# src/models/popt_speech_model.py

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig


def _get_project_and_popt_root() -> Tuple[Path, Path]:
    """
    Returns:
        project_root: <...>/Brain-Treebank
        popt_root:    <...>/PopulationTransformer  (sibling repo)
    """
    this_file = Path(__file__).resolve()
    # src/models/popt_speech_model.py -> src/models -> src -> Brain-Treebank
    project_root = this_file.parents[2]
    popt_root = project_root.parent / "PopulationTransformer"
    return project_root, popt_root


# Make PopulationTransformer repo importable as top-level 'models'
PROJECT_ROOT, POPT_ROOT = _get_project_and_popt_root()
if str(POPT_ROOT) not in sys.path:
    sys.path.insert(0, str(POPT_ROOT))

# Now these refer to PopulationTransformer/models/*
from models.pt_downstream_model import PtDownstreamModel  # type: ignore
from models import build_model  # type: ignore


class PopTSpeechModel(nn.Module):
    """
    Thin wrapper around PopT's PtDownstreamModel for brain-only speech decoding.

    This uses the original PopulationTransformer implementation (Option A).

    Args:
        cfg_path:       Path to pt_downstream_model.yaml (or equivalent).
        upstream_path:  Path to pretrained_popt_brainbert_stft.pth.

    Expected brain input:
        brain: (B, N, F=768)  # BrainBERT electrode features

    Forward:
        logits = model(brain)            # (B,)
        cls    = model.cls_embedding(...)# (B, hidden_dim)
    """

    def __init__(self, cfg_path: Union[str, Path], upstream_path: Union[str, Path]):
        super().__init__()

        cfg_path = Path(cfg_path)
        upstream_path = Path(upstream_path)

        if not cfg_path.is_file():
            raise FileNotFoundError(f"[PopTSpeechModel] PopT config not found: {cfg_path}")
        if not upstream_path.is_file():
            raise FileNotFoundError(
                f"[PopTSpeechModel] PopT upstream weights not found: {upstream_path}"
            )

        # Load config with OmegaConf
        cfg = OmegaConf.load(str(cfg_path))
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # We expect an 'upstream_cfg' subtree describing the encoder
        if "upstream_cfg" not in cfg:
            raise KeyError(
                "[PopTSpeechModel] 'upstream_cfg' missing in "
                f"{cfg_path}. Please ensure this YAML matches PopT's pt_downstream_model.yaml."
            )

        # Attach upstream_path into cfg so PtDownstreamModel.build_model can see it
        cfg.upstream_path = str(upstream_path)

        # Some PopT code expects these attributes to exist
        if "hidden_dim" not in cfg:
            raise KeyError(
                "[PopTSpeechModel] 'hidden_dim' missing in PopT downstream config. "
                "Expected something like hidden_dim: 512"
            )
        if "input_dim" not in cfg:
            # default for BrainBERT features
            cfg.input_dim = 768

        self.cfg: DictConfig = cfg

        # Build downstream model (this will internally call build_model() on upstream_cfg
        # and load the PopT weights from cfg.upstream_path)
        self.model = PtDownstreamModel()
        self.model.build_model(self.cfg)

        # Cache whether the upstream uses multi-subject positional encoding
        upstream_cfg = self.cfg.upstream_cfg
        self.position_encoding = str(
            upstream_cfg.get("position_encoding", "positional_encoding")
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def hidden_dim(self) -> int:
        return int(self.cfg.hidden_dim)

    # ------------------------------------------------------------------
    # Helpers to create mask + positions
    # ------------------------------------------------------------------
    def _make_pos_and_mask(
        self, brain: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Given:
            brain: (B, N, F)

        Returns:
            src_key_mask: (B, L) bool mask, where L is the sequence length
                          AFTER positional encodings are applied.
            positions:
                - if positional_encoding == 'multi_subj_position_encoding':
                      (coords, seq_id)
                  coords: (B, N, 3) long
                  seq_id: (B, N)   long
                - else:
                      positions: (B, N) long indices [0..N-1]
        """
        if brain.dim() != 3:
            raise ValueError(
                f"[PopTSpeechModel] Expected brain to be (B, N, F), got shape={brain.shape}"
            )

        B, N, _ = brain.shape
        device = brain.device

        if self.position_encoding == "multi_subj_position_encoding":
            # Dummy integer coordinates and sequence IDs
            # (everything at index 0 in the pos-encoding table)
            coords = torch.zeros(B, N, 3, dtype=torch.long, device=device)
            seq_id = torch.zeros(B, N, dtype=torch.long, device=device)
            positions = (coords, seq_id)
            # MultiSubjBrainPositionalEncoding adds a CLS token internally,
            # so final sequence length is N + 1.
            seq_len = N + 1
        else:
            # Simple 1D positions [0, 1, ..., N-1]
            positions = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
            seq_len = N

        # No padding -> mask is all False
        src_key_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        return src_key_mask, positions

    # ------------------------------------------------------------------
    # Forward + CLS embedding
    # ------------------------------------------------------------------
    def forward(self, brain: torch.Tensor) -> torch.Tensor:
        """
        Args:
            brain: (B, N, F)

        Returns:
            logits: (B,) speech logits before sigmoid
        """
        src_key_mask, positions = self._make_pos_and_mask(brain)
        # PtDownstreamModel.forward returns (B, 1)
        logits = self.model(brain, src_key_mask, positions)
        return logits.squeeze(-1)

    def cls_embedding(self, brain: torch.Tensor) -> torch.Tensor:
        """
        Get CLS embedding from the upstream encoder (for analysis / jitter).

        Returns:
            cls: (B, hidden_dim)
        """
        src_key_mask, positions = self._make_pos_and_mask(brain)

        # upstream(..., intermediate_rep=True) returns sequence reps
        outs = self.model.upstream(
            brain, src_key_mask, positions, intermediate_rep=True
        )

        if isinstance(outs, tuple):
            output_specs = outs[0]
        else:
            output_specs = outs

        # CLS is always at index 0
        cls = output_specs[:, 0, :]  # (B, hidden_dim)
        return cls
