# src/models/popt_speech_model.py

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf


class PopTSpeechModel(nn.Module):
    """
    Thin wrapper around PopT's PtDownstreamModel for brain-only speech decoding.

    Usage:
        model = PopTSpeechModel(
            cfg_path=".../PopulationTransformer/conf/model/pt_downstream_model.yaml",
            upstream_path=".../pretrained_popt_brainbert_stft.pth",
        )
        logits = model(brain)            # (B,)
        cls = model.cls_embedding(brain) # (B, hidden_dim)
    """

    def __init__(self, cfg_path: str, upstream_path: str):
        super().__init__()

        cfg_path = Path(cfg_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"[PopTSpeechModel] PopT config not found: {cfg_path}")

        # PopT repo root is in .../Baseline Replication/PopulationTransformer
        # This file is .../Baseline Replication/Brain-Treebank/src/models/popt_speech_model.py
        # So we go up 4 levels to 'Baseline Replication' then down to 'PopulationTransformer'
        popt_root = Path(__file__).parents[3] / "PopulationTransformer"
        if str(popt_root) not in sys.path:
            sys.path.insert(0, str(popt_root))

        # Import PtDownstreamModel from the PopT repo
        from models.pt_downstream_model import PtDownstreamModel  # type: ignore

        # Load YAML config (should be pt_downstream_model.yaml)
        cfg = OmegaConf.load(str(cfg_path))

        # Sanity check: downstream config must have upstream_cfg
        if "upstream_cfg" not in cfg:
            raise RuntimeError(
                "[PopTSpeechModel] Expected 'upstream_cfg' in the PopT downstream "
                f"config, but it is missing in {cfg_path}. "
                "You likely passed pt_custom_model.yaml instead of pt_downstream_model.yaml."
            )

        # Inject upstream_path (checkpoint) into cfg
        cfg.upstream_path = upstream_path

        self.cfg = cfg

        # Build downstream model (this will load upstream encoder + linear head)
        self.model = PtDownstreamModel()
        self.model.build_model(self.cfg)

    @property
    def hidden_dim(self) -> int:
        # PopT hidden_dim (e.g., 512)
        return int(self.cfg.hidden_dim)

    def _make_pos_and_mask(self, brain: torch.Tensor):
        """
        Given brain: (B, N, F), create PopT-style positions and src_key_mask
        assuming no padding (all electrodes valid).
        """
        B, N, _ = brain.shape
        device = brain.device

        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        src_key_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        # Check if upstream expects multi-subject encoding (coords + seq_id)
        upstream_cfg = self.cfg.get("upstream_cfg", {})
        if upstream_cfg.get("position_encoding") == "multi_subj_position_encoding":
            # Generate dummy coordinates and sequence IDs for compatibility
            # In a real scenario, these should come from the dataset
            coords = torch.zeros(B, N, 3, dtype=torch.long, device=device)
            seq_id = torch.zeros(B, N, dtype=torch.long, device=device)
            return src_key_mask, (coords, seq_id)
            
        return src_key_mask, positions

    def forward(self, brain: torch.Tensor) -> torch.Tensor:
        """
        brain: (B, N, F=768) PopT BrainBERT embeddings per electrode
        Returns:
            logits: (B,) speech logits (before sigmoid)
        """
        B, N, F = brain.shape
        device = brain.device
        
        # Prepend dummy CLS token (zeros)
        cls_token = torch.zeros(B, 1, F, device=device)
        brain_with_cls = torch.cat([cls_token, brain], dim=1)
        
        src_key_mask, positions = self._make_pos_and_mask(brain)
        
        # Prepend mask for CLS token (False = not masked)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        src_key_mask = torch.cat([cls_mask, src_key_mask], dim=1)
        
        logits = self.model(brain_with_cls, src_key_mask, positions)  # (B, 1)
        return logits.squeeze(-1)

    def cls_embedding(self, brain: torch.Tensor) -> torch.Tensor:
        """
        Returns CLS embedding from upstream PopT encoder: (B, hidden_dim).
        Useful for jitter / representation analyses.
        """
        B, N, F = brain.shape
        device = brain.device
        
        # Prepend dummy CLS token (zeros)
        cls_token = torch.zeros(B, 1, F, device=device)
        brain_with_cls = torch.cat([cls_token, brain], dim=1)

        src_key_mask, positions = self._make_pos_and_mask(brain)
        
        # Prepend mask for CLS token
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        src_key_mask = torch.cat([cls_mask, src_key_mask], dim=1)

        # upstream(..., intermediate_rep=True) returns sequence reps
        outs = self.model.upstream(
            brain_with_cls, src_key_mask, positions, intermediate_rep=True
        )

        if isinstance(outs, tuple):
            output_specs = outs[0]
        else:
            output_specs = outs

        cls = output_specs[:, 0, :]  # (B, hidden_dim)
        return cls
