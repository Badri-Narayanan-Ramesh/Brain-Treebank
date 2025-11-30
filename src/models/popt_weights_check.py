# src/models/popt_weights_check.py

from __future__ import annotations

import yaml
import torch

from .popt_speech_model import PopTSpeechModel


def main():
    # 1) Load your Brain-Treebank config
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]

    # 2) Get PopT config + upstream path
    if "popt_cfg_path" not in model_cfg:
        raise KeyError(
            "popt_cfg_path is missing in configs/base.yaml under 'model'.\n"
            "Please add, for example:\n"
            "  model:\n"
            "    popt_cfg_path: "
            "\"C:/Users/badri/OneDrive/Documents/EE 675 Neural Learning/Baseline Replication/PopulationTransformer/conf/model/pt_downstream_model.yaml\""
        )

    if "popt_upstream_path" not in model_cfg:
        raise KeyError(
            "popt_upstream_path is missing in configs/base.yaml under 'model'.\n"
            "Please add, for example:\n"
            "  model:\n"
            "    popt_upstream_path: "
            "\"C:/Users/badri/OneDrive/Documents/EE 675 Neural Learning/Baseline Replication/PopulationTransformer/pretrained_weights/popt_pretrained_weights/pretrained_popt_brainbert_stft.pth\""
        )

    popt_cfg_path = model_cfg["popt_cfg_path"]
    popt_upstream_path = model_cfg["popt_upstream_path"]

    print(f"[Check] Using PopT config: {popt_cfg_path}")
    print(f"[Check] Using PopT upstream: {popt_upstream_path}")

    # 3) Build PopTSpeechModel (PtDownstreamModel + upstream weights)
    model = PopTSpeechModel(
        cfg_path=popt_cfg_path,
        upstream_path=popt_upstream_path,
    )

    # 4) Fake batch to verify shapes
    B, N, F = 2, 50, 768
    x = torch.randn(B, N, F)

    with torch.no_grad():
        logits = model(x)                # (B,)
        cls = model.cls_embedding(x)     # (B, hidden_dim)

    print(f"[Check] logits shape: {logits.shape}")
    print(f"[Check] cls shape:    {cls.shape}")
    print(f"[Check] hidden_dim:   {model.hidden_dim}")
    print(
        "[Check] If this ran without error and shapes look correct, "
        "PopT downstream + upstream weights are loading properly via Option A."
    )


if __name__ == "__main__":
    main()
