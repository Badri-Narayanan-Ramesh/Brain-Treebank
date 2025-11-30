# src/training/train_contrastive_v3.py

import os
import argparse
from typing import Iterable, Tuple, Optional

import yaml
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset_for_brain_and_audio import make_dataloaders_from_config
from src.data.dataset_popt_speech import make_popt_speech_dataloaders_from_config
from src.models.brain_transformer import PopulationTransformer
from src.models.audio_encoder import ConvAudioEncoder
from src.models.projection_heads import ProjectionHead
from src.models.popt_speech_model import PopTSpeechModel
from src.training.losses import InfoNCELoss, temporal_smoothing_loss
from src.training.metrics import (
    compute_retrieval_at_k_sampled,
    compute_auc,
    compute_embedding_jitter,
)


# -------------------------------------------------------------------------
# Config utils
# -------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_trainable_params(*modules: nn.Module):
    """Yield only parameters that require gradients from a list of modules."""
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.requires_grad:
                yield p


# -------------------------------------------------------------------------
# EVAL: generic contrastive + speech (old path)
# -------------------------------------------------------------------------
def evaluate_generic(
    loader,
    brain_encoder,
    audio_encoder,
    brain_proj,
    audio_proj,
    speech_head,
    device,
    brain_only: bool,
) -> Tuple[float, float, float, float]:
    """
    Evaluate when using PopulationTransformer + optional audio/contrastive.

    For PopTSpeechDataset (brain-only):
        batch = (brain, labels, meta)

    For BrainAudioDataset:
        batch = (brain, audio, labels, meta)
    """
    brain_encoder.eval()
    speech_head.eval()
    if (not brain_only) and audio_encoder is not None:
        audio_encoder.eval()
        if brain_proj is not None:
            brain_proj.eval()
        if audio_proj is not None:
            audio_proj.eval()

    all_scores = []
    all_labels = []
    all_brain_z = []
    all_u = []
    all_v = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                brain, labels, meta = batch
                audio = None
            elif len(batch) == 4:
                brain, audio, labels, meta = batch
            else:
                raise ValueError(f"Unexpected batch structure with len={len(batch)}")

            brain = brain.to(device)
            labels = labels.float().to(device)

            z_brain, _ = brain_encoder(brain)  # z_brain: (B, d_model)
            logits = speech_head(z_brain).squeeze(-1)  # (B,)
            scores = torch.sigmoid(logits)

            all_scores.append(scores)
            all_labels.append(labels)
            all_brain_z.append(z_brain)

            if (not brain_only) and audio is not None:
                audio = audio.to(device)
                z_audio = audio_encoder(audio)
                u = brain_proj(z_brain)
                v = audio_proj(z_audio)
                all_u.append(u)
                all_v.append(v)

    if not all_scores:
        return 0.0, 0.0, 0.5, 0.0

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_brain_z = torch.cat(all_brain_z, dim=0)

    auc = compute_auc(all_scores, all_labels)
    jitter = compute_embedding_jitter(all_brain_z, meta=None)

    if (not brain_only) and all_u and all_v:
        all_u = torch.cat(all_u, dim=0)
        all_v = torch.cat(all_v, dim=0)
        ret2, chance = compute_retrieval_at_k_sampled(all_u, all_v, k=2)
    else:
        ret2, chance = 0.0, 0.0

    print(f"[Eval] AUC(model) = {auc:.4f}")
    return ret2, chance, auc, jitter


# -------------------------------------------------------------------------
# EVAL: PopT downstream (Option A)
# -------------------------------------------------------------------------
def evaluate_popt(
    loader,
    popt_model: PopTSpeechModel,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Evaluate when using the full PopT downstream model via PopTSpeechModel
    on PopTSpeechDataset (brain, labels, meta).
    """
    popt_model.eval()

    all_scores = []
    all_labels = []
    all_cls = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) != 3:
                raise ValueError(
                    "[evaluate_popt] Expected PopTSpeechDataset yielding (brain, labels, meta)."
                )
            brain, labels, meta = batch
            brain = brain.to(device)
            labels = labels.float().to(device)

            logits = popt_model(brain)            # (B,)
            scores = torch.sigmoid(logits)        # (B,)
            cls = popt_model.cls_embedding(brain) # (B, hidden_dim)

            all_scores.append(scores)
            all_labels.append(labels)
            all_cls.append(cls)

    if not all_scores:
        return 0.0, 0.0, 0.5, 0.0

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_cls = torch.cat(all_cls, dim=0)

    auc = compute_auc(all_scores, all_labels)
    jitter = compute_embedding_jitter(all_cls, meta=None)

    # No contrastive or audio retrieval in this mode
    ret2, chance = 0.0, 0.0
    print(f"[Eval-PopT] AUC(model) = {auc:.4f}")
    return ret2, chance, auc, jitter


# -------------------------------------------------------------------------
# TRAIN
# -------------------------------------------------------------------------
def train(config_path: str):
    config = load_config(config_path)
    brain_only = bool(config["training"].get("brain_only", False))
    use_popt_speech = bool(config["data"].get("use_popt_speech", False))
    model_cfg = config["model"]
    use_popt_downstream = bool(model_cfg.get("use_popt_downstream", False))

    # Device
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # The name of the file to write summary metrics into.  Default is
    # 'metrics.json'.  This value can be overridden via
    # config['logging']['metrics_file'].
    metrics_file = config.get("logging", {}).get("metrics_file", "metrics.json")

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    if use_popt_speech:
        print("[Data] Using PopTSpeechDataset (brain-only speech decoding)")
        train_loader, val_loader = make_popt_speech_dataloaders_from_config(config)
    else:
        print("[Data] Using BrainAudioDataset (brain + proxy audio)")
        train_loader, val_loader = make_dataloaders_from_config(config)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # ---------------------------------------------------------------------
    # Branch 1: PopT downstream model (Option A, your new path)
    # ---------------------------------------------------------------------
    if use_popt_speech and use_popt_downstream:
        print("[Model] Using PopTSpeechModel (full PopT downstream)")

        popt_cfg_path = model_cfg["popt_cfg_path"]
        popt_upstream_path = model_cfg["popt_upstream_path"]

        popt_model = PopTSpeechModel(
            cfg_path=popt_cfg_path,
            upstream_path=popt_upstream_path,
        ).to(device)

        bce = nn.BCEWithLogitsLoss().to(device)
        lambda_smooth = float(config["training"].get("lambda_smooth", 0.0))

        params = list(iter_trainable_params(popt_model))
        optimizer = optim.AdamW(
            params,
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        epochs = int(config["training"]["epochs"])
        log_interval = int(config["logging"].get("log_interval", 50))
        val_interval = int(config["logging"].get("val_interval", 1))
        ckpt_dir = config["logging"].get("ckpt_dir", "./checkpoints_popt_downstream")
        os.makedirs(ckpt_dir, exist_ok=True)

        train_losses = []
        val_aucs = []
        val_retrievals = []
        val_jitters = []

        for epoch in range(1, epochs + 1):
            popt_model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"[PopT] Epoch {epoch}/{epochs}")

            for step, batch in enumerate(pbar, start=1):
                if len(batch) != 3:
                    raise ValueError(
                        "[train-PopT] Expected PopTSpeechDataset batch "
                        "of (brain, labels, meta)."
                    )
                brain, labels, meta = batch
                brain = brain.to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                # forward
                logits = popt_model(brain)            # (B,)
                cls = popt_model.cls_embedding(brain) # (B, hidden_dim)

                loss_cls = bce(logits, labels)
                loss_sm = temporal_smoothing_loss(cls, meta, lambda_smooth)
                loss = loss_cls + loss_sm

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if step % log_interval == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.3f}",
                            "L_cls": f"{loss_cls.item():.3f}",
                            "L_sm": f"{loss_sm.item():.3f}",
                        }
                    )

            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"[PopT][Epoch {epoch}] Train Loss: {avg_loss:.4f}")

            # ------------------------ Validation ------------------------
            if epoch % val_interval == 0 or epoch == epochs:
                ret2, chance, auc, jitter = evaluate_popt(
                    val_loader,
                    popt_model,
                    device,
                )
                val_retrievals.append(ret2)
                val_aucs.append(auc)
                val_jitters.append(jitter)

                print(
                    f"[PopT][Epoch {epoch}] "
                    f"Retrieval@2: {ret2:.4f} (chance: {chance:.4f}), "
                    f"AUC: {auc:.4f}, Jitter: {jitter:.4f}"
                )

                # Save best checkpoint by AUC
                if auc >= max(val_aucs):
                    ckpt_path = os.path.join(ckpt_dir, f"best_epoch_{epoch}.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "popt_model": popt_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "val_auc": auc,
                        },
                        ckpt_path,
                    )
                    print(f"[PopT][Checkpoint] Saved best model to {ckpt_path}")

        # ------------------------ Plots ------------------------
        epochs_range = np.arange(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs_range, train_losses, label="Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PopT downstream: Training loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "train_loss.png"))
        plt.close()

        if val_aucs:
            plt.figure()
            plt.plot(epochs_range[: len(val_aucs)], val_aucs, label="Val AUC")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.title("PopT downstream: Validation AUC")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, "val_auc.png"))
            plt.close()

        if val_jitters:
            plt.figure()
            plt.plot(epochs_range[: len(val_jitters)], val_jitters, label="Val jitter")
            plt.xlabel("Epoch")
            plt.ylabel("Embedding jitter")
            plt.title("PopT downstream: Validation jitter")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, "val_jitter.png"))
            plt.close()

        # At the end of the PopT branch, summarise and save metrics.  We
        # compute the best validation AUC and final values of AUC,
        # jitter, and retrieval.  These metrics are written to a JSON
        # file named by `metrics_file` in the checkpoint directory.
        try:
            best_auc = max(val_aucs) if val_aucs else 0.0
            final_auc = val_aucs[-1] if val_aucs else 0.0
            final_jitter = val_jitters[-1] if val_jitters else 0.0
            final_retrieval = val_retrievals[-1] if val_retrievals else 0.0
        except Exception:
            best_auc = final_auc = final_jitter = final_retrieval = 0.0
        metrics = {
            "best_auc": float(best_auc),
            "final_auc": float(final_auc),
            "final_jitter": float(final_jitter),
            "final_retrieval": float(final_retrieval),
        }
        metrics_path = os.path.join(ckpt_dir, metrics_file)
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f)
            print(f"[PopT][Metrics] Saved metrics to {metrics_path}")
        except Exception as e:
            print(f"[PopT][Metrics] Failed to save metrics: {e}")
        return  # done with PopT branch

    # ---------------------------------------------------------------------
    # Branch 2: Original Brain-Treebank model (PopulationTransformer + audio)
    # ---------------------------------------------------------------------
    print("[Model] Using PopulationTransformer + contrastive path (generic mode)")

    d_model = int(model_cfg["d_model"])
    d_contrastive = int(model_cfg.get("d_contrastive", 64))

    brain_input_dim = int(model_cfg.get("brain_input_dim", 768))
    coord_dim = int(model_cfg.get("coord_dim", 0))

    # Brain encoder
    brain_encoder = PopulationTransformer(
        input_dim=brain_input_dim,
        d_model=d_model,
        coord_dim=coord_dim,
        n_heads=int(model_cfg.get("n_heads", 8)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    # Audio encoder only used if we are not in brain-only mode
    audio_encoder = (
        ConvAudioEncoder(
            input_dim=None,
            d_model=int(model_cfg.get("audio_d_model", 128)),
            hidden_dim=int(model_cfg.get("audio_hidden_dim", 256)),
            num_layers=int(model_cfg.get("audio_num_layers", 3)),
            kernel_size=int(model_cfg.get("audio_kernel_size", 5)),
            dropout=float(model_cfg.get("audio_dropout", 0.1)),
        ).to(device)
        if not brain_only
        else None
    )

    # Projection heads
    brain_proj = (
        ProjectionHead(d_model, d_contrastive).to(device) if not brain_only else None
    )
    audio_proj = (
        ProjectionHead(int(model_cfg.get("audio_d_model", 128)), d_contrastive).to(
            device
        )
        if not brain_only
        else None
    )

    # Speech head
    speech_head = nn.Linear(d_model, 1).to(device)

    # Load upstream weights (best-effort)
    upstream_path = model_cfg.get("upstream_path", None)
    if upstream_path:
        print(f"[PopT-like] Loading pretrained weights from: {upstream_path}")
        state = torch.load(upstream_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
        missing, unexpected = brain_encoder.load_state_dict(state, strict=False)
        print("[PopT-like] Missing keys:")
        for k in missing:
            print("   ", k)
        print("[PopT-like] Unexpected keys:")
        for k in unexpected:
            print("   ", k)
        print(
            f"[PopT-like] Loaded with missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    # Optionally freeze brain encoder
    freeze_brain = bool(model_cfg.get("freeze_brain", False))
    if freeze_brain:
        print("[PopT-like] Freezing brain encoder parameters")
        for p in brain_encoder.parameters():
            p.requires_grad = False

    # Losses & optimizer
    contrastive_loss = (
        InfoNCELoss(temperature=float(config["training"]["temperature"])).to(device)
        if not brain_only
        else None
    )
    bce = nn.BCEWithLogitsLoss().to(device)

    lambda_cls = float(config["training"].get("lambda_cls", 0.0))
    lambda_smooth = float(config["training"].get("lambda_smooth", 0.0))

    params = list(
        iter_trainable_params(
            brain_encoder,
            speech_head,
            *([audio_encoder, brain_proj, audio_proj] if not brain_only else []),
        )
    )
    optimizer = optim.AdamW(
        params,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    # Training loop
    epochs = int(config["training"]["epochs"])
    log_interval = int(config["logging"].get("log_interval", 50))
    val_interval = int(config["logging"].get("val_interval", 1))
    ckpt_dir = config["logging"].get("ckpt_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_losses = []
    val_aucs = []
    val_retrievals = []
    val_jitters = []

    for epoch in range(1, epochs + 1):
        brain_encoder.train()
        if (not brain_only) and audio_encoder is not None:
            audio_encoder.train()
            if brain_proj is not None:
                brain_proj.train()
            if audio_proj is not None:
                audio_proj.train()
        speech_head.train()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for step, batch in enumerate(pbar, start=1):
            if len(batch) == 3:
                brain, labels, meta = batch
                audio = None
            elif len(batch) == 4:
                brain, audio, labels, meta = batch
            else:
                raise ValueError(f"Unexpected batch with len={len(batch)}")

            brain = brain.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            z_brain, _ = brain_encoder(brain)

            if brain_only:
                logits = speech_head(z_brain).squeeze(-1)
                loss_cls = bce(logits, labels)
                loss_sm = temporal_smoothing_loss(z_brain, meta, lambda_smooth)
                loss_con = torch.tensor(0.0, device=device)
                loss = loss_cls + loss_sm
            else:
                if audio is None:
                    raise RuntimeError(
                        "Received batch without audio while brain_only=False"
                    )
                audio = audio.to(device)
                z_audio = audio_encoder(audio)
                u = brain_proj(z_brain)
                v = audio_proj(z_audio)

                loss_con = contrastive_loss(u, v)
                logits = speech_head(z_brain).squeeze(-1)
                loss_cls = bce(logits, labels)
                loss_sm = temporal_smoothing_loss(z_brain, meta, lambda_smooth)
                loss = loss_con + lambda_cls * loss_cls + loss_sm

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.3f}",
                        "L_con": f"{loss_con.item():.3f}",
                        "L_cls": f"{loss_cls.item():.3f}",
                        "L_sm": f"{loss_sm.item():.3f}",
                    }
                )

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        # Validation
        if epoch % val_interval == 0 or epoch == epochs:
            ret2, chance, auc, jitter = evaluate_generic(
                val_loader,
                brain_encoder,
                audio_encoder,
                brain_proj,
                audio_proj,
                speech_head,
                device,
                brain_only=brain_only,
            )
            val_retrievals.append(ret2)
            val_aucs.append(auc)
            val_jitters.append(jitter)

            print(
                f"[Epoch {epoch}] "
                f"Retrieval@2: {ret2:.4f} (chance: {chance:.4f}), "
                f"AUC: {auc:.4f}, Jitter: {jitter:.4f}"
            )

            # Save best checkpoint by AUC
            if auc >= max(val_aucs):
                ckpt_path = os.path.join(ckpt_dir, f"best_epoch_{epoch}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "brain_encoder": brain_encoder.state_dict(),
                        "speech_head": speech_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_auc": auc,
                    },
                    ckpt_path,
                )
                print(f"[Checkpoint] Saved best model to {ckpt_path}")

    # Plots
    epochs_range = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "train_loss.png"))
    plt.close()

    if val_aucs:
        plt.figure()
        plt.plot(epochs_range[: len(val_aucs)], val_aucs, label="Val AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title("Validation AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "val_auc.png"))
        plt.close()

        if val_jitters:
            plt.figure()
            plt.plot(epochs_range[: len(val_jitters)], val_jitters, label="Val jitter")
            plt.xlabel("Epoch")
            plt.ylabel("Embedding jitter")
            plt.title("Validation jitter")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, "val_jitter.png"))
            plt.close()

        # Write summary metrics for the generic branch.  We compute the
        # best and final values of AUC, jitter and retrieval.  These are
        # written to the same checkpoint directory as the plots.
        try:
            best_auc = max(val_aucs) if val_aucs else 0.0
            final_auc = val_aucs[-1] if val_aucs else 0.0
            final_jitter = val_jitters[-1] if val_jitters else 0.0
            final_retrieval = val_retrievals[-1] if val_retrievals else 0.0
        except Exception:
            best_auc = final_auc = final_jitter = final_retrieval = 0.0

        metrics = {
            "best_auc": float(best_auc),
            "final_auc": float(final_auc),
            "final_jitter": float(final_jitter),
            "final_retrieval": float(final_retrieval),
        }
        metrics_path = os.path.join(ckpt_dir, metrics_file)
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f)
            print(f"[Metrics] Saved metrics to {metrics_path}")
        except Exception as e:
            print(f"[Metrics] Failed to save metrics: {e}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()
    train(args.config)
