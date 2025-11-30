"""
run_experiments.py
====================

This script orchestrates a series of experiments for the Brain–Treebank
speech decoding project.  It automates running both supervised and
contrastive variants of the model with and without temporal smoothing,
using a single base YAML configuration as a starting point.  Each
experiment is executed by invoking the training routine from
``src.training.train_contrastive_v3`` with a temporary configuration
file.  After training, the script collects summary metrics (best AUC,
final AUC, final jitter, final retrieval) written out by the training
code to a JSON file in the experiment’s checkpoint directory.  A final
summary of all experiments is written to ``summary.json`` in the
specified output directory.

Usage::

    python run_experiments.py --config configs/base.yaml --output_dir results

Arguments:
    --config: Path to a YAML configuration file that defines your
        BrainTreebank/Popt training setup.  This file will be copied
        and modified internally for each experiment.
    --output_dir: Directory where per–experiment subdirectories and
        metrics files will be written.  The script will create the
        directory if it does not exist.
    --smoothing_lambda: Optional float specifying the smoothing
        strength (lambda_smooth) for the runs that include temporal
        smoothing.  Defaults to 0.1.

The experiments run are:

    1. ``sup_no_smooth``: Supervised PopT decoding with no smoothing.
    2. ``sup_smooth``: Supervised PopT decoding with smoothing enabled.
    3. ``con_no_smooth``: Contrastive brain–audio training without smoothing.
    4. ``con_smooth``: Contrastive brain–audio training with smoothing.

Each experiment writes its checkpoints and metrics into a
subdirectory named after the experiment under ``output_dir``.

Note: This script does not perform any training itself in the
``__main__`` block when imported.  It simply coordinates calls to
``train()`` from the underlying training module.  It assumes that
``train_contrastive_v3.py`` has been patched to write a ``metrics.json``
file in its checkpoint directory (see train_contrastive_v3 for details).
"""

from __future__ import annotations

import argparse
import os
import copy
import json
from typing import Dict, Any

import yaml

from src.training.train_contrastive_v3 import train  # type: ignore


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str) -> None:
    """Write a Python dictionary to a YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def run_experiment(
    name: str,
    base_cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    output_dir: str,
) -> Dict[str, float]:
    """
    Run a single experiment by modifying a base configuration and
    invoking the training routine.  Returns the loaded metrics for
    the experiment as a dictionary.

    Parameters
    ----------
    name : str
        Name of the experiment; used to create subdirectories and
        temporary configuration filenames.
    base_cfg : dict
        The base configuration dictionary to copy and modify.
    overrides : dict
        A set of configuration overrides.  Recognized keys are:
            - use_popt_speech (bool)
            - use_popt_downstream (bool)
            - brain_only (bool)
            - lambda_smooth (float)
            - use_contrastive (bool) [unused here, for clarity]
    output_dir : str
        Root directory where experiment-specific directories and
        configuration files will be written.

    Returns
    -------
    metrics : dict
        A dictionary containing summary metrics produced by the
        training routine.  Keys include ``best_auc``, ``final_auc``,
        ``final_jitter``, and ``final_retrieval``.
    """
    # Deep copy base configuration to avoid side effects
    cfg = copy.deepcopy(base_cfg)

    # Update configuration with override values
    data_cfg = cfg.setdefault("data", {})
    model_cfg = cfg.setdefault("model", {})
    training_cfg = cfg.setdefault("training", {})
    logging_cfg = cfg.setdefault("logging", {})

    # Determine experiment-specific settings
    use_popt_speech = overrides.get("use_popt_speech", data_cfg.get("use_popt_speech", False))
    use_popt_downstream = overrides.get("use_popt_downstream", model_cfg.get("use_popt_downstream", False))
    brain_only = overrides.get("brain_only", training_cfg.get("brain_only", False))
    lambda_smooth = overrides.get("lambda_smooth", training_cfg.get("lambda_smooth", 0.0))

    # Apply overrides
    data_cfg["use_popt_speech"] = bool(use_popt_speech)
    model_cfg["use_popt_downstream"] = bool(use_popt_downstream)
    training_cfg["brain_only"] = bool(brain_only)
    training_cfg["lambda_smooth"] = float(lambda_smooth)

    # Set up experiment-specific directories
    exp_dir = os.path.join(output_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    logging_cfg["ckpt_dir"] = exp_dir
    logging_cfg["metrics_file"] = "metrics.json"

    # Write temporary configuration file
    tmp_cfg_path = os.path.join(exp_dir, f"{name}_config.yaml")
    save_config(cfg, tmp_cfg_path)

    print(f"[Experiment] Running '{name}' with config saved to {tmp_cfg_path}")

    # Invoke training.  This call will train the model and write metrics
    # into the ckpt_dir.  The train function expects a path to a YAML
    # configuration file.
    train(tmp_cfg_path)

    # Load metrics after training completes.  If the metrics file is
    # missing or unreadable, return an empty dictionary.
    metrics_path = os.path.join(exp_dir, "metrics.json")
    metrics: Dict[str, float]
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"[Experiment] Warning: Could not load metrics for '{name}': {e}")
        metrics = {}

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple speech decoding experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_output",
        help="Directory to write experiment outputs and metrics",
    )
    parser.add_argument(
        "--smoothing_lambda",
        type=float,
        default=0.1,
        help="Smoothing strength (lambda_smooth) for runs with smoothing",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the base configuration
    base_cfg = load_config(args.config)

    # Define the four experiments.  The ``use_contrastive`` flag is
    # included for clarity but not used directly in this helper.  It
    # indicates whether the model should use audio features (False for
    # supervised PopT, True for contrastive).  In this script we
    # control that behaviour via use_popt_speech/use_popt_downstream
    # and brain_only.  Adjust as needed.
    experiments = [
        (
            "sup_no_smooth",
            {
                "use_popt_speech": True,
                "use_popt_downstream": True,
                "brain_only": True,
                "lambda_smooth": 0.0,
                "use_contrastive": False,
            },
        ),
        (
            "sup_smooth",
            {
                "use_popt_speech": True,
                "use_popt_downstream": True,
                "brain_only": True,
                "lambda_smooth": args.smoothing_lambda,
                "use_contrastive": False,
            },
        ),
        (
            "con_no_smooth",
            {
                "use_popt_speech": False,
                "use_popt_downstream": False,
                "brain_only": False,
                "lambda_smooth": 0.0,
                "use_contrastive": True,
            },
        ),
        (
            "con_smooth",
            {
                "use_popt_speech": False,
                "use_popt_downstream": False,
                "brain_only": False,
                "lambda_smooth": args.smoothing_lambda,
                "use_contrastive": True,
            },
        ),
    ]

    # Run all experiments sequentially and collect their metrics
    summary: Dict[str, Dict[str, float]] = {}
    for name, overrides in experiments:
        metrics = run_experiment(name, base_cfg, overrides, args.output_dir)
        summary[name] = metrics

    # Save a combined summary for convenience
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary] All experiment metrics saved to {summary_path}")


if __name__ == "__main__":
    main()