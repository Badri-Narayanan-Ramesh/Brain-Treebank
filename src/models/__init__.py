# src/models/__init__.py
"""
Lightweight models package for Brain-Treebank.

Important:
- Do NOT import the external PopT 'models' package here.
- PopT lives in:  <repo-root>/PopulationTransformer/models
  and is accessed only via src.models.popt_speech_model.PopTSpeechModel.

This file should *not* contain MODEL_REGISTRY, register_model,
or importlib.import_module('models.*') calls.
"""

from .brain_transformer import PopulationTransformer
from .audio_encoder import ConvAudioEncoder
from .projection_heads import ProjectionHead

__all__ = [
    "PopulationTransformer",
    "ConvAudioEncoder",
    "ProjectionHead",
]
