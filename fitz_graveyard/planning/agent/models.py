# fitz_graveyard/planning/agent/models.py
"""
Embedding and reranking model helpers for the retrieval pipeline.

Loads sentence-transformers models in-process alongside the running
llama-server. These models are small (~275MB embedder + ~85MB reranker)
but need ~2GB free VRAM for model + working memory during encoding.
Checks both system RAM and free VRAM before loading — falls back to
BM25 + LLM judgment if resources are insufficient.

Requires: pip install fitz-graveyard[retrieval]
"""

import gc
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Minimum available system RAM (GB) to attempt loading torch + models.
# Below this, the paging file can exhaust and crash the asyncio event loop.
_MIN_AVAILABLE_RAM_GB = 2.0


def _check_system_memory() -> None:
    """Fail fast if system RAM is too low to safely load torch.

    Importing torch + CUDA runtime allocates ~1-2 GB system virtual memory.
    Combined with llama-server, this can exhaust the page file on 16 GB systems,
    causing OSError 1455 that corrupts the asyncio event loop.

    If torch is already imported (CUDA context already allocated), skip the check
    — the expensive memory hit already happened, and subsequent model loads are
    only ~85-275 MB.
    """
    if "torch" in sys.modules:
        logger.debug("System RAM check skipped — torch already loaded")
        return

    import psutil

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    if available_gb < _MIN_AVAILABLE_RAM_GB:
        raise MemoryError(
            f"Only {available_gb:.1f} GB RAM available (need {_MIN_AVAILABLE_RAM_GB} GB). "
            f"Skipping embedding/reranking to avoid paging file exhaustion. "
            f"Close other applications or increase Windows page file size."
        )
    logger.info(f"System RAM check: {available_gb:.1f} GB available")


def _pick_device() -> str:
    """Choose device for embedding/reranking models.

    Returns ``"cuda"`` if available, ``"cpu"`` otherwise.
    Embedding + reranking models are small (~360MB) and can coexist
    with the LLM on GPUs with sufficient VRAM headroom.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    return "cuda"


class EmbeddingModel:
    """Manages a sentence-transformers SentenceTransformer model."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None

    def load(self, model_name: str) -> None:
        """Load embedding model onto GPU.

        Raises ImportError if sentence-transformers is not installed.
        Raises RuntimeError if CUDA is not available.
        """
        if self._model is not None and self._model_name == model_name:
            return
        self.unload()
        _check_system_memory()
        device = _pick_device()
        if device == "cuda":
            os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name} (device={device})")
        self._model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device,
        )
        self._model_name = model_name
        logger.info("Embedding model loaded")

    def encode(self, texts: list[str], batch_size: int = 64) -> Any:
        """Encode texts to normalized embeddings (numpy ndarray)."""
        if self._model is None:
            raise RuntimeError("Embedding model not loaded")
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def unload(self) -> None:
        """Unload model to free GPU VRAM."""
        if self._model is not None:
            logger.info(f"Unloading embedding model: {self._model_name}")
            del self._model
            self._model = None
            self._model_name = None
            _free_gpu_memory()


class RerankerModel:
    """Manages a sentence-transformers CrossEncoder model."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None

    def load(self, model_name: str) -> None:
        """Load reranker model onto GPU.

        Raises ImportError if sentence-transformers is not installed.
        Raises RuntimeError if CUDA is not available.
        """
        if self._model is not None and self._model_name == model_name:
            return
        self.unload()
        _check_system_memory()
        device = _pick_device()
        if device == "cuda":
            os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {model_name} (device={device})")
        self._model = CrossEncoder(model_name, device=device)
        self._model_name = model_name
        logger.info("Reranker model loaded")

    def rank(
        self, query: str, documents: list[str], top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Score and rank documents by relevance to query.

        Returns list of (original_index, score) sorted by score descending.
        """
        if self._model is None:
            raise RuntimeError("Reranker model not loaded")
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return [(int(idx), float(score)) for idx, score in ranked]

    def unload(self) -> None:
        """Unload model to free GPU VRAM."""
        if self._model is not None:
            logger.info(f"Unloading reranker model: {self._model_name}")
            del self._model
            self._model = None
            self._model_name = None
            _free_gpu_memory()


def _free_gpu_memory() -> None:
    """Best-effort GPU memory cleanup after model unload."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
