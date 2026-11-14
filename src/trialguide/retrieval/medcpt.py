"""Optional MedCPT encoders and reusable, validated trial-embedding indexes."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

DEFAULT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
DEFAULT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
# These immutable revisions identify the published safetensors commits on Hugging Face.
DEFAULT_QUERY_REVISION = "1aa262155a9128ad7d10a64fa782aeefc751f93d"
DEFAULT_ARTICLE_REVISION = "9de8685cd4789a21e222b44831b75dc78bf2e039"
SUPPORTED_COHORTS = ("sigir", "trec_2021", "trec_2022")
CACHE_FORMAT_VERSION = 1
_CACHE_METADATA = "metadata.json"
_CACHE_EMBEDDINGS = "article_embeddings.npy"
_MEMORY_ARTICLE_CACHE: OrderedDict[tuple, "MedCPTIndex"] = OrderedDict()
_MEMORY_ARTICLE_CACHE_LIMIT = 1
_QUERY_ENCODERS: dict[tuple, tuple[Callable, str]] = {}


@dataclass(frozen=True)
class MedCPTConfig:
    """Model IDs, immutable revisions, token limits, and output dimension."""

    query_model_id: str = DEFAULT_QUERY_MODEL
    article_model_id: str = DEFAULT_ARTICLE_MODEL
    query_revision: str = DEFAULT_QUERY_REVISION
    article_revision: str = DEFAULT_ARTICLE_REVISION
    query_max_length: int = 256
    article_max_length: int = 512
    embedding_dimension: int = 768

    # Reject incomplete model pinning and unusable tensor limits at configuration time.
    def __post_init__(self) -> None:
        """Require both model revisions and positive sequence and embedding sizes."""
        if not self.query_model_id or not self.article_model_id:
            raise ValueError("query_model_id and article_model_id must be non-empty")
        if not self.query_revision or not self.article_revision:
            raise ValueError("query_revision and article_revision must be immutable non-empty revisions")
        if self.query_max_length <= 0 or self.article_max_length <= 0:
            raise ValueError("token limits must be positive")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

    # Record tokenizer settings in one stable shape for cache validation.
    def tokenizer_settings(self) -> dict:
        """Return the tokenizer inputs that affect query and article embeddings."""
        return {
            "query": {
                "model_id": self.query_model_id,
                "revision": self.query_revision,
                "max_length": self.query_max_length,
                "truncation": True,
                "padding": True,
                "sentence_pair": False,
            },
            "article": {
                "model_id": self.article_model_id,
                "revision": self.article_revision,
                "max_length": self.article_max_length,
                "truncation": True,
                "padding": True,
                "sentence_pair": ["title", "text"],
            },
        }


@dataclass
class MedCPTIndex:
    """Article vectors aligned to the ordered trial IDs and validated cache metadata."""

    trial_ids: list[str]
    embeddings: np.ndarray
    metadata: dict

    # Return cached vectors in caller-supplied ID order without relying on row order.
    def vectors_for(self, trial_ids: list[str]) -> np.ndarray:
        """Select cached vectors by ID and preserve the requested order."""
        positions = {trial_id: row for row, trial_id in enumerate(self.trial_ids)}
        missing = [trial_id for trial_id in trial_ids if trial_id not in positions]
        if missing:
            raise ValueError(f"trial ID {missing[0]!r} is missing from the MedCPT index")
        return self.embeddings[[positions[trial_id] for trial_id in trial_ids]]


# Build query text using only visible sentence contents, in sentence-ID order.
def build_medcpt_query(note) -> str:
    """Format a VisibleNote without sentence-number prefixes or patient identifiers."""
    from trialguide.schemas import VisibleNote

    if not isinstance(note, VisibleNote):
        raise TypeError("MedCPT query construction accepts only a VisibleNote")
    return "\n".join(text for _, text in sorted(note.sentences.items()))


# Validate corpus documents before they can be fingerprinted or encoded.
def _validated_trials(trials: Iterable[dict]) -> list[dict]:
    """Validate required TrialGPT article fields and unique trial IDs."""
    prepared = []
    seen = set()
    for row, trial in enumerate(trials, start=1):
        if not isinstance(trial, dict):
            raise TypeError(f"trial record {row} must be a dictionary")
        trial_id = trial.get("_id")
        if not isinstance(trial_id, str) or not trial_id.strip():
            raise ValueError(f"trial record {row} _id must be a non-empty string")
        if trial_id in seen:
            raise ValueError(f"duplicate trial ID: {trial_id}")
        seen.add(trial_id)
        for field in ("title", "text"):
            value = trial.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"trial record {row} {field} must be a non-empty string")
        prepared.append(trial)
    return prepared


# Fingerprint the full canonical corpus so any changed record invalidates its index.
def _corpus_fingerprint(trials: list[dict]) -> str:
    """Hash canonical records in trial-ID order, independent of input row ordering."""
    canonical = json.dumps(
        sorted(trials, key=lambda trial: trial["_id"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# Resolve hardware only when an encoder is requested, keeping package import optional.
def _resolve_device(torch, device: str, fp16: bool) -> tuple[str, bool]:
    """Select CUDA or CPU and disable half precision on CPU."""
    cuda_available = bool(torch.cuda.is_available())
    if device == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    elif device == "cuda" or device.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(f"CUDA device {device!r} was requested but CUDA is unavailable")
        resolved = device
    elif device == "cpu":
        resolved = "cpu"
    else:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or a CUDA device such as 'cuda:0'")
    return resolved, bool(fp16 and resolved.startswith("cuda"))


# Load one encoder and return a reusable CLS-vector batch function.
def _load_hf_encoder(model_id: str, revision: str, max_length: int, device: str,
                     fp16: bool) -> tuple[Callable, str]:
    """Load a pinned HF encoder lazily and expose its tokenized CLS representations."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("MedCPT requires optional torch and transformers packages") from exc

    resolved_device, use_fp16 = _resolve_device(torch, device, fp16)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(
            model_id, revision=revision, use_safetensors=True
        ).to(resolved_device)
        if use_fp16:
            model = model.half()
        model.eval()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                f"MedCPT model load ran out of memory on {resolved_device}; use CPU or a smaller runtime"
            ) from exc
        raise

    def encode_batch(inputs: list) -> np.ndarray:
        """Tokenize and encode a batch, returning final-layer CLS vectors as float32."""
        try:
            encoded = tokenizer(
                inputs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(resolved_device)
            with torch.inference_mode():
                output = model(**encoded).last_hidden_state[:, 0, :]
            return output.detach().to(device="cpu", dtype=torch.float32).numpy()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    f"MedCPT encoding ran out of memory on {resolved_device}; reduce batch size or use CPU"
                ) from exc
            raise

    return encode_batch, resolved_device


# Encode ordered title/text pairs in bounded batches and validate the final matrix.
def _encode_article_batches(trials: list[dict], encoder: Callable, batch_size: int,
                            embedding_dimension: int) -> np.ndarray:
    """Batch paired trial articles and return a finite float32 matrix aligned to trials."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    blocks = []
    for start in range(0, len(trials), batch_size):
        batch = trials[start:start + batch_size]
        pairs = [[trial["title"], trial["text"]] for trial in batch]
        try:
            block = np.asarray(encoder(pairs), dtype=np.float32)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    f"MedCPT article batch at row {start + 1} ran out of memory; reduce batch_size"
                ) from exc
            raise
        expected_shape = (len(batch), embedding_dimension)
        if block.shape != expected_shape:
            raise ValueError(f"MedCPT article encoder returned shape {block.shape}; expected {expected_shape}")
        if not np.isfinite(block).all():
            raise ValueError(f"MedCPT article encoder returned non-finite values in batch at row {start + 1}")
        blocks.append(block)
    if not blocks:
        return np.empty((0, embedding_dimension), dtype=np.float32)
    return np.concatenate(blocks, axis=0)


# Hash persisted arrays so the JSON metadata acts as a commit marker for a complete cache.
def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one cache artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Atomically replace one JSON artifact after its complete temporary file is flushed.
def _atomic_write_json(path: Path, value: dict) -> None:
    """Write UTF-8 JSON by replacing the destination from a same-directory temporary file."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Atomically persist NumPy data without enabling pickle serialization.
def _atomic_write_embeddings(path: Path, embeddings: np.ndarray) -> None:
    """Save a float32 NPY array through a flushed temporary file and atomic replacement."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Build expected settings shared by cache writes and validation.
def _cache_metadata(trials: list[dict], cohort: str, config: MedCPTConfig,
                    embeddings: np.ndarray) -> dict:
    """Describe model pins, tokenizer settings, corpus identity, and vector alignment."""
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "cohort": cohort,
        "query_model_id": config.query_model_id,
        "query_revision": config.query_revision,
        "article_model_id": config.article_model_id,
        "article_revision": config.article_revision,
        "tokenizer_configuration": config.tokenizer_settings(),
        "query_max_length": config.query_max_length,
        "article_max_length": config.article_max_length,
        "article_input": ["title", "text"],
        "embedding_dimension": config.embedding_dimension,
        "dtype": str(embeddings.dtype),
        "trial_count": len(trials),
        "trial_ids": [trial["_id"] for trial in trials],
        "corpus_fingerprint": _corpus_fingerprint(trials),
    }


# Reject an index/config pairing that would compare vectors from different encoders.
def _validate_index_config(index: MedCPTIndex, config: MedCPTConfig) -> None:
    """Ensure a scorer's query encoder settings match the index's recorded settings."""
    checks = {
        "query_model_id": config.query_model_id,
        "query_revision": config.query_revision,
        "article_model_id": config.article_model_id,
        "article_revision": config.article_revision,
        "tokenizer_configuration": config.tokenizer_settings(),
        "query_max_length": config.query_max_length,
        "article_max_length": config.article_max_length,
        "embedding_dimension": config.embedding_dimension,
    }
    for field, value in checks.items():
        if index.metadata.get(field) != value:
            raise ValueError(f"MedCPT index configuration mismatch: {field}")


# Build and atomically save the article vectors for one shared cohort corpus.
def build_corpus_index(trials: Iterable[dict], cohort: str, cache_dir: Path,
                       config: MedCPTConfig = MedCPTConfig(), *, batch_size: int = 16,
                       device: str = "auto", fp16: bool = False) -> MedCPTIndex:
    """Encode and persist a cohort's shared TrialGPT corpus once, using paired title/text inputs."""
    if cohort not in SUPPORTED_COHORTS:
        raise ValueError(f"unsupported MedCPT cohort {cohort!r}; expected one of {SUPPORTED_COHORTS}")
    prepared = _validated_trials(trials)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / _CACHE_METADATA
    embedding_path = directory / _CACHE_EMBEDDINGS
    if metadata_path.exists() or embedding_path.exists():
        raise FileExistsError(f"MedCPT cache artifacts already exist in {directory}; load or choose another directory")

    if prepared:
        encoder, resolved_device = _load_hf_encoder(
            config.article_model_id,
            config.article_revision,
            config.article_max_length,
            device,
            fp16,
        )
        embeddings = _encode_article_batches(prepared, encoder, batch_size, config.embedding_dimension)
        del encoder
        gc.collect()
        try:
            import torch
            if resolved_device.startswith("cuda"):
                torch.cuda.empty_cache()
        except ImportError:
            pass
    else:
        embeddings = np.empty((0, config.embedding_dimension), dtype=np.float32)

    metadata = _cache_metadata(prepared, cohort, config, embeddings)
    _atomic_write_embeddings(embedding_path, embeddings)
    metadata["embedding_sha256"] = _file_sha256(embedding_path)
    _atomic_write_json(metadata_path, metadata)
    return MedCPTIndex(metadata["trial_ids"], embeddings, metadata)


# Load a cache without importing torch, transformers, or model weights.
def load_corpus_index(cache_dir: Path, cohort: str, config: MedCPTConfig = MedCPTConfig(), *,
                      trials: Iterable[dict] | None = None) -> MedCPTIndex:
    """Load an NPY/JSON index and validate its cohort, model pins, corpus, shape, and digest."""
    if cohort not in SUPPORTED_COHORTS:
        raise ValueError(f"unsupported MedCPT cohort {cohort!r}; expected one of {SUPPORTED_COHORTS}")
    directory = Path(cache_dir)
    metadata_path = directory / _CACHE_METADATA
    embedding_path = directory / _CACHE_EMBEDDINGS
    if not metadata_path.is_file() or not embedding_path.is_file():
        raise ValueError(f"incomplete MedCPT cache at {directory}: expected metadata.json and article_embeddings.npy")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MedCPT cache metadata {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid MedCPT cache metadata {metadata_path}: expected a JSON object")

    expected = {
        "format_version": CACHE_FORMAT_VERSION,
        "cohort": cohort,
        "query_model_id": config.query_model_id,
        "query_revision": config.query_revision,
        "article_model_id": config.article_model_id,
        "article_revision": config.article_revision,
        "tokenizer_configuration": config.tokenizer_settings(),
        "query_max_length": config.query_max_length,
        "article_max_length": config.article_max_length,
        "article_input": ["title", "text"],
        "embedding_dimension": config.embedding_dimension,
        "dtype": "float32",
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"incompatible MedCPT cache {metadata_path}: {field} does not match requested configuration")

    trial_ids = metadata.get("trial_ids")
    trial_count = metadata.get("trial_count")
    if (not isinstance(trial_ids, list) or len(trial_ids) != trial_count
            or any(not isinstance(trial_id, str) or not trial_id.strip() for trial_id in trial_ids)
            or len(set(trial_ids)) != len(trial_ids)):
        raise ValueError(f"invalid trial-ID alignment in MedCPT cache metadata {metadata_path}")
    if trials is not None:
        prepared = _validated_trials(trials)
        if len(prepared) != trial_count or _corpus_fingerprint(prepared) != metadata.get("corpus_fingerprint"):
            raise ValueError("MedCPT cache corpus fingerprint does not match the supplied trial corpus")

    if not isinstance(metadata.get("corpus_fingerprint"), str) or len(metadata["corpus_fingerprint"]) != 64:
        raise ValueError(f"invalid corpus fingerprint in MedCPT cache metadata {metadata_path}")
    if metadata.get("embedding_sha256") != _file_sha256(embedding_path):
        raise ValueError(f"MedCPT embedding checksum mismatch for {embedding_path}")
    try:
        embeddings = np.load(embedding_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load MedCPT embeddings {embedding_path}: {exc}") from exc
    expected_shape = (trial_count, config.embedding_dimension)
    if embeddings.shape != expected_shape:
        raise ValueError(f"invalid MedCPT embedding shape {embeddings.shape}; expected {expected_shape}")
    if embeddings.dtype != np.dtype("float32") or not np.isfinite(embeddings).all():
        raise ValueError("MedCPT embeddings must be finite float32 values")
    return MedCPTIndex(trial_ids, embeddings, metadata)


# Reuse a valid index or build it only when the requested cache directory is empty.
def get_or_build_corpus_index(trials: Iterable[dict], cohort: str, cache_dir: Path,
                              config: MedCPTConfig = MedCPTConfig(), *, batch_size: int = 16,
                              device: str = "auto", fp16: bool = False) -> MedCPTIndex:
    """Load a matching cache without model loading, otherwise build a new index."""
    directory = Path(cache_dir)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"MedCPT cache path is not a directory: {directory}")
    if directory.is_dir() and any(directory.iterdir()):
        return load_corpus_index(directory, cohort, config, trials=trials)
    return build_corpus_index(
        trials,
        cohort,
        directory,
        config,
        batch_size=batch_size,
        device=device,
        fp16=fp16,
    )


# Load the query encoder once per process and keep device placement consistent.
def _query_encoder(config: MedCPTConfig, device: str, fp16: bool) -> tuple[Callable, str]:
    """Get a process-cached query encoder for the requested model revision and device."""
    key = (config.query_model_id, config.query_revision, config.query_max_length, device, fp16)
    if key not in _QUERY_ENCODERS:
        _QUERY_ENCODERS[key] = _load_hf_encoder(
            config.query_model_id,
            config.query_revision,
            config.query_max_length,
            device,
            fp16,
        )
    return _QUERY_ENCODERS[key]


# Embed a query with the pinned query tokenizer and return its CLS vector.
def _encode_query(query_text: str, config: MedCPTConfig, device: str, fp16: bool) -> np.ndarray:
    """Encode one non-empty query and validate its finite vector shape."""
    if not query_text.strip():
        return np.empty((0,), dtype=np.float32)
    encoder, _ = _query_encoder(config, device, fp16)
    vector = np.asarray(encoder([query_text]), dtype=np.float32)
    expected_shape = (1, config.embedding_dimension)
    if vector.shape != expected_shape or not np.isfinite(vector).all():
        raise ValueError(f"MedCPT query encoder returned invalid output; expected finite shape {expected_shape}")
    return vector[0]


# Score a query against an index by dot product while preserving caller trial order.
def _score_index(query_text: str, trial_ids: list[str], index: MedCPTIndex,
                 config: MedCPTConfig, device: str, fp16: bool) -> list[float]:
    """Return deterministic inner-product scores mapped to the requested trial IDs."""
    if not query_text.strip() or not trial_ids:
        return []
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("duplicate trial IDs supplied for MedCPT scoring")
    query_vector = _encode_query(query_text, config, device, fp16)
    article_vectors = index.vectors_for(trial_ids)
    if article_vectors.shape[1] != len(query_vector):
        raise ValueError("MedCPT query and article embedding dimensions do not match")
    return [float(score) for score in article_vectors @ query_vector]


class MedCPTIndexScorer:
    """Callable retrieval adapter that scores indexed trials by ID, never re-encoding articles."""

    def __init__(self, index: MedCPTIndex, config: MedCPTConfig = MedCPTConfig(), *,
                 device: str = "auto", fp16: bool = False):
        """Bind a loaded cache to its pinned query encoder configuration."""
        _validate_index_config(index, config)
        self.index = index
        self.config = config
        self.device = device
        self.fp16 = fp16

    # Validate the supplied shared corpus before using its ID-aligned vectors.
    def score_trials(self, query_text: str, trials: list[dict]) -> list[float]:
        """Check corpus identity and score the requested corpus using its cached article vectors."""
        prepared = _validated_trials(trials)
        if _corpus_fingerprint(prepared) != self.index.metadata.get("corpus_fingerprint"):
            raise ValueError("MedCPT index does not match the trials supplied to retrieval")
        return _score_index(
            query_text,
            [trial["_id"] for trial in prepared],
            self.index,
            self.config,
            self.device,
            self.fp16,
        )

    # Keep compatibility with the scorer callable protocol used by retrieve_trials.
    def __call__(self, query_text: str, trial_texts: list[str]) -> list[float]:
        """Reject text-only calls because an index scorer requires IDs for safe alignment."""
        raise TypeError("MedCPTIndexScorer must be used through retrieve_trials() with trial dictionaries")


# Construct an adapter accepted by the existing retrieve_trials scorer hook.
def make_medcpt_scorer(index: MedCPTIndex, config: MedCPTConfig = MedCPTConfig(), *,
                       device: str = "auto", fp16: bool = False) -> MedCPTIndexScorer:
    """Create an indexed MedCPT scorer that reuses cached article vectors."""
    return MedCPTIndexScorer(index, config, device=device, fp16=fp16)


# Encode a corpus only once per process for the legacy no-scorer retrieval path.
def score_medcpt_trials(query_text: str, trials: list[dict], config: MedCPTConfig = MedCPTConfig(), *,
                        batch_size: int = 16, device: str = "auto", fp16: bool = False) -> list[float]:
    """Use a bounded in-memory article cache, then score a new query with its query encoder."""
    if not query_text.strip() or not trials:
        return []
    prepared = _validated_trials(trials)
    fingerprint = _corpus_fingerprint(prepared)
    key = (fingerprint, config, batch_size, device, fp16)
    index = _MEMORY_ARTICLE_CACHE.get(key)
    if index is None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        encoder, _ = _load_hf_encoder(
            config.article_model_id,
            config.article_revision,
            config.article_max_length,
            device,
            fp16,
        )
        embeddings = _encode_article_batches(prepared, encoder, batch_size, config.embedding_dimension)
        del encoder
        gc.collect()
        try:
            import torch
            if device.startswith("cuda") or (device == "auto" and torch.cuda.is_available()):
                torch.cuda.empty_cache()
        except ImportError:
            pass
        metadata = _cache_metadata(prepared, "in_memory", config, embeddings)
        index = MedCPTIndex(metadata["trial_ids"], embeddings, metadata)
        _MEMORY_ARTICLE_CACHE[key] = index
        while len(_MEMORY_ARTICLE_CACHE) > _MEMORY_ARTICLE_CACHE_LIMIT:
            _MEMORY_ARTICLE_CACHE.popitem(last=False)
    else:
        _MEMORY_ARTICLE_CACHE.move_to_end(key)
    return _score_index(
        query_text,
        [trial["_id"] for trial in prepared],
        index,
        config,
        device,
        fp16,
    )


__all__ = [
    "MedCPTConfig",
    "MedCPTIndex",
    "MedCPTIndexScorer",
    "build_medcpt_query",
    "build_corpus_index",
    "load_corpus_index",
    "get_or_build_corpus_index",
    "make_medcpt_scorer",
    "score_medcpt_trials",
    "DEFAULT_QUERY_MODEL",
    "DEFAULT_ARTICLE_MODEL",
    "DEFAULT_QUERY_REVISION",
    "DEFAULT_ARTICLE_REVISION",
]
