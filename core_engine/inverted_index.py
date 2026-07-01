"""
Memory-optimized Inverted Index for ~3M documents on 16GB RAM.

Key optimizations over the naive nested-dict approach:
- String doc_ids stored ONCE in a list; postings use 4-byte integer IDs.
- Posting lists packed as binary (7 bytes/posting vs ~80 bytes as Python tuples).
- Shards spilled to disk under memory pressure; merged as raw bytes (not Python objects).
- Pickle serialization (3-10x smaller than JSON).
- Shard files kept until final save completes (crash-safe).
"""

import gc
import os
import pickle
import struct
import tempfile
from collections import Counter, defaultdict

import psutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIELD_TO_ID = {"title": 0, "abstract": 1}
ID_TO_FIELD = {0: "title", 1: "abstract"}

_TOTAL_RAM = psutil.virtual_memory().total
DEFAULT_MEMORY_CEILING_BYTES = min(10 * (1 << 30), int(_TOTAL_RAM * 0.65))
GC_INTERVAL_DOCS = 50_000

# Each posting: (doc_int_id: uint32, field_id: uint8, count: uint16) = 7 bytes
_POSTING_FMT = struct.Struct("<IBH")
_POSTING_SIZE = _POSTING_FMT.size  # 7


def _rss_bytes():
    return psutil.Process(os.getpid()).memory_info().rss


def _rss_gb():
    return _rss_bytes() / (1 << 30)


def _pack_postings(postings):
    """Pack list[tuple[int,int,int]] → bytes."""
    buf = bytearray(len(postings) * _POSTING_SIZE)
    for i, (d, f, c) in enumerate(postings):
        _POSTING_FMT.pack_into(buf, i * _POSTING_SIZE, d, f, c)
    return bytes(buf)


def _unpack_postings(data):
    """Unpack bytes → list[tuple[int,int,int]]."""
    n = len(data) // _POSTING_SIZE
    return [_POSTING_FMT.unpack_from(data, i * _POSTING_SIZE) for i in range(n)]


class InvertedIndex:
    def __init__(self, memory_ceiling_bytes=DEFAULT_MEMORY_CEILING_BYTES):
        # --- Build-time (list[tuple] for easy appending) ---
        self._build_index: dict[str, list[tuple]] = defaultdict(list)

        # --- Packed index (bytes, used after load / save) ---
        self._packed_index: dict[str, bytes] = {}

        # --- Doc ID mapping ---
        self.doc_id_to_int: dict[str, int] = {}
        self.int_to_doc_id: list[str] = []

        # --- Doc lengths: int_id → (title_len, abstract_len) ---
        self.doc_lengths: dict[int, tuple[int, int]] = {}

        # --- Build-time bookkeeping ---
        self._memory_ceiling = memory_ceiling_bytes
        self._shard_paths: list[str] = []
        self._docs_added = 0

        # --- Compatibility proxy for LexicalSearcher ---
        self.document_lengths: _DocLengthProxy | None = None

    # ------------------------------------------------------------------
    # Public: Indexing
    # ------------------------------------------------------------------
    def add_document(self, doc_id: str, field_name: str, tokens: list[str]):
        int_id = self._get_or_create_doc_int(doc_id)
        field_id = FIELD_TO_ID[field_name]

        # Store document length
        existing = self.doc_lengths.get(int_id, (0, 0))
        if field_id == 0:
            self.doc_lengths[int_id] = (len(tokens), existing[1])
        else:
            self.doc_lengths[int_id] = (existing[0], len(tokens))

        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            self._build_index[term].append((int_id, field_id, min(count, 65535)))

        self._docs_added += 1
        if self._docs_added % GC_INTERVAL_DOCS == 0:
            gc.collect()
            if _rss_bytes() > self._memory_ceiling:
                print(
                    f"  ⚠ RSS={_rss_gb():.1f}GB exceeds ceiling "
                    f"({self._memory_ceiling / (1 << 30):.1f}GB). "
                    f"Spilling shard to disk..."
                )
                self._spill_shard()

    # ------------------------------------------------------------------
    # Internal: Shard management
    # ------------------------------------------------------------------
    def _get_or_create_doc_int(self, doc_id: str) -> int:
        int_id = self.doc_id_to_int.get(doc_id)
        if int_id is None:
            int_id = len(self.int_to_doc_id)
            self.doc_id_to_int[doc_id] = int_id
            self.int_to_doc_id.append(doc_id)
        return int_id

    def _spill_shard(self):
        """Pack current build_index to bytes and write to a temp shard file."""
        packed_shard = {}
        for term, postings in self._build_index.items():
            packed_shard[term] = _pack_postings(postings)

        shard_path = tempfile.mktemp(suffix=".shard.pkl", prefix="idx_")
        with open(shard_path, "wb") as f:
            pickle.dump(packed_shard, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._shard_paths.append(shard_path)
        self._build_index.clear()
        self._build_index = defaultdict(list)
        del packed_shard
        gc.collect()
        print(f"    Shard written: {shard_path} | RSS after spill: {_rss_gb():.1f}GB")

    # ------------------------------------------------------------------
    # Public: Save / Load
    # ------------------------------------------------------------------
    def save_to_disk(self, path: str):
        """
        Merge all shards + remaining build_index as PACKED BYTES, then save.

        Memory profile during merge:
          - merged dict[str, bytearray] grows to ~1.5-2 GB (compact binary)
          - one shard loaded at a time adds ~20-50 MB
          - Peak ≈ 2.5 GB — safe on 16 GB RAM
        """
        print("  Packing remaining in-memory postings...")
        merged: dict[str, bytearray] = {}
        for term, postings in self._build_index.items():
            merged[term] = bytearray(_pack_postings(postings))
        self._build_index.clear()
        self._build_index = defaultdict(list)
        gc.collect()

        if self._shard_paths:
            print(f"  Merging {len(self._shard_paths)} shard(s) as packed bytes...")
            for i, shard_path in enumerate(self._shard_paths):
                with open(shard_path, "rb") as f:
                    shard: dict[str, bytes] = pickle.load(f)
                for term, packed in shard.items():
                    if term in merged:
                        merged[term].extend(packed)
                    else:
                        merged[term] = bytearray(packed)
                del shard
                gc.collect()
                if (i + 1) % 10 == 0 or i == len(self._shard_paths) - 1:
                    print(
                        f"    Merged {i + 1}/{len(self._shard_paths)} shards | "
                        f"RSS: {_rss_gb():.1f}GB"
                    )

        # Convert bytearrays → bytes (saves ~12 bytes per entry, releases overalloc)
        packed_index = {term: bytes(buf) for term, buf in merged.items()}
        del merged
        gc.collect()

        data = {
            "int_to_doc_id": self.int_to_doc_id,
            "packed_index": packed_index,
            "doc_lengths": self.doc_lengths,
        }

        # Estimate output size and check disk space
        total_posting_bytes = sum(len(v) for v in packed_index.values())
        estimated_size = int(total_posting_bytes * 1.3)  # ~30% overhead for pickle framing
        dest_dir = os.path.dirname(os.path.abspath(path)) or "."
        free_bytes = os.statvfs(dest_dir).f_bavail * os.statvfs(dest_dir).f_frsize
        if estimated_size > free_bytes:
            raise OSError(
                f"Not enough disk space to save index. "
                f"Need ~{estimated_size / (1 << 30):.1f} GB, "
                f"have {free_bytes / (1 << 30):.1f} GB free in {dest_dir}. "
                f"Free up space and call save_to_disk() again (no re-index needed)."
            )

        # Atomic save: write to temp file, then rename
        tmp_path = path + ".tmp"
        print(f"  Writing to {tmp_path} (~{estimated_size / (1 << 30):.1f} GB)...")
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

        # Only delete shards AFTER successful save
        for sp in self._shard_paths:
            try:
                os.unlink(sp)
            except OSError:
                pass
        self._shard_paths.clear()

        n_terms = len(packed_index)
        n_postings = sum(len(v) // _POSTING_SIZE for v in packed_index.values())
        file_mb = os.path.getsize(path) / (1 << 20)
        print(
            f"  ✓ Index saved: {len(self.int_to_doc_id):,} docs, "
            f"{n_terms:,} terms, {n_postings:,} postings, "
            f"{file_mb:.1f} MB on disk."
        )

        # Build proxy for immediate use
        self._packed_index = packed_index
        self.document_lengths = _DocLengthProxy(
            self.doc_lengths, self.int_to_doc_id, self.doc_id_to_int
        )

    def load_from_disk(self, path: str):
        #print(f"Loading index from {path}...")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.int_to_doc_id = data["int_to_doc_id"]
        self.doc_id_to_int = {did: i for i, did in enumerate(self.int_to_doc_id)}
        self.doc_lengths = data["doc_lengths"]
        self._packed_index = data["packed_index"]

        del data
        gc.collect()

        self.document_lengths = _DocLengthProxy(
            self.doc_lengths, self.int_to_doc_id, self.doc_id_to_int
        )

        n_terms = len(self._packed_index)
        n_postings = sum(
            len(v) // _POSTING_SIZE for v in self._packed_index.values()
        )
        """print(
            f"  ✓ Loaded: {len(self.int_to_doc_id):,} docs, "
            f"{n_terms:,} terms, {n_postings:,} postings. "
            f"RSS: {_rss_gb():.1f}GB"
        )"""

    # ------------------------------------------------------------------
    # Public: Query-time posting access
    # ------------------------------------------------------------------
    def get_postings(self, term: str):
        """Return unpacked postings for a term: list[(doc_int_id, field_id, count)]."""
        buf = self._packed_index.get(term)
        if buf is None:
            return []
        return _unpack_postings(buf)

    def get_postings_count(self, term: str) -> int:
        """Number of postings for a term (for IDF)."""
        buf = self._packed_index.get(term)
        if buf is None:
            return 0
        return len(buf) // _POSTING_SIZE

    @property
    def index(self):
        """Compatibility: returns the packed index dict for `term in self.index`."""
        return self._packed_index


# ---------------------------------------------------------------------------
# Compatibility proxy for LexicalSearcher
# ---------------------------------------------------------------------------
class _DocLengthProxy:
    """Maps string doc_id → {field_name: length} without duplicating strings."""

    def __init__(self, doc_lengths, int_to_doc_id, doc_id_to_int):
        self._dl = doc_lengths
        self._i2d = int_to_doc_id
        self._d2i = doc_id_to_int

    def __len__(self):
        return len(self._dl)

    def __contains__(self, doc_id):
        return doc_id in self._d2i

    def __getitem__(self, doc_id):
        int_id = self._d2i[doc_id]
        tl, al = self._dl[int_id]
        return {"title": tl, "abstract": al}

    def items(self):
        for int_id, (tl, al) in self._dl.items():
            yield self._i2d[int_id], {"title": tl, "abstract": al}

    def __iter__(self):
        for int_id in self._dl:
            yield self._i2d[int_id]

    def get(self, doc_id, default=None):
        int_id = self._d2i.get(doc_id)
        if int_id is None:
            return default
        tl, al = self._dl[int_id]
        return {"title": tl, "abstract": al}
