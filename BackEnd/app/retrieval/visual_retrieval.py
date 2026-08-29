"""Module 3.1 — Visual Retrieval Tools.

Cung cấp 4 tool deterministic (không LLM, không reasoning):
``frame_search``, ``shot_search``, ``clip_search``,
``image_similarity_search`` (alias tường minh của
``frame_search(image_ref=...)``, khớp đúng tên tool liệt kê trong
``ProposalOnlinePipeline.md`` §7.1). Mỗi tool nhận vào 1 câu query (text
hoặc ảnh), encode thành vector CLIP, tìm trong FAISS index tương ứng rồi
resolve kết quả về ``SearchHit[]`` — contract chuẩn dùng chung cho mọi
retrieval tool trong hệ thống (xem ``ProposalOnlinePipeline.md`` §7).

Thiết kế gốc: ``Markdown_Doc/module_visual_retrieval_tools.md``.

So với thiết kế gốc, phần implement này có 2 điều chỉnh quan trọng:

1. ``FaissIndexRegistry`` chỉ trả về ``(faiss_id, raw_score)`` thô — việc
   resolve ``faiss_id -> canonical reference`` được gộp thành **đúng 1 câu
   JOIN** trong ``PostgreManager`` (``get_frame_hits_by_faiss_ids`` /
   ``get_clip_hits_by_faiss_ids`` / ``get_shot_hits_by_faiss_ids``), thay vì
   2 bước "lookup mapping theo faiss_id" rồi "lookup canonical ref theo
   entity_id cho từng hit" như bản thiết kế — bước 2 đó là N+1 query, ngược
   với mục tiêu "1 query DB duy nhất" mà chính thiết kế gốc đề ra.
2. ``shot_search()`` cần thêm ``model_version`` và ``pooling_method`` khi
   resolve (bảng ``ShotEmbeddingRecord`` unique theo cả 2 trường này, xem
   ``sql_models.py``), nên ``VisualSearchConfig`` có thêm 2 trường so với
   bản thiết kế ban đầu.

Ba tool dùng chung 1 pipeline nội bộ 2 bước:
``_search_and_rank()`` (gọi FAISS, đánh rank) rồi mỗi tool tự resolve +
dựng ``SearchHit`` theo đúng entity type của mình (không ép chung 1 hàm
generic — mỗi entity type có shape canonical ref khác nhau: frame có
``frame_id``, clip có ``clip_id``+``shot_id``, shot chỉ có ``shot_id``).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from BackEnd.app.contracts.models import SearchHit
from BackEnd.app.Database.postgre_manager import PostgreManager

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Registry — quản lý vòng đời của 3 FAISS index (singleton)
# ---------------------------------------------------------------------------


class FaissIndexRegistry:
    """Load và giữ 3 FAISS index (frame/clip/shot) trong RAM suốt vòng đời
    service.

    Đây là nơi DUY NHẤT trong hệ thống được phép gọi ``faiss.Index.search()``
    trực tiếp. Mọi tool ở tầng trên (``VisualRetrievalTools``) PHẢI gọi qua
    ``search_frame_index`` / ``search_clip_index`` / ``search_shot_index``
    bên dưới, không tự ý gọi ``index.search()`` ở nơi khác. Tách lớp này để:

    1. Test được phần "gọi FAISS đúng" độc lập, không cần mock DB.
    2. Nếu sau này đổi loại index (vd ``IndexFlatIP`` -> ``IndexIVFFlat`` để
       tăng tốc), chỉ sửa trong lớp này, không đụng tới ``VisualRetrievalTools``.

    An toàn cho việc ĐỌC đồng thời (FAISS ``IndexFlatIP`` không có state ghi
    trong lúc serving). Việc khởi tạo singleton dùng double-checked locking
    để tránh load trùng 3 index khi nhiều request đầu tiên tới cùng lúc.
    """

    _instance: "FaissIndexRegistry | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, index_dir: str | None = None) -> None:
        resolved_dir = index_dir or os.getenv("FAISS_INDEX_DIR")
        if not resolved_dir:
            raise RuntimeError(
                "FAISS_INDEX_DIR chưa được cấu hình. "
                "Set biến môi trường FAISS_INDEX_DIR hoặc truyền index_dir trực tiếp."
            )

        self.index_dir = resolved_dir
        self.frame_index: "faiss.Index" = self._load("frame.faiss")
        self.clip_index: "faiss.Index" = self._load("clip.faiss")
        self.shot_index: "faiss.Index" = self._load("shot.faiss")

    def _load(self, filename: str) -> "faiss.Index":
        import faiss  # import trễ để module này import được kể cả khi chưa cài faiss

        path = os.path.join(self.index_dir, filename)
        try:
            return faiss.read_index(path)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Không đọc được FAISS index tại '{path}'. "
                f"Kiểm tra biến môi trường FAISS_INDEX_DIR và đảm bảo file "
                f".faiss đã được tải về (xem §8 module_visual_retrieval_tools.md)."
            ) from exc

    @classmethod
    def get_instance(cls, index_dir: str | None = None) -> "FaissIndexRegistry":
        """Lazy singleton — gọi 1 lần lúc app startup, các nơi khác dùng lại
        instance này thay vì tự tạo mới (tránh load lại 3 index tốn RAM/thời
        gian mỗi request)."""

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(index_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Xoá singleton hiện tại. CHỈ dùng trong test để cô lập giữa các
        test case (mỗi test cần load fixture index riêng)."""

        with cls._instance_lock:
            cls._instance = None

    # -- Hàm truy vấn FAISS (điểm chốt duy nhất gọi index.search()) --------

    @staticmethod
    def _raw_search(
        index: "faiss.Index", vector: np.ndarray, top_k: int
    ) -> list[tuple[int, float]]:
        """Gọi FAISS search thô cho 1 index bất kỳ.

        Input:
            index: 1 trong self.frame_index / self.clip_index / self.shot_index.
            vector: vector query đã L2-normalize, shape (512,) hoặc (1, 512).
            top_k: số kết quả mong muốn.

        Output:
            list[(faiss_id, raw_score)] đã lọc bỏ slot rỗng, giữ nguyên thứ
            tự FAISS trả về (score giảm dần — FAISS tự sort, không cần sort lại).
        """

        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        scores, faiss_ids = index.search(vector, top_k)

        return [
            (int(fid), float(score))
            for fid, score in zip(faiss_ids[0], scores[0])
            if fid != -1  # FAISS trả -1 khi index có ít hơn top_k vector
        ]

    @staticmethod
    def _raw_subset_search(
        index: "faiss.Index",
        vector: np.ndarray,
        faiss_ids: list[int],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Exact inner-product search over every vector in an allowed ID subset."""

        if not faiss_ids or top_k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        ids = np.asarray(list(dict.fromkeys(faiss_ids)), dtype=np.int64)
        try:
            vectors = index.reconstruct_batch(ids)
        except (AttributeError, RuntimeError, TypeError):
            vectors = np.stack([index.reconstruct(int(faiss_id)) for faiss_id in ids])
        matrix = np.asarray(vectors, dtype=np.float32)
        scores = matrix @ query
        count = min(top_k, len(ids))
        if count == len(ids):
            selected = np.arange(len(ids))
        else:
            selected = np.argpartition(scores, -count)[-count:]
        ordered = sorted(selected, key=lambda idx: (-float(scores[idx]), int(ids[idx])))
        return [(int(ids[idx]), float(scores[idx])) for idx in ordered]

    def search_frame_index(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Truy vấn ``frame.faiss``. Dùng bởi ``VisualRetrievalTools.frame_search()``."""

        return self._raw_search(self.frame_index, vector, top_k)

    def search_clip_index(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Truy vấn ``clip.faiss``. Dùng bởi ``VisualRetrievalTools.clip_search()``."""

        return self._raw_search(self.clip_index, vector, top_k)

    def search_shot_index(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Truy vấn ``shot.faiss`` (đã pooled sẵn ở Offline). Dùng bởi
        ``VisualRetrievalTools.shot_search()``."""

        return self._raw_search(self.shot_index, vector, top_k)

    def search_frame_subset(
        self, vector: np.ndarray, faiss_ids: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        return self._raw_subset_search(self.frame_index, vector, faiss_ids, top_k)

    def search_clip_subset(
        self, vector: np.ndarray, faiss_ids: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        return self._raw_subset_search(self.clip_index, vector, faiss_ids, top_k)

    def search_shot_subset(
        self, vector: np.ndarray, faiss_ids: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        return self._raw_subset_search(self.shot_index, vector, faiss_ids, top_k)


class _SearchIndexFn(Protocol):
    """Kiểu của 3 hàm ``search_*_index`` — dùng để type-hint tham số
    ``search_fn`` trong ``VisualRetrievalTools`` mà không phải import
    ``FaissIndexRegistry`` làm tham số cụ thể (dễ mock trong test)."""

    def __call__(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]: ...


# ---------------------------------------------------------------------------
# 2. Embedder — encode text/image thành vector CLIP
# ---------------------------------------------------------------------------


DEFAULT_TEXT_EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_FAISS_INDEX_MODEL_NAME = "clip-ViT-B-32"


class ClipEmbedder:
    """Encode text/image into the vector space of the existing CLIP FAISS indexes.

    The multilingual text model is distilled into the original CLIP ViT-B/32
    embedding space. Therefore Vietnamese text can query the existing image
    indexes without rebuilding them. Image-to-image retrieval still uses the
    original CLIP checkpoint because the multilingual checkpoint only provides
    a text encoder.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL_NAME,
        device: str = "cpu",
        image_model_name: str = DEFAULT_FAISS_INDEX_MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self.image_model_name = image_model_name
        self.device = device
        self._text_model = self._load_model(model_name, device)
        self._image_model = self._text_model if model_name == image_model_name else None

    def _load_model(self, model_name: str, device: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu thư viện 'sentence-transformers'. "
                "Cài đặt qua 'uv add sentence-transformers pillow' trước khi "
                "dùng ClipEmbedder."
            ) from exc
        return SentenceTransformer(model_name, device=device)

    def encode_text(self, text: str) -> np.ndarray:
        """Trả về vector đã L2-normalize, dtype float32, shape (512,)."""

        vector = self._text_model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return vector.astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode an ordered text batch in one model call."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._text_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_image(self, image_path: str) -> np.ndarray:
        """Trả về vector đã L2-normalize, dtype float32, shape (512,)."""

        from PIL import Image

        if self._image_model is None:
            self._image_model = self._load_model(self.image_model_name, self.device)
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            vector = self._image_model.encode(
                image, convert_to_numpy=True, normalize_embeddings=True
            )
        return vector.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Cấu hình tìm kiếm
# ---------------------------------------------------------------------------

# Safety ceiling — cứng, không vượt quá bất kể caller yêu cầu gì.
# FAISS IndexFlatIP search 1000/366k vectors ~1.5ms, an toàn cho production.
ABSOLUTE_MAX_TOP_K = 1000


@dataclass(frozen=True)
class VisualSearchConfig:
    """Cấu hình cho 1 instance ``VisualRetrievalTools``.

    ``default_top_k`` là giá trị mặc định khi caller không truyền top_k.
    Caller có thể truyền bất kỳ giá trị nào, nhưng sẽ bị cap bởi
    ``ABSOLUTE_MAX_TOP_K`` (1000) — thay vì bị cap bởi ``max_top_k``
    cũ (100) vốn quá thấp cho nhiều trường hợp truy xuất.

    ``index_version``/``model_version`` mặc định lấy đúng giá trị đang có
    trong dump dữ liệu thật (``Markdown_Doc/aic_hcmc_full.sql``): cả 3 bảng
    ``*EmbeddingRecord`` đều dùng ``index_version=0`` (KHÔNG phải 1 —
    ``module_visual_retrieval_tools.md`` §0 ghi "index_version = 1" là số
    giả định lúc thiết kế, chưa đối chiếu dữ liệu thật), và
    ``ShotEmbeddingRecord.model_version`` là chuỗi ``"0"`` (không phải
    ``"v1"``). Dùng sai 2 giá trị này khiến MỌI hit đều không resolve được
    (âm thầm bị bỏ qua ở ``_log_unresolved``, không raise lỗi nào) — xem
    ``module_visual_retrieval_tools_implementation.md`` §7.
    """

    # This identifies the model used when the existing FAISS index was built.
    # It deliberately differs from the multilingual model used for text queries.
    model_name: str = DEFAULT_FAISS_INDEX_MODEL_NAME
    model_version: str = "0"
    pooling_method: str = "mean"
    index_version: int = 0
    default_top_k: int = 200
    max_top_k: int = ABSOLUTE_MAX_TOP_K


VisualRetriever = Literal["frame_search", "clip_search", "shot_search"]


@dataclass(frozen=True)
class VisualSearchRequest:
    """One text query in an ordered native visual retrieval batch."""

    retriever: VisualRetriever
    query: str
    top_k: int | None = None
    event_id: str | None = None
    tool_call_id: str | None = None
    video_ids: list[str] | None = None
    start_ms: int | None = None
    end_ms: int | None = None


# ---------------------------------------------------------------------------
# 4. VisualRetrievalTools — 3 tool chính
# ---------------------------------------------------------------------------


class VisualRetrievalTools:
    """Module 3.1 — Visual Retrieval Tools.

    Input: text query hoặc ``image_ref``. Output: ``SearchHit[]``, đã resolve
    đầy đủ canonical reference. Không merge, không rank liên-nguồn, không
    gọi VLM — những việc đó thuộc Module 4/5/7.
    """

    def __init__(
        self,
        registry: FaissIndexRegistry,
        embedder: ClipEmbedder,
        db_mng: PostgreManager,
        config: VisualSearchConfig | None = None,
    ) -> None:
        self.registry = registry
        self.embedder = embedder
        self.db_mng = db_mng
        self.config = config or VisualSearchConfig()
        self._subset_ids_cache: dict[
            tuple[str, tuple[str, ...], int | None, int | None], list[int]
        ] = {}
        self._subset_ids_lock = threading.Lock()

    # -- Public API -------------------------------------------------------

    def frame_search(
        self,
        query: str | None = None,
        image_ref: str | None = None,
        top_k: int | None = None,
        event_id: str | None = None,
        tool_call_id: str | None = None,
        video_ids: list[str] | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[SearchHit]:
        """Text -> frame hoặc image -> frame. Case ``image_ref`` chính là
        "image similarity search" (tìm ảnh tương tự 1 ảnh cho trước) —
        ``image_similarity_search()`` bên dưới chỉ là alias gọi lại hàm này,
        không có pipeline resolve riêng."""

        vector = self._encode_frame_query(query, image_ref)
        return self._frame_search_from_vector(
            vector,
            top_k=top_k,
            event_id=event_id,
            tool_call_id=tool_call_id,
            video_ids=video_ids,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def image_similarity_search(
        self,
        image_ref: str,
        top_k: int | None = None,
        event_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> list[SearchHit]:
        """Alias tường minh của ``frame_search(image_ref=...)``, khớp đúng
        tên tool liệt kê trong ``ProposalOnlinePipeline.md`` §7.1.

        Không tự triển khai lại pipeline — chỉ gọi thẳng ``frame_search()``
        để tránh trùng lặp logic encode/resolve (2 tool cùng tìm trên
        ``frame.faiss``, chỉ khác cách sinh vector query: text hay ảnh)."""

        return self.frame_search(
            image_ref=image_ref,
            top_k=top_k,
            event_id=event_id,
            tool_call_id=tool_call_id,
        )

    def clip_search(
        self,
        query: str,
        top_k: int | None = None,
        event_id: str | None = None,
        tool_call_id: str | None = None,
        video_ids: list[str] | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[SearchHit]:
        vector = self.embedder.encode_text(query)
        return self._clip_search_from_vector(
            vector,
            top_k=top_k,
            event_id=event_id,
            tool_call_id=tool_call_id,
            video_ids=video_ids,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def shot_search(
        self,
        query: str,
        top_k: int | None = None,
        event_id: str | None = None,
        tool_call_id: str | None = None,
        video_ids: list[str] | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[SearchHit]:
        """Query thẳng ``shot.faiss`` (đã pooled sẵn ở Offline) — không
        derive runtime từ frame embeddings."""

        vector = self.embedder.encode_text(query)
        return self._shot_search_from_vector(
            vector,
            top_k=top_k,
            event_id=event_id,
            tool_call_id=tool_call_id,
            video_ids=video_ids,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def search_many(
        self,
        requests: list[VisualSearchRequest],
    ) -> list[list[SearchHit]]:
        """Run an ordered mixed frame/clip/shot text batch.

        All request texts are encoded together. Search and canonical resolve
        retain each request's own retriever, limits, scope, and provenance.
        """

        if not requests:
            return []
        vectors = self.embedder.encode_texts([request.query for request in requests])
        if len(vectors) != len(requests):
            raise RuntimeError(
                "encode_texts() returned a different number of vectors than requests"
            )

        results: list[list[SearchHit]] = []
        for request, vector in zip(requests, vectors):
            common = {
                "top_k": request.top_k,
                "event_id": request.event_id,
                "tool_call_id": request.tool_call_id,
                "video_ids": request.video_ids,
                "start_ms": request.start_ms,
                "end_ms": request.end_ms,
            }
            if request.retriever == "frame_search":
                hits = self._frame_search_from_vector(vector, **common)
            elif request.retriever == "clip_search":
                hits = self._clip_search_from_vector(vector, **common)
            elif request.retriever == "shot_search":
                hits = self._shot_search_from_vector(vector, **common)
            else:
                raise ValueError(f"Unsupported visual retriever: {request.retriever!r}")
            results.append(hits)
        return results

    # -- Internal -----------------------------------------------------------

    def _frame_search_from_vector(
        self,
        vector: np.ndarray,
        *,
        top_k: int | None,
        event_id: str | None,
        tool_call_id: str | None,
        video_ids: list[str] | None,
        start_ms: int | None,
        end_ms: int | None,
    ) -> list[SearchHit]:
        if video_ids is None:
            ranked = self._search_and_rank(vector, self.registry.search_frame_index, top_k)
        else:
            subset_ids = self._cached_subset_ids(
                "frame",
                video_ids,
                start_ms,
                end_ms,
                lambda: self.db_mng.get_frame_faiss_ids_for_videos(
                    video_ids,
                    index_version=self.config.index_version,
                    model_name=self.config.model_name,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
            )
            ranked = self._search_subset_and_rank(
                vector, self.registry.search_frame_subset, subset_ids, top_k
            )
        if not ranked:
            return []

        faiss_ids = [faiss_id for faiss_id, _, _ in ranked]
        hits_by_faiss_id = {
            hit.faiss_id: hit
            for hit in self.db_mng.get_frame_hits_by_faiss_ids(
                faiss_ids,
                index_version=self.config.index_version,
                model_name=self.config.model_name,
            )
        }

        results: list[SearchHit] = []
        for faiss_id, raw_score, rank in ranked:
            hit = hits_by_faiss_id.get(faiss_id)
            if hit is None:
                self._log_unresolved("frame", faiss_id)
                continue
            results.append(
                SearchHit(
                    entity_type="frame",
                    entity_id=hit.frame_id,
                    video_id=hit.video_id,
                    shot_id=hit.shot_id,
                    frame_id=hit.frame_id,
                    start_ms=hit.start_ms,
                    end_ms=hit.end_ms,
                    source="frame_embedding",
                    rank=rank,
                    raw_score=raw_score,
                    tool_call_id=tool_call_id,
                    event_id=event_id,
                )
            )
        return results

    def _clip_search_from_vector(
        self,
        vector: np.ndarray,
        *,
        top_k: int | None,
        event_id: str | None,
        tool_call_id: str | None,
        video_ids: list[str] | None,
        start_ms: int | None,
        end_ms: int | None,
    ) -> list[SearchHit]:
        if video_ids is None:
            ranked = self._search_and_rank(vector, self.registry.search_clip_index, top_k)
        else:
            subset_ids = self._cached_subset_ids(
                "clip",
                video_ids,
                start_ms,
                end_ms,
                lambda: self.db_mng.get_clip_faiss_ids_for_videos(
                    video_ids,
                    index_version=self.config.index_version,
                    model_name=self.config.model_name,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
            )
            ranked = self._search_subset_and_rank(
                vector, self.registry.search_clip_subset, subset_ids, top_k
            )
        if not ranked:
            return []

        faiss_ids = [faiss_id for faiss_id, _, _ in ranked]
        hits_by_faiss_id = {
            hit.faiss_id: hit
            for hit in self.db_mng.get_clip_hits_by_faiss_ids(
                faiss_ids,
                index_version=self.config.index_version,
                model_name=self.config.model_name,
            )
        }

        results: list[SearchHit] = []
        for faiss_id, raw_score, rank in ranked:
            hit = hits_by_faiss_id.get(faiss_id)
            if hit is None:
                self._log_unresolved("clip", faiss_id)
                continue
            results.append(
                SearchHit(
                    entity_type="clip",
                    entity_id=hit.clip_id,
                    video_id=hit.video_id,
                    shot_id=hit.shot_id,
                    clip_id=hit.clip_id,
                    start_ms=hit.start_ms,
                    end_ms=hit.end_ms,
                    source="clip_embedding",
                    rank=rank,
                    raw_score=raw_score,
                    tool_call_id=tool_call_id,
                    event_id=event_id,
                )
            )
        return results

    def _shot_search_from_vector(
        self,
        vector: np.ndarray,
        *,
        top_k: int | None,
        event_id: str | None,
        tool_call_id: str | None,
        video_ids: list[str] | None,
        start_ms: int | None,
        end_ms: int | None,
    ) -> list[SearchHit]:
        if video_ids is None:
            ranked = self._search_and_rank(vector, self.registry.search_shot_index, top_k)
        else:
            subset_ids = self._cached_subset_ids(
                "shot",
                video_ids,
                start_ms,
                end_ms,
                lambda: self.db_mng.get_shot_faiss_ids_for_videos(
                    video_ids,
                    index_version=self.config.index_version,
                    model_name=self.config.model_name,
                    model_version=self.config.model_version,
                    pooling_method=self.config.pooling_method,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
            )
            ranked = self._search_subset_and_rank(
                vector, self.registry.search_shot_subset, subset_ids, top_k
            )
        if not ranked:
            return []

        faiss_ids = [faiss_id for faiss_id, _, _ in ranked]
        hits_by_faiss_id = {
            hit.faiss_id: hit
            for hit in self.db_mng.get_shot_hits_by_faiss_ids(
                faiss_ids,
                index_version=self.config.index_version,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                pooling_method=self.config.pooling_method,
            )
        }

        results: list[SearchHit] = []
        for faiss_id, raw_score, rank in ranked:
            hit = hits_by_faiss_id.get(faiss_id)
            if hit is None:
                self._log_unresolved("shot", faiss_id)
                continue
            results.append(
                SearchHit(
                    entity_type="shot",
                    entity_id=hit.shot_id,
                    video_id=hit.video_id,
                    shot_id=hit.shot_id,
                    start_ms=hit.start_ms,
                    end_ms=hit.end_ms,
                    source="shot_embedding",
                    rank=rank,
                    raw_score=raw_score,
                    tool_call_id=tool_call_id,
                    event_id=event_id,
                )
            )
        return results

    def _encode_frame_query(
        self, query: str | None, image_ref: str | None
    ) -> np.ndarray:
        """Bắt buộc đúng 1 trong 2 tham số ``query``/``image_ref`` được set
        (mutually exclusive) — validate trước khi động tới FAISS/DB."""

        if (query is None) == (image_ref is None):
            raise ValueError(
                "frame_search() yêu cầu đúng 1 trong 2 tham số: query hoặc image_ref."
            )
        if query is not None:
            return self.embedder.encode_text(query)

        try:
            frame_record = self.db_mng.get_frame_record_by_frame_id(image_ref)
        except ValueError as exc:
            raise ValueError(f"Không tìm thấy ảnh cho image_ref={image_ref!r}.") from exc
        if frame_record.frame_path is None:
            raise ValueError(
                f"Frame {image_ref!r} tồn tại nhưng không có ảnh lưu trữ (frame_path=None)."
            )
        return self.embedder.encode_image(str(frame_record.frame_path))

    def _cached_subset_ids(
        self,
        entity_type: str,
        video_ids: list[str],
        start_ms: int | None,
        end_ms: int | None,
        loader,
    ) -> list[int]:
        key = (entity_type, tuple(sorted(set(video_ids))), start_ms, end_ms)
        with self._subset_ids_lock:
            cached = self._subset_ids_cache.get(key)
        if cached is not None:
            return cached
        loaded = list(loader())
        with self._subset_ids_lock:
            return self._subset_ids_cache.setdefault(key, loaded)

    def _search_and_rank(
        self,
        vector: np.ndarray,
        search_fn: _SearchIndexFn,
        top_k: int | None,
    ) -> list[tuple[int, float, int]]:
        """Bước dùng chung của cả 3 tool: clamp top_k, gọi đúng hàm
        ``search_*_index()`` tương ứng, gắn rank (1-based, theo đúng thứ tự
        FAISS trả về — không sort lại)."""

        k = self._resolve_top_k(top_k)
        raw_hits = search_fn(vector, k)
        return [
            (faiss_id, raw_score, rank)
            for rank, (faiss_id, raw_score) in enumerate(raw_hits, start=1)
        ]

    def _search_subset_and_rank(
        self,
        vector: np.ndarray,
        search_fn,
        faiss_ids: list[int],
        top_k: int | None,
    ) -> list[tuple[int, float, int]]:
        k = self._resolve_top_k(top_k)
        raw_hits = search_fn(vector, faiss_ids, k)
        return [
            (faiss_id, raw_score, rank)
            for rank, (faiss_id, raw_score) in enumerate(raw_hits, start=1)
        ]

    def _resolve_top_k(self, top_k: int | None) -> int:
        k = top_k if top_k is not None else self.config.default_top_k
        return min(k, self.config.max_top_k, ABSOLUTE_MAX_TOP_K)

    @staticmethod
    def _log_unresolved(entity_type: str, faiss_id: int) -> None:
        """faiss_id không resolve được sang PostgreSQL -> lệch dữ liệu
        DB<->FAISS. Bỏ qua hit này thay vì raise, để không sập cả batch
        search vì 1 bản ghi lệch — nhưng log lại rõ ràng để phát hiện được
        khi giám sát (xem §7 module_visual_retrieval_tools.md: script kiểm
        tra đồng bộ DB<->FAISS phải chạy trước khi dùng module)."""

        logger.warning(
            "Không resolve được faiss_id=%s (entity_type=%s) sang PostgreSQL. "
            "Có thể DB và FAISS index đang lệch dữ liệu — chạy "
            "scripts/check_faiss_db_sync.py để kiểm tra.",
            faiss_id,
            entity_type,
        )


# ---------------------------------------------------------------------------
# 5. Factory — wiring mặc định đọc từ biến môi trường
# ---------------------------------------------------------------------------


def build_default_visual_retrieval_tools(
    *,
    config: VisualSearchConfig | None = None,
    db_mng: PostgreManager | None = None,
) -> VisualRetrievalTools:
    """Dựng 1 ``VisualRetrievalTools`` dùng singleton ``FaissIndexRegistry``
    và cấu hình đọc từ biến môi trường (``FAISS_INDEX_DIR``,
    ``TEXT_EMBEDDING_MODEL_NAME``, ``FAISS_INDEX_MODEL_NAME``,
    ``CLIP_MODEL_DEVICE`` — xem ``.env.example``).

    Gọi 1 lần lúc app khởi động (vd trong FastAPI ``lifespan``), không gọi
    lại trong mỗi request vì sẽ load lại model/index.
    """

    faiss_index_model_name = os.getenv(
        "FAISS_INDEX_MODEL_NAME",
        os.getenv("CLIP_MODEL_NAME", DEFAULT_FAISS_INDEX_MODEL_NAME),
    )
    resolved_config = config or VisualSearchConfig(model_name=faiss_index_model_name)
    registry = FaissIndexRegistry.get_instance()
    embedder = ClipEmbedder(
        model_name=os.getenv(
            "TEXT_EMBEDDING_MODEL_NAME", DEFAULT_TEXT_EMBEDDING_MODEL_NAME
        ),
        device=os.getenv("CLIP_MODEL_DEVICE", "cpu"),
        image_model_name=faiss_index_model_name,
    )
    return VisualRetrievalTools(
        registry=registry,
        embedder=embedder,
        db_mng=db_mng or PostgreManager(),
        config=resolved_config,
    )
