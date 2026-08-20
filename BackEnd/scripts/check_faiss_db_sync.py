"""Script kiểm tra đồng bộ dữ liệu giữa PostgreSQL và FAISS index.

Bắt buộc chạy 1 lần trước khi Visual Retrieval Tools (Module 3.1) đi vào
dùng thật, và mỗi khi nhận file ``.faiss`` mới (xem §7
``Markdown_Doc/module_visual_retrieval_tools.md``).

Cách chạy (từ thư mục gốc repo):

    python -m BackEnd.scripts.check_faiss_db_sync

Nếu có dòng MISMATCH: KHÔNG đưa module vào dùng, báo lại nhóm để xác minh
nguồn gốc file ``.faiss`` (build từ batch nào, đã đồng bộ ghi DB chưa)
trước khi tiếp tục.
"""

from __future__ import annotations

import os
import sys

import faiss

from BackEnd.app.Database.postgre_manager import PostgreManager

# (tên file .faiss, entity_type tương ứng, index_version cần kiểm tra)
# index_version=0 khớp dữ liệu thật hiện có (Markdown_Doc/aic_hcmc_full.sql):
# cả 3 bảng *EmbeddingRecord đều dùng index_version=0, không phải 1.
_CHECKS: list[tuple[str, str, int]] = [
    ("frame.faiss", "frame", 0),
    ("clip.faiss", "clip", 0),
    ("shot.faiss", "shot", 0),
]


def check(index_dir: str, db_mng: PostgreManager) -> bool:
    """In kết quả so sánh ``ntotal`` của từng file .faiss với số dòng
    embedding record tương ứng trong PostgreSQL.

    Trả về True nếu tất cả khớp, False nếu có ít nhất 1 MISMATCH.
    """

    all_ok = True
    for filename, entity_type, index_version in _CHECKS:
        path = os.path.join(index_dir, filename)
        index = faiss.read_index(path)
        faiss_count = index.ntotal
        db_count = db_mng.count_embedding_records(entity_type, index_version)

        is_ok = faiss_count == db_count
        all_ok = all_ok and is_ok
        status = "OK" if is_ok else "MISMATCH"
        print(
            f"[{status}] {entity_type}: FAISS ntotal={faiss_count} | "
            f"DB count={db_count} (index_version={index_version})"
        )
    return all_ok


if __name__ == "__main__":
    faiss_index_dir = os.getenv("FAISS_INDEX_DIR", "BackEnd/data/faiss")
    is_synced = check(faiss_index_dir, PostgreManager())
    sys.exit(0 if is_synced else 1)
