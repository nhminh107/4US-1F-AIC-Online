"""End-to-End AIC Competition CLI Runner (Pipeline V2).

Interactive terminal runner that:
1. Auto-starts & healthchecks Docker runtime.
2. Provides an interactive menu to test KIS, VQA, TRAKE tasks.
3. Executes queries via Pipeline V2.
4. Exports 2 files:
   - BTC-compliant unzipped CSV file (<query_id>.csv)
   - Interactive HTML Visual Inspector (<query_id>_preview.html) & image links file.
5. Summarizes top results in terminal with auto-browser preview option.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows UTF-8 stdout safeguard
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' package is required. Run: pip install requests")
    sys.exit(1)

from scripts.competition.docker_guard import ensure_docker_runtime, check_api_health
from scripts.competition.exporter import ResultExporter, format_timestamp

API_QUERY_URL = "http://localhost:8000/api/v1/query"
QUERIES_DIR = PROJECT_ROOT / "queries"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def print_banner() -> None:
    print("\n" + "=" * 68)
    print("       AIC ONLINE COMPETITION INTERACTIVE RUNNER (V2)")
    print("=" * 68)


def list_query_files() -> list[Path]:
    """List available sample query files in queries/ directory."""
    if not QUERIES_DIR.exists():
        return []
    return sorted(list(QUERIES_DIR.glob("*.txt")))


def select_query_from_file() -> tuple[str, str, str] | None:
    """Prompt user to select a query file from queries/ directory."""
    files = list_query_files()
    if not files:
        print("[!] No query files found in queries/ directory.")
        return None

    print("\nAvailable query files:")
    for idx, f in enumerate(files, start=1):
        print(f"  [{idx:2d}] {f.name}")
    print("  [ 0] Back to main menu")

    choice = input("\nSelect file number: ").strip()
    if not choice.isdigit() or int(choice) <= 0 or int(choice) > len(files):
        return None

    selected_file = files[int(choice) - 1]
    content = selected_file.read_text(encoding="utf-8").strip()

    # Determine task from filename
    fname = selected_file.stem.lower()
    if "kis" in fname:
        task = "KIS"
    elif "qa" in fname or "vqa" in fname:
        task = "VQA"
    elif "trake" in fname:
        task = "TRAKE"
    else:
        task = "KIS"

    query_id = selected_file.stem
    return task, query_id, content


def run_query(
    *,
    task: str,
    query_id: str,
    prompt: str,
    ask_open_browser: bool = True,
) -> dict[str, Any] | None:
    """Send query to API V2, export submission & HTML preview, and show summary."""
    print(f"\n[*] Sending {task} query to Pipeline V2 API...")
    print(f"    Query ID: {query_id}")
    print(f"    Prompt  : {prompt[:90]}{'...' if len(prompt) > 90 else ''}\n")

    payload = {
        "prompt": prompt,
        "session_id": query_id,
        "task_hint": task,
        "top_k": {
            "result_top_k": 100,
        },
    }

    start_time = time.perf_counter()
    try:
        resp = requests.post(API_QUERY_URL, json=payload, timeout=120)
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return None

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if resp.status_code != 200:
        print(f"[ERROR] API returned error HTTP {resp.status_code}: {resp.text}")
        return None

    data = resp.json()
    response_task = data.get("task")
    if response_task != task:
        print(
            f"[ERROR] API task mismatch: requested {task}, returned {response_task}. "
            "Refusing to export with the wrong BTC format."
        )
        return None

    # Export outputs (CSV for BTC + HTML for visual review + TXT links)
    exporter = ResultExporter(output_base_dir=OUTPUTS_DIR)
    exported = exporter.export(
        query_id=query_id,
        task=task,
        prompt=prompt,
        api_response=data,
        latency_ms=latency_ms,
    )

    # Print summary
    verif = data.get("verification", {})
    verif_status = verif.get("status", "unknown")
    verif_conf = verif.get("confidence", 0.0)

    print("=" * 68)
    print(f" [V2 EXECUTION SUCCESS]  Latency: {latency_ms:.1f}ms | Verifier: {verif_status.upper()} ({verif_conf:.2f})")
    print("=" * 68)

    if task == "VQA" and data.get("answer"):
        print(f"[*] ANSWER: \033[92m{data.get('answer')}\033[0m")
        print("-" * 68)

    # Show top-5 results in terminal
    if task == "TRAKE":
        sequences = data.get("results", data.get("sequences", []))
        print(f"Top Matched Video Sequences (Found {len(sequences)}):")
        for i, seq in enumerate(sequences[:3], start=1):
            vid = seq.get("video_id")
            score = seq.get("sequence_score", 0.0)
            print(f"  Seq #{i} | Video: {vid} | Score: {score:.4f}")
            for evt in seq.get("events", []):
                eid = evt.get("event_id")
                fidx = evt.get("frame_idx")
                t_str = format_timestamp(evt.get("start_ms"))
                print(f"     -> {eid}: Frame {fidx:<6} ({t_str})")
    else:
        results = data.get("results", [])
        print(f"Top 5 Ranked Frames (Total {len(results)} items):")
        for i, item in enumerate(results[:5], start=1):
            vid = item.get("video_id")
            fidx = item.get("frame_idx")
            score = item.get("score", 0.0)
            t_str = format_timestamp(item.get("start_ms"))
            ans_str = f" | Ans: {item.get('answer')}" if item.get("answer") else ""
            print(f"  #{i:02d} | Video: {vid:<10} | Frame: {fidx:<6} | Score: {score:.4f} | Time: {t_str}{ans_str}")

    print("\n" + "=" * 68)
    print(" [OUTPUT ARTIFACTS GENERATED]")
    print(f"  1. File nộp BTC (CSV không zip): {exported['csv_file']}")
    print(f"  2. File kiểm tra ảnh (HTML):    {exported['html_file']}")
    print(f"  3. File danh sách link ảnh:     {exported['txt_file']}")
    print(f"  4. Raw API response (JSON):     {exported['response_file']}")
    print(f"  5. Retrieval audit (JSON):      {exported['audit_file']}")
    print("=" * 68)

    # Ask to open in browser if interactive
    if ask_open_browser:
        try:
            open_now = input("\n[?] Bạn có muốn mở ngay trang HTML kiểm tra ảnh trên trình duyệt? (Y/n): ").strip().lower()
            if open_now in ("", "y", "yes"):
                webbrowser.open(exported["html_file"].resolve().as_uri())
                print(f"[*] Đã mở {exported['html_file'].name} trên trình duyệt!")
        except Exception as e:
            print(f"[!] Không thể mở trình duyệt: {e}")

    return data


def main_interactive_loop() -> None:
    """Main interactive loop for the competition runner."""
    print_banner()

    # Step 1: Ensure Docker runtime is up & healthy
    print("[*] Kiểm tra hệ thống Docker runtime...")
    if not ensure_docker_runtime():
        print("\n[FATAL] Không thể kết nối hoặc khởi động Docker runtime. Vui lòng kiểm tra Docker Desktop.")
        sys.exit(1)

    while True:
        print_banner()
        print(" CHỌN BÀI THI HOẶC CHỨC NĂNG:")
        print("  [1] Bài 1: KIS   - Known Item Search (Tìm khoảnh khắc / cảnh video)")
        print("  [2] Bài 2: VQA   - Visual Question Answering (Hỏi đáp nội dung video)")
        print("  [3] Bài 3: TRAKE - Temporal Action Tracking (Tìm chuỗi hành động E1, E2, ...)")
        print("  [4] Nạp đề từ file có sẵn (Thư mục queries/)")
        print("  [5] Kiểm tra lại sức khỏe hệ thống (Health check / Docker)")
        print("  [0] Thoát")
        print("-" * 68)

        choice = input("Nhập lựa chọn của bạn (0-5): ").strip()

        if choice == "0":
            print("\n[!] Tạm biệt! Chúc bạn thi đấu đạt kết quả tốt nhất.")
            break

        elif choice in ("1", "2", "3"):
            task_map = {"1": "KIS", "2": "VQA", "3": "TRAKE"}
            task = task_map[choice]
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_query_id = f"query_{task.lower()}_{timestamp_str}"

            print(f"\n--- [BÀI THI: {task}] ---")
            query_id_input = input(f"Nhập mã đề Query ID (mặc định: '{default_query_id}'): ").strip()
            query_id = query_id_input if query_id_input else default_query_id

            print("\nNhập/dán nội dung prompt câu hỏi (nhấn Enter để xác nhận):")
            prompt = input("> ").strip()
            if not prompt:
                print("[!] Prompt không được để trống!")
                continue

            run_query(task=task, query_id=query_id, prompt=prompt)

        elif choice == "4":
            selected = select_query_from_file()
            if selected:
                task, query_id, content = selected
                print(f"\n[*] Đã nạp từ file: {query_id} (Dạng bài: {task})")
                print(f"[*] Prompt: {content}\n")
                confirm = input("Chạy truy vấn này? (Y/n): ").strip().lower()
                if confirm in ("", "y", "yes"):
                    run_query(task=task, query_id=query_id, prompt=content)

        elif choice == "5":
            print("\n[*] Đang kiểm tra sức khỏe hệ thống...")
            ensure_docker_runtime(force_restart=False)

        else:
            print("[!] Lựa chọn không hợp lệ, vui lòng chọn từ 0 đến 5.")

        input("\nNhấn Enter để tiếp tục...")


if __name__ == "__main__":
    try:
        main_interactive_loop()
    except KeyboardInterrupt:
        print("\n\n[!] Đã dừng chương trình.")
        sys.exit(0)
