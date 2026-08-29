"""Result export utilities for AIC competition: BTC CSV submission and Visual HTML/Text inspector."""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any

# CDN prefix for frame images (matches agent_tools/write_outputs.py)
R2_PREFIX = "https://pub-c8f3587e831a418ebf0d427203860188.r2.dev/"


def full_url(raw: str) -> str:
    """Convert a DB frame_path to a full R2 CDN URL."""
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    return R2_PREFIX + raw.lstrip("/").replace("\\", "/")


def format_timestamp(ms: int | None) -> str:
    """Format milliseconds into MM:SS.mmm format."""
    if ms is None or ms < 0:
        return "N/A"
    total_seconds = ms / 1000.0
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"


class ResultExporter:
    """Export query results to BTC-compliant CSV and Visual HTML preview."""

    def __init__(self, output_base_dir: str | Path = "outputs") -> None:
        self.output_base_dir = Path(output_base_dir)

    def export(
        self,
        *,
        query_id: str,
        task: str,
        prompt: str,
        api_response: dict[str, Any],
        latency_ms: float = 0.0,
    ) -> dict[str, Path]:
        """Export all output artifacts for a query run.

        Returns:
            dict with keys 'csv_file', 'html_file', 'txt_file' pointing to the created paths.
        """
        task_upper = task.upper()
        query_dir = self.output_base_dir / query_id
        query_dir.mkdir(parents=True, exist_ok=True)

        csv_path = query_dir / f"{query_id}.csv"
        html_path = query_dir / f"{query_id}_preview.html"
        txt_path = query_dir / f"{query_id}_image_links.txt"
        response_path = query_dir / f"{query_id}_response.json"
        audit_path = query_dir / f"{query_id}_audit.json"

        response_path.write_text(
            json.dumps(api_response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        audit_path.write_text(
            json.dumps(
                self._build_audit(query_id, task_upper, api_response, latency_ms),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # 1. Export BTC Submission CSV
        self._export_csv(api_response, task_upper, csv_path)

        # 2. Export Image Links TXT
        self._export_links_txt(api_response, task_upper, txt_path)

        # 3. Export Visual HTML Preview
        self._export_html(
            query_id=query_id,
            task=task_upper,
            prompt=prompt,
            api_response=api_response,
            latency_ms=latency_ms,
            output_path=html_path,
        )

        return {
            "csv_file": csv_path,
            "html_file": html_path,
            "txt_file": txt_path,
            "response_file": response_path,
            "audit_file": audit_path,
        }

    @staticmethod
    def _build_audit(
        query_id: str,
        requested_task: str,
        response: dict[str, Any],
        latency_ms: float,
    ) -> dict[str, Any]:
        results = response.get("results", [])
        return {
            "query_id": query_id,
            "requested_task": requested_task,
            "task": response.get("task"),
            "task_matches_request": response.get("task") == requested_task,
            "execution_path": response.get("execution_path"),
            "latency_ms": round(latency_ms, 3),
            "search_hit_count": response.get("search_hit_count", 0),
            "candidate_count": response.get("candidate_count", 0),
            "result_count": len(results),
            "structured_query": response.get("structured_query"),
            "retrieval_v2_plan": response.get("retrieval_v2_plan"),
            "tool_calls": response.get("tool_calls", []),
            "retrieval_v2_session": response.get("retrieval_v2_session"),
            "trake_status": response.get("trake_status"),
            "replan_required": response.get("replan_required", False),
            "missing_event_ids": response.get("missing_event_ids", []),
            "warnings": response.get("warnings", []),
        }

    def _export_csv(
        self,
        response: dict[str, Any],
        task: str,
        output_path: Path,
    ) -> None:
        """Write unheaded UTF-8 CSV compliant with BTC rules."""
        rows: list[list[Any]] = []

        if task == "TRAKE":
            sequences = response.get("results", response.get("sequences", []))
            for seq in sequences:
                events = seq.get("events", [])
                for evt in events:
                    vid = evt.get("video_id") or seq.get("video_id", "")
                    fidx = evt.get("frame_idx", 0)
                    rows.append([vid, fidx])
        else:
            results = response.get("results", [])
            global_answer = response.get("answer") or ""
            for item in results:
                vid = item.get("video_id", "")
                fidx = item.get("frame_idx", 0)
                if task == "VQA":
                    ans = item.get("answer") or global_answer
                    rows.append([vid, fidx, ans])
                else:
                    rows.append([vid, fidx])

        # Limit to top 100 rows
        rows = rows[:100]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)

    def _export_links_txt(
        self,
        response: dict[str, Any],
        task: str,
        output_path: Path,
    ) -> None:
        """Write plain text file with full R2 CDN image URLs and frame info."""
        lines: list[str] = [
            f"# Image Links and Verification Details (Task: {task})",
            f"# Generated at: {output_path.name}",
            "# " + "=" * 70,
            "",
        ]

        if task == "TRAKE":
            sequences = response.get("results", response.get("sequences", []))
            for s_idx, seq in enumerate(sequences, start=1):
                vid = seq.get("video_id", "")
                score = seq.get("sequence_score", 0.0)
                lines.append(f"--- Sequence #{s_idx} | Video: {vid} | Score: {score:.4f} ---")
                for evt in seq.get("events", []):
                    eid = evt.get("event_id", "")
                    fidx = evt.get("frame_idx", 0)
                    start_ms = evt.get("start_ms", 0)
                    img_url = full_url(evt.get("img_url", ""))
                    lines.append(
                        f"  [{eid}] Frame: {fidx} | Time: {format_timestamp(start_ms)} | "
                        f"Image: {img_url}"
                    )
                lines.append("")
        else:
            results = response.get("results", [])
            for r_idx, item in enumerate(results, start=1):
                vid = item.get("video_id", "")
                fidx = item.get("frame_idx", 0)
                score = item.get("score", 0.0)
                start_ms = item.get("start_ms", 0)
                img_url = full_url(item.get("img_url", ""))
                ans_str = f" | Answer: {item.get('answer')}" if task == "VQA" and item.get("answer") else ""
                lines.append(
                    f"#{r_idx:03d} | Video: {vid} | Frame: {fidx:<6} | Score: {score:.4f} | "
                    f"Time: {format_timestamp(start_ms)}{ans_str}\n"
                    f"      Image: {img_url}"
                )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _export_html(
        self,
        *,
        query_id: str,
        task: str,
        prompt: str,
        api_response: dict[str, Any],
        latency_ms: float,
        output_path: Path,
    ) -> None:
        """Render modern, responsive dark-mode HTML visual gallery with R2 CDN images."""
        verif = api_response.get("verification", {})
        verif_status = verif.get("status", "unknown")
        verif_conf = verif.get("confidence", 0.0)
        global_ans = api_response.get("answer", "")

        status_color = "#10b981" if verif_status == "accepted" else "#ef4444" if verif_status == "rejected" else "#f59e0b"

        # Build cards
        cards_html = []
        if task == "TRAKE":
            sequences = api_response.get("results", api_response.get("sequences", []))
            for s_idx, seq in enumerate(sequences, start=1):
                vid = seq.get("video_id", "")
                score = seq.get("sequence_score", 0.0)
                events = seq.get("events", [])
                events_html = []
                for evt in events:
                    eid = evt.get("event_id", "")
                    fidx = evt.get("frame_idx", 0)
                    start_ms = evt.get("start_ms", 0)
                    evt_score = evt.get("score", 0.0)
                    cdn_img = full_url(evt.get("img_url", ""))

                    events_html.append(f"""
                    <div class="event-item">
                        <div class="event-tag">{html.escape(str(eid))}</div>
                        <div class="img-wrap">
                            <a href="{cdn_img}" target="_blank" title="Open full image">
                                <img src="{cdn_img}" alt="Frame {fidx}" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'200\\' height=\\'120\\' viewBox=\\'0 0 200 120\\'><rect fill=\\'%231e293b\\' width=\\'200\\' height=\\'120\\'/><text fill=\\'%2394a3b8\\' x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\'>Frame {fidx}</text></svg>'">
                            </a>
                        </div>
                        <div class="meta-row">
                            <span class="fidx">Frame: <b>{fidx}</b></span>
                            <span class="time">{format_timestamp(start_ms)}</span>
                        </div>
                        <button class="copy-btn" onclick="navigator.clipboard.writeText('{cdn_img}')" title="Copy image URL">📋</button>
                    </div>
                    """)

                cards_html.append(f"""
                <div class="seq-card">
                    <div class="seq-header">
                        <span class="seq-rank">#{s_idx}</span>
                        <span class="seq-vid">Video: <b>{html.escape(vid)}</b></span>
                        <span class="seq-score">Match Score: <b>{score:.4f}</b></span>
                    </div>
                    <div class="events-grid">
                        {''.join(events_html)}
                    </div>
                </div>
                """)
        else:
            results = api_response.get("results", [])
            for r_idx, item in enumerate(results, start=1):
                vid = item.get("video_id", "")
                fidx = item.get("frame_idx", 0)
                score = item.get("score", 0.0)
                start_ms = item.get("start_ms", 0)
                cdn_img = full_url(item.get("img_url", ""))
                item_ans = item.get("answer") or ""

                ans_badge = f'<div class="ans-badge">💬 {html.escape(item_ans)}</div>' if item_ans else ""

                cards_html.append(f"""
                <div class="card">
                    <div class="card-header">
                        <span class="rank">#{r_idx}</span>
                        <span class="vid">{html.escape(vid)}</span>
                        <span class="score">{score:.4f}</span>
                    </div>
                    <div class="img-wrap">
                        <a href="{cdn_img}" target="_blank" title="Click to view full image">
                            <img src="{cdn_img}" alt="{html.escape(vid)} frame {fidx}" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'240\\' height=\\'135\\' viewBox=\\'0 0 240 135\\'><rect fill=\\'%231e293b\\' width=\\'240\\' height=\\'135\\'/><text fill=\\'%2394a3b8\\' x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\'>Frame {fidx}</text></svg>'">
                        </a>
                    </div>
                    <div class="card-footer">
                        <div class="meta-row">
                            <span class="fidx">Frame: <b>{fidx}</b></span>
                            <span class="time">{format_timestamp(start_ms)}</span>
                        </div>
                        {ans_badge}
                        <button class="copy-btn" onclick="navigator.clipboard.writeText('{cdn_img}')" title="Copy image URL">📋 Copy URL</button>
                    </div>
                </div>
                """)

        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIC Visual Inspector - {html.escape(query_id)}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-hover: #1f2937;
            --border: #1e293b;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --primary: #38bdf8;
            --accent: #818cf8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }}
        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 16px;
        }}
        .title-area {{ display: flex; align-items: center; gap: 12px; }}
        .badge-task {{
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            color: white;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }}
        h1 {{ font-size: 1.5rem; font-weight: 800; color: #fff; }}
        .metrics-bar {{
            display: flex;
            gap: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}
        .metric-item {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 8px;
        }}
        .metric-item b {{ color: var(--primary); }}
        .prompt-box {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 14px 18px;
            font-size: 0.95rem;
            color: #cbd5e1;
        }}
        .prompt-box b {{ color: var(--primary); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
        }}
        .card:hover {{
            transform: translateY(-4px);
            border-color: var(--primary);
            box-shadow: 0 12px 24px rgba(56, 189, 248, 0.15);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: rgba(0,0,0,0.2);
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .rank {{
            background: #1e293b;
            color: var(--primary);
            padding: 2px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .vid {{ font-family: 'JetBrains Mono', monospace; color: #cbd5e1; }}
        .score {{ color: var(--accent); font-family: 'JetBrains Mono', monospace; font-weight: 700; }}
        .img-wrap {{
            width: 100%;
            aspect-ratio: 16/9;
            background: #0b0f17;
            position: relative;
            overflow: hidden;
        }}
        .img-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }}
        .img-wrap:hover img {{ transform: scale(1.05); }}
        .card-footer {{
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 0.8rem;
        }}
        .meta-row {{
            display: flex;
            justify-content: space-between;
            color: var(--text-dim);
            font-family: 'JetBrains Mono', monospace;
        }}
        .meta-row b {{ color: #e2e8f0; }}
        .ans-badge {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            margin-top: 4px;
            word-break: break-word;
        }}
        .copy-btn {{
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: var(--primary);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-top: 4px;
        }}
        .copy-btn:hover {{
            background: rgba(56, 189, 248, 0.25);
            border-color: var(--primary);
        }}
        .copy-btn:active {{
            transform: scale(0.95);
        }}
        /* TRAKE specific styling */
        .seq-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        .seq-header {{
            display: flex;
            gap: 16px;
            align-items: center;
            margin-bottom: 14px;
            font-size: 0.95rem;
        }}
        .seq-rank {{
            background: var(--accent);
            color: white;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .events-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }}
        .event-item {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
            position: relative;
        }}
        .event-tag {{
            position: absolute;
            top: 14px;
            left: 14px;
            z-index: 10;
            background: rgba(0, 0, 0, 0.75);
            color: #38bdf8;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 2px 8px;
            border-radius: 6px;
            border: 1px solid rgba(56, 189, 248, 0.4);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-top">
                <div class="title-area">
                    <span class="badge-task">{html.escape(task)}</span>
                    <h1>{html.escape(query_id)}</h1>
                </div>
                <div class="metrics-bar">
                    <div class="metric-item">Time: <b>{latency_ms:.1f}ms</b></div>
                    <div class="metric-item">Verifier: <b style="color: {status_color}; text-transform: uppercase;">{verif_status}</b> ({verif_conf:.2f})</div>
                    <div class="metric-item">Total Results: <b>{len(api_response.get('results', api_response.get('sequences', [])))}</b></div>
                </div>
            </div>
            <div class="prompt-box">
                <b>Prompt:</b> {html.escape(prompt)}
            </div>
            {f'<div class="prompt-box" style="margin-top: 10px; border-color: rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.08);"><b>Overall Answer:</b> <span style="color: #34d399; font-weight: 600;">{html.escape(global_ans)}</span></div>' if global_ans else ''}
        </header>

        <div class="{ 'events-container' if task == 'TRAKE' else 'grid' }">
            {''.join(cards_html)}
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)


__all__ = ["ResultExporter", "full_url", "R2_PREFIX"]
