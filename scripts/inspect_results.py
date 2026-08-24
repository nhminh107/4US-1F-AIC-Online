import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("scripts/results_investigation.json", encoding="utf-8") as f:
    data = json.load(f)

for q_id, terms in data.items():
    print(f"\n==================== {q_id} ====================")
    for term, hits in terms.items():
        if hits:
            print(f"--- '{term}' ---")
            for h in hits[:2]:
                print(f"  [{h['source']}] video: {h['video_id']}, frame_idx: {h['frame_idx']}, time_ms: {h['timestamp_ms']}, score: {h['score']:.2f}")
                print(f"    text: {h['content'][:150]}")
