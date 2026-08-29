import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('scripts/results_investigation.json', encoding='utf-8') as f:
    data = json.load(f)

for q_id, q_data in data.items():
    print(f"\n[{q_id}]")
    for keyword, hits in q_data.items():
        if not hits: continue
        print(f"  Keyword: {keyword}")
        for i, hit in enumerate(hits[:2]):
            content = hit['content'].replace('\n', ' ')
            print(f"    - {hit['video_id']} @ {hit['timestamp_ms']}ms (score: {hit['score']}): {content[:100]}...")
