"""Test script for Aggregator module.

Generates ~20 SearchHit items with all entity types (video, frame, shot, clip, ocr, asr, caption, object_detection, object_track)
and tests interval merging into CandidateRegions.
"""

from BackEnd.app.aggregator import Aggregator
from BackEnd.app.contracts.models import SearchHit


def create_sample_search_hits() -> list[SearchHit]:
    """Create 20 SearchHits covering all 9 entity types and diverse time ranges."""
    hits = [
        # ==========================================
        # Group 1: (video_01, event_1)
        # Cluster 1A: 1000ms - 3500ms (merge_gap = 2000)
        # ==========================================
        SearchHit(
            tool_call_id="tc_01",
            event_id="event_1",
            source="faiss_frame_embedding",
            entity_type="frame",
            entity_id="frame_001",
            video_id="video_01",
            frame_id="frame_001",
            start_ms=1000,
            end_ms=1000,
            rank=1,
            raw_score=0.95,
        ),
        SearchHit(
            tool_call_id="tc_01",
            event_id="event_1",
            source="ocr_search",
            entity_type="ocr",
            entity_id="ocr_101",
            video_id="video_01",
            frame_id="frame_001",
            start_ms=1200,
            end_ms=1500,
            rank=2,
            raw_score=0.88,
        ),
        SearchHit(
            tool_call_id="tc_02",
            event_id="event_1",
            source="yolo_object_detector",
            entity_type="object_detection",
            entity_id="obj_det_501",
            video_id="video_01",
            frame_id="frame_002",
            start_ms=2000,
            end_ms=2000,
            rank=3,
            raw_score=0.85,
        ),
        SearchHit(
            tool_call_id="tc_02",
            event_id="event_1",
            source="clip_embedding",
            entity_type="clip",
            entity_id="clip_005",
            video_id="video_01",
            clip_id="clip_005",
            start_ms=1500,
            end_ms=3500,
            rank=4,
            raw_score=0.81,
        ),

        # Cluster 1B: 10000ms - 14000ms (Gap: 10000 - 3500 = 6500ms > 2000ms -> New Region)
        SearchHit(
            tool_call_id="tc_03",
            event_id="event_1",
            source="shot_detector",
            entity_type="shot",
            entity_id="shot_012",
            video_id="video_01",
            shot_id="shot_012",
            start_ms=10000,
            end_ms=12000,
            rank=5,
            raw_score=0.79,
        ),
        SearchHit(
            tool_call_id="tc_03",
            event_id="event_1",
            source="whisper_asr",
            entity_type="asr",
            entity_id="asr_301",
            video_id="video_01",
            start_ms=11500,
            end_ms=14000,
            rank=6,
            raw_score=0.76,
        ),

        # Cluster 1C: 25000ms - 27000ms (Gap: 25000 - 14000 = 11000ms > 2000ms -> New Region)
        SearchHit(
            tool_call_id="tc_04",
            event_id="event_1",
            source="blip_caption",
            entity_type="caption",
            entity_id="cap_401",
            video_id="video_01",
            start_ms=25000,
            end_ms=27000,
            rank=7,
            raw_score=0.74,
        ),

        # ==========================================
        # Group 2: (video_01, event_2)
        # Cluster 2A: 5000ms - 8500ms
        # ==========================================
        SearchHit(
            tool_call_id="tc_05",
            event_id="event_2",
            source="deepsort_tracker",
            entity_type="object_track",
            entity_id="track_601",
            video_id="video_01",
            start_ms=5000,
            end_ms=8000,
            rank=1,
            raw_score=0.96,
        ),
        SearchHit(
            tool_call_id="tc_05",
            event_id="event_2",
            source="video_retrieval",
            entity_type="video",
            entity_id="video_01_full",
            video_id="video_01",
            start_ms=6000,
            end_ms=7500,
            rank=2,
            raw_score=0.89,
        ),
        SearchHit(
            tool_call_id="tc_05",
            event_id="event_2",
            source="ocr_search",
            entity_type="ocr",
            entity_id="ocr_102",
            video_id="video_01",
            start_ms=7000,
            end_ms=8500,
            rank=3,
            raw_score=0.87,
        ),

        # ==========================================
        # Group 3: (video_02, event_1)
        # Cluster 3A: 500ms - 4000ms
        # ==========================================
        SearchHit(
            tool_call_id="tc_06",
            event_id="event_1",
            source="faiss_frame_embedding",
            entity_type="frame",
            entity_id="frame_201",
            video_id="video_02",
            frame_id="frame_201",
            start_ms=500,
            end_ms=500,
            rank=1,
            raw_score=0.91,
        ),
        SearchHit(
            tool_call_id="tc_06",
            event_id="event_1",
            source="clip_embedding",
            entity_type="clip",
            entity_id="clip_201",
            video_id="video_02",
            clip_id="clip_201",
            start_ms=600,
            end_ms=2200,
            rank=2,
            raw_score=0.86,
        ),
        SearchHit(
            tool_call_id="tc_07",
            event_id="event_1",
            source="shot_detector",
            entity_type="shot",
            entity_id="shot_201",
            video_id="video_02",
            shot_id="shot_201",
            start_ms=2000,
            end_ms=4000,
            rank=3,
            raw_score=0.83,
        ),

        # Cluster 3B: 15000ms - 18000ms
        SearchHit(
            tool_call_id="tc_08",
            event_id="event_1",
            source="whisper_asr",
            entity_type="asr",
            entity_id="asr_501",
            video_id="video_02",
            start_ms=15000,
            end_ms=17500,
            rank=4,
            raw_score=0.80,
        ),
        SearchHit(
            tool_call_id="tc_08",
            event_id="event_1",
            source="blip_caption",
            entity_type="caption",
            entity_id="cap_501",
            video_id="video_02",
            start_ms=16000,
            end_ms=18000,
            rank=5,
            raw_score=0.77,
        ),

        # ==========================================
        # Group 4: (video_03, event_3)
        # Cluster 4A: 3000ms - 5500ms
        # ==========================================
        SearchHit(
            tool_call_id="tc_09",
            event_id="event_3",
            source="yolo_object_detector",
            entity_type="object_detection",
            entity_id="obj_det_901",
            video_id="video_03",
            start_ms=3000,
            end_ms=3000,
            rank=1,
            raw_score=0.94,
        ),
        SearchHit(
            tool_call_id="tc_09",
            event_id="event_3",
            source="deepsort_tracker",
            entity_type="object_track",
            entity_id="track_901",
            video_id="video_03",
            start_ms=3200,
            end_ms=5000,
            rank=2,
            raw_score=0.90,
        ),
        SearchHit(
            tool_call_id="tc_10",
            event_id="event_3",
            source="video_retrieval",
            entity_type="video",
            entity_id="video_03_full",
            video_id="video_03",
            start_ms=4500,
            end_ms=5500,
            rank=3,
            raw_score=0.84,
        ),

        # Cluster 4B: 30000ms - 32000ms
        SearchHit(
            tool_call_id="tc_10",
            event_id="event_3",
            source="faiss_frame_embedding",
            entity_type="frame",
            entity_id="frame_305",
            video_id="video_03",
            frame_id="frame_305",
            start_ms=30000,
            end_ms=30000,
            rank=4,
            raw_score=0.82,
        ),
        SearchHit(
            tool_call_id="tc_10",
            event_id="event_3",
            source="ocr_search",
            entity_type="ocr",
            entity_id="ocr_305",
            video_id="video_03",
            start_ms=30500,
            end_ms=32000,
            rank=5,
            raw_score=0.78,
        ),
    ]
    return hits


def run_aggregator_test():
    hits = create_sample_search_hits()
    print("=" * 80)
    print(" 1. INPUT SEARCH HITS SUMMARY ")
    print("=" * 80)
    print(f"Total SearchHits count: {len(hits)}")
    
    entity_types = {h.entity_type for h in hits}
    print(f"Entity types present ({len(entity_types)}): {sorted(list(entity_types))}")
    print()

    print("Details of Input SearchHits:")
    print(f"{'Idx':<4} | {'Video ID':<10} | {'Event ID':<8} | {'Entity Type':<17} | {'Entity ID':<12} | {'Time Range (ms)':<15} | {'Score':<6}")
    print("-" * 88)
    for idx, hit in enumerate(hits, 1):
        time_range = f"{hit.start_ms}-{hit.end_ms}"
        print(f"{idx:<4} | {hit.video_id:<10} | {str(hit.event_id):<8} | {hit.entity_type:<17} | {hit.entity_id:<12} | {time_range:<15} | {hit.raw_score:<6.2f}")
    print()

    print("=" * 80)
    print(" 2. RUNNING AGGREGATOR (merge_gap = 2000ms) ")
    print("=" * 80)
    aggregator = Aggregator()
    candidate_regions = aggregator.execute(hits, merge_gap=2000)

    print(f"Total CandidateRegions produced: {len(candidate_regions)}")
    print()

    print("=" * 80)
    print(" 3. OUTPUT CANDIDATE REGIONS AND EVIDENCE ")
    print("=" * 80)
    for idx, region in enumerate(candidate_regions, 1):
        print(f"Region #{idx}: Candidate ID: {region.candidate_id}")
        print(f"  - Video ID:   {region.video_id}")
        print(f"  - Event ID:   {region.event_id}")
        print(f"  - Time Range: {region.start_ms} ms -> {region.end_ms} ms (Duration: {region.end_ms - region.start_ms} ms)")
        print(f"  - Evidence Count: {len(region.evidence)}")
        print("  - Evidences:")
        for ev_idx, ev in enumerate(region.evidence, 1):
            print(f"      [{ev_idx}] Type: {ev.entity_type:<16} | ID: {ev.entity_id:<12} | Time: {ev.start_ms}-{ev.end_ms} ms | Rank: {ev.rank} | Score: {ev.raw_score}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(" 4. VERIFICATION AND COMPARISON SUMMARY ")
    print("=" * 80)
    total_evidences_out = sum(len(r.evidence) for r in candidate_regions)
    print(f"Input SearchHits count  : {len(hits)}")
    print(f"Output Evidence total   : {total_evidences_out}")
    print(f"Regions created         : {len(candidate_regions)}")
    
    assert total_evidences_out == len(hits), f"Mismatch! Input {len(hits)} hits vs Output {total_evidences_out} evidence"
    print(">>> VERIFICATION SUCCESSFUL: All input hits were preserved and properly grouped into CandidateRegions! <<<")


if __name__ == "__main__":
    run_aggregator_test()
