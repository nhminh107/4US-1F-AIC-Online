from BackEnd.app.contracts.models import CandidateEvidence, CandidateRegion
from BackEnd.app.Fusion.fusion_and_ranking import FusionRanking
from BackEnd.app.Fusion.dynamic_weight import LLM_DynamicWeight
import asyncio


# ==============================================================================
# PROMPT 1 TEST CASES (25 CandidateRegions)
# Theme: Countryside, rural scenery, boy riding a water buffalo, rice fields.
# Entity types used: frame, clip, shot, ocr, asr, object_detection, object_track.
# (entity_type="caption" removed)
# ==============================================================================
candidate_regions_prompt_1: list[CandidateRegion] = [
    CandidateRegion(
        candidate_id="cand_p1_001",
        video_id="video_countryside_01",
        event_id="event_rural_01",
        start_ms=2000,
        end_ms=8000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_c01_001", start_ms=2000, end_ms=2000, rank=1, raw_score=0.96),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_c01_01", start_ms=2000, end_ms=8000, rank=2, raw_score=0.91),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_buffalo_boy_01", start_ms=3000, end_ms=5000, rank=1, raw_score=0.94),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_002",
        video_id="video_countryside_01",
        event_id="event_rural_01",
        start_ms=12000,
        end_ms=18000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_c01_02", start_ms=12000, end_ms=18000, rank=4, raw_score=0.85),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_boy_riding_01", start_ms=12500, end_ms=17500, rank=2, raw_score=0.89),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_003",
        video_id="video_village_02",
        event_id="event_rural_02",
        start_ms=5000,
        end_ms=10000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_v02_050", start_ms=5000, end_ms=5000, rank=3, raw_score=0.89),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_v02_01", start_ms=5500, end_ms=9500, rank=1, raw_score=0.93),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_004",
        video_id="video_village_02",
        event_id="event_rural_02",
        start_ms=25000,
        end_ms=32000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_v02_04", start_ms=25000, end_ms=32000, rank=6, raw_score=0.80),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_buffalo_02", start_ms=26000, end_ms=30000, rank=4, raw_score=0.84),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_005",
        video_id="video_farm_03",
        event_id="event_rural_03",
        start_ms=1000,
        end_ms=6000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_f03_010", start_ms=1000, end_ms=1000, rank=2, raw_score=0.92),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_f03_01", start_ms=1000, end_ms=6000, rank=1, raw_score=0.95),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_f03_01", start_ms=1500, end_ms=3000, rank=5, raw_score=0.78),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_006",
        video_id="video_farm_03",
        event_id="event_rural_03",
        start_ms=15000,
        end_ms=21000,
        evidence=[
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_f03_03", start_ms=15000, end_ms=21000, rank=3, raw_score=0.87),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_f03_03", start_ms=15000, end_ms=21000, rank=5, raw_score=0.83),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_007",
        video_id="video_paddy_field_04",
        event_id="event_rural_04",
        start_ms=4000,
        end_ms=9500,
        evidence=[
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_buffalo_04", start_ms=4000, end_ms=9000, rank=1, raw_score=0.97),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_person_04", start_ms=4500, end_ms=8500, rank=2, raw_score=0.91),
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_pf04_040", start_ms=4000, end_ms=4000, rank=4, raw_score=0.86),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_008",
        video_id="video_paddy_field_04",
        event_id="event_rural_04",
        start_ms=18000,
        end_ms=24000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_pf04_03", start_ms=18000, end_ms=24000, rank=3, raw_score=0.88),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_pf04_02", start_ms=19000, end_ms=23000, rank=6, raw_score=0.79),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_009",
        video_id="video_vietnam_rural_05",
        event_id="event_rural_05",
        start_ms=3000,
        end_ms=7000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_vr05_030", start_ms=3000, end_ms=3000, rank=1, raw_score=0.94),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_vr05_01", start_ms=3000, end_ms=7000, rank=2, raw_score=0.91),
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_vr05_01", start_ms=3000, end_ms=7000, rank=5, raw_score=0.82),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_010",
        video_id="video_vietnam_rural_05",
        event_id="event_rural_05",
        start_ms=10000,
        end_ms=15000,
        evidence=[
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_boy_05", start_ms=10500, end_ms=14500, rank=3, raw_score=0.88),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_011",
        video_id="video_pastoral_06",
        event_id="event_rural_06",
        start_ms=500,
        end_ms=5500,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_pas06_01", start_ms=500, end_ms=5500, rank=1, raw_score=0.96),
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_pas06_005", start_ms=500, end_ms=500, rank=2, raw_score=0.93),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_boy_buffalo_06", start_ms=1000, end_ms=5000, rank=3, raw_score=0.89),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_012",
        video_id="video_pastoral_06",
        event_id="event_rural_06",
        start_ms=20000,
        end_ms=27000,
        evidence=[
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_pas06_03", start_ms=20500, end_ms=26000, rank=2, raw_score=0.91),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_pas06_01", start_ms=21000, end_ms=24000, rank=7, raw_score=0.75),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_013",
        video_id="video_river_village_07",
        event_id="event_rural_07",
        start_ms=8000,
        end_ms=14000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_rv07_080", start_ms=8000, end_ms=8000, rank=2, raw_score=0.90),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_rv07_02", start_ms=8000, end_ms=14000, rank=3, raw_score=0.87),
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_rv07_02", start_ms=8000, end_ms=14000, rank=4, raw_score=0.84),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_014",
        video_id="video_river_village_07",
        event_id="event_rural_07",
        start_ms=30000,
        end_ms=36000,
        evidence=[
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_buffalo_07", start_ms=30500, end_ms=35000, rank=1, raw_score=0.93),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_015",
        video_id="video_country_life_08",
        event_id="event_rural_08",
        start_ms=2000,
        end_ms=7500,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_cl08_01", start_ms=2000, end_ms=7500, rank=1, raw_score=0.95),
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_cl08_020", start_ms=2000, end_ms=2000, rank=3, raw_score=0.89),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_boy_08", start_ms=2500, end_ms=7000, rank=2, raw_score=0.92),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_cl08_01", start_ms=3000, end_ms=6500, rank=5, raw_score=0.81),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_016",
        video_id="video_country_life_08",
        event_id="event_rural_08",
        start_ms=14000,
        end_ms=19000,
        evidence=[
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_cl08_03", start_ms=14000, end_ms=19000, rank=4, raw_score=0.84),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_cl08_03", start_ms=14000, end_ms=19000, rank=6, raw_score=0.79),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_cl08_01", start_ms=15000, end_ms=17000, rank=8, raw_score=0.72),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_017",
        video_id="video_buffalo_boy_09",
        event_id="event_rural_09",
        start_ms=0,
        end_ms=5000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_bb09_000", start_ms=0, end_ms=0, rank=1, raw_score=0.97),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_riding_09", start_ms=500, end_ms=4500, rank=2, raw_score=0.93),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_018",
        video_id="video_buffalo_boy_09",
        event_id="event_rural_09",
        start_ms=11000,
        end_ms=16500,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_bb09_02", start_ms=11000, end_ms=16500, rank=2, raw_score=0.91),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_buffalo_09", start_ms=11500, end_ms=16000, rank=1, raw_score=0.94),
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_bb09_110", start_ms=11000, end_ms=11000, rank=5, raw_score=0.83),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_019",
        video_id="video_fields_10",
        event_id="event_rural_10",
        start_ms=6000,
        end_ms=11000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_f10_060", start_ms=6000, end_ms=6000, rank=3, raw_score=0.88),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_f10_01", start_ms=6500, end_ms=10500, rank=1, raw_score=0.94),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_020",
        video_id="video_fields_10",
        event_id="event_rural_10",
        start_ms=22000,
        end_ms=28000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_f10_04", start_ms=22000, end_ms=28000, rank=4, raw_score=0.86),
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_f10_04", start_ms=22000, end_ms=28000, rank=5, raw_score=0.82),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_boy_10", start_ms=23000, end_ms=27000, rank=3, raw_score=0.87),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_021",
        video_id="video_hometown_11",
        event_id="event_rural_11",
        start_ms=1500,
        end_ms=6500,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_ht11_015", start_ms=1500, end_ms=1500, rank=1, raw_score=0.95),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_ht11_01", start_ms=1500, end_ms=6500, rank=2, raw_score=0.91),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_022",
        video_id="video_hometown_11",
        event_id="event_rural_11",
        start_ms=13000,
        end_ms=17500,
        evidence=[
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_water_buffalo_11", start_ms=13500, end_ms=17000, rank=2, raw_score=0.89),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_ht11_01", start_ms=14000, end_ms=16000, rank=6, raw_score=0.76),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_023",
        video_id="video_harvest_12",
        event_id="event_rural_12",
        start_ms=4500,
        end_ms=9000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_hv12_045", start_ms=4500, end_ms=4500, rank=2, raw_score=0.92),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_hv12_01", start_ms=4500, end_ms=9000, rank=1, raw_score=0.96),
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_hv12_01", start_ms=4500, end_ms=9000, rank=3, raw_score=0.88),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_hv12_01", start_ms=5000, end_ms=8500, rank=5, raw_score=0.82),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_024",
        video_id="video_harvest_12",
        event_id="event_rural_12",
        start_ms=20000,
        end_ms=25000,
        evidence=[
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_boy_riding_12", start_ms=20500, end_ms=24500, rank=1, raw_score=0.92),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p1_025",
        video_id="video_green_meadow_13",
        event_id="event_rural_13",
        start_ms=7000,
        end_ms=12000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_gm13_02", start_ms=7000, end_ms=12000, rank=1, raw_score=0.94),
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_gm13_070", start_ms=7000, end_ms=7000, rank=3, raw_score=0.89),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_buffalo_boy_13", start_ms=7500, end_ms=11500, rank=2, raw_score=0.91),
        ],
    ),
]


# ==============================================================================
# PROMPT 2 TEST CASES (25 CandidateRegions)
# Theme: Sports award ceremony, player receiving trophy/medal on podium.
# Entity types used: frame, clip, shot, ocr, asr, object_detection, object_track.
# (entity_type="caption" removed)
# ==============================================================================
candidate_regions_prompt_2: list[CandidateRegion] = [
    CandidateRegion(
        candidate_id="cand_p2_001",
        video_id="video_sports_award_01",
        event_id="event_sports_01",
        start_ms=5000,
        end_ms=12000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_sa01_050", start_ms=5000, end_ms=5000, rank=1, raw_score=0.97),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_sa01_01", start_ms=5000, end_ms=12000, rank=2, raw_score=0.93),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_podium_01", start_ms=6000, end_ms=10000, rank=4, raw_score=0.85),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_002",
        video_id="video_sports_award_01",
        event_id="event_sports_01",
        start_ms=20000,
        end_ms=28000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_sa01_03", start_ms=20000, end_ms=28000, rank=3, raw_score=0.88),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_trophy_01", start_ms=21000, end_ms=27000, rank=1, raw_score=0.94),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_sa01_02", start_ms=20500, end_ms=26000, rank=5, raw_score=0.81),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_003",
        video_id="video_football_cup_02",
        event_id="event_sports_02",
        start_ms=3000,
        end_ms=9000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_fc02_030", start_ms=3000, end_ms=3000, rank=2, raw_score=0.91),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_fc02_01", start_ms=3000, end_ms=9000, rank=1, raw_score=0.95),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_player_podium_02", start_ms=3500, end_ms=8500, rank=3, raw_score=0.87),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_004",
        video_id="video_football_cup_02",
        event_id="event_sports_02",
        start_ms=15000,
        end_ms=22000,
        evidence=[
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_fc02_03", start_ms=15000, end_ms=22000, rank=4, raw_score=0.84),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_fc02_champion", start_ms=16000, end_ms=20000, rank=2, raw_score=0.90),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_005",
        video_id="video_olympic_ceremony_03",
        event_id="event_sports_03",
        start_ms=10000,
        end_ms=18000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_oc03_100", start_ms=10000, end_ms=10000, rank=1, raw_score=0.98),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_oc03_02", start_ms=10000, end_ms=18000, rank=2, raw_score=0.94),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_oc03_gold_medal", start_ms=11000, end_ms=17000, rank=1, raw_score=0.96),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_medal_03", start_ms=11500, end_ms=16500, rank=3, raw_score=0.90),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_006",
        video_id="video_olympic_ceremony_03",
        event_id="event_sports_03",
        start_ms=30000,
        end_ms=37000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_oc03_05", start_ms=30000, end_ms=37000, rank=4, raw_score=0.85),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_oc03_gold", start_ms=31000, end_ms=35000, rank=5, raw_score=0.81),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_007",
        video_id="video_championship_04",
        event_id="event_sports_04",
        start_ms=2000,
        end_ms=7500,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_cs04_020", start_ms=2000, end_ms=2000, rank=3, raw_score=0.89),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_winner_04", start_ms=2500, end_ms=7000, rank=1, raw_score=0.95),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_cs04_01", start_ms=2000, end_ms=7500, rank=2, raw_score=0.91),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_008",
        video_id="video_championship_04",
        event_id="event_sports_04",
        start_ms=12000,
        end_ms=19000,
        evidence=[
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_cs04_02", start_ms=12000, end_ms=19000, rank=5, raw_score=0.82),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_cs04_award", start_ms=13000, end_ms=18000, rank=4, raw_score=0.84),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_009",
        video_id="video_podium_victory_05",
        event_id="event_sports_05",
        start_ms=4000,
        end_ms=10000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_pv05_040", start_ms=4000, end_ms=4000, rank=1, raw_score=0.96),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_pv05_01", start_ms=4000, end_ms=10000, rank=2, raw_score=0.92),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_pv05_1st", start_ms=4500, end_ms=8500, rank=3, raw_score=0.88),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_podium_05", start_ms=4000, end_ms=9500, rank=1, raw_score=0.94),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_010",
        video_id="video_podium_victory_05",
        event_id="event_sports_05",
        start_ms=25000,
        end_ms=31000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_pv05_04", start_ms=25000, end_ms=31000, rank=5, raw_score=0.83),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_athlete_05", start_ms=25500, end_ms=30500, rank=2, raw_score=0.89),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_011",
        video_id="video_medal_ceremony_06",
        event_id="event_sports_06",
        start_ms=1000,
        end_ms=6500,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_mc06_010", start_ms=1000, end_ms=1000, rank=2, raw_score=0.91),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_mc06_01", start_ms=1000, end_ms=6500, rank=1, raw_score=0.95),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_mc06_01", start_ms=1500, end_ms=5500, rank=3, raw_score=0.87),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_012",
        video_id="video_medal_ceremony_06",
        event_id="event_sports_06",
        start_ms=14000,
        end_ms=20000,
        evidence=[
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_mc06_03", start_ms=14000, end_ms=20000, rank=4, raw_score=0.83),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_mc06_winner", start_ms=15000, end_ms=18500, rank=2, raw_score=0.88),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_013",
        video_id="video_trophy_presentation_07",
        event_id="event_sports_07",
        start_ms=6000,
        end_ms=13000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_tp07_060", start_ms=6000, end_ms=6000, rank=1, raw_score=0.97),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_tp07_01", start_ms=6000, end_ms=13000, rank=2, raw_score=0.93),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_cup_07", start_ms=6500, end_ms=12000, rank=1, raw_score=0.95),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_player_07", start_ms=6000, end_ms=12500, rank=3, raw_score=0.89),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_014",
        video_id="video_trophy_presentation_07",
        event_id="event_sports_07",
        start_ms=22000,
        end_ms=29000,
        evidence=[
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_tp07_congratulations", start_ms=22500, end_ms=28000, rank=2, raw_score=0.90),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_tp07_mvp", start_ms=23000, end_ms=27000, rank=4, raw_score=0.82),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_015",
        video_id="video_tournament_final_08",
        event_id="event_sports_08",
        start_ms=8000,
        end_ms=15000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_tf08_02", start_ms=8000, end_ms=15000, rank=1, raw_score=0.94),
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_tf08_080", start_ms=8000, end_ms=8000, rank=3, raw_score=0.88),
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_tf08_02", start_ms=8000, end_ms=15000, rank=4, raw_score=0.83),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_016",
        video_id="video_tournament_final_08",
        event_id="event_sports_08",
        start_ms=32000,
        end_ms=38000,
        evidence=[
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_podium_08", start_ms=32500, end_ms=37500, rank=2, raw_score=0.89),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_017",
        video_id="video_golden_ball_09",
        event_id="event_sports_09",
        start_ms=3000,
        end_ms=8500,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_gb09_030", start_ms=3000, end_ms=3000, rank=1, raw_score=0.96),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_gb09_01", start_ms=3000, end_ms=8500, rank=2, raw_score=0.92),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_gb09_balondor", start_ms=3500, end_ms=7500, rank=1, raw_score=0.95),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_018",
        video_id="video_golden_ball_09",
        event_id="event_sports_09",
        start_ms=18000,
        end_ms=24000,
        evidence=[
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_footballer_09", start_ms=18500, end_ms=23500, rank=2, raw_score=0.88),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_gb09_speech", start_ms=19000, end_ms=23000, rank=3, raw_score=0.85),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_019",
        video_id="video_stadium_glory_10",
        event_id="event_sports_10",
        start_ms=500,
        end_ms=6000,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_sg10_01", start_ms=500, end_ms=6000, rank=1, raw_score=0.95),
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_sg10_005", start_ms=500, end_ms=500, rank=2, raw_score=0.90),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_athlete_10", start_ms=1000, end_ms=5500, rank=3, raw_score=0.87),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_020",
        video_id="video_stadium_glory_10",
        event_id="event_sports_10",
        start_ms=16000,
        end_ms=23000,
        evidence=[
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_sg10_03", start_ms=16000, end_ms=23000, rank=5, raw_score=0.80),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_sg10_03", start_ms=16000, end_ms=23000, rank=3, raw_score=0.86),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_sg10_ceremony", start_ms=17000, end_ms=21000, rank=2, raw_score=0.89),
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_sg10_announcement", start_ms=16500, end_ms=22000, rank=4, raw_score=0.83),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_021",
        video_id="video_athlete_honored_11",
        event_id="event_sports_11",
        start_ms=7000,
        end_ms=13500,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_ah11_070", start_ms=7000, end_ms=7000, rank=1, raw_score=0.94),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_ah11_02", start_ms=7000, end_ms=13500, rank=2, raw_score=0.91),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_022",
        video_id="video_athlete_honored_11",
        event_id="event_sports_11",
        start_ms=27000,
        end_ms=33000,
        evidence=[
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_podium_step_11", start_ms=27500, end_ms=32500, rank=2, raw_score=0.88),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_stepping_up_11", start_ms=27000, end_ms=32000, rank=1, raw_score=0.92),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_023",
        video_id="video_winner_podium_12",
        event_id="event_sports_12",
        start_ms=2500,
        end_ms=8000,
        evidence=[
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_wp12_025", start_ms=2500, end_ms=2500, rank=2, raw_score=0.93),
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_wp12_01", start_ms=2500, end_ms=8000, rank=1, raw_score=0.96),
            CandidateEvidence(source="shot_detector", entity_type="shot", entity_id="shot_wp12_01", start_ms=2500, end_ms=8000, rank=4, raw_score=0.84),
            CandidateEvidence(source="ocr_search", entity_type="ocr", entity_id="ocr_wp12_gold", start_ms=3000, end_ms=7000, rank=3, raw_score=0.89),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_024",
        video_id="video_winner_podium_12",
        event_id="event_sports_12",
        start_ms=19000,
        end_ms=26000,
        evidence=[
            CandidateEvidence(source="whisper_asr", entity_type="asr", entity_id="asr_wp12_cheering", start_ms=19500, end_ms=25500, rank=3, raw_score=0.86),
            CandidateEvidence(source="yolo_object_detector", entity_type="object_detection", entity_id="obj_trophy_12", start_ms=20000, end_ms=25000, rank=1, raw_score=0.93),
        ],
    ),
    CandidateRegion(
        candidate_id="cand_p2_025",
        video_id="video_grand_stage_13",
        event_id="event_sports_13",
        start_ms=9000,
        end_ms=15500,
        evidence=[
            CandidateEvidence(source="clip_embedding", entity_type="clip", entity_id="clip_gs13_02", start_ms=9000, end_ms=15500, rank=1, raw_score=0.95),
            CandidateEvidence(source="faiss_frame_embedding", entity_type="frame", entity_id="frame_gs13_090", start_ms=9000, end_ms=9000, rank=3, raw_score=0.89),
            CandidateEvidence(source="deepsort_tracker", entity_type="object_track", entity_id="track_player_podium_13", start_ms=9500, end_ms=15000, rank=2, raw_score=0.92),
        ],
    ),
]

async def main():
    user_prompt_1 = "Hãy tìm cho tôi khung cảnh đồng quê có cậu bé đang cưỡi trâu"
    user_prompt_2 = "Tìm cho tôi khung cảnh 1 cầu thủ lên bục nhận giải thưởng"

    llm = LLM_DynamicWeight()
    fusion = FusionRanking()

    #Test1
    llm_out1= await llm.dynamic_weight(user_prompt=user_prompt_1)
    res1 = fusion.fusion_and_ranking(candidate_regions_prompt_1, llm_out1)

    for item in res1:
        print(item)

    #Test2
    llm_out2 = await llm.dynamic_weight(user_prompt=user_prompt_2)
    res2 = fusion.fusion_and_ranking(candidate_regions_prompt_2, llm_out2)
    for item in res2: 
        print(item)

if __name__ == "__main__":
    asyncio.run(main())


