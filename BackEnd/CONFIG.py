FRAME_EMBEDDING_WEIGHT = 1.0
SHOT_EMBEDDING_WEIGHT = 1.0 
CLIP_EMBEDDING_WEIGHT = 1.0
OCR_WEIGHT = 1.0
ASR_WEIGHT = 1.0
TRACK_WEIGHT = 1.0
RRF_K = 60
RANKING_MODEL = "gpt-oss-20b"
STD_BUFF = 1.5
DETECT_WEIGHT = 1.0

# ==== Module 6A - KIS Handler ====
# Số lượng RankedCandidateRegion tối đa (đầu danh sách, đã sort theo fusion_score)
# mà KIS Handler sẽ xử lý trong 1 lần gọi execute().
TOP_N_KIS = 10

# Tỉ lệ (0-1) tính từ 2 đầu mút [start_ms, end_ms] của CandidateRegion.
# Nếu "anchor" (mốc thời gian trọng tâm suy ra từ evidence) rơi vào vùng sát biên
# này thì coi là rủi ro "khoảnh khắc thật nằm ngoài region" -> phải mở rộng tìm
# kiếm frame sang shot lân cận thay vì chỉ tin vùng [start_ms, end_ms] gốc.
KIS_EDGE_RATIO = 0.1

# Số shot lân cận (trước/sau) được lấy thêm khi 1 shot đã biết nhưng chưa có
# frame nào trong CSDL (Level 2 - Neighbor Expansion qua get_temporal_neighbors).
KIS_NEIGHBOR_SHOT_COUNT = 1