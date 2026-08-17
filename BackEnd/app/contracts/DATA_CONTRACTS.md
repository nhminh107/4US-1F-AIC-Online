# Data contracts của Online Pipeline

Tài liệu này mô tả các schema trong `models.py`. Contract là dữ liệu trao đổi
giữa các module; mỗi module chỉ nên phụ thuộc vào contract, không phụ thuộc vào
cách module khác truy vấn PostgreSQL, FAISS hoặc Elasticsearch.

## Quy ước chung

- Mọi mốc thời gian dùng `*_ms`, đơn vị millisecond.
- Một khoảng thời gian hợp lệ có `end_ms >= start_ms`. Hai giá trị bằng nhau
  biểu diễn một thời điểm, ví dụ keyframe.
- ID phải là canonical ID có thể resolve ở Shared Data & Evidence Layer.
- `confidence` luôn nằm trong `[0, 1]`; `KISResult.score` cũng dùng thang này.
  `raw_score`, `fusion_score` và `sequence_score` giữ nguyên thang điểm của
  module tạo ra chúng.
- Tất cả contract kế thừa `ContractModel`: immutable sau khi tạo và từ chối
  field không thuộc schema. Điều này giúp phát hiện sớm lỗi tích hợp module.

## Contract nền tảng

| Contract | Ý nghĩa | Producer → Consumer |
|---|---|---|
| `ContractModel` | Cấu hình Pydantic dùng chung: immutable và không nhận field lạ. | Tất cả contract |
| `TimeRangeModel` | Base cho dữ liệu gắn với một khoảng/thời điểm trong video. | `SearchHit`, candidate, evidence, result |
| `CanonicalEntityRef` | Địa chỉ chuẩn tới video và tùy chọn shot, clip, frame cùng thời gian. | Shared Evidence Layer → utility tools/UI |
| `Provenance` | Nguồn gốc evidence: model, version, lần chạy và confidence. | Offline pipeline → Shared Evidence Layer |
| `EvidenceItem` | Contract evidence tổng quát, giữ lại cho các use case cần dữ liệu phẳng. | Shared Evidence Layer → use case generic |
| `TemporalNeighbors` | Các entity chuẩn đứng trước và sau một entity/candidate. | Shared Evidence Layer → KIS/UI |
| `MediaReference` | Đường dẫn local hoặc URL để hiển thị entity trên UI. | Shared Evidence Layer → UI |

### `CanonicalEntityRef`

Được dùng khi một module cần trỏ chính xác đến dữ liệu Offline mà không tự JOIN
giữa các bảng. `video_id` và khoảng thời gian là bắt buộc; `shot_id`, `clip_id`,
`frame_id` có mặt khi entity tương ứng đã được resolve. Các ID chi tiết đều
optional: một vùng candidate có thể chỉ cần `video_id` và thời gian, trong khi
OCR/object evidence cần thêm `frame_id` để truy vết chính xác.

### `EvidenceItem` và `Provenance`

`EvidenceItem` là contract tổng quát cho trường hợp cần một danh sách evidence
phẳng. Vì kế thừa `CanonicalEntityRef`, item luôn có `video_id`, thời gian và có
thể mang `frame_id`, `shot_id`, `clip_id`. `EvidenceBundle` không dùng type này
nữa: bundle dùng các contract chuyên biệt để không làm mất field đặc thù của OCR,
object detection, track và caption.

## Query và planning

| Contract | Ý nghĩa | Producer → Consumer |
|---|---|---|
| `RawQuery` | Input nguyên bản từ người dùng. | UI → Intent Extractor |
| `Event` | Một event được tách từ câu truy vấn. | Intent Extractor → `StructuredQuery` |
| `TemporalConstraint` | Quan hệ thứ tự/gap giữa hai event. | Intent Extractor → TRAKE Aligner |
| `StructuredQuery` | Ý nghĩa truy vấn đã chuẩn hóa cho toàn pipeline. | Intent Extractor → Fast Path/Planner/Fusion/Handlers |
| `ToolCall` | Một yêu cầu retrieval do Planner tạo. | Planner → Retrieval Tools |

### `RawQuery`

`text` là field tối thiểu. `query_id` và `session_id` có thể được UI hoặc service
tạo sau; `image_ref`, `video_ref` phục vụ image/video query; `feedback` chứa
phản hồi tự do ở phiên hiện tại.

### `StructuredQuery`

`task` xác định nhánh xử lý: `KIS`, `VQA` hoặc `TRAKE`. Contract giữ `question`
cho VQA, các query/constraint theo modality và feedback đã chuẩn hóa. Với TRAKE,
`events` chứa các event cần tìm, còn `temporal_constraints` mô tả cặp `before` /
`after`, có thể đặt `max_gap_ms` và `allow_overlap`.

`temporal_constraint` (số ít) vẫn được chấp nhận khi đọc payload cũ và được cung
cấp như một property tương thích ngược; payload mới phải dùng
`temporal_constraints`.

### `ToolCall`

Là lời gọi deterministic tới retrieval tool. `parameters` chứa input riêng của
tool, ví dụ `query`, `top_k` hoặc `object_class`. Khi Planner tìm một event
TRAKE, `event_id` phải được giữ để Candidate Aggregator không gộp evidence của
các event khác nhau.

## Retrieval, aggregation và ranking

| Contract | Ý nghĩa | Producer → Consumer |
|---|---|---|
| `SearchHit` | Kết quả chuẩn của mọi retrieval tool. | Fast Path/Retrieval Tools → Candidate Aggregator |
| `CandidateEvidence` | Phần evidence được giữ lại khi gộp hit. | Candidate Aggregator → Candidate/Fusion |
| `CandidateRegion` | Vùng thời gian gom các hit cùng video và event. | Candidate Aggregator → Hybrid Fusion |
| `ConstraintResult` | Kết quả hard/negative constraint của Fusion. | Hybrid Fusion → ranked candidate |
| `RankedCandidateRegion` | Candidate đã có fusion score và trạng thái constraint. | Hybrid Fusion → KIS/VQA/TRAKE |

### `SearchHit`

Mọi `frame_search`, `clip_search`, `ocr_search`, `asr_search`, `object_search`
và các tool khác phải trả contract này. `source` cho biết retriever; `entity_type`
và `entity_id` xác định evidence. Vì kế thừa `CanonicalEntityRef`, hit đã có
`video_id`, thời gian và có thể kèm `frame_id`, `shot_id`, `clip_id` theo
granularity của retriever. `rank` bắt đầu từ 1, `raw_score` được giữ nguyên theo
retriever để Fusion chuẩn hóa theo source khi cần. `tool_call_id` liên kết với
Planner và `event_id` bảo toàn ngữ cảnh TRAKE.

### `CandidateRegion` và `RankedCandidateRegion`

Aggregator chỉ gộp evidence overlap/gần nhau trong **cùng video và cùng event**
thành `CandidateRegion`. Nó không chấm điểm cuối. Hybrid Fusion bổ sung
`fusion_score` và `constraint_result` để tạo `RankedCandidateRegion`; evidence
gốc vẫn phải được giữ để Handlers và Verifier giải thích kết quả.

## Evidence và kết quả cuối

| Contract | Ý nghĩa | Producer → Consumer |
|---|---|---|
| `EvidenceBundle` | Context đa modality của một vùng video. | Shared Evidence Layer → VQA/Verifier |
| `KISResult` | Khoảnh khắc/frame đại diện cho Known-Item Search. | KIS Handler → Verifier/UI |
| `VQAResult` | Câu trả lời được VLM sinh từ evidence. | VQA Handler → Verifier/UI |
| `TemporalEventResult` | Candidate được chọn cho một event trong chuỗi. | TRAKE Aligner → `TemporalSequence` |
| `TemporalSequence` | Chuỗi event cùng video thỏa thời gian. | TRAKE Aligner → Verifier/UI |
| `VerifiedResult` | Phán quyết kiểm chứng có chọn lọc. | Verifier → UI |

### `EvidenceBundle`

Bundle được tải theo một `video_id` và vùng thời gian. Mỗi danh sách giữ đúng
contract do adapter tạo ra: `ShotMetadata`, `ClipWindowMetadata`,
`FrameMetadata`, `OCRResult`, `TranscriptSegmentResult`, `CaptionResult`,
`ObjectDetectionResult` và `ObjectTrackResult`. Shot và clip giữ context cấu
trúc/media, còn các modality còn lại cung cấp evidence cho VQA/Verifier.

### `KISResult`

KIS Handler chọn một `representative_frame_id` trong vùng tốt nhất. `score` là
điểm kết quả chuẩn hóa, `evidence_ids` giúp UI và Verifier giải thích vì sao
frame này được chọn.

### `VQAResult`

`status="answered"` biểu thị đã có câu trả lời; `status="uncertain"` dùng khi
evidence không đủ và không được ép VLM đoán. `evidence_ids` chỉ chứa ID thuộc
EvidenceBundle đã đưa vào quá trình trả lời.

### `TemporalSequence`

TRAKE Aligner chọn các `TemporalEventResult` trong cùng `video_id`, bảo đảm thứ
tự và gap theo `TemporalConstraint`. `sequence_score` đánh giá cả chuỗi, khác
với `fusion_score` vốn chỉ đánh giá một candidate/event.

### `VerifiedResult`

Verifier trả một trong ba trạng thái: `accepted`, `rejected`, `uncertain`.
`supporting_evidence_ids` là bằng chứng hỗ trợ kết luận; `failed_constraints`
liệt kê constraint không đạt để UI hoặc Planner có thể hiển thị/replan đúng lý do.
