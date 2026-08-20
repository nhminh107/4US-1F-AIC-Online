# API Endpoints cho FrontEnd

Base URL khi chạy local:

```text
http://localhost:8000/api/v1
```

## 1. Gửi truy vấn

```http
POST /api/v1/query
Content-Type: application/json
```

Input tối thiểu:

```json
{
  "prompt": "Tìm khung cảnh xe buýt đi trong thành phố"
}
```

Input đầy đủ:

```json
{
  "prompt": "Tìm khung cảnh xe buýt đi trong thành phố",
  "feedback": null,
  "session_id": "session-01",
  "top_k": {
    "clip_search": 30,
    "frame_search": 30,
    "shot_search": 30,
    "ocr_search": 20,
    "asr_search": 20,
    "object_search": 30,
    "track_search": 30,
    "result_top_k": 10
  }
}
```

Có **8 tham số K**:

| Tham số | Ý nghĩa |
|---|---|
| `clip_search` | Số clip lấy từ FAISS |
| `frame_search` | Số frame lấy từ FAISS |
| `shot_search` | Số shot lấy từ FAISS |
| `ocr_search` | Số kết quả chữ trong hình |
| `asr_search` | Số kết quả lời nói |
| `object_search` | Số kết quả object detection |
| `track_search` | Số kết quả tracking |
| `result_top_k` | Số kết quả cuối trả cho FrontEnd |

Có thể bỏ `top_k` hoặc chỉ truyền các giá trị muốn thay đổi. Backend tự dùng giá
trị mặc định cho những trường còn thiếu.

Không gửi `feedback: "string"`. Nếu không có feedback, hãy gửi `null` hoặc bỏ
trường này.

## 2. Output KIS và VQA

KIS và VQA trả cùng cấu trúc `results`. VQA chỉ hiển thị ảnh để người dùng tự
trả lời.

```json
{
  "query_id": "query_123",
  "task": "KIS",
  "results": [
    {
      "video_id": "L22_V014",
      "start_ms": 898967,
      "end_ms": 898967,
      "representative_frame_id": "L22_V014_230",
      "display_frame_id": "L22_V014_230",
      "frame_idx": 26969,
      "img_url": "data/keyframes/L22_V014/230.jpg",
      "score": 1.0,
      "evidence_ids": []
    }
  ]
}
```

Các field FrontEnd cần chú ý:

- `display_frame_id`: ID frame thực tế cần hiển thị.
- `frame_idx`: vị trí frame trong video, lấy từ PostgreSQL.
- `img_url`: đường dẫn ảnh lưu trong PostgreSQL.
- `start_ms`, `end_ms`: thời gian trong video, đơn vị millisecond.
- `score`: điểm xếp hạng; số lớn hơn được ưu tiên.

Nếu frame retrieval là `extracted`, ba field `display_frame_id`, `frame_idx` và
`img_url` sẽ thuộc frame `official` gần nhất.

## 3. Output TRAKE

TRAKE trả danh sách chuỗi sự kiện. Mỗi event có một ảnh để hiển thị:

```json
{
  "task": "TRAKE",
  "trake_status": "success",
  "replan_required": false,
  "results": [
    {
      "video_id": "L22_V014",
      "sequence_score": 0.91,
      "events": [
        {
          "event_id": "E1",
          "candidate_id": "candidate-01",
          "start_ms": 10000,
          "end_ms": 15000,
          "display_frame_id": "L22_V014_010",
          "frame_idx": 300,
          "img_url": "data/keyframes/L22_V014/010.jpg"
        }
      ]
    }
  ]
}
```

Nếu `replan_required=true`, dữ liệu retrieval chưa đủ để tạo chuỗi sự kiện.

## 4. Hiển thị ảnh

Thêm tiền tố: "https://pub-c8f3587e831a418ebf0d427203860188.r2.dev/" vào trước img_url để lấy link ảnh chính xác. 
Sau đó chỉ cần thêm HTML hiển thị ảnh là ok

## 5. Health check

```http
GET /api/v1/health
```

```json
{
  "status": "ok",
  "selective_verifier_enabled": false
}
```
