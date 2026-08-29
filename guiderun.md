# Huong Dan Chay Online Pipeline Bang Docker

File nay dung cho folder Online da co `release_package`.

## 1. Dieu kien can co

Trong `release_package` phai co:

```text
release_package/
  postgres_data/
  elasticsearch_data/
  artifacts/faiss/
    frame.faiss
    clip.faiss
    shot.faiss
    faiss_index.json
  data/
```

Luu y: `release_package/data` can chua anh keyframe neu muon route
`/api/v1/frames/{frame_id}` tra anh truc tiep. Database hien luu duong dan
anh dang `data/keyframes/...`, nen trong Docker anh can nam tai
`/app/data/keyframes/...`.

## 2. Cau hinh .env

File `.env` dung cho Docker nen giu cac gia tri service name noi bo:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/aic_hcmc
ELASTICSEARCH_URL=http://elasticsearch:9200
FAISS_INDEX_DIR=/app/artifacts/faiss
```

De goi endpoint `/api/v1/query`, can thay:

```env
FPT_API_KEY=your-real-api-key
FPT_BASE_URL=https://mkp-api.fptcloud.com
```

Neu chi check health/data/search truc tiep thi chua can key that.

Mac dinh runtime nay uu tien CUDA cho online API:

```env
API_GPUS=all
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
TORCH_PACKAGE=torch==2.3.1+cu121
CLIP_MODEL_DEVICE=cuda
```

Neu may khong co NVIDIA GPU hoac Docker chua cai NVIDIA Container Toolkit, doi
ve CPU:

```env
API_GPUS=
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
TORCH_PACKAGE=torch==2.3.1+cpu
CLIP_MODEL_DEVICE=cpu
```

Sau khi doi CUDA/CPU phai build lai image `api`.

## 3. Chay full runtime

Tai root project:

```bat
docker compose --env-file .env -f docker-compose.runtime.yml up -d --build
```

Hoac dung script bat Docker va test data tu dong:

```bat
scripts\docker_up_test_data.bat
```

Lenh nay se chay:

- `postgres`: doc `release_package/postgres_data`
- `elasticsearch`: doc `release_package/elasticsearch_data`
- `api`: chay `uvicorn BackEnd.main:app`
- kiem tra `torch.cuda.is_available()` neu `.env` dang de
  `CLIP_MODEL_DEVICE=cuda`

API mac dinh o:

```text
http://localhost:8000
```

## 4. Kiem tra nhanh

Kiem tra container:

```bat
docker compose --env-file .env -f docker-compose.runtime.yml ps
```

Kiem tra API:

```bat
curl http://localhost:8000/api/v1/health
```

Kiem tra PostgreSQL <-> FAISS:

```bat
docker compose --env-file .env -f docker-compose.runtime.yml --profile tools run --rm data-tools python -m BackEnd.scripts.check_faiss_db_sync
```

Ket qua dung la:

```text
[OK] frame: FAISS ntotal=366380 | DB count=366380 (index_version=0)
[OK] clip: FAISS ntotal=115835 | DB count=115835 (index_version=0)
[OK] shot: FAISS ntotal=96796 | DB count=96796 (index_version=0)
```

Kiem tra Elasticsearch:

```bat
docker compose --env-file .env -f docker-compose.runtime.yml --profile tools run --rm data-tools python scripts/search_es_direct.py tin --source ocr --top-k 3
```

## 5. Goi query online

Sau khi da co `FPT_API_KEY` that:

Chay menu co ban de chon bai 1/2/3, khong can Python tren may host:

```bat
scripts\run_online_task.bat
```

Trong menu, script se hoi theo thu tu:

1. Chon bai `1/2/3`.
2. Nhap `result_top_k` muon tra ve, vi du `30`.
3. Nhap prompt truy van.

Chay truc tiep tung bai de test BTC:

```bat
scripts\run_online_task.bat 1
scripts\run_online_task.bat 2
scripts\run_online_task.bat 3
```

Neu muon chi dinh K ngay tren terminal:

```bat
scripts\run_online_task.bat 1 --preset balanced --k 30 "tim canh co xe buyt tren duong"
scripts\run_online_task.bat 2 --preset balanced --k 30 "nguoi trong canh dang cam vat gi?"
scripts\run_online_task.bat 3 --preset accurate --k 30 "tim video co chuoi su kien: nguoi di vao, sau do ngoi xuong"
```

Hoac truyen prompt rieng:

```bat
scripts\run_online_task.bat 1 "tim canh co xe buyt tren duong"
scripts\run_online_task.bat 2 "nguoi trong canh dang cam vat gi?"
scripts\run_online_task.bat 3 "tim video co chuoi su kien: nguoi di vao, sau do ngoi xuong"
```

Chay ca 3 bai bang prompt mac dinh:

```bat
scripts\run_online_task.bat all
```

Kiem tra payload gui len API ma chua goi LLM:

```bat
scripts\run_online_task.bat 1 --dry-run
scripts\run_online_task.bat all --preset accurate --dry-run
```

Preset trong script:

- `fast`: dung khi can phan hoi nhanh, recall vua phai.
- `balanced`: mac dinh, nen dung cho test de thi thong thuong.
- `accurate`: tang recall cho cau kho, TRAKE, cau co OCR/ASR/object; cham hon.

Tat ca preset van goi endpoint online chuan `POST /api/v1/query`; script khong
tu search truc tiep vao PostgreSQL, Elasticsearch hay FAISS.

Neu may co Python host thi co the dung ban Python tuong duong:

```bat
python scripts\run_online_task.py 1
python scripts\run_online_task.py 2
python scripts\run_online_task.py 3
```

Trong menu:

- `1`: KIS
- `2`: VQA
- `3`: TRAKE

Script se tu them tien to `KIS:`, `VQA:`, hoac `TRAKE:` vao prompt neu ban
chua nhap tien to. Viec nay giup `/api/v1/query` phan loai dung bai vi API
hien khong co field `task` rieng trong request.

Ket qua terminal se in cac field BTC can dung:

- KIS/VQA: `video_id`, `start_ms`, `end_ms`, `representative_frame_id`,
  `display_frame_id`, `frame_idx`, `img_url`, `image_url`.
- TRAKE: `trake_status`, `replan_required`, `missing_event_ids`, tung event co
  `display_frame_id`, `frame_idx`, `img_url`, `image_url`.

Theo `BackEnd/app/api/API_ENDPOINTS.md`, VQA hien tai la che do human-review:
backend tra anh bang chung de nguoi dung tu tra loi, chua tu sinh cau tra loi
VQA bang VLM.

Hoac goi truc tiep bang `curl`:

```bat
curl -X POST http://localhost:8000/api/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"tim canh co nguoi dang di bo\",\"top_k\":{\"clip_search\":10,\"result_top_k\":5}}"
```

Lan dau goi visual query co the cham vi container can tai/cache model
`sentence-transformers/clip-ViT-B-32-multilingual-v1`. Cache duoc giu trong
Docker volume `model-cache`, nen cac lan sau se nhanh hon.

## 6. Tat runtime

Tat container nhung giu data:

```bat
docker compose --env-file .env -f docker-compose.runtime.yml down
```

Khong dung `-v` neu khong muon xoa Docker volume cache model.
