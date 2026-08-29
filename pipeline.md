# Thiết kế Pipeline V2 truy xuất video cho KIS, QA và TRAKE

> Cập nhật ngày 24/08/2026 sau khi đối chiếu dữ liệu runtime, thể lệ, FAQ,
> thông báo chính thức của AIC26 và toàn bộ artifact của các lượt truy xuất DB
> trực tiếp đã thực hiện. Kiến trúc V2 không phụ thuộc agent trên critical
> path; agent chỉ là extension point có thể bật sau.

## 0. Trạng thái triển khai ngày 24/08/2026

V2 deterministic đã được cài song song với V1, nằm chủ yếu tại
`BackEnd/app/retrieval_v2/` và được nối vào `BackEnd/app/api/pipeline.py` qua
feature flag. Compose mặc định vẫn để `RETRIEVAL_V2_ENABLED=false`; runtime local
hiện đã bật V2, mount hai artifact retrieval và báo readiness qua `/health`.

Đã triển khai:

- atom typed theo modality, role, entity/action/count/relation/attribute và
  event; visual atom bắt buộc English, OCR/ASR giữ nguyên ngôn ngữ nguồn;
- prompt theo năm vai trò và weight dựa trên corpus IDF khi có artifact;
- global progressive-depth, canonical moment cap 1.600-2.000, video-index lane
  có quota riêng và local search theo video + time window;
- batched text embedding, subset FAISS cache, native filter PostgreSQL/ES và
  failure isolation theo từng call/retriever family;
- `MomentBand`, coverage matrix tách retrieval/verification, family-aware score,
  hard gate tri-state, scoped rejection và tối đa hai diagnostic retry;
- sequence alignment cho KIS, official-frame selector query-aware, monotonic
  official-frame assignment cho TRAKE, grounded QA contract và exporter dùng
  đúng schema API `results`;
- audit log gồm coverage/gate/review/round, versioned corpus/video-index builder,
  fixture replay và production-controller smoke replay;
- corpus stats đã build từ metadata visual của 873 video; video-level index đã
  build 3.492 representative từ 96.796 shot cho đủ 873 video;
- reranker tầng hai chấm lại official frame theo từng visual atom bằng CLIP;
  grounded VQA provider được tự động nối khi có `VQA_API_KEY` và `VQA_MODEL`;
- regression tests cho wrong video/moment, modality leak, no-gain retry,
  Vietnamese fallback, TRAKE row limit và API/exporter contract.

Chưa được coi là hoàn thành production:

- chưa có VLM/storyboard reviewer chấm atom `PASS/FAIL`; reviewer fallback vẫn
  trả `uncertain`, không giả vờ đã xem ảnh. Reranker official-frame hiện dùng
  CLIP cùng embedding family, chưa phải SigLIP/cross-encoder độc lập;
- câu VQA thuần thị giác dùng grounded provider khi cấu hình model; không có
  provider thì trả `uncertain`, còn OCR/ASR chỉ trả lời khi query khai báo đúng nguồn;
- fixture `reviewed` là artifact đã xác nhận, không phải kết quả model. Phần
  `controller_smoke` chỉ chứng minh orchestration, không phải live-retrieval
  quality benchmark;
- Docker live smoke đã cứu query mưa từ không có kết quả lên video đúng
  `L21_V026` ở hạng 4 và giảm plan từ 30 xuống 15 atom. Chưa chạy đủ năm KIS,
  QA, TRAKE và chưa có p95 profile nên chưa đạt release gate;
- chưa đủ held-out label để calibrate confidence và quyết định bỏ V1.

Điều kiện bật V2 production vẫn là bốn gate ở mục 18.3; unit test xanh không
thay thế các gate đó.

## 1. Mục tiêu

Tài liệu này đề xuất kiến trúc thay thế cho phần truy xuất online hiện tại, với
bốn mục tiêu bắt buộc:

1. Tìm đúng loại bằng chứng mà câu truy vấn mô tả. Một đối tượng nhìn thấy trong
   ảnh chỉ được tìm bằng dữ liệu thị giác/object; từ đó xuất hiện trong OCR hoặc
   ASR không được phép làm video tăng hạng.
2. Phục vụ đủ ba loại bài: KIS, QA (được đặt tên `VQA` trong contract hiện tại)
   và TRAKE.
3. Giữ recall cao để có thể xuất tối đa 100 đáp án, nhưng ưu tiên mạnh độ chính
   xác của các vị trí đầu.
4. Chạy tốt khi chưa có agent. Sau này có thể cắm agent vào các interface lập kế
   hoạch, phản biện và xem ảnh mà không thay retrieval core.

Nguyên tắc trung tâm của thiết kế là **modality firewall**: mỗi mệnh đề trong
truy vấn có một loại bằng chứng được phép. Không có bước nào được tự động phát
tán cùng một từ khóa sang tất cả nguồn dữ liệu.

---

## 2. Kết luận audit pipeline hiện tại

### 2.1. Luồng đang chạy

Entry point chính nằm tại `BackEnd/app/api/pipeline.py`:

```text
Raw prompt
  -> Intent Extractor
  -> Fast Path hoặc Query Planner
  -> FAISS / Elasticsearch / PostgreSQL tools
  -> Aggregator
  -> FusionRanking
  -> KISHandler hoặc TrakeTemporalAligner
  -> Selective Verifier
  -> API result
```

Các nguồn dữ liệu hiện có là phù hợp để xây pipeline mới:

| Nguồn | Dữ liệu | Vai trò đúng |
|---|---|---|
| `frame.faiss` | embedding frame | chi tiết/đối tượng/trạng thái tại một thời điểm |
| `clip.faiss` | embedding clip window | hành động và diễn biến ngắn |
| `shot.faiss` | embedding shot | cảnh rộng, bối cảnh và bố cục |
| `aic_hcm2026_text_ocr_active` | chữ nhìn thấy | visible text |
| `aic_hcm2026_text_transcript_active` | lời nói | spoken content |
| PostgreSQL object detection | object theo frame | class, số lượng, vị trí |
| PostgreSQL object track | object liên tục trong shot | chuyển động và tính liên tục |
| PostgreSQL | metadata/canonical mapping | nguồn chân lý cho video, thời gian và frame |

### 2.2. Elasticsearch gộp physical index nhưng tách evidence bằng filtered alias

Runtime hiện chỉ có một physical index `aic_hcm2026_text_v1`. Các loại dữ liệu
được expose qua filtered alias:

- OCR: `aic_hcm2026_text_ocr_active`;
- ASR: `aic_hcm2026_text_transcript_active`;
- object: `aic_hcm2026_text_object_active`;
- metadata: `aic_hcm2026_text_metadata_active`;
- caption: `aic_hcm2026_text_caption_active`.

Mỗi alias có filter `term source_type`, vì vậy `ocr_search` và `asr_search` hiện
không đọc chéo dữ liệu dù cùng physical index. Cách lưu này không phải nguyên
nhân trực tiếp của lỗi. Caption alias tồn tại nhưng có 0 document và
`caption_search` cũng đã bị gỡ khỏi online code.

Hiện tượng một noun thị giác khớp lời nói/chữ chỉ xảy ra nếu code query thẳng
physical index, dùng wildcard alias sai cách, Intent Extractor phân loại sai,
hoặc Fusion chấp nhận evidence không được query yêu cầu.

Pipeline mới cấm physical index ở retrieval registry. Chỉ typed alias được phép
tham gia. Plan validator phải kiểm tra cả tool lẫn alias đích trước khi chạy.

### 2.3. Các vấn đề cần sửa

#### P0 - Không có contract bảo vệ modality

`StructuredQuery` hiện chỉ có các list rời như `visual_queries`,
`ocr_constraints`, `asr_constraints`, `object_constraints`. Nó không lưu:

- mệnh đề nào trong câu gốc sinh ra constraint;
- loại bằng chứng bắt buộc hay chỉ hỗ trợ;
- retriever nào được phép hoặc bị cấm;
- polarity `MUST`, `SHOULD`, `MUST_NOT`;
- event mà constraint thuộc về trong mọi trường hợp.

LLM có thể gán “phi thuyền” vào OCR/ASR dù người dùng nói “ảnh có hình phi
thuyền”. Sau đó hệ thống không còn đủ provenance để phát hiện lỗi.

#### P0 - Fusion cộng điểm theo nguồn có mặt, không theo ý định truy vấn

`BackEnd/app/Fusion/fusion_and_ranking.py` đang:

- đặt tất cả trọng số frame, clip, shot, OCR, ASR, object và track bằng `1.0`;
- dùng best rank của mỗi `entity_type`, bỏ qua `raw_score`;
- tăng weight cho loại evidence chỉ vì loại đó có hit trong batch;
- không tính coverage theo từng mệnh đề;
- kiểm tra negative constraint bằng `entity_id` và `source`, không phải nội dung;
- chỉ kiểm tra có một object hit bất kỳ, không kiểm tra đúng class được yêu cầu.

Đây là nơi bằng chứng sai modality có thể được hợp thức hóa và tăng hạng.

#### P0 - QA hiện chưa trả lời câu hỏi

Trong `BackEnd/app/api/pipeline.py`, nhánh `VQA` dùng thẳng `KISHandler` và trả
danh sách ảnh cho người xem. `BackEnd/app/vqa/README.md` cũng xác nhận handler
không gọi model và không sinh answer. Contract có `VQAResult`, client VLM và
verifier liên quan, nhưng chúng chưa được nối vào active pipeline.

#### P1 - Fast Path gọi thừa visual retriever

Với mỗi visual query, `BackEnd/app/fast_path/runner.py` gọi đồng thời frame,
clip và shot search. Cách này tăng chi phí, tạo nhiều hit tương quan và khiến
fusion tưởng rằng có ba bằng chứng độc lập, trong khi chúng có thể đến từ cùng
một embedding concept.

`QueryPlanner` có hướng dẫn chọn `top_k` theo độ đặc hiệu, nhưng
`OnlinePipeline._apply_top_k()` lại ghi đè bằng cấu hình cố định của request.
Vì vậy planner hiện không thực sự điều chỉnh được retrieval depth theo atom.

#### P1 - Aggregator trộn hit theo thời gian nhưng không biết mệnh đề

`BackEnd/app/aggregator.py` nhóm theo `(video_id, event_id)` rồi nối mọi hit gần
nhau trong 1 giây. OCR, ASR, visual và object có thể bị gom vào cùng region dù
không phục vụ cùng constraint. `CandidateEvidence` không có `query_atom_id`,
modality, match type hoặc vai trò required/supporting.

#### P1 - Object count hiện có thể đếm sai

`BackEnd/app/retrieval_tools/object.py` nhóm detection theo shot nếu frame có
`shot_id`, rồi dùng tổng số detection trong cả nhóm làm `min_count`. Một người
được detect ở nhiều frame có thể bị hiểu nhầm thành nhiều người cùng lúc. Count
phải group theo `frame_id`; track chỉ dùng để kiểm tra tính liên tục.

#### P1 - TRAKE second pass quá hẹp

Khi thiếu event, pipeline chỉ gọi lại `clip_search(top_k=500)` và chỉ giữ video
đã xuất hiện ở pass đầu. Video đúng nhưng chưa có trong shortlist không thể được
cứu; event cần OCR, object hoặc frame-level detail cũng không được tìm đúng
modality.

#### P1 - Serving top-k và submission top-k đang bị gộp

`result_top_k` mặc định lấy từ `VQA_MAX_CANDIDATES=10` rồi được dùng cho cả KIS
và TRAKE. UI top-10 hợp lý cho tương tác, nhưng submission được phép 100 dòng.
V2 phải tách `retrieval_k`, `rerank_k`, `review_k`, `display_k` và `submission_k`;
không dùng một biến top-k cho năm mục tiêu khác nhau.

#### P2 - KIS chọn frame chưa query-aware

`KISHandler` chọn frame gần temporal anchor nhưng chưa dùng nội dung query để
rerank các frame trong region. Một region đúng video nhưng frame đại diện có thể
không hiển thị hành động/đối tượng quan trọng nhất.

#### P2 - Verifier chưa đủ ngân sách visual review

`VerificationConfig` mặc định `max_vlm_calls_per_result=0`,
`max_reranker_calls_per_result=0`, chỉ kiểm tra tối đa ba candidate. Đây là
selective deterministic verifier, chưa phải top-5 visual review theo yêu cầu vận
hành thực tế.

### 2.4. Hồ sơ dữ liệu runtime

Các số dưới đây được đo trực tiếp trên PostgreSQL, FAISS và Elasticsearch đang
chạy ngày 22/08/2026:

| Thành phần | Quy mô/coverage | Hệ quả thiết kế |
|---|---:|---|
| Video | 873 | Có thể chấm video-level coverage trên toàn corpus |
| Thời lượng video | median 324 giây, trung bình 539 giây, p95 1.433 giây | Cần coarse-to-fine; một video có thể rất dài |
| Frame | 366.380 | Global frame search vẫn khả thi |
| Official frame | 177.321, đủ 873 video | Chỉ dùng cho output và review ổn định |
| Extracted frame | 189.059, đủ 873 video | Dùng để tăng recall và định vị moment |
| Clip window | 115.835; median 3,2 giây, tối đa 10 giây | Retriever chính cho action/moment |
| Shot | 96.796; median 2,72 giây nhưng max 1.171 giây | Phải chặn shot outlier trước aggregation |
| FAISS | frame 366.380, clip 115.835, shot 96.796 | Cả ba index khớp DB ở `index_version=0` |
| OCR | 1.270.545 region trên 169.742 frame, đủ 873 video | ES gom regions thành một document/frame |
| ASR | 134.742 segment trên 859/873 video | Thiếu ASR không được xem là bằng chứng phủ định |
| Object detection | 1.332.453 detection | Có thể làm count/spatial verification mạnh |
| Object tracking | 228.070 track, 1.240.464 observation | Hữu ích cho continuity/movement trong shot |
| Object vocabulary | 601 class | Allow-list 30 class hiện tại làm mất nhiều recall |
| Caption | 0 | Không đưa caption retrieval vào V2 ban đầu |

Một OCR document ES tương ứng một frame và chứa nhiều `regions` nested. Vì vậy
169.742 OCR document so với 1.270.545 PostgreSQL row là khác granularity, không
phải mất dữ liệu.

### 2.5. Ràng buộc từ thể lệ và thông báo chính thức

- Mỗi query nộp một CSV không header, UTF-8, delimiter dấu phẩy, tối đa 100 dòng.
- KIS: `video_id,frame_idx`.
- QA: `video_id,frame_idx,answer`; answer tối đa 100 ký tự và phải CSV-escape.
- TRAKE: một `video_id` và đúng N frame cho N event, theo đúng thứ tự event.
- File ZIP phải chứa thư mục `submission/`, không nén trực tiếp các CSV.
- Video YouTube private vẫn là candidate hợp lệ; dữ liệu BTC cung cấp mới là
  nguồn chính.
- Public leaderboard chỉ phản ánh khoảng 50% đáp án. Không tune pipeline chỉ
  theo public feedback; QA/TRAKE có thể chỉ hiện ở private trong một số đợt.
- Rules chung ghi tối đa 3 lần nộp, nhưng thông báo ngày 21/08/2026 tăng đợt 1
  lên 4. Số lần nộp phải là cấu hình theo round, không được hard-code.
- Trang rules không công khai công thức điểm cụ thể; công thức thuộc từng vòng.
  Pipeline phải hỗ trợ ranking-aware top-100 nhưng không giả định score formula
  không có trong dữ liệu cấu hình.

### 2.6. Quyết định xây lại

Không nên vá tiếp `OnlinePipeline` hiện tại. Nên tạo `OnlinePipelineV2` chạy
song song, shadow-evaluate rồi chuyển traffic. Phạm vi giữ và thay:

| Giữ lại | Xây lại/thay thế |
|---|---|
| PostgreSQL schema và `PostgreManager` | `StructuredQuery` dạng list rời |
| FAISS registry, ID mapping và sync check | Fast Path/Query Planner chia đôi logic |
| filtered ES aliases | Aggregator merge mù modality |
| official-frame resolver | Fusion equal-weight/source-presence boost |
| evidence service | KIS anchor-only moment selection |
| TRAKE DP/beam aligner core | QA dùng KIS output, chưa trả answer |
| deterministic verifier dùng được | submission/output chưa thành module khép kín |

Đây là rebuild orchestration và ranking, không phải rebuild database từ đầu.

### 2.7. Bài học từ quy trình truy xuất DB trực tiếp đã dùng

Các lượt giải trực tiếp trước đây không dùng online pipeline. Tool đọc
PostgreSQL, typed Elasticsearch alias và FAISS theo từng lệnh độc lập. Quy trình
thực tế thành công là:

```text
probe dữ liệu
  -> tách mô tả thành nhiều prompt visual tiếng Anh ngắn
  -> global frame/clip/shot search với recall cao
  -> gộp RRF để lập danh sách giả thuyết video
  -> xem top frame và tải evidence theo video/time window
  -> kiểm tra atom còn thiếu, cảnh trước/sau và frame lân cận
  -> loại video hoặc chỉ loại moment sai
  -> đổi prompt/retriever cho atom yếu rồi tìm lại
  -> khi đúng video: quét dày official frame trong dải thời gian
  -> xếp center/transition frame tốt nhất lên đầu và xuất CSV/TXT
```

Các artifact cho năm query KIS cho thấy global visual search mỗi query sinh
khoảng 1.600-2.000 candidate. Query 4 và 20 tìm đúng video ngay ở đầu; query 19
và 18 có cả top 5 ban đầu sai video; query 24 tìm đúng video nhưng sai moment.
Vì vậy, `review top 5` không phải điểm kết thúc. Nó là một phép đo để quyết định
retry ở phạm vi atom, moment hay video.

Điểm mạnh cần giữ trong V2:

- tool theo modality độc lập và DB read-only;
- nhiều prompt visual tiếng Anh, mỗi prompt nhấn một nhóm thuộc tính;
- retrieval sâu để bảo vệ recall;
- xem evidence theo video và cửa sổ thời gian, không chỉ xem một ảnh;
- neighbor expansion và map cuối về official frame;
- có URL fallback để ảnh luôn review được;
- sau khi chắc video/moment, dùng nhiều official frame gần nhau để phủ sai số.

Điểm không nên bê nguyên vào production:

- clip/shot bị map về midpoint quá sớm, làm mất extent thật;
- RRF coi frame/clip/shot như phiếu độc lập dù cùng embedding family;
- không có `atom_id`, score calibration, reject scope hoặc retry state;
- thao tác xem ảnh và ghi nhớ video đã loại nằm trong trí nhớ người làm;
- một top-k rất lớn được tạo nhưng chưa có cơ chế refinement có cấu trúc;
- phân bổ dòng submission được quyết định thủ công sau khi đã biết video đúng.

### 2.8. Ma trận so sánh ba cách

| Năng lực | Truy xuất DB trực tiếp | Online V1 hiện tại | V2 sau thiết kế này |
|---|---|---|---|
| Modality routing | tốt do người vận hành chọn tool | dễ rò OCR/ASR qua intent/fusion | policy compiler + firewall |
| Prompt decomposition | nhiều prompt thủ công | list query nhưng thiếu provenance | typed atom + prompt variants |
| Global recall | cao, thường top-k 1.000 mỗi lane | phụ thuộc top-k request cố định | adaptive dual-lane |
| Local narrowing | evidence/neighbors thủ công | yếu | filtered local search + refinement loop |
| Đơn vị candidate | frame rời sau khi midpoint resolve | temporal region merge theo khoảng gần | `MomentBand` giữ extent và atom coverage |
| Temporal KIS | người làm xem diễn biến trước/sau | anchor frame | internal `KIS_SEQUENCE` alignment |
| Hard filter | quyết định thủ công, khó tái hiện | rải rác và có lỗi semantics | một `HardConstraintEngine` ba trạng thái |
| Trí nhớ retry | file JSON và trí nhớ người làm | gần như không có | `SearchSessionState` + hypothesis ledger |
| Visual review | hiệu quả nhưng thủ công | mặc định gần như tắt | gate có budget, có escalation |
| Top-100 | tốt sau khi đã xác nhận đáp án | serving top-k lẫn submission top-k | allocation theo loại bất định |
| QA | có thể soi evidence nhưng chưa thành pipeline | chưa sinh answer | grounded answer + verifier |
| TRAKE | có thể làm nhưng tốn thao tác | có DP, second pass hẹp | dual-lane + local aligner |
| Audit/replay | artifact rời | log chưa đủ provenance | artifact version hóa từng round |

Kết luận: V2 không nên mô phỏng một lệnh search trực tiếp. Nó phải tự động hóa
chính **vòng lặp suy giảm bất định** đã giúp direct retrieval tìm ra đáp án khi
top đầu ban đầu sai.

---

## 3. Nguyên tắc thiết kế mới

### 3.1. Nói gì thì tìm đúng thứ đó

Mỗi phần có nghĩa của truy vấn được chuyển thành một `QueryAtom`. Atom là đơn
vị nhỏ nhất được truy xuất, chấm điểm và kiểm tra.

| Nội dung người dùng | Evidence modality | Retriever được phép |
|---|---|---|
| đối tượng, màu sắc, cảnh, người, bố cục | `VISUAL` | frame/clip/shot FAISS, visual caption |
| object thuộc class detector | `OBJECT` | object detection, có thể thêm visual fallback |
| hành động/chuyển động | `ACTION` | clip FAISS, track; frame chỉ hỗ trợ |
| số lượng object nhìn thấy | `OBJECT_COUNT` | detection group theo frame |
| quan hệ không gian | `SPATIAL` | detection boxes hoặc VLM reranker |
| chữ xuất hiện trên hình/bảng/biển | `VISIBLE_TEXT` | chỉ OCR |
| lời nói/nghe thấy/người nói | `SPOKEN_TEXT` | chỉ ASR |
| nội dung âm thanh không phải lời nói | `AUDIO_EVENT` | audio embedding trong tương lai |
| title/description/keywords của video | `METADATA` | PostgreSQL metadata search |
| trước/sau/sau đó/cùng lúc | `TEMPORAL` | aligner, không phải retriever |

`VISIBLE_TEXT` và `SPOKEN_TEXT` chỉ được kích hoạt bởi tín hiệu ngôn ngữ rõ ràng
hoặc cấu trúc câu hỏi yêu cầu đọc/nghe. Việc một noun có thể xuất hiện trong OCR
hay ASR không phải lý do để gọi hai nguồn này.

### 3.2. Phân biệt retrieval evidence và verification context

- **Retrieval evidence** được phép ảnh hưởng thứ hạng candidate.
- **Verification context** chỉ được tải sau khi đã có candidate để giải thích,
  kiểm tra hoặc trả lời; nó không được hồi tố cộng điểm nếu modality không được
  query yêu cầu.

Ví dụ query “ảnh có hình phi thuyền” chỉ retrieval bằng visual/object. Sau khi
có top candidate, hệ thống có thể tải OCR/ASR quanh đó cho QA hoặc người review,
nhưng từ “phi thuyền” trong các text này không làm candidate tăng hạng.

### 3.3. V2 không phụ thuộc agent nhưng phải agent-ready

Release V2 đầu tiên dùng một structured parser call có schema, deterministic
policy compiler, deterministic router và deterministic ranking. Đây không phải
agent loop: model không tự chọn tool, không quan sát kết quả rồi hành động tiếp.

Các interface sau phải được định nghĩa từ đầu để sau này có thể cắm agent:

```text
QueryInterpreter -> RetrievalPlan
PlanReviewer     -> PlanReview
CandidateReviewer -> CandidateReview
AnswerGenerator -> GroundedAnswer
```

Implementation mặc định của `PlanReviewer` và `CandidateReviewer` là no-op hoặc
rule-based. Khi bật agent sau này, input/output contract không đổi.

Agent phù hợp cho parsing phức tạp, expansion, critique và visual review. Agent
không được:

- tự gọi index tổng hợp;
- thay modality sau khi policy compiler đã khóa;
- tự đặt weight trực tiếp trong scoring;
- bỏ hard constraint;
- sửa output BTC hoặc canonical ID;
- truy cập SQL/FAISS tùy ý ngoài tool contract.

Mọi output agent phải là structured JSON, qua schema validation và policy
validation trước khi thực thi.

---

## 4. Contract đề xuất

### 4.1. QueryAtom

```yaml
QueryAtom:
  atom_id: A1
  event_id: E1 | null
  original_text: "có hình phi thuyền"
  retrieval_texts:
    - "a visible spacecraft in the scene"
    - "spaceship shown in the image"
  modality: VISUAL
  subtype: OBJECT_APPEARANCE
  role: REQUIRED          # REQUIRED | SUPPORTING
  operator: MUST          # MUST | SHOULD | MUST_NOT
  allowed_retrievers:
    - frame_search
    - clip_search
  forbidden_retrievers:
    - ocr_search
    - asr_search
  granularity: MOMENT     # FRAME | MOMENT | SHOT | VIDEO
  links: []               # spatial/count/temporal links to other atoms
  confidence: 0.96
```

Các enum chính:

```text
Modality = VISUAL | OBJECT | ACTION | OBJECT_COUNT | SPATIAL |
           VISIBLE_TEXT | SPOKEN_TEXT | AUDIO_EVENT | METADATA | TEMPORAL

Role     = REQUIRED | SUPPORTING
Operator = MUST | SHOULD | MUST_NOT
```

### 4.2. RetrievalPlan

```yaml
RetrievalPlan:
  query_id: query-p1-x
  task: KIS
  atoms: [A1, A2, ...]
  tool_calls:
    - call_id: TC1
      atom_id: A1
      event_id: null
      retriever: frame_search
      query: "a visible spacecraft in the scene"
      top_k: 300
      profile: visual_object
  hard_constraints: [A1]
  temporal_constraints: []
  review_budget:
    shortlist_size: 20
    visual_candidates: 5
    max_retry_rounds: 2
```

### 4.3. SearchHit và evidence mới

Mỗi hit phải thêm tối thiểu:

```text
atom_id, event_id, modality, role, operator,
retriever_name, retriever_version, index_version,
raw_score, normalized_score, rank, match_type,
video_id, frame/clip/shot ID, start_ms, end_ms
```

`match_type` gồm `semantic`, `exact`, `phrase`, `fuzzy`, `detector`, `track`,
`metadata`. Text evidence nên lưu matched span và confidence phục vụ debug,
nhưng API công khai chỉ trả phần cần thiết.

### 4.4. MomentBand

Không đưa từng hit hoặc từng frame lân cận thành candidate cuối độc lập.
`MomentBand` là đơn vị trung tâm của KIS/QA và là node đầu vào của TRAKE:

```yaml
MomentBand:
  band_id: MB1
  video_id: L22_V021
  start_ms: 612000
  end_ms: 648000
  peak_ms: 631200
  event_id: E1
  atom_evidence:
    A1: [H17, H22]
    A2: [H31]
  coverage_status:
    A1: PASS
    A2: UNKNOWN
  source_families: [legacy_clip_b32, object_detector_v3]
  canonical_frame_candidates: []
```

Band được tạo từ extent thật của frame/clip/shot/ASR rồi hợp nhất theo overlap
và query relation. Chỉ khi chuẩn bị review/output mới lấy representative frame.
Nhờ đó local search không mất cảnh chuyển tiếp và không biến nhiều frame gần
nhau thành nhiều bằng chứng độc lập.

### 4.5. SearchSessionState và hypothesis ledger

Mỗi query có state bền vững để retry không lặp lại cùng sai lầm:

```text
query/plan/version, current_round, budget_used,
hypotheses[video_id + band_id],
atom coverage, positive evidence, contradictions, unknowns,
rejection scope/reason, prompt variants đã chạy,
retriever/index family đã chạy, review verdict và retry history
```

Một hypothesis có thể ở `ACTIVE`, `CONFIRMED`, `REJECTED` hoặc `SUPERSEDED`.
Reject luôn có phạm vi `PLAN`, `VIDEO`, `MOMENT_BAND`, `FRAME` hoặc
`SUBMISSION_ROW`. Ví dụ query 24 phải loại moment sai trong video đúng, không
được loại toàn bộ video; query 18/19 cần loại video top đầu rồi mở shortlist.

### 4.6. ConstraintDecision

Mọi filter cứng trả một decision có cấu trúc:

```yaml
ConstraintDecision:
  constraint_id: C4
  status: PASS             # PASS | FAIL | UNKNOWN
  scope: MOMENT_BAND       # PLAN | VIDEO | MOMENT_BAND | FRAME | SUBMISSION_ROW
  reason_code: TEMPORAL_ORDER_VIOLATION
  evidence_ids: [H12, H19]
  confidence: 1.0
```

`UNKNOWN` không đồng nghĩa `FAIL`. Thiếu detection, thiếu ASR/OCR, CLIP score
thấp hoặc verifier không chắc đều phải giữ là `UNKNOWN` trừ khi có bằng chứng
đủ tin cậy về mâu thuẫn.

### 4.7. Candidate score breakdown

Mỗi candidate cần giữ:

- điểm theo từng atom;
- required coverage;
- supporting bonus;
- hard constraint status;
- negative evidence;
- temporal consistency;
- nguồn nào bị policy loại bỏ;
- lý do rerank/review.

Không thể tối ưu pipeline nếu chỉ còn một `fusion_score` không giải thích được.

---

## 5. Kiến trúc mục tiêu

```text
User query
  |
  v
Round Manifest + Task classifier + typed Query Interpreter
  |
  v
Query Atomizer
  |  (one structured call or deterministic parser; schema validated)
  v
Modality Policy Compiler  <--- deterministic firewall
  |
  v
Task-aware Retrieval Planner
  |---- Visual router -> frame / clip / shot FAISS
  |---- Object router -> detection / track / spatial
  |---- Text router   -> OCR only / ASR only
  |---- Metadata router
  |
  |-----------------------------|
  v                             v
Global moment lane        Video shortlist lane
  |                             |
  |                       filtered local search
  |-----------------------------|
                |
                v
Per-retriever calibration + family-aware deduplication
  |
  v
Atom-aware Candidate Builder
  |
  v
Constraint-aware Fusion + coverage ranking
  |
  +---- KIS Moment Selector
  +---- QA Evidence Answerer
  +---- TRAKE Video Shortlist + Temporal Aligner
  |
  v
Deterministic verification + optional reviewer hook
  |
  v
Canonical frame resolver -> score-aware top-100 builder
  -> BTC validator -> submission/*.csv + frame_links + ZIP
```

### 5.1. Hai retrieval lane để tránh cascade error

Corpus chỉ có 873 video nhưng mỗi video khá dài. V2 nên chạy hai lane rồi lấy
union thay vì chỉ coarse-to-fine cứng:

1. **Global moment lane:** search trực tiếp frame/clip/shot toàn corpus. Lane này
   cứu video đúng khi video-level aggregation yếu.
2. **Video shortlist lane:** gom evidence theo atom để chấm cả 873 video, lấy
   shortlist rồi search dày trong từng video. Lane này tìm chính xác moment và
   đặc biệt quan trọng cho TRAKE.

Candidate từ hai lane được dedup theo canonical video/time. Không candidate nào
được cộng bonus chỉ vì xuất hiện ở cả frame, clip và shot nếu chúng cùng một
visual embedding family.

### 5.2. Pipeline state machine

```text
PARSED -> POLICY_VALIDATED -> RETRIEVED -> BANDS_BUILT -> RANKED
                                      |                    |
                                      |                    v
                                      +<-- REFINED <-- DIAGNOSED
                                                           |
                                                           v
TASK_RESOLVED -> VERIFIED -> DENSE_SWEEP -> FORMATTED -> VALIDATED
```

`DIAGNOSED` xác định thất bại nằm ở plan, video, moment hay atom; `REFINED` chỉ
tạo tool call mới cho phần yếu. Mỗi state persist artifact JSON có version.
Retry không chạy lại toàn pipeline vô điều kiện và không quên hypothesis đã bị
loại. `DENSE_SWEEP` chỉ chạy khi đã có hypothesis video/moment đủ mạnh. Thiết kế
này cho phép thêm reviewer/agent sau mà không làm thay đổi state cốt lõi.

---

## 6. Các giai đoạn xử lý

### Giai đoạn A - Task classification và atomization

1. Xác định `KIS`, `QA/VQA` hoặc `TRAKE`.
2. Tách câu thành event; với KIS/QA thông thường có một event ngầm.
3. Tách event thành các atom độc lập.
4. Xác định modality từ cue và nghĩa câu, không dựa đơn thuần vào noun.
5. Gắn `REQUIRED`, `SUPPORTING`, `MUST_NOT`.
6. Biên dịch temporal/spatial/count relation thành link giữa atom.
7. Lưu nguyên văn tiếng Việt cho audit.

Policy cue tối thiểu:

| Cue | Kết quả |
|---|---|
| “có/nhìn thấy/hình/cảnh/mặc/cầm/đứng” | visual/action/object |
| “chữ/dòng chữ/bảng ghi/logo có chữ/biển ghi” | OCR |
| “nói/đọc/nghe/lời thoại/phát thanh” | ASR |
| “sau đó/trước khi/cuối cùng” | temporal relation |
| “không có/không xuất hiện” | negative atom, không suy luận từ missing detection |

Nếu confidence modality thấp, ambiguity resolver chỉ được chọn trong tập phương
án policy đã tạo sẵn. Không được mặc định gọi mọi nguồn. MVP dùng rule-based
resolver; agent critic có thể thay implementation sau. Với noun thuần thị giác,
default là `VISUAL`.

### Giai đoạn B - Chuẩn hóa query theo modality

#### Visual

- Giữ bản tiếng Việt để log.
- Tạo một bản tiếng Anh trung thành cho CLIP.
- Có thể tạo 2-3 paraphrase ngắn, mỗi bản tập trung một thuộc tính quan trọng.
- Không thêm text/OCR cue vào prompt visual.
- Không viết prompt quá dài như toàn bộ đoạn truyện cho từng event.

Encoder hiện tại là multilingual CLIP nên tiếng Việt có thể chạy, nhưng English
normalization vẫn nên là profile mặc định và phải được kiểm chứng bằng ablation
trên benchmark nội bộ. Không coi việc dịch là chân lý cố định.

#### OCR và ASR

- Giữ tiếng Việt gốc.
- Chuẩn hóa Unicode, khoảng trắng, dấu câu và biến thể không dấu.
- Tạo exact phrase, token AND, fuzzy và semantic variant theo cascade.
- Không dịch sang tiếng Anh trừ khi query thực sự nói về text tiếng Anh.

#### Object

- Thay allow-list 30 class bằng catalog sinh từ 601 `classid` trong DB, có
  multilingual alias và version. Chỉ class thực sự có detection mới được
  planner chọn.
- Dùng `aic_hcm2026_text_object_active` để shortlist frame nhanh theo exact
  `objects.raw`; tải PostgreSQL boxes để count/spatial verification.
- Map sang detector class bằng dictionary có version.
- Nếu object ngoài ontology, dùng visual retrieval; không ép vào class gần nhất.
- Count và spatial relation phải giữ ở atom/link, không nhét vào query string.

### Giai đoạn C - Task-aware retrieval routing

#### Visual router

| Loại atom | Retriever chính | Retriever hỗ trợ |
|---|---|---|
| object/attribute nhỏ, trạng thái tĩnh | frame | clip |
| hành động hoặc thay đổi vài giây | clip | frame/track |
| cảnh rộng, địa điểm, bố cục | shot | clip |
| chuỗi hành động trong một event | clip | shot |
| fine-grained moment trong video đã shortlist | frame có filter video | clip |

Không gọi cả ba index mặc định. Chỉ dùng retriever hỗ trợ nếu expected gain đủ
lớn hoặc pass đầu thiếu recall.

Shot dài bất thường phải bị giới hạn. Shot trên một ngưỡng cấu hình, khởi điểm
30 giây, không được dùng nguyên extent để tạo candidate mà phải chia theo clip
window hoặc local frame peaks. Dữ liệu có shot dài tới 1.171 giây; giữ nguyên sẽ
kéo gần cả video vào một region giả.

#### Visual model strategy

Ba index hiện hành đều khai báo `clip-ViT-B-32` và đang đồng bộ, nên giữ chúng
làm baseline lane. Không xóa index cũ trước khi model mới chứng minh tốt hơn.

V2 dùng `EmbeddingIndexRegistry` để có thể chạy song song nhiều family:

```text
legacy_clip_b32       -> baseline, rollback và ensemble diversity
multilingual_image_v2 -> image-text model mạnh hơn cho frame/scene
temporal_video_v1     -> video-text model có temporal modeling cho action
```

Ứng viên đầu tiên nên benchmark là SigLIP 2 ở image lane vì hỗ trợ đa ngôn ngữ,
retrieval và localization. Action lane cần một video-text encoder có mô hình hóa
thời gian, không chỉ mean-pool image embeddings. Model được chọn bằng Recall@100,
MRR và latency trên query nội bộ, không chọn chỉ theo benchmark công bố.

Trong giai đoạn chuyển đổi, English translation là query chính cho CLIP hiện
tại; query tiếng Việt chạy như lane phụ có weight được tune. Với model đa ngôn
ngữ mới, giữ cả hai và ablation trước khi bỏ bản dịch.

#### Text router

OCR và ASR là hai tool độc lập cả về alias, query builder và score calibration.

OCR cascade:

1. exact/phrase;
2. token AND;
3. fuzzy có ngưỡng;
4. semantic text embedding nếu exact recall thấp.

ASR cascade tương tự nhưng cho phép time segment dài và normalization từ nói.
Kết quả phải resolve lại qua PostgreSQL trước khi dùng.

#### Caption

Caption sinh từ ảnh là **visual-derived evidence**, không phải OCR/ASR. Nếu khôi
phục caption retrieval, tạo alias riêng `aic_visual_caption_active`, gắn target
frame/clip/shot và model provenance. Caption chỉ được dùng cho atom visual với
trọng số hỗ trợ; không đưa trở lại index text hợp nhất.

#### Metadata

Metadata chỉ được gọi khi câu hỏi nêu title, mô tả chương trình, nguồn video,
địa danh/nhân vật mà metadata có giá trị. Metadata không được tự động cộng điểm
cho mọi visual query vì description có thể kể nội dung không xuất hiện tại
moment đang tìm.

### Giai đoạn D - Per-retriever calibration và dedup

Raw score giữa FAISS, BM25, fuzzy OCR và detector không cùng thang đo. Mỗi
retriever cần calibration độc lập bằng labeled validation set:

```text
p_relevance = calibrator[retriever, profile](raw_score, rank, query_features)
```

Có thể bắt đầu bằng RRF trong từng atom, sau đó dùng isotonic regression hoặc
logistic calibration. Không trộn raw score chưa hiệu chuẩn.

Dedup theo canonical entity và temporal overlap:

- cùng video + cùng atom + frame gần nhau: giữ hit tốt nhất, lưu provenance còn lại;
- frame/clip/shot cùng region không được tính như ba bằng chứng độc lập;
- giới hạn số hit mỗi video trong pass đầu để tăng diversity;
- giữ một quota nhỏ cho long-tail videos.

### Giai đoạn E - Atom-aware Candidate Builder

Thay vì merge mọi hit gần nhau, builder thực hiện:

1. Group theo `(video_id, event_id, atom_id)`.
2. Tạo proposal region từ temporal extent của hit gốc.
3. Mở rộng cửa sổ theo loại retriever, ví dụ frame +/- 1.5 giây, clip dùng
   extent thật, ASR dùng segment thật.
4. Join proposal của các atom chỉ khi chúng cần đồng thời và có temporal
   overlap/gap phù hợp.
5. Giữ riêng evidence map `atom_id -> hits`.
6. Candidate ID ổn định dựa trên video, event và time bucket, không dựa vào thứ
   tự enumerate.

Với atom hỗ trợ, evidence ở gần có thể tăng confidence nhẹ. Với atom bắt buộc,
candidate không có hit mới chỉ là `UNKNOWN`, vì retriever có thể false-negative.
Candidate bị loại cứng khi có contradiction đáng tin cậy; còn thiếu coverage bị
penalty lớn và không được promote vào nhóm precision cho tới khi verification
xác nhận.

### Giai đoạn F - Constraint-aware fusion

#### Score theo atom

```text
S_atom(c, a) = max_h [calibrated_score(h)]
               + small_consensus_bonus(distinct_retriever_families)
```

Frame, clip và shot đều thuộc cùng `visual_embedding_family`; không được tính là
ba family độc lập.

#### Coverage bắt buộc

```text
coverage_required = count(required atoms matched) / count(required atoms)
```

Nếu một atom là hard `MUST`, candidate `FAIL` bị loại; candidate `UNKNOWN` không
được coi là đã đạt coverage. Nếu parser chưa chắc và atom ở soft mode, dùng
geometric mean để một atom rất yếu không bị che bởi nhiều hit phụ:

```text
S_required = geometric_mean(epsilon + S_atom for required atoms)
S_support  = weighted_sum(S_atom for supporting atoms)
```

#### Điểm cuối

```text
S_candidate = W_required * S_required
            + W_support  * S_support
            + W_temporal * temporal_consistency
            + W_review   * review_score
            - missing_penalty
            - negative_penalty
            - duplicate_penalty
```

Weight lấy từ task/profile đã version hóa và được tune offline; tuyệt đối không
tăng weight chỉ vì source có trả hit. MVP chọn profile bằng task/rule. Agent mở
rộng sau này chỉ được chọn profile trong allow-list, không được sinh weight tự do.

Hard negative chỉ được kết luận khi có positive evidence về điều bị cấm. Không
được kết luận “không có X” chỉ vì detector không thấy X.

### Giai đoạn G - HardConstraintEngine tập trung

Không để logic lọc cứng nằm rải ở planner, fusion, task handler và formatter.
Một engine duy nhất chạy ở bốn gate:

| Gate | Kiểm tra | Ví dụ `FAIL` chắc chắn |
|---|---|---|
| Plan | tool/modality/alias hợp lệ | gọi ASR cho atom visual-only; query concrete ES index |
| Candidate | ID, extent, contradiction | video/frame không tồn tại; positive evidence về điều `MUST_NOT` |
| Task | semantics theo task | thứ tự thời gian sai; TRAKE khác video hoặc thiếu event |
| Submission | format/canonical output | frame không official; trùng dòng; sai số cột/arity |

Reason code tối thiểu:

```text
FORBIDDEN_MODALITY, FORBIDDEN_INDEX, INVALID_ENTITY,
POSITIVE_CONSTRAINT_CONTRADICTION, TEMPORAL_ORDER_VIOLATION,
WRONG_OBJECT_COUNT, NEGATIVE_CONSTRAINT_CONFIRMED,
INVALID_OFFICIAL_FRAME, TRAKE_EVENT_ARITY_MISMATCH,
UNGROUNDED_QA_ANSWER, DUPLICATE_SUBMISSION_ROW
```

Quy tắc quan trọng:

1. Chỉ `FAIL` mới bị loại; `UNKNOWN` vẫn có thể đi rescue lane nhưng không vào
   precision tier nếu thiếu atom bắt buộc.
2. Detector không thấy object, OCR/ASR thiếu dữ liệu, score thấp hoặc VLM không
   chắc đều là `UNKNOWN`, không phải bằng chứng phủ định.
3. Count chỉ `PASS` khi đủ detection đồng thời trong cùng frame. Chỉ `FAIL` khi
   nguồn có coverage/độ tin cậy đủ để khẳng định count sai; còn lại `UNKNOWN`.
4. Reject phải đúng scope. Một frame sai không loại cả band; một band sai không
   loại cả video; plan vi phạm modality thì chặn trước khi gọi tool.
5. Verification context không được nâng thành retrieval evidence. Muốn promote
   một evidence mới phải tạo atom/tool call hợp lệ, chạy lại policy gate và ghi
   provenance trong round kế tiếp.

Engine thuần deterministic và version hóa. Agent/reviewer chỉ cung cấp evidence
hoặc verdict có confidence; không được tự tạo hard rule hay bỏ qua decision.

---

## 7. Pipeline riêng cho KIS

### 7.1. Retrieval

KIS có một format output nhưng hai execution profile nội bộ:

- `KIS_MOMENT`: các atom đồng thời trong một cảnh/moment;
- `KIS_SEQUENCE`: mô tả có “sau đó”, “tiếp đến”, “kết thúc”, nhiều hành động
  hoặc scene transition. Profile này tách event và align nhẹ trong cùng video,
  nhưng cuối cùng vẫn xuất một cặp `video_id,frame_idx` mỗi dòng.

Luồng retrieval:

1. Atomize mô tả thành scene, object, action, count, spatial, OCR hoặc ASR.
2. Với `KIS_SEQUENCE`, giữ order và scene boundary thay vì nhét cả đoạn vào một
   prompt CLIP dài.
3. Retrieval high-recall theo đúng modality, ưu tiên 2-4 prompt visual ngắn cho
   các atom phân biệt mạnh.
4. Tạo `MomentBand`, shortlist khoảng 200-500 band tùy độ khó và chấm video
   hypothesis theo số atom/event khác nhau được phủ.
5. Rank theo required coverage trước, score sau; ban đầu cap 2-3 band/video để
   giữ khả năng cứu video long-tail.
6. Với top video hypothesis, chạy local search theo `video_id`, giữ extent thật
   và tăng mật độ quanh event lân cận.

### 7.2. Precise moment selection

Với top bands:

1. tải các official/extracted frame trong band và neighbor shot;
2. rerank frame bằng prompt atom ngắn, không dùng toàn bộ mô tả dài;
3. ưu tiên frame hiển thị rõ required object/action;
4. với sequence, kiểm tra storyboard trước/sau và chọn frame đại diện theo event
   được BTC yêu cầu hoặc theo frame có tổng coverage tốt nhất;
5. map extracted frame về official frame gần nhất;
6. bảo đảm `frame_idx`, `img_url`, `display_frame_id` cùng một record.

Thay `KISHandler` anchor-only bằng query-aware moment selector. Anchor vẫn là
fallback tốt khi không có frame reranker.

### 7.3. Review, chẩn đoán và refinement

- Top 1-10 ưu tiên precision và review kỹ.
- Tạo montage cho tối đa 5 candidate đầu, mỗi candidate có center frame và 1-2
  neighbor frame.
- MVP cho người vận hành review qua montage và ghi `match`, `partial`,
  `mismatch`, atom nào thiếu cùng confidence. Reviewer agent có thể thay đúng
  interface này ở phase sau.
- Reviewer không tự tạo candidate. Verdict đi vào `DIAGNOSED`, từ đó
  orchestrator deterministic mới lập retry call hợp lệ.

Các trigger retry rõ ràng:

```text
top 5 đều mismatch/partial thấp
required atom coverage thấp
score margin hoặc agreement giữa prompt thấp
top results bị một video/source family chi phối
đúng video nhưng sai moment hoặc sequence thiếu một event
ảnh review 404/extracted-only và chưa map được official frame
```

Action retry được chọn theo chẩn đoán, không đổi modality tùy tiện:

- tách/viết lại prompt của atom yếu;
- đổi granularity visual hợp lệ, tăng retrieval depth hoặc rescue quota;
- local search và neighbor sweep trong video đúng;
- loại đúng frame/band/video đã có contradiction;
- mở rộng video shortlist nếu hypothesis hiện tại đều thất bại;
- dùng OCR/ASR exact chỉ khi atom gốc thật sự yêu cầu đọc/nghe.

Review mặc định xem 5 hypothesis đầu. Nếu cả 5 sai, hệ thống bắt buộc refinement
thay vì dừng; sau retry chỉ review candidate mới hoặc band đã thay đổi, tổng tối
đa theo budget. Đây là điểm khác quan trọng so với “xem top 5 một lần”.

### 7.4. Xây danh sách KIS theo mục tiêu cuộc thi

Do tối đa 100 dòng và thứ hạng đầu quan trọng, allocation phụ thuộc loại bất
định đã đo được, không dùng một diversity rule cố định:

| Trạng thái | Chiến lược dòng submission |
|---|---|
| chưa chắc video | chia quota cho nhiều video hypothesis, mỗi video ít band |
| chắc video, chưa chắc moment | tập trung nhiều band/time bucket trong video đó |
| chắc video và band | dense official-frame sweep trong band, xếp center-out theo coverage |
| sequence có event biên chưa chắc | rải frame quanh transition/event có uncertainty cao |
| nhiều hypothesis gần điểm | precision tier trước, sau đó MMR có quota theo lane/video |

Builder thực hiện:

1. Top 1-10 chỉ lấy hypothesis có required coverage cao và không có `UNKNOWN`
   nghiêm trọng chưa review.
2. Giữ score theo band; frame gần nhau trong cùng band là uncertainty coverage,
   không được giả làm nhiều independent retrieval hit.
3. Khi confidence video cao, cho phép nhiều frame cùng video/moment. Đây chính
   là cách các đáp án direct retrieval đúng được tạo sau khi video đã xác nhận.
4. Khi confidence video thấp, cap frame/video và dành rescue quota cho lane độc
   lập để tránh correlated failure.
5. Map sang official frame, xếp frame theo atom visibility/transition relevance,
   rồi dedup đúng cặp `(video_id, frame_idx)`.
6. Không cố điền đủ 100 nếu phần đuôi chỉ gồm `FAIL` hoặc row không hợp lệ; vẫn
   có thể dùng `UNKNOWN` đã qua rescue trong diversity tier nếu còn budget.

Các tham số này thuộc `RoundManifest.scoring_profile`. Nếu BTC công bố công thức
điểm của vòng, profile có thể đổi mà không sửa retrieval core.

---

## 8. Pipeline riêng cho QA/VQA

QA cần hai bài toán nối tiếp: tìm evidence rồi sinh answer. Không thể dùng nhánh
KIS hiện tại làm output cuối.

### 8.1. Question analysis

Ngoài retrieval atom, tạo `AnswerSpec`:

```yaml
answer_type: NUMBER | TEXT | COLOR | PERSON | PLACE | BOOLEAN | OTHER
answer_source: VISUAL | OCR | ASR | MIXED
normalization: short_vietnamese
max_length: 100
```

Ví dụ:

- “Có bao nhiêu người?” -> object count/visual evidence, không dùng ASR.
- “Dòng chữ trên bảng là gì?” -> OCR required, frame visual là context.
- “Người đàn ông nói gì?” -> ASR required, visual giúp xác định speaker/time.
- “Chiếc áo màu gì?” -> visual only.

### 8.2. Evidence retrieval và answer generation

1. Retrieve candidate regions theo `answer_source` và locator atoms.
2. Tạo evidence pack gồm top frame, neighbor frames, OCR/ASR chỉ khi được phép,
   object boxes và timestamp.
3. Grounded QA model/VLM trả answer, confidence và evidence IDs bằng một
   structured call; đây không phải agent loop.
4. Chạy answer aggregation trên nhiều candidate/evidence pack.
5. Verify answer với deterministic rule nếu có thể: count, exact OCR, exact ASR,
   boolean constraint.
6. Nếu không đủ evidence, trả `uncertain`; không suy đoán.

### 8.3. Ranking đáp án

Rank theo tích của ba thành phần:

```text
P(answer correct) ~= retrieval_confidence
                  * evidence_grounding_confidence
                  * answer_consensus
```

Kết quả BTC là `video_id,frame_idx,answer`, answer tối đa 100 ký tự. Frame phải
là moment chứa bằng chứng trả lời, không chỉ là frame locator ban đầu.

Vì rules vừa mô tả so khớp ngữ nghĩa vừa lưu ý answer được so dưới dạng chuỗi,
V2 dùng answer canonicalization bảo thủ: câu trả lời ngắn, đúng loại, bỏ diễn
giải thừa. Chỉ sinh một số rất nhỏ variant như `5` và `Năm người` khi
`RoundManifest` cho phép và validation chứng minh variant có lợi; không tiêu hết
100 dòng bằng các paraphrase tương đương.

---

## 9. Pipeline riêng cho TRAKE

### 9.1. Event decomposition

- Mỗi event có atom riêng và ID ổn định `E1..En`.
- Atom luôn mang `event_id`; OCR/ASR/object constraint không được để event null.
- Temporal relation được lưu tách biệt: order, min/max gap, overlap, same shot
  nếu query nêu rõ.
- Tách một event quá dài thành sub-atom đồng thời, không biến chúng thành các
  event nối tiếp giả.

### 9.2. Hai tầng truy xuất

#### Tầng 1 - Video shortlist

1. Retrieve high-recall riêng cho từng event đúng modality.
2. Tính event coverage theo video.
3. Giữ video có nhiều event khác nhau, không dựa vào tổng số hit.
4. Dành quota rescue cho video thiếu một event nhưng các event còn lại rất mạnh.

#### Tầng 2 - Search within shortlisted video

1. Chạy lại từng event với filter `video_id`.
2. Tăng temporal density quanh candidate của event lân cận.
3. Dùng đúng retriever của atom còn thiếu; không mặc định clip-only.
4. Nếu không có feasible video, mở rộng shortlist toàn corpus cho atom hiếm nhất
   thay vì khóa vào video đã có ở pass đầu.

### 9.3. Temporal alignment

Giữ `TrakeTemporalAligner` và DP/beam hiện có vì boundary của module hợp lý,
nhưng input phải là candidate đã atom-aware. Sequence score cần gồm:

- event relevance;
- coverage đủ mọi event;
- order và gap constraint;
- overlap penalty;
- transition plausibility;
- first-occurrence policy khi đề yêu cầu lần đầu;
- duplicate/near-identical sequence penalty.

Top sequences được visual reviewer xem dưới dạng storyboard theo đúng thứ tự
event. Reviewer chấm từng event và toàn chuỗi; nếu một event sai, retry chỉ event
đó trong cùng video trước khi mở toàn corpus.

### 9.4. Xây top-100 sequence

- Mỗi dòng phải có đúng N frame, cùng video và đúng thứ tự tăng thời gian.
- Tạo variant sequence có kiểm soát từ các local maxima quanh từng event; không
  lấy Cartesian product mọi frame lân cận.
- Ưu tiên thay một event có uncertainty cao, giữ các event chắc chắn cố định.
- Dedup sequence tuyệt đối và near-duplicate theo time tolerance.
- Chia diversity budget giữa nhiều video trước khi tạo quá nhiều jitter trong
  một video.
- Submission validator loại toàn bộ dòng sai arity/order trước khi ghi CSV.

---

## 10. Extension agent review, triển khai sau

Phần này không nằm trong MVP của `OnlinePipelineV2`. MVP vẫn phải đạt benchmark
và sinh submission hoàn chỉnh khi mọi implementation agent là no-op. Extension
chỉ được bật qua feature flag sau shadow evaluation.

### 10.1. Các agent và ranh giới

| Agent | Nhiệm vụ | Không được làm |
|---|---|---|
| Intent Agent | tách task/event/atom | gọi DB hoặc chọn weight |
| Plan Critic | tìm atom thiếu, modality sai | thêm modality ngoài policy |
| Retrieval Critic | đọc score breakdown, đề xuất retry | tự tạo hit/candidate |
| Visual Review Agent | xem montage/storyboard top candidate | search toàn corpus |
| QA Answer Agent | trả answer từ evidence pack | dùng kiến thức ngoài evidence |
| Submission Validator | kiểm format/canonical frame | thay đổi ranking ngữ nghĩa |

Không cần nhiều agent chạy tự do. Một orchestrator deterministic gọi đúng agent
ở đúng gate sẽ rẻ hơn, nhanh hơn và dễ debug.

### 10.2. Review loop có giới hạn

```text
Round 0: retrieve + rank
  -> nếu confidence/margin tốt: dừng
  -> nếu top candidates mơ hồ: review tối đa 5
  -> critic xác định atom yếu hoặc contradiction
Round 1: retry atom yếu với query expansion / retriever hỗ trợ hợp lệ
  -> merge + rerank + review lại phần thay đổi
Round 2: rescue search hoặc dừng ở uncertain
```

Budget mặc định đề xuất:

| Task | Visual review | Retry | Candidate review |
|---|---:|---:|---:|
| KIS | 1 batch | tối đa 2 | top 5 |
| QA | 1-3 evidence packs | tối đa 1 | top 3 |
| TRAKE | 1 storyboard batch | tối đa 2 | top 3 sequences |

Review phải chạy sau deterministic filters và trước final top-10. Không dùng
VLM để xem hàng trăm ảnh.

---

## 11. Thiết kế index và dữ liệu

### 11.1. Tách API theo evidence semantics

Không bắt buộc tách physical Elasticsearch index. Filtered alias hiện tại đủ
an toàn nếu registry cấm concrete index và startup check xác minh filter. Public
retriever/API đề xuất:

```text
aic_visual_frame_active       -> FAISS frame version hiện hành
aic_visual_clip_active        -> FAISS clip version hiện hành
aic_visual_shot_active        -> FAISS shot version hiện hành
aic_hcm2026_text_ocr_active        -> filter source_type=ocr
aic_hcm2026_text_transcript_active -> filter source_type=transcript
aic_hcm2026_text_object_active     -> filter source_type=object
aic_hcm2026_text_metadata_active   -> filter source_type=video_metadata
```

Caption không có dữ liệu nên chưa đăng ký retriever. Khi bổ sung caption, nó là
visual-derived evidence và chỉ được mở qua filtered alias riêng.

Registry phải mô tả rõ `evidence_modality`, alias filter kỳ vọng, model/index
version và canonical resolver. Readiness check fail nếu alias mất filter hoặc
trỏ nhầm index.

Không dùng một `content` index chung làm public retriever. Nếu cần một physical
index vì vận hành, query builder bắt buộc filter `source_type` và application
chỉ expose các wrapper typed như `ocr_search`, `asr_search`.

### 11.2. PostgreSQL là nguồn chân lý

- Resolve mọi FAISS/ES hit về PostgreSQL.
- FAISS ID phải đi qua embedding record đúng `index_version`, model và pooling.
- Output luôn dùng official `frame_idx`.
- Track thuộc shot; không suy luận liên tục qua shot boundary nếu chưa nối track.
- Object count group theo frame, không theo shot.

### 11.3. Version và reproducibility

Mỗi run log:

```text
query_parser_version, policy_version, plan_version,
embedding_model, faiss_index_version,
ocr_index_alias+concrete_version, asr_index_alias+concrete_version,
fusion_profile_version, verifier_model_version, random_seed
```

Không log secret hoặc toàn bộ nội dung nhạy cảm.

---

## 12. Thay đổi đề xuất theo module

| Module hiện tại | Thay đổi |
|---|---|
| module mới `round_manifest/` | số dòng, số lần nộp, scoring profile và task rules theo vòng |
| `contracts/models.py` | thêm `QueryAtom`, modality/role/operator, `RetrievalPlan`, score breakdown |
| `intent_extractor/prompts.py` | prompt typed atom và cue OCR/ASR rõ ràng |
| `intent_extractor/extractor.py` | schema validation, confidence và deterministic fallback |
| module mới `retrieval_policy/` | policy compiler, allow/deny retriever, plan validator |
| module mới `constraint_engine/` | tri-state hard filter, reject scope/reason và bốn validation gate |
| module mới `search_session/` | hypothesis ledger, retry budget, artifact replay và state machine |
| module mới `moment_band/` | tạo/merge temporal extent, atom coverage và representative-frame candidates |
| `fast_path/runner.py` | thay bằng deterministic task-aware router, không gọi ba visual index mặc định |
| `query_planner/` | planner chỉ chọn call trong allow-list của từng atom; giữ top-k adaptive |
| `retrieval_tools/text.py` | wrapper typed, exact/fuzzy/semantic cascade, source filter bắt buộc |
| `retrieval_tools/object.py` | sửa count theo frame, trả class/count/box provenance |
| `retrieval/visual_retrieval.py` | cache embedding, batch query, optional video/time filter |
| `aggregator.py` | candidate builder theo atom và temporal relation |
| `Fusion/` | thay bằng calibrated atom-aware fusion; bỏ dynamic source-presence boost |
| `KIS/kis_handler.py` | `KIS_MOMENT`/`KIS_SEQUENCE`, query-aware frame reranker và uncertainty allocation |
| `vqa/handler.py` | nối retrieval evidence -> VLM answer -> grounding verifier |
| `trake/` | video shortlist + within-video retrieval trước aligner |
| `verification/` | bật selective VLM review có budget; deterministic check trước |
| `api/pipeline.py` | orchestrator mới, stage diagnostics, retry loop |
| `api/models.py` | output QA thực, debug plan tùy feature flag |
| module mới `submission/` | top-100 builder, CSV/TXT, thư mục `submission/`, ZIP và validator |
| module mới `embedding_registry/` | nhiều index family, version, shadow index và rollback |

`dynamic_weight.py` không nên gọi LLM để sinh weight runtime. Nếu cần agent chọn
chiến lược, nó chỉ chọn một profile đã được benchmark và version hóa.

---

## 13. Lộ trình migration

Triển khai dưới entry point `/api/v2/query` và package riêng; không sửa dần
orchestrator cũ đến trạng thái nửa V1/nửa V2. V1 chỉ dùng làm baseline và
fallback cho tới khi V2 qua shadow evaluation.

### Phase 0 - Baseline và observability

- Tạo query set có ground truth cho cả ba task.
- Log intent, tool calls, hit provenance, latency và score breakdown hiện tại.
- Đo recall@K, MRR/nDCG và modality violation rate.
- Chạy `check_faiss_db_sync` và kiểm alias ES trước benchmark.
- Tạo `RoundManifest` từ rules/announcement thay vì hard-code số lần nộp.

**Exit:** có baseline có thể tái hiện và biết query nào gọi sai modality.

### Phase 1 - Modality firewall

- Thêm `QueryAtom`, policy compiler và plan validator.
- Thêm `HardConstraintEngine` với `PASS/FAIL/UNKNOWN`, reject reason và scope.
- Giữ retrieval/fusion cũ phía sau adapter để giảm blast radius.
- Chặn OCR/ASR khi query không có cue tương ứng.
- Thêm contract tests và audit log forbidden calls.

**Exit:** modality violation rate bằng 0 trên bộ test bắt buộc.

### Phase 2 - Atom-aware aggregation và fusion

- Carry `atom_id` từ tool call đến candidate.
- Tạo `MomentBand` và giữ temporal extent tới task resolver.
- Persist `SearchSessionState` và hypothesis ledger qua từng retrieval round.
- Calibrate score từng retriever.
- Fusion theo required coverage, bỏ source-presence boost.
- Sửa object count và negative constraint.

**Exit:** Recall@100 không giảm đáng kể, MRR/top-5 precision tăng trên validation.

### Phase 3 - Task handlers hoàn chỉnh

- KIS precise moment/sequence reranker, deterministic refinement và top-100
  allocation theo uncertainty.
- QA sinh answer có grounding.
- TRAKE video shortlist và modality-aware second pass.
- Submission formatter cho ba format BTC.

**Exit:** end-to-end tạo đúng CSV/TXT, QA có answer, TRAKE đủ event cùng video.

### Phase 4 - Agent review

- Thêm montage/storyboard builder.
- Bật Visual Review Agent theo gate và budget.
- Thêm retrieval critic + retry loop tối đa hai round.
- A/B test có/không review.

**Exit:** review tăng top-5 precision đủ lớn so với latency/cost và không làm giảm
tính tái hiện của retrieval nền.

### Phase 5 - Tuning và rollout

- Tune top-k/profile trên held-out set.
- Build shadow index cho image-text model mới và temporal video model.
- Chỉ promote index nếu tăng retrieval metric sau khi tính cả latency/GPU RAM.
- Shadow-run pipeline mới cùng pipeline cũ.
- So sánh theo task, modality và loại query.
- Canary rollout, giữ feature flag để fallback.

---

## 14. Kiểm thử bắt buộc

### 14.1. Modality firewall tests

| Query | Tool bắt buộc | Tool bị cấm |
|---|---|---|
| “ảnh có hình phi thuyền” | visual/object | OCR, ASR |
| “dòng chữ PHI THUYỀN trên bảng” | OCR | ASR; visual chỉ context |
| “người đàn ông nói phi thuyền” | ASR | OCR |
| “người mặc áo đỏ cạnh bảng ghi London Zoo” | visual + OCR | ASR |
| “sau đó phát thanh viên nói X” | visual event + ASR event | OCR |

Test phải assert tool-call plan, không chỉ assert final result.

### 14.2. Contract và data tests

- Mọi hit có `atom_id`, modality, event ID đúng.
- `MomentBand` giữ extent clip/shot thật; không bị thay bằng midpoint trước local
  search và task resolution.
- `SearchSessionState` replay cho kết quả giống nhau và không chạy lại tool call
  đã hoàn tất khi retry một atom khác.
- ES OCR wrapper không thể query ASR alias và ngược lại.
- Concrete unified index query bị policy từ chối; startup check xác minh alias
  có đúng `source_type` filter.
- FAISS hit resolve đúng mapping version.
- Extracted frame map về một official frame nhất quán.
- Object count được tính trong cùng frame.
- Object catalog phản ánh class có thật trong DB, không giới hạn cứng 30 class.
- Shot outlier không tạo candidate region dài hàng phút.
- TRAKE event evidence không bị event null.
- Submission test đúng 100 dòng tối đa, không header, QA escaping, TRAKE arity,
  thư mục `submission/` và ZIP layout.

### 14.3. Hard-constraint tests

- Forbidden tool/index trả `FAIL` ở plan gate trước mọi I/O.
- Thiếu detector/ASR/OCR trả `UNKNOWN`, không tự chuyển thành `FAIL`.
- Positive contradiction trả đúng reason code và chỉ loại đúng scope.
- Một moment sai không loại toàn video; một frame 404 không loại band nếu còn
  official frame resolve được.
- Object count không cộng detection qua nhiều frame; evidence coverage yếu trả
  `UNKNOWN`.
- KIS sequence, TRAKE arity/order, QA grounding và submission canonical frame
  đều qua cùng engine nhưng dùng rule set theo task.
- Verification evidence không được cộng retrieval score nếu chưa qua policy gate
  của round mới.

### 14.4. Ranking và refinement tests

- Hit sai modality không đóng góp score.
- Ba visual granularity cùng concept không được tính thành ba independent votes.
- Candidate đủ hai required atom xếp trên candidate rất mạnh ở một atom nhưng
  thiếu atom còn lại.
- Negative constraint dùng evidence content/class thực.
- Diversity không loại toàn bộ candidate tốt trong cùng video.
- Khi top 5 sai video, retry mở video shortlist thay vì chỉ rerank top 5 cũ.
- Khi đúng video nhưng sai moment, reject `MOMENT_BAND` không reject `VIDEO`.
- Khi video/band confidence cao, top-100 cho phép dense official-frame sweep;
  khi video confidence thấp, allocation tăng cross-video diversity.
- `KIS_SEQUENCE` xếp band đủ đúng thứ tự trên band có các noun đúng nhưng thứ tự
  hành động sai.

### 14.5. Metrics

| Nhóm | Metrics |
|---|---|
| Retrieval | Recall@10/50/100, MRR, nDCG@10, video recall |
| Modality | forbidden-call rate, wrong-modality contribution rate |
| Moment | frame accuracy, temporal IoU, distance tới ground-truth frame |
| QA | answer accuracy/EM/F1, grounded-answer rate, abstention quality |
| TRAKE | event recall, feasible-video recall, full-sequence recall, order accuracy |
| Refinement | hypothesis survival, retry success, wrong-scope rejection rate, rounds/query |
| Filter | false-reject rate, `UNKNOWN` resolution rate, reason-code distribution |
| System | p50/p95 latency, ES/FAISS calls, VLM calls, cost/query, cache hit rate |

Đánh giá riêng theo visual-only, OCR-only, ASR-only, mixed và temporal query.
Không tune và báo cáo trên cùng một query set.

### 14.6. Gold set và hard-negative set

Mỗi query đã giải cần được lưu thành evaluation record, không chỉ CSV cuối:

```text
query text, task, typed atoms,
relevant video IDs, accepted time/frame intervals,
QA accepted answers hoặc TRAKE event intervals,
confirmed false-positive videos, modality failure labels
```

Tạo riêng hard negatives có cùng noun nhưng khác modality, ví dụ noun xuất hiện
trong ASR/OCR nhưng không xuất hiện trong ảnh. Đây là bộ test trực tiếp cho lỗi
người dùng đang gặp. Chia train/dev/test theo query và nguồn video để tránh tune
vào cùng chương trình/video. Public leaderboard chỉ là tín hiệu tham khảo vì
không phủ toàn bộ đáp án.

Top-5 review thủ công về sau phải ghi lại atom thiếu/sai để trở thành label cho
reranker và calibration, thay vì chỉ sửa thứ hạng của một lần chạy.

---

## 15. Tối ưu hiệu năng

1. Cache text embedding theo `(model_version, normalized_query)`.
2. Batch nhiều visual atom trong một lần encode và FAISS search.
3. Chạy song song các retriever của **các atom độc lập**, không chạy tool bị cấm.
4. Dùng adaptive top-k: atom hiếm/exact cần K nhỏ, visual broad cần K lớn.
5. Pass 1 có per-video cap; pass 2 search trong shortlist video.
6. Batch resolve FAISS IDs và frame metadata để tránh N+1 PostgreSQL query.
7. Chỉ dựng montage và gọi VLM sau deterministic rerank.
8. Cache evidence pack theo `(video_id, start_ms, end_ms, modalities)`.
9. Đặt timeout/circuit breaker riêng cho từng retriever; timeout một nguồn không
   được âm thầm đổi sang modality khác.
10. Warm FAISS indexes và model khi service khởi động; expose readiness riêng.

Mục tiêu latency ban đầu nên được đo từ baseline trước khi chốt SLA. Thiết kế
hai pass thường giảm tổng số hit cần fusion dù pass đầu có recall cao.

### 15.1. Budget khởi điểm để benchmark

Đây là giá trị bootstrap, không phải hằng số cuối cùng:

| Task/profile | Global retrieval | Video shortlist | Local search | Final |
|---|---:|---:|---:|---:|
| KIS object/detail | frame tối đa 1.000, clip 300 khi cần | 40 video | 50 moment/video | 100 dòng |
| KIS action | clip tối đa 1.000, frame 300 | 50 video | 50 moment/video | 100 dòng |
| KIS scene | shot 500, clip 300 | 40 video | 30 moment/video | 100 dòng |
| OCR/ASR exact | lấy theo chunk 200 đến khi score/margin ổn định | 50 video | +/- neighbor window | 100 dòng |
| QA | retriever theo `AnswerSpec`, tối đa 1.000 | 30 video | 20 moment/video | tối đa 100 dòng |
| TRAKE | tối đa 1.000/event | 80 video + rescue quota | 30 moment/event/video | 100 sequence |

Adaptive stop dựa trên unique-video gain, score drop và required-atom coverage.
Không dừng chỉ vì đã đủ 100 raw hit; 100 dòng submission cần được chọn từ một
candidate pool lớn hơn đáng kể.

---

## 16. Ví dụ end-to-end

### 16.1. Query visual “phi thuyền”

Input:

```text
Tìm cảnh có hình phi thuyền bay phía trên thành phố.
```

Atoms:

```text
A1 VISUAL/ACTION REQUIRED: spacecraft flying
A2 VISUAL/SCENE  REQUIRED: above a city
```

Plan hợp lệ:

```text
A1 -> clip_search, optional frame_search second pass
A2 -> shot_search
OCR -> forbidden
ASR -> forbidden
```

Fusion chỉ xếp cao candidate có coverage A1 và A2 trong cùng temporal region.
Chữ “phi thuyền” trên phụ đề hoặc lời nói không tạo score.

### 16.2. Query mixed modality

Input:

```text
Một đàn sư tử ở khu nuôi dưỡng, phía trước có bảng London Zoo.
```

Atoms:

```text
A1 VISUAL REQUIRED: group of lions in an enclosure
A2 VISIBLE_TEXT REQUIRED: "London Zoo"
A3 VISUAL SUPPORTING: information board in front
```

Plan gọi clip/frame cho A1/A3 và OCR cho A2. OCR “London Zoo” chỉ có ý nghĩa khi
time overlap với region có sư tử; một bản tin nói “London Zoo” không được gọi ASR
và không tăng hạng.

### 16.3. TRAKE

Input dạng “E1 bản đồ có bốn công trình, sau đó E2 đập nhìn từ trên cao, tiếp
đến E3 cận cảnh con đập dưới mưa” tạo ba event visual. Tầng 1 shortlist video
theo coverage E1-E3; tầng 2 tìm dense candidate từng event trong các video đó;
aligner chỉ tạo sequence cùng video và tăng thời gian. Storyboard reviewer xem
ba cột E1-E3 thay vì xem từng ảnh rời.

---

## 17. Tiêu chí hoàn thành

Pipeline được coi là hoàn chỉnh khi đáp ứng đồng thời:

- Không có forbidden modality call trên test suite.
- MVP chạy end-to-end khi agent/reviewer extension bị tắt.
- Mỗi score cuối truy ngược được tới atom và evidence.
- OCR/ASR wrapper dùng đúng filtered alias; concrete unified index không nằm
  trong registry public.
- KIS trả tối đa 100 `video_id,frame_idx`, top 5 có thể visual review.
- QA trả `video_id,frame_idx,answer` có evidence grounding.
- TRAKE trả đủ một frame/event, cùng video và đúng thứ tự.
- Object count tính theo frame.
- Hard filter dùng `PASS/FAIL/UNKNOWN`, có reason code và đúng reject scope.
- KIS top-5 sai vẫn có refinement path; đúng video nhưng sai moment không bị
  loại cả video.
- Top-100 allocation đổi theo video/moment uncertainty và có thể replay.
- Retry có budget, không vòng lặp agent vô hạn.
- Output dùng official frame và đúng format BTC, đồng thời sinh file link ảnh.
- ZIP có đúng thư mục `submission/`; rule thay đổi theo round qua manifest.
- Có benchmark held-out, latency/cost report và feature flag rollback.

---

## 18. Đánh giá thiết kế sau phản biện

### 18.1. Mức độ khớp với cách truy xuất đã giải được bài

Sau lần sửa này, thiết kế đã bao phủ các cơ chế cốt lõi của direct retrieval:
multi-prompt đúng modality, global high-recall, video hypothesis, local evidence,
neighbor expansion, retry theo atom, visual review, official-frame sweep và
ranking-aware output. Điểm còn lại mang tính triển khai/benchmark, không còn là
lỗ hổng kiến trúc chính.

| Trục đánh giá | Trạng thái | Nhận xét |
|---|---|---|
| Đúng modality | tốt | firewall + typed alias + promotion rule chặn OCR/ASR lẫn visual |
| Recall ban đầu | tốt về thiết kế | dual-lane và adaptive top-k giữ được ưu điểm direct search |
| Thoát top-5 sai | tốt về thiết kế | có diagnosis, scoped rejection và deterministic refinement |
| Định vị đúng moment | tốt hơn rõ rệt | `MomentBand`, local search và dense sweep thay midpoint/anchor sớm |
| Multi-event KIS | hợp lý | `KIS_SEQUENCE` tái dùng temporal alignment nhẹ |
| Hard filter | đầy đủ về contract | tri-state tránh false reject, bốn gate tránh rule rải rác |
| Top-100 | phù hợp dữ liệu thực tế | allocation theo uncertainty thay diversity cố định |
| QA | đã nối grounded provider, chưa chứng minh | cần model config và gold eval cho legend/count |
| TRAKE | đã có missing-event second pass | cần benchmark video shortlist recall và full-sequence recall |
| Agent readiness | tốt | state/artifact/tool contract đủ để cắm critic/reviewer sau |
| Runtime readiness | artifacts ready, chưa promotion | 397 tests xanh, health ready; còn thiếu full live benchmark, VLM review và p95 shadow |

### 18.2. Rủi ro triển khai còn phải giải

1. FAISS hiện không có filter `video_id` tự nhiên. Local search phải benchmark
   hai phương án: reconstruct vector rồi lọc tập ID của video, hoặc build cache/
   sub-index theo shortlisted videos. Không được giả định API filter đã tồn tại.
2. Score calibration cần label đủ theo từng retriever/profile. Trước khi đủ
   label, dùng family-aware RRF làm baseline và log raw score; không học weight
   trên vài query đã biết đáp án.
3. `UNKNOWN` có thể giữ quá nhiều candidate. Cần budget rescue, deadline và
   completeness threshold theo tier, nhưng không được giải quyết bằng hard
   reject thiếu bằng chứng.
4. Dense sweep có thể lãng phí nhiều dòng nếu confidence video bị overestimate.
   Phải calibrate video/band confidence và giữ một rescue quota tối thiểu.
5. KIS sequence và TRAKE cần annotation interval, không chỉ một ground-truth
   frame, nếu không moment/transition model sẽ bị tune sai.
6. VLM review có thể không ổn định. MVP phải ghi verdict/provenance, khóa
   temperature/profile và luôn chạy được khi reviewer tắt.
7. Latency hai lane + refinement có thể cao. Cần cache, batch, adaptive stop và
   per-query budget; không giảm latency bằng cách phá modality firewall.

### 18.3. Kết luận go/no-go

Thiết kế hiện **đủ chặt để bắt đầu triển khai V2 theo phase**, nhưng **chưa đủ
bằng chứng để thay V1 ngay**. Go cho Phase 0-2; no-go cho production cutover cho
tới khi qua bốn gate: modality violation bằng 0, không giảm Recall@100, tăng
top-5 precision trên held-out set, và submission validator đạt 100% fixture.
Agent không nằm trong các gate này.

---

## 19. Quyết định kiến trúc khuyến nghị

Thứ tự đầu tư hiệu quả nhất là:

1. Tạo `OnlinePipelineV2` song song, giữ nguyên data layer và V1 fallback.
2. Làm modality firewall và provenance theo atom. Đây là sửa lỗi trực tiếp
   cho hiện tượng “mô tả hình nhưng khớp OCR/ASR”.
3. Đưa mọi hard rule vào `HardConstraintEngine` tri-state; không hard-code filter
   rải rác trong retriever/handler.
4. Viết lại aggregation/fusion quanh `MomentBand`, required coverage và
   `SearchSessionState`; không cố tune các weight đều hiện tại.
5. Hoàn thiện deterministic refinement, KIS sequence và uncertainty-aware
   top-100 trước khi thêm visual agent.
6. Hoàn thiện QA vì nhánh đang chạy chưa thực sự trả lời câu hỏi.
7. Nâng TRAKE thành dual-lane + within-video search, giữ DP aligner hiện có.
8. Mở full 601-class object catalog và sửa count theo frame.
9. Benchmark index ảnh/video thế hệ mới dưới dạng shadow indexes.
10. Chỉ thêm visual agent review sau khi deterministic retrieval ổn định.

Agent review là lớp tăng precision, không phải cách che một retrieval plan thiếu
kiểu dữ liệu. Khi modality và provenance đúng ngay từ đầu, agent mới có thể phản
biện đúng candidate, retry đúng atom và giúp hệ thống tốt lên qua benchmark thay
vì chỉ tạo cảm giác thông minh ở runtime.

---

## 20. Nguồn đã đối chiếu

- [Thể lệ AIC26](https://sotuyenaic.oj.io.vn/rules/), truy cập 22/08/2026.
- [FAQ AIC26](https://sotuyenaic.oj.io.vn/faq/), truy cập 22/08/2026.
- [Thông báo AIC26](https://sotuyenaic.oj.io.vn/announcements/), truy cập
  22/08/2026.
- `C:/Users/ADMIN/Downloads/db_description.md` và schema/code runtime trong repo.
- `.aic-agent-work/` và các output direct retrieval trong workspace: prompt,
  global result, evidence window, review manifest và submission đã xác nhận.
- PostgreSQL, Elasticsearch aliases/mappings và FAISS sync report của runtime.
- [SigLIP 2 paper](https://arxiv.org/abs/2502.14786), chỉ dùng làm ứng viên
  benchmark; chưa phải quyết định model production.
- [CLIP4Clip paper](https://arxiv.org/abs/2104.08860), tham khảo nhu cầu temporal
  modeling cho video-text retrieval.
