-- Indexes to accelerate object, visual, and track searches in AIC HCMC
CREATE INDEX IF NOT EXISTS idx_class_id_name_lower ON classid (LOWER(class_name));
CREATE INDEX IF NOT EXISTS idx_class_id_id_lower ON classid (LOWER(class_id));

CREATE INDEX IF NOT EXISTS idx_object_detections_class_frame ON objectdetection (class_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_object_detections_frame_id ON objectdetection (frame_id);

CREATE INDEX IF NOT EXISTS idx_object_tracks_class_shot ON objecttrack (class_id, shot_id);
CREATE INDEX IF NOT EXISTS idx_object_tracks_shot_id ON objecttrack (shot_id);

CREATE INDEX IF NOT EXISTS idx_frames_video_shot_time ON frame (video_id, shot_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_frames_video_time ON frame (video_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_frame_frame_id ON frame (frame_id);

-- Critical FAISS resolution indexes for instant top-1000 lookup
CREATE INDEX IF NOT EXISTS idx_frame_embedding_faiss ON frameembeddingrecord (faiss_id, index_version, model_name);
CREATE INDEX IF NOT EXISTS idx_clip_embedding_faiss ON clipembeddingrecord (faiss_id, index_version, model_name);
CREATE INDEX IF NOT EXISTS idx_shot_embedding_faiss ON shotembeddingrecord (faiss_id, index_version, model_name);

CREATE INDEX IF NOT EXISTS idx_frame_embedding_frame_id ON frameembeddingrecord (frame_id);
CREATE INDEX IF NOT EXISTS idx_clip_embedding_clip_id ON clipembeddingrecord (clip_id);
CREATE INDEX IF NOT EXISTS idx_shot_embedding_shot_id ON shotembeddingrecord (shot_id);

CREATE INDEX IF NOT EXISTS idx_clipwindow_clip_id ON clipwindow (clip_id);
CREATE INDEX IF NOT EXISTS idx_clipwindow_shot_id ON clipwindow (shot_id);
CREATE INDEX IF NOT EXISTS idx_shot_shot_id ON shot (shot_id);
