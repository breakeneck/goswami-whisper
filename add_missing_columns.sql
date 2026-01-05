-- Run this to add missing columns to the transcriptions table
-- Execute: mysql -u goswami -p goswami_whisper < add_missing_columns.sql
-- Or via Docker: docker exec -i goswami-whisper-mysql-1 mysql -u goswami -pgoswamipassword goswami_whisper < add_missing_columns.sql

ALTER TABLE transcriptions
    ADD COLUMN IF NOT EXISTS progress FLOAT DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS duration_seconds FLOAT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS current_position FLOAT DEFAULT 0.0;

