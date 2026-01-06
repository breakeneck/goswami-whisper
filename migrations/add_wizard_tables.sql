-- Migration script to add new tables for the wizard workflow
-- Run this if the tables are not automatically created

-- Uploads table (main table for uploaded files)
CREATE TABLE IF NOT EXISTS uploads (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    original_filename VARCHAR(500),
    file_path VARCHAR(1000),
    source_url TEXT,
    file_size BIGINT,
    duration_seconds FLOAT,
    is_indexed BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at),
    INDEX idx_is_indexed (is_indexed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Transcribes table (transcribed text with provider info)
CREATE TABLE IF NOT EXISTS transcribes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    upload_id INT NOT NULL,
    text LONGTEXT,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE,
    INDEX idx_upload_id (upload_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Contents table (formatted text with provider info)
CREATE TABLE IF NOT EXISTS contents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    transcribe_id INT NOT NULL,
    text LONGTEXT,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    error_message TEXT,
    is_indexed BOOLEAN DEFAULT FALSE,
    duration_seconds FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transcribe_id) REFERENCES transcribes(id) ON DELETE CASCADE,
    INDEX idx_transcribe_id (transcribe_id),
    INDEX idx_status (status),
    INDEX idx_is_indexed (is_indexed),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add some useful views
CREATE OR REPLACE VIEW v_upload_summary AS
SELECT
    u.id,
    u.filename,
    u.original_filename,
    u.duration_seconds,
    u.is_indexed,
    u.created_at,
    COUNT(DISTINCT t.id) as transcribe_count,
    COUNT(DISTINCT c.id) as content_count,
    MAX(CASE WHEN c.is_indexed THEN 1 ELSE 0 END) as has_indexed_content
FROM uploads u
LEFT JOIN transcribes t ON t.upload_id = u.id
LEFT JOIN contents c ON c.transcribe_id = t.id
GROUP BY u.id;

