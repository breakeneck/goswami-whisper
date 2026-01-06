-- Migration script to add timing columns to existing tables
-- Run this if you already have the tables created

-- Add index_duration_seconds to uploads table
ALTER TABLE uploads
ADD COLUMN IF NOT EXISTS index_duration_seconds FLOAT;

-- Add duration_seconds to transcribes table
ALTER TABLE transcribes
ADD COLUMN IF NOT EXISTS duration_seconds FLOAT;

-- Add duration_seconds to contents table
ALTER TABLE contents
ADD COLUMN IF NOT EXISTS duration_seconds FLOAT;

