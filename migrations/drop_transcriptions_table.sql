-- Migration script to drop the deprecated transcriptions table
-- Run this after migrating all data to the new uploads/transcribes/contents structure
-- WARNING: This will permanently delete all data in the transcriptions table

-- Drop the transcriptions table
DROP TABLE IF EXISTS transcriptions;

