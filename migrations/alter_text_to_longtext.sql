-- Migration script to change text columns from TEXT to LONGTEXT
-- Run this if you get "Data too long for column 'text'" errors
-- TEXT in MySQL is limited to 65,535 characters
-- LONGTEXT supports up to 4GB

-- Alter text column in transcribes table
ALTER TABLE transcribes
MODIFY COLUMN text LONGTEXT;

-- Alter text column in contents table (formatted text)
ALTER TABLE contents
MODIFY COLUMN text LONGTEXT;

