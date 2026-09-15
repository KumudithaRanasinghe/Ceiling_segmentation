-- =============================================================================
-- Ceiling AI Database Initialization Script for MySQL / MySQL Workbench
-- =============================================================================

CREATE DATABASE IF NOT EXISTS `ceiling_ai` 
  CHARACTER SET utf8mb4 
  COLLATE utf8mb4_unicode_ci;

USE `ceiling_ai`;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS `users` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY COMMENT 'UUID v4',
    `email` VARCHAR(255) NOT NULL UNIQUE COMMENT 'Unique lowercased email',
    `username` VARCHAR(64) NOT NULL UNIQUE,
    `full_name` VARCHAR(255) NULL,
    `hashed_password` VARCHAR(60) NOT NULL COMMENT 'Bcrypt hash',
    `refresh_token_hash` TEXT NULL,
    `roles` TEXT NOT NULL COMMENT 'JSON array of roles e.g. ["user", "admin"]',
    `is_active` BOOLEAN NOT NULL DEFAULT TRUE,
    `is_superuser` BOOLEAN NOT NULL DEFAULT FALSE,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX `idx_users_email` (`email`),
    INDEX `idx_users_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Segmentation Jobs Table
CREATE TABLE IF NOT EXISTS `segmentation_jobs` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY COMMENT 'UUID v4 job_id',
    `user_id` VARCHAR(36) NOT NULL,
    `status` VARCHAR(16) NOT NULL DEFAULT 'success' COMMENT 'pending|running|success|failed',
    `error_detail` TEXT NULL,
    `image_filename` VARCHAR(255) NULL,
    `image_size_bytes` INT NULL,
    `content_type` VARCHAR(64) NULL,
    `pixels_per_meter` FLOAT NULL,
    `full_result_json` LONGTEXT NULL COMMENT 'Serialized SegmentationResponse JSON blob',
    `inference_time_ms` FLOAT NULL,
    `model_version` VARCHAR(64) NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `completed_at` DATETIME NULL,
    `expires_at` DATETIME NULL,
    INDEX `idx_jobs_user_id` (`user_id`),
    INDEX `idx_jobs_created_at` (`created_at`),
    INDEX `idx_jobs_expires_at` (`expires_at`),
    INDEX `idx_jobs_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. Audit Logs Table
CREATE TABLE IF NOT EXISTS `audit_logs` (
    `id` VARCHAR(36) NOT NULL PRIMARY KEY,
    `actor_id` VARCHAR(36) NOT NULL,
    `actor_email` VARCHAR(255) NOT NULL,
    `action` VARCHAR(64) NOT NULL,
    `resource_type` VARCHAR(64) NOT NULL,
    `resource_id` VARCHAR(64) NULL,
    `details_json` TEXT NULL,
    `ip_address` VARCHAR(45) NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX `idx_audit_actor` (`actor_id`),
    INDEX `idx_audit_action` (`action`),
    INDEX `idx_audit_resource` (`resource_type`),
    INDEX `idx_audit_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. System Settings Table
CREATE TABLE IF NOT EXISTS `system_settings` (
    `key` VARCHAR(64) NOT NULL PRIMARY KEY,
    `value_json` LONGTEXT NOT NULL,
    `category` VARCHAR(64) NOT NULL DEFAULT 'general',
    `description` VARCHAR(255) NULL,
    `updated_by` VARCHAR(36) NULL,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX `idx_settings_category` (`category`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
