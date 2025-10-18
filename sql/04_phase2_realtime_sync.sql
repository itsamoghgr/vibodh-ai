-- =====================================================
-- Vibodh AI - Phase 2: Real-time Sync & Memory
-- =====================================================
-- This migration adds tables and functions for:
-- 1. Real-time Slack event tracking
-- 2. Enhanced AI memory with importance and sources
-- 3. Nightly summarization support
-- =====================================================

-- Step 1: Create events table for real-time event tracking
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('slack', 'notion', 'drive', 'jira', 'hubspot')),
    actor_id TEXT,  -- User ID from the source system
    payload JSONB NOT NULL,  -- Full event payload
    happened_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add foreign key constraint if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'events_org_id_fkey'
    ) THEN
        ALTER TABLE events
        ADD CONSTRAINT events_org_id_fkey
        FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE;
    END IF;
END $$;

-- Create indexes for events table
CREATE INDEX IF NOT EXISTS events_org_id_idx ON events(org_id);
CREATE INDEX IF NOT EXISTS events_source_idx ON events(source);
CREATE INDEX IF NOT EXISTS events_happened_at_idx ON events(happened_at DESC);
CREATE INDEX IF NOT EXISTS events_org_source_time_idx ON events(org_id, source, happened_at DESC);

-- Enable RLS on events table
ALTER TABLE events ENABLE ROW LEVEL SECURITY;

-- RLS policies for events
CREATE POLICY "Users can view events for their organization"
ON events FOR SELECT
USING (
    org_id IN (
        SELECT org_id FROM profiles WHERE id = auth.uid()
    )
);

CREATE POLICY "Service can insert events"
ON events FOR INSERT
WITH CHECK (true);  -- Allow service role to insert

-- Step 2: Update ai_memory table schema
-- Add importance and source_refs columns if they don't exist
ALTER TABLE ai_memory
    ADD COLUMN IF NOT EXISTS importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    ADD COLUMN IF NOT EXISTS source_refs JSONB DEFAULT '[]'::jsonb;

-- Add index for importance sorting
CREATE INDEX IF NOT EXISTS ai_memory_importance_idx ON ai_memory(importance DESC);
CREATE INDEX IF NOT EXISTS ai_memory_org_importance_idx ON ai_memory(org_id, importance DESC, created_at DESC);

-- Enable RLS on ai_memory if not already enabled
ALTER TABLE ai_memory ENABLE ROW LEVEL SECURITY;

-- RLS policies for ai_memory (drop existing if they exist, then recreate)
DROP POLICY IF EXISTS "Users can view ai_memory for their organization" ON ai_memory;
CREATE POLICY "Users can view ai_memory for their organization"
ON ai_memory FOR SELECT
USING (
    org_id IN (
        SELECT org_id FROM profiles WHERE id = auth.uid()
    )
);

DROP POLICY IF EXISTS "Service can insert ai_memory" ON ai_memory;
CREATE POLICY "Service can insert ai_memory"
ON ai_memory FOR INSERT
WITH CHECK (true);  -- Allow service role to insert

DROP POLICY IF EXISTS "Service can update ai_memory" ON ai_memory;
CREATE POLICY "Service can update ai_memory"
ON ai_memory FOR UPDATE
USING (true);  -- Allow service role to update

-- Step 3: Create function to get recent events for summarization
CREATE OR REPLACE FUNCTION get_recent_events(
    filter_org_id UUID,
    filter_source TEXT DEFAULT 'slack',
    hours_back INT DEFAULT 24
)
RETURNS TABLE (
    id UUID,
    source TEXT,
    actor_id TEXT,
    payload JSONB,
    happened_at TIMESTAMPTZ
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.source,
        e.actor_id,
        e.payload,
        e.happened_at
    FROM events e
    WHERE e.org_id = filter_org_id
        AND e.source = filter_source
        AND e.happened_at >= NOW() - (hours_back || ' hours')::INTERVAL
    ORDER BY e.happened_at DESC;
END;
$$;

-- Step 4: Create function to get summarization-ready documents
CREATE OR REPLACE FUNCTION get_documents_for_summary(
    filter_org_id UUID,
    filter_source TEXT DEFAULT 'slack',
    hours_back INT DEFAULT 24,
    min_docs INT DEFAULT 5
)
RETURNS TABLE (
    channel_id TEXT,
    channel_name TEXT,
    message_count BIGINT,
    recent_content TEXT[]
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.channel_id,
        d.channel_name,
        COUNT(d.id) AS message_count,
        ARRAY_AGG(d.content ORDER BY d.created_at DESC) AS recent_content
    FROM documents d
    WHERE d.org_id = filter_org_id
        AND d.source_type = filter_source
        AND d.created_at >= NOW() - (hours_back || ' hours')::INTERVAL
    GROUP BY d.channel_id, d.channel_name
    HAVING COUNT(d.id) >= min_docs
    ORDER BY message_count DESC;
END;
$$;

-- Step 5: Create function to track webhook status
CREATE OR REPLACE FUNCTION update_connection_webhook_status(
    connection_uuid UUID,
    last_event_time TIMESTAMPTZ
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE connections
    SET
        last_sync_at = last_event_time,
        metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
            'last_webhook_event', last_event_time,
            'webhook_active', true
        ),
        updated_at = NOW()
    WHERE id = connection_uuid;
END;
$$;

-- Step 6: Grant permissions
GRANT SELECT ON events TO authenticated;
GRANT INSERT ON events TO service_role;  -- For real-time sync
GRANT EXECUTE ON FUNCTION get_recent_events TO authenticated;
GRANT EXECUTE ON FUNCTION get_documents_for_summary TO authenticated;
GRANT EXECUTE ON FUNCTION update_connection_webhook_status TO service_role;

-- Step 7: Create materialized view for daily activity summary (optional)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_activity_summary AS
SELECT
    org_id,
    source,
    DATE(happened_at) AS activity_date,
    COUNT(*) AS event_count,
    COUNT(DISTINCT actor_id) AS unique_users,
    MIN(happened_at) AS first_event,
    MAX(happened_at) AS last_event
FROM events
GROUP BY org_id, source, DATE(happened_at);

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS daily_activity_summary_unique_idx
ON daily_activity_summary(org_id, source, activity_date);

-- Create function to refresh the materialized view
CREATE OR REPLACE FUNCTION refresh_daily_activity()
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_activity_summary;
END;
$$;

-- =====================================================
-- Post-Migration Instructions
-- =====================================================
--
-- 1. Run this SQL in Supabase SQL Editor
--
-- 2. Test events table:
--    SELECT * FROM events WHERE org_id = 'your-org-id';
--
-- 3. Test ai_memory updates:
--    SELECT * FROM ai_memory WHERE importance > 0.7 ORDER BY created_at DESC LIMIT 10;
--
-- 4. Test recent events function:
--    SELECT * FROM get_recent_events('your-org-id', 'slack', 24);
--
-- 5. Configure Slack Events API:
--    - Request URL: https://your-domain.com/api/slack/events
--    - Subscribe to: message.channels, message.groups
--    - Enable Event Subscriptions in Slack App settings
--
-- 6. Verify webhook is receiving events:
--    - Send a test message in Slack
--    - Check events table: SELECT COUNT(*) FROM events WHERE created_at > NOW() - INTERVAL '5 minutes';
--
-- =====================================================
