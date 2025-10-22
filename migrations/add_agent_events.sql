-- Migration: Add agent_events table for cross-agent communication
-- Description: Enables pub/sub event system for inter-agent coordination
-- Created: 2025-10-22

-- Create agent_events table
CREATE TABLE IF NOT EXISTS agent_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    event_type TEXT NOT NULL,  -- Backend validates via AgentEventType enum
    source_agent TEXT NOT NULL,
    target_agent TEXT,  -- NULL = broadcast to all agents
    payload JSONB NOT NULL DEFAULT '{}',
    priority TEXT NOT NULL DEFAULT 'normal',  -- Backend validates (low/normal/high/critical)
    status TEXT NOT NULL DEFAULT 'pending',  -- Backend validates (pending/consumed/expired)
    consumed_by TEXT,
    consumed_at TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_events_org_id ON agent_events(org_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_status ON agent_events(status);
CREATE INDEX IF NOT EXISTS idx_agent_events_event_type ON agent_events(event_type);
CREATE INDEX IF NOT EXISTS idx_agent_events_target_agent ON agent_events(target_agent);
CREATE INDEX IF NOT EXISTS idx_agent_events_created_at ON agent_events(created_at);

-- Composite index for common query pattern (org + status + created_at)
CREATE INDEX IF NOT EXISTS idx_agent_events_org_status_created
    ON agent_events(org_id, status, created_at);

-- Enable Row Level Security
ALTER TABLE agent_events ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only access events for their organization
CREATE POLICY agent_events_org_isolation ON agent_events
    FOR ALL
    USING (org_id IN (
        SELECT org_id FROM profiles WHERE id = auth.uid()
    ));

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON agent_events TO authenticated;

-- Add comment
COMMENT ON TABLE agent_events IS 'Event bus for inter-agent communication and coordination';
COMMENT ON COLUMN agent_events.event_type IS 'Type of event (campaign_completed, task_created, etc.)';
COMMENT ON COLUMN agent_events.source_agent IS 'Agent that published the event';
COMMENT ON COLUMN agent_events.target_agent IS 'Specific agent to receive event (NULL = broadcast)';
COMMENT ON COLUMN agent_events.payload IS 'Event-specific data payload';
COMMENT ON COLUMN agent_events.priority IS 'Event priority level';
COMMENT ON COLUMN agent_events.status IS 'Event processing status';
COMMENT ON COLUMN agent_events.consumed_by IS 'Agent that consumed/processed this event';
