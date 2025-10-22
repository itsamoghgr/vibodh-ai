-- Migration: Add integration_health_checks table for monitoring integration health
-- Description: Tracks health status of external integrations (Slack, ClickUp, Email, etc.)
-- Created: 2025-10-22

-- Create integration_health_checks table
CREATE TABLE IF NOT EXISTS integration_health_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    integration TEXT NOT NULL,  -- Backend validates via IntegrationType enum
    status TEXT NOT NULL,  -- Backend validates via IntegrationStatus enum
    message TEXT,
    response_time_ms INT,  -- API response time in milliseconds
    metadata JSONB DEFAULT '{}',  -- Additional health check metadata
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_integration_health_org_id ON integration_health_checks(org_id);
CREATE INDEX IF NOT EXISTS idx_integration_health_integration ON integration_health_checks(integration);
CREATE INDEX IF NOT EXISTS idx_integration_health_status ON integration_health_checks(status);
CREATE INDEX IF NOT EXISTS idx_integration_health_checked_at ON integration_health_checks(checked_at);

-- Composite index for common query pattern (org + integration + checked_at)
CREATE INDEX IF NOT EXISTS idx_integration_health_org_integration_checked
    ON integration_health_checks(org_id, integration, checked_at DESC);

-- Enable Row Level Security
ALTER TABLE integration_health_checks ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only access health checks for their organization
CREATE POLICY integration_health_checks_org_isolation ON integration_health_checks
    FOR ALL
    USING (org_id IN (
        SELECT org_id FROM profiles WHERE id = auth.uid()
    ));

-- Grant permissions
GRANT SELECT, INSERT ON integration_health_checks TO authenticated;

-- Add comments
COMMENT ON TABLE integration_health_checks IS 'Tracks health and connectivity status of external integrations';
COMMENT ON COLUMN integration_health_checks.integration IS 'Type of integration (slack/clickup/email/openai/groq)';
COMMENT ON COLUMN integration_health_checks.status IS 'Health status (healthy/degraded/failed/unknown)';
COMMENT ON COLUMN integration_health_checks.response_time_ms IS 'API response time in milliseconds';
COMMENT ON COLUMN integration_health_checks.metadata IS 'Additional health check details';
