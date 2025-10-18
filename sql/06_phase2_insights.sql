-- Phase 2 - Step 3: AI Insights Table
-- This table stores AI-generated insights about organizational patterns, trends, and recommendations

CREATE TABLE IF NOT EXISTS ai_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('project', 'team', 'trend', 'risk', 'general')),
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    recommendations TEXT,
    confidence FLOAT DEFAULT 0.8 CHECK (confidence >= 0 AND confidence <= 1),
    source_refs JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only access insights from their organization
CREATE POLICY ai_insights_org_isolation
    ON ai_insights
    FOR ALL
    USING (org_id IN (
        SELECT org_id FROM profiles WHERE id = auth.uid()
    ));

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ai_insights_org_id ON ai_insights(org_id);
CREATE INDEX IF NOT EXISTS idx_ai_insights_created_at ON ai_insights(org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_insights_category ON ai_insights(org_id, category);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_ai_insights_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ai_insights_updated_at_trigger
    BEFORE UPDATE ON ai_insights
    FOR EACH ROW
    EXECUTE FUNCTION update_ai_insights_updated_at();

-- Helper function to get recent insights
CREATE OR REPLACE FUNCTION get_recent_insights(
    org_uuid UUID,
    insight_limit INT DEFAULT 10,
    insight_category TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    category TEXT,
    title TEXT,
    summary TEXT,
    recommendations TEXT,
    confidence FLOAT,
    source_refs JSONB,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ai.id,
        ai.category,
        ai.title,
        ai.summary,
        ai.recommendations,
        ai.confidence,
        ai.source_refs,
        ai.created_at
    FROM ai_insights ai
    WHERE ai.org_id = org_uuid
        AND (insight_category IS NULL OR ai.category = insight_category)
    ORDER BY ai.created_at DESC
    LIMIT insight_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Helper function to get insight statistics
CREATE OR REPLACE FUNCTION get_insight_stats(org_uuid UUID)
RETURNS TABLE (
    total_insights BIGINT,
    project_insights BIGINT,
    team_insights BIGINT,
    trend_insights BIGINT,
    risk_insights BIGINT,
    general_insights BIGINT,
    avg_confidence FLOAT,
    last_generated TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT AS total_insights,
        COUNT(*) FILTER (WHERE category = 'project')::BIGINT AS project_insights,
        COUNT(*) FILTER (WHERE category = 'team')::BIGINT AS team_insights,
        COUNT(*) FILTER (WHERE category = 'trend')::BIGINT AS trend_insights,
        COUNT(*) FILTER (WHERE category = 'risk')::BIGINT AS risk_insights,
        COUNT(*) FILTER (WHERE category = 'general')::BIGINT AS general_insights,
        AVG(confidence)::FLOAT AS avg_confidence,
        MAX(created_at) AS last_generated
    FROM ai_insights
    WHERE org_id = org_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON TABLE ai_insights IS 'AI-generated insights about organizational patterns, trends, and recommendations';
COMMENT ON COLUMN ai_insights.category IS 'Category: project, team, trend, risk, or general';
COMMENT ON COLUMN ai_insights.confidence IS 'Confidence score from 0 to 1';
COMMENT ON COLUMN ai_insights.source_refs IS 'JSON array of references (entity IDs, document IDs, etc.)';
