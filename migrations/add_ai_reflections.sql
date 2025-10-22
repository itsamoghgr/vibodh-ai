-- Migration: Add ai_reflections table for agent self-reflection and learning
-- Description: Stores agent reflections after action execution for adaptive learning
-- Created: 2025-10-22

-- Create ai_reflections table
CREATE TABLE IF NOT EXISTS ai_reflections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    agent_type TEXT NOT NULL,  -- Backend validates agent type
    plan_id UUID,  -- Reference to ai_action_plans
    action_id UUID,  -- Reference to ai_actions_executed

    -- Reflection metadata
    reflection_type TEXT NOT NULL,  -- Backend validates (execution/planning/learning/performance)
    trigger_event TEXT,  -- What triggered this reflection

    -- Reflection content
    summary TEXT NOT NULL,  -- High-level summary of the reflection
    insights JSONB DEFAULT '[]',  -- Array of specific insights learned
    patterns_discovered JSONB DEFAULT '[]',  -- Patterns identified
    improvements_suggested JSONB DEFAULT '[]',  -- Suggestions for future improvement

    -- Performance metrics
    overall_success BOOLEAN,
    confidence_score NUMERIC CHECK (confidence_score >= 0 AND confidence_score <= 1),
    performance_metrics JSONB DEFAULT '{}',  -- Execution metrics

    -- Learning indicators
    learning_points JSONB DEFAULT '[]',  -- Specific learning points for adaptive engine
    preference_updates JSONB DEFAULT '[]',  -- Suggested preference adjustments

    -- Context
    context JSONB DEFAULT '{}',  -- Context at time of reflection
    metadata JSONB DEFAULT '{}',  -- Additional metadata

    -- Adaptive engine integration
    ingested_by_adaptive_engine BOOLEAN DEFAULT false,
    ingested_at TIMESTAMPTZ,
    adaptive_actions_taken JSONB DEFAULT '[]',  -- Actions taken by adaptive engine based on reflection

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT ai_reflections_org_id_fkey FOREIGN KEY (org_id) REFERENCES organizations(id),
    CONSTRAINT ai_reflections_plan_id_fkey FOREIGN KEY (plan_id) REFERENCES ai_action_plans(id),
    CONSTRAINT ai_reflections_action_id_fkey FOREIGN KEY (action_id) REFERENCES ai_actions_executed(id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ai_reflections_org_id ON ai_reflections(org_id);
CREATE INDEX IF NOT EXISTS idx_ai_reflections_agent_type ON ai_reflections(agent_type);
CREATE INDEX IF NOT EXISTS idx_ai_reflections_plan_id ON ai_reflections(plan_id);
CREATE INDEX IF NOT EXISTS idx_ai_reflections_reflection_type ON ai_reflections(reflection_type);
CREATE INDEX IF NOT EXISTS idx_ai_reflections_created_at ON ai_reflections(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_reflections_ingested ON ai_reflections(ingested_by_adaptive_engine);

-- Composite index for adaptive engine queries (org + not ingested + created_at)
CREATE INDEX IF NOT EXISTS idx_ai_reflections_adaptive_queue
    ON ai_reflections(org_id, created_at DESC)
    WHERE ingested_by_adaptive_engine = false;

-- Composite index for agent performance analysis
CREATE INDEX IF NOT EXISTS idx_ai_reflections_agent_performance
    ON ai_reflections(org_id, agent_type, created_at DESC);

-- Enable Row Level Security
ALTER TABLE ai_reflections ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only access reflections for their organization
CREATE POLICY ai_reflections_org_isolation ON ai_reflections
    FOR ALL
    USING (org_id IN (
        SELECT org_id FROM profiles WHERE id = auth.uid()
    ));

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON ai_reflections TO authenticated;

-- Add comments
COMMENT ON TABLE ai_reflections IS 'Agent self-reflections for adaptive learning and continuous improvement';
COMMENT ON COLUMN ai_reflections.reflection_type IS 'Type of reflection (execution/planning/learning/performance)';
COMMENT ON COLUMN ai_reflections.insights IS 'Specific insights learned from this experience';
COMMENT ON COLUMN ai_reflections.patterns_discovered IS 'Behavioral or contextual patterns identified';
COMMENT ON COLUMN ai_reflections.learning_points IS 'Specific points for adaptive engine to learn from';
COMMENT ON COLUMN ai_reflections.preference_updates IS 'Suggested updates to agent preferences';
COMMENT ON COLUMN ai_reflections.ingested_by_adaptive_engine IS 'Whether adaptive engine has processed this reflection';
COMMENT ON COLUMN ai_reflections.adaptive_actions_taken IS 'Actions taken by adaptive engine based on this reflection';
