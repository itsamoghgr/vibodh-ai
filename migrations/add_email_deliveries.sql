-- Migration: Add email_deliveries table for tracking email delivery
-- Description: Enables tracking of email sends, deliveries, and failures
-- Created: 2025-10-22

-- Create email_deliveries table
CREATE TABLE IF NOT EXISTS email_deliveries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL,
    recipients TEXT[] NOT NULL,  -- Array of recipient email addresses
    subject TEXT NOT NULL,
    status TEXT NOT NULL,  -- Backend validates (sent/failed/bounced/delivered)
    error TEXT,  -- Error message if failed
    attempt INT NOT NULL DEFAULT 1,
    message_id TEXT,  -- SMTP message ID
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at TIMESTAMPTZ,  -- When delivery was confirmed (if available)
    bounced_at TIMESTAMPTZ,  -- When bounce was detected (if available)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_email_deliveries_org_id ON email_deliveries(org_id);
CREATE INDEX IF NOT EXISTS idx_email_deliveries_status ON email_deliveries(status);
CREATE INDEX IF NOT EXISTS idx_email_deliveries_sent_at ON email_deliveries(sent_at);

-- Composite index for common query pattern (org + status + sent_at)
CREATE INDEX IF NOT EXISTS idx_email_deliveries_org_status_sent
    ON email_deliveries(org_id, status, sent_at);

-- Enable Row Level Security
ALTER TABLE email_deliveries ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only access deliveries for their organization
CREATE POLICY email_deliveries_org_isolation ON email_deliveries
    FOR ALL
    USING (org_id IN (
        SELECT org_id FROM profiles WHERE id = auth.uid()
    ));

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON email_deliveries TO authenticated;

-- Add comments
COMMENT ON TABLE email_deliveries IS 'Tracks email delivery status and history';
COMMENT ON COLUMN email_deliveries.recipients IS 'Array of recipient email addresses';
COMMENT ON COLUMN email_deliveries.status IS 'Delivery status (sent/failed/bounced/delivered)';
COMMENT ON COLUMN email_deliveries.attempt IS 'Number of send attempts';
COMMENT ON COLUMN email_deliveries.message_id IS 'SMTP message ID for tracking';
