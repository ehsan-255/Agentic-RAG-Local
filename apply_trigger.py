#!/usr/bin/env python
"""
Script to apply only the trigger SQL to the database.
This specifically applies the trigger to automatically update counts in documentation_sources.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("apply_trigger")

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def apply_trigger_sql():
    """Apply just the trigger part of the SQL to update the counts."""
    try:
        # Import schema module
        from src.db.schema import get_connection, release_connection
        
        # Define the SQL for just the trigger
        trigger_sql = """
-- Create or replace the trigger function to update documentation_sources aggregates
CREATE OR REPLACE FUNCTION update_source_counts() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE documentation_sources
        SET 
            pages_count = (
                SELECT COUNT(DISTINCT url)
                FROM site_pages 
                WHERE metadata->>'source_id' = NEW.metadata->>'source_id'
            ),
            chunks_count = (
                SELECT COUNT(*) 
                FROM site_pages 
                WHERE metadata->>'source_id' = NEW.metadata->>'source_id'
            )
        WHERE source_id = NEW.metadata->>'source_id';
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE documentation_sources
        SET 
            pages_count = (
                SELECT COUNT(DISTINCT url)
                FROM site_pages 
                WHERE metadata->>'source_id' = OLD.metadata->>'source_id'
            ),
            chunks_count = (
                SELECT COUNT(*) 
                FROM site_pages 
                WHERE metadata->>'source_id' = OLD.metadata->>'source_id'
            )
        WHERE source_id = OLD.metadata->>'source_id';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Drop any existing trigger (if present) and create the trigger to call the function
DROP TRIGGER IF EXISTS update_source_counts_trigger ON site_pages;

CREATE TRIGGER update_source_counts_trigger
AFTER INSERT OR DELETE ON site_pages
FOR EACH ROW
EXECUTE FUNCTION update_source_counts();
        """
        
        # Get database connection
        logger.info("Connecting to database...")
        conn = get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return False
        
        try:
            # Execute the trigger SQL
            logger.info("Applying trigger SQL...")
            cur = conn.cursor()
            cur.execute(trigger_sql)
            conn.commit()
            logger.info("Trigger applied successfully!")
            
            # Update existing records to fix current counts
            logger.info("Updating counts for existing records...")
            update_sql = """
            -- Update all documentation sources with correct counts
            UPDATE documentation_sources src
            SET 
                pages_count = (
                    SELECT COUNT(DISTINCT url)
                    FROM site_pages 
                    WHERE metadata->>'source_id' = src.source_id
                ),
                chunks_count = (
                    SELECT COUNT(*) 
                    FROM site_pages 
                    WHERE metadata->>'source_id' = src.source_id
                );
            """
            cur.execute(update_sql)
            conn.commit()
            logger.info("Counts updated for all sources!")
            
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error applying trigger SQL: {e}")
            return False
        finally:
            release_connection(conn)
            logger.info("Database connection closed")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting trigger application process...")
    if apply_trigger_sql():
        logger.info("Successfully applied trigger to update documentation_sources counts")
    else:
        logger.error("Failed to apply trigger") 