-- Physical delete of all rows owned by a user.
-- Step 1: count rows (used on dry-run and to populate the audit row).
-- Step 2 (non-dry-run): cascade delete in dependency order.
--
-- Order matters:
--   1. memory_history references memories(memory_id)
--   2. entity_mentions references entities(entity_id) and memories(memory_id)
--   3. entities (owned by user_id)
--   4. episodes (owned by user_id)
--   5. memories (owned by user_id)
--   6. session_buffer (owned by user_id)
--   7. contradiction_audit (owned by user_id)
--   8. processing_log (user_id, or legacy rows whose session belongs to user)
--   9. reextraction_jobs (owned by user_id)
--  10. user_settings (owned by user_id)
--
-- Reflections (memories with source_memory_ids pointing at deleted rows
-- owned by OTHER users) are soft-invalidated by setting valid_until = now()
-- so provenance remains queryable for the audit period.

-- Counts used for the audit row.
SELECT
    (SELECT COUNT(*) FROM memory.memories WHERE user_id = $1) AS rows_memories,
    (SELECT COUNT(*) FROM memory.entities WHERE user_id = $1) AS rows_entities,
    (SELECT COUNT(*) FROM memory.entity_mentions em
        JOIN memory.entities e ON em.entity_id = e.entity_id
        WHERE e.user_id = $1) AS rows_mentions,
    (SELECT COUNT(*) FROM memory.episodes WHERE user_id = $1) AS rows_episodes,
    (SELECT COUNT(*) FROM memory.memory_history mh
        JOIN memory.memories m ON mh.memory_id = m.memory_id
        WHERE m.user_id = $1) AS rows_history,
    (SELECT COUNT(*) FROM memory.session_buffer WHERE user_id = $1) AS rows_session_buffer,
    (SELECT COUNT(*) FROM memory.contradiction_audit WHERE user_id = $1) AS rows_contradiction_audit,
    (SELECT COUNT(*) FROM memory.processing_log
        WHERE user_id = $1
           OR session_id IN (
               SELECT session_id FROM memory.session_buffer WHERE user_id = $1
           )) AS rows_processing_log,
    (SELECT COUNT(*) FROM memory.reextraction_jobs WHERE user_id = $1) AS rows_reextraction_jobs;

-- Cascade delete (non-dry-run), in dependency order.
DELETE FROM memory.memory_history
    WHERE memory_id IN (SELECT memory_id FROM memory.memories WHERE user_id = $1);
DELETE FROM memory.entity_mentions
    WHERE entity_id IN (SELECT entity_id FROM memory.entities WHERE user_id = $1);
DELETE FROM memory.entity_mentions
    WHERE memory_id IN (SELECT memory_id FROM memory.memories WHERE user_id = $1);
DELETE FROM memory.entities WHERE user_id = $1;
DELETE FROM memory.episodes WHERE user_id = $1;
DELETE FROM memory.contradiction_audit WHERE user_id = $1;
DELETE FROM memory.reextraction_jobs WHERE user_id = $1;
DELETE FROM memory.processing_log
    WHERE user_id = $1
       OR session_id IN (SELECT session_id FROM memory.session_buffer WHERE user_id = $1);
DELETE FROM memory.session_buffer WHERE user_id = $1;
DELETE FROM memory.user_settings WHERE user_id = $1;
DELETE FROM memory.memories WHERE user_id = $1;
