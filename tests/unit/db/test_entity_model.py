"""Unit tests for Entity and EntityMention pydantic models."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.entity import Entity, EntityMention


class TestEntityDefaults:
    def test_entity_defaults(self):
        """Entity created with required fields only should have correct defaults."""
        user_id = uuid.uuid4()
        entity = Entity(
            user_id=user_id,
            entity_name="Nike",
            entity_type="brand",
        )

        assert entity.entity_id is not None
        assert isinstance(entity.entity_id, uuid.UUID)
        assert entity.user_id == user_id
        assert entity.agent_id == uuid.UUID("00000000-0000-0000-0000-000000000000")
        assert entity.entity_name == "Nike"
        assert entity.entity_type == "brand"
        assert entity.embedding is None
        assert entity.facts == {}
        assert entity.confidence == 1.0
        assert entity.mention_count == 0
        assert entity.source_memory_ids == []
        assert entity.metadata == {}
        assert entity.created_at is not None
        assert entity.updated_at is not None

    def test_entity_normalized_name_auto(self):
        """normalized_name is automatically set to stripped lowercase of entity_name."""
        entity = Entity(
            user_id=uuid.uuid4(),
            entity_name="  Nike Inc. ",
            entity_type="organization",
        )
        assert entity.normalized_name == "nike inc."

    def test_entity_normalized_name_explicit(self):
        """An explicit normalized_name is preserved as-is."""
        entity = Entity(
            user_id=uuid.uuid4(),
            entity_name="Nike",
            entity_type="brand",
            normalized_name="nike_custom",
        )
        assert entity.normalized_name == "nike_custom"

    def test_entity_normalized_name_plain_strip_lower(self):
        """Plain name with mixed case and surrounding whitespace is normalized."""
        entity = Entity(
            user_id=uuid.uuid4(),
            entity_name="  OpenAI  ",
            entity_type="organization",
        )
        assert entity.normalized_name == "openai"


class TestEntityMentionDefaults:
    def test_entity_mention_defaults(self):
        """EntityMention created with required fields should have auto-generated id and occurred_at."""
        entity_id = uuid.uuid4()
        memory_id = uuid.uuid4()

        mention = EntityMention(entity_id=entity_id, memory_id=memory_id)

        assert isinstance(mention.id, uuid.UUID)
        assert mention.entity_id == entity_id
        assert mention.memory_id == memory_id
        assert mention.mention_text == ""
        assert mention.context == ""
        assert isinstance(mention.occurred_at, datetime)

    def test_entity_mention_id_is_unique(self):
        """Two EntityMention instances get distinct UUIDs by default."""
        entity_id = uuid.uuid4()
        memory_id = uuid.uuid4()

        m1 = EntityMention(entity_id=entity_id, memory_id=memory_id)
        m2 = EntityMention(entity_id=entity_id, memory_id=memory_id)

        assert m1.id != m2.id

    def test_entity_mention_occurred_at_is_utc(self):
        """occurred_at is timezone-aware (UTC)."""
        mention = EntityMention(entity_id=uuid.uuid4(), memory_id=uuid.uuid4())
        assert mention.occurred_at.tzinfo is not None


class TestEntityAllFields:
    def test_entity_all_fields(self):
        """Creating an Entity with all fields set should round-trip correctly."""
        entity_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        source_mem_id = uuid.uuid4()
        embedding = [0.1] * 768
        now = datetime.now(timezone.utc)

        entity = Entity(
            entity_id=entity_id,
            user_id=user_id,
            agent_id=agent_id,
            entity_name="Anthropic",
            entity_type="organization",
            normalized_name="anthropic",
            embedding=embedding,
            facts={"founded": 2021, "hq": "San Francisco"},
            confidence=0.95,
            mention_count=3,
            source_memory_ids=[source_mem_id],
            metadata={"source": "extraction"},
            created_at=now,
            updated_at=now,
        )

        assert entity.entity_id == entity_id
        assert entity.user_id == user_id
        assert entity.agent_id == agent_id
        assert entity.entity_name == "Anthropic"
        assert entity.entity_type == "organization"
        assert entity.normalized_name == "anthropic"
        assert entity.embedding == embedding
        assert entity.facts == {"founded": 2021, "hq": "San Francisco"}
        assert entity.confidence == 0.95
        assert entity.mention_count == 3
        assert entity.source_memory_ids == [source_mem_id]
        assert entity.metadata == {"source": "extraction"}
        assert entity.created_at == now
        assert entity.updated_at == now
