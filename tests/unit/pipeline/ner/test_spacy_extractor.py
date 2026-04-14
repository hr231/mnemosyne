from __future__ import annotations

import pytest

try:
    import spacy
    spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SPACY_AVAILABLE,
    reason="spaCy or en_core_web_sm not available",
)

from mnemosyne.pipeline.ner.spacy_extractor import RawEntity, extract_entities_spacy


def _find(entities: list[RawEntity], entity_type: str) -> list[RawEntity]:
    return [e for e in entities if e.entity_type == entity_type]


def test_extract_person():
    results = extract_entities_spacy("John Smith went to the store")
    persons = _find(results, "person")
    assert len(persons) >= 1
    names = [e.name for e in persons]
    assert any("John" in n or "Smith" in n for n in names)


def test_extract_organization():
    results = extract_entities_spacy("Apple Inc. released a new product")
    orgs = _find(results, "organization")
    assert len(orgs) >= 1
    assert any("Apple" in e.name for e in orgs)


def test_extract_location():
    results = extract_entities_spacy("She lives in New York")
    locations = _find(results, "location")
    assert len(locations) >= 1
    assert any("New York" in e.name for e in locations)


def test_no_entities():
    results = extract_entities_spacy("Hello world")
    assert isinstance(results, list)


def test_returns_raw_entity_format():
    results = extract_entities_spacy("John Smith works at Apple Inc.")
    assert len(results) >= 1
    for entity in results:
        assert isinstance(entity, RawEntity)
        assert entity.name
        assert entity.entity_type
        assert entity.source == "spacy"
        assert isinstance(entity.mention_text, str)
        assert isinstance(entity.context, str)
