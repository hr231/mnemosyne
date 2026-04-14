from __future__ import annotations

import os

import pytest

# GLiNER requires model download — gate behind env var
if not os.environ.get("MNEMOSYNE_NER"):
    pytest.skip("Set MNEMOSYNE_NER=1 to run GLiNER tests", allow_module_level=True)

from mnemosyne.pipeline.ner.gliner_extractor import extract_entities_gliner
from mnemosyne.pipeline.ner.spacy_extractor import RawEntity


def test_extract_brand():
    results = extract_entities_gliner("I love Nike shoes")
    assert isinstance(results, list)
    brands = [e for e in results if e.entity_type == "brand"]
    assert len(brands) >= 1
    assert any("Nike" in e.name for e in brands)


def test_returns_raw_entity_format():
    results = extract_entities_gliner("I bought an Adidas jacket last week")
    assert isinstance(results, list)
    for entity in results:
        assert isinstance(entity, RawEntity)
        assert entity.name
        assert entity.entity_type
        assert entity.source == "gliner"
