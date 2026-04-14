from __future__ import annotations

import pytest

from mnemosyne.pipeline.ner.router import merge_entities, should_use_llm
from mnemosyne.pipeline.ner.spacy_extractor import RawEntity


def _make(name: str, entity_type: str, source: str, mention_text: str = "", context: str = "") -> RawEntity:
    return RawEntity(
        name=name,
        entity_type=entity_type,
        mention_text=mention_text or name,
        context=context,
        source=source,
    )


def test_merge_dedup_by_name():
    spacy = [_make("Nike", "organization", "spacy")]
    gliner = [_make("Nike", "brand", "gliner")]
    result = merge_entities(spacy, gliner, [])
    assert len(result) == 1


def test_spacy_type_wins_for_standard():
    """When the entity name is the same, spaCy's standard type wins over GLiNER's."""
    spacy = [_make("Google", "organization", "spacy")]
    gliner = [_make("Google", "brand", "gliner")]
    result = merge_entities(spacy, gliner, [])
    assert len(result) == 1
    assert result[0].entity_type == "organization"


def test_gliner_type_wins_for_domain():
    """For a non-standard entity that spaCy does not find, GLiNER's label is kept."""
    gliner = [_make("Gore-Tex", "material", "gliner")]
    result = merge_entities([], gliner, [])
    assert len(result) == 1
    assert result[0].entity_type == "material"


def test_gliner_type_wins_when_spacy_has_non_standard():
    """If spaCy produces a product type and GLiNER produces brand, GLiNER wins."""
    spacy = [_make("Adidas", "product", "spacy")]
    gliner = [_make("Adidas", "brand", "gliner")]
    result = merge_entities(spacy, gliner, [])
    assert len(result) == 1
    # product is not in STANDARD_TYPES, so GLiNER's brand should win
    assert result[0].entity_type == "brand"


def test_longest_mention_wins():
    short = _make("Nike", "organization", "spacy", mention_text="Nike")
    long = _make("Nike", "brand", "gliner", mention_text="Nike Inc.")
    result = merge_entities([short], [long], [])
    assert len(result) == 1
    assert result[0].mention_text == "Nike Inc."


def test_priority_ordering_spacy_over_gliner():
    spacy = [_make("Apple", "organization", "spacy")]
    gliner = [_make("Apple", "brand", "gliner")]
    result = merge_entities(spacy, gliner, [])
    assert result[0].source == "spacy"


def test_priority_ordering_gliner_over_llm():
    gliner = [_make("Tesla", "organization", "gliner")]
    llm = [_make("Tesla", "brand", "llm")]
    result = merge_entities([], gliner, llm)
    assert result[0].source == "gliner"


def test_priority_ordering_spacy_over_llm():
    spacy = [_make("Paris", "location", "spacy")]
    llm = [_make("Paris", "concept", "llm")]
    result = merge_entities(spacy, [], llm)
    assert result[0].source == "spacy"


def test_all_unique_entities_kept():
    spacy = [_make("John Smith", "person", "spacy")]
    gliner = [_make("Nike", "brand", "gliner")]
    llm = [_make("Paris", "location", "llm")]
    result = merge_entities(spacy, gliner, llm)
    assert len(result) == 3


def test_case_insensitive_dedup():
    """Names differing only in case should deduplicate."""
    spacy = [_make("apple", "organization", "spacy")]
    gliner = [_make("Apple", "brand", "gliner")]
    result = merge_entities(spacy, gliner, [])
    assert len(result) == 1


def test_should_use_llm_when_no_entities():
    text = "The user has very specific preferences about interior design aesthetics."
    result = should_use_llm(text, spacy_count=0, gliner_count=0)
    assert result is True


def test_should_not_use_llm_when_entities_found():
    text = "John Smith works at Apple Inc."
    result = should_use_llm(text, spacy_count=2, gliner_count=0)
    assert result is False


def test_should_not_use_llm_for_very_short_text():
    """Very short text with no entities should not trigger LLM (nothing to extract)."""
    result = should_use_llm("Hi", spacy_count=0, gliner_count=0)
    assert result is False


def test_longest_context_kept():
    short_ctx = _make("Nike", "organization", "spacy", context="Nike.")
    long_ctx = _make("Nike", "brand", "gliner", context="I really love Nike shoes from the store.")
    result = merge_entities([short_ctx], [long_ctx], [])
    assert len(result) == 1
    assert result[0].context == "I really love Nike shoes from the store."
