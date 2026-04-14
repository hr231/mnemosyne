from __future__ import annotations

import os

import pytest

# All tests in this module require torch + transformers + model download.
# Gate them behind MNEMOSYNE_NLI=1 so CI does not download ~400 MB on every run.
pytestmark = pytest.mark.skipif(
    os.environ.get("MNEMOSYNE_NLI", "") not in ("1", "true", "yes"),
    reason="MNEMOSYNE_NLI not enabled — set MNEMOSYNE_NLI=1 to run NLI tests",
)


class TestPredictNLI:
    def test_returns_nli_result(self):
        """predict_nli returns an NLIResult with all three probability fields."""
        from mnemosyne.pipeline.nli import predict_nli, NLIResult

        result = predict_nli("The sky is blue", "The sky is blue")
        assert isinstance(result, NLIResult)
        assert hasattr(result, "entailment")
        assert hasattr(result, "contradiction")
        assert hasattr(result, "neutral")
        # Scores are probabilities and should sum to approximately 1.
        total = result.entailment + result.contradiction + result.neutral
        assert abs(total - 1.0) < 0.01, f"scores must sum to ~1.0, got {total}"

    def test_predict_contradiction(self):
        """Factually contradictory sentences score high on contradiction."""
        from mnemosyne.pipeline.nli import predict_nli

        # "The sky is blue" vs "The sky is green" are mutually exclusive claims.
        result = predict_nli("The sky is blue", "The sky is green")
        assert result.contradiction > 0.5, (
            f"Expected contradiction > 0.5 for contradictory pair, got {result.contradiction}"
        )

    def test_predict_entailment(self):
        """Paraphrases / semantically equivalent sentences score high on entailment."""
        from mnemosyne.pipeline.nli import predict_nli

        result = predict_nli("Dogs are animals", "Canines are animals")
        assert result.entailment > 0.5, (
            f"Expected entailment > 0.5 for entailing pair, got {result.entailment}"
        )

    def test_predict_neutral(self):
        """Unrelated sentences score high on neutral (neither entailment nor contradiction)."""
        from mnemosyne.pipeline.nli import predict_nli

        result = predict_nli("The sky is blue", "Pizza is delicious")
        assert result.neutral > 0.5, (
            f"Expected neutral > 0.5 for unrelated pair, got {result.neutral}"
        )
