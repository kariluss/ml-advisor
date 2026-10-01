"""Módulos de lógica pura (pandas, scipy, sklearn).

Não dependem de LLM. Aceitam um ``BaseAgent`` opcional para enriquecer
a saída com narrativa interpretativa.
"""

from ml_advisor.core.data_analyzer import (
    ColumnSummary,
    DataAnalyzer,
    DatasetSummary,
)
from ml_advisor.core.hypothesis_tester import HypothesisTester, TestResult
from ml_advisor.core.ml_recommender import (
    DatasetProfile,
    MLRecommender,
    Recommendation,
)

__all__ = [
    "ColumnSummary",
    "DataAnalyzer",
    "DatasetProfile",
    "DatasetSummary",
    "HypothesisTester",
    "MLRecommender",
    "Recommendation",
    "TestResult",
]
