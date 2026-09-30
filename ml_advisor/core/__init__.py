"""Módulos de lógica pura (pandas, scipy, sklearn).

Não dependem de LLM. Aceitam um ``BaseAgent`` opcional para enriquecer
a saída com narrativa interpretativa.
"""

from ml_advisor.core.data_analyzer import (
    ColumnSummary,
    DataAnalyzer,
    DatasetSummary,
)

__all__ = ["ColumnSummary", "DataAnalyzer", "DatasetSummary"]
