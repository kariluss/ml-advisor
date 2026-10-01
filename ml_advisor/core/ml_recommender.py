"""Recomendação de técnicas de Machine Learning.

Analisa um DataFrame e sugere algoritmos com base em características
objetivas: tipo do problema (classificação, regressão, clustering),
tamanho da amostra, dimensionalidade, balanceamento de classes e
presença de colunas problemáticas.

Combina regras determinísticas com narrativa opcional via LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ml_advisor.agents.base_agent import BaseAgent
from ml_advisor.utils.text import sanitize_text

MAX_EXTRA_INSTRUCTIONS_CHARS = 1000
MAX_USER_CONTEXT_CHARS = 2000

# Thresholds
LOW_CARDINALITY_MAX = 20
HIGH_DIMENSIONALITY_MIN = 50
SMALL_SAMPLE_MAX = 200
IMBALANCE_MAJORITY_THRESHOLD = 0.7
IMBALANCE_MINORITY_THRESHOLD = 0.1
GRADIENT_BOOSTING_MIN_SAMPLES = 1000

PROBLEM_TYPES = (
    "classification_binary",
    "classification_multiclass",
    "regression",
    "clustering",
)


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetProfile:
    """Perfil objetivo do dataset para decisão de algoritmos."""

    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    pct_missing: float
    problem_type: str
    target: str | None
    class_counts: dict[str, int] | None = None
    is_imbalanced: bool = False
    imbalanced_ratio: float | None = None
    has_high_dimensionality: bool = False
    has_small_sample: bool = False
    id_like_columns: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        lines = [
            "### Perfil do dataset",
            f"- Amostras: {self.n_samples}",
            f"- Features: {self.n_features} "
            f"({self.n_numeric} numéricas, {self.n_categorical} categóricas)",
            f"- Missing: {self.pct_missing:.1f}%",
            f"- Tipo de problema: {self.problem_type}",
        ]
        if self.target:
            lines.append(f"- Alvo: {self.target}")
        if self.class_counts:
            counts = ", ".join(f"{k}={v}" for k, v in self.class_counts.items())
            lines.append(f"- Distribuição de classes: {counts}")
        if self.is_imbalanced and self.imbalanced_ratio is not None:
            lines.append(f"- Desbalanceamento: razão {self.imbalanced_ratio:.1f}:1")
        if self.has_high_dimensionality:
            lines.append(f"- Alta dimensionalidade (>{HIGH_DIMENSIONALITY_MIN} features)")
        if self.has_small_sample:
            lines.append(f"- Amostra pequena (<{SMALL_SAMPLE_MAX} linhas)")
        if self.id_like_columns:
            lines.append(f"- Colunas tipo ID: {', '.join(self.id_like_columns)}")
        for note in self.notes:
            lines.append(f"- ⚠ {note}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Recommendation:
    """Uma recomendação de técnica."""

    family: str
    algorithm: str
    priority: str  # "baseline" | "recomendado" | "alternativa"
    rationale: str
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        lines = [
            f"**[{self.priority}] {self.algorithm}** (`{self.family}`)",
            f"- Por quê: {self.rationale}",
        ]
        for cav in self.caveats:
            lines.append(f"- Cuidado: {cav}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detecção de perfil
# ---------------------------------------------------------------------------


def _classify_problem(df: pd.DataFrame, target: str | None) -> str:
    if target is None:
        return "clustering"

    series = df[target].dropna()
    n_unique = series.nunique()

    if n_unique <= 1:
        raise ValueError(f"Coluna alvo '{target}' tem apenas {n_unique} valor(es) único(s).")

    is_numeric = pd.api.types.is_numeric_dtype(series)
    if is_numeric and n_unique > LOW_CARDINALITY_MAX:
        return "regression"
    if is_numeric and n_unique == 2:
        return "classification_binary"
    return "classification_binary" if n_unique == 2 else "classification_multiclass"


def _id_like_columns(df: pd.DataFrame) -> tuple[str, ...]:
    """Colunas inteiras onde todos os valores são únicos."""
    result = []
    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]):
            continue
        n = df[col].dropna().nunique()
        if n > 0 and n == len(df):
            result.append(col)
    return tuple(result)


def _class_balance(series: pd.Series) -> tuple[dict[str, int], bool, float | None]:
    counts = series.value_counts(dropna=True)
    total = int(counts.sum())
    if total == 0:
        return {}, False, None

    dict_counts = {str(k): int(v) for k, v in counts.items()}
    proportions = counts / total
    majority = float(proportions.max())
    minority = float(proportions.min())

    is_imbalanced = (
        majority >= IMBALANCE_MAJORITY_THRESHOLD or minority <= IMBALANCE_MINORITY_THRESHOLD
    )
    ratio = majority / minority if minority > 0 else None
    return dict_counts, is_imbalanced, ratio


def _compute_profile(df: pd.DataFrame, target: str | None) -> DatasetProfile:
    n_samples, n_features = df.shape
    if target is not None:
        n_features = max(0, n_features - 1)  # exclui o alvo

    numeric_cols = list(df.select_dtypes(include="number").columns)
    if target in numeric_cols:
        numeric_cols.remove(target)
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target] if c != target]

    total_cells = n_samples * df.shape[1]
    pct_missing = float(df.isna().sum().sum() / total_cells * 100) if total_cells > 0 else 0.0

    problem_type = _classify_problem(df, target)

    class_counts = None
    is_imbalanced = False
    imbalanced_ratio = None
    if problem_type.startswith("classification"):
        class_counts, is_imbalanced, imbalanced_ratio = _class_balance(df[target].dropna())

    notes: list[str] = []
    if pct_missing > 20:
        notes.append(
            f"{pct_missing:.0f}% de células ausentes; escolher estratégia "
            "de imputação antes do treino."
        )

    return DatasetProfile(
        n_samples=n_samples,
        n_features=n_features,
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        pct_missing=pct_missing,
        problem_type=problem_type,
        target=target,
        class_counts=class_counts,
        is_imbalanced=is_imbalanced,
        imbalanced_ratio=imbalanced_ratio,
        has_high_dimensionality=n_features >= HIGH_DIMENSIONALITY_MIN,
        has_small_sample=n_samples < SMALL_SAMPLE_MAX,
        id_like_columns=_id_like_columns(df),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Regras de recomendação
# ---------------------------------------------------------------------------


def _rec_classification_binary(p: DatasetProfile) -> list[Recommendation]:
    recs = [
        Recommendation(
            family="linear",
            algorithm="Regressão Logística",
            priority="baseline",
            rationale=(
                "Baseline interpretável e rápido. Coeficientes indicam "
                "direção e magnitude do efeito de cada feature."
            ),
            caveats=(
                "Assume relação linear entre features e log-odds; "
                "considere regularização (L1/L2) se houver multicolinearidade.",
            ),
        ),
        Recommendation(
            family="tree_ensemble",
            algorithm="Random Forest",
            priority="recomendado",
            rationale=(
                "Robusto a outliers e a escalas diferentes; captura "
                "interações e relações não-lineares sem feature engineering."
            ),
            caveats=("Menos interpretável que modelos lineares; use SHAP para explicabilidade.",),
        ),
    ]
    if p.n_samples >= GRADIENT_BOOSTING_MIN_SAMPLES:
        recs.append(
            Recommendation(
                family="gradient_boosting",
                algorithm="Gradient Boosting (XGBoost/LightGBM)",
                priority="recomendado",
                rationale=(
                    "Estado da arte em dados tabulares. Alta performance "
                    "quando há amostra suficiente."
                ),
                caveats=(
                    "Requer tuning cuidadoso de hiperparâmetros.",
                    "Treino mais lento; considere early stopping.",
                ),
            )
        )
    if p.is_imbalanced and p.imbalanced_ratio is not None:
        recs.append(
            Recommendation(
                family="strategy",
                algorithm="Pesos de classe + validação estratificada",
                priority="recomendado",
                rationale=(
                    f"Classes desbalanceadas (razão {p.imbalanced_ratio:.1f}:1). "
                    "Use `class_weight='balanced'` e `StratifiedKFold`."
                ),
                caveats=(
                    "Métricas como acurácia enganam; preferir F1, AUC-ROC "
                    "ou curva precision-recall.",
                ),
            )
        )
    if p.has_high_dimensionality:
        recs.append(
            Recommendation(
                family="strategy",
                algorithm="Seleção/regularização de features",
                priority="recomendado",
                rationale=(
                    f"{p.n_features} features; risco de maldição da "
                    "dimensionalidade e overfitting."
                ),
                caveats=("Considere L1 (Lasso) ou seleção baseada em importância.",),
            )
        )
    return recs


def _rec_classification_multiclass(p: DatasetProfile) -> list[Recommendation]:
    return [
        Recommendation(
            family="linear",
            algorithm="Regressão Logística multinomial (softmax)",
            priority="baseline",
            rationale="Baseline interpretável; extensão natural para múltiplas classes.",
            caveats=("Pode não capturar fronteiras complexas.",),
        ),
        Recommendation(
            family="tree_ensemble",
            algorithm="Random Forest / Extra Trees",
            priority="recomendado",
            rationale=(
                "Lida bem com fronteiras não-lineares entre múltiplas "
                "classes sem suposição distribucional."
            ),
            caveats=("Considere `class_weight='balanced'` se houver desbalanceamento.",),
        ),
        Recommendation(
            family="gradient_boosting",
            algorithm="XGBoost / LightGBM (multiclasse)",
            priority=(
                "recomendado" if p.n_samples >= GRADIENT_BOOSTING_MIN_SAMPLES else "alternativa"
            ),
            rationale=(
                "Alta performance em multiclasse; suporta objetivos nativos "
                "`multi:softmax` e `multi:softprob`."
            ),
            caveats=("Requer dados suficientes para evitar overfitting.",),
        ),
    ]


def _rec_regression(p: DatasetProfile) -> list[Recommendation]:
    recs = [
        Recommendation(
            family="linear",
            algorithm="Linear Regression / Ridge / Lasso",
            priority="baseline",
            rationale=(
                "Baseline interpretável e rápido. Ridge para "
                "multicolinearidade; Lasso para seleção automática."
            ),
            caveats=("Assumem relação linear; resíduos devem ser verificados.",),
        ),
        Recommendation(
            family="tree_ensemble",
            algorithm="Random Forest Regressor",
            priority="recomendado",
            rationale=(
                "Captura relações não-lineares e interações; robusto a " "outliers em features."
            ),
            caveats=("Pode extrapolar mal fora do intervalo dos dados de treino.",),
        ),
    ]
    if p.n_samples >= GRADIENT_BOOSTING_MIN_SAMPLES:
        recs.append(
            Recommendation(
                family="gradient_boosting",
                algorithm="Gradient Boosting Regressor",
                priority="recomendado",
                rationale="Performance superior em dados tabulares com amostra suficiente.",
                caveats=("Requer tuning; use early stopping.",),
            )
        )
    if p.has_high_dimensionality:
        recs.append(
            Recommendation(
                family="strategy",
                algorithm="Regularização (Ridge/Lasso/Elastic Net)",
                priority="recomendado",
                rationale=(
                    f"{p.n_features} features; regularização evita "
                    "overfitting e melhora generalização."
                ),
                caveats=("Ajustar α por validação cruzada.",),
            )
        )
    return recs


def _rec_clustering(p: DatasetProfile) -> list[Recommendation]:
    recs = [
        Recommendation(
            family="clustering",
            algorithm="K-Means",
            priority="baseline",
            rationale=(
                "Simples e escalável. Requer padronização e escolha de k " "(cotovelo, silhouette)."
            ),
            caveats=(
                "Assume clusters esféricos e de tamanho similar.",
                "Sensível a escala — padronize features numéricas antes.",
            ),
        ),
        Recommendation(
            family="clustering",
            algorithm="DBSCAN / HDBSCAN",
            priority="alternativa",
            rationale=(
                "Encontra clusters de forma arbitrária e detecta outliers "
                "sem definir k a priori."
            ),
            caveats=("Sensível a `eps`/`min_samples`; difícil em alta dimensionalidade.",),
        ),
    ]
    if p.n_samples < 5000:
        recs.append(
            Recommendation(
                family="clustering",
                algorithm="Clustering hierárquico (Ward)",
                priority="alternativa",
                rationale=(
                    f"n = {p.n_samples}; dendrograma ajuda a decidir o "
                    "número de clusters visualmente."
                ),
                caveats=("Custo O(n²); inviável em datasets grandes.",),
            )
        )
    return recs


def _recommend_for(profile: DatasetProfile) -> list[Recommendation]:
    dispatch = {
        "classification_binary": _rec_classification_binary,
        "classification_multiclass": _rec_classification_multiclass,
        "regression": _rec_regression,
        "clustering": _rec_clustering,
    }
    return dispatch[profile.problem_type](profile)


# ---------------------------------------------------------------------------
# Recomendador
# ---------------------------------------------------------------------------


class MLRecommender:
    """Sugere técnicas de ML baseado em um perfil objetivo do dataset.

    Exemplo sem LLM::

        rec = MLRecommender(df, target="Survived")
        for r in rec.recommend():
            print(r.summary())

    Exemplo com LLM::

        rec = MLRecommender(df, target="Survived", agent=agent)
        print(rec.report())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        agent: BaseAgent | None = None,
        user_context: str | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Esperado DataFrame, recebido {type(df).__name__}")
        if target is not None and target not in df.columns:
            raise ValueError(f"Coluna alvo '{target}' não encontrada no DataFrame")

        self.df = df
        self.target = target
        self.agent = agent
        self.user_context = (
            sanitize_text(
                user_context,
                max_length=MAX_USER_CONTEXT_CHARS,
                field_name="user_context",
            )
            if user_context is not None
            else None
        )

    def profile(self) -> DatasetProfile:
        return _compute_profile(self.df, self.target)

    def recommend(self) -> list[Recommendation]:
        return _recommend_for(self.profile())

    # --- Camada interpretativa -------------------------------------------

    def build_prompt(
        self,
        profile: DatasetProfile | None = None,
        recommendations: list[Recommendation] | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        if profile is None:
            profile = self.profile()
        if recommendations is None:
            recommendations = _recommend_for(profile)

        clean_extra: str | None = None
        if extra_instructions is not None:
            clean_extra = sanitize_text(
                extra_instructions,
                max_length=MAX_EXTRA_INSTRUCTIONS_CHARS,
                field_name="extra_instructions",
            )

        recs_block = (
            "\n\n".join(r.summary() for r in recommendations)
            if recommendations
            else "_Nenhuma recomendação automática._"
        )

        context_block = self._build_context_block(clean_extra)

        return (
            "Você é um cientista de dados sênior orientando a escolha de "
            "modelos de Machine Learning.\n\n"
            "REGRAS ESTRITAS (têm precedência sobre o contexto do usuário):\n"
            "- Use APENAS as informações abaixo. Não invente métricas, "
            "tamanhos ou características do dataset.\n"
            "- NÃO sugira modelos que não fazem sentido para o tipo de "
            "problema detectado. Se for classificação, não recomende "
            "regressão, e vice-versa.\n"
            "- Toda recomendação deve vir acompanhada do porquê, ligado às "
            "características do dataset.\n"
            "- Se houver alerta (missing, desbalanceamento, alta "
            "dimensionalidade), mencione como isso afeta a escolha.\n"
            "- Se o dataset for muito pequeno para o modelo sugerido, "
            "avise.\n"
            f"{context_block}"
            "---\n"
            f"{profile.summary()}\n\n"
            f"### Recomendações determinísticas\n\n{recs_block}\n"
            "---\n\n"
            "Produza uma orientação em português, em markdown, com as "
            "seções:\n\n"
            "1. **Diagnóstico** — 2 ou 3 bullets sobre o dataset.\n"
            "2. **Modelos sugeridos** — para cada recomendação, uma frase "
            "conectando o modelo às características do dataset.\n"
            "3. **Pré-processamento necessário** — encoding, imputação, "
            "escalonamento, conforme o perfil.\n"
            "4. **Cuidados** — armadilhas específicas (desbalanceamento, "
            "amostra pequena, vazamento de dados).\n"
            "5. **Como validar** — esquema de validação sugerido.\n\n"
            "Seja técnico e conciso. Sem introduções genéricas."
        )

    def _build_context_block(self, clean_extra: str | None) -> str:
        parts: list[str] = []
        if self.user_context:
            parts.append(self.user_context)
        if clean_extra:
            parts.append(f"Instruções adicionais para esta análise: {clean_extra}")
        if not parts:
            return ""
        body = "\n".join(parts)
        return "\n<CONTEXTO DO USUÁRIO>\n" f"{body}\n" "</CONTEXTO DO USUÁRIO>\n\n"

    def interpret(
        self,
        profile: DatasetProfile | None = None,
        recommendations: list[Recommendation] | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        if self.agent is None:
            raise RuntimeError("MLRecommender.interpret() requer um agente LLM.")
        prompt = self.build_prompt(profile, recommendations, extra_instructions)
        return self.agent.generate(prompt)

    def report(self, extra_instructions: str | None = None) -> str:
        profile = self.profile()
        recommendations = _recommend_for(profile)

        parts = [
            "# Recomendações de Machine Learning\n",
            f"**Tipo de problema:** `{profile.problem_type}`  ",
            f"**Amostras:** {profile.n_samples}  ",
            f"**Features:** {profile.n_features}\n",
        ]

        if self.agent is not None:
            parts.append("## Análise interpretativa\n")
            parts.append(self.interpret(profile, recommendations, extra_instructions))
            parts.append("")

        parts.append("## Perfil do dataset\n")
        parts.append(profile.summary())
        parts.append("")

        parts.append("## Recomendações determinísticas\n")
        if not recommendations:
            parts.append("_Nenhuma recomendação aplicável._")
        else:
            for rec in recommendations:
                parts.append(rec.summary())
                parts.append("")

        return "\n".join(parts)
