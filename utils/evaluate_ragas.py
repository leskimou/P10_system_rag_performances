"""Évaluation de la qualité du RAG avec RAGAS (faithfulness, context_precision,
context_recall, answer_relevancy).

Le dataset (utils/ragas_dataset.json) contient les questions, réponses et
contextes déjà générés par le chatbot (champs "answer" et "contexts", remplis
via utils/ans_cont_recup_ragas.py), pour éviter de refaire les appels au
chatbot à chaque évaluation. Nécessite MISTRAL_API_KEY pour le LLM juge.

Lancer avec :

    uv run python utils/evaluate_ragas.py

Le LLM juge (mistral-large-latest) attend 90s et retente automatiquement sur 429.
Les embeddings juges (utilisés par answer_relevancy) sont générés via l'API
Mistral (mistral-embed), comme le reste du pipeline RAG du projet.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from utils.config import MISTRAL_API_KEY

sys.stdout.reconfigure(encoding="utf-8")

DATASET_PATH = Path(__file__).parent / "ragas_dataset.json"
SCORE_THRESHOLD = 0.7
_RETRY_WAIT = 90
_MAX_RETRIES = 8

_ALL_METRICS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


class _MistralWithRateRetry(ChatMistralAI):
    """Réessaie automatiquement après 90s sur les erreurs 429.

    Corrige aussi `_combine_llm_outputs` : l'implémentation de langchain_mistralai
    plante avec un TypeError dès qu'on demande plusieurs générations (n>1, cas de
    AnswerRelevancy avec strictness=3) car l'API Mistral renvoie un sous-dict
    `prompt_tokens_details` dans `usage`, sur lequel le code fait `+=` sans
    vérifier le type. Ragas avale silencieusement cette exception et renvoie NaN
    pour la métrique concernée.
    """

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                continue
            token_usage = output.get("token_usage")
            if not token_usage:
                continue
            for k, v in token_usage.items():
                if isinstance(v, dict):
                    continue
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        for attempt in range(_MAX_RETRIES):
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < _MAX_RETRIES - 1:
                    print(f"\n[Rate limit 429] attente {_RETRY_WAIT}s (tentative {attempt + 1}/{_MAX_RETRIES - 1})...")
                    time.sleep(_RETRY_WAIT)
                else:
                    raise

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        for attempt in range(_MAX_RETRIES):
            try:
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < _MAX_RETRIES - 1:
                    print(f"\n[Rate limit 429] attente {_RETRY_WAIT}s (tentative {attempt + 1}/{_MAX_RETRIES - 1})...")
                    await asyncio.sleep(_RETRY_WAIT)
                else:
                    raise


def load_dataset() -> EvaluationDataset:
    with open(DATASET_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    missing = [s["question"] for s in samples if not s.get("answer") or not s.get("contexts")]
    if missing:
        raise SystemExit(
            f"{len(missing)} question(s) sans réponse/contexte dans {DATASET_PATH}.\n"
            "Lancez d'abord : uv run python utils/ans_cont_recup_ragas.py"
        )

    rows = [
        {
            "user_input": sample["question"],
            "response": sample["answer"],
            "retrieved_contexts": sample["contexts"],
            "reference": sample["ground_truth"],
        }
        for sample in samples
    ]
    return EvaluationDataset.from_list(rows)


def run_evaluation():
    dataset = load_dataset()

    # mistral-small-latest respecte mal le JSON strict attendu par RAGAS pour les
    # métriques à verdict (faithfulness/context_precision/context_recall), ce qui
    # fait échouer le parsing même après les retries de RAGAS et renvoie NaN.
    # mistral-large-latest est plus fiable sur ce point, donc on l'utilise pour
    # toutes les métriques juges.
    judge_llm = LangchainLLMWrapper(
        _MistralWithRateRetry(model="mistral-large-latest", temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(model="mistral-embed", api_key=MISTRAL_API_KEY)
    )

    faithfulness = Faithfulness()
    faithfulness.llm = judge_llm
    context_precision = ContextPrecision()
    context_precision.llm = judge_llm
    context_recall = ContextRecall()
    context_recall.llm = judge_llm
    answer_relevancy = AnswerRelevancy(embeddings=judge_embeddings)
    answer_relevancy.llm = judge_llm

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        run_config=RunConfig(max_workers=1, timeout=600),
    )
    return result.to_pandas()


def report(df) -> bool:
    cols = ["user_input"] + [m for m in _ALL_METRICS if m in df.columns]
    print(f"\n{'='*60}\nRapport RAGAS ({len(df)} questions)\n{'='*60}")
    print(df[cols].to_string(index=False))

    print("\n--- Moyennes ---")
    all_pass = True
    for m in _ALL_METRICS:
        if m not in df.columns:
            continue
        scores = df[m].dropna()
        if scores.empty:
            print(f"  {m:<25} N/A (toutes les valeurs sont NaN)")
            continue
        mean_score = scores.mean()
        passed = mean_score >= SCORE_THRESHOLD
        all_pass = all_pass and passed
        status = "OK" if passed else "ECHEC"
        print(f"  {m:<25} {mean_score:.3f}  [{status}]")
    print("=" * 60)
    return all_pass


def main() -> int:
    if not MISTRAL_API_KEY:
        print("MISTRAL_API_KEY non défini. Définissez-le dans le fichier .env.")
        return 1

    df = run_evaluation()
    success = report(df)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
