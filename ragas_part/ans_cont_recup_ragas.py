"""Remplit ragas_part/ragas_dataset.json avec les réponses et contextes générés par le
chatbot pour chaque question, afin d'alimenter evaluate_ragas.py sans refaire les
appels au modèle à chaque évaluation.

Lancer avec :

    uv run python ragas_part/ans_cont_recup_ragas.py
"""
import json
import sys
import time
from pathlib import Path

# Permet de lancer ce fichier directement (uv run python ragas_part/ans_cont_recup_ragas.py) :
# sans ça, Python ajoute ragas_part/ à sys.path au lieu de la racine du projet, et
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.chatbot import ask_with_context

sys.stdout.reconfigure(encoding="utf-8")

DATASET_PATH = Path(__file__).parent / "ragas_dataset.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    samples = json.load(f)

total = len(samples)

for i, sample in enumerate(samples, start=1):
    print(f"\n[{i}/{total}] Question : {sample['question']}")
    answer, contexts = ask_with_context(sample["question"])
    sample["answer"] = answer
    sample["contexts"] = contexts

    print(f"Réponse : {answer}")
    print("Contextes récupérés :")
    for j, ctx in enumerate(contexts, 1):
        print(f"  [{j}] {ctx}")

    # Sauvegarde après chaque question pour ne rien perdre en cas de quota dépassé
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    if i < total:
        if i % 4 == 0:
            print("Pause de 1m30 entre les blocs de 4 questions (limite API Mistral)...")
            time.sleep(90)
        else:
            time.sleep(2)  # limite le débit d'appels à l'API Mistral
