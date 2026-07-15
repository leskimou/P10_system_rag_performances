.DEFAULT_GOAL := help
.PHONY: help install run index load-db ragas-dataset ragas-eval ragas-refresh test clean

help: ## Affiche la liste des commandes disponibles
	@echo "Commandes disponibles :"
	@echo "  make install       Installe les dependances du projet (uv sync)"
	@echo "  make run           Lance l'application Streamlit (MistralChat.py)"
	@echo "  make index         Indexe les documents du dossier inputs/ dans FAISS"
	@echo "  make load-db       Charge regular_NBA.xlsx dans la base PostgreSQL"
	@echo "  make ragas-dataset Genere les reponses/contextes du dataset RAGAS"
	@echo "  make ragas-eval    Evalue le pipeline RAG avec RAGAS"
	@echo "  make ragas-refresh Reindexe, regenere le dataset RAGAS et relance l'evaluation"
	@echo "  make test          Lance tous les tests (unitaires, fonctionnels, integration)"
	@echo "  make clean         Supprime les caches Python (__pycache__)"

install: ## Installe les dependances via uv
	uv sync

run: ## Lance l'application Streamlit
	uv run streamlit run MistralChat.py

index: ## Indexe les documents dans le vector store FAISS
	uv run python indexer.py

load-db: ## Charge le fichier Excel NBA dans la base PostgreSQL
	uv run python Sql_db/load_excel_to_db.py

ragas-dataset: ## Regenere les reponses/contextes du dataset RAGAS
	uv run python ragas_part/ans_cont_recup_ragas.py

ragas-eval: ## Lance l'evaluation RAGAS du pipeline RAG
	uv run python ragas_part/evaluate_ragas.py

ragas-refresh: index ragas-dataset ragas-eval ## Reindexe, regenere le dataset RAGAS et relance l'evaluation

test: ## Lance tous les tests (unitaires, fonctionnels, integration)
	uv run pytest tests/ -v

clean: ## Nettoie les caches Python
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
