# Assistant RAG avec Mistral

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🤖 **Génération de réponses** avec les modèles Mistral (Small ou Large)
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.9+ 
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer l'environnement virtuel et installer les dépendances**

```bash
uv sync
```

3. **Activer l'environnement virtuel** (optionnel, `uv run` s'en charge automatiquement)

```bash
# Sur Windows
.venv\Scripts\activate
# Sur macOS/Linux
source .venv/bin/activate
```

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
.
├── MistralChat.py          # Application Streamlit principale
├── indexer.py              # Script pour indexer les documents
├── inputs/                 # Dossier pour les documents sources
├── vector_db/              # Dossier pour l'index FAISS et les chunks
├── database/               # Base de données SQLite pour les interactions
└── utils/                  # Modules utilitaires
    ├── config.py           # Configuration de l'application
    ├── database.py         # Gestion de la base de données
    └── vector_store.py     # Gestion de l'index vectoriel

```

## Utilisation

### 1. Ajouter des documents

Placez vos documents dans le dossier `inputs/`. Les formats supportés sont :
- PDF
- TXT
- DOCX
- CSV
- JSON

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
python indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

### 3. Lancer l'application

```bash
streamlit run MistralChat.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.


## Modules principaux

### `utils/vector_store.py`

Gère l'index vectoriel FAISS et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Mistral
- Création et interrogation de l'index FAISS

### `utils/query_classifier.py`

Détermine si une requête nécessite une recherche RAG :
- Analyse des mots-clés
- Classification avec le modèle Mistral
- Détection des questions spécifiques vs générales

### `utils/database.py`

Gère la base de données SQLite pour les interactions :
- Enregistrement des questions et réponses
- Stockage des feedbacks utilisateurs
- Récupération des statistiques

### `utils/schemas.py`

Définit tous les modèles Pydantic du pipeline RAG.

## Pydantic & Pydantic AI dans le projet

Le pipeline RAG s'appuie sur l'écosystème Pydantic à chaque étape, du chargement des documents jusqu'à la génération de la réponse :

- **Validation des données à chaque frontière du pipeline** (`utils/schemas.py`) : un modèle dédié par étape — `RawDocument` → `CleanedDocument` → `Chunk` → `EmbeddedChunk` → `SearchResult` — remplace les `dict` non typés qui circulaient auparavant entre `data_loader.py`, `vector_store.py` et `chatbot.py`. Chaque modèle impose ses propres contraintes (ex. `page_content` non vide via `Field(min_length=1)`), ce qui fait échouer tôt et explicitement un document mal formé plutôt que de propager une erreur silencieuse plus loin dans le pipeline.
- **Échecs gérés sans interrompre tout le pipeline** : `build_cleaned_document()` (`utils/data_loader.py`) capture les `ValidationError` pour ignorer et logguer un document invalide sans stopper l'indexation des autres documents.
- **Validation des entrées utilisateur** : `RAGQuery` (`utils/schemas.py`) valide et nettoie la question posée (longueur, non-vide) avant qu'elle ne soit envoyée au modèle, dans `generate_answer()` (`utils/chatbot.py`).
- **Sortie structurée du LLM avec Pydantic AI** : le chatbot utilise `pydantic_ai.Agent(MistralModel(...), output_type=RAGAnswer)` (`utils/chatbot.py`) pour forcer le modèle Mistral à renvoyer une réponse conforme au schéma `RAGAnswer`, plutôt qu'un texte libre à parser manuellement.
- **Observabilité avec Pydantic Logfire** : `utils/observability.py` configure Logfire (`logfire.instrument_pydantic_ai()`) pour tracer automatiquement les appels de l'agent Pydantic AI ; `utils/chatbot.py` ajoute des spans (`search_context`, `generate_answer`, `ask_with_context`) pour suivre chaque étape du pipeline RAG en local, sans compte cloud.

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut

