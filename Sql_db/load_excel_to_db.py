"""Charge le fichier inputs/regular_NBA.xlsx dans une base PostgreSQL.

Feuilles chargées :
- "Données NBA" -> table `player_stats` (stats saison régulière par joueur)
- "Equipe"      -> table `teams` (code équipe -> nom complet)

À chaque exécution, les tables sont recréées (DROP + CREATE) à partir du xlsx.
"""

import datetime
import re
import sys
from pathlib import Path

# Permet de lancer ce fichier directement (python Sql_db/load_excel_to_db.py) :
# sans ça, Python ajoute Sql_db/ à sys.path au lieu de la racine du projet, et
# `from utils.config import ...` échoue.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import create_engine

from utils.config import POSTGRES_URL

XLSX_FILE = Path(__file__).resolve().parent.parent / "inputs" / "regular_NBA.xlsx"

# Excel a mal interprété l'en-tête "3PM" comme une heure (15:00 = 3PM).
MISPARSED_TIME_COLUMN = datetime.time(15, 0)
MISPARSED_TIME_COLUMN_NAME = "3PM"


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("%", "_pct").replace("+/-", "plus_minus")
    name = name.replace("/", "_per_")
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = f"n{name}"  # un identifiant SQL ne doit pas commencer par un chiffre
    return name


def load_player_stats(xlsx_file: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_file, sheet_name="Données NBA", header=1)
    df = df.rename(columns={MISPARSED_TIME_COLUMN: MISPARSED_TIME_COLUMN_NAME})
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def load_teams(xlsx_file: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_file, sheet_name="Equipe")
    df.columns = ["team_code", "team_name"]
    return df


def main() -> None:
    if not POSTGRES_URL:
        sys.exit(
            "Configuration PostgreSQL incomplète : renseignez DB_HOST, DB_PORT, "
            "DB_NAME, DB_USER et DB_PWD dans le fichier .env."
        )

    try:
        player_stats = load_player_stats(XLSX_FILE)
        teams = load_teams(XLSX_FILE)
    except FileNotFoundError:
        sys.exit(f"Fichier introuvable : {XLSX_FILE}")

    engine = create_engine(POSTGRES_URL)

    teams.to_sql("teams", engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE teams ADD PRIMARY KEY (team_code)"
        )

    player_stats.to_sql("player_stats", engine, if_exists="replace", index=False)

    print(f"teams: {len(teams)} lignes insérées")
    print(f"player_stats: {len(player_stats)} lignes insérées")


if __name__ == "__main__":
    main()
