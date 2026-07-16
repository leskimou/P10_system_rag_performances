# utils/observability.py
"""Configuration de Pydantic Logfire.

Envoie vers l'interface web Logfire si LOGFIRE_TOKEN est défini dans
l'environnement (.env), sinon reste entièrement local (console uniquement).
Un handler stdlib logging est aussi attaché pour que tous les
logging.info/warning/error déjà présents dans le projet (Sql_db/sql_tool.py,
utils/chatbot.py, utils/data_loader.py, utils/vector_store.py) remontent
automatiquement dans Logfire, sans modifier ces fichiers.
"""
import logging
import os

import logfire

_configured = False


def configure_logfire() -> None:
    """Configure Logfire (cloud si LOGFIRE_TOKEN présent, sinon local) et
    branche le logging standard dessus."""
    global _configured
    if _configured:
        return

    logfire.configure(
        token=os.getenv("LOGFIRE_TOKEN"),
        send_to_logfire="if-token-present",
    )
    logfire.instrument_pydantic_ai()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logfire.LogfireLoggingHandler())

    _configured = True
