# utils/observability.py
"""Configuration de Pydantic Logfire (mode console locale, sans compte cloud)."""
import logfire

_configured = False


def configure_logfire() -> None:
    """Configure Logfire pour tracer la chaîne RAG/LLM dans la console.

    Aucun token n'est requis : send_to_logfire=False garde tout en local.
    """
    global _configured
    if _configured:
        return
    logfire.configure(send_to_logfire=False)
    logfire.instrument_pydantic_ai()
    _configured = True
