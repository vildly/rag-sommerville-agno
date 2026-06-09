"""
Общая конфигурация примера: переключатель эмбеддингов (Ollama / OpenAI),
подключение к pgvector и сборка объекта Knowledge.

И index_textbook.py, и ask.py импортируют функции отсюда, чтобы индексация
и поиск всегда использовали один и тот же эмбеддер и одну и ту же таблицу.
"""

import os

from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

# --- Настройки через переменные окружения (см. .env.example) ------------------

# Какой провайдер эмбеддингов использовать: "ollama" (локально) или "openai".
EMBEDDER = os.getenv("EMBEDDER", "ollama").lower()

# Строка подключения к Postgres с расширением pgvector.
# Формат agno: postgresql+psycopg://<user>:<pass>@<host>:<port>/<db>
DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg://ai:ai@localhost:5532/ai",
)

# Путь к PDF учебника.
# По умолчанию ищем файл в самой папке проекта; если там нет — на уровень выше
# (для случая, когда репозиторий лежит внутри папки курса).
_here = os.path.dirname(os.path.abspath(__file__))
_pdf_name = "Software Engineering - Ian Sommerville.pdf"
_pdf_local = os.path.join(_here, _pdf_name)
_pdf_parent = os.path.join(os.path.dirname(_here), _pdf_name)
PDF_PATH = os.getenv(
    "PDF_PATH",
    _pdf_local if os.path.exists(_pdf_local) else _pdf_parent,
)

# Размер чанка и перекрытие (в символах, не токенах).
# Примерно 1 токен ~ 4 символа, поэтому 4000 символов ~ 1000 токенов.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "400"))


def build_embedder():
    """
    Возвращает объект эмбеддера в зависимости от EMBEDDER.

    Важно: у разных моделей разная размерность вектора
    (nomic-embed-text: 768, text-embedding-3-small: 1536),
    поэтому для каждого провайдера используется своя таблица в pgvector.
    """
    if EMBEDDER == "openai":
        from agno.knowledge.embedder.openai import OpenAIEmbedder

        # Требуется переменная окружения OPENAI_API_KEY.
        return OpenAIEmbedder(id="text-embedding-3-small")

    if EMBEDDER == "ollama":
        from agno.knowledge.embedder.ollama import OllamaEmbedder

        # Требуется запущенный Ollama и команда: ollama pull nomic-embed-text
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return OllamaEmbedder(id="nomic-embed-text", dimensions=768, host=host)

    raise ValueError(
        f"Неизвестный EMBEDDER={EMBEDDER!r}. Допустимо: 'ollama' или 'openai'."
    )


def table_name() -> str:
    """Имя таблицы зависит от провайдера, чтобы не смешивать векторы разной размерности."""
    return f"sommerville_{EMBEDDER}"


def build_knowledge() -> Knowledge:
    """
    Собирает Knowledge поверх pgvector с выбранным эмбеддером.

    Один и тот же объект используется и для записи (index_textbook.py),
    и для чтения (ask.py).
    """
    return Knowledge(
        vector_db=PgVector(
            table_name=table_name(),
            db_url=DB_URL,
            embedder=build_embedder(),
        ),
    )
