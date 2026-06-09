"""
Интерфейс к индексу через Agno Agent.

Агент сам решает, когда искать по учебнику (search_knowledge=True), достаёт
релевантные чанки из pgvector и отвечает на их основе со ссылками на источник.

Запуск с вопросом из командной строки:
    python ask.py "What is the difference between verification and validation?"

Запуск в интерактивном режиме (вопрос-ответ в цикле):
    python ask.py
"""

import os
import sys

from agno.agent import Agent

import config


def build_model():
    """
    Чат-модель для агента. По умолчанию совпадает с провайдером эмбеддингов,
    чтобы локальный сценарий (Ollama) работал без API-ключей.
    """
    provider = os.getenv("CHAT_PROVIDER", config.EMBEDDER).lower()

    if provider == "openai":
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id=os.getenv("CHAT_MODEL", "gpt-4o-mini"))

    from agno.models.ollama import Ollama

    return Ollama(id=os.getenv("CHAT_MODEL", "llama3.1"))


def build_agent() -> Agent:
    return Agent(
        model=build_model(),
        knowledge=config.build_knowledge(),
        # Даём агенту инструмент поиска по базе знаний и просим
        # опираться только на найденное.
        search_knowledge=True,
        instructions=[
            "Отвечай по учебнику Sommerville 'Software Engineering'.",
            "Используй поиск по базе знаний для каждого вопроса.",
            "Опирайся только на найденные фрагменты, не выдумывай.",
            "В конце ответа укажи страницы-источники, если они есть в метаданных.",
        ],
        markdown=True,
    )


def main() -> None:
    agent = build_agent()

    # Вопрос можно передать аргументом командной строки.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        agent.print_response(question, markdown=True)
        return

    # Иначе интерактивный режим.
    print("Спрашивайте про учебник Sommerville. Пустая строка или 'exit' для выхода.")
    while True:
        try:
            question = input("\nВопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        agent.print_response(question, markdown=True)


if __name__ == "__main__":
    main()
