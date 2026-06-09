"""
Шаг индексации: читаем PDF учебника, режем на чанки, считаем эмбеддинги
и складываем в pgvector.

Запуск:
    python index_textbook.py

Скрипт идемпотентен в рамках примера: повторный запуск можно делать с флагом
--recreate, чтобы пересоздать таблицу с нуля (например, после смены эмбеддера
или размера чанка).
"""

import argparse
import sys

from agno.knowledge.chunking.fixed import FixedSizeChunking
from agno.knowledge.reader.pdf_reader import PDFReader

import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексация учебника в pgvector")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Пересоздать таблицу перед индексацией (удалит старые векторы)",
    )
    args = parser.parse_args()

    print(f"Эмбеддер:   {config.EMBEDDER}")
    print(f"Таблица:    {config.table_name()}")
    print(f"PDF:        {config.PDF_PATH}")
    print(f"Чанк:       {config.CHUNK_SIZE} символов, перекрытие {config.CHUNK_OVERLAP}")
    print("-" * 60)

    knowledge = config.build_knowledge()

    if args.recreate:
        # Очищаем содержимое перед повторной загрузкой.
        print("Пересоздаю таблицу (--recreate)...")
        knowledge.vector_db.drop()

    if not knowledge.vector_db.table_exists():
        knowledge.vector_db.create()

    # Чанкинг фиксированного размера с перекрытием.
    reader = PDFReader(
        chunking_strategy=FixedSizeChunking(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        ),
    )

    print("Читаю PDF, считаю эмбеддинги и пишу в pgvector...")
    print("(на полном учебнике это может занять несколько минут)")

    # insert() прочитает PDF, разобьёт на чанки, посчитает векторы и запишет
    # их в pgvector вместе с метаданными (имя файла, номер страницы).
    knowledge.insert(path=config.PDF_PATH, reader=reader)

    print("-" * 60)
    print("Готово. Индекс заполнен. Теперь можно запускать: python ask.py")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print(
            f"Не найден PDF: {config.PDF_PATH}\n"
            "Укажите путь через переменную окружения PDF_PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
