# Путь A: vector RAG по учебнику Sommerville (на agno)

Минимальный рабочий пример из лекции про AI-агентов: берём PDF учебника, режем
на чанки, векторизуем, складываем в pgvector и спрашиваем через Agno Agent.

Реализованы все пункты чеклиста, кроме отдельного MCP-сервера: интерфейс к
индексу сделан напрямую через Agno Agent (он сам ходит в базу знаний).

Что показывает пример:

1. чанкинг PDF и эмбеддинги (на выбор: локально через Ollama или через OpenAI API);
2. хранение векторов в Postgres с расширением pgvector;
3. поиск и ответы через Agno Agent со ссылкой на источник.

Проверено на agno 2.6.9.

## Что понадобится

- Python 3.10+ (в примере проверялось на 3.10 и 3.12).
- Docker (для запуска pgvector одной командой).
- Для локального варианта: установленный Ollama (https://ollama.com).
- Для варианта с OpenAI: ключ `OPENAI_API_KEY`.

## Шаг 0: окружение

```bash
cd rag-sommerville-agno
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

PDF учебника по умолчанию берётся из папки курса (на уровень выше):
`Software Engineering - Ian Sommerville.pdf`. Свой PDF можно указать через
`PDF_PATH` в `.env`.

## Шаг 1: поднять pgvector

```bash
docker compose up -d
```

Поднимется Postgres с pgvector на порту `5532` (чтобы не конфликтовать с
локальным Postgres на `5432`). Строка подключения уже прописана в `.env`.

## Шаг 2: выбрать эмбеддинги

Вариант по умолчанию: локально через Ollama. Один раз скачиваем модель:

```bash
ollama pull nomic-embed-text
```

Чтобы переключиться на OpenAI, в `.env` поставьте:

```
EMBEDDER=openai
OPENAI_API_KEY=sk-...
```

Для каждого провайдера создаётся своя таблица (`sommerville_ollama` или
`sommerville_openai`), потому что у моделей разная размерность вектора
(nomic-embed-text: 768, text-embedding-3-small: 1536). Менять провайдера
можно без конфликтов, индексы не пересекаются.

## Шаг 3: проиндексировать учебник

```bash
# подгрузить переменные из .env
export $(grep -v '^#' .env | xargs)

python index_textbook.py
```

На полном учебнике индексация занимает несколько минут (зависит от машины и
провайдера эмбеддингов). Чтобы пересоздать таблицу с нуля, например после смены
размера чанка:

```bash
python index_textbook.py --recreate
```

## Шаг 4: спросить

Одним вопросом из командной строки:

```bash
python ask.py "What is the difference between verification and validation?"
```

Или в интерактивном режиме:

```bash
python ask.py
```

Агент сам решает искать по учебнику (`search_knowledge=True`), достаёт топ
релевантных чанков из pgvector и отвечает на их основе.

## Как это устроено

Три файла, всё общее вынесено в `config.py`:

- `config.py`: переключатель эмбеддингов, подключение к pgvector, сборка
  объекта `Knowledge`. И индексация, и поиск используют одну и ту же
  конфигурацию, поэтому таблица и эмбеддер всегда согласованы.
- `index_textbook.py`: чтение PDF, чанкинг с перекрытием (`FixedSizeChunking`),
  запись векторов в pgvector через `Knowledge.insert`.
- `ask.py`: `Agent` поверх `Knowledge`, демо-вопросы и интерактивный режим.

Размер чанка задаётся в символах (`CHUNK_SIZE`, `CHUNK_OVERLAP` в `.env`).
Ориентир: примерно 4 символа на токен, то есть 4000 символов это около
1000 токенов.

## Куда расти

- Заменить `FixedSizeChunking` на `RecursiveChunking` или `SemanticChunking`
  (модули `agno.knowledge.chunking.recursive` и `.semantic`) и сравнить выдачу.
- Поднять pgvector в облаке или перейти на Qdrant для прод-нагрузки.
- Обернуть тот же индекс тонким MCP-сервером с инструментом
  `search_textbook(query, k=5)`: пункт из чеклиста, который здесь намеренно
  заменён на прямой Agno Agent. Логика поиска переиспользуется один в один.

## Если что-то не работает

- `connection refused` на 5532: контейнер pgvector не поднялся, проверьте
  `docker compose ps` и `docker compose logs pgvector`.
- Ollama: убедитесь, что сервис запущен (`ollama list`) и модель скачана
  (`ollama pull nomic-embed-text`).
- Размерность вектора не совпала: вы сменили эмбеддер, но пишете в старую
  таблицу. Запустите `python index_textbook.py --recreate`.
- Не найден PDF: задайте абсолютный путь в `PDF_PATH`.
