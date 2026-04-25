# Prompts (proprietary know-how)

В этой папке живут LLM system-prompts, которые драйвят Claude-based фичи проекта (LLM-Critic, Hypothesis Generator, и любые будущие). Они — интеллектуальная собственность проекта и **gitignored**: сами `.md` файлы в репо не попадают, только этот README.

## Архитектура

- Каждый prompt — `<feature-name>.md` (kebab-case).
- Имя файла маппится 1-в-1 на Python-модуль, который его подгружает через
  `app.backend.prompt_loader.load_prompt(name)`.
- Полное содержимое файла идёт в Claude `system` параметр с включённым
  prompt caching (ephemeral cache).

## Какие prompts должны быть

Если вы свежеклонировали repo, эти файлы нужны локально:

- `hypothesis_generator.md` — для `app/backend/hypothesis_generator.py`
- `llm_critic.md` — для `app/backend/critic_llm.py`

## Как получить

Связаться с владельцем проекта. Намерение: код открытый, prompts — know-how.

## Формат

Plain Markdown. Заголовки/секции опциональны, frontmatter не нужен.
Содержимое целиком уходит в Claude как system message.

## Версионирование локально

Можно держать `prompts/hypothesis_generator.v1.md`, `.v2.md` и т.п. для A/B,
loader работает по точному имени `<name>.md`, остальные просто игнорируются.
