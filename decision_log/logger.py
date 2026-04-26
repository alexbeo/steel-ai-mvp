"""
Decision Log — структурированная память проекта.

Каждое техническое решение сохраняется с контекстом, альтернативами, reasoning.
Это компенсирует отсутствие persistent memory у LLM-агентов.

Через N месяцев вы (или новый агент) можете восстановить reasoning любого
решения, не поднимая его с нуля.

Использование:
    from decision_log.logger import log_decision, query_decisions
    
    log_decision(
        phase="training",
        decision="Использовать XGBoost с max_depth=5",
        alternatives_considered=["CatBoost", "MLP", "Random Forest"],
        reasoning="Датасет маленький (2100 записей). MLP переобучится. "
                  "XGBoost vs CatBoost — выбрали XGBoost из-за better community support "
                  "и проще интеграция с SHAP.",
        context={"dataset_size": 2100, "n_features": 24, "target": "yield_strength"},
        author="orchestrator",
    )
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).parent / "decisions.db"


def _init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            phase TEXT NOT NULL,
            decision TEXT NOT NULL,
            alternatives_considered TEXT,
            reasoning TEXT NOT NULL,
            context_json TEXT,
            author TEXT,
            outcome TEXT,
            revised_at TEXT,
            tags TEXT
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_phase ON decisions(phase);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON decisions(timestamp);")
    conn.commit()
    conn.close()


def log_decision(
    phase: str,
    decision: str,
    reasoning: str,
    alternatives_considered: list[str] | None = None,
    context: dict | None = None,
    author: str = "unknown",
    tags: list[str] | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """
    Записывает техническое решение в log. Возвращает ID записи.

    phase: data_acquisition | preprocessing | feature_engineering |
           training | inverse_design | validation | reporting | meta
    decision: короткое название решения
    reasoning: развёрнутое объяснение, почему именно так
    alternatives_considered: что рассматривали и отвергли
    context: dict с relevant metrics/params на момент решения
    author: orchestrator | data_curator | model_trainer | user | ...
    tags: ["xgboost", "overfitting_fix", "client_feedback", ...]
    """
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """
        INSERT INTO decisions
            (timestamp, phase, decision, alternatives_considered, reasoning, 
             context_json, author, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            phase,
            decision,
            json.dumps(alternatives_considered or [], ensure_ascii=False),
            reasoning,
            json.dumps(context or {}, ensure_ascii=False, default=str),
            author,
            json.dumps(tags or [], ensure_ascii=False),
        ),
    )
    decision_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return decision_id


def update_outcome(
    decision_id: int,
    outcome: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """
    Обновляет запись, отмечая результат решения после применения.
    Полезно для обучения: решение сработало / провалилось.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE decisions SET outcome = ?, revised_at = ?
        WHERE id = ?
        """,
        (outcome, datetime.utcnow().isoformat(), decision_id),
    )
    conn.commit()
    conn.close()


def query_decisions(
    phase: str | None = None,
    tag: str | None = None,
    keyword: str | None = None,
    limit: int = 50,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Поиск решений. Используется агентами, чтобы не переизобретать колесо.
    """
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM decisions WHERE 1=1"
    params = []
    if phase:
        query += " AND phase = ?"
        params.append(phase)
    if tag:
        query += " AND tags LIKE ?"
        params.append(f"%{tag}%")
    if keyword:
        query += " AND (decision LIKE ? OR reasoning LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    result = []
    for row in rows:
        d = dict(row)
        d["alternatives_considered"] = json.loads(d.get("alternatives_considered") or "[]")
        d["context"] = json.loads(d.get("context_json") or "{}")
        d["tags"] = json.loads(d.get("tags") or "[]")
        d.pop("context_json", None)
        result.append(d)
    return result


def get_decision_by_id(decision_id: int, db_path: Path = DEFAULT_DB_PATH) -> dict | None:
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["alternatives_considered"] = json.loads(d.get("alternatives_considered") or "[]")
    d["context"] = json.loads(d.get("context_json") or "{}")
    d["tags"] = json.loads(d.get("tags") or "[]")
    d.pop("context_json", None)
    return d


def summarize_project_history(db_path: Path = DEFAULT_DB_PATH) -> str:
    """
    Генерирует текстовое summary всех решений для передачи в LLM-контекст.
    Используется Orchestrator в начале каждой новой сессии.
    """
    decisions = query_decisions(limit=100, db_path=db_path)
    if not decisions:
        return "Decision log пуст. Это первая сессия работы над проектом."
    
    by_phase: dict[str, list[dict]] = {}
    for d in decisions:
        by_phase.setdefault(d["phase"], []).append(d)
    
    lines = [f"# История решений проекта ({len(decisions)} записей)\n"]
    for phase, items in sorted(by_phase.items()):
        lines.append(f"\n## Фаза: {phase} ({len(items)} решений)\n")
        for d in items[:5]:  # топ-5 по каждой фазе
            lines.append(f"- **{d['decision']}** ({d['timestamp'][:10]})")
            lines.append(f"  Reasoning: {d['reasoning'][:200]}")
            if d.get("outcome"):
                lines.append(f"  Outcome: {d['outcome']}")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = Path(f.name)
    
    id1 = log_decision(
        phase="training",
        decision="Использовать time-based split вместо random",
        alternatives_considered=["random_split", "stratified_kfold"],
        reasoning="В данных NIMS HSLA есть год выпуска. Random split создаст data leakage "
                  "через correlated heats одной кампании. Time-based split даёт честную оценку.",
        context={"dataset_size": 2847, "has_time_column": True},
        author="model_trainer",
        tags=["validation", "data_leakage"],
        db_path=test_db,
    )
    id2 = log_decision(
        phase="training",
        decision="XGBoost с max_depth=5",
        alternatives_considered=["max_depth=10", "max_depth=3"],
        reasoning="Optuna search показала, что max_depth=5 даёт стабильный R²=0.87. "
                  "max_depth=10 overfitted (train 0.99, val 0.71). max_depth=3 underfitted.",
        context={"optuna_best_r2": 0.87, "trials": 100},
        author="model_trainer",
        tags=["xgboost", "hyperparameters"],
        db_path=test_db,
    )
    update_outcome(id1, "Подтверждено: R² на hold-out 2025 = 0.83, стабильно", db_path=test_db)
    
    print("=== Все решения ===")
    for d in query_decisions(db_path=test_db):
        print(f"[{d['id']}] {d['phase']}: {d['decision']} (by {d['author']})")
        print(f"    outcome: {d.get('outcome') or 'not set'}")
    
    print("\n=== Поиск по тегу 'xgboost' ===")
    for d in query_decisions(tag="xgboost", db_path=test_db):
        print(f"- {d['decision']}")
    
    print("\n=== Summary для LLM-контекста ===")
    print(summarize_project_history(db_path=test_db))
    
    test_db.unlink()
