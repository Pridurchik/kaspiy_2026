"""Хранилище платформы (SQLite): алерты всех источников, решения инспектора, слои обзора."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

PRIO_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, type TEXT, category TEXT, priority TEXT, risk INTEGER,
                status TEXT, start_ts TEXT, created REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS feedback (alert_id TEXT, status TEXT, note TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
            """)

    def reset(self):
        with self.lock:
            self.db.executescript("DELETE FROM alerts; DELETE FROM feedback; DELETE FROM kv;")
            self.db.commit()

    def upsert_alert(self, a: dict) -> dict:
        f = a.get("fusion") or {}
        a.setdefault("status", "new")
        a.setdefault("received_at", time.time())
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?)",
                            (a["id"], a.get("type", "acoustic"), a["category"], f.get("priority", a.get("severity", "medium")),
                             int(f.get("risk") or 0), a["status"], a.get("start_ts"), a["received_at"],
                             json.dumps(a, ensure_ascii=False, default=float)))
            self.db.commit()
        return a

    def alerts(self, status: str | None = None, limit: int = 200, visible_only: bool = True) -> list[dict]:
        q, args = "SELECT payload FROM alerts", []
        if status:
            q += " WHERE status = ?"
            args.append(status)
        with self.lock:
            rows = [json.loads(r[0]) for r in self.db.execute(q, args)]
        if visible_only:
            rows = [r for r in rows if not r.get("hidden")]
        rows.sort(key=lambda a: (a["status"] in ("closed", "false_alarm"), PRIO_ORDER.get((a.get("fusion") or {}).get("priority"), 5),
                                 -int((a.get("fusion") or {}).get("risk") or 0), a.get("start_ts") or ""))
        return rows[:limit]

    def alert(self, aid: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT payload FROM alerts WHERE id = ?", (aid,)).fetchone()
        return json.loads(r[0]) if r else None

    def set_status(self, aid: str, status: str, note: str = "") -> dict | None:
        a = self.alert(aid)
        if not a:
            return None
        a["status"] = status
        a.setdefault("history", []).append({"status": status, "note": note, "ts": time.time()})
        self.upsert_alert(a)
        with self.lock:
            self.db.execute("INSERT INTO feedback VALUES (?,?,?,?)", (aid, status, note, time.time()))
            self.db.commit()
        return a

    def feedback_stats(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT status, COUNT(*) FROM feedback GROUP BY status").fetchall()
        return dict(rows)

    def put(self, key: str, value) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False, default=float)))
            self.db.commit()

    def get(self, key: str, default=None):
        with self.lock:
            r = self.db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(r[0]) if r else default
