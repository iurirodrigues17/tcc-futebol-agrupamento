#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import sqlite3
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple, Iterable, Optional

EVENT_FIELDS = ["goal", "shoton", "shotoff", "foulcommit", "card", "cross", "corner", "possession"]

ALIASES_COMMON = {
    "event_index": ["idx", "index", "id", "_idx", "event.index"],
    "elapsed": ["elapsed", "minute", "min", "time", "event.time", "event.minute"],
    "elapsed_plus": ["elapsed_plus", "stoppage", "extra", "injurytime", "stoppage_time"],
    "team": ["team", "team_id", "teamapiid", "side", "team.side", "team_api_id"],
    "player1": ["player1", "player", "player_id", "playerid", "player.apiid", "player_api_id"],
    "player2": ["player2", "secondaryplayer", "secondary_player", "second_player", "assist2", "player2_id"],
    "assist": ["assist", "assist_id", "assist_player", "assistplayer"],
    "comment": ["comment", "note", "notes", "text", "msg"],
}
ALIASES_BY_TYPE: Dict[str, Dict[str, List[str]]] = {
    "goal": {
        **ALIASES_COMMON,
        "shot_place": ["shot.place", "shotplace", "shot_place", "place"],
        "shot_outcome": ["shot.outcome", "shotoutcome", "shot_outcome", "outcome"],
        "own_goal": ["owngoal", "own_goal"],
        "penalty": ["penalty", "is_penalty"],
    },
    "shoton": {
        **ALIASES_COMMON,
        "shot_place": ["shot.place", "shotplace", "shot_place", "place"],
        "shot_outcome": ["shot.outcome", "shotoutcome", "shot_outcome", "outcome"],
        "key_pass": ["keypass", "key_pass"],
    },
    "shotoff": {
        **ALIASES_COMMON,
        "shot_place": ["shot.place", "shotplace", "shot_place", "place"],
        "shot_outcome": ["shot.outcome", "shotoutcome", "shot_outcome", "outcome"],
        "key_pass": ["keypass", "key_pass"],
    },
    "foulcommit": {
        **ALIASES_COMMON,
        "card_type": ["cardtype", "card_type", "card.type"],
        "foul_type": ["foultype", "foul_type", "foul.type"],
        "penalty": ["penalty", "is_penalty"],
    },
    "card": {
        **ALIASES_COMMON,
        "card_type": ["cardtype", "card_type", "card.type", "type"],
    },
    "cross": {
        **ALIASES_COMMON,
        "outcome": ["outcome", "cross.outcome"],
    },
    "corner": {
        **ALIASES_COMMON,
        "outcome": ["outcome", "corner.outcome"],
    },
    "possession": {
        **ALIASES_COMMON,
        "possession": ["possession", "value", "pct", "possession.value"],
    },
}

DDL = r"""
BEGIN;

CREATE TABLE IF NOT EXISTS goal_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  event_index INTEGER,
  elapsed INTEGER,
  elapsed_plus INTEGER,
  team TEXT,
  player1 TEXT,
  player2 TEXT,
  assist TEXT,
  shot_place TEXT,
  shot_outcome TEXT,
  own_goal INTEGER,
  penalty INTEGER,
  comment TEXT,
  raw_tag TEXT,
  FOREIGN KEY (match_id) REFERENCES Match(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS goal_event_attr (
  event_id INTEGER NOT NULL,
  attr_key TEXT NOT NULL,
  attr_value TEXT,
  FOREIGN KEY (event_id) REFERENCES goal_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS shoton_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  event_index INTEGER,
  elapsed INTEGER,
  elapsed_plus INTEGER,
  team TEXT,
  player1 TEXT,
  player2 TEXT,
  shot_place TEXT,
  shot_outcome TEXT,
  key_pass INTEGER,
  comment TEXT,
  raw_tag TEXT,
  FOREIGN KEY (match_id) REFERENCES Match(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS shoton_event_attr (
  event_id INTEGER NOT NULL,
  attr_key TEXT NOT NULL,
  attr_value TEXT,
  FOREIGN KEY (event_id) REFERENCES shoton_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS shotoff_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  event_index INTEGER,
  elapsed INTEGER,
  elapsed_plus INTEGER,
  team TEXT,
  player1 TEXT,
  player2 TEXT,
  shot_place TEXT,
  shot_outcome TEXT,
  key_pass INTEGER,
  comment TEXT,
  raw_tag TEXT,
  FOREIGN KEY (match_id) REFERENCES Match(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS shotoff_event_attr (
  event_id INTEGER NOT NULL,
  attr_key TEXT NOT NULL,
  attr_value TEXT,
  FOREIGN KEY (event_id) REFERENCES shotoff_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS foulcommit_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  event_index INTEGER,
  elapsed INTEGER,
  elapsed_plus INTEGER,
  team TEXT,
  player1 TEXT,
  card_type TEXT,
  foul_type TEXT,
  penalty INTEGER,
  comment TEXT,
  raw_tag TEXT,
  FOREIGN KEY (match_id) REFERENCES Match(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS foulcommit_event_attr (
  event_id INTEGER NOT NULL,
  attr_key TEXT NOT NULL,
  attr_value TEXT,
  FOREIGN KEY (event_id) REFERENCES foulcommit_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS card_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  event_index INTEGER,
  elapsed INTEGER,
  elapsed_plus INTEGER,
  team TEXT,
  player1 TEXT,
  card_type TEXT,
  comment TEXT,
  raw_tag TEXT,
  FOREIGN KEY (match_id) REFERENCES Match(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS card_event_attr (
  event_id INTEGER NOT NULL,
  attr_key TEXT NOT NULL,
  attr_value TEXT,
  FOREIGN KEY (event_id) REFERENCES card_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cross_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  event_index INTEGER,
  elapsed INTEGER,
  elapsed_plus INTEGER,
  team TEXT,
  player1 TEXT,
  outcome TEXT,
  comment TEXT,
  raw_tag TEXT,
  FOREIGN KEY (match_id) REFERENCES Match(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cross_event_attr (
  event_id INTEGER NOT NULL,
  attr_key TEXT NOT NULL,
  attr_value TEXT,
  FOREIGN KEY (event_id) REFERENCES cross_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS corner_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  event_index INTEGER,
  elapsed INTEGER,
  elapsed_plus INTEGER,
  team TEXT,
  player1 TEXT,
  outcome TEXT,
  comment TEXT,
  raw_tag TEXT,
  FOREIGN KEY (match_id) REFERENCES Match(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS corner_event_attr (
  event_id INTEGER NOT NULL,
  attr_key TEXT NOT NULL,
  attr_value TEXT,
  FOREIGN KEY (event_id) REFERENCES corner_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS possession_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_id INTEGER NOT NULL,
  event_index INTEGER,
  elapsed INTEGER,
  elapsed_plus INTEGER,
  team TEXT,
  possession TEXT,
  comment TEXT,
  raw_tag TEXT,
  FOREIGN KEY (match_id) REFERENCES Match(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS possession_event_attr (
  event_id INTEGER NOT NULL,
  attr_key TEXT NOT NULL,
  attr_value TEXT,
  FOREIGN KEY (event_id) REFERENCES possession_events(id) ON DELETE CASCADE
);

COMMIT;
"""

def normalize_tag(tag: str) -> str:
    tag = tag.split('}')[-1]
    tag = tag.replace('-', '_')
    return tag.lower()

def flatten_xml(elem: ET.Element, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    for k, v in elem.attrib.items():
        out[f"{prefix}{normalize_tag(k)}"] = (v or "").strip()
    txt = (elem.text or "").strip()
    if txt and len(list(elem)) == 0:
        out[prefix[:-1] if prefix.endswith('.') else prefix] = txt
    for child in list(elem):
        tag = normalize_tag(child.tag)
        flatten_xml(child, prefix=f"{prefix}{tag}.", out=out)
    return out

def parse_fragment(xml_text: Optional[str]) -> List[ET.Element]:
    if not xml_text or not xml_text.strip():
        return []
    xml_text = xml_text.strip()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        try:
            root = ET.fromstring(f"<root>{xml_text}</root>")
        except ET.ParseError:
            return []
    nodes = list(root.iterfind('.//event'))
    if not nodes:
        nodes = list(root)
    return nodes

def pick(flat: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in flat and flat[k] not in (None, ""):
            return str(flat[k])
    return None

def to_bool(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "sim"):
        return 1
    if s in ("0", "false", "no", "n", "nao", "não"):
        return 0
    return None

def to_int(x: Optional[str]) -> Optional[int]:
    if x is None or str(x).strip() == "":
        return None
    try:
        return int(str(x).strip())
    except ValueError:
        m = re.search(r"-?\d+", str(x))
        return int(m.group(0)) if m else None

def build_core_and_attrs(flat: Dict[str, Any], evtype: str) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    aliases = ALIASES_BY_TYPE[evtype]
    core: Dict[str, Any] = {
        "event_index": to_int(pick(flat, aliases.get("event_index", []))),
        "elapsed": to_int(pick(flat, aliases.get("elapsed", []))),
        "elapsed_plus": to_int(pick(flat, aliases.get("elapsed_plus", []))),
        "team": pick(flat, aliases.get("team", [])),
        "player1": pick(flat, aliases.get("player1", [])),
        "player2": pick(flat, aliases.get("player2", [])),
        "assist": pick(flat, aliases.get("assist", [])),
        "comment": pick(flat, aliases.get("comment", [])),
    }

    if evtype in ("goal", "shoton", "shotoff"):
        core["shot_place"] = pick(flat, aliases.get("shot_place", []))
        core["shot_outcome"] = pick(flat, aliases.get("shot_outcome", []))
    if evtype in ("goal", "foulcommit"):
        core["penalty"] = to_bool(pick(flat, aliases.get("penalty", [])))
    if evtype == "goal":
        core["own_goal"] = to_bool(pick(flat, aliases.get("own_goal", [])))
    if evtype in ("shoton", "shotoff"):
        core["key_pass"] = to_bool(pick(flat, aliases.get("key_pass", [])))
    if evtype == "foulcommit":
        core["card_type"] = pick(flat, aliases.get("card_type", []))
        core["foul_type"] = pick(flat, aliases.get("foul_type", []))
    if evtype == "card":
        core["card_type"] = pick(flat, aliases.get("card_type", []))
    if evtype in ("cross", "corner"):
        core["outcome"] = pick(flat, aliases.get("outcome", []))
    if evtype == "possession":
        core["possession"] = pick(flat, aliases.get("possession", []))

    # atributos remanescentes
    used = set()
    for v in ALIASES_COMMON.values():
        used.update(v)
    for k in ("goal","shoton","shotoff","foulcommit","card","cross","corner","possession"):
        if k in ALIASES_BY_TYPE:
            for v in ALIASES_BY_TYPE[k].values():
                if isinstance(v, list):
                    used.update(v)
    attrs = [(k, str(v)) for k, v in flat.items() if k not in used and v is not None and str(v) != ""]

    return core, attrs

def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()

def insert_core_and_attrs(conn: sqlite3.Connection, evtype: str, match_id: int, raw_tag: str, core: Dict[str, Any], attrs: List[Tuple[str, str]]) -> None:
    cur = conn.cursor()
    table_map = {
        "goal": ("goal_events", "goal_event_attr"),
        "shoton": ("shoton_events", "shoton_event_attr"),
        "shotoff": ("shotoff_events", "shotoff_event_attr"),
        "foulcommit": ("foulcommit_events", "foulcommit_event_attr"),
        "card": ("card_events", "card_event_attr"),
        "cross": ("cross_events", "cross_event_attr"),
        "corner": ("corner_events", "corner_event_attr"),
        "possession": ("possession_events", "possession_event_attr"),
    }
    core_tbl, attr_tbl = table_map[evtype]

    # colunas core por tipo (somente as que EXISTEM no DDL daquele tipo)
    base_common = ["match_id", "event_index", "elapsed", "elapsed_plus", "team", "comment"]
    extra_common = {
        "goal": ["player1", "player2", "assist", "shot_place", "shot_outcome", "own_goal", "penalty"],
        "shoton": ["player1", "player2", "shot_place", "shot_outcome", "key_pass"],
        "shotoff": ["player1", "player2", "shot_place", "shot_outcome", "key_pass"],
        "foulcommit": ["player1", "card_type", "foul_type", "penalty"],
        "card": ["player1", "card_type"],
        "cross": ["player1", "outcome"],
        "corner": ["player1", "outcome"],
        "possession": ["possession"],  # sem player1/player2
    }
    cols = base_common + extra_common[evtype] + ["raw_tag"]

    # montar lista de valores conforme cols
    value_map = {
        "match_id": match_id,
        "event_index": core.get("event_index"),
        "elapsed": core.get("elapsed"),
        "elapsed_plus": core.get("elapsed_plus"),
        "team": core.get("team"),
        "player1": core.get("player1"),
        "player2": core.get("player2"),
        "assist": core.get("assist"),
        "shot_place": core.get("shot_place"),
        "shot_outcome": core.get("shot_outcome"),
        "own_goal": core.get("own_goal"),
        "penalty": core.get("penalty"),
        "key_pass": core.get("key_pass"),
        "card_type": core.get("card_type"),
        "foul_type": core.get("foul_type"),
        "outcome": core.get("outcome"),
        "possession": core.get("possession"),
        "comment": core.get("comment"),
        "raw_tag": raw_tag,
    }
    values = [ value_map[c] for c in cols ]

    placeholders = ",".join(["?"] * len(values))
    cur.execute(f"INSERT INTO {core_tbl} ({','.join(cols)}) VALUES ({placeholders});", values)
    event_id = cur.lastrowid

    if attrs:
        cur.executemany(
            f"INSERT INTO {attr_tbl} (event_id, attr_key, attr_value) VALUES (?,?,?);",
            [(event_id, k, v) for (k, v) in attrs]
        )

def run_etl(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_schema(conn)

    BATCH = 2000
    offset = 0
    cols = "id, goal, shoton, shotoff, foulcommit, card, cross, corner, possession"

    while True:
        rows = conn.execute(f"SELECT {cols} FROM Match LIMIT {BATCH} OFFSET {offset};").fetchall()
        if not rows:
            break
        for r in rows:
            match_id = r[0]
            xml_map = {
                "goal": r[1],
                "shoton": r[2],
                "shotoff": r[3],
                "foulcommit": r[4],
                "card": r[5],
                "cross": r[6],
                "corner": r[7],
                "possession": r[8],
            }
            for evtype, xml_text in xml_map.items():
                nodes = parse_fragment(xml_text)
                for node in nodes:
                    flat = flatten_xml(node, prefix="")
                    core, attrs = build_core_and_attrs(flat, evtype)
                    raw_tag = ET.tostring(node, encoding="unicode")
                    insert_core_and_attrs(conn, evtype, match_id, raw_tag, core, attrs)
        conn.commit()
        offset += BATCH

    conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python etl_events_core_attr_v2.py /caminho/para/database.sqlite")
        sys.exit(1)
    run_etl(sys.argv[1])
