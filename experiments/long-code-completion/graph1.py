#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_block_graph_viz_modified.py

在原脚本基础上增加：
1) 依赖方向统一为：dot 中 a -> b 表示 a 是被依赖方，b 是依赖方
2) 统计每个语义块的被依赖数 used_by_total
3) 统计每个语义块的依赖数 depends_on_total
4) 在标题、JSON、日志里同时输出这两类统计
"""

from __future__ import annotations

import ast
import html
import json
import re
import sys
import textwrap
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

try:
    import pydot  # type: ignore
except Exception:
    pydot = None

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import split_code as semantic_splitter
except Exception as _import_err:
    semantic_splitter = None
    _SPLIT_IMPORT_ERROR = _import_err
else:
    _SPLIT_IMPORT_ERROR = None


# =========================================================
# Config
# =========================================================

USE_INLINE_SOURCE = True
SOURCE_FILE_PATH = r""
FUNCTION_NAME = "update"
JOERN_HOME = r"/home/zhangmanqing/wyh/joern-cli"
WORK_ROOT = Path(".semantic_block_viz")
OUT_DIR = None

N_COLS = 6
BLOCK_BASE_WIDTH = 12.2
X_GAP = 1.0
Y_GAP = 0.72

DRAW_NODE_CODE_SNIPPETS = True
DRAW_ONLY_CROSS_BLOCK_EDGES = True
DRAW_EDGE_COUNT_LABELS = False
SHOW_BLOCK_OVERLAY_STATS = False

DEFAULT_BLOCK_PPLS: List[float] = [
    0.94, 1.31, 1.31, 1.31, 0.62, 0.75, 2.00, 1.50, 1.75, 1.31,
    0.00, 2.12, 1.75, 1.06, 1.62, 2.12, -0.19, 0.12, 0.12, 2.00,
    1.62, 2.25, 2.12, 2.38, 0.12, 1.88, 1.88, 0.75, 1.31, 1.31,
    1.31, 0.62, 1.75, 1.88, 2.00, 1.31, 2.50, 2.50, 2.00, 1.75,
    1.19,
]

SOURCE_CODE = """
def update(self, full=False):

    # Marker
    tStart = time.time()
    
    # The current time is further adjusted by the 'timeOffset' attribute
    # to help with tracking.
    tNow = datetime.utcnow() + self.timeOffset
    
    # Update the observer with the current time
    self.observer.date = tNow.strftime("%Y/%m/%d %H:%M:%S.%f")
    
    # Loop through the satellites and update them
    for tier,sat in zip(self.tiers, self.satellites):
        if not full:
            if tier != self.currentTier and sat.catalog_number != self.tracking:
                continue
                
        sat.compute(self.observer)
        
        if sat.alt > 0:
            ## If the satellite is up, check and see if it is 
            ## the one that we should be tracking.
            if sat.catalog_number == self.tracking:
                ### Is there a telescope to use?
                if self.telescope is not None:
                    #### Apply a perpendicular correction to the
                    #### track to help with tracking
                    ra, dec = getPointFromBearing(sat.ra, sat.dec, sat.bearing+math.pi/2, self.crossTrackOffset)
                    
                    #### Radians -> hours/degrees
                    ra = ra*_rad2hr
                    dec = dec*_rad2deg
                    
                    #### Command the telescope
                    self.telescope.moveToPosition(ra, dec, blocking=False)
                    
        else:
            ## If it is no longer visible, check and see if it is
            ## the satellite that we were tracking so that we can
            ## stop.
            if sat.catalog_number == self.tracking:
                self.stopTracking()
                
    # Final time to figure out how much time we spent calculating 
    # positions.
    tStop = time.time()
    
    # Update the tier being computed
    self.currentTier = (self.currentTier + 1) % self.nTiers
    
    # Done
    return tStop-tStart
"""

DATA_EDGE_HINTS = (
    "ddg", "data dependency", "data-dependency", "data dependence", "data-dependence",
    "reaching_def", "reaching def", "def-use", "def use", "use-def", "use def", "dataflow",
)
CONTROL_EDGE_HINTS = (
    "cdg", "control dependency", "control-dependency", "control dependence",
    "control-dependence", "cfg", "control flow", "branch",
)


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def normalize_source(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def dedent_source_text(text: str) -> str:
    text = normalize_source(text)
    return textwrap.dedent(text).lstrip("\n").rstrip() + "\n"


def load_source_text() -> str:
    if USE_INLINE_SOURCE:
        return dedent_source_text(SOURCE_CODE)
    if not SOURCE_FILE_PATH.strip():
        raise ValueError("SOURCE_FILE_PATH is empty and USE_INLINE_SOURCE is False.")
    path = Path(SOURCE_FILE_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    return dedent_source_text(path.read_text(encoding="utf-8", errors="ignore"))


def is_code_line(text: str) -> bool:
    s = text.strip()
    return bool(s) and not s.startswith("#")


def normalize_spaces(text: str) -> str:
    text = text.replace("\t", "    ")
    text = re.sub(r"[ \f\v]+", " ", text)
    return text.strip()


def wrap_code_text(text: str, width: int = 28) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return [""]
    out: List[str] = []
    for part in text.split("\n"):
        if not part:
            out.append("")
            continue
        chunks = textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False)
        out.extend(chunks or [part])
    return out or [""]


def unique_keep_order(seq: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for x in seq:
        key = x if isinstance(x, (str, int, float, tuple)) else repr(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def dot_attr_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return dot_attr_to_str(value[0])
        return " ".join(dot_attr_to_str(x) for x in value if x is not None)
    s = str(value)
    s = html.unescape(s)
    s = s.replace("\\l", "\n").replace("\\n", "\n").replace("\\r", "\n")
    s = s.replace("<BR/>", "\n").replace("<BR />", "\n").replace("<BR>", "\n")
    s = s.replace('\\"', '"')
    return s.strip()


def normalize_graph_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in attrs.items():
        if isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        if isinstance(v, (list, tuple)):
            v = [dot_attr_to_str(x) for x in v]
        else:
            v = dot_attr_to_str(v)
        out[str(k)] = v
    return out


def parse_label(raw_label: str) -> Tuple[str, Optional[int], str]:
    if not raw_label:
        return "", None, ""
    s = dot_attr_to_str(raw_label).strip()
    s = strip_quotes(s)
    s = s.strip("<>").strip()
    if "\n" in s:
        first, rest = s.split("\n", 1)
    else:
        first, rest = s, ""

    m = re.match(r"^\s*(?P<kind>[^,]+)\s*,\s*(?P<line>\d+)\s*(?P<tail>.*)$", first)
    if m:
        kind = m.group("kind").strip()
        line_no = int(m.group("line"))
        tail = m.group("tail").strip()
        code = normalize_spaces((tail + " " + rest).strip())
        return kind, line_no, code

    m2 = re.match(r"^\s*(?P<kind>[^,]+)\s*,\s*(?P<line>\d+)\s*$", first)
    if m2:
        kind = m2.group("kind").strip()
        line_no = int(m2.group("line"))
        return kind, line_no, normalize_spaces(rest)

    return first.strip(), None, normalize_spaces(rest)


def edge_text(attrs: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("label", "kind", "type", "edgeType", "relation", "rel", "name", "tag"):
        if key in attrs and attrs.get(key) not in (None, ""):
            parts.append(dot_attr_to_str(attrs.get(key)))
    if "attrs" in attrs and isinstance(attrs["attrs"], dict):
        inner = attrs["attrs"]
        for key in ("label", "kind", "type", "edgeType", "relation", "rel", "name", "tag"):
            if key in inner and inner.get(key) not in (None, ""):
                parts.append(dot_attr_to_str(inner.get(key)))
    if not parts and attrs:
        parts.append(" ".join(f"{k}={dot_attr_to_str(v)}" for k, v in attrs.items() if v not in (None, "")))
    return " | ".join(x for x in parts if x)


def is_data_dependency_edge(edata: Dict[str, Any]) -> bool:
    text = edge_text(edata).lower()
    if any(tok in text for tok in CONTROL_EDGE_HINTS):
        return False
    return any(tok in text for tok in DATA_EDGE_HINTS) or "ddg" in text


def is_control_dependency_edge(edata: Dict[str, Any]) -> bool:
    text = edge_text(edata).lower()
    if any(tok in text for tok in DATA_EDGE_HINTS):
        return False
    return any(tok in text for tok in CONTROL_EDGE_HINTS) or "cdg" in text


def choose_best_font() -> str:
    candidates = [
        "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimHei",
        "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return "DejaVu Sans"


FONT_FAMILY = choose_best_font()
plt.rcParams["font.sans-serif"] = [FONT_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# split_code.py interface
# =========================================================

def require_splitter() -> Any:
    if semantic_splitter is None:
        raise ImportError(f"Cannot import split_code.py: {_SPLIT_IMPORT_ERROR}")
    required = [
        "parse_python_ast",
        "function_span",
        "build_stmt_infos_for_function",
        "generate_pdg_with_joern",
        "choose_candidate_graphs",
        "merge_graphs",
        "build_line_graph_from_merged_pdg",
        "build_semantic_blocks",
    ]
    missing = [name for name in required if not hasattr(semantic_splitter, name)]
    if missing:
        raise AttributeError(f"split_code.py is missing required interfaces: {missing}")
    return semantic_splitter


def parse_source_for_splitter(source_text: str) -> Tuple[str, ast.AST, Any]:
    parsed_source = dedent_source_text(source_text)
    splitter = require_splitter()
    tree = splitter.parse_python_ast(parsed_source)
    return parsed_source, tree, splitter


def detect_function_node(tree: ast.AST, function_name: str) -> ast.AST:
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node, "name", None) == function_name:
            return node
    raise ValueError(f"Function {function_name!r} not found in source.")


# =========================================================
# DOT parsing
# =========================================================

def infer_line_no_from_node_attrs(data: Dict[str, Any]) -> Optional[int]:
    if isinstance(data.get("line_no"), int):
        return int(data["line_no"])
    if isinstance(data.get("lineNo"), int):
        return int(data["lineNo"])
    for key in ("line", "lineno", "line_number", "lineNumber", "start_line", "startLine"):
        v = data.get(key)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())

    raw = str(data.get("raw_label") or data.get("label") or data.get("code") or "")
    _, line_no, _ = parse_label(raw)
    if isinstance(line_no, int):
        return line_no

    for key in ("raw_label", "label", "code"):
        s = str(data.get(key) or "")
        m = re.search(r"(?<!\d)(\d{1,5})(?!\d)", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def parse_dot_file(dot_file: Path) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["dot_file"] = str(dot_file)
    graph.graph["dot_name"] = dot_file.name

    if pydot is not None:
        pydot_graphs = pydot.graph_from_dot_file(str(dot_file))
        if pydot_graphs:
            for pg in pydot_graphs:
                nxg = nx.drawing.nx_pydot.from_pydot(pg)
                if not nxg.is_multigraph():
                    tmp = nx.MultiDiGraph()
                    tmp.add_nodes_from(nxg.nodes(data=True))
                    tmp.add_edges_from((u, v, dict(d)) for u, v, d in nxg.edges(data=True))
                    nxg = tmp

                for nid, data in nxg.nodes(data=True):
                    nid = str(nid).strip().strip('"')
                    attrs = normalize_graph_attrs(dict(data))
                    label = attrs.get("label", "")
                    kind, line_no, code = parse_label(label)
                    node_attr = {
                        "raw_label": label,
                        "kind": attrs.get("kind", kind),
                        "line_no": line_no if isinstance(line_no, int) else None,
                        "code": attrs.get("code", code),
                        "attrs": attrs,
                        "dot_file": dot_file.name,
                    }
                    if graph.has_node(nid):
                        cur = graph.nodes[nid]
                        for k, v in node_attr.items():
                            if k == "attrs" and isinstance(v, dict):
                                cur.setdefault("attrs", {})
                                cur["attrs"].update(v)
                            else:
                                cur[k] = v if k not in cur or cur.get(k) in (None, "", [], {}) else cur[k]
                        cur.setdefault("dot_files", [])
                        if dot_file.name not in cur["dot_files"]:
                            cur["dot_files"].append(dot_file.name)
                    else:
                        node_attr["dot_files"] = [dot_file.name]
                        graph.add_node(nid, **node_attr)

                for u, v, k, data in nxg.edges(keys=True, data=True):
                    su = str(u).strip().strip('"')
                    sv = str(v).strip().strip('"')
                    attrs = normalize_graph_attrs(dict(data))
                    label = attrs.get("label", "") or attrs.get("taillabel", "") or attrs.get("headlabel", "")
                    graph.add_edge(su, sv, key=f"{dot_file.name}:{k}", label=label, attrs=attrs, dot_file=dot_file.name)
            return graph

    node_re = re.compile(r'^\s*"?(?P<id>[^"]+?)"?\s*\[(?P<attrs>.*)\]\s*;?\s*$')
    edge_re = re.compile(r'^\s*"?(?P<src>[^"\s]+)"?\s*->\s*"?(?P<dst>[^"\s]+)"?(?:\s*\[(?P<attrs>.*)\])?\s*;?\s*$')
    text = dot_file.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//") or line.startswith("/*"):
            continue
        if line.startswith("digraph") or line in {"{", "}"}:
            continue
        if line.startswith("graph [") or line.startswith("node [") or line.startswith("edge [") or line.startswith("subgraph "):
            continue

        nm = node_re.match(raw_line)
        if nm:
            nid = nm.group("id").strip().strip('"')
            attrs_raw = nm.group("attrs")
            label = ""
            m = re.search(r'\blabel\s*=\s*(.+)$', attrs_raw)
            if m:
                label = m.group(1).strip().rstrip(";")
            kind, line_no, code = parse_label(label)
            graph.add_node(
                nid,
                raw_label=label,
                kind=kind,
                line_no=line_no,
                code=code,
                attrs={},
                dot_file=dot_file.name,
                dot_files=[dot_file.name],
            )
            continue

        em = edge_re.match(raw_line)
        if em:
            src = em.group("src").strip().strip('"')
            dst = em.group("dst").strip().strip('"')
            attrs = em.group("attrs") or ""
            label = ""
            m = re.search(r'\blabel\s*=\s*(.+)$', attrs)
            if m:
                label = m.group(1).strip().rstrip(";")
            graph.add_edge(src, dst, key=f"{dot_file.name}:{graph.number_of_edges()}", label=label, attrs={}, dot_file=dot_file.name)

    return graph


def merge_raw_graphs(graphs: List[nx.MultiDiGraph]) -> nx.MultiDiGraph:
    merged = nx.MultiDiGraph()
    for G in graphs:
        for nid, data in G.nodes(data=True):
            if merged.has_node(nid):
                cur = merged.nodes[nid]
                for k, v in data.items():
                    if k == "dot_files":
                        cur.setdefault("dot_files", [])
                        for item in v if isinstance(v, list) else [v]:
                            if item not in cur["dot_files"]:
                                cur["dot_files"].append(item)
                    elif k == "attrs" and isinstance(v, dict):
                        cur.setdefault("attrs", {})
                        cur["attrs"].update(v)
                    else:
                        if cur.get(k) in (None, "", [], {}):
                            cur[k] = v
            else:
                merged.add_node(nid, **dict(data))
        for u, v, key, data in G.edges(keys=True, data=True):
            merged.add_edge(u, v, key=f"{G.graph.get('dot_name', 'dot')}::{key}", **dict(data))
    return merged


def choose_candidate_dot_files(splitter: Any, pdg_dir: Path, function_name: str, func_start_line: int, func_end_line: int) -> List[Path]:
    candidate_graphs = splitter.choose_candidate_graphs(pdg_dir, function_name, func_start_line, func_end_line)
    dot_files = sorted({Path(g.graph.get("dot_file", "")) for g in candidate_graphs if g.graph.get("dot_file")})
    if dot_files:
        return dot_files
    return sorted(pdg_dir.glob("*.dot"))


# =========================================================
# Blocks and line ownership
# =========================================================

def stmt_text(stmt: Any, source_lines: List[str]) -> str:
    start = safe_int(getattr(stmt, "start_line", 0), 0)
    end = safe_int(getattr(stmt, "end_line", start), start)
    if start < 1 or end < start or end > len(source_lines):
        return str(getattr(stmt, "text", "") or "")
    return "\n".join(source_lines[start - 1:end]).rstrip()


def collect_block_candidate_lines(block: Any, source_lines: List[str]) -> List[int]:
    line_nos: List[int] = []
    stmt_infos = list(getattr(block, "stmt_infos", []))
    if stmt_infos:
        for stmt in stmt_infos:
            start = safe_int(getattr(stmt, "start_line", 0), 0)
            end = safe_int(getattr(stmt, "end_line", start), start)
            for ln in range(start, end + 1):
                if 1 <= ln <= len(source_lines) and is_code_line(source_lines[ln - 1]):
                    line_nos.append(ln)
    else:
        start = safe_int(getattr(block, "start_line", 0), 0)
        end = safe_int(getattr(block, "end_line", start), start)
        for ln in range(start, end + 1):
            if 1 <= ln <= len(source_lines) and is_code_line(source_lines[ln - 1]):
                line_nos.append(ln)
    return unique_keep_order(line_nos)


def build_line_to_block_map(blocks: List[Any], source_lines: List[str]) -> Dict[int, int]:
    candidates: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for blk in blocks:
        bid = safe_int(getattr(blk, "block_id", 0), 0)
        line_nos = collect_block_candidate_lines(blk, source_lines)
        if not line_nos:
            continue
        span_len = max(1, max(line_nos) - min(line_nos) + 1)
        depth = safe_int(getattr(blk, "depth", 0), 0)
        for ln in line_nos:
            candidates[ln].append((span_len, depth, bid))

    line_to_block: Dict[int, int] = {}
    for ln, items in candidates.items():
        items.sort(key=lambda x: (x[0], x[1], x[2]))
        line_to_block[ln] = items[0][2]
    return line_to_block


def block_own_lines(block: Any, source_lines: List[str], line_to_block: Dict[int, int]) -> List[int]:
    bid = safe_int(getattr(block, "block_id", 0), 0)
    candidates = collect_block_candidate_lines(block, source_lines)
    return unique_keep_order([ln for ln in candidates if line_to_block.get(ln) == bid])


def print_blocks(blocks: List[Any], source_lines: List[str], ppl_list: List[float], line_to_block: Dict[int, int]) -> None:
    print("\n" + "=" * 100)
    print("Semantic block split result")
    print("=" * 100)
    for blk in blocks:
        bid = safe_int(getattr(blk, "block_id", 0), 0)
        ppl = ppl_list[bid - 1] if 0 <= bid - 1 < len(ppl_list) else None
        owned = block_own_lines(blk, source_lines, line_to_block)
        print(f"\n[Block {bid}]")
        print("-" * 100)
        print(
            f"kind={getattr(blk, 'kind', '')}  ast_kind={getattr(blk, 'ast_kind', '')}  depth={getattr(blk, 'depth', 0)}  "
            f"span={getattr(blk, 'start_line', None)}-{getattr(blk, 'end_line', None)}  "
            f"stmt_count={len(getattr(blk, 'stmt_infos', []))}  ppl={ppl}"
        )
        print(f"owned_lines: {owned}")
        for ln in owned:
            print(f"  L{ln}: {source_lines[ln - 1].rstrip()}")


# =========================================================
# Raw dot -> line-level dependency graph
# 方向统一：raw a->b 表示 a 被依赖，b 依赖 a
# 在 line_graph 里反转为：b -> a（依赖方 -> 被依赖方）
# =========================================================

def build_dependency_line_graph_from_raw_pdg(
    merged_raw: nx.MultiDiGraph,
    source_lines: List[str],
    func_start_line: int,
    func_end_line: int,
) -> Tuple[nx.DiGraph, Dict[str, str], Dict[str, Any]]:
    line_graph = nx.DiGraph()

    code_line_nos = [
        ln for ln in range(func_start_line, func_end_line + 1)
        if 1 <= ln <= len(source_lines) and is_code_line(source_lines[ln - 1])
    ]
    code_line_set = set(code_line_nos)

    for ln in code_line_nos:
        line_id = f"L{ln}"
        line_graph.add_node(
            line_id,
            line_no=ln,
            text=source_lines[ln - 1].rstrip("\n"),
            raw_node_ids=[],
            raw_kinds=[],
            raw_labels=[],
            raw_self_edges=0,
            dep_in=0,
            dep_out=0,
            used_by_count=0,
            depends_on_count=0,
        )

    raw_node_to_line: Dict[str, str] = {}
    skipped_nodes: List[Dict[str, Any]] = []
    for nid, data in merged_raw.nodes(data=True):
        line_no = infer_line_no_from_node_attrs(data)
        if line_no is None:
            skipped_nodes.append({"raw_node_id": nid, "reason": "no_line_no"})
            continue
        if not (func_start_line <= line_no <= func_end_line):
            continue
        if line_no not in code_line_set:
            skipped_nodes.append({"raw_node_id": nid, "line_no": line_no, "reason": "non_code_line"})
            continue

        line_id = f"L{line_no}"
        raw_node_to_line[nid] = line_id
        node = line_graph.nodes[line_id]
        node["raw_node_ids"].append(nid)
        kind = str(data.get("kind") or "").strip()
        if kind:
            node["raw_kinds"].append(kind)
        label = str(data.get("raw_label") or data.get("code") or data.get("label") or "")
        if label:
            node["raw_labels"].append(label)

    skipped_edges: List[Dict[str, Any]] = []
    edge_role_counter = defaultdict(int)

    for src, dst, key, edata in merged_raw.edges(keys=True, data=True):
        relation = "data" if is_data_dependency_edge(edata) else "control" if is_control_dependency_edge(edata) else "other"
        if relation == "data":
            edge_role_counter["data"] += 1
        elif relation == "control":
            edge_role_counter["control_or_cfg"] += 1
        else:
            edge_role_counter["other_unknown"] += 1
            continue

        src_line = raw_node_to_line.get(src)
        dst_line = raw_node_to_line.get(dst)
        if src_line is None or dst_line is None:
            skipped_edges.append({"src": src, "dst": dst, "reason": "unmapped_endpoints", "label": edata.get("label", "")})
            continue

        # raw: src -> dst, 这里表示 src 被依赖，dst 依赖 src
        dependent_line = dst_line
        dependency_line = src_line
        if dependent_line == dependency_line:
            line_graph.nodes[dependency_line]["raw_self_edges"] += 1
            continue

        line_graph.nodes[dependency_line]["used_by_count"] += 1
        line_graph.nodes[dependent_line]["depends_on_count"] += 1
        line_graph.nodes[dependency_line]["dep_in"] += 1
        line_graph.nodes[dependent_line]["dep_out"] += 1

        label = str(edata.get("label") or "").strip()
        attrs = edata.get("attrs") or {}
        if line_graph.has_edge(dependent_line, dependency_line):
            existing = line_graph.edges[dependent_line, dependency_line]
            existing["count"] = int(existing.get("count", 0)) + 1
            if label and label not in existing.get("labels", []):
                existing.setdefault("labels", []).append(label)
            existing.setdefault("rel_types", set())
            if isinstance(existing["rel_types"], set):
                existing["rel_types"].add(relation)
            existing.setdefault("raw_edge_keys", []).append(key)
            existing.setdefault("raw_edges", []).append({
                "src": src, "dst": dst, "key": key, "label": label, "attrs": attrs, "relation": relation,
            })
        else:
            line_graph.add_edge(
                dependent_line,
                dependency_line,
                count=1,
                labels=[label] if label else [],
                rel_types={relation},
                raw_edge_keys=[key],
                raw_edges=[{
                    "src": src, "dst": dst, "key": key, "label": label, "attrs": attrs, "relation": relation,
                }],
            )

    diagnostics = {
        "code_line_nos": code_line_nos,
        "mapped_line_nodes": sum(1 for nid in line_graph.nodes if line_graph.nodes[nid]["raw_node_ids"]),
        "unmapped_code_lines": [ln for ln in code_line_nos if f"L{ln}" not in line_graph or not line_graph.nodes[f"L{ln}"]["raw_node_ids"]],
        "skipped_nodes": skipped_nodes,
        "skipped_edges": skipped_edges,
        "raw_node_to_line": raw_node_to_line,
        "edge_role_counter": dict(edge_role_counter),
    }
    return line_graph, raw_node_to_line, diagnostics


# =========================================================
# Block statistics
# =========================================================

def compute_block_stats(
    blocks: List[Any],
    source_lines: List[str],
    line_graph: nx.DiGraph,
    ppl_list: List[float],
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], Dict[int, int]]:
    line_to_block = build_line_to_block_map(blocks, source_lines)

    block_stats_by_id: Dict[int, Dict[str, Any]] = {}
    for blk in blocks:
        bid = safe_int(getattr(blk, "block_id", 0), 0)
        idx0 = bid - 1
        ppl = ppl_list[idx0] if 0 <= idx0 < len(ppl_list) else None
        line_nos = block_own_lines(blk, source_lines, line_to_block)
        block_stats_by_id[bid] = {
            "block_id": bid,
            "block_index": idx0,
            "ppl": ppl,
            "stmt_count": len(getattr(blk, "stmt_infos", [])),
            "physical_line_count": len(line_nos),
            "line_nos": line_nos,
            "used_by_total": 0,
            "depends_on_total": 0,
            "internal_edges": 0,
            "ddg_used_by_total": 0,
            "ddg_depends_on_total": 0,
            "cdg_used_by_total": 0,
            "cdg_depends_on_total": 0,
        }

    inter_block_edges: List[Dict[str, Any]] = []
    for src, dst, data in line_graph.edges(data=True):
        src_ln = safe_int(line_graph.nodes[src].get("line_no", 0), 0)
        dst_ln = safe_int(line_graph.nodes[dst].get("line_no", 0), 0)
        src_bid = line_to_block.get(src_ln)
        dst_bid = line_to_block.get(dst_ln)
        if src_bid is None or dst_bid is None:
            continue

        cnt = safe_int(data.get("count", 1), 1)
        rel_types = data.get("rel_types", set())
        labels = list(data.get("labels", [])) if isinstance(data.get("labels", []), list) else []

        if src_bid == dst_bid:
            block_stats_by_id[src_bid]["internal_edges"] += cnt
            continue

        # line_graph 方向：依赖方 -> 被依赖方
        # 所以 src_block 是依赖方，dst_block 是被依赖方
        block_stats_by_id[src_bid]["depends_on_total"] += cnt
        block_stats_by_id[dst_bid]["used_by_total"] += cnt
        if isinstance(rel_types, set):
            if "data" in rel_types:
                block_stats_by_id[src_bid]["ddg_depends_on_total"] += cnt
                block_stats_by_id[dst_bid]["ddg_used_by_total"] += cnt
            if "control" in rel_types:
                block_stats_by_id[src_bid]["cdg_depends_on_total"] += cnt
                block_stats_by_id[dst_bid]["cdg_used_by_total"] += cnt

        inter_block_edges.append({
            "src_line": src_ln,
            "dst_line": dst_ln,
            "src_block_id": src_bid,
            "dst_block_id": dst_bid,
            "count": cnt,
            "labels": labels,
            "rel_types": sorted(list(rel_types)) if isinstance(rel_types, set) else [],
            "raw_edges": data.get("raw_edges", []),
        })

    return block_stats_by_id, inter_block_edges, line_to_block


def semantic_dependency_count(stats: Dict[str, Any]) -> int:
    """
    Dependency score used by l7 block selection.

    Keep l2's successful behavior as the default: count how many later blocks
    depend on this block. Consumer-side dependency counts are exposed in the
    stats for analysis, but not mixed into the score here because l4/l5-style
    bidirectional scoring moved selection away from several exact-match cases.
    """
    used_by_total = safe_int(stats.get("used_by_total", 0), 0)
    return max(0, used_by_total)


# =========================================================
# Layout and drawing
# =========================================================

def chunk_blocks_contiguously(blocks: List[Any], n_cols: int) -> List[List[Any]]:
    n = len(blocks)
    n_cols = max(1, min(n_cols, n if n > 0 else 1))
    base = n // n_cols
    rem = n % n_cols
    out: List[List[Any]] = []
    start = 0
    for c in range(n_cols):
        size = base + (1 if c < rem else 0)
        out.append(blocks[start:start + size])
        start += size
    return out


def estimate_block_height(block: Any, source_lines: List[str], line_to_block: Dict[int, int]) -> float:
    line_nos = block_own_lines(block, source_lines, line_to_block)
    if not line_nos:
        return 1.7
    total_rows = 1
    for ln in line_nos:
        total_rows += max(1, len(wrap_code_text(source_lines[ln - 1], width=27)))
    return 0.95 + 0.52 * total_rows


def build_layout(
    blocks: List[Any],
    source_lines: List[str],
    line_to_block: Dict[int, int],
    n_cols: int = N_COLS,
    x_gap: float = X_GAP,
    y_gap: float = Y_GAP,
    base_width: float = BLOCK_BASE_WIDTH,
) -> Tuple[Dict[int, Dict[str, float]], float, float]:
    cols = chunk_blocks_contiguously(blocks, n_cols)
    col_heights: List[float] = []
    for col in cols:
        h = 0.0
        for blk in col:
            h += estimate_block_height(blk, source_lines, line_to_block)
            h += y_gap
        col_heights.append(h)

    total_height = max(col_heights) + 2.0 if col_heights else 10.0
    total_width = n_cols * base_width + (n_cols + 1) * x_gap

    block_layout: Dict[int, Dict[str, float]] = {}
    for col_idx, col in enumerate(cols):
        x = x_gap + col_idx * (base_width + x_gap)
        y_cursor = total_height - y_gap
        for blk in col:
            bid = safe_int(getattr(blk, "block_id", 0), 0)
            h = estimate_block_height(blk, source_lines, line_to_block)
            y_cursor -= h
            block_layout[bid] = {"x": x, "y": y_cursor, "w": base_width, "h": h, "col": col_idx}
            y_cursor -= y_gap
    return block_layout, total_width, total_height


def add_block_patch(ax, x: float, y: float, w: float, h: float, facecolor: str = "#F8FAFF", edgecolor: str = "#5A5A7A") -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.05,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=0.97,
        zorder=1,
    ))


def draw_arrow(ax, p1: Tuple[float, float], p2: Tuple[float, float], color: str = "#4C78A8", alpha: float = 0.40, lw: float = 2.0, rad: float = 0.0, zorder: int = 0) -> None:
    ax.add_patch(FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    ))


def edge_rad(src: Tuple[float, float], dst: Tuple[float, float], salt: int = 0) -> float:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    base = 0.22 if abs(dx) > 2.5 else 0.12
    sign = -1.0 if ((hash((round(src[0], 1), round(src[1], 1), round(dst[0], 1), round(dst[1], 1), salt)) & 1) == 0) else 1.0
    if abs(dy) > 10.0:
        return 0.0
    return sign * base


def block_title(blk: Any, stats: Dict[str, Any]) -> str:
    bid = safe_int(getattr(blk, "block_id", 0), 0)
    ppl = stats.get("ppl", None)
    ppl_text = "NA" if ppl is None else f"{ppl:.2f}"
    return (
        f"B{bid} | ppl={ppl_text} | "
        f"used_by={stats.get('used_by_total', 0)} | "
        f"depends_on={stats.get('depends_on_total', 0)} | "
        f"lines={stats.get('physical_line_count', 0)}"
    )


def line_y_in_block(block_layout: Dict[str, float], line_index: int, total_lines: int) -> float:
    if total_lines <= 0:
        return block_layout["y"] + block_layout["h"] / 2.0
    top = block_layout["y"] + block_layout["h"]
    bottom = block_layout["y"] + 0.22
    usable = max(0.4, top - bottom)
    ratio = (line_index + 1) / (total_lines + 1)
    return top - ratio * usable


def choose_anchor_points(src_block_layout: Dict[str, float], dst_block_layout: Dict[str, float], src_y: float) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    sx, sy, sw, sh = src_block_layout["x"], src_block_layout["y"], src_block_layout["w"], src_block_layout["h"]
    dx, dy, dw, dh = dst_block_layout["x"], dst_block_layout["y"], dst_block_layout["w"], dst_block_layout["h"]
    src_center = (sx + sw / 2.0, sy + sh / 2.0)
    dst_center = (dx + dw / 2.0, dy + dh / 2.0)
    delta_x = dst_center[0] - src_center[0]
    delta_y = dst_center[1] - src_center[1]
    start_x = sx + sw + 0.10 if delta_x >= 0 else sx - 0.10
    start_y = min(max(src_y, sy + 0.20), sy + sh - 0.20)
    end_x = dx - 0.12 if delta_x >= 0 else dx + dw + 0.12
    end_y = min(max(dst_center[1], dy + 0.20), dy + dh - 0.20)
    if abs(delta_x) < 3.0:
        start_x = min(max(src_center[0], sx + 0.40), sx + sw - 0.40)
        start_y = sy + sh + 0.10 if delta_y >= 0 else sy - 0.10
        end_x = min(max(dst_center[0], dx + 0.40), dx + dw - 0.40)
        end_y = dy - 0.12 if delta_y >= 0 else dy + dh + 0.12
    return (start_x, start_y), (end_x, end_y), delta_x


def draw_semantic_block_graph(
    blocks: List[Any],
    source_lines: List[str],
    line_graph: nx.DiGraph,
    ppl_list: List[float],
    out_dir: Path,
    function_name: str,
    source_path: Optional[Path] = None,
    n_cols: int = N_COLS,
    dpi: int = 220,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    block_stats_by_id, inter_block_edges, line_to_block = compute_block_stats(blocks, source_lines, line_graph, ppl_list)
    blocks_sorted = sorted(blocks, key=lambda b: safe_int(getattr(b, "block_id", 0), 0))
    block_layout, total_width, total_height = build_layout(blocks_sorted, source_lines, line_to_block, n_cols=n_cols)

    fig_w = max(18.0, total_width / 1.18)
    fig_h = max(12.0, total_height / 1.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    edge_color = "#3B6EA8"
    box_face = "#F8FAFF"
    box_edge = "#4F5B76"
    header_face = "#E7EEF9"

    line_pos: Dict[str, Tuple[float, float]] = {}
    line_block_map: Dict[str, int] = {}

    for blk in blocks_sorted:
        bid = safe_int(getattr(blk, "block_id", 0), 0)
        layout = block_layout.get(bid)
        if layout is None:
            continue
        x, y, w, h = layout["x"], layout["y"], layout["w"], layout["h"]
        add_block_patch(ax, x, y, w, h, facecolor=box_face, edgecolor=box_edge)

        title_h = min(1.0, max(0.80, 0.10 * h))
        ax.add_patch(FancyBboxPatch(
            (x + 0.07, y + h - title_h - 0.07),
            w - 0.14, title_h,
            boxstyle="round,pad=0.012,rounding_size=0.14",
            linewidth=0.0,
            facecolor=header_face,
            alpha=1.0,
            zorder=2,
        ))

        stats = block_stats_by_id.get(bid, {})
        ax.text(
            x + 0.14,
            y + h - 0.26,
            block_title(blk, stats),
            fontsize=7.5,
            fontweight="bold",
            family="monospace",
            va="top",
            ha="left",
            color="#1F2937",
            zorder=3,
        )

        if SHOW_BLOCK_OVERLAY_STATS:
            ax.text(
                x + w - 0.14,
                y + 0.10,
                f"internal={stats.get('internal_edges', 0)}",
                fontsize=6.2,
                family="monospace",
                va="bottom",
                ha="right",
                color="#374151",
                zorder=3,
            )

        owned_lines = block_own_lines(blk, source_lines, line_to_block)
        if not owned_lines:
            ax.text(x + 0.14, y + h - title_h - 0.16, "<empty>", fontsize=6.8, family="monospace", va="top", ha="left", color="#6B7280", zorder=3)
            continue

        cur_y = y + h - title_h - 0.16
        for idx, ln in enumerate(owned_lines):
            line_id = f"L{ln}"
            text = source_lines[ln - 1].rstrip("\n")
            wrapped = wrap_code_text(text, width=27)
            display = [f"L{ln}: {wrapped[0]}"] + [f"    {wline}" for wline in wrapped[1:]]
            center_y = line_y_in_block(layout, idx, len(owned_lines))
            line_pos[line_id] = (x + 0.26, center_y)
            line_block_map[line_id] = bid
            ax.text(
                x + 0.14,
                cur_y,
                "\n".join(display),
                fontsize=6.25,
                family="monospace",
                va="top",
                ha="left",
                color="#111827",
                zorder=3,
                linespacing=1.06,
            )
            cur_y -= 0.28 * len(display) + 0.08

    # 按线级边聚合，画跨块箭头
    pair_edges: Dict[Tuple[str, int], Dict[str, Any]] = OrderedDict()
    for src, dst, data in line_graph.edges(data=True):
        sb = line_block_map.get(src)
        db = line_block_map.get(dst)
        if sb is None or db is None:
            continue
        if DRAW_ONLY_CROSS_BLOCK_EDGES and sb == db:
            continue

        key = (src, db)
        if key not in pair_edges:
            pair_edges[key] = {"count": 0, "labels": set(), "rel_types": set(), "src_block": sb, "dst_block": db}
        pair_edges[key]["count"] += safe_int(data.get("count", 1), 1)
        for lab in data.get("labels", []) if isinstance(data.get("labels", []), list) else []:
            if lab:
                pair_edges[key]["labels"].add(lab)
        rel_types = data.get("rel_types", set())
        if isinstance(rel_types, set):
            pair_edges[key]["rel_types"].update(rel_types)

    all_pairs = list(pair_edges.items())
    all_pairs.sort(key=lambda kv: (-kv[1]["count"], kv[0][0], kv[0][1]))

    pair_group_index: Dict[Tuple[int, int], int] = defaultdict(int)
    for idx, ((src, dst_block_id), info) in enumerate(all_pairs):
        if src not in line_pos:
            continue
        src_bid = info["src_block"]
        dst_bid = info["dst_block"]
        if src_bid not in block_layout or dst_bid not in block_layout:
            continue

        src_layout = block_layout[src_bid]
        dst_layout = block_layout[dst_bid]
        src_ln = int(src[1:])
        src_blk = next(b for b in blocks_sorted if safe_int(getattr(b, "block_id", 0), 0) == src_bid)
        src_owned_lines = block_own_lines(src_blk, source_lines, line_to_block)
        try:
            src_idx = src_owned_lines.index(src_ln)
        except Exception:
            src_idx = 0
        src_y = line_y_in_block(src_layout, src_idx, max(1, len(src_owned_lines)))

        p1, p2, _ = choose_anchor_points(src_layout, dst_layout, src_y)
        group = (min(src_bid, dst_bid), max(src_bid, dst_bid))
        pair_group_index[group] += 1
        order = pair_group_index[group]
        rad = edge_rad(p1, p2, salt=idx) + (0.03 * ((order % 5) - 2))

        rel_types = info.get("rel_types", set())
        is_control = "control" in rel_types and "data" not in rel_types
        color = "#8A63D2" if is_control else edge_color
        alpha = 0.38 if "data" in rel_types else 0.28
        lw = 2.2 if info["count"] == 1 else 2.8
        draw_arrow(ax, p1, p2, color=color, alpha=alpha, lw=lw, rad=rad, zorder=0)

        if DRAW_EDGE_COUNT_LABELS and info["count"] > 1:
            mx = (p1[0] + p2[0]) / 2.0
            my = (p1[1] + p2[1]) / 2.0
            ax.text(mx, my, f"x{info['count']}", fontsize=6.2, family="monospace", color="#B45309", ha="center", va="center", zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.74))

    fig.suptitle(f"Semantic Block Dependency Graph - {function_name}", fontsize=15, fontweight="bold", y=0.995)
    if source_path is not None:
        fig.text(0.5, 0.975, f"Source: {source_path}", ha="center", va="top", fontsize=8.7, color="#6B7280")

    png_path = out_dir / "semantic_block_dependency_graph.png"
    pdf_path = out_dir / "semantic_block_dependency_graph.pdf"
    svg_path = out_dir / "semantic_block_dependency_graph.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.35)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    graph_data = {
        "function_name": function_name,
        "source_path": str(source_path) if source_path else None,
        "block_count": len(blocks),
        "blocks": [],
        "block_stats": {str(k): v for k, v in block_stats_by_id.items()},
        "inter_block_edges": inter_block_edges,
        "files": {"png": str(png_path), "pdf": str(pdf_path), "svg": str(svg_path)},
        "line_positions": {k: [v[0], v[1]] for k, v in line_pos.items()},
        "line_block_map": line_block_map,
    }

    for blk in blocks_sorted:
        bid = safe_int(getattr(blk, "block_id", 0), 0)
        idx0 = bid - 1
        stats = block_stats_by_id.get(bid, {})
        ppl = ppl_list[idx0] if 0 <= idx0 < len(ppl_list) else None
        owned_lines = block_own_lines(blk, source_lines, line_to_block)
        graph_data["blocks"].append({
            "block_id": bid,
            "block_index": idx0,
            "kind": getattr(blk, "kind", ""),
            "ast_kind": getattr(blk, "ast_kind", ""),
            "depth": getattr(blk, "depth", 0),
            "start_line": getattr(blk, "start_line", None),
            "end_line": getattr(blk, "end_line", None),
            "ppl": ppl,
            "stmt_count": stats.get("stmt_count", 0),
            "physical_line_count": stats.get("physical_line_count", 0),
            "used_by_total": stats.get("used_by_total", 0),
            "depends_on_total": stats.get("depends_on_total", 0),
            "internal_edges": stats.get("internal_edges", 0),
            "ddg_used_by_total": stats.get("ddg_used_by_total", 0),
            "ddg_depends_on_total": stats.get("ddg_depends_on_total", 0),
            "cdg_used_by_total": stats.get("cdg_used_by_total", 0),
            "cdg_depends_on_total": stats.get("cdg_depends_on_total", 0),
            "physical_lines": owned_lines,
            "code": getattr(blk, "code", ""),
            "stmt_infos": [
                {
                    "node_id": getattr(stmt, "node_id", ""),
                    "start_line": getattr(stmt, "start_line", None),
                    "end_line": getattr(stmt, "end_line", None),
                    "kind": getattr(stmt, "kind", ""),
                    "text": stmt_text(stmt, source_lines),
                    "depth": getattr(stmt, "depth", 0),
                    "raw_node_ids": getattr(stmt, "raw_node_ids", []),
                    "raw_kinds": getattr(stmt, "raw_kinds", []),
                    "raw_labels": getattr(stmt, "raw_labels", []),
                }
                for stmt in getattr(blk, "stmt_infos", [])
            ],
        })

    json_path = out_dir / "semantic_block_dependency_graph.json"
    json_path.write_text(json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph_data


# =========================================================
# Pipeline
# =========================================================

def save_log_file(out_dir: Path, lines: List[str]) -> None:
    (out_dir / "semantic_blocks_log.txt").write_text("\n".join(lines), encoding="utf-8")


def build_pipeline(source_text: str, function_name: str, joern_home: Optional[Path], work_root: Path) -> Dict[str, Any]:
    splitter = require_splitter()

    parsed_source, tree, _ = parse_source_for_splitter(source_text)
    source_lines = parsed_source.splitlines()
    func_node = detect_function_node(tree, function_name)
    func_start_line, func_end_line = splitter.function_span(func_node)

    ensure_dir(work_root)
    source_file = work_root / "input.py"
    source_file.write_text(parsed_source, encoding="utf-8")

    if joern_home is None:
        joern_home = Path(JOERN_HOME).expanduser().resolve()
    if not joern_home.exists():
        raise FileNotFoundError(f"JOERN_HOME does not exist: {joern_home}")

    print(f"[INFO] Target function: {function_name}")
    print(f"[INFO] Function span: {func_start_line}-{func_end_line}")
    print(f"[INFO] Work root: {work_root}")

    pdg_dir = Path(splitter.generate_pdg_with_joern(source_file, work_root, joern_home))
    print(f"[INFO] PDG export dir: {pdg_dir}")

    dot_files = choose_candidate_dot_files(splitter, pdg_dir, function_name, func_start_line, func_end_line)
    if not dot_files:
        raise FileNotFoundError(f"No usable dot files found under {pdg_dir}")

    print(f"[INFO] Selected dot files: {len(dot_files)}")
    for f in dot_files:
        print(f"   - {f.name}")

    parsed_graphs: List[nx.MultiDiGraph] = []
    parse_dump: List[Dict[str, Any]] = []
    for dot_file in dot_files:
        G = parse_dot_file(dot_file)
        parsed_graphs.append(G)
        parse_dump.append({"dot_file": str(dot_file), "node_count": G.number_of_nodes(), "edge_count": G.number_of_edges()})

    merged_raw = merge_raw_graphs(parsed_graphs)
    raw_edge_roles = defaultdict(int)
    for _, _, _, edata in merged_raw.edges(keys=True, data=True):
        if is_control_dependency_edge(edata):
            raw_edge_roles["control_or_cfg"] += 1
        elif is_data_dependency_edge(edata):
            raw_edge_roles["data"] += 1
        else:
            raw_edge_roles["other_unknown"] += 1

    print(f"[INFO] Merged raw dot: nodes={merged_raw.number_of_nodes()}, edges={merged_raw.number_of_edges()}")
    print(f"[INFO] Raw edge type stats: {dict(raw_edge_roles)}")

    stmt_infos, stmt_by_ast_id, span_by_line = splitter.build_stmt_infos_for_function(
        source_text=parsed_source,
        source_lines=source_lines,
        func_node=func_node,
    )

    splitter_line_graph = splitter.build_line_graph_from_merged_pdg(
        merged_raw=merged_raw,
        stmt_infos=stmt_infos,
        span_by_line=span_by_line,
        func_start_line=func_start_line,
        func_end_line=func_end_line,
    )
    print(f"[INFO] splitter line graph: nodes={splitter_line_graph.number_of_nodes()}, edges={splitter_line_graph.number_of_edges()}")

    blocks = splitter.build_semantic_blocks(
        func_node=func_node,
        source_text=parsed_source,
        source_lines=source_lines,
        line_graph=splitter_line_graph,
        stmt_by_ast_id=stmt_by_ast_id,
    )
    blocks = sorted(blocks, key=lambda b: safe_int(getattr(b, "block_id", 0), 0))

    line_to_block_preview = build_line_to_block_map(blocks, source_lines)
    print(f"[INFO] semantic blocks: {len(blocks)}")
    print_blocks(blocks, source_lines, DEFAULT_BLOCK_PPLS, line_to_block_preview)

    data_line_graph, raw_node_to_line, diagnostics = build_dependency_line_graph_from_raw_pdg(
        merged_raw=merged_raw,
        source_lines=source_lines,
        func_start_line=func_start_line,
        func_end_line=func_end_line,
    )

    code_line_nos = diagnostics["code_line_nos"]
    mapped_lines = sorted(ln for ln in code_line_nos if f"L{ln}" in data_line_graph and data_line_graph.nodes[f"L{ln}"]["raw_node_ids"])
    missing_lines = sorted(ln for ln in code_line_nos if f"L{ln}" not in data_line_graph or not data_line_graph.nodes[f"L{ln}"]["raw_node_ids"])

    print("\n[INFO] Line coverage check (dependency-only)")
    print(f"       code lines: {len(code_line_nos)}")
    print(f"       mapped lines: {len(mapped_lines)}")
    print(f"       missing lines: {len(missing_lines)}")
    if missing_lines:
        print(f"       missing line numbers: {missing_lines}")

    block_stats_by_id, inter_block_edges, line_to_block = compute_block_stats(
        blocks=blocks,
        source_lines=source_lines,
        line_graph=data_line_graph,
        ppl_list=DEFAULT_BLOCK_PPLS,
    )

    out_dir = Path(OUT_DIR).expanduser().resolve() if OUT_DIR else (work_root / "result")
    ensure_dir(out_dir)

    graph_data = draw_semantic_block_graph(
        blocks=blocks,
        source_lines=source_lines,
        line_graph=data_line_graph,
        ppl_list=DEFAULT_BLOCK_PPLS,
        out_dir=out_dir,
        function_name=function_name,
        source_path=source_file if not USE_INLINE_SOURCE else None,
        n_cols=N_COLS,
    )

    summary = {
        "function_name": function_name,
        "function_span": [func_start_line, func_end_line],
        "dot_files": [str(p) for p in dot_files],
        "raw_graph": {
            "nodes": merged_raw.number_of_nodes(),
            "edges": merged_raw.number_of_edges(),
            "edge_roles": dict(raw_edge_roles),
        },
        "splitter_graph": {
            "nodes": splitter_line_graph.number_of_nodes(),
            "edges": splitter_line_graph.number_of_edges(),
        },
        "dependency_graph": {
            "nodes": data_line_graph.number_of_nodes(),
            "edges": data_line_graph.number_of_edges(),
        },
        "coverage": {
            "code_lines": len(code_line_nos),
            "mapped_lines": len(mapped_lines),
            "missing_lines": missing_lines,
        },
        "blocks": graph_data["blocks"],
        "block_stats": {str(k): v for k, v in block_stats_by_id.items()},
        "inter_block_edges": inter_block_edges,
        "unmapped_raw_nodes": diagnostics["skipped_nodes"],
        "unmapped_raw_edges": diagnostics["skipped_edges"],
        "raw_node_to_line": raw_node_to_line,
        "line_to_block": {str(k): v for k, v in line_to_block.items()},
        "files": graph_data["files"],
    }

    (out_dir / "semantic_block_dependency_graph.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "dot_parse_dump.json").write_text(json.dumps(parse_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    save_log_file(
        out_dir,
        [
            "=" * 100,
            "Semantic block log",
            "=" * 100,
            f"function: {function_name}",
            f"range: {func_start_line}-{func_end_line}",
            f"dot files: {len(dot_files)}",
            f"raw merged nodes: {merged_raw.number_of_nodes()}",
            f"raw merged edges: {merged_raw.number_of_edges()}",
            f"raw edge roles: {dict(raw_edge_roles)}",
            f"splitter line graph nodes: {splitter_line_graph.number_of_nodes()}",
            f"splitter line graph edges: {splitter_line_graph.number_of_edges()}",
            f"dependency line graph nodes: {data_line_graph.number_of_nodes()}",
            f"dependency line graph edges: {data_line_graph.number_of_edges()}",
            f"code lines: {len(code_line_nos)}",
            f"mapped lines: {len(mapped_lines)}",
            f"missing lines: {len(missing_lines)}",
            f"missing line numbers: {missing_lines}",
            "",
            "Block stats:",
        ]
        + [
            f"  B{bid}: ppl={stats.get('ppl')}  lines={stats.get('line_nos', [])}  "
            f"stmt={stats.get('stmt_count', 0)}  used_by={stats.get('used_by_total', 0)}  depends_on={stats.get('depends_on_total', 0)}"
            for bid, stats in sorted(block_stats_by_id.items())
        ]
    )

    print(f"\n[OK] Output written to: {out_dir}")
    print("     - semantic_block_dependency_graph.png")
    print("     - semantic_block_dependency_graph.pdf")
    print("     - semantic_block_dependency_graph.svg")
    print("     - semantic_block_dependency_graph.json")
    print("     - semantic_blocks_log.txt")
    print("     - dot_parse_dump.json")

    return summary


def main() -> None:
    source_text = load_source_text()
    work_root = WORK_ROOT.expanduser().resolve()
    joern_home = Path(JOERN_HOME).expanduser().resolve()
    build_pipeline(source_text=source_text, function_name=FUNCTION_NAME, joern_home=joern_home, work_root=work_root)


if __name__ == "__main__":
    main()
