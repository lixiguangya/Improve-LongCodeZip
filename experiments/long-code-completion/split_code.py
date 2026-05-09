
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_code_fixed_semantic_blocks.py

修复重点：
1) Joern 导出的 dot 不改，改的是“选择/合并/映射”逻辑
2) 优先读取 0.dot；若没有函数专用 dot，则把 module dot 当作函数图的回退来源
3) raise / pass / return 等终结语句不再被吞掉，作为独立 atomic 语句保留
4) if / elif / else 分支严格拆分；分支体 <= 阈值时整体作为一个块，否则拆 header 后递归
5) 支持 function / class / module / suite 四种输入
"""

from __future__ import annotations

import ast
import datetime
import html
import json
import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from functools import lru_cache
import logging

import networkx as nx

try:
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package
except Exception:
    RefactoringTool = None
    get_fixers_from_package = None


def suppress_lib2to3_noise() -> None:
    for name in (
        "refactor",
        "lib2to3",
        "lib2to3.refactor",
        "lib2to3.fixes",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


suppress_lib2to3_noise()


# =========================================================
# 1) 配置区
# =========================================================

JOERN_HOME = r"/home/nwpu_wyh/joern-cli"

# 这里保留和你原来一样的入口风格；实际运行时你可以直接改这两个常量
SOURCE_CODE = r"""

	def _retrieve_schema_from_server(self, url, conn=None):
        tmpdir = None
        has_conn = conn is not None

        self.debug('retrieving schema for SchemaCache url=%s conn=%s', url, conn)

        try:
            if api.env.context == 'server' and conn is None:
                # FIXME: is this really what we want to do?
                # This seems like this logic is in the wrong place and may conflict with other state.
                try:
                    # Create a new credentials cache for this Apache process
                    tmpdir = tempfile.mkdtemp(prefix = "tmp-")
                    ccache_file = 'FILE:%s/ccache' % tmpdir
                    krbcontext = krbV.default_context()
                    principal = str('HTTP/%s@%s' % (api.env.host, api.env.realm))
                    keytab = krbV.Keytab(name='/etc/httpd/conf/ipa.keytab', context=krbcontext)
                    principal = krbV.Principal(name=principal, context=krbcontext)
                    prev_ccache = os.environ.get('KRB5CCNAME')
                    os.environ['KRB5CCNAME'] = ccache_file
                    ccache = krbV.CCache(name=ccache_file, context=krbcontext, primary_principal=principal)
                    ccache.init(principal)
                    ccache.init_creds_keytab(keytab=keytab, principal=principal)
                except krbV.Krb5Error, e:
                    raise StandardError('Unable to retrieve LDAP schema. Error initializing principal %s in %s: %s' % (principal.name, '/etc/httpd/conf/ipa.keytab', str(e)))
                finally:
                    if prev_ccache is not None:
                        os.environ['KRB5CCNAME'] = prev_ccache


            if conn is None:
                conn = IPASimpleLDAPObject(url)
                if url.startswith('ldapi://'):
                    conn.set_option(_ldap.OPT_HOST_NAME, api.env.host)
                conn.sasl_interactive_bind_s(None, SASL_AUTH)

            try:
                schema_entry = conn.search_s('cn=schema', _ldap.SCOPE_BASE,
                    attrlist=['attributetypes', 'objectclasses'])[0]
            except _ldap.NO_SUCH_OBJECT:
                # try different location for schema
                # openldap has schema located in cn=subschema
                self.debug('cn=schema not found, fallback to cn=subschema')
                schema_entry = conn.search_s('cn=subschema', _ldap.SCOPE_BASE,
                    attrlist=['attributetypes', 'objectclasses'])[0]
            if not has_conn:
                conn.unbind_s()
        except _ldap.SERVER_DOWN:
            raise NetworkError(uri=url,
                               error=u'LDAP Server Down, unable to retrieve LDAP schema')
        except _ldap.LDAPError, e:
            desc = e.args[0]['desc'].strip()
            info = e.args[0].get('info', '').strip()
            raise DatabaseError(desc = u'uri=%s' % url,
                                info = u'Unable to retrieve LDAP schema: %s: %s' % (desc, info))
        except IndexError:
            # no 'cn=schema' entry in LDAP? some servers use 'cn=subschema'
            # TODO: DS uses 'cn=schema', support for other server?
            #       raise a more appropriate exception
            raise
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir)

        return _ldap.schema.SubSchema(schema_entry[1])

"""
FUNCTION_NAME = "_retrieve_schema_from_server"
SOURCE_SUFFIX = ".py"
PYTHON_VENV_DIR = ""

# 分块阈值
CONTROL_LINE_LIMIT = 4
MAX_SIMPLE_GROUP_SIZE = 6
MAX_SIMPLE_GROUP_SPAN = 6
MAX_SIMPLE_LINE_GAP = 1
MAX_SIMPLE_INIT_GROUP_SIZE = 4

# PDG 节点过滤：不参与映射的抽象节点
EXCLUDE_KINDS = {
    "METHOD", "METHOD_RETURN",
    "PARAM", "PARAMETER", "PARAMETER_IN", "PARAMETER_OUT",
    "BLOCK", "LOCAL", "TYPE", "TYPE_DECL", "TYPE_REF",
    "MEMBER", "FIELD_IDENTIFIER", "IDENTIFIER", "LITERAL",
}


# =========================================================
# 2) 数据结构
# =========================================================

@dataclass
class StmtInfo:
    node_id: str
    start_line: int
    end_line: int
    kind: str
    text: str
    depth: int
    is_exec: bool
    ast_node_id: int
    raw_node_ids: List[str] = field(default_factory=list)
    raw_kinds: List[str] = field(default_factory=list)
    raw_labels: List[str] = field(default_factory=list)

    @property
    def span_len(self) -> int:
        return max(1, self.end_line - self.start_line + 1)


@dataclass
class SemanticBlock:
    block_id: int
    kind: str
    ast_kind: str
    depth: int
    start_line: int
    end_line: int
    scope_chain: List[Dict[str, Any]]
    code: str
    stmt_infos: List[StmtInfo]
    pdg_connected: bool
    synthetic: bool = False

    @property
    def node_ids(self) -> List[str]:
        return [x.node_id for x in self.stmt_infos]


# =========================================================
# 3) 通用工具
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def which(cmd: str, joern_home: Optional[Path] = None) -> str:
    if joern_home:
        candidate = joern_home / cmd
        if candidate.exists():
            return str(candidate)
    found = shutil.which(cmd)
    if found:
        return found
    raise FileNotFoundError(f"找不到命令 {cmd}，请检查 JOERN_HOME 或 PATH。")

def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)

def normalize_source(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


def detect_leading_indent_prefix(source_text: str) -> str:
    """Return the common leading indentation of the first non-empty line, normalized to spaces."""
    for line in source_text.splitlines():
        if line.strip():
            expanded = line.expandtabs(4)
            return re.match(r"^ *", expanded).group(0)
    return ""


def dedent_source_text(source_text: str) -> str:
    """Shift the whole snippet left by its shared leading indentation, line by line."""
    normalized = normalize_source(source_text)
    lines = normalized.splitlines()
    if not lines:
        return normalized

    expanded_lines = [line.expandtabs(4) for line in lines]
    non_empty = [line for line in expanded_lines if line.strip()]
    if not non_empty:
        return normalized

    indent_width = min(len(re.match(r"^ *", line).group(0)) for line in non_empty)
    if indent_width <= 0:
        return normalized

    indent_prefix = " " * indent_width
    shifted: List[str] = []
    for line in expanded_lines:
        if not line.strip():
            shifted.append("")
            continue
        if line.startswith(indent_prefix):
            shifted.append(line[indent_width:])
        else:
            leading = len(re.match(r"^ *", line).group(0))
            shifted.append(line[min(leading, indent_width):])
    return "\n".join(shifted).rstrip("\n") + "\n"


def reindent_text(text: str, prefix: str) -> str:
    if not prefix:
        return text
    lines = text.splitlines()
    return "\n".join((prefix + line) if line.strip() else line for line in lines)


def _comment_context_for_block(source_lines: List[str], block: SemanticBlock) -> Tuple[int, int]:
    """Compute the render span for a block while keeping nearby leading comments attached."""
    start = block.start_line
    end = block.end_line
    if start < 1 or end < start:
        return start, end
    start = _comment_context_start_line(source_lines, start)
    return start, end


def reindent_blocks_for_output(blocks: List[SemanticBlock], prefix: str, source_lines: Optional[List[str]] = None) -> List[SemanticBlock]:
    if not prefix and source_lines is None:
        return blocks

    rendered = deepcopy(blocks)
    if not rendered:
        return rendered

    spans: List[Tuple[int, int]] = []
    for block in rendered:
        if source_lines is not None:
            spans.append(_comment_context_for_block(source_lines, block))
        else:
            spans.append((block.start_line, block.end_line))

    ordered_indices = sorted(
        range(len(rendered)),
        key=lambda i: (
            rendered[i].start_line if rendered[i].start_line >= 0 else 10**9,
            rendered[i].end_line if rendered[i].end_line >= 0 else 10**9,
            rendered[i].depth,
            i,
        ),
    )

    for prev_idx, cur_idx in zip(ordered_indices, ordered_indices[1:]):
        cur_start, _ = spans[cur_idx]
        prev_start, prev_end = spans[prev_idx]
        if cur_start > 0 and prev_end >= cur_start:
            spans[prev_idx] = (prev_start, cur_start - 1)

    protected_floor = 1
    for idx in ordered_indices:
        block = rendered[idx]
        start, end = spans[idx]

        if block.kind == "definition":
            if start > 0:
                block.start_line = start
            if end >= start and end > 0:
                block.end_line = end
            if prefix:
                block.code = reindent_text(block.code, prefix)
                for stmt in block.stmt_infos:
                    stmt.text = reindent_text(stmt.text, prefix)
            rendered_len = len(block.code.splitlines()) if block.code else 0
            if rendered_len > 0 and block.start_line > 0:
                protected_floor = max(protected_floor, block.start_line + rendered_len)
            continue

        if source_lines is not None and start > 0 and end >= start:
            start = max(start, protected_floor)
            if block.kind in {"control", "atomic"}:
                sliced = slice_source_lines(source_lines, start, end)
                if sliced.strip():
                    block.code = sliced

        if start > 0:
            block.start_line = start
        if end >= start and end > 0:
            block.end_line = end

        if prefix:
            block.code = reindent_text(block.code, prefix)
            for stmt in block.stmt_infos:
                stmt.text = reindent_text(stmt.text, prefix)

    return rendered


def _comment_context_start_line(source_lines: List[str], stmt_start_line: int, lower_bound: int = 1) -> int:
    """Extend a rendered span upward to include the nearest preceding comment block.

    Only a contiguous comment/blank block that is directly attached to the statement
    is included. The optional lower_bound prevents a child block from stealing a
    comment span that has already been assigned to an enclosing definition block.
    """
    if not source_lines or stmt_start_line <= 1:
        return stmt_start_line

    lower_bound = max(1, lower_bound)
    if stmt_start_line <= lower_bound:
        return stmt_start_line

    idx = min(stmt_start_line - 1, len(source_lines))
    if idx <= 0:
        return stmt_start_line

    # Skip blank lines immediately above the statement, but never cross the lower bound.
    while idx > lower_bound - 1 and not source_lines[idx - 1].strip():
        idx -= 1

    # Only attach the block if the closest non-blank line is a comment.
    if idx <= lower_bound - 1 or not source_lines[idx - 1].lstrip().startswith("#"):
        return stmt_start_line

    # Include the whole contiguous comment/blank region.
    while idx > lower_bound - 1 and (not source_lines[idx - 1].strip() or source_lines[idx - 1].lstrip().startswith("#")):
        idx -= 1

    return max(idx + 1, lower_bound)


def render_stmt_text_with_context(info: StmtInfo, source_lines: List[str], lower_bound: int = 1) -> str:
    """Render a statement together with its leading comment block, if any."""
    start_line = _comment_context_start_line(source_lines, info.start_line, lower_bound=lower_bound)
    rendered = slice_source_lines(source_lines, start_line, info.end_line)
    return rendered if rendered.strip() else info.text


def attach_leading_comment_context(
    code: str,
    source_lines: Optional[List[str]],
    start_line: int,
    end_line: int,
    lower_bound: int = 1,
) -> Tuple[str, int, int]:
    """Attach the closest leading comment block to a rendered semantic block.

    A comment belongs to the following block only when it is the nearest
    non-blank line above that block. This keeps explanatory comments with the
    code they describe, while avoiding broad upward grabs across real code.
    """
    if not source_lines or start_line < 1 or end_line < start_line:
        return code, start_line, end_line
    context_start = _comment_context_start_line(
        source_lines,
        start_line,
        lower_bound=lower_bound,
    )
    if context_start >= start_line:
        return code, start_line, end_line
    prefix = slice_source_lines(source_lines, context_start, start_line - 1)
    if not prefix.strip():
        return code, start_line, end_line
    if code and code.strip():
        return f"{prefix}\n{code.rstrip()}", context_start, end_line
    rendered = slice_source_lines(source_lines, context_start, end_line)
    return rendered, context_start, end_line


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def is_docstring_expr(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )

def is_simple_stmt_node(stmt: ast.AST) -> bool:
    return isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr,
                             ast.Pass, ast.Assert, ast.Import, ast.ImportFrom, ast.Delete,
                             ast.Raise, ast.Return, ast.Break, ast.Continue))

def is_terminal_stmt_node(stmt: ast.AST) -> bool:
    return isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue))

TERMINAL_KINDS = {"raise", "pass", "return", "break", "continue"}
IMPORT_KINDS = {"import", "importfrom"}
ASSIGN_KINDS = {"assign", "annassign", "augassign"}


def _identifier_set_from_text(text: str) -> set:
    if not text:
        return set()
    stop_words = {
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
        "while", "with", "yield",
    }
    return {
        ident
        for ident in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
        if ident not in stop_words and not ident.startswith("__")
    }


def _lhs_identifier_set_from_text(text: str) -> set:
    if not text:
        return set()
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if "=" not in first_line:
        return set()
    lhs = first_line.split("=", 1)[0]
    lhs = re.sub(r"^\s*(global|nonlocal)\s+", "", lhs)
    return _identifier_set_from_text(lhs)


def _has_call_syntax(text: str) -> bool:
    return bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_\.]*\s*\(", text or ""))


def _is_io_like_text(text: str) -> bool:
    low = (text or "").lower()
    return bool(
        re.search(r"\bopen\s*\(", low)
        or re.search(r"\.(read|write|readlines|writelines)\s*\(", low)
        or re.search(r"\b(read|write)\s*\(", low)
        or "'rb'" in low
        or '"rb"' in low
        or "'wb'" in low
        or '"wb"' in low
    )


def _is_side_effect_like_text(text: str) -> bool:
    low = (text or "").lower()
    return bool(
        re.search(
            r"\.(append|extend|update|remove|pop|setdefault|init|bind|connect|send|debug|info|warning|error|set_option)\s*\(",
            low,
        )
        or re.search(r"\b(os\.environ|setattr|delattr|shutil\.|os\.remove|os\.unlink|os\.mkdir|os\.makedirs)\b", low)
    )


def atomic_stmt_role(info: StmtInfo) -> str:
    if info.kind in TERMINAL_KINDS:
        return "terminal"
    if info.kind in IMPORT_KINDS:
        return "import"
    if _is_io_like_text(info.text):
        return "io"
    if info.kind in ASSIGN_KINDS:
        return "assign_call" if _has_call_syntax(info.text) else "assign"
    if _is_side_effect_like_text(info.text):
        return "side_effect"
    if _has_call_syntax(info.text):
        return "call"
    return info.kind or "other"


def _stmt_infos_connected(a: StmtInfo, b: StmtInfo, line_graph: Optional[nx.DiGraph]) -> bool:
    if line_graph is None:
        return False
    if a.node_id not in line_graph or b.node_id not in line_graph:
        return False
    if line_graph.has_edge(a.node_id, b.node_id) or line_graph.has_edge(b.node_id, a.node_id):
        return True
    sub = line_graph.subgraph([a.node_id, b.node_id]).to_undirected()
    return nx.is_connected(sub) if sub.number_of_nodes() == 2 else False


def _stmt_infos_identifier_overlap(a: StmtInfo, b: StmtInfo) -> bool:
    a_ids = _identifier_set_from_text(a.text)
    b_ids = _identifier_set_from_text(b.text)
    if not a_ids or not b_ids:
        return False
    if a_ids & b_ids:
        return True
    a_lhs = _lhs_identifier_set_from_text(a.text)
    b_lhs = _lhs_identifier_set_from_text(b.text)
    return bool((a_lhs & b_ids) or (b_lhs & a_ids))


def _info_has_leading_comment(info: StmtInfo, source_lines: Optional[List[str]]) -> bool:
    if not source_lines:
        return False
    start = _comment_context_start_line(source_lines, info.start_line)
    return start < info.start_line

def is_opaque_definition_node(stmt: ast.AST) -> bool:
    return isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

def is_control_stmt(stmt: ast.AST) -> bool:
    if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
        return True
    return hasattr(ast, "Match") and isinstance(stmt, ast.Match)

def ast_kind(stmt: ast.AST) -> str:
    if isinstance(stmt, ast.If):
        return "if"
    if isinstance(stmt, ast.For):
        return "for"
    if isinstance(stmt, ast.AsyncFor):
        return "for"
    if isinstance(stmt, ast.While):
        return "while"
    if isinstance(stmt, ast.Try):
        return "try"
    if isinstance(stmt, ast.With):
        return "with"
    if isinstance(stmt, ast.AsyncWith):
        return "with"
    if hasattr(ast, "Match") and isinstance(stmt, ast.Match):
        return "match"
    if isinstance(stmt, ast.Return):
        return "return"
    if isinstance(stmt, ast.Raise):
        return "raise"
    if isinstance(stmt, ast.Break):
        return "break"
    if isinstance(stmt, ast.Continue):
        return "continue"
    if isinstance(stmt, ast.Pass):
        return "pass"
    if isinstance(stmt, ast.Assert):
        return "assert"
    if isinstance(stmt, ast.Assign):
        return "assign"
    if isinstance(stmt, ast.AnnAssign):
        return "annassign"
    if isinstance(stmt, ast.AugAssign):
        return "augassign"
    if isinstance(stmt, ast.Expr):
        return "expr"
    if isinstance(stmt, ast.Import):
        return "import"
    if isinstance(stmt, ast.ImportFrom):
        return "importfrom"
    if isinstance(stmt, ast.Delete):
        return "delete"
    if isinstance(stmt, ast.FunctionDef):
        return "functiondef"
    if isinstance(stmt, ast.AsyncFunctionDef):
        return "asyncfunctiondef"
    if isinstance(stmt, ast.ClassDef):
        return "classdef"
    return stmt.__class__.__name__.lower()

def slice_source_lines(source_lines: List[str], start_line: int, end_line: int) -> str:
    if not source_lines:
        return ""
    start_line = max(1, start_line)
    end_line = max(start_line, end_line)
    end_line = min(end_line, len(source_lines))
    if start_line > len(source_lines) or start_line > end_line:
        return ""
    return "\n".join(source_lines[start_line - 1:end_line]).rstrip()

def indent_of_line(source_lines: List[str], line_no: int) -> str:
    if line_no < 1 or line_no > len(source_lines):
        return ""
    line = source_lines[line_no - 1]
    return re.match(r"^\s*", line).group(0) if line else ""

def first_stmt_line(stmts: List[ast.AST]) -> Optional[int]:
    lines = [getattr(s, "lineno", None) for s in stmts if getattr(s, "lineno", None) is not None]
    return min(lines) if lines else None

def get_stmt_text(source_text: str, source_lines: List[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None:
        return ""
    if end is None:
        end = start
    text = slice_source_lines(source_lines, start, end)
    if text.strip():
        return text.rstrip("\n")
    try:
        seg = ast.get_source_segment(source_text, node)
        if seg is not None:
            return seg.rstrip("\n")
    except Exception:
        pass
    return ""

def control_header_text(
    node: ast.AST,
    source_lines: List[str],
    body_stmts: Optional[List[ast.AST]] = None,
    default_text: str = "",
) -> str:
    start = getattr(node, "lineno", None)
    if start is None:
        return default_text
    end = getattr(node, "end_lineno", start)
    if body_stmts:
        body_start = first_stmt_line(body_stmts)
        if body_start is not None and body_start > start:
            end = body_start - 1
    text = slice_source_lines(source_lines, start, end)
    return text if text.strip() else default_text

def scope_item(kind: str, label: str, line_no: int) -> Dict[str, Any]:
    return {"kind": kind, "label": label, "line_no": line_no}

def scope_signature(scope_chain: List[Dict[str, Any]]) -> str:
    if not scope_chain:
        return "∅"
    return " > ".join(f"{x['kind']}@{x['line_no']}" for x in scope_chain)


# =========================================================
# 4) Python AST 解析（兼容 Python2 风格）
# =========================================================

@lru_cache(maxsize=1)
def _cached_refactor_tool():
    if RefactoringTool is None or get_fixers_from_package is None:
        return None
    try:
        return RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    except Exception:
        return None


def _needs_py2_conversion(source_text: str) -> bool:
    if not source_text:
        return False
    patterns = (
        r"^\s*print\s+>>",
        r"^\s*print\s+[^\(].*$",
        r"^\s*except\s+[^:,\s]+\s*,\s*[A-Za-z_]\w*\s*:",
        r"^\s*raise\s+[A-Za-z_][\w\.]*\s*,\s*.+$",
    )
    return any(re.search(p, source_text, flags=re.MULTILINE) for p in patterns)


def _py2_to_py3_source(source_text: str) -> str:
    source_text = normalize_source(source_text)
    converted = source_text

    if _needs_py2_conversion(source_text):
        tool = _cached_refactor_tool()
        if tool is not None:
            try:
                converted = str(tool.refactor_string(source_text, "snippet.py"))
            except Exception:
                converted = source_text

        converted = re.sub(r'(?m)^(\s*)print\s*>>\s*([^,]+),\s*(.+)$', r'\1print(\3, file=\2)', converted)
        converted = re.sub(
            r'(?m)^(\s*)print\s+(?!\()(.*\S)\s*$',
            lambda m: f"{m.group(1)}print({m.group(2)})",
            converted,
        )
        converted = re.sub(
            r'(?m)^(\s*)except\s+([^:,]+)\s*,\s*([A-Za-z_]\w*)\s*:',
            r'\1except \2 as \3:',
            converted,
        )

        def _fix_old_raise(m: re.Match) -> str:
            indent = m.group(1)
            exc = m.group(2).strip()
            msg = m.group(3).strip()
            msg = re.sub(r',\s*$', '', msg)
            return f"{indent}raise {exc}({msg})"

        converted = re.sub(r'(?m)^(\s*)raise\s+([A-Za-z_][\w\.]*)\s*,\s*(.+)$', _fix_old_raise, converted)

    return converted

def _shift_ast_locations(tree: ast.AST, line_delta: int) -> None:
    for node in ast.walk(tree):
        if hasattr(node, "lineno") and getattr(node, "lineno") is not None:
            node.lineno = max(1, int(node.lineno) + line_delta)
        if hasattr(node, "end_lineno") and getattr(node, "end_lineno") is not None:
            node.end_lineno = max(1, int(node.end_lineno) + line_delta)

def _parse_wrapped_suite_nodes(source_text: str, base_line_offset: int) -> List[ast.stmt]:
    source_text = normalize_source(source_text)
    parse_text = _py2_to_py3_source(source_text)
    expanded = parse_text.expandtabs(4)

    def _extract_body(tree: ast.AST) -> List[ast.stmt]:
        if isinstance(tree, ast.Module) and len(tree.body) == 1 and isinstance(tree.body[0], ast.If):
            return list(getattr(tree.body[0], "body", []))
        if isinstance(tree, ast.Module):
            return list(tree.body)
        return list(getattr(tree, "body", []))

    # 1) 先尝试直接解析去掉公共缩进后的文本。
    for candidate in (expanded, textwrap.dedent(expanded)):
        try:
            tree = ast.parse(candidate)
            module = ast.Module(body=_extract_body(tree), type_ignores=[])
            _shift_ast_locations(module, base_line_offset)
            return list(module.body)
        except (SyntaxError, IndentationError):
            pass

    # 2) 再尝试包裹成 suite。
    try:
        wrapped = "if True:\n" + textwrap.indent(textwrap.dedent(expanded), "    ")
        tree = ast.parse(wrapped)
        suite_nodes = _extract_body(tree)
        module = ast.Module(body=suite_nodes, type_ignores=[])
        _shift_ast_locations(module, base_line_offset - 1)
        return list(module.body)
    except (SyntaxError, IndentationError):
        pass

    # 3) 最后做保守的逐行回退，避免整段丢失。
    body: List[ast.stmt] = []
    lines = source_text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        j = len(lines)
        parsed = False
        while j > i:
            frag = "\n".join(lines[i:j]).rstrip("\n") + "\n"
            frag_parse = _py2_to_py3_source(frag)
            try:
                frag_tree = ast.parse(frag_parse)
                _shift_ast_locations(frag_tree, base_line_offset + i)
                body.extend(getattr(frag_tree, "body", []))
                i = j
                parsed = True
                break
            except Exception:
                j -= 1
        if not parsed:
            i += 1
    return body

def _parse_source_segmented(source_text: str, base_line_offset: int = 0) -> ast.Module:
    source_text = normalize_source(source_text)
    parse_text = _py2_to_py3_source(source_text)
    lines = source_text.splitlines()

    if not any(line.strip() for line in lines):
        return ast.Module(body=[], type_ignores=[])

    # 1) 原样解析；失败后尝试去掉公共缩进再解析。
    for candidate in (parse_text, textwrap.dedent(parse_text)):
        try:
            tree = ast.parse(candidate)
            _shift_ast_locations(tree, base_line_offset)
            if not isinstance(tree, ast.Module):
                tree = ast.Module(body=getattr(tree, "body", []), type_ignores=[])
            return tree
        except (SyntaxError, IndentationError):
            pass

    exp_lines = [line.expandtabs(4) for line in lines]
    first_nonempty = next((i for i, line in enumerate(exp_lines) if line.strip()), None)
    if first_nonempty is None:
        return ast.Module(body=[], type_ignores=[])

    first_indent = len(re.match(r"^[ ]*", exp_lines[first_nonempty]).group(0))
    if first_indent > 0:
        split_idx = len(lines)
        for j in range(first_nonempty + 1, len(lines)):
            if not exp_lines[j].strip():
                continue
            cur_indent = len(re.match(r"^[ ]*", exp_lines[j]).group(0))
            if cur_indent < first_indent:
                split_idx = j
                break

        leading_text = "\n".join(lines[:split_idx]).rstrip("\n") + "\n"
        leading_nodes = _parse_wrapped_suite_nodes(leading_text, base_line_offset)
        body: List[ast.stmt] = list(leading_nodes)
        if split_idx < len(lines):
            rest_text = "\n".join(lines[split_idx:]).rstrip("\n") + "\n"
            rest_tree = _parse_source_segmented(rest_text, base_line_offset + split_idx)
            body.extend(getattr(rest_tree, "body", []))
        return ast.Module(body=body, type_ignores=[])

    # 2) 逐段兜底，避免整段失败时丢块
    body: List[ast.stmt] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        j = len(lines)
        parsed = False
        while j > i:
            frag = "\n".join(lines[i:j]).rstrip("\n") + "\n"
            frag_parse = _py2_to_py3_source(frag)
            try:
                frag_tree = ast.parse(frag_parse)
                _shift_ast_locations(frag_tree, base_line_offset + i)
                body.extend(getattr(frag_tree, "body", []))
                i = j
                parsed = True
                break
            except Exception:
                j -= 1
        if not parsed:
            i += 1
    return ast.Module(body=body, type_ignores=[])

def parse_python_ast(source_text: str) -> ast.AST:
    return _parse_source_segmented(normalize_source(source_text), base_line_offset=0)

def find_target_function(tree: ast.AST, function_name: str) -> ast.AST:
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    raise ValueError(f"在源码中没有找到函数 {function_name}")

def function_span(func_node: ast.AST) -> Tuple[int, int]:
    start_line = getattr(func_node, "lineno", None)
    end_line = getattr(func_node, "end_lineno", None)
    if start_line is None or end_line is None:
        raise ValueError("需要 Python 3.8+ 的 end_lineno 才能精确定位函数范围。")
    decorator_list = getattr(func_node, "decorator_list", [])
    if decorator_list:
        dec_start = min(getattr(d, "lineno", start_line) for d in decorator_list)
        start_line = min(start_line, dec_start)
    return start_line, end_line

def is_standard_function_snippet(tree: ast.AST) -> bool:
    body = [n for n in getattr(tree, "body", []) if not is_docstring_expr(n)]
    return len(body) == 1 and isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef))

def is_standard_class_snippet(tree: ast.AST) -> bool:
    body = [n for n in getattr(tree, "body", []) if not is_docstring_expr(n)]
    return len(body) == 1 and isinstance(body[0], ast.ClassDef)

def iter_child_stmt_lists(node: ast.AST) -> Iterable[List[ast.AST]]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []
    child_lists: List[List[ast.AST]] = []
    if hasattr(node, "body") and isinstance(getattr(node, "body"), list):
        body = getattr(node, "body")
        if body and all(isinstance(x, ast.AST) for x in body):
            child_lists.append(body)
    if isinstance(node, ast.If):
        if node.orelse:
            child_lists.append(node.orelse)
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        if node.orelse:
            child_lists.append(node.orelse)
    elif isinstance(node, ast.Try):
        for h in node.handlers:
            if getattr(h, "body", None):
                child_lists.append(h.body)
        if node.orelse:
            child_lists.append(node.orelse)
        if node.finalbody:
            child_lists.append(node.finalbody)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        if node.body:
            child_lists.append(node.body)
    if hasattr(ast, "Match") and isinstance(node, ast.Match):
        for case in node.cases:
            if getattr(case, "body", None):
                child_lists.append(case.body)
    if isinstance(node, ast.ExceptHandler):
        if getattr(node, "body", None):
            child_lists.append(node.body)
    return child_lists

def collect_stmt_infos(
    source_text: str,
    source_lines: List[str],
    node: ast.AST,
    depth: int,
    stmt_infos: List[StmtInfo],
    stmt_by_ast_id: Dict[int, StmtInfo],
    span_by_line: Dict[int, List[StmtInfo]],
) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        if start_line is None:
            return
        if end_line is None:
            end_line = start_line
        seg = get_stmt_text(source_text, source_lines, node)
        kind = ast_kind(node)
        node_id = f"{start_line}:{end_line}:{kind}:{len(stmt_infos)}"
        info = StmtInfo(node_id, start_line, end_line, kind, seg, depth, False, id(node))
        stmt_infos.append(info)
        stmt_by_ast_id[id(node)] = info
        for line in range(start_line, end_line + 1):
            span_by_line.setdefault(line, []).append(info)
        return

    if not isinstance(node, ast.stmt):
        return
    if is_docstring_expr(node):
        return

    start_line = getattr(node, "lineno", None)
    end_line = getattr(node, "end_lineno", None)
    if start_line is None:
        return
    if end_line is None:
        end_line = start_line
    seg = get_stmt_text(source_text, source_lines, node)
    kind = ast_kind(node)
    node_id = f"{start_line}:{end_line}:{kind}:{len(stmt_infos)}"
    info = StmtInfo(node_id, start_line, end_line, kind, seg, depth, True, id(node))
    stmt_infos.append(info)
    stmt_by_ast_id[id(node)] = info
    for line in range(start_line, end_line + 1):
        span_by_line.setdefault(line, []).append(info)

    for child_list in iter_child_stmt_lists(node):
        for child in child_list:
            collect_stmt_infos(source_text, source_lines, child, depth + 1, stmt_infos, stmt_by_ast_id, span_by_line)

def build_stmt_infos_for_stmts(
    source_text: str,
    source_lines: List[str],
    stmts: List[ast.AST],
) -> Tuple[List[StmtInfo], Dict[int, StmtInfo], Dict[int, List[StmtInfo]]]:
    stmt_infos: List[StmtInfo] = []
    stmt_by_ast_id: Dict[int, StmtInfo] = {}
    span_by_line: Dict[int, List[StmtInfo]] = {}
    for stmt in stmts:
        collect_stmt_infos(source_text, source_lines, stmt, 1, stmt_infos, stmt_by_ast_id, span_by_line)
    stmt_infos.sort(key=lambda x: (x.start_line, x.end_line, x.kind))
    return stmt_infos, stmt_by_ast_id, span_by_line

def build_stmt_infos_for_function(
    source_text: str,
    source_lines: List[str],
    func_node: ast.AST,
) -> Tuple[List[StmtInfo], Dict[int, StmtInfo], Dict[int, List[StmtInfo]]]:
    return build_stmt_infos_for_stmts(source_text, source_lines, getattr(func_node, "body", []))

def build_stmt_infos_for_root(
    source_text: str,
    source_lines: List[str],
    stmts: List[ast.AST],
) -> Tuple[List[StmtInfo], Dict[int, StmtInfo], Dict[int, List[StmtInfo]]]:
    return build_stmt_infos_for_stmts(source_text, source_lines, stmts)

def find_best_stmt_for_line(line_no: int, span_by_line: Dict[int, List[StmtInfo]]) -> Optional[StmtInfo]:
    candidates = span_by_line.get(line_no, [])
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x.span_len, -x.depth, x.start_line, x.end_line))[0]


# =========================================================
# 5) 控制结构复杂度判断
# =========================================================

def _stmt_start_line(node: ast.AST) -> int:
    line = getattr(node, "lineno", None)
    return int(line) if line is not None else -1

def _stmt_end_line(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return int(end)
    line = getattr(node, "lineno", None)
    return int(line) if line is not None else -1

def _span_len(start_line: int, end_line: int) -> int:
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    return end_line - start_line + 1

def _effective_code_line_count(source_lines: List[str], start_line: int, end_line: int) -> int:
    if not source_lines:
        return _span_len(start_line, end_line)
    n = len(source_lines)
    s = max(1, start_line)
    e = min(n, end_line)
    if e < s:
        return 0
    count = 0
    for idx in range(s - 1, e):
        line = source_lines[idx].strip()
        if not line or line.startswith("#"):
            continue
        count += 1
    return count if count > 0 else _span_len(start_line, end_line)

def _section_should_be_whole(source_lines: List[str], start_line: int, end_line: int) -> bool:
    return _effective_code_line_count(source_lines, start_line, end_line) <= CONTROL_LINE_LIMIT


def _suite_effective_code_count(stmts: List[ast.AST], source_lines: List[str]) -> int:
    """Estimate the effective size of a suite using immediate statement spans."""
    total = 0
    for stmt in stmts:
        if stmt is None or is_docstring_expr(stmt):
            continue
        start = _stmt_start_line(stmt)
        end = _stmt_end_line(stmt)
        if start > 0 and end >= start:
            total += _effective_code_line_count(source_lines, start, end)
    return total


def _body_should_be_whole(source_lines: List[str], body_stmts: List[ast.AST]) -> bool:
    """Keep a control statement whole when its direct body is small enough."""
    return _suite_effective_code_count(body_stmts, source_lines) <= CONTROL_LINE_LIMIT


def control_should_be_kept_whole(node: ast.AST, source_lines: Optional[List[str]] = None) -> bool:
    """Fallback rule for control nodes not handled by a dedicated splitter."""
    body = list(getattr(node, "body", [])) if hasattr(node, "body") else []
    if source_lines is not None and body:
        return _body_should_be_whole(source_lines, body)
    start = _stmt_start_line(node)
    end = _stmt_end_line(node)
    if source_lines is not None:
        return _section_should_be_whole(source_lines, start, end)
    return _span_len(start, end) <= CONTROL_LINE_LIMIT

def _whole_block_span(start_line: int, end_line: int) -> Tuple[int, int]:
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    return start_line, end_line

def _header_end_from_body(start_line: int, body_stmts: List[ast.AST]) -> int:
    body_first = first_stmt_line(body_stmts)
    if body_first is None or body_first <= start_line:
        return start_line
    return body_first - 1


# =========================================================
# 6) Joern 调用
# =========================================================

def generate_pdg_with_joern(source_file: Path, work_dir: Path, joern_home: Optional[Path]) -> Path:
    joern_parse = which("joern-parse", joern_home)
    joern_export = which("joern-export", joern_home)

    input_dir = work_dir / "input_src"
    output_dir = work_dir / "pdg_out"

    ensure_dir(input_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    shutil.copy2(source_file, input_dir / source_file.name)

    parse_cmd = [joern_parse, str(input_dir), "--language", "PYTHONSRC"]
    if PYTHON_VENV_DIR.strip():
        parse_cmd += ["--frontend-args", "--venvDir", PYTHON_VENV_DIR.strip()]

    run_cmd(parse_cmd, cwd=work_dir)
    run_cmd([joern_export, "--repr", "pdg", "--out", str(output_dir)], cwd=work_dir)
    return output_dir


# =========================================================
# 7) DOT 解析与合并
# =========================================================

NODE_RE = re.compile(r'^\s*"?(?P<id>[^"\s]+)"?\s*\[(?P<attrs>.*)\]\s*;?\s*$')
EDGE_RE = re.compile(r'^\s*"?(?P<src>[^"\s]+)"?\s*->\s*"?(?P<dst>[^"\s]+)"?(?:\s*\[(?P<attrs>.*)\])?\s*;?\s*$')

def extract_attr_value(attrs: str, key: str) -> str:
    if not attrs:
        return ""
    m = re.search(rf"\b{re.escape(key)}\s*=\s*(.+)$", attrs)
    if not m:
        return ""
    val = m.group(1).strip()
    val = re.sub(r"[;,]\s*$", "", val).strip()
    return val

def parse_label(raw_label: str) -> Tuple[str, Optional[int], str]:
    if not raw_label:
        return "", None, ""
    s = html.unescape(str(raw_label)).strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    s = s.replace("<BR/>", "\n").replace("<BR />", "\n").replace("<BR>", "\n").strip()

    if "\n" in s:
        first, rest = s.split("\n", 1)
    else:
        first, rest = s, ""

    m = re.match(r"^\s*(?P<kind>[^,]+)\s*,\s*(?P<line>\d+)\s*$", first)
    if m:
        return m.group("kind").strip(), int(m.group("line")), normalize_spaces(rest)

    m2 = re.match(r"^\s*(?P<kind>[^,]+)\s*,\s*(?P<line>\d+)\s*(?P<tail>.*)$", first)
    if m2:
        kind = m2.group("kind").strip()
        line_no = int(m2.group("line"))
        tail = m2.group("tail").strip()
        code = normalize_spaces((tail + " " + rest).strip())
        return kind, line_no, code

    return first.strip(), None, normalize_spaces(rest)

def parse_dot_file(dot_file: Path) -> nx.DiGraph:
    G = nx.DiGraph()
    G.graph["name"] = dot_file.name
    G.graph["dot_file"] = str(dot_file)
    text = dot_file.read_text(encoding="utf-8", errors="ignore")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("//") or line.startswith("/*"):
            continue
        if line.startswith("digraph") or line == "{" or line == "}":
            continue
        if line.startswith("graph [") or line.startswith("node [") or line.startswith("edge ["):
            continue
        if line.startswith("subgraph "):
            continue

        nm = NODE_RE.match(raw_line)
        if nm:
            nid = nm.group("id").strip().strip('"')
            attrs = nm.group("attrs")
            label = extract_attr_value(attrs, "label")
            kind, line_no, code = parse_label(label)
            G.add_node(nid, raw_label=label, kind=kind, line_no=line_no, code=code, dot_file=dot_file.name)
            continue

        em = EDGE_RE.match(raw_line)
        if em:
            src = em.group("src").strip().strip('"')
            dst = em.group("dst").strip().strip('"')
            attrs = em.group("attrs") or ""
            label = extract_attr_value(attrs, "label")
            G.add_edge(src, dst, label=label, dot_file=dot_file.name)
            continue

    return G

def _graph_has_module_method(G: nx.DiGraph) -> bool:
    for _, data in G.nodes(data=True):
        if (data.get("kind") or "").strip() == "METHOD":
            code = (data.get("code") or "").strip()
            if code == "<module>" or "<module>" in code:
                return True
    return False

def choose_candidate_graphs(
    dot_dir: Path,
    function_name: str,
    func_start_line: int,
    func_end_line: int,
) -> List[nx.DiGraph]:
    dot_files = sorted(dot_dir.glob("*.dot"))
    if not dot_files:
        raise FileNotFoundError(f"{dot_dir} 下没有找到 .dot 文件")

    parsed: List[nx.DiGraph] = []
    for dot_file in dot_files:
        try:
            parsed.append(parse_dot_file(dot_file))
        except Exception:
            continue

    if not parsed:
        return [parse_dot_file(f) for f in dot_files]

    zero_graphs = [g for g in parsed if Path(g.graph.get("dot_file", "")).name == "0.dot"]
    name_matched: List[nx.DiGraph] = []
    line_matched: List[nx.DiGraph] = []
    module_like: List[nx.DiGraph] = []

    for G in parsed:
        has_method_match = False
        has_line_overlap = False
        has_module = _graph_has_module_method(G)

        for _, data in G.nodes(data=True):
            kind = (data.get("kind") or "").strip()
            code = (data.get("code") or "").strip()
            line_no = data.get("line_no")

            if kind == "METHOD" and code and function_name in code:
                if not code.startswith("<operator>") and code != "<global>":
                    has_method_match = True

            if isinstance(line_no, int) and func_start_line <= line_no <= func_end_line:
                has_line_overlap = True

        if has_method_match:
            name_matched.append(G)
        elif has_line_overlap:
            line_matched.append(G)
        if has_module:
            module_like.append(G)

    if name_matched:
        return name_matched

    if zero_graphs:
        # 0.dot 是优先级最高的回退：若它是 module 图，就用它；否则配合行号回退
        if any(_graph_has_module_method(g) for g in zero_graphs):
            return zero_graphs
        if line_matched:
            return line_matched

    if line_matched:
        return line_matched

    if module_like:
        return module_like

    return parsed

def merge_graphs(graphs: List[nx.DiGraph]) -> nx.DiGraph:
    merged = nx.DiGraph()
    for G in graphs:
        for nid, data in G.nodes(data=True):
            if nid not in merged:
                merged.add_node(nid, **dict(data))
            else:
                cur = merged.nodes[nid]
                for k, v in data.items():
                    if cur.get(k) in (None, "", [], {}):
                        if v not in (None, "", [], {}):
                            cur[k] = v
                    elif k == "raw_label" and v and v != cur.get(k):
                        cur[k] = cur[k] if len(str(cur[k])) >= len(str(v)) else v

        for src, dst, data in G.edges(data=True):
            if merged.has_edge(src, dst):
                existing = merged.edges[src, dst]
                old_label = existing.get("label", "")
                new_label = data.get("label", "")
                labels = set()
                if old_label:
                    labels.add(str(old_label))
                if new_label:
                    labels.add(str(new_label))
                existing["label"] = " | ".join(sorted(labels))
            else:
                merged.add_edge(src, dst, **dict(data))
    return merged


# =========================================================
# 8) 原始 PDG -> 行级语句图
# =========================================================

def build_line_graph_from_merged_pdg(
    merged_raw: nx.DiGraph,
    stmt_infos: List[StmtInfo],
    span_by_line: Dict[int, List[StmtInfo]],
    func_start_line: int,
    func_end_line: int,
) -> nx.DiGraph:
    line_graph = nx.DiGraph()

    for nid, data in merged_raw.nodes(data=True):
        line_no = data.get("line_no")
        if not isinstance(line_no, int):
            continue
        if not (func_start_line <= line_no <= func_end_line):
            continue
        stmt = find_best_stmt_for_line(line_no, span_by_line)
        if stmt is None:
            continue
        kind = (data.get("kind") or "").strip()
        if kind in EXCLUDE_KINDS:
            continue
        stmt.raw_node_ids.append(str(nid))
        stmt.raw_kinds.append(kind)
        raw_label = str(data.get("raw_label") or data.get("code") or "")
        if raw_label:
            stmt.raw_labels.append(raw_label)

    for stmt in stmt_infos:
        line_graph.add_node(
            stmt.node_id,
            line_no=stmt.start_line,
            end_line=stmt.end_line,
            kind=stmt.kind,
            text=stmt.text,
            depth=stmt.depth,
        )

    node_lookup = {stmt.node_id: stmt for stmt in stmt_infos}
    for nid, data in merged_raw.nodes(data=True):
        line_no = data.get("line_no")
        if not isinstance(line_no, int):
            continue
        if not (func_start_line <= line_no <= func_end_line):
            continue
        stmt = find_best_stmt_for_line(line_no, span_by_line)
        if stmt is None:
            continue
        kind = (data.get("kind") or "").strip()
        if kind in EXCLUDE_KINDS:
            continue
        stmt.raw_node_ids.append(str(nid))
        stmt.raw_kinds.append(kind)
        raw_label = str(data.get("raw_label") or data.get("code") or "")
        if raw_label:
            stmt.raw_labels.append(raw_label)

    def best_stmt_for_nid(nid: str) -> Optional[StmtInfo]:
        data = merged_raw.nodes[nid]
        line_no = data.get("line_no")
        if isinstance(line_no, int):
            return find_best_stmt_for_line(line_no, span_by_line)
        return None

    for src, dst, edata in merged_raw.edges(data=True):
        sstmt = best_stmt_for_nid(src)
        dstmt = best_stmt_for_nid(dst)
        if sstmt is None or dstmt is None:
            continue
        if sstmt.node_id == dstmt.node_id:
            continue
        label = (edata.get("label") or "").strip()
        if line_graph.has_edge(sstmt.node_id, dstmt.node_id):
            existing = line_graph.edges[sstmt.node_id, dstmt.node_id]
            etypes = set(existing.get("edge_types", []))
            etypes.add("pdg")
            existing["edge_types"] = sorted(etypes)
            if label:
                labels = set(existing.get("labels", []))
                labels.add(label)
                existing["labels"] = sorted(labels)
        else:
            line_graph.add_edge(
                sstmt.node_id,
                dstmt.node_id,
                labels=[label] if label else [],
                edge_types=["pdg"],
            )

    ordered_stmt_ids = [stmt.node_id for stmt in sorted(stmt_infos, key=lambda x: (x.start_line, x.end_line, x.kind))]
    for a, b in zip(ordered_stmt_ids, ordered_stmt_ids[1:]):
        if a not in line_graph or b not in line_graph:
            continue
        if line_graph.has_edge(a, b):
            existing = line_graph.edges[a, b]
            etypes = set(existing.get("edge_types", []))
            etypes.add("seq")
            existing["edge_types"] = sorted(etypes)
        else:
            line_graph.add_edge(a, b, labels=[], edge_types=["seq"])

    return line_graph


# =========================================================
# 9) 语义块划分
# =========================================================

def stmt_info_for_ast(stmt_by_ast_id: Dict[int, StmtInfo], stmt: ast.AST) -> Optional[StmtInfo]:
    return stmt_by_ast_id.get(id(stmt))

def block_connected(line_graph: Optional[nx.DiGraph], stmt_infos: List[StmtInfo]) -> bool:
    if line_graph is None:
        return False
    if not stmt_infos:
        return False
    if len(stmt_infos) == 1:
        return True
    node_ids = [x.node_id for x in stmt_infos if x.node_id in line_graph]
    if len(node_ids) <= 1:
        return True
    sub = line_graph.subgraph(node_ids).to_undirected()
    return nx.is_connected(sub) if sub.number_of_nodes() > 0 else False

def make_block(
    block_id: int,
    kind: str,
    ast_kind_name: str,
    depth: int,
    scope_chain: List[Dict[str, Any]],
    stmt_infos: List[StmtInfo],
    code: str,
    line_graph: Optional[nx.DiGraph],
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    synthetic: bool = False,
    source_lines: Optional[List[str]] = None,
    attach_leading_comments: bool = False,
    comment_lower_bound: int = 1,
) -> SemanticBlock:
    if stmt_infos:
        s = min(x.start_line for x in stmt_infos) if start_line is None else start_line
        e = max(x.end_line for x in stmt_infos) if end_line is None else end_line
    else:
        s = start_line if start_line is not None else -1
        e = end_line if end_line is not None else -1

    if attach_leading_comments:
        code, s, e = attach_leading_comment_context(
            code,
            source_lines,
            s,
            e,
            lower_bound=comment_lower_bound,
        )

    return SemanticBlock(
        block_id=block_id,
        kind=kind,
        ast_kind=ast_kind_name,
        depth=depth,
        start_line=s,
        end_line=e,
        scope_chain=scope_chain,
        code=code.rstrip("\n"),
        stmt_infos=stmt_infos,
        pdg_connected=block_connected(line_graph, stmt_infos),
        synthetic=synthetic,
    )

def _can_merge_atomic_infos(
    cur: List[StmtInfo],
    nxt: StmtInfo,
    line_graph: Optional[nx.DiGraph],
    source_lines: Optional[List[str]] = None,
) -> bool:
    if not cur:
        return True

    last = cur[-1]
    if last.kind in TERMINAL_KINDS or nxt.kind in TERMINAL_KINDS:
        return False
    if _info_has_leading_comment(nxt, source_lines):
        return False

    gap = nxt.start_line - last.end_line
    span_after = max(nxt.end_line, cur[-1].end_line) - min(cur[0].start_line, nxt.start_line) + 1
    if (
        len(cur) >= MAX_SIMPLE_GROUP_SIZE
        or gap > MAX_SIMPLE_LINE_GAP + 1
        or span_after > MAX_SIMPLE_GROUP_SPAN
    ):
        return False

    cur_roles = {atomic_stmt_role(info) for info in cur}
    nxt_role = atomic_stmt_role(nxt)

    if cur_roles == {"import"} and nxt_role == "import":
        return True
    if "import" in cur_roles or nxt_role == "import":
        return False

    # Keep IO and side-effect calls as crisp blocks unless there is an immediate
    # data relation to the previous statement. These lines are often the evidence
    # a completion task needs, so hiding them inside a generic setup block hurts.
    relation_to_last = _stmt_infos_connected(last, nxt, line_graph) or _stmt_infos_identifier_overlap(last, nxt)
    sensitive_roles = {"io", "side_effect", "call", "assign_call"}
    if nxt_role in sensitive_roles or cur_roles & sensitive_roles:
        return relation_to_last

    if cur_roles <= {"assign"} and nxt_role == "assign":
        return len(cur) < MAX_SIMPLE_INIT_GROUP_SIZE

    return relation_to_last


def _refine_disconnected_atomic_chunk(
    chunk: List[StmtInfo],
    line_graph: Optional[nx.DiGraph],
) -> List[List[StmtInfo]]:
    if line_graph is None or len(chunk) <= 2:
        return [chunk]

    node_ids = [x.node_id for x in chunk if x.node_id in line_graph]
    if len(node_ids) <= 1:
        return [chunk]

    sub = line_graph.subgraph(node_ids).to_undirected()
    if nx.is_connected(sub):
        return [chunk]

    comps = list(nx.connected_components(sub))
    comp_by_node: Dict[str, int] = {}
    for comp_idx, comp in enumerate(comps):
        for node_id in comp:
            comp_by_node[node_id] = comp_idx

    refined: List[List[StmtInfo]] = []
    cur: List[StmtInfo] = []
    cur_key: Any = None
    for info in chunk:
        key = comp_by_node.get(info.node_id)
        if key is None:
            key = ("missing", info.node_id)
        if not cur or key == cur_key:
            cur.append(info)
            cur_key = key
            continue
        refined.append(cur)
        cur = [info]
        cur_key = key
    if cur:
        refined.append(cur)
    return refined


def chunk_atomic_infos(
    infos: List[StmtInfo],
    line_graph: Optional[nx.DiGraph],
    source_lines: Optional[List[str]] = None,
) -> List[List[StmtInfo]]:
    """
    终结语句默认独立；连续原子语句只在有清晰局部语义关系时合并。
    """
    if not infos:
        return []
    infos = sorted(infos, key=lambda x: (x.start_line, x.end_line, x.kind))

    if len(infos) == 1:
        return [infos]

    chunks: List[List[StmtInfo]] = []
    cur: List[StmtInfo] = [infos[0]]
    for nxt in infos[1:]:
        if _can_merge_atomic_infos(cur, nxt, line_graph, source_lines=source_lines):
            cur.append(nxt)
        else:
            chunks.append(cur)
            cur = [nxt]
    if cur:
        chunks.append(cur)

    refined: List[List[StmtInfo]] = []
    for chunk in chunks:
        refined.extend(_refine_disconnected_atomic_chunk(chunk, line_graph))
    return refined

def split_atomic_run_into_blocks(
    atomic_stmts: List[ast.AST],
    scope_chain: List[Dict[str, Any]],
    depth: int,
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    source_lines: List[str],
) -> List[SemanticBlock]:
    infos: List[StmtInfo] = []
    for stmt in atomic_stmts:
        info = stmt_info_for_ast(stmt_by_ast_id, stmt)
        if info is not None:
            infos.append(info)
    if not infos:
        return []

    blocks: List[SemanticBlock] = []
    for chunk in chunk_atomic_infos(infos, line_graph, source_lines=source_lines):
        rendered_parts = [render_stmt_text_with_context(x, source_lines) for x in chunk]
        code = "\n".join(rendered_parts).rstrip()
        blocks.append(
            make_block(
                block_id=0,
                kind="atomic",
                ast_kind_name="group" if len(chunk) > 1 else chunk[0].kind,
                depth=depth,
                scope_chain=scope_chain,
                stmt_infos=chunk,
                code=code,
                line_graph=line_graph,
            )
        )
    return blocks

def collect_branch_header_block(
    header_kind: str,
    header_text: str,
    line_no: int,
    scope_chain: List[Dict[str, Any]],
    depth: int,
    line_graph: Optional[nx.DiGraph],
    stmt_info: Optional[StmtInfo] = None,
    synthetic: bool = False,
    end_line: Optional[int] = None,
    source_lines: Optional[List[str]] = None,
) -> SemanticBlock:
    stmt_infos = [stmt_info] if stmt_info is not None else []
    start = stmt_info.start_line if stmt_info is not None else line_no
    end = stmt_info.start_line if stmt_info is not None else line_no if end_line is None else end_line
    if end_line is not None:
        end = end_line
    return make_block(
        block_id=0,
        kind="control",
        ast_kind_name=header_kind,
        depth=depth,
        scope_chain=scope_chain,
        stmt_infos=stmt_infos,
        code=header_text,
        line_graph=line_graph,
        start_line=start,
        end_line=end,
        synthetic=synthetic,
        source_lines=source_lines,
        attach_leading_comments=True,
    )

def _make_whole_section_block(
    *,
    section_kind: str,
    start_line: int,
    end_line: int,
    source_lines: List[str],
    scope_chain: List[Dict[str, Any]],
    depth: int,
    line_graph: Optional[nx.DiGraph],
    stmt_info: Optional[StmtInfo] = None,
    synthetic: bool = False,
) -> SemanticBlock:
    code = slice_source_lines(source_lines, start_line, end_line)
    return make_block(
        block_id=0,
        kind="control",
        ast_kind_name=section_kind,
        depth=depth,
        scope_chain=scope_chain,
        stmt_infos=[stmt_info] if stmt_info is not None else [],
        code=code,
        line_graph=line_graph,
        start_line=start_line,
        end_line=end_line,
        synthetic=synthetic or stmt_info is None,
        source_lines=source_lines,
        attach_leading_comments=True,
    )

def _emit_section_or_recurse(
    *,
    section_kind: str,
    header_text: str,
    start_line: int,
    end_line: int,
    body_stmts: List[ast.AST],
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    base_scope_chain: List[Dict[str, Any]],
    branch_scope: List[Dict[str, Any]],
    depth: int,
    stmt_info: Optional[StmtInfo] = None,
    synthetic: bool = False,
) -> List[SemanticBlock]:
    start_line, end_line = _whole_block_span(start_line, end_line)
    if _body_should_be_whole(source_lines, body_stmts):
        return [_make_whole_section_block(
            section_kind=section_kind,
            start_line=start_line, end_line=end_line,
            source_lines=source_lines, scope_chain=branch_scope,
            depth=depth, line_graph=line_graph,
            stmt_info=stmt_info, synthetic=synthetic,
        )]

    header_end = _header_end_from_body(start_line, body_stmts)
    blocks = [collect_branch_header_block(
        header_kind=section_kind, header_text=header_text, line_no=start_line,
        scope_chain=base_scope_chain, depth=depth, line_graph=line_graph,
        stmt_info=stmt_info, synthetic=synthetic, end_line=header_end,
        source_lines=source_lines,
    )]

    if body_stmts:
        blocks.extend(segment_suite(
            body_stmts, source_text, source_lines, line_graph, stmt_by_ast_id,
            branch_scope, depth + 1
        ))
    return blocks


def split_definition_node(
    stmt: ast.AST,
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    scope_chain: List[Dict[str, Any]],
    depth: int,
    recurse_body: bool = True,
) -> List[SemanticBlock]:
    info = stmt_info_for_ast(stmt_by_ast_id, stmt)
    if info is None:
        return []

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body_first_line = first_stmt_line(getattr(stmt, "body", []))
        header_end = body_first_line - 1 if body_first_line is not None and body_first_line > info.start_line else info.start_line
        header_text = control_header_text(stmt, source_lines, body_stmts=getattr(stmt, "body", []), default_text=f"def {getattr(stmt, 'name', 'function')}(...):")
        if not header_text.strip():
            header_text = f"def {getattr(stmt, 'name', 'function')}(...):"
        def_block = make_block(
            0, "definition", info.kind, depth, scope_chain, [info], header_text, line_graph,
            start_line=info.start_line, end_line=header_end, synthetic=False,
            source_lines=source_lines, attach_leading_comments=True,
        )
    else:
        decorator_list = getattr(stmt, "decorator_list", [])
        start_line = getattr(stmt, "lineno", info.start_line)
        if decorator_list:
            start_line = min(start_line, min(getattr(d, "lineno", start_line) for d in decorator_list))
        body_first_line = first_stmt_line(getattr(stmt, "body", []))
        header_end = body_first_line - 1 if body_first_line is not None and body_first_line > start_line else start_line
        header_text = control_header_text(stmt, source_lines, body_stmts=getattr(stmt, "body", []), default_text=f"class {getattr(stmt, 'name', 'Class')}:")
        if not header_text.strip():
            header_text = f"class {getattr(stmt, 'name', 'Class')}:"
        def_block = make_block(
            0, "definition", info.kind, depth, scope_chain, [info], header_text, line_graph,
            start_line=start_line, end_line=header_end, synthetic=False,
            source_lines=source_lines, attach_leading_comments=True,
        )

    blocks: List[SemanticBlock] = [def_block]

    if recurse_body and hasattr(stmt, "body"):
        body_stmts = [s for s in getattr(stmt, "body", []) if not is_docstring_expr(s)]
        _, local_stmt_by_ast_id, _ = build_stmt_infos_for_stmts(source_text, source_lines, body_stmts)
        blocks.extend(segment_suite(
            body_stmts,
            source_text=source_text,
            source_lines=source_lines,
            line_graph=None,
            stmt_by_ast_id=local_stmt_by_ast_id,
            scope_chain=scope_chain + [scope_item(info.kind, getattr(stmt, "name", info.kind), info.start_line)],
            depth=depth + 1,
        ))
    return blocks

def segment_suite(
    stmts: List[ast.AST],
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    scope_chain: List[Dict[str, Any]],
    depth: int,
) -> List[SemanticBlock]:
    blocks: List[SemanticBlock] = []
    pending_atomic: List[ast.AST] = []

    def flush_pending_atomic() -> None:
        nonlocal pending_atomic
        if not pending_atomic:
            return
        blocks.extend(split_atomic_run_into_blocks(
            pending_atomic, scope_chain=scope_chain, depth=depth,
            line_graph=line_graph, stmt_by_ast_id=stmt_by_ast_id,
            source_lines=source_lines,
        ))
        pending_atomic = []

    for stmt in stmts:
        if stmt is None or is_docstring_expr(stmt):
            continue

        if is_opaque_definition_node(stmt):
            flush_pending_atomic()
            blocks.extend(split_definition_node(
                stmt=stmt, source_text=source_text, source_lines=source_lines,
                line_graph=line_graph, stmt_by_ast_id=stmt_by_ast_id,
                scope_chain=scope_chain, depth=depth, recurse_body=True
            ))
            continue

        if is_simple_stmt_node(stmt) or is_terminal_stmt_node(stmt):
            pending_atomic.append(stmt)
            continue

        if is_control_stmt(stmt):
            flush_pending_atomic()
            blocks.extend(segment_control_stmt(
                stmt=stmt, source_text=source_text, source_lines=source_lines,
                line_graph=line_graph, stmt_by_ast_id=stmt_by_ast_id,
                scope_chain=scope_chain, depth=depth
            ))
            continue

        pending_atomic.append(stmt)

    flush_pending_atomic()
    return blocks

def segment_control_stmt(
    stmt: ast.AST,
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    scope_chain: List[Dict[str, Any]],
    depth: int,
) -> List[SemanticBlock]:
    if isinstance(stmt, ast.If):
        return segment_if_chain(stmt, source_text, source_lines, line_graph, stmt_by_ast_id, scope_chain, depth)
    if isinstance(stmt, ast.Try):
        return segment_try_stmt(stmt, source_text, source_lines, line_graph, stmt_by_ast_id, scope_chain, depth)
    if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        return segment_loop_stmt(stmt, source_text, source_lines, line_graph, stmt_by_ast_id, scope_chain, depth)
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return segment_with_stmt(stmt, source_text, source_lines, line_graph, stmt_by_ast_id, scope_chain, depth)
    if hasattr(ast, "Match") and isinstance(stmt, ast.Match):
        return segment_match_stmt(stmt, source_text, source_lines, line_graph, stmt_by_ast_id, scope_chain, depth)

    if control_should_be_kept_whole(stmt, source_lines):
        info = stmt_info_for_ast(stmt_by_ast_id, stmt)
        if info is None:
            start = _stmt_start_line(stmt)
            end = _stmt_end_line(stmt)
            if start > 0 and end >= start:
                return [_make_whole_section_block(
                    section_kind=ast_kind(stmt), start_line=start, end_line=end,
                    source_lines=source_lines, scope_chain=scope_chain,
                    depth=depth, line_graph=line_graph, synthetic=True
                )]
            return []
        return [make_block(
            0, "control", info.kind, depth, scope_chain, [info], info.text, line_graph,
            synthetic=False, source_lines=source_lines, attach_leading_comments=True,
        )]

    info = stmt_info_for_ast(stmt_by_ast_id, stmt)
    if info is None:
        return []
    return [make_block(
        0, "control", info.kind, depth, scope_chain, [info], info.text, line_graph,
        synthetic=False, source_lines=source_lines, attach_leading_comments=True,
    )]

def _indent_width_at_line(source_lines: List[str], line_no: int) -> int:
    if line_no < 1 or line_no > len(source_lines):
        return -1
    return len(re.match(r"^[ ]*", source_lines[line_no - 1].expandtabs(4)).group(0))


def _last_body_end_line(body_stmts: List[ast.AST]) -> int:
    ends = [_stmt_end_line(s) for s in body_stmts if isinstance(s, ast.AST)]
    ends = [e for e in ends if e > 0]
    return max(ends) if ends else -1


def _is_actual_elif(current: ast.If, candidate: ast.AST, source_lines: List[str]) -> bool:
    if not isinstance(candidate, ast.If):
        return False
    cur_line = _stmt_start_line(current)
    cand_line = _stmt_start_line(candidate)
    if cur_line <= 0 or cand_line <= 0:
        return False
    cur_indent = _indent_width_at_line(source_lines, cur_line)
    cand_indent = _indent_width_at_line(source_lines, cand_line)
    return cur_indent >= 0 and cand_indent >= 0 and cand_indent == cur_indent


def _if_chain_next_elif(current: ast.If, source_lines: List[str]) -> Optional[ast.If]:
    if len(current.orelse) != 1:
        return None
    candidate = current.orelse[0]
    if _is_actual_elif(current, candidate, source_lines):
        return candidate
    return None


def _branch_code_end_for_small_clause(current: ast.If, source_lines: List[str], next_clause_start: Optional[int]) -> int:
    start_line = _stmt_start_line(current)
    body_end = _last_body_end_line(list(getattr(current, "body", [])))
    if body_end < start_line:
        body_end = start_line
    return body_end
def _find_else_header_line(current: ast.If, source_lines: List[str]) -> int:
    body_first = first_stmt_line(list(current.orelse))
    if body_first is None:
        return _stmt_start_line(current)
    cur_indent = _indent_width_at_line(source_lines, _stmt_start_line(current))
    search_start = max(1, body_first - 1)
    search_end = max(1, _stmt_start_line(current) + 1)
    for line_no in range(search_start, search_end - 1, -1):
        if line_no < 1 or line_no > len(source_lines):
            continue
        stripped = source_lines[line_no - 1].strip()
        if not stripped:
            continue
        if _indent_width_at_line(source_lines, line_no) != cur_indent:
            continue
        if stripped.startswith('else:') or stripped == 'else:':
            return line_no
    return max(_stmt_start_line(current) + 1, body_first - 1)




def segment_if_chain(
    if_node: ast.If,
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    scope_chain: List[Dict[str, Any]],
    depth: int,
) -> List[SemanticBlock]:
    blocks: List[SemanticBlock] = []
    current: ast.If = if_node
    branch_kind = "if"

    while True:
        info = stmt_info_for_ast(stmt_by_ast_id, current)
        start_line = info.start_line if info is not None else _stmt_start_line(current)
        body_stmts = list(getattr(current, "body", []))
        header = control_header_text(current, source_lines, body_stmts=body_stmts, default_text=f"{branch_kind}:")

        next_elif = _if_chain_next_elif(current, source_lines)
        next_clause_start = _stmt_start_line(next_elif) if next_elif is not None else None
        branch_scope = scope_chain + [scope_item(branch_kind, header, getattr(current, "lineno", -1))]

        if _body_should_be_whole(source_lines, body_stmts):
            branch_end = _branch_code_end_for_small_clause(current, source_lines, next_clause_start)
            blocks.append(_make_whole_section_block(
                section_kind=branch_kind,
                start_line=start_line,
                end_line=branch_end,
                source_lines=source_lines,
                scope_chain=branch_scope,
                depth=depth,
                line_graph=line_graph,
                stmt_info=info,
                synthetic=(info is None),
            ))
        else:
            header_end = _header_end_from_body(start_line, body_stmts)
            blocks.append(collect_branch_header_block(
                header_kind=branch_kind,
                header_text=header,
                line_no=start_line,
                scope_chain=scope_chain,
                depth=depth,
                line_graph=line_graph,
                stmt_info=info,
                synthetic=(info is None),
                end_line=header_end,
                source_lines=source_lines,
            ))
            blocks.extend(segment_suite(
                body_stmts,
                source_text,
                source_lines,
                line_graph,
                stmt_by_ast_id,
                branch_scope,
                depth + 1,
            ))

        if next_elif is not None:
            current = next_elif
            branch_kind = "elif"
            continue

        if current.orelse:
            else_line = _find_else_header_line(current, source_lines)
            else_end = _last_body_end_line(list(current.orelse))
            if else_end < else_line:
                else_end = _stmt_end_line(current)
            else_header = indent_of_line(source_lines, else_line) + "else:"
            else_scope = scope_chain + [scope_item("else", else_header, else_line)]
            if _body_should_be_whole(source_lines, list(current.orelse)):
                blocks.append(_make_whole_section_block(
                    section_kind="else",
                    start_line=else_line,
                    end_line=else_end,
                    source_lines=source_lines,
                    scope_chain=else_scope,
                    depth=depth,
                    line_graph=line_graph,
                    stmt_info=None,
                    synthetic=True,
                ))
            else:
                blocks.append(collect_branch_header_block(
                    header_kind="else",
                    header_text=else_header,
                    line_no=else_line,
                    scope_chain=scope_chain,
                    depth=depth,
                    line_graph=line_graph,
                    stmt_info=None,
                    synthetic=True,
                    end_line=else_line,
                    source_lines=source_lines,
                ))
                blocks.extend(segment_suite(
                    list(current.orelse),
                    source_text,
                    source_lines,
                    line_graph,
                    stmt_by_ast_id,
                    else_scope,
                    depth + 1,
                ))
        break
    return blocks


def segment_loop_stmt(
    stmt: ast.AST,
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    scope_chain: List[Dict[str, Any]],
    depth: int,
) -> List[SemanticBlock]:
    info = stmt_info_for_ast(stmt_by_ast_id, stmt)
    kind = ast_kind(stmt)
    body_stmts = list(getattr(stmt, "body", []))
    header_text = control_header_text(stmt, source_lines, body_stmts=body_stmts, default_text=f"{kind}:")
    start_line = info.start_line if info is not None else _stmt_start_line(stmt)
    overall_end = _stmt_end_line(stmt)
    orelse = getattr(stmt, "orelse", [])

    if _body_should_be_whole(source_lines, body_stmts):
        return [_make_whole_section_block(
            section_kind=kind,
            start_line=start_line,
            end_line=overall_end,
            source_lines=source_lines,
            scope_chain=scope_chain,
            depth=depth,
            line_graph=line_graph,
            stmt_info=info,
            synthetic=(info is None),
        )]

    body_end = overall_end
    if orelse:
        first_orelse_line = first_stmt_line(orelse)
        if first_orelse_line is not None and first_orelse_line > 1:
            body_end = min(body_end, first_orelse_line - 2)

    body_scope = scope_chain + [scope_item(kind, header_text, start_line)]
    blocks = _emit_section_or_recurse(
        section_kind=kind,
        header_text=header_text,
        start_line=start_line,
        end_line=body_end,
        body_stmts=body_stmts,
        source_text=source_text,
        source_lines=source_lines,
        line_graph=line_graph,
        stmt_by_ast_id=stmt_by_ast_id,
        base_scope_chain=scope_chain,
        branch_scope=body_scope,
        depth=depth,
        stmt_info=info,
        synthetic=(info is None),
    )

    if orelse:
        first_orelse_line = first_stmt_line(orelse)
        else_line = (first_orelse_line - 1) if first_orelse_line is not None and first_orelse_line > 0 else overall_end
        else_header = indent_of_line(source_lines, getattr(stmt, "lineno", 1)) + "else:"
        else_scope = scope_chain + [scope_item("else", else_header, else_line)]
        if _body_should_be_whole(source_lines, orelse):
            blocks.append(_make_whole_section_block(
                section_kind="else",
                start_line=else_line,
                end_line=overall_end,
                source_lines=source_lines,
                scope_chain=else_scope,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
            ))
        else:
            blocks.append(collect_branch_header_block(
                header_kind="else",
                header_text=else_header,
                line_no=else_line,
                scope_chain=scope_chain,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
                end_line=else_line,
                source_lines=source_lines,
            ))
            blocks.extend(segment_suite(
                orelse,
                source_text,
                source_lines,
                line_graph,
                stmt_by_ast_id,
                else_scope,
                depth + 1,
            ))
    return blocks


def segment_with_stmt(
    stmt: ast.AST,
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    scope_chain: List[Dict[str, Any]],
    depth: int,
) -> List[SemanticBlock]:
    info = stmt_info_for_ast(stmt_by_ast_id, stmt)
    kind = ast_kind(stmt)
    body_stmts = list(getattr(stmt, "body", []))
    header_text = control_header_text(stmt, source_lines, body_stmts=body_stmts, default_text=f"{kind}:")
    start_line = info.start_line if info is not None else _stmt_start_line(stmt)
    end_line = _stmt_end_line(stmt)
    body_scope = scope_chain + [scope_item(kind, header_text, start_line)]
    if _body_should_be_whole(source_lines, body_stmts):
        return [_make_whole_section_block(
            section_kind=kind,
            start_line=start_line,
            end_line=end_line,
            source_lines=source_lines,
            scope_chain=scope_chain,
            depth=depth,
            line_graph=line_graph,
            stmt_info=info,
            synthetic=(info is None),
        )]
    return _emit_section_or_recurse(
        section_kind=kind,
        header_text=header_text,
        start_line=start_line,
        end_line=end_line,
        body_stmts=body_stmts,
        source_text=source_text,
        source_lines=source_lines,
        line_graph=line_graph,
        stmt_by_ast_id=stmt_by_ast_id,
        base_scope_chain=scope_chain,
        branch_scope=body_scope,
        depth=depth,
        stmt_info=info,
        synthetic=(info is None),
    )


def match_case_start_line(case: Any) -> Optional[int]:
    pattern = getattr(case, "pattern", None)
    guard = getattr(case, "guard", None)
    candidates = []
    if pattern is not None and getattr(pattern, "lineno", None) is not None:
        candidates.append(pattern.lineno)
    if guard is not None and getattr(guard, "lineno", None) is not None:
        candidates.append(guard.lineno)
    body_start = first_stmt_line(getattr(case, "body", []))
    if body_start is not None:
        candidates.append(body_start)
    return min(candidates) if candidates else None

def match_case_header_text(case: Any, source_lines: List[str]) -> str:
    start = match_case_start_line(case)
    if start is None:
        return "case:"
    body_start = first_stmt_line(getattr(case, "body", []))
    end = start if body_start is None or body_start <= start else body_start - 1
    text = slice_source_lines(source_lines, start, end)
    return text if text.strip() else "case:"

def _next_clause_end_line(next_clause_lines: List[int], overall_end: int, start_line: int) -> int:
    valid_lines = [line for line in next_clause_lines if isinstance(line, int) and line > 0]
    end_line = min(valid_lines) - 1 if valid_lines else overall_end
    return max(end_line, start_line)


def segment_try_stmt(
    stmt: ast.Try,
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    scope_chain: List[Dict[str, Any]],
    depth: int,
) -> List[SemanticBlock]:
    """Split try/except/else/finally like the if/elif/else chain.

    Rule:
    - Each clause is handled independently.
    - If a clause body is short enough (<= CONTROL_LINE_LIMIT effective lines),
      keep the whole clause as one block.
    - Otherwise, emit the clause header block and recurse into its body.
    """
    blocks: List[SemanticBlock] = []
    info = stmt_info_for_ast(stmt_by_ast_id, stmt)
    start_line = info.start_line if info is not None else getattr(stmt, "lineno", -1)
    overall_end = _stmt_end_line(stmt)
    header = control_header_text(stmt, source_lines, body_stmts=stmt.body, default_text="try:")

    # ---------- try: clause ----------
    try_next_clause_lines: List[int] = []
    for handler in stmt.handlers:
        if getattr(handler, "lineno", None) is not None:
            try_next_clause_lines.append(int(handler.lineno))
    if stmt.orelse:
        else_first = first_stmt_line(stmt.orelse)
        if else_first is not None and else_first > 0:
            try_next_clause_lines.append(else_first - 1)
    if stmt.finalbody:
        fin_first = first_stmt_line(stmt.finalbody)
        if fin_first is not None and fin_first > 0:
            try_next_clause_lines.append(fin_first - 1)
    try_end = _next_clause_end_line(try_next_clause_lines, overall_end, start_line)

    try_scope = scope_chain + [scope_item("try", header, getattr(stmt, "lineno", -1))]
    if _body_should_be_whole(source_lines, stmt.body):
        blocks.append(_make_whole_section_block(
            section_kind="try",
            start_line=start_line,
            end_line=try_end,
            source_lines=source_lines,
            scope_chain=try_scope,
            depth=depth,
            line_graph=line_graph,
            stmt_info=info,
            synthetic=(info is None),
        ))
    else:
        blocks.append(collect_branch_header_block(
            header_kind="try",
            header_text=header,
            line_no=start_line,
            scope_chain=scope_chain,
            depth=depth,
            line_graph=line_graph,
            stmt_info=info,
            synthetic=(info is None),
            end_line=_header_end_from_body(start_line, stmt.body),
            source_lines=source_lines,
        ))
        blocks.extend(segment_suite(
            stmt.body,
            source_text,
            source_lines,
            line_graph,
            stmt_by_ast_id,
            try_scope,
            depth + 1,
        ))

    # ---------- except: clauses ----------
    for idx, handler in enumerate(stmt.handlers):
        handler_label = control_header_text(handler, source_lines, body_stmts=handler.body, default_text="except:")
        handler_line = getattr(handler, "lineno", getattr(stmt, "lineno", -1))

        next_clause_lines: List[int] = []
        for nxt in stmt.handlers[idx + 1:]:
            if getattr(nxt, "lineno", None) is not None:
                next_clause_lines.append(int(nxt.lineno))
        if stmt.orelse:
            else_first = first_stmt_line(stmt.orelse)
            if else_first is not None and else_first > 0:
                next_clause_lines.append(else_first - 1)
        if stmt.finalbody:
            fin_first = first_stmt_line(stmt.finalbody)
            if fin_first is not None and fin_first > 0:
                next_clause_lines.append(fin_first - 1)
        handler_end = _next_clause_end_line(next_clause_lines, overall_end, handler_line)

        except_scope = scope_chain + [scope_item("except", handler_label, handler_line)]
        if _body_should_be_whole(source_lines, handler.body):
            blocks.append(_make_whole_section_block(
                section_kind="except",
                start_line=handler_line,
                end_line=handler_end,
                source_lines=source_lines,
                scope_chain=except_scope,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
            ))
        else:
            blocks.append(collect_branch_header_block(
                header_kind="except",
                header_text=handler_label,
                line_no=handler_line,
                scope_chain=scope_chain,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
                end_line=_header_end_from_body(handler_line, handler.body),
                source_lines=source_lines,
            ))
            blocks.extend(segment_suite(
                handler.body,
                source_text,
                source_lines,
                line_graph,
                stmt_by_ast_id,
                except_scope,
                depth + 1,
            ))

    # ---------- else: clause ----------
    if stmt.orelse:
        else_first = first_stmt_line(stmt.orelse)
        else_line = (else_first - 1) if else_first is not None and else_first > 0 else overall_end

        next_clause_lines = []
        if stmt.finalbody:
            fin_first = first_stmt_line(stmt.finalbody)
            if fin_first is not None and fin_first > 0:
                next_clause_lines.append(fin_first - 1)
        else_end = _next_clause_end_line(next_clause_lines, overall_end, else_line)

        else_header = indent_of_line(source_lines, getattr(stmt, "lineno", 1)) + "else:"
        else_scope = scope_chain + [scope_item("else", else_header, else_line)]
        if _body_should_be_whole(source_lines, stmt.orelse):
            blocks.append(_make_whole_section_block(
                section_kind="else",
                start_line=else_line,
                end_line=else_end,
                source_lines=source_lines,
                scope_chain=else_scope,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
            ))
        else:
            blocks.append(collect_branch_header_block(
                header_kind="else",
                header_text=else_header,
                line_no=else_line,
                scope_chain=scope_chain,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
                end_line=else_line,
                source_lines=source_lines,
            ))
            blocks.extend(segment_suite(
                stmt.orelse,
                source_text,
                source_lines,
                line_graph,
                stmt_by_ast_id,
                else_scope,
                depth + 1,
            ))

    # ---------- finally: clause ----------
    if stmt.finalbody:
        fin_first = first_stmt_line(stmt.finalbody)
        fin_line = (fin_first - 1) if fin_first is not None and fin_first > 0 else overall_end
        fin_scope = scope_chain + [scope_item("finally", indent_of_line(source_lines, getattr(stmt, "lineno", 1)) + "finally:", fin_line)]
        if _body_should_be_whole(source_lines, stmt.finalbody):
            blocks.append(_make_whole_section_block(
                section_kind="finally",
                start_line=fin_line,
                end_line=overall_end,
                source_lines=source_lines,
                scope_chain=fin_scope,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
            ))
        else:
            blocks.append(collect_branch_header_block(
                header_kind="finally",
                header_text=indent_of_line(source_lines, getattr(stmt, "lineno", 1)) + "finally:",
                line_no=fin_line,
                scope_chain=scope_chain,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
                end_line=fin_line,
                source_lines=source_lines,
            ))
            blocks.extend(segment_suite(
                stmt.finalbody,
                source_text,
                source_lines,
                line_graph,
                stmt_by_ast_id,
                fin_scope,
                depth + 1,
            ))

    return blocks


def segment_match_stmt(
    stmt: ast.AST,
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
    scope_chain: List[Dict[str, Any]],
    depth: int,
) -> List[SemanticBlock]:
    info = stmt_info_for_ast(stmt_by_ast_id, stmt)
    start_line = info.start_line if info is not None else _stmt_start_line(stmt)
    end_line = _stmt_end_line(stmt)
    cases = list(getattr(stmt, "cases", []))
    case_bodies = [c for c in cases if getattr(c, "body", None)]
    header = control_header_text(stmt, source_lines, body_stmts=case_bodies, default_text="match:")
    if _body_should_be_whole(source_lines, case_bodies):
        return [_make_whole_section_block(
            section_kind="match",
            start_line=start_line,
            end_line=end_line,
            source_lines=source_lines,
            scope_chain=scope_chain,
            depth=depth,
            line_graph=line_graph,
            stmt_info=info,
            synthetic=(info is None),
        )]

    blocks: List[SemanticBlock] = [collect_branch_header_block(
        header_kind="match", header_text=header, line_no=start_line,
        scope_chain=scope_chain, depth=depth, line_graph=line_graph,
        stmt_info=info, synthetic=(info is None), source_lines=source_lines,
    )]
    match_scope = scope_chain + [scope_item("match", header, getattr(stmt, "lineno", -1))]
    for idx, case in enumerate(cases):
        case_body = getattr(case, "body", [])
        case_line = match_case_start_line(case) or getattr(stmt, "lineno", -1)
        case_label = match_case_header_text(case, source_lines)
        next_case_line = match_case_start_line(cases[idx + 1]) if idx + 1 < len(cases) else None
        case_end = end_line
        if next_case_line is not None and next_case_line > 0:
            case_end = next_case_line - 1
        case_scope = match_scope + [scope_item("case", case_label, case_line)]
        if _body_should_be_whole(source_lines, case_body):
            blocks.append(_make_whole_section_block(
                section_kind="case",
                start_line=case_line,
                end_line=case_end,
                source_lines=source_lines,
                scope_chain=case_scope,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
            ))
        else:
            blocks.append(collect_branch_header_block(
                header_kind="case",
                header_text=case_label,
                line_no=case_line,
                scope_chain=scope_chain,
                depth=depth,
                line_graph=line_graph,
                stmt_info=None,
                synthetic=True,
            ))
            blocks.extend(segment_suite(case_body, source_text, source_lines, line_graph, stmt_by_ast_id, case_scope, depth + 1))
    return blocks


def build_function_header_block(func_node: ast.AST, source_lines: List[str], line_graph: Optional[nx.DiGraph] = None) -> SemanticBlock:
    func_name = getattr(func_node, "name", "function")
    try:
        func_start_line, _ = function_span(func_node)
    except Exception:
        func_start_line = getattr(func_node, "lineno", -1)

    body = list(getattr(func_node, "body", []))
    body_first_line = first_stmt_line(body)

    header_end_line = func_start_line
    header_text = f"def {func_name}(...):"

    # 把函数头下紧跟的 docstring 视为函数说明，纳入 Block 1。
    # 这样 Block 1 会变成：def 行 + 三引号多行注释；后续语义块从真正可执行语句开始。
    if body and is_docstring_expr(body[0]):
        doc_node = body[0]
        doc_end = getattr(doc_node, "end_lineno", getattr(doc_node, "lineno", func_start_line))
        if doc_end is not None and doc_end >= func_start_line:
            header_end_line = int(doc_end)
            header_text = slice_source_lines(source_lines, func_start_line, header_end_line)
            if not header_text.strip():
                header_text = control_header_text(
                    func_node,
                    source_lines,
                    body_stmts=body[1:],
                    default_text=f"def {func_name}(...):",
                )
    else:
        if body_first_line is not None and func_start_line > 0 and body_first_line > func_start_line:
            header_end_line = body_first_line - 1
        header_text = control_header_text(
            func_node,
            source_lines,
            body_stmts=body,
            default_text=f"def {func_name}(...):",
        )
        if not header_text.strip():
            header_text = f"def {func_name}(...):"

    return make_block(
        0,
        "definition",
        ast_kind(func_node),
        0,
        [],
        [],
        header_text,
        line_graph,
        start_line=func_start_line,
        end_line=header_end_line,
        synthetic=True,
        source_lines=source_lines,
        attach_leading_comments=True,
    )

def build_class_header_block(class_node: ast.ClassDef, source_lines: List[str], line_graph: Optional[nx.DiGraph] = None) -> SemanticBlock:
    class_start = getattr(class_node, "lineno", -1)
    decorator_list = getattr(class_node, "decorator_list", [])
    if decorator_list:
        class_start = min(class_start, min(getattr(d, "lineno", class_start) for d in decorator_list))
    body_first_line = first_stmt_line(getattr(class_node, "body", []))
    header_end = body_first_line - 1 if body_first_line is not None and body_first_line > class_start else class_start
    header_text = control_header_text(class_node, source_lines, body_stmts=getattr(class_node, "body", []), default_text=f"class {class_node.name}:")
    if not header_text.strip():
        header_text = f"class {class_node.name}:"
    return make_block(
        0, "definition", "classdef", 0, [], [], header_text, line_graph,
        start_line=class_start, end_line=header_end, synthetic=True,
        source_lines=source_lines, attach_leading_comments=True,
    )

def build_semantic_blocks(
    func_node: ast.AST,
    source_text: str,
    source_lines: List[str],
    line_graph: Optional[nx.DiGraph],
    stmt_by_ast_id: Dict[int, StmtInfo],
) -> List[SemanticBlock]:
    blocks: List[SemanticBlock] = []
    header_block = build_function_header_block(func_node, source_lines, line_graph=line_graph)
    if header_block.code.strip():
        blocks.append(header_block)

    body_blocks = segment_suite(
        getattr(func_node, "body", []), source_text, source_lines, line_graph,
        stmt_by_ast_id, [], 1
    )
    blocks.extend(body_blocks)

    blocks = [b for b in blocks if b.code.strip() or b.stmt_infos or b.synthetic]
    blocks.sort(key=lambda b: (b.start_line if b.start_line >= 0 else 10**9, b.end_line if b.end_line >= 0 else 10**9, b.depth, b.kind))

    meaningful = [b for b in blocks if b.code.strip()]
    if len(meaningful) <= 1:
        alt = build_suite_semantic_blocks(
            [n for n in getattr(func_node, "body", []) if not is_docstring_expr(n)],
            source_text,
            source_lines,
            depth=1,
        )
        alt = [b for b in alt if b.code.strip() or b.stmt_infos or b.synthetic]
        if alt:
            blocks = ([header_block] if header_block.code.strip() else []) + alt

    for idx, blk in enumerate(blocks, 1):
        blk.block_id = idx
    return blocks

def build_class_semantic_blocks(class_node: ast.ClassDef, source_text: str, source_lines: List[str]) -> List[SemanticBlock]:
    body_stmts = [n for n in getattr(class_node, "body", []) if not is_docstring_expr(n)]
    _, stmt_by_ast_id, _ = build_stmt_infos_for_stmts(source_text, source_lines, body_stmts)
    blocks: List[SemanticBlock] = []
    header_block = build_class_header_block(class_node, source_lines, line_graph=None)
    if header_block.code.strip():
        blocks.append(header_block)
    blocks.extend(build_suite_semantic_blocks(body_stmts, source_text, source_lines, depth=1))
    blocks = [b for b in blocks if b.code.strip() or b.stmt_infos or b.synthetic]
    blocks.sort(key=lambda b: (b.start_line if b.start_line >= 0 else 10**9, b.end_line if b.end_line >= 0 else 10**9, b.depth, b.kind))
    for idx, blk in enumerate(blocks, 1):
        blk.block_id = idx
    return blocks

def build_suite_semantic_blocks(stmts: List[ast.AST], source_text: str, source_lines: List[str], depth: int = 0) -> List[SemanticBlock]:
    stmts = [n for n in stmts if not is_docstring_expr(n)]
    _, stmt_by_ast_id, _ = build_stmt_infos_for_stmts(source_text, source_lines, stmts)
    blocks = segment_suite(stmts, source_text, source_lines, line_graph=None, stmt_by_ast_id=stmt_by_ast_id, scope_chain=[], depth=depth)
    blocks = [b for b in blocks if b.code.strip() or b.stmt_infos or b.synthetic]
    blocks.sort(key=lambda b: (b.start_line if b.start_line >= 0 else 10**9, b.end_line if b.end_line >= 0 else 10**9, b.depth, b.kind))
    for idx, blk in enumerate(blocks, 1):
        blk.block_id = idx
    return blocks

def build_module_semantic_blocks(tree: ast.AST, source_text: str, source_lines: List[str]) -> List[SemanticBlock]:
    return build_suite_semantic_blocks(getattr(tree, "body", []), source_text, source_lines, depth=0)


# =========================================================
# 10) 输出
# =========================================================

def stmt_info_record(info: StmtInfo) -> Dict[str, Any]:
    return {
        "node_id": info.node_id,
        "start_line": info.start_line,
        "end_line": info.end_line,
        "kind": info.kind,
        "text": info.text,
        "depth": info.depth,
        "is_exec": info.is_exec,
        "ast_node_id": info.ast_node_id,
        "raw_node_ids": info.raw_node_ids,
        "raw_kinds": info.raw_kinds,
        "raw_labels": info.raw_labels,
    }

def block_to_record(block: SemanticBlock) -> Dict[str, Any]:
    return {
        "block_id": block.block_id,
        "kind": block.kind,
        "ast_kind": block.ast_kind,
        "depth": block.depth,
        "start_line": block.start_line,
        "end_line": block.end_line,
        "scope_chain": block.scope_chain,
        "scope_signature": scope_signature(block.scope_chain),
        "code": block.code,
        "pdg_connected": block.pdg_connected,
        "synthetic": block.synthetic,
        "stmt_infos": [stmt_info_record(s) for s in block.stmt_infos],
        "node_ids": block.node_ids,
        "stmt_count": len(block.stmt_infos),
    }

def print_blocks(blocks: List[SemanticBlock]) -> None:
    print("\n" + "=" * 100)
    print("语义块划分结果")
    print("=" * 100)
    for block in blocks:
        print(f"\n📦 Block {block.block_id}")
        print("-" * 100)
        print(
            f"kind={block.kind}  ast_kind={block.ast_kind}  depth={block.depth}  "
            f"lines={block.start_line}-{block.end_line}  pdg_connected={block.pdg_connected}  synthetic={block.synthetic}"
        )
        print(f"scope: {scope_signature(block.scope_chain)}")
        print(block.code)

def save_results(
    out_dir: Path,
    source_text: str,
    source_lines: List[str],
    func_node: ast.AST,
    raw_graph: nx.DiGraph,
    line_graph: nx.DiGraph,
    blocks: List[SemanticBlock],
    dot_files_used: List[str],
) -> None:
    ensure_dir(out_dir)
    func_start_line, func_end_line = function_span(func_node)

    json_data = {
        "mode": "function_pdg",
        "function_name": getattr(func_node, "name", FUNCTION_NAME),
        "function_start_line": func_start_line,
        "function_end_line": func_end_line,
        "dot_files_used": dot_files_used,
        "raw_graph": {"node_count": raw_graph.number_of_nodes(), "edge_count": raw_graph.number_of_edges()},
        "line_graph": {
            "node_count": line_graph.number_of_nodes(),
            "edge_count": line_graph.number_of_edges(),
            "nodes": [{"id": nid, **data} for nid, data in line_graph.nodes(data=True)],
            "edges": [{"src": src, "dst": dst, **data} for src, dst, data in line_graph.edges(data=True)],
        },
        "blocks": [block_to_record(block) for block in blocks],
        "source_lines": [{"line_no": i + 1, "text": line} for i, line in enumerate(source_lines)],
        "source_text": source_text,
    }

    (out_dir / "semantic_blocks.json").write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_lines: List[str] = []
    txt_lines.append("=" * 100)
    txt_lines.append("语义块划分结果")
    txt_lines.append("=" * 100)
    txt_lines.append(f"函数：{getattr(func_node, 'name', FUNCTION_NAME)}")
    txt_lines.append(f"函数范围：{func_start_line}-{func_end_line}")
    txt_lines.append(f"使用的 dot 文件数：{len(dot_files_used)}")
    txt_lines.append(f"原始图：节点 {raw_graph.number_of_nodes()}，边 {raw_graph.number_of_edges()}")
    txt_lines.append(f"行级图：节点 {line_graph.number_of_nodes()}，边 {line_graph.number_of_edges()}")
    txt_lines.append(f"语义块数：{len(blocks)}")
    for block in blocks:
        txt_lines.append("")
        txt_lines.append(f"📦 Block {block.block_id}")
        txt_lines.append("-" * 100)
        txt_lines.append(
            f"kind={block.kind}  ast_kind={block.ast_kind}  depth={block.depth}  "
            f"lines={block.start_line}-{block.end_line}  pdg_connected={block.pdg_connected}  synthetic={block.synthetic}"
        )
        txt_lines.append(f"scope: {scope_signature(block.scope_chain)}")
        txt_lines.append(block.code)

    (out_dir / "semantic_blocks.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"\n[OK] 结果已保存到：{out_dir}")
    print("     - semantic_blocks.json")
    print("     - semantic_blocks.txt")

def save_results_ast_only(
    out_dir: Path,
    source_text: str,
    source_lines: List[str],
    root_kind: str,
    root_name: str,
    blocks: List[SemanticBlock],
    root_start_line: Optional[int] = None,
    root_end_line: Optional[int] = None,
) -> None:
    ensure_dir(out_dir)
    json_data = {
        "mode": f"{root_kind}_ast_only",
        "root_kind": root_kind,
        "root_name": root_name,
        "root_start_line": root_start_line,
        "root_end_line": root_end_line,
        "blocks": [block_to_record(block) for block in blocks],
        "source_lines": [{"line_no": i + 1, "text": line} for i, line in enumerate(source_lines)],
        "source_text": source_text,
    }
    (out_dir / "semantic_blocks.json").write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_lines: List[str] = []
    txt_lines.append("=" * 100)
    txt_lines.append("语义块划分结果（AST-only）")
    txt_lines.append("=" * 100)
    txt_lines.append(f"根节点类型：{root_kind}")
    txt_lines.append(f"根节点名称：{root_name}")
    if root_start_line is not None and root_end_line is not None:
        txt_lines.append(f"根节点范围：{root_start_line}-{root_end_line}")
    txt_lines.append(f"语义块数：{len(blocks)}")
    for block in blocks:
        txt_lines.append("")
        txt_lines.append(f"📦 Block {block.block_id}")
        txt_lines.append("-" * 100)
        txt_lines.append(
            f"kind={block.kind}  ast_kind={block.ast_kind}  depth={block.depth}  "
            f"lines={block.start_line}-{block.end_line}  pdg_connected={block.pdg_connected}  synthetic={block.synthetic}"
        )
        txt_lines.append(f"scope: {scope_signature(block.scope_chain)}")
        txt_lines.append(block.code)
    (out_dir / "semantic_blocks.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"\n[OK] 结果已保存到：{out_dir}")
    print("     - semantic_blocks.json")
    print("     - semantic_blocks.txt")


# =========================================================
# 11) 主流程
# =========================================================

def main() -> None:
    joern_home = Path(JOERN_HOME).expanduser().resolve()
    if not joern_home.exists():
        raise FileNotFoundError(f"JOERN_HOME 不存在：{joern_home}")

    source_text_raw = normalize_source(SOURCE_CODE)
    indent_prefix = detect_leading_indent_prefix(source_text_raw)

    source_text = dedent_source_text(source_text_raw)

    if indent_prefix:
        print(f"[INFO] 检测到 SOURCE_CODE 存在统一前置缩进：{len(indent_prefix)} 个空格，已逐行左移后再解析；输出时会恢复该缩进。")

    source_lines = source_text.splitlines()
    source_lines_raw = source_text_raw.splitlines()
    tree = parse_python_ast(source_text)

    work_dir = Path.cwd() / f"pdg_work_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(work_dir)

    source_file = work_dir / f"test_func{SOURCE_SUFFIX}"
    source_file.write_text(source_text, encoding="utf-8")

    blocks_output_prefix = indent_prefix

    # 0) 缩进 suite：AST-only
    if getattr(tree, "_wrapped_suite_mode", False):
        print("[INFO] 模式：indented-suite / AST-only")
        suite_stmts = []
        if getattr(tree, "body", []):
            top = tree.body[0]
            if isinstance(top, ast.If) and isinstance(top.test, ast.Constant) and top.test.value is True:
                suite_stmts = getattr(top, "body", [])
            else:
                suite_stmts = getattr(tree, "body", [])
        blocks = build_suite_semantic_blocks(suite_stmts, source_text, source_lines, depth=0)
        blocks_out = reindent_blocks_for_output(blocks, blocks_output_prefix, source_lines=source_lines)
        print_blocks(blocks_out)
        out_dir = work_dir / "results"
        save_results_ast_only(out_dir=out_dir, source_text=source_text_raw, source_lines=source_lines_raw,
                              root_kind="suite", root_name="indented_block", blocks=blocks_out)
        return

    # 1) 标准 def：走 PDG
    if is_standard_function_snippet(tree):
        func_node = find_target_function(tree, getattr(tree.body[0], "name"))
        func_start_line, func_end_line = function_span(func_node)

        print("[INFO] 模式：function")
        print(f"[INFO] 函数名：{func_node.name}")
        print(f"[INFO] 函数起止行：{func_start_line}-{func_end_line}")

        pdg_dir = generate_pdg_with_joern(source_file, work_dir, joern_home)
        print(f"[INFO] PDG 导出目录：{pdg_dir}")

        candidate_graphs = choose_candidate_graphs(pdg_dir, func_node.name, func_start_line, func_end_line)
        dot_files_used = sorted({g.graph.get("dot_file", "") for g in candidate_graphs if g.graph.get("dot_file")})
        print(f"[INFO] 命中的 dot 数量：{len(candidate_graphs)}")
        print(f"[INFO] 使用的 dot：{dot_files_used}")

        merged_raw = merge_graphs(candidate_graphs)
        print(f"[INFO] 合并后原始图：节点数 {merged_raw.number_of_nodes()}，边数 {merged_raw.number_of_edges()}")

        stmt_infos, stmt_by_ast_id, span_by_line = build_stmt_infos_for_function(
            source_text=source_text, source_lines=source_lines, func_node=func_node
        )

        line_graph = build_line_graph_from_merged_pdg(
            merged_raw=merged_raw,
            stmt_infos=stmt_infos,
            span_by_line=span_by_line,
            func_start_line=func_start_line,
            func_end_line=func_end_line,
        )
        print(f"[INFO] 行级图：节点数 {line_graph.number_of_nodes()}，边数 {line_graph.number_of_edges()}")

        blocks = build_semantic_blocks(
            func_node=func_node,
            source_text=source_text,
            source_lines=source_lines,
            line_graph=line_graph,
            stmt_by_ast_id=stmt_by_ast_id,
        )
        blocks_out = reindent_blocks_for_output(blocks, blocks_output_prefix, source_lines=source_lines)
        print_blocks(blocks_out)

        out_dir = work_dir / "results"
        save_results(
            out_dir=out_dir,
            source_text=source_text_raw,
            source_lines=source_lines_raw,
            func_node=func_node,
            raw_graph=merged_raw,
            line_graph=line_graph,
            blocks=blocks_out,
            dot_files_used=dot_files_used,
        )
        return

    # 2) class：AST-only
    if is_standard_class_snippet(tree):
        class_node = tree.body[0]
        print("[INFO] 模式：class-only")
        blocks = build_class_semantic_blocks(class_node, source_text, source_lines)
        blocks_out = reindent_blocks_for_output(blocks, blocks_output_prefix, source_lines=source_lines)
        print_blocks(blocks_out)
        out_dir = work_dir / "results"
        save_results_ast_only(
            out_dir=out_dir,
            source_text=source_text_raw,
            source_lines=source_lines_raw,
            root_kind="class",
            root_name=getattr(class_node, "name", "class"),
            blocks=blocks_out,
            root_start_line=getattr(class_node, "lineno", None),
            root_end_line=getattr(class_node, "end_lineno", None),
        )
        return

    # 3) module：AST-only
    print("[INFO] 模式：module-only")
    blocks = build_module_semantic_blocks(tree, source_text, source_lines)
    blocks_out = reindent_blocks_for_output(blocks, blocks_output_prefix, source_lines=source_lines)
    print_blocks(blocks_out)
    out_dir = work_dir / "results"
    save_results_ast_only(
        out_dir=out_dir,
        source_text=source_text_raw,
        source_lines=source_lines_raw,
        root_kind="module",
        root_name="module",
        blocks=blocks_out,
    )

if __name__ == "__main__":
    main()
