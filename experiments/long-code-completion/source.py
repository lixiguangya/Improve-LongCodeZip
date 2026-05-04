#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdg_semantic_block_py.py

目标：
1) 输入是一个 Python 函数（不是整个程序）
2) 调用 Joern 生成 PDG 并导出所有 .dot
3) 过滤出与目标函数相关的 dot，并合并成完整图
4) 将完整图投影为“行级语句图”
   - 每个节点都是一个完整语句
   - 节点带有行号（start_line/end_line）
   - 过滤掉方法、参数、标识符等非语句节点
5) 结合 Python AST + PDG，对语义块进行划分
   - 顶层顺序语句合并为块
   - if/elif/else、for、while、try、with、match 等复合语句作为结构块
   - 图用于投影、连通性校验和块内依赖展示
6) 输出 JSON / TXT

依赖：
    pip install networkx

注意：
- 本版本不依赖 pydot / pygraphviz
- 直接手写解析 Joern 导出的 dot
- 针对 Python 函数，使用 AST 定位函数范围和语句范围，比括号计数更稳
"""

from __future__ import annotations

import ast
import datetime
import html
import json
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterable

import networkx as nx


# =========================================================
# 1. 配置区
# =========================================================

JOERN_HOME = r"/home/zhangmanqing/wyh/joern-cli"

# 输入：一个 Python 函数
SOURCE_CODE = r"""
	def net_install(self,after_download):
        # initialise the profile, from the server if any
        if self.profile:
            profile_data = self.get_data("profile",self.profile)
        elif self.system:
            profile_data = self.get_data("system",self.system)
        elif self.image:
            profile_data = self.get_data("image",self.image)
        else:
            # shouldn't end up here, right?
            profile_data = {}

        if profile_data.get("kickstart","") != "":

            # fix URLs
            if profile_data["kickstart"][0] == "/" or profile_data["template_remote_kickstarts"]:
               if not self.system:
                   profile_data["kickstart"] = "http://%s/cblr/svc/op/ks/profile/%s" % (profile_data['http_server'], profile_data['name'])
               else:
                   profile_data["kickstart"] = "http://%s/cblr/svc/op/ks/system/%s" % (profile_data['http_server'], profile_data['name'])
                
            # find_kickstart source tree in the kickstart file
            self.get_install_tree_from_kickstart(profile_data)

            # if we found an install_tree, and we don't have a kernel or initrd
            # use the ones in the install_tree
            if self.safe_load(profile_data,"install_tree"):
                if not self.safe_load(profile_data,"kernel"):
                    profile_data["kernel"] = profile_data["install_tree"] + "/images/pxeboot/vmlinuz"

                if not self.safe_load(profile_data,"initrd"):
                    profile_data["initrd"] = profile_data["install_tree"] + "/images/pxeboot/initrd.img"


        # find the correct file download location 
        if not self.is_virt:
            if os.path.exists("/boot/efi/EFI/redhat/elilo.conf"):
                # elilo itanium support, may actually still work
                download = "/boot/efi/EFI/redhat"
            else:
                # whew, we have a sane bootloader
                download = "/boot"

        else:
            # ensure we have a good virt type choice and know where
            # to download the kernel/initrd
            if self.virt_type is None:
                self.virt_type = self.safe_load(profile_data,'virt_type',default=None)
            if self.virt_type is None or self.virt_type == "":
                self.virt_type = "auto"

            # if virt type is auto, reset it to a value we can actually use
            if self.virt_type == "auto":

                if profile_data.get("xml_file","") != "":
                    raise InfoException("xmlfile based installations are not supported")

                elif profile_data.has_key("file"):
                    print "- ISO or Image based installation, always uses --virt-type=qemu"
                    self.virt_type = "qemu"
                    
                else:
                    # FIXME: auto never selects vmware, maybe it should if we find it?

                    if not ANCIENT_PYTHON:
                        cmd = sub_process.Popen("/bin/uname -r", stdout=sub_process.PIPE, shell=True)
                        uname_str = cmd.communicate()[0]
                        if uname_str.find("xen") != -1:
                            self.virt_type = "xenpv"
                        elif os.path.exists("/usr/bin/qemu-img"):
                            self.virt_type = "qemu"
                        else:
                            # assume Xen, we'll check to see if virt-type is really usable later.
                            raise InfoException, "Not running a Xen kernel and qemu is not installed"

                print "- no virt-type specified, auto-selecting %s" % self.virt_type

            # now that we've figured out our virt-type, let's see if it is really usable
            # rather than showing obscure error messages from Xen to the user :)

            if self.virt_type in [ "xenpv", "xenfv" ]:
                cmd = sub_process.Popen("uname -r", stdout=sub_process.PIPE, shell=True)
                uname_str = cmd.communicate()[0]
                # correct kernel on dom0?
                if uname_str.find("xen") == -1:
                   raise InfoException("kernel-xen needs to be in use")
                # xend installed?
                if not os.path.exists("/usr/sbin/xend"):
                   raise InfoException("xen package needs to be installed")
                # xend running?
                rc = sub_process.call("/usr/sbin/xend status", stderr=None, stdout=None, shell=True)
                if rc != 0:
                   raise InfoException("xend needs to be started")

            # for qemu
            if self.virt_type == "qemu":
                # qemu package installed?
                if not os.path.exists("/usr/bin/qemu-img"):
                    raise InfoException("qemu package needs to be installed")
                # is libvirt new enough?
                cmd = sub_process.Popen("rpm -q python-virtinst", stdout=sub_process.PIPE, shell=True)
                version_str = cmd.communicate()[0]
                if version_str.find("virtinst-0.1") != -1 or version_str.find("virtinst-0.0") != -1:
                    raise InfoException("need python-virtinst >= 0.2 to do installs for qemu/kvm")

            # for vmware
            if self.virt_type == "vmware" or self.virt_type == "vmwarew":
                # FIXME: if any vmware specific checks are required (for deps) do them here.
                pass

            if self.virt_type == "virt-image":
                if not os.path.exists("/usr/bin/virt-image"):
                    raise InfoException("virt-image not present, downlevel virt-install package?")

            # for both virt types
            if os.path.exists("/etc/rc.d/init.d/libvirtd"):
                rc = sub_process.call("/sbin/service libvirtd status", stdout=None, shell=True)
                if rc != 0:
                    # libvirt running?
                    raise InfoException("libvirtd needs to be running")


            if self.virt_type in [ "xenpv" ]:
                # we need to fetch the kernel/initrd to do this
                download = "/var/lib/xen" 
            elif self.virt_type in [ "xenfv", "vmware", "vmwarew" ] :
                # we are downloading sufficient metadata to initiate PXE, no D/L needed
                download = None 
            else: # qemu
                # fullvirt, can use set_location in virtinst library, no D/L needed yet
                download = None 

        # download required files
        if not self.is_display and download is not None:
           self.get_distro_files(profile_data, download)
  
        # perform specified action
        after_download(self, profile_data)
"""

SOURCE_SUFFIX = ".py"
FUNCTION_NAME = "net_install"

# 如果函数依赖第三方包，可填虚拟环境路径；不需要就留空字符串
PYTHON_VENV_DIR = ""  # 例如："/some/path/venv"

# Joern 里常见的非语句节点类型
EXCLUDE_KINDS = {
    "METHOD",
    "METHOD_RETURN",
    "PARAM",
    "PARAMETER",
    "PARAMETER_IN",
    "PARAMETER_OUT",
    "BLOCK",
    "LOCAL",
    "TYPE",
    "TYPE_DECL",
    "TYPE_REF",
    "MEMBER",
    "FIELD_IDENTIFIER",
    "IDENTIFIER",
    "LITERAL",
}

# Python 语义结构块核心类型
CONTROL_AST_KINDS = {
    "if",
    "for",
    "while",
    "try",
    "with",
    "match",
}

TERMINAL_AST_KINDS = {
    "return",
    "raise",
    "break",
    "continue",
}

SIMPLE_SINGLETON_KINDS = {
    "pass",
    "assert",
    "assign",
    "annassign",
    "augassign",
    "expr",
    "import",
    "importfrom",
    "delete",
}

# =========================================================
# 2. 数据结构
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
    raw_node_ids: List[str]
    raw_kinds: List[str]
    raw_labels: List[str]

    @property
    def span_len(self) -> int:
        return max(1, self.end_line - self.start_line + 1)


@dataclass
class Region:
    region_kind: str  # simple / compound / terminal
    ast_kind: str
    stmt_infos: List[StmtInfo]

    @property
    def start_line(self) -> int:
        return min(x.start_line for x in self.stmt_infos)

    @property
    def end_line(self) -> int:
        return max(x.end_line for x in self.stmt_infos)

    @property
    def node_ids(self) -> List[str]:
        return [x.node_id for x in self.stmt_infos]


# =========================================================
# 3. 通用工具
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
    """
    去掉前后空白并统一缩进，适合内嵌 Python 函数源码。
    """
    return textwrap.dedent(source).strip("\n") + "\n"


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def strip_comments_and_strings(line: str) -> str:
    """
    轻量版：去掉单行注释和字符串内容，足够用于定位/分类。
    """
    s = line
    s = re.sub(r"#.*$", "", s)
    s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
    s = re.sub(r"'(?:\\.|[^'\\])*'", "''", s)
    return s


def is_docstring_expr(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def is_simple_stmt_node(stmt: ast.AST) -> bool:
    return isinstance(
        stmt,
        (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Expr,
            ast.Pass,
            ast.Assert,
            ast.Import,
            ast.ImportFrom,
            ast.Delete,
        ),
    )


def is_terminal_stmt_node(stmt: ast.AST) -> bool:
    return isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue))


def is_compound_stmt_node(stmt: ast.AST) -> bool:
    if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
        return True
    if hasattr(ast, "Match") and isinstance(stmt, getattr(ast, "Match")):
        return True
    return False


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
    if hasattr(ast, "Match") and isinstance(stmt, getattr(ast, "Match")):
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
    return stmt.__class__.__name__.lower()


def node_sort_key_from_line(start_line: int, end_line: int, kind: str) -> Tuple[int, int, str]:
    return (start_line, end_line, kind)


# =========================================================
# 4. Python AST：定位目标函数和语句范围
# =========================================================

def parse_python_ast(source_text: str) -> ast.AST:
    return ast.parse(source_text)


def find_target_function(tree: ast.AST, function_name: str) -> ast.AST:
    """
    优先找顶层函数；找不到再全树搜索。
    支持 FunctionDef / AsyncFunctionDef。
    """
    # 顶层优先
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node

    # 全树兜底
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node

    raise ValueError(f"在源码中没有找到函数 {function_name}")


def function_span(func_node: ast.AST) -> Tuple[int, int]:
    """
    返回函数起止行号（1-based）。
    如果有装饰器，起始行为最上方装饰器行。
    """
    start_line = getattr(func_node, "lineno", None)
    end_line = getattr(func_node, "end_lineno", None)

    if start_line is None or end_line is None:
        raise ValueError("Python AST 需要 end_lineno 才能精确定位函数范围，请使用较新的 Python 版本。")

    decorator_list = getattr(func_node, "decorator_list", [])
    if decorator_list:
        dec_start = min(getattr(d, "lineno", start_line) for d in decorator_list)
        start_line = min(start_line, dec_start)

    return start_line, end_line


def iter_child_stmt_lists(node: ast.AST) -> Iterable[List[ast.AST]]:
    """
    返回一个节点内部所有“子语句列表”。
    用于递归收集函数内全部语句。
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # 不进入嵌套函数/类，避免把嵌套定义混进当前函数语义块
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
            if hasattr(h, "body") and isinstance(h.body, list) and h.body:
                child_lists.append(h.body)
        if node.orelse:
            child_lists.append(node.orelse)
        if node.finalbody:
            child_lists.append(node.finalbody)

    elif isinstance(node, ast.With):
        pass

    elif isinstance(node, ast.AsyncWith):
        pass

    elif hasattr(ast, "Match") and isinstance(node, getattr(ast, "Match")):
        for case in node.cases:
            if hasattr(case, "body") and isinstance(case.body, list) and case.body:
                child_lists.append(case.body)

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
    """
    递归收集函数内全部“语句节点”。
    只保留真正的语句，不保留嵌套函数/类定义，不保留 docstring。
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # 不把嵌套函数/类作为当前函数块的一部分
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

    # 语句文本：优先使用 AST segment，失败则回退到行切片
    seg = None
    try:
        seg = ast.get_source_segment(source_text, node)
    except Exception:
        seg = None

    if seg is None:
        seg = "\n".join(source_lines[start_line - 1 : end_line])
    seg = seg.rstrip("\n")

    kind = ast_kind(node)
    is_exec = True

    node_id = f"{start_line}:{end_line}:{kind}:{len(stmt_infos)}"

    info = StmtInfo(
        node_id=node_id,
        start_line=start_line,
        end_line=end_line,
        kind=kind,
        text=seg,
        depth=depth,
        is_exec=is_exec,
        ast_node_id=id(node),
        raw_node_ids=[],
        raw_kinds=[],
        raw_labels=[],
    )

    stmt_infos.append(info)
    stmt_by_ast_id[id(node)] = info

    for line in range(start_line, end_line + 1):
        span_by_line.setdefault(line, []).append(info)

    for child_list in iter_child_stmt_lists(node):
        for child in child_list:
            collect_stmt_infos(
                source_text=source_text,
                source_lines=source_lines,
                node=child,
                depth=depth + 1,
                stmt_infos=stmt_infos,
                stmt_by_ast_id=stmt_by_ast_id,
                span_by_line=span_by_line,
            )


def build_stmt_infos_for_function(
    source_text: str,
    source_lines: List[str],
    func_node: ast.AST,
) -> Tuple[List[StmtInfo], Dict[int, StmtInfo], Dict[int, List[StmtInfo]]]:
    """
    返回：
      - 全部语句信息（函数内）
      - AST node id -> StmtInfo
      - line_no -> 所有覆盖该行的 StmtInfo（用于 raw PDG 节点映射）
    """
    stmt_infos: List[StmtInfo] = []
    stmt_by_ast_id: Dict[int, StmtInfo] = {}
    span_by_line: Dict[int, List[StmtInfo]] = {}

    # 先收集函数顶层 body
    body = getattr(func_node, "body", [])
    for stmt in body:
        collect_stmt_infos(
            source_text=source_text,
            source_lines=source_lines,
            node=stmt,
            depth=1,
            stmt_infos=stmt_infos,
            stmt_by_ast_id=stmt_by_ast_id,
            span_by_line=span_by_line,
        )

    stmt_infos.sort(key=lambda x: (x.start_line, x.end_line, x.kind))
    return stmt_infos, stmt_by_ast_id, span_by_line


def find_best_stmt_for_line(line_no: int, span_by_line: Dict[int, List[StmtInfo]]) -> Optional[StmtInfo]:
    """
    一个 raw 节点的 line_no 可能落在多层语句 span 内。
    这里选“覆盖该行的最具体语句”：
      1) span 最短优先
      2) depth 更深优先
      3) start_line 更小优先
    """
    candidates = span_by_line.get(line_no, [])
    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda x: (x.span_len, -x.depth, x.start_line, x.end_line),
    )
    return candidates[0]


# =========================================================
# 5. Joern 调用
# =========================================================

def generate_pdg_with_joern(source_file: Path, work_dir: Path, joern_home: Optional[Path]) -> Path:
    """
    Python frontend：
      joern-parse input_dir --language PYTHONSRC [--frontend-args ...]
      joern-export --repr pdg --out outdir
    """
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
# 6. DOT 解析与合并
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
    """
    解析 Joern dot label：
        METHOD, 4<BR/>analyze_orders
        CONTROL_STRUCTURE, 8<BR/>if x
        CALL, 10<BR/>foo(...)
    """
    if not raw_label:
        return "", None, ""

    s = html.unescape(str(raw_label)).strip()

    # 去掉外层引号
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()

    # 去掉可能的外层 <>
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()

    # 统一换行
    s = s.replace("<BR/>", "\n").replace("<BR />", "\n").replace("<BR>", "\n").strip()

    if "\n" in s:
        first, rest = s.split("\n", 1)
    else:
        first, rest = s, ""

    # first 形如：METHOD, 4
    m = re.match(r"^\s*(?P<kind>[^,]+)\s*,\s*(?P<line>\d+)\s*$", first)
    if m:
        kind = m.group("kind").strip()
        line_no = int(m.group("line"))
        code = normalize_spaces(rest)
        return kind, line_no, code

    # first 形如：METHOD, 4 some tail
    m2 = re.match(r"^\s*(?P<kind>[^,]+)\s*,\s*(?P<line>\d+)\s*(?P<tail>.*)$", first)
    if m2:
        kind = m2.group("kind").strip()
        line_no = int(m2.group("line"))
        tail = m2.group("tail").strip()
        code = normalize_spaces((tail + " " + rest).strip())
        return kind, line_no, code

    return first.strip(), None, normalize_spaces(rest)


def parse_dot_file(dot_file: Path) -> nx.DiGraph:
    """
    手写解析 Joern 导出的 dot，避免 pydot 兼容问题。
    """
    G = nx.DiGraph()
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

            G.add_node(
                nid,
                raw_label=label,
                kind=kind,
                line_no=line_no,
                code=code,
                dot_file=dot_file.name,
            )
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


def choose_candidate_graphs(
    dot_dir: Path,
    function_name: str,
    func_start_line: int,
    func_end_line: int,
) -> List[nx.DiGraph]:
    """
    过滤出与目标 Python 函数相关的 dot：
    1) 优先 METHOD 名称匹配 function_name
    2) 其次按节点 line_no 与函数行范围重叠
    3) 如果都没有，则作为兜底返回所有 dot
    """
    dot_files = sorted(dot_dir.glob("*.dot"))
    if not dot_files:
        raise FileNotFoundError(f"{dot_dir} 下没有找到 .dot 文件")

    name_matched: List[nx.DiGraph] = []
    line_matched: List[nx.DiGraph] = []

    for dot_file in dot_files:
        try:
            G = parse_dot_file(dot_file)
        except Exception:
            continue

        has_method_match = False
        has_line_overlap = False

        for _, data in G.nodes(data=True):
            kind = (data.get("kind") or "").strip()
            code = (data.get("code") or "").strip()
            line_no = data.get("line_no")

            # 目标函数 METHOD
            if kind == "METHOD" and code and function_name in code:
                # 排除全局 / operator
                if not code.startswith("<operator>") and code != "<global>":
                    has_method_match = True

            # 行号重叠
            if isinstance(line_no, int) and func_start_line <= line_no <= func_end_line:
                has_line_overlap = True

        if has_method_match:
            name_matched.append(G)
        elif has_line_overlap:
            line_matched.append(G)

    if name_matched:
        return name_matched
    if line_matched:
        return line_matched

    return [parse_dot_file(f) for f in dot_files]


def merge_graphs(graphs: List[nx.DiGraph]) -> nx.DiGraph:
    """
    合并多个 dot 图：
    - 节点按 id union
    - 边按 (src, dst) union
    """
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
                        # 保留更完整的 label 信息
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
# 7. 原始 PDG -> 行级语句图
# =========================================================

def build_line_graph_from_merged_pdg(
    merged_raw: nx.DiGraph,
    stmt_infos: List[StmtInfo],
    span_by_line: Dict[int, List[StmtInfo]],
    func_start_line: int,
    func_end_line: int,
) -> nx.DiGraph:
    """
    将原始 PDG 投影为“行级语句图”：
    - 节点必须是完整语句
    - 节点带 start_line / end_line
    - 过滤掉 METHOD / PARAM / IDENTIFIER 等非语句节点
    """
    line_graph = nx.DiGraph()

    # 让每一条 raw node 映射到“最具体”的语句节点
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

        # 将 raw 节点信息挂到语句节点上
        stmt.raw_node_ids.append(str(nid))
        stmt.raw_kinds.append(kind)
        raw_label = str(data.get("raw_label") or data.get("code") or "")
        if raw_label:
            stmt.raw_labels.append(raw_label)

    # 添加语句节点
    for stmt in stmt_infos:
        line_graph.add_node(
            stmt.node_id,
            line_no=stmt.start_line,
            end_line=stmt.end_line,
            kind=stmt.kind,
            depth=stmt.depth,
            text=stmt.text,
            raw_node_ids=sorted(set(stmt.raw_node_ids)),
            raw_kinds=sorted(set(k for k in stmt.raw_kinds if k)),
            raw_labels=sorted(set(k for k in stmt.raw_labels if k)),
            covered_by_pdg=(len(stmt.raw_node_ids) > 0),
            span_len=stmt.span_len,
        )

    # PDG 边投影到语句节点
    for src, dst, data in merged_raw.edges(data=True):
        sdata = merged_raw.nodes[src]
        ddata = merged_raw.nodes[dst]

        sline = sdata.get("line_no")
        dline = ddata.get("line_no")

        if not isinstance(sline, int) or not isinstance(dline, int):
            continue
        if not (func_start_line <= sline <= func_end_line):
            continue
        if not (func_start_line <= dline <= func_end_line):
            continue

        sstmt = find_best_stmt_for_line(sline, span_by_line)
        dstmt = find_best_stmt_for_line(dline, span_by_line)
        if sstmt is None or dstmt is None:
            continue
        if sstmt.node_id == dstmt.node_id:
            continue

        label = str(data.get("label") or "")
        if line_graph.has_edge(sstmt.node_id, dstmt.node_id):
            existing = line_graph.edges[sstmt.node_id, dstmt.node_id]
            labels = set(existing.get("labels", []))
            if label:
                labels.add(label)
            existing["labels"] = sorted(labels)
            etypes = set(existing.get("edge_types", []))
            etypes.add("pdg")
            existing["edge_types"] = sorted(etypes)
        else:
            line_graph.add_edge(
                sstmt.node_id,
                dstmt.node_id,
                labels=[label] if label else [],
                edge_types=["pdg"],
            )

    # 顺序边：按语句出现顺序连接
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
# 8. Python 语义块划分
# =========================================================

def collect_span_stmt_infos(stmt_infos: List[StmtInfo], start_line: int, end_line: int) -> List[StmtInfo]:
    return sorted(
        [
            x for x in stmt_infos
            if x.start_line >= start_line and x.end_line <= end_line
        ],
        key=lambda x: (x.start_line, x.end_line, x.kind),
    )


def is_top_level_stmt(stmt: ast.AST) -> bool:
    return isinstance(stmt, ast.stmt) and not is_docstring_expr(stmt)


def region_kind_for_stmt(stmt: ast.AST) -> str:
    if is_compound_stmt_node(stmt):
        return "compound"
    if is_terminal_stmt_node(stmt):
        return "terminal"
    return "simple"


def build_top_level_regions(
    func_node: ast.AST,
    stmt_infos: List[StmtInfo],
    stmt_by_ast_id: Dict[int, StmtInfo],
) -> List[Region]:
    """
    用 Python AST 的函数体顶层语句作为骨架：
    - 连续简单语句合并为一个 simple region
    - 复合语句（if/for/while/try/with/match）整体作为 compound region
    - return / raise / break / continue 通常单独作为 terminal region
    """
    top_body = [stmt for stmt in getattr(func_node, "body", []) if is_top_level_stmt(stmt)]
    regions: List[Region] = []

    i = 0
    while i < len(top_body):
        stmt = top_body[i]
        info = stmt_by_ast_id.get(id(stmt))
        if info is None:
            i += 1
            continue

        kind = region_kind_for_stmt(stmt)

        if kind == "compound":
            region_stmt_infos = collect_span_stmt_infos(stmt_infos, info.start_line, info.end_line)
            regions.append(Region(region_kind="compound", ast_kind=info.kind, stmt_infos=region_stmt_infos))
            i += 1
            continue

        if kind == "terminal":
            regions.append(Region(region_kind="terminal", ast_kind=info.kind, stmt_infos=[info]))
            i += 1
            continue

        # simple：合并连续简单语句
        simple_infos = [info]
        j = i + 1
        while j < len(top_body):
            nxt = top_body[j]
            nxt_info = stmt_by_ast_id.get(id(nxt))
            if nxt_info is None:
                j += 1
                continue

            nxt_kind = region_kind_for_stmt(nxt)
            if nxt_kind in {"compound", "terminal"}:
                break

            simple_infos.append(nxt_info)
            j += 1

        regions.append(Region(region_kind="simple", ast_kind="simple", stmt_infos=simple_infos))
        i = j

    # 按起始行排序
    regions.sort(key=lambda r: (r.start_line, r.end_line))
    return regions


def split_simple_region_by_connectivity(line_graph: nx.DiGraph, region: Region) -> List[List[str]]:
    """
    simple region 如果在行级图上断开，则按连通分量拆分。
    compound region 不拆分，避免把 if/elif/else 或 try/except 结构拆散。
    """
    nodes = region.node_ids
    if len(nodes) <= 1:
        return [nodes]

    sub = line_graph.subgraph(nodes).to_undirected()
    comps = list(nx.connected_components(sub))
    if len(comps) <= 1:
        return [sorted(nodes, key=lambda nid: (line_graph.nodes[nid]["line_no"], line_graph.nodes[nid]["end_line"]))]

    ordered_blocks = []
    for comp in sorted(comps, key=lambda c: min(line_graph.nodes[nid]["line_no"] for nid in c)):
        ordered_blocks.append(
            sorted(
                list(comp),
                key=lambda nid: (line_graph.nodes[nid]["line_no"], line_graph.nodes[nid]["end_line"]),
            )
        )
    return ordered_blocks


def split_into_semantic_blocks(
    line_graph: nx.DiGraph,
    stmt_infos: List[StmtInfo],
    stmt_by_ast_id: Dict[int, StmtInfo],
    func_node: ast.AST,
) -> List[List[str]]:
    """
    语义块划分策略（Python 函数版）：
    1) 由 AST 的顶层语句决定块骨架；
    2) simple block = 连续简单语句；
    3) compound block = 一个完整复合语句（if/for/while/try/with/match）；
    4) graph 只作为投影和 connectivity 校验，不强行打碎复合结构。
    """
    regions = build_top_level_regions(func_node, stmt_infos, stmt_by_ast_id)

    blocks: List[List[str]] = []

    for region in regions:
        if region.region_kind == "simple":
            parts = split_simple_region_by_connectivity(line_graph, region)
            blocks.extend(parts)
        else:
            # compound / terminal：保持整体结构
            blocks.append(
                sorted(
                    region.node_ids,
                    key=lambda nid: (line_graph.nodes[nid]["line_no"], line_graph.nodes[nid]["end_line"]),
                )
            )

    # 清理空块、去重、排序
    cleaned: List[List[str]] = []
    seen: Set[str] = set()

    for block in blocks:
        if not block:
            continue
        block = [nid for nid in block if nid in line_graph]
        if not block:
            continue

        # 去重但保序
        uniq = []
        for nid in block:
            if nid not in seen:
                uniq.append(nid)
                seen.add(nid)

        if uniq:
            cleaned.append(uniq)

    cleaned.sort(key=lambda blk: min(line_graph.nodes[nid]["line_no"] for nid in blk))
    return cleaned


# =========================================================
# 9. 输出
# =========================================================

def node_record(line_graph: nx.DiGraph, nid: str) -> Dict:
    data = line_graph.nodes[nid]
    return {
        "node_id": nid,
        "line_no": data.get("line_no"),
        "end_line": data.get("end_line"),
        "kind": data.get("kind", ""),
        "depth": data.get("depth", None),
        "text": data.get("text", ""),
        "covered_by_pdg": data.get("covered_by_pdg", False),
        "raw_node_ids": data.get("raw_node_ids", []),
        "raw_kinds": data.get("raw_kinds", []),
        "raw_labels": data.get("raw_labels", []),
    }


def block_to_records(line_graph: nx.DiGraph, block: List[str]) -> List[Dict]:
    return [node_record(line_graph, nid) for nid in block]


def print_blocks(line_graph: nx.DiGraph, blocks: List[List[str]]) -> None:
    print("\n" + "=" * 100)
    print("语义块划分结果（Python 函数 / 行级语句节点）")
    print("=" * 100)

    for i, block in enumerate(blocks, 1):
        print(f"\n📦 Block {i}")
        print("-" * 100)
        for r in block_to_records(line_graph, block):
            flag = "PDG" if r["covered_by_pdg"] else "SRC"
            print(f"[{r['line_no']}-{r['end_line']}] ({r['kind']} | {flag}) {r['text']}")


def save_results(
    out_dir: Path,
    source_text: str,
    source_lines: List[str],
    func_node: ast.AST,
    raw_graph: nx.DiGraph,
    line_graph: nx.DiGraph,
    blocks: List[List[str]],
    dot_files_used: List[str],
) -> None:
    ensure_dir(out_dir)

    func_start_line, func_end_line = function_span(func_node)

    json_data = {
        "function_name": getattr(func_node, "name", FUNCTION_NAME),
        "function_start_line": func_start_line,
        "function_end_line": func_end_line,
        "dot_files_used": dot_files_used,
        "raw_graph": {
            "node_count": raw_graph.number_of_nodes(),
            "edge_count": raw_graph.number_of_edges(),
        },
        "line_graph": {
            "node_count": line_graph.number_of_nodes(),
            "edge_count": line_graph.number_of_edges(),
            "nodes": [
                {"id": nid, **data}
                for nid, data in line_graph.nodes(data=True)
            ],
            "edges": [
                {"src": src, "dst": dst, **data}
                for src, dst, data in line_graph.edges(data=True)
            ],
        },
        "blocks": [
            {
                "block_id": idx + 1,
                "nodes": block_to_records(line_graph, block),
            }
            for idx, block in enumerate(blocks)
        ],
        "source_lines": [
            {"line_no": i + 1, "text": line}
            for i, line in enumerate(source_lines)
        ],
        "source_text": source_text,
    }

    (out_dir / "semantic_blocks.json").write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    txt_lines: List[str] = []
    txt_lines.append("=" * 100)
    txt_lines.append("语义块划分结果（Python 函数 / 行级语句节点）")
    txt_lines.append("=" * 100)
    txt_lines.append(f"函数：{getattr(func_node, 'name', FUNCTION_NAME)}")
    txt_lines.append(f"函数范围：{func_start_line}-{func_end_line}")
    txt_lines.append(f"使用的 dot 文件数：{len(dot_files_used)}")
    txt_lines.append(f"原始图：节点 {raw_graph.number_of_nodes()}，边 {raw_graph.number_of_edges()}")
    txt_lines.append(f"行级图：节点 {line_graph.number_of_nodes()}，边 {line_graph.number_of_edges()}")

    for i, block in enumerate(blocks, 1):
        txt_lines.append("")
        txt_lines.append(f"📦 Block {i}")
        txt_lines.append("-" * 100)
        for r in block_to_records(line_graph, block):
            flag = "PDG" if r["covered_by_pdg"] else "SRC"
            txt_lines.append(f"[{r['line_no']}-{r['end_line']}] ({r['kind']} | {flag}) {r['text']}")

    (out_dir / "semantic_blocks.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"\n[OK] 结果已保存到：{out_dir}")
    print("     - semantic_blocks.json")
    print("     - semantic_blocks.txt")


# =========================================================
# 10. 主流程
# =========================================================

def main() -> None:
    joern_home = Path(JOERN_HOME).expanduser().resolve()
    if not joern_home.exists():
        raise FileNotFoundError(f"JOERN_HOME 不存在：{joern_home}")

    source_text = normalize_source(SOURCE_CODE)
    source_lines = source_text.splitlines()

    tree = parse_python_ast(source_text)
    func_node = find_target_function(tree, FUNCTION_NAME)
    func_start_line, func_end_line = function_span(func_node)

    work_dir = Path.cwd() / f"pdg_work_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(work_dir)

    source_file = work_dir / f"test_func{SOURCE_SUFFIX}"
    source_file.write_text(source_text, encoding="utf-8")

    print(f"[INFO] Joern 目录：{joern_home}")
    print(f"[INFO] 工作目录：{work_dir}")
    print(f"[INFO] 源码文件：{source_file}")
    print(f"[INFO] 函数名：{FUNCTION_NAME}")
    print(f"[INFO] 函数起止行：{func_start_line}-{func_end_line}")

    pdg_dir = generate_pdg_with_joern(source_file, work_dir, joern_home)
    print(f"[INFO] PDG 导出目录：{pdg_dir}")

    candidate_graphs = choose_candidate_graphs(
        pdg_dir,
        FUNCTION_NAME,
        func_start_line,
        func_end_line,
    )
    dot_files_used = sorted({g.graph.get("name", "") for g in candidate_graphs if isinstance(g, nx.DiGraph)})
    print(f"[INFO] 命中的 dot 数量：{len(candidate_graphs)}")

    merged_raw = merge_graphs(candidate_graphs)
    print(f"[INFO] 合并后原始图：节点数 {merged_raw.number_of_nodes()}，边数 {merged_raw.number_of_edges()}")

    stmt_infos, stmt_by_ast_id, span_by_line = build_stmt_infos_for_function(
        source_text=source_text,
        source_lines=source_lines,
        func_node=func_node,
    )

    line_graph = build_line_graph_from_merged_pdg(
        merged_raw=merged_raw,
        stmt_infos=stmt_infos,
        span_by_line=span_by_line,
        func_start_line=func_start_line,
        func_end_line=func_end_line,
    )
    print(f"[INFO] 行级图：节点数 {line_graph.number_of_nodes()}，边数 {line_graph.number_of_edges()}")

    blocks = split_into_semantic_blocks(
        line_graph=line_graph,
        stmt_infos=stmt_infos,
        stmt_by_ast_id=stmt_by_ast_id,
        func_node=func_node,
    )
    print_blocks(line_graph, blocks)

    out_dir = work_dir / "results"
    save_results(
        out_dir=out_dir,
        source_text=source_text,
        source_lines=source_lines,
        func_node=func_node,
        raw_graph=merged_raw,
        line_graph=line_graph,
        blocks=blocks,
        dot_files_used=sorted({g.graph.get("dot_file", "") for g in candidate_graphs}),
    )


if __name__ == "__main__":
    main()
