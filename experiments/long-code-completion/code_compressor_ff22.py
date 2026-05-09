# ff22: split_code1 fragmented semantic blocks + high-variance entropy fallback.
# 从 top10 和 data.log 看，split_code2 的合并/注释附着会吞掉预算；本版用更碎的 split_code1，
# 并把 test/doc/control_loop/class/schema/decorator 续写回退到 entropy，保留 literal/assign_call 的语义块优势。
import torch
import numpy as np
from typing import Any, List, Union, Tuple, Dict, Optional
import re
import math
import zlib
import hashlib
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm
import copy
import bisect
import json
import ast
import textwrap
import logging
import random
from collections import defaultdict
import networkx as nx
from functools import lru_cache
from loguru import logger

# 新增导入
import sys
import tempfile
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import split_code1 as semantic_splitter
except Exception as _split_code_import_error:
    semantic_splitter = None
    logger.warning(f"Failed to import split_code1.py: {_split_code_import_error}")

try:
    import graph as semantic_graph_viz
except Exception as _graph_import_error:
    semantic_graph_viz = None
    logger.warning(f"Failed to import graph.py: {_graph_import_error}")

class EntropyChunking:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Entropy-based text chunking implementation"""
        logger.debug(f"Loading Entropy chunking model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.debug(f"Entropy chunking model loaded on device: {self.device}")

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, inserting empty lines for double newlines"""
        # First replace double newlines with a special marker
        text_with_markers = text.replace('\n\n', '\n__EMPTY_LINE__\n')
        
        # Split by single newlines
        lines = text_with_markers.split('\n')
        
        # Process lines: replace markers with empty strings, keep original lines
        sentences = []
        for line in lines:
            if line == '__EMPTY_LINE__':
                sentences.append(' ')  # Empty line for double newline breaks
            else:
                sentences.append(line)  # Keep original line with indentation
        
        return sentences

    def calculate_sentence_ppl(self, sentences: List[str]) -> List[float]:
        """Calculate perplexity for each sentence based on preceding context"""
        ppls = []
        
        for i, sentence in enumerate(sentences):
            if i == 0:
                context = ""
                target = sentence
            else:
                context = "\n".join(sentences[:i])
                target = sentence
            
            ppl = self._compute_ppl(context, target)
            ppls.append(ppl)
        
        return ppls

    def _compute_ppl(self, context: str, target: str) -> float:
        """Compute perplexity of target text given context"""
        # Handle empty target lines
        if not target:
            return 0.0  # Assign zero perplexity to empty lines
            
        if context:
            full_text = context + "\n" + target
            context_tokens = self.tokenizer(context + "\n", return_tensors="pt", add_special_tokens=True)
            context_length = context_tokens.input_ids.shape[1]
        else:
            full_text = target
            context_length = 0
        
        inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        if context_length > 0:
            target_logits = logits[0, context_length-1:-1]
            target_labels = inputs.input_ids[0, context_length:]
        else:
            target_logits = logits[0, :-1]
            target_labels = inputs.input_ids[0, 1:]
        
        if len(target_labels) > 0:
            log_probs = torch.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs[torch.arange(len(target_labels)), target_labels]
            avg_log_prob = token_log_probs.mean().item()
            ppl = math.exp(-avg_log_prob)
        else:
            ppl = float('inf')

        # take log2 of ppl
        ppl = math.log2(ppl)
        
        return ppl

    def calculate_adaptive_thresholds(self, ppls: List[float], k: float = 0.2) -> dict:
        """Calculate adaptive thresholds using different statistical methods"""
        # Filter out infinite and NaN values
        valid_ppls = [p for p in ppls if not math.isinf(p) and not math.isnan(p) and p > 0]
        
        if len(valid_ppls) < 3:
            # Fallback to fixed threshold if not enough valid data
            return {
                'std': 0.5,
                'robust_std': 0.5,
                'iqr': 0.5,
                'mad': 0.5
            }
        
        valid_ppls = np.array(valid_ppls)
        
        # Method 1: Standard deviation based
        mean_ppl = np.mean(valid_ppls)
        std_ppl = np.std(valid_ppls)
        threshold_std = mean_ppl + k * std_ppl
        
        # Method 2: Robust standard deviation (using median and MAD)
        median_ppl = np.median(valid_ppls)
        mad = np.median(np.abs(valid_ppls - median_ppl))
        robust_std = mad * 1.4826  # Convert MAD to robust std estimate
        threshold_robust_std = median_ppl + k * robust_std
        
        # Method 3: IQR based (Interquartile Range)
        q25 = np.percentile(valid_ppls, 25)
        q75 = np.percentile(valid_ppls, 75)
        iqr = q75 - q25
        threshold_iqr = q75 + k * iqr
        
        # Method 4: MAD based (Median Absolute Deviation)
        threshold_mad = median_ppl + k * mad
        
        return {
            'std': threshold_std,
            'robust_std': threshold_robust_std,
            'iqr': threshold_iqr,
            'mad': threshold_mad
        }

    def find_ppl_spikes_adaptive(self, values: List[float], method: str = 'std', k: float = 0.2) -> tuple:
        """Find PPL spikes using adaptive threshold based on statistical method"""
        thresholds = self.calculate_adaptive_thresholds(values, k)
        threshold = thresholds[method]
        
        spike_indices = []
        
        for i in range(1, len(values) - 1):
            current = values[i]
            left = values[i - 1]
            right = values[i + 1]
            
            # Skip infinite or NaN values
            if math.isinf(current) or math.isnan(current):
                continue
            if math.isinf(left) or math.isnan(left):
                left = current
            if math.isinf(right) or math.isnan(right):
                right = current
            
            # Check if current PPL is significantly higher than both neighbors
            left_diff = current - left
            right_diff = current - right
            
            # Condition: Current PPL is higher than both neighbors with adaptive threshold
            if (left_diff >= threshold or right_diff >= threshold) and (left_diff >= 0 and right_diff >= 0):
                spike_indices.append(i)
        
        return spike_indices, threshold

    def chunk_text_adaptive(self, text: str, method: str = 'std', k: float = 0.2) -> tuple:
        """Perform PPL-based text chunking using adaptive spike detection"""
        sentences = self.split_into_sentences(text)
        ppls = self.calculate_sentence_ppl(sentences)
        spike_indices, threshold = self.find_ppl_spikes_adaptive(ppls, method, k)
        
        chunks = []
        # Split at spike points (after the spike line)
        split_points = [0] + [idx + 1 for idx in spike_indices] + [len(sentences)]
        
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = "\n".join(chunk_sentences)
            chunks.append(chunk_text)
        
        return chunks, sentences, ppls, spike_indices




class ProgramAnalysisSemanticChunking:
    """
    Module-aware adapter over split_code.py.

    Behavior:
    1) Walks the whole module instead of only the first top-level function/class.
    2) Keeps the existing function-level PDG semantic splitter from split_code.py.
    3) Does NOT merge top-level miscellaneous statements into one block.
    4) Does NOT merge class-level miscellaneous statements into one block.
    5) Preserves decorators for functions/classes/statements where possible.
    """

    def __init__(
        self,
        joern_home: Optional[str] = None,
        work_root: Optional[Union[str, Path]] = None,
        fallback_entropy: Optional[object] = None,
    ):
        self.fallback_entropy = fallback_entropy
        self.work_root = Path(work_root or ".semantic_chunk_cache").expanduser().resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, tuple] = {}
        self._pdg_cache: Dict[str, Path] = {}
        self._graph_cache: Dict[str, List[int]] = {}

        self.joern_home = Path(joern_home).expanduser().resolve() if joern_home else None
        self.module = semantic_splitter
        self.impl = None

        self._required_helpers = [
            "parse_python_ast",
            "function_span",
            "build_stmt_infos_for_function",
            "generate_pdg_with_joern",
            "choose_candidate_graphs",
            "merge_graphs",
            "build_line_graph_from_merged_pdg",
            "build_semantic_blocks",
            "build_suite_semantic_blocks",
            "build_module_semantic_blocks",
        ]

        if self.module is None:
            logger.warning("Failed to import split_code.py; semantic chunking will use fallback mode.")
            return

        missing = [name for name in self._required_helpers if not hasattr(self.module, name)]
        if missing:
            logger.warning(
                "split_code.py does not expose the expected helper functions "
                f"{missing}; semantic chunking will use fallback mode."
            )
        else:
            logger.debug("ProgramAnalysisSemanticChunking initialized from split_code.py helper functions.")

    @staticmethod
    def _is_docstring_expr(stmt: ast.AST) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    @staticmethod
    def _leading_indent(text: str) -> str:
        for line in text.splitlines():
            if line.strip():
                m = re.match(r"^\s*", line)
                return m.group(0) if m else ""
        return ""

    @staticmethod
    def _reindent_block(text: str, indent: str) -> str:
        if not text:
            return text
        out = []
        for line in text.splitlines():
            if line.strip():
                out.append(f"{indent}{line}")
            else:
                out.append(line)
        return "\n".join(out).rstrip("\n")

    @staticmethod
    def _node_text(source_text: str, source_lines: List[str], node: ast.AST) -> str:
        try:
            seg = ast.get_source_segment(source_text, node)
            if seg is not None and seg.strip():
                return seg.rstrip("\n")
        except Exception:
            pass

        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None:
            return ""
        if end is None:
            end = start
        start = max(1, int(start))
        end = min(max(start, int(end)), len(source_lines))
        return "\n".join(source_lines[start - 1:end]).rstrip("\n")

    @staticmethod
    def _node_source_with_decorators(source_lines: List[str], node: ast.AST) -> str:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)

        decorator_list = getattr(node, "decorator_list", None)
        if decorator_list:
            dec_lines = [
                getattr(d, "lineno", None)
                for d in decorator_list
                if getattr(d, "lineno", None) is not None
            ]
            if dec_lines:
                start = min(dec_lines)

        if start is None:
            return ""
        if end is None:
            end = start
        start = max(1, int(start))
        end = min(len(source_lines), int(end))
        return "\n".join(source_lines[start - 1:end]).rstrip("\n")

    @staticmethod
    def _fallback_line_chunks(text: str) -> List[str]:
        lines = text.splitlines()
        if not lines:
            return [text] if text else []

        chunks: List[str] = []
        cur: List[str] = []
        for line in lines:
            if line.strip():
                cur.append(line)
            else:
                if cur:
                    chunks.append("\n".join(cur).rstrip())
                    cur = []
        if cur:
            chunks.append("\n".join(cur).rstrip())

        return chunks if chunks else ([text] if text.strip() else [])

    def _fallback_result(self, text: str) -> tuple:
        chunks = self._fallback_line_chunks(text)
        sentences = text.splitlines()
        ppls = [float(len(c.splitlines())) for c in chunks]
        spike_indices = [i for i in range(max(0, len(chunks) - 1))]
        return chunks, sentences, ppls, spike_indices

    def _source_key(self, source_text: str) -> str:
        return hashlib.md5(source_text.encode("utf-8")).hexdigest()[:16]

    def _ensure_pdg_dir(self, source_text: str, source_file_name: str = "snippet.py") -> Optional[Path]:
        if self.module is None:
            return None

        key = self._source_key(source_text)
        if key in self._pdg_cache:
            return self._pdg_cache[key]

        missing = [name for name in self._required_helpers if not hasattr(self.module, name)]
        if missing:
            return None

        work_dir = self.work_root / f"semantic_{key}"
        work_dir.mkdir(parents=True, exist_ok=True)

        source_file = work_dir / source_file_name
        source_file.write_text(source_text, encoding="utf-8")

        generate_pdg_with_joern = getattr(self.module, "generate_pdg_with_joern")
        pdg_dir = generate_pdg_with_joern(source_file, work_dir, self.joern_home)
        pdg_dir = Path(pdg_dir)

        self._pdg_cache[key] = pdg_dir
        return pdg_dir

    def _analyze_single_function(
        self,
        func_node: ast.AST,
        source_text: str,
        source_lines: List[str],
    ) -> List[str]:
        if self.module is None:
            return [self._node_source_with_decorators(source_lines, func_node)]

        missing = [name for name in self._required_helpers if not hasattr(self.module, name)]
        if missing:
            return [self._node_source_with_decorators(source_lines, func_node)]

        try:
            parse_python_ast = getattr(self.module, "parse_python_ast")
            function_span = getattr(self.module, "function_span")
            build_stmt_infos_for_function = getattr(self.module, "build_stmt_infos_for_function")
            choose_candidate_graphs = getattr(self.module, "choose_candidate_graphs")
            merge_graphs = getattr(self.module, "merge_graphs")
            build_line_graph_from_merged_pdg = getattr(self.module, "build_line_graph_from_merged_pdg")
            build_semantic_blocks = getattr(self.module, "build_semantic_blocks")
            build_suite_semantic_blocks = getattr(self.module, "build_suite_semantic_blocks", None)

            func_source = self._node_source_with_decorators(source_lines, func_node)
            if not func_source.strip():
                return []

            analysis_source = textwrap.dedent(func_source)
            if not analysis_source.endswith("\n"):
                analysis_source += "\n"
            tree = parse_python_ast(analysis_source)

            target = None
            for node in getattr(tree, "body", []):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    target = node
                    break
            if target is None:
                return [func_source.rstrip()]

            pdg_dir = self._ensure_pdg_dir(analysis_source, "snippet.py")
            if pdg_dir is None:
                return [func_source.rstrip()]

            func_start_line, func_end_line = function_span(target)
            candidate_graphs = choose_candidate_graphs(pdg_dir, target.name, func_start_line, func_end_line)
            if not candidate_graphs:
                return [func_source.rstrip()]

            merged_raw = merge_graphs(candidate_graphs)
            raw_lines = func_source.splitlines()

            stmt_infos, stmt_by_ast_id, span_by_line = build_stmt_infos_for_function(
                source_text=analysis_source,
                source_lines=raw_lines,
                func_node=target,
            )

            line_graph = build_line_graph_from_merged_pdg(
                merged_raw=merged_raw,
                stmt_infos=stmt_infos,
                span_by_line=span_by_line,
                func_start_line=func_start_line,
                func_end_line=func_end_line,
            )

            blocks = build_semantic_blocks(
                func_node=target,
                source_text=analysis_source,
                source_lines=raw_lines,
                line_graph=line_graph,
                stmt_by_ast_id=stmt_by_ast_id,
            )

            chunks = [blk.code.rstrip() for blk in blocks if blk.code and blk.code.strip()]

            if len(chunks) <= 1 and build_suite_semantic_blocks is not None:
                fallback_blocks = build_suite_semantic_blocks(
                    [n for n in getattr(target, "body", []) if not self._is_docstring_expr(n)],
                    analysis_source,
                    raw_lines,
                    depth=1,
                )
                fallback_chunks = [b.code.rstrip() for b in fallback_blocks if getattr(b, "code", "").strip()]
                if fallback_chunks:
                    return fallback_chunks

            if len(chunks) <= 1:
                return self._fallback_line_chunks(func_source)

            return chunks

        except Exception as e:
            logger.warning(f"Semantic split failed for {getattr(func_node, 'name', '<anonymous>')}, fallback to whole node: {e}")
            return [self._node_source_with_decorators(source_lines, func_node)]

    def _split_class_node(self, class_node: ast.ClassDef, source_text: str, source_lines: List[str]) -> List[str]:
        chunks: List[str] = []
        class_start = getattr(class_node, "lineno", None)
        if class_start is None:
            return []

        decorator_list = getattr(class_node, "decorator_list", None)
        if decorator_list:
            dec_lines = [
                getattr(d, "lineno", None)
                for d in decorator_list
                if getattr(d, "lineno", None) is not None
            ]
            if dec_lines:
                class_start = min(dec_lines)

        body_first_line = None
        for stmt in getattr(class_node, "body", []):
            ln = getattr(stmt, "lineno", None)
            if ln is not None:
                body_first_line = ln
                break

        header_end = getattr(class_node, "lineno", class_start)
        if body_first_line is not None and body_first_line > class_start:
            header_end = body_first_line - 1

        header_text = "\n".join(source_lines[class_start - 1:header_end]).rstrip("\n")
        if header_text.strip():
            chunks.append(header_text)

        build_suite_semantic_blocks = getattr(self.module, "build_suite_semantic_blocks", None) if self.module else None

        for stmt in getattr(class_node, "body", []):
            if self._is_docstring_expr(stmt):
                continue

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(self._analyze_single_function(stmt, source_text, source_lines))
                continue

            if isinstance(stmt, ast.ClassDef):
                chunks.extend(self._split_class_node(stmt, source_text, source_lines))
                continue

            text = self._node_source_with_decorators(source_lines, stmt)
            if text.strip():
                chunks.append(text.rstrip())

        if len(chunks) <= 1 and build_suite_semantic_blocks is not None:
            try:
                body = [n for n in getattr(class_node, "body", []) if not self._is_docstring_expr(n)]
                blocks = build_suite_semantic_blocks(body, source_text, source_lines, depth=1)
                alt = [b.code.rstrip() for b in blocks if getattr(b, "code", "").strip()]
                if alt:
                    chunks = [header_text] + alt if header_text.strip() else alt
            except Exception:
                pass

        return [c for c in chunks if c.strip()]

    def _split_module(self, tree: ast.AST, source_text: str, source_lines: List[str]) -> List[str]:
        chunks: List[str] = []

        for node in getattr(tree, "body", []):
            if self._is_docstring_expr(node):
                continue

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(self._analyze_single_function(node, source_text, source_lines))
                continue

            if isinstance(node, ast.ClassDef):
                chunks.extend(self._split_class_node(node, source_text, source_lines))
                continue

            text = self._node_source_with_decorators(source_lines, node)
            if text.strip():
                chunks.append(text.rstrip())

        return [c for c in chunks if c.strip()]

    def chunk_text_adaptive(
        self,
        text: str,
        method: str = "std",
        k: float = 0.2,
        language: str = "python",
    ) -> tuple:
        if language.lower() != "python":
            return self._fallback_result(text)

        if self.module is None:
            return self._fallback_result(text)

        missing = [name for name in self._required_helpers if not hasattr(self.module, name)]
        if missing:
            return self._fallback_result(text)

        try:
            raw_text = text
            analysis_source = textwrap.dedent(raw_text)
            if not analysis_source.endswith("\n"):
                analysis_source += "\n"
            if not analysis_source.strip():
                return [], [], [], []

            parse_python_ast = getattr(self.module, "parse_python_ast")
            tree = parse_python_ast(analysis_source)
            raw_source_lines = raw_text.splitlines()

            chunks = self._split_module(tree, analysis_source, raw_source_lines)

            if not chunks:
                return self._fallback_result(text)

            chunks = [chunk.rstrip("\n") for chunk in chunks if chunk.strip()]

            sentences = raw_text.splitlines()
            ppls = [float(len(c.splitlines())) for c in chunks]
            spike_indices = [i for i in range(max(0, len(chunks) - 1))]
            return chunks, sentences, ppls, spike_indices

        except Exception as e:
            logger.warning(f"ProgramAnalysisSemanticChunking failed, falling back to line chunks: {e}")
            return self._fallback_result(text)


class CodeCompressor:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4",
        device_map: str = "cuda",
        model_config: dict = {},
    ):
        """
        Initialize the CodeCompressor with a language model for compression.
        
        Args:
            model_name: The name of the model to load from HuggingFace
            device_map: Device to load the model on
            model_config: Additional configuration for the model
        """
        self.model_name = model_name
        self.device = device_map
        self.model_config = model_config
        self.load_model(model_name, device_map, model_config)
        
        # Initialize Entropy chunking with smaller model
        logger.debug("Initializing Entropy chunking...")
        self.entropy_chunking = EntropyChunking()
        self.semantic_chunker = ProgramAnalysisSemanticChunking(
            joern_home=getattr(semantic_splitter, "JOERN_HOME", None) if semantic_splitter is not None else None,
            fallback_entropy=self.entropy_chunking,
        )

        # Graph cache for raw-dot based dependency statistics
        self._graph_cache: Dict[str, Any] = {}
        self._graph_cache_meta: Dict[str, Any] = {}

        # Add caching system for model outputs and token information
        self.cache = {
            "token_length": {},      # Cache for token length by text
            "encodings": {},         # Cache for tokenizer encodings
            "perplexity": {},        # Cache for perplexity calculations
            "conditional_ppl": {},   # Cache for conditional perplexity
            "context_rankings": {},  # Cache for context rankings
            "chunk_ppl": {},         # Cache for coarse chunk PPL scores
        }
        self.max_cache_size = 1000   # Limit cache size to prevent memory issues
        
        # Joern / semantic block analysis helpers
        self.joern_home = Path(getattr(semantic_splitter, "JOERN_HOME", "/home/zhangmanqing/wyh/joern-cli")).expanduser().resolve() if semantic_splitter is not None else Path("/home/zhangmanqing/wyh/joern-cli").expanduser().resolve()
        self.semantic_work_root = Path(".semantic_chunk_cache").expanduser().resolve()
        self.semantic_work_root.mkdir(parents=True, exist_ok=True)

        # set up the max position embeddings and cache bos num
        self.max_position_embeddings = getattr(self.model.config, "max_position_embeddings", 4096)
        self.cache_bos_num = 10
        self.prefix_bos_num = 100
        self.context_idxs = []
    
    def load_model(
        self, model_name: str, device_map: str = "cuda", model_config: dict = {}
    ):
        """
        Load the language model and tokenizer.
        
        Args:
            model_name: The name of the model to load
            device_map: Device to load the model on
            model_config: Additional configuration for the model
        """
        logger.debug(f"Loading model {model_name} on {device_map}")
        torch_dtype = torch.bfloat16 if "torch_dtype" not in model_config else model_config["torch_dtype"]
        # model_kwargs = {"device_map": device_map, "torch_dtype": torch_dtype, "trust_remote_code": True}
        model_kwargs = {"device_map": device_map, "torch_dtype": torch_dtype, "trust_remote_code": True}
        
        for k, v in model_config.items():
            model_kwargs[k] = v
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        logger.debug("Model and tokenizer loaded successfully")
        
    def _manage_cache_size(self, cache_type):
        """
        Manage cache size by removing oldest entries when cache exceeds max size.
        
        Args:
            cache_type: The type of cache to manage
        """
        if len(self.cache[cache_type]) > self.max_cache_size:
            # Remove 20% of the oldest entries
            remove_count = int(self.max_cache_size * 0.2)
            keys_to_remove = list(self.cache[cache_type].keys())[:remove_count]
            for key in keys_to_remove:
                del self.cache[cache_type][key]
        
    @staticmethod
    def _edge_text(attrs: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in ("label", "kind", "type", "edgeType", "relation", "rel", "name", "tag"):
            if key in attrs and attrs.get(key) not in (None, ""):
                parts.append(str(attrs.get(key)))
        if "attrs" in attrs and isinstance(attrs["attrs"], dict):
            inner = attrs["attrs"]
            for key in ("label", "kind", "type", "edgeType", "relation", "rel", "name", "tag"):
                if key in inner and inner.get(key) not in (None, ""):
                    parts.append(str(inner.get(key)))
        if not parts and attrs:
            parts.append(" ".join(f"{k}={v}" for k, v in attrs.items() if v not in (None, "")))
        return " | ".join(x for x in parts if x)

    @staticmethod
    def _is_data_dependency_edge(edata: Dict[str, Any]) -> bool:
        text = CodeCompressor._edge_text(edata).lower()
        control_hints = ("cdg", "control dependency", "control-dependency", "control dependence", "control-dependence", "cfg", "control flow", "branch")
        data_hints = ("ddg", "data dependency", "data-dependency", "data dependence", "data-dependence", "reaching_def", "reaching def", "def-use", "def use", "use-def", "use def", "dataflow")
        if any(tok in text for tok in control_hints):
            return False
        return any(tok in text for tok in data_hints) or "ddg" in text

    @staticmethod
    def _is_control_dependency_edge(edata: Dict[str, Any]) -> bool:
        text = CodeCompressor._edge_text(edata).lower()
        control_hints = ("cdg", "control dependency", "control-dependency", "control dependence", "control-dependence", "cfg", "control flow", "branch")
        data_hints = ("ddg", "data dependency", "data-dependency", "data dependence", "data-dependence", "reaching_def", "reaching def", "def-use", "def use", "use-def", "use def", "dataflow")
        if any(tok in text for tok in data_hints):
            return False
        return any(tok in text for tok in control_hints) or "cdg" in text

    @staticmethod
    def _infer_line_no_from_node_attrs(data: Dict[str, Any]) -> Optional[int]:
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
        m = re.search(r"(?<!\d)(\d{1,5})(?!\d)", raw)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        return None

    def _build_dependency_line_graph_from_raw_pdg(
        self,
        merged_raw: Any,
        source_lines: List[str],
        func_start_line: int,
        func_end_line: int,
    ) -> Tuple[nx.DiGraph, Dict[str, str], Dict[str, Any]]:
        line_graph = nx.DiGraph()

        code_line_nos = [
            ln for ln in range(func_start_line, func_end_line + 1)
            if 1 <= ln <= len(source_lines) and source_lines[ln - 1].strip() and not source_lines[ln - 1].lstrip().startswith("#")
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
            )

        raw_node_to_line: Dict[str, str] = {}
        skipped_nodes: List[Dict[str, Any]] = []
        for nid, data in merged_raw.nodes(data=True):
            line_no = self._infer_line_no_from_node_attrs(data)
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
            relation = "data" if self._is_data_dependency_edge(edata) else "control" if self._is_control_dependency_edge(edata) else "other"
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

            dependent_line = dst_line
            dependency_line = src_line
            if dependent_line == dependency_line:
                line_graph.nodes[dependency_line]["raw_self_edges"] += 1
                continue

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
                    "src": src,
                    "dst": dst,
                    "key": key,
                    "label": label,
                    "attrs": attrs,
                    "relation": relation,
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
                        "src": src,
                        "dst": dst,
                        "key": key,
                        "label": label,
                        "attrs": attrs,
                        "relation": relation,
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

    def get_token_length(
        self,
        text: str,
        add_special_tokens: bool = True,
    ):
        """
        Get the number of tokens in the given text.
        
        Args:
            text: The text to tokenize
            add_special_tokens: Whether to count special tokens
            
        Returns:
            The number of tokens
        """
        # Create a cache key based on text and parameters
        cache_key = f"{text}_{add_special_tokens}"
        
        # Check if result is in cache
        if cache_key in self.cache["token_length"]:
            return self.cache["token_length"][cache_key]
        
        # Calculate token length if not in cache
        token_length = len(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))
        
        # Store in cache
        self.cache["token_length"][cache_key] = token_length
        self._manage_cache_size("token_length")
        
        return token_length
    
    def get_ppl(
        self,
        text: str,
        granularity: str = "line",
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None,
        condition_mode: str = "none",
        condition_pos_id: int = 0,
    ):


        cache_key = f"{text}_{granularity}_{condition_mode}_{condition_pos_id}"
        if past_key_values is None and not return_kv and cache_key in self.cache["perplexity"]:
            print("[CACHE HIT]")
            return self.cache["perplexity"][cache_key]
        
        # ================= 原逻辑 =================
        if input_ids is None:
            encoding_key = text
            if encoding_key in self.cache["encodings"]:
                cached_encoding = self.cache["encodings"][encoding_key]
                input_ids = cached_encoding["input_ids"]
                attention_mask = cached_encoding["attention_mask"]
                print("[ENCODING CACHE HIT]")
            else:
                encoding = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True
                )
                input_ids = encoding["input_ids"].to(self.model.device)
                attention_mask = encoding["attention_mask"].to(self.model.device)

                
                self.cache["encodings"][encoding_key] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                self._manage_cache_size("encodings")
        
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
            
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values,
                return_dict=True,
                output_hidden_states=True,
                use_cache=True,
            )
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length+1:end].contiguous()

        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]


        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)

        if condition_mode == "prefix":
            loss = loss[condition_pos_id:]

        mean_loss = loss.mean() if len(loss) > 0 else torch.tensor(0.0)
        ppl = torch.exp(mean_loss).item() if mean_loss.item() != float('inf') else float('inf')


        result = {
            "loss": loss,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lines_info": [],
            "segments": [text] if text else [],
            "ppl": ppl,
        }
        
        if return_kv:
            result["past_key_values"] = outputs.past_key_values
        else:
            self.cache["perplexity"][cache_key] = result
            self._manage_cache_size("perplexity")
            
        return result
    
    def __get_lines_info(self, lines, input_ids, loss):
        """
        Get information about each line including start/end positions and importance.
        
        Args:
            lines: List of lines in the text
            input_ids: Token IDs for the entire text
            loss: Per-token loss values
            
        Returns:
            List of dictionaries with line information
        """
        line_info = []
        cumulative_tokens = 0
        
        input_ids_list = input_ids.cpu().tolist()
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Encode each line to find its token length
            line_tokens = self.tokenizer.encode(line, add_special_tokens=False)
            line_length = len(line_tokens)
            
            # Find position in the tokenized text
            start_pos = cumulative_tokens
            end_pos = start_pos + line_length
            
            # Calculate mean loss (importance) for this line
            # Loss might be shorter than the token IDs due to shifting
            if isinstance(loss, torch.Tensor) and start_pos < len(loss) and end_pos <= len(loss):
                line_loss = loss[start_pos:end_pos].mean().item()
            else:
                # Handle edge cases
                line_loss = float("inf")
            
            line_info.append({
                "line": line,
                "start": start_pos,
                "end": end_pos,
                "importance": line_loss,
                "tokens": line_length
            })
            
            cumulative_tokens += line_length
            
        return line_info
    
    def get_prefix_length(self, prefix: str, text: str):
        """
        Calculate the length of a prefix in tokens when concatenated with a text.
        
        Args:
            prefix: The prefix text
            text: The main text
            
        Returns:
            Length of the prefix in tokens
        """
        possible_prefix_token = max(self.get_token_length(prefix, False) - 3, 1)
        full_input_ids = self.tokenizer(prefix + text[:100], add_special_tokens=False).input_ids
        
        for i in range(possible_prefix_token, len(full_input_ids)):
            cur_prefix = self.tokenizer.decode(full_input_ids[:i])
            if cur_prefix == prefix:
                break
                
        return i
    
    def get_condition_ppl(
        self,
        text: str,
        question: str,
        condition_in_question: str = "none",
        granularity: str = "line",
    ):
        cache_key = f"{text}_{question}_{condition_in_question}_{granularity}"
        
        if cache_key in self.cache["conditional_ppl"]:
            print("[CACHE HIT]")
            return self.cache["conditional_ppl"][cache_key]
        
        if condition_in_question == "none":
            result = self.get_ppl(
                text=text, granularity=granularity, condition_mode="none"
            )
            ppl_value = result["ppl"]

        else:

            question_ppl_without_context = self.get_ppl(
                text=question, 
                granularity=granularity
            )["ppl"]

            full_input = text + "\n\n" + question

            prefix = text + "\n\n"
            condition_pos = self.get_token_length(prefix, add_special_tokens=True)

            question_ppl_with_context = self.get_ppl(
                text=full_input, 
                granularity=granularity,
                condition_mode="prefix",
                condition_pos_id=condition_pos
            )["ppl"]


            ppl_value = question_ppl_without_context - question_ppl_with_context


        self.cache["conditional_ppl"][cache_key] = ppl_value
        self._manage_cache_size("conditional_ppl")
        
        return ppl_value
       
    def control_context_budget(
        self,
        context_list: List[str],
        target_token: float,
        question: str = "",
        reorder_context: str = "original",
        condition_in_question: str = "none",
        force_context_ids: List[int] = None,
        force_context_number: int = None,
        context_budget: str = "+100",
        dynamic_context_compression_ratio: float = 0.0,
    ):
        """
        Control token budget for contexts based on relevance ranking, following LongLLMLingua.
        
        Args:
            context_list: List of contexts
            target_token: Target number of tokens
            question: Question for relevance ranking
            reorder_context: How to reorder contexts ("original", "importance", "two_stage")
            condition_in_question: Mode for conditional ranking
            force_context_ids: List of context IDs to always include
            force_context_number: Number of contexts to forcibly include
            context_budget: String expression to modify target token budget
            dynamic_context_compression_ratio: Ratio for dynamic compression (0.0-1.0)
            
        Returns:
            Selected contexts, their indices, and dynamic ratios
        """
        logger.debug(f"Controlling context budget with target_token={target_token}")
        start_time = time.time()
        
        if not context_list:
            return [], [], []
        
        # Get token counts for each context
        logger.debug("Calculating token lengths for contexts")
        context_tokens_length = [self.get_token_length(context) for context in context_list]
        
        # If total tokens already fit within budget, return all contexts
        total_tokens = sum(context_tokens_length)
        if total_tokens <= target_token:
            logger.debug(f"All contexts fit within budget ({total_tokens} <= {target_token})")
            end_time = time.time()
            logger.debug(f"Context budget control completed in {end_time - start_time:.2f} seconds")
            return context_list, list(range(len(context_list))), [0.0] * len(context_list)
        
        # Rank contexts by relevance if question is provided
        logger.debug("Ranking contexts by relevance")
        if question:
            # Get perplexity change for each context with the question
            context_ppl_changes = []
            for d, dl in zip(context_list, context_tokens_length):
                # Calculate how much this context reduces question perplexity
                ppl_change = self.get_condition_ppl(
                    d,
                    question,
                    condition_in_question,
                )
                # Apply length adjustment factor similar to before
                context_ppl_changes.append(ppl_change - dl * 2 / 250 * 0)
            
            # Sort by perplexity change - higher is better (more reduction in question perplexity)
            demonstrations_sort = sorted(enumerate(context_ppl_changes), key=lambda x: -x[1])
        else:
            # Without question, use default ordering
            demonstrations_sort = [(i, 0) for i in range(len(context_list))]
        
        # Extract ranking for later use
        self.context_idxs.append([x for idx, (x, _) in enumerate(demonstrations_sort)])
        
        # Calculate the target token budget with context_budget expression
        if target_token < 0:
            target_token = 100
        target_token = eval("target_token" + context_budget)
        
        # Initialize selected context tracking
        used = force_context_ids if force_context_ids is not None else []
        
        # Select contexts until we reach the token budget
        for idx, _ in demonstrations_sort:
            if idx >= len(context_tokens_length):
                continue
            target_token -= context_tokens_length[idx]
            if idx not in used:
                used.append(idx)
            if target_token < 0 or (
                force_context_number is not None and len(used) >= force_context_number
            ):
                break
        
        # Store original selection order
        original_used = used.copy()
        
        # Reorder contexts if requested
        if reorder_context == "original":
            used = sorted(used)
        elif reorder_context == "two_stage":
            l, r = [_ for idx, _ in enumerate(used) if idx % 2 == 0], [
                _ for idx, _ in enumerate(used) if idx % 2 == 1
            ]
            used = l + r[::-1]
        
        # Calculate dynamic compression ratios if requested
        if dynamic_context_compression_ratio > 0:
            N = len(used)
            dynamic_ratio = [
                i * (abs(dynamic_context_compression_ratio) / (N - 1)) if N > 1 else 0
                for i in range(-(N - 1), N, 2)
            ][::-1]
            dynamic_ratio_map = {i: j for i, j in zip(original_used, dynamic_ratio)}
            dynamic_ratio = [dynamic_ratio_map[i] for i in used]
        else:
            dynamic_ratio = [0.0] * len(used)
        
        # Build list of selected contexts
        selected_contexts = [context_list[idx] for idx in used if idx < len(context_list)]
        
        end_time = time.time()
        logger.debug(f"Selected {len(selected_contexts)} contexts out of {len(context_list)}")
        logger.debug(f"Context budget control completed in {end_time - start_time:.2f} seconds")
        
        return selected_contexts, used, dynamic_ratio, demonstrations_sort
    
    def compress_code_file(
        self,
        code: str,
        query: str = "",
        instruction: str = "",
        rate: float = 0.5,
        target_token: float = -1,
        language: str = "python",
        use_iterative_compression: bool = True,
        iterative_size: int = 200,
        dynamic_compression_ratio: float = 0.2,
        context_budget: str = "+100",
        rank_only: bool = False,
        fine_ratio: float = None,
        fine_grained_importance_method: str = "conditional_ppl",
        min_lines_for_fine_grained: int = 5,
        importance_beta: float = 0.5,
        use_knapsack: bool = True,
        coarse_expand_ratio: float = 1.4,
        use_semantic_fine_blocks: bool = True,
        semantic_fine_budget_factor: float = 0.94,
        semantic_min_fine_ratio: float = 0.42,
        semantic_docstring_fallback: bool = True,
    ):
        """
        Compress a code file by first splitting it into function-based chunks and then compressing.
        Functions are prioritized based on query relevance, similar to LongLLMLingua.
        
        Args:
            code: The code to compress
            query: Query to prioritize relevant functions
            instruction: Additional instruction to guide compression
            rate: Compression rate for coarse-grained (function level) compression (0.0-1.0)
            target_token: Target number of tokens (alternative to rate)
            language: Programming language of the code
            use_iterative_compression: Whether to use iterative compression
            iterative_size: Size of each iteration for iterative compression
            dynamic_compression_ratio: Ratio for dynamic compression
            context_budget: String expression to modify token budget
            rank_only: If True, just rank and select contexts without fine-grained compression
            fine_ratio: Ratio for fine-grained line selection (0.0-1.0). If None, uses `rate`.
            fine_grained_importance_method: Method for scoring line importance ('contrastive_perplexity' or 'conditional_ppl'). Defaults to 'conditional_ppl'.
            min_lines_for_fine_grained: Minimum number of lines a function must have to undergo fine-grained compression (otherwise kept fully).
            importance_beta: Controls how much function importance affects its individual compression rate during fine-grained compression.
            use_knapsack: Whether to use knapsack algorithm for block selection (True) or greedy line-by-line approach (False).
            use_semantic_fine_blocks: Prefer split_code2.py semantic blocks for normal Python functions during fine-grained compression.
            semantic_fine_budget_factor: Extra token-budget multiplier for semantic blocks; lower values improve compression.
            semantic_min_fine_ratio: Lower bound for semantic-block fine ratio to avoid over-pruning important functions.
            semantic_docstring_fallback: Use entropy blocks when the completion query is inside a docstring.
            
        Returns:
            Compressed code and statistics with the following structure:
            {
                "original_code": Original uncompressed code,
                "compressed_code": Compressed code,
                "compressed_prompt": Complete compressed prompt with instruction and query,
                "original_tokens": Number of tokens in original code,
                "compressed_tokens": Number of tokens in compressed code,
                "final_compressed_tokens": Number of tokens in final compressed prompt,
                "compression_ratio": Ratio of compressed to original tokens,
                "function_compressions": Details about compression for each function,
                "selected_functions": Indices of selected functions,
                "demonstrations_sort": Ranking of functions by importance,
                "compressed_chunks": List of compressed code chunks
                "fine_grained_method_used": The method used for fine-grained importance scoring.
            }
        """
        logger.debug(f"Starting code file compression with rate={rate}, target_token={target_token}, language={language}")
        start_time = time.time()
        
        # Split code into function-based chunks.
        # Keep the original lexical order by default.  Hoisting nested functions
        # changed local context shape in several completion cases and made the
        # combined coarse+fine setting unstable.
        logger.debug("Splitting code into function-based chunks")
        preprocessed_code = code

        code_chunks = self.split_code_by_functions(preprocessed_code, language=language)
        logger.debug(f"Split code into {len(code_chunks)} chunks")
        
        # Calculate total tokens
        logger.debug("Calculating total tokens")
        total_tokens = sum(self.get_token_length(chunk) for chunk in code_chunks)
        logger.debug(f"Total tokens: {total_tokens}")

        # Determine target_token based on rate if not specified
        original_target_token = target_token # Store original value if provided
        if target_token <= 0:
            if rate <= 0:
                 # Default target if both rate and target_token are invalid
                target_token = int(total_tokens * 0.5)
                logger.warning(f"Rate and target_token invalid, defaulting target_token to {target_token}")
            else:
                target_token = int(total_tokens * rate)
        logger.debug(f"Coarse Target tokens: {target_token}")
        

        # Build function signature dictionary + call graph statistics
        logger.debug("Building function signature dictionary and call-graph statistics")
        function_chunk_stats = self._build_function_call_stats(code_chunks, language=language)

        # Print detailed coarse-grained statistics
        logger.info("\n" + "=" * 100)
        logger.info("粗粒度函数块调用统计")
        logger.info("=" * 100)

        function_candidates: List[Dict[str, Any]] = []
        query_for_ranking = query.strip()

        for info in function_chunk_stats:
            idx = int(info["index"])
            chunk = info["chunk"]
            chunk_tokens = int(info["tokens"])
            call_score = int(info["called_by_count"]) if info["is_function"] else 0
            raw_ppl = float(info.get("ppl", 0.0))
            ppl_change = 0.0

            if query_for_ranking:
                try:
                    ppl_change = float(self.get_condition_ppl(chunk, query_for_ranking, "prefix"))
                except Exception as e:
                    logger.warning(f"[粗粒度] 计算 Chunk {idx} 的 ppl_change 失败，回退为 0.0: {e}")
                    ppl_change = 0.0
            else:
                ppl_change = 0.0

            function_candidates.append({
                "orig_index": idx,
                "chunk": chunk,
                "function_name": info["function_name"],
                "signature_text": info["signature_text"],
                "is_function": bool(info["is_function"]),
                "ppl": float(ppl_change),
                "raw_ppl": float(raw_ppl),
                "ppl_change": float(ppl_change),
                "call_score": float(call_score),
                "call_count": int(info["call_count"]),
                "called_by_count": int(info["called_by_count"]),
                "calls": list(info["calls"]),
                "called_by": list(info["called_by"]),
                "calls_text": list(info["calls_text"]),
                "called_by_text": list(info["called_by_text"]),
                "tokens": chunk_tokens,
            })

            if info["is_function"]:
                logger.info(
                    f"[Func {idx}] {info['signature_text'] or info['function_name'] or '<unknown>'}\n"
                    f"  tokens={chunk_tokens}, raw_ppl={raw_ppl:.6f}, ppl_change={ppl_change:.6f}, call_count={int(info['call_count'])}, "
                    f"called_by_count={int(info['called_by_count'])}, call_total={call_score}\n"
                    f"  calls -> {info['calls_text'] or []}\n"
                    f"  called_by <- {info['called_by_text'] or []}"
                )
            else:
                preview = chunk.strip().split("\n", 1)[0][:100] if chunk.strip() else "<EMPTY>"
                logger.info(
                    f"[Chunk {idx}] 非函数块/普通块\n"
                    f"  tokens={chunk_tokens}, raw_ppl={raw_ppl:.6f}, ppl_change={ppl_change:.6f}, call_total=0, preview={preview!r}"
                )

        function_rank_scores: Dict[int, float] = {}
        if function_candidates:
            score_matrix = np.array(
                [[float(item["ppl_change"]) ] for item in function_candidates],
                dtype=float,
            )
            score_norm = self._coarse_minmax_normalize(score_matrix)
            for item, row in zip(function_candidates, score_norm.tolist()):
                function_rank_scores[int(item["orig_index"])] = float(row[0])

        logger.info("[粗粒度] 归一化后的单块得分（基于 ppl_change，用于后续细粒度分配）:")
        for idx in range(len(code_chunks)):
            score = function_rank_scores.get(idx, 0.0)
            logger.info(f"  Chunk {idx}: score={score:.6f}")

        demonstrations_sort = sorted(function_rank_scores.items(), key=lambda x: (-x[1], x[0]))

        # Determine target_token based on rate if not specified
        original_target_token = target_token
        if target_token <= 0:
            if rate <= 0:
                target_token = int(total_tokens * 0.5)
                logger.warning(f"Rate and target_token invalid, defaulting target_token to {target_token}")
            else:
                target_token = int(total_tokens * rate)
        logger.debug(f"Coarse Target tokens: {target_token}")
        logger.info(f"[粗粒度] 粗粒度阶段不考虑保留块；使用原始预算 target_token={target_token}，并在选择器内额外加 context_budget={context_budget}")

        # Use two-stage ranking to select coarse chunks
        selected_function_indices: set = set()
        preserved_chunk_indices: set = set()  # defensive: kept empty; coarse stage no longer uses preserved blocks
        coarse_selection_info: Dict[str, Any] = {}
        if language.lower() == "python" and function_candidates:
            logger.info(f"[粗粒度] 使用两阶段排序：先 ppl_change，再调用数；r={coarse_expand_ratio}, k=ceil(r*p)，并使用稳定排序")
            selected_function_indices, coarse_selection_info = self._nsga_function_selection(
                items=function_candidates,
                target_tokens=target_token,
                max_archive=512,
                language=language,
                expansion_ratio=coarse_expand_ratio,
                context_budget=context_budget,
            )
            selected_indices = sorted(set(selected_function_indices))
        else:
            logger.warning("[粗粒度] 非 Python 或无函数块，回退到原有上下文控制策略")
            selected_contexts, selected_indices, dynamic_ratios, demonstrations_sort = self.control_context_budget(
                code_chunks,
                target_token=target_token,
                question=query,
                reorder_context="original",
                condition_in_question="prefix",
                context_budget=context_budget,
                dynamic_context_compression_ratio=dynamic_compression_ratio,
            )
            selected_function_indices = set(i for i in selected_indices if i < len(code_chunks))
            coarse_selection_info = {"method": "context_budget_fallback"}

        # Use selected chunks as is for the coarse stage
        logger.debug("Using rank-only mode: selecting top chunks without fine-grained compression")
        compressed_chunks = []
        compressed_tokens = 0
        function_compressions = {}

        selected_indices = sorted(set(selected_function_indices))

        for i, chunk in enumerate(code_chunks):
            if i in selected_indices:
                compressed_chunks.append(chunk)
                chunk_tokens = self.get_token_length(chunk)
                compressed_tokens += chunk_tokens
                function_compressions[i] = {
                    "original_tokens": chunk_tokens,
                    "compressed_tokens": chunk_tokens,
                    "compression_ratio": 1.0,
                    "call_count": int(function_chunk_stats[i]["call_count"]) if i < len(function_chunk_stats) else 0,
                    "called_by_count": int(function_chunk_stats[i]["called_by_count"]) if i < len(function_chunk_stats) else 0,
                    "call_score": int(function_chunk_stats[i]["called_by_count"]) if i < len(function_chunk_stats) else 0,
                    "preserved": False,
                }
            else:
                comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
                omission_text = f"{comment_marker} ... "
                compressed_chunks.append(omission_text)
                compressed_tokens += self.get_token_length(omission_text)
                function_compressions[i] = {
                    "original_tokens": self.get_token_length(chunk),
                    "compressed_tokens": self.get_token_length(omission_text),
                    "compression_ratio": self.get_token_length(omission_text) / max(self.get_token_length(chunk), 1),
                    "call_count": int(function_chunk_stats[i]["call_count"]) if i < len(function_chunk_stats) else 0,
                    "called_by_count": int(function_chunk_stats[i]["called_by_count"]) if i < len(function_chunk_stats) else 0,
                    "call_score": int(function_chunk_stats[i]["called_by_count"]) if i < len(function_chunk_stats) else 0,
                    "preserved": False,
                }

        compressed_code = "\n\n".join(compressed_chunks)

        # --- Post-join cleanup for consecutive omission markers ---
        logger.debug("Cleaning up consecutive omission markers after joining...")
        lines = compressed_code.split("\n")
        cleaned_lines = []
        last_non_empty_line_was_omission = False
        comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
        omission_marker_content = f"{comment_marker} ...".strip() # Content to check against

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                # Keep empty lines
                cleaned_lines.append(line)
                # Don't reset the flag here, wait for a non-empty line
            elif stripped_line == omission_marker_content:
                if last_non_empty_line_was_omission:
                    # Skip this consecutive omission marker line
                    logger.debug(f"Skipping line: '{line}' (consecutive omission)")
                    continue
                else:
                    # Keep the first omission marker line
                    cleaned_lines.append(line)
                    last_non_empty_line_was_omission = True
            else:
                # Regular code line
                cleaned_lines.append(line)
                last_non_empty_line_was_omission = False

        compressed_code = "\n".join(cleaned_lines)
        logger.debug("Cleanup finished.")
        # --- End post-join cleanup ---


        output = f"{instruction}\n\n{compressed_code}\n\n{query}\n{instruction}"
        
        # Calculate actual compressed tokens
        final_compressed_tokens = self.get_token_length(output)
        
        end_time = time.time()
        logger.debug(f"Code file compression completed in {end_time - start_time:.2f} seconds")
        logger.debug(f"Compression ratio: {compressed_tokens / total_tokens if total_tokens > 0 else 1.0:.2f}")
        
        if rank_only:
            return {
                "original_code": code,
                "compressed_code": compressed_code,
                "compressed_prompt": output,
                "original_tokens": total_tokens,
                "compressed_tokens": compressed_tokens,
                "final_compressed_tokens": final_compressed_tokens,
                "compression_ratio": compressed_tokens / total_tokens if total_tokens > 0 else 1.0,
                "function_compressions": function_compressions,
                "selected_functions": sorted(selected_function_indices),
                "demonstrations_sort": demonstrations_sort,
                "compressed_chunks": compressed_chunks,
                "fine_grained_method_used": None,
            }
        else:
            # enter fine-grained compression
            logger.debug(f"Starting fine-grained compression on selected functions using method: {fine_grained_importance_method}")

            # --- Dynamic Fine-grained Rate Allocation ---
            logger.debug("Calculating dynamic fine-grained compression rates...")

            # 1. Collect data for selected functions
            selected_functions_data = []
            importance_map = {idx: score for idx, score in demonstrations_sort} # Map index to score
            total_lines_selected = 0
            for i in selected_indices:
                if i < len(code_chunks):
                    chunk = code_chunks[i]
                    # Use simple line splitting for allocation efficiency
                    lines = chunk.split("\n")
                    line_count = len(lines)
                    score = importance_map.get(i, 0.0) # Default score 0 if not found
                    selected_functions_data.append({
                        "index": i,
                        "lines": lines,
                        "line_count": line_count,
                        "score": score
                    })
                    total_lines_selected += line_count
                else:
                     logger.warning(f"Selected index {i} is out of bounds for code_chunks (length {len(code_chunks)})")


            # 2. Calculate overall fine-grained target lines
            current_fine_ratio = fine_ratio if fine_ratio is not None else rate # Use rate if fine_ratio not set
            if original_target_token > 0: # If target_token was set explicitly, derive ratio from it for fine-grained stage
                 # Estimate target lines based on the ratio of selected tokens to total tokens, then apply fine_ratio
                 selected_tokens = sum(self.get_token_length(code_chunks[d['index']]) for d in selected_functions_data)
                 effective_coarse_rate = selected_tokens / total_tokens if total_tokens > 0 else 1.0
                 # Use the user-provided fine_ratio, or fall back to rate/coarse target estimate
                 fine_target_rate = current_fine_ratio
                 logger.debug(f"Using fine_ratio={fine_target_rate} for fine-grained target calculation.")
                 target_total_lines = int(total_lines_selected * fine_target_rate)

            else: # Calculate target based on fine_ratio/rate directly applied to selected lines
                 target_total_lines = int(total_lines_selected * current_fine_ratio)
                 logger.debug(f"Using current_fine_ratio={current_fine_ratio} for fine-grained target calculation.")

            logger.debug(f"Total lines in selected functions: {total_lines_selected}")
            logger.debug(f"Target total lines after fine-grained compression: {target_total_lines}")

            # 3. Separate small and large functions
            small_functions = []
            large_functions = []
            lines_in_small_functions = 0
            lines_in_large_functions = 0

            for data in selected_functions_data:
                if data["line_count"] < min_lines_for_fine_grained:
                    small_functions.append(data)
                    lines_in_small_functions += data["line_count"]
                else:
                    large_functions.append(data)
                    lines_in_large_functions += data["line_count"]

            logger.debug(f"Found {len(small_functions)} small functions (< {min_lines_for_fine_grained} lines) with {lines_in_small_functions} total lines (will be kept).")
            logger.debug(f"Found {len(large_functions)} large functions (>= {min_lines_for_fine_grained} lines) with {lines_in_large_functions} total lines.")

            # 4. Calculate target lines for large functions
            target_lines_for_large = max(0, target_total_lines - lines_in_small_functions)
            logger.debug(f"Target lines to keep from large functions: {target_lines_for_large}")

            # 5. Allocate rates for large functions
            function_fine_ratios = {} # Map: index -> individual_fine_ratio

            if not large_functions or lines_in_large_functions == 0:
                 logger.debug("No large functions to compress further or zero lines in large functions.")
                 global_rate_for_large = 1.0 if lines_in_large_functions > 0 else 0.0 # Should be 0 if lines_in_large_functions is 0
            elif target_lines_for_large <= 0:
                 logger.debug("Target lines for large functions is <= 0. Setting rates to 0.")
                 global_rate_for_large = 0.0
            elif target_lines_for_large >= lines_in_large_functions:
                 logger.debug("Target lines for large functions >= total lines. Setting rates to 1.0.")
                 global_rate_for_large = 1.0
            else:
                global_rate_for_large = target_lines_for_large / lines_in_large_functions
                logger.debug(f"Global target rate for large functions: {global_rate_for_large:.4f}")

                # Normalize scores for weighting (MinMax scaling)
                scores = [d["score"] for d in large_functions]
                valid_scores = [s for s in scores if not math.isinf(s) and not math.isnan(s)]

                if not valid_scores or max(valid_scores) == min(valid_scores):
                    logger.debug("Scores are uniform or invalid, using global rate for all large functions.")
                    for data in large_functions:
                        function_fine_ratios[data["index"]] = global_rate_for_large
                else:
                    min_score = min(valid_scores)
                    max_score = max(valid_scores)
                    normalized_scores = [(s - min_score) / (max_score - min_score) if not math.isinf(s) and not math.isnan(s) else 0.0 for s in scores] # Normalize to [0, 1], default 0 for invalid

                    # Calculate initial biased rates
                    initial_rates = []
                    for norm_score in normalized_scores:
                        # Bias rate: higher score -> higher rate (closer to 1)
                        # Beta controls sensitivity. beta=0 -> uniform rate. beta=1 -> max sensitivity.
                        biased_rate = global_rate_for_large * (1 + importance_beta * (norm_score - 0.5) * 2) # Scale norm_score diff to [-beta, beta]
                        clamped_rate = max(0.0, min(1.0, biased_rate)) # Clamp to [0, 1]
                        initial_rates.append(clamped_rate)

                    # Calculate actual lines kept with initial rates
                    actual_lines_kept = sum(initial_rates[i] * large_functions[i]["line_count"] for i in range(len(large_functions)))
                    logger.debug(f"Initial biased rates calculated. Estimated lines kept: {actual_lines_kept:.1f}")

                    # Adjust rates proportionally to meet target
                    if actual_lines_kept > 0 and abs(actual_lines_kept - target_lines_for_large) > 1: # Adjust if difference is significant
                        adjustment_factor = target_lines_for_large / actual_lines_kept
                        logger.debug(f"Adjusting rates by factor: {adjustment_factor:.4f}")
                        final_rates = [max(0.0, min(1.0, r * adjustment_factor)) for r in initial_rates] # Adjust and clamp again
                    else:
                        logger.debug("Initial rates are close enough or actual_lines_kept is 0, no adjustment needed.")
                        final_rates = initial_rates

                    for i, data in enumerate(large_functions):
                        function_fine_ratios[data["index"]] = final_rates[i]

            # Set rate 1.0 for small functions
            for data in small_functions:
                function_fine_ratios[data["index"]] = 1.0

            # --- End Dynamic Allocation ---


            # Apply fine-grained compression to each selected function
            fine_compressed_chunks = []
            compressed_tokens = 0
            function_compressions = {}

            # Define a smoothing window size for moving average
            smoothing_window = 5
            # fine_ratio = fine_ratio if fine_ratio is not None else rate # Use the same ratio by default if fine_ratio not specified # Removed, using individual ratios now

            # Process each chunk in the original order
            # Use tqdm.auto for compatibility
            fine_grained_pbar = tqdm(enumerate(code_chunks), total=len(code_chunks), desc="Fine-Grained Compression", leave=False)
            for i, chunk in fine_grained_pbar:
            # for i, chunk in enumerate(code_chunks):
                if i in selected_indices:
                    # This function was selected during coarse-grained compression
                    individual_fine_ratio = function_fine_ratios.get(i) # Get dynamically assigned ratio
                    if individual_fine_ratio is None:
                         logger.error(f"Missing fine-grained ratio for selected function index {i}. Skipping fine-grained compression for this chunk.")
                         individual_fine_ratio = 1.0 # Fallback to keep the chunk

                    # ff7: prefer split_code2 semantic blocks as the fine-grained
                    # unit.  They preserve AST/control-flow boundaries and skip
                    # docstring/comment prose that made c_ff1.log under-compress.
                    # Fall back to the first group's entropy path when semantic
                    # splitting is unavailable or unsafe for docstring completion.
                    current_chunk = code_chunks[i]
                    chunks = []
                    sentences = []
                    ppls = []
                    spike_indices = []
                    block_dependency_counts = []
                    semantic_blocks = []
                    semantic_chunks = []
                    semantic_dep_counts = []
                    semantic_attempted = False
                    chunk_source = "entropy"
                    block_metadata: List[Dict[str, Any]] = []
                    block_joiner = "\n\n"
                    query_inside_docstring = self._query_is_inside_docstring(query)
                    normal_python_function = language.lower() == "python" and self._is_normal_python_function_chunk(current_chunk)

                    if (
                        use_semantic_fine_blocks
                        and normal_python_function
                        and not (semantic_docstring_fallback and query_inside_docstring)
                        and self._classify_completion_context(query).get("family", "general") not in {"test", "control", "control_loop", "doc_text", "class_body", "api_schema_table", "decorator"}
                    ):
                        semantic_attempted = True
                        semantic_chunks, semantic_dep_counts, semantic_blocks = self._get_semantic_blocks_and_dependency_counts(
                            current_chunk, language=language
                        )

                        if semantic_blocks and self._semantic_chunks_are_usable(current_chunk, semantic_chunks):
                            chunks, block_dependency_counts, block_metadata = self._postprocess_split_code2_blocks(
                                func_idx=i,
                                chunks=semantic_chunks,
                                dependency_counts=list(semantic_dep_counts),
                                semantic_blocks=semantic_blocks,
                                query=query,
                                language=language,
                            )
                            chunk_source = "split_code2_semantic"
                            block_joiner = "\n"
                            logger.debug(
                                f"Func {i}: using split_code2 semantic blocks "
                                f"({len(semantic_chunks)} -> {len(chunks)} blocks) for fine-grained compression."
                            )
                        else:
                            logger.debug(
                                f"Func {i}: split_code2 semantic blocks unavailable or too coarse; "
                                "falling back to entropy chunks."
                            )

                    if not chunks:
                        chunks, sentences, ppls, spike_indices = self.entropy_chunking.chunk_text_adaptive(
                            current_chunk,
                            method="std",
                            k=0.2,
                        )
                        if not chunks:
                            chunks = [current_chunk] if current_chunk.strip() else []
                        block_dependency_counts = [0.0] * len(chunks)

                        if normal_python_function:
                            # 第一组原路径：计算 split_code2/graph.py 的依赖，但投影到 entropy block。
                            if not semantic_attempted:
                                semantic_chunks, semantic_dep_counts, semantic_blocks = self._get_semantic_blocks_and_dependency_counts(
                                    current_chunk, language=language
                                )

                            if semantic_blocks and len(semantic_dep_counts) == len(semantic_blocks):
                                block_dependency_counts = self._project_semantic_dependencies_to_entropy_chunks(
                                    current_chunk,
                                    chunks,
                                    semantic_blocks,
                                    semantic_dep_counts,
                                )
                            elif semantic_chunks:
                                logger.debug(
                                    f"Func {i}: semantic dependency info unavailable or misaligned; "
                                    "using entropy chunks with zero dependency feature."
                                )
                        else:
                            logger.debug(f"Func {i} is not a normal Python function chunk; using EntropyChunking.")
                    else:
                        sentences = current_chunk.splitlines()
                        ppls = [float(len(c.splitlines())) for c in chunks]
                        spike_indices = [j for j in range(max(0, len(chunks) - 1))]

                    self._log_semantic_chunks(
                        i,
                        code_chunks[i],
                        chunks,
                        language,
                        block_source=chunk_source,
                        block_metadata=block_metadata,
                        block_dependency_counts=block_dependency_counts,
                    )
                    # Use chunks as lines, but preserve all chunks including empty ones to maintain formatting
                    chunk_lines = chunks  # Keep all chunks to preserve \n\n and formatting
                    chunk_line_count = len([chunk for chunk in chunk_lines if chunk.strip()])  # Count only non-empty for logic
                    original_nonempty_line_count = len([line for line in current_chunk.splitlines() if line.strip()])
                    chunk_score = importance_map.get(i, float('nan')) # Get score
                    if len(block_dependency_counts) != len(chunk_lines):
                        logger.warning(
                            f"[NSGA-II] dependency count length mismatch for Func {i}: "
                            f"chunks={len(chunk_lines)}, dep_counts={len(block_dependency_counts)}; "
                            f"padding/truncating instead of zeroing."
                        )
                        if len(block_dependency_counts) < len(chunk_lines):
                            block_dependency_counts = block_dependency_counts + [0] * (len(chunk_lines) - len(block_dependency_counts))
                        else:
                            block_dependency_counts = block_dependency_counts[:len(chunk_lines)]

                    logger.debug(
                        f"Processing Func {i}: BlockSource={chunk_source}, Blocks={len(chunk_lines)}, "
                        f"Non-empty={chunk_line_count}, OriginalNonEmptyLines={original_nonempty_line_count}, "
                        f"Score={chunk_score:.4f}, Assigned FineRatio={individual_fine_ratio:.4f}"
                    )


                    # Skip fine-grained compression if ratio is 1.0 (or close) or function is small
                    if individual_fine_ratio >= 0.999 or original_nonempty_line_count < min_lines_for_fine_grained or chunk_line_count <= 1:
                        if individual_fine_ratio >= 0.999:
                            note = "Kept (Ratio=1.0)"
                        elif chunk_line_count <= 1:
                            note = "Kept (Only one fine-grained block)"
                        else:
                            note = f"Kept (Small Func < {min_lines_for_fine_grained} lines)"
                        logger.debug(f"  - {note}")
                        fine_compressed_chunks.append(chunk)
                        chunk_tokens = self.get_token_length(chunk)
                        compressed_tokens += chunk_tokens
                        function_compressions[i] = {
                            "original_tokens": chunk_tokens,
                            "compressed_tokens": chunk_tokens,
                            "compression_ratio": 1.0,
                            "individual_fine_ratio": individual_fine_ratio,
                            "note": note,
                            "importance_method": None # No line importance calculation needed
                        }
                        continue # Move to next chunk


                    # Apply fine-grained compression only if the function is large enough
                    # and we're not in rank-only mode (already checked) and ratio < 1.0
                    if original_nonempty_line_count >= min_lines_for_fine_grained and chunk_line_count > 1 and individual_fine_ratio < 0.999:
                        effective_fine_ratio = individual_fine_ratio
                        if chunk_source == "split_code2_semantic":
                            effective_fine_ratio = max(
                                semantic_min_fine_ratio,
                                min(individual_fine_ratio, individual_fine_ratio * semantic_fine_budget_factor),
                            )
                        logger.debug(
                            f"  - Applying fine-grained compression with ratio {effective_fine_ratio:.4f} "
                            f"(assigned={individual_fine_ratio:.4f}, source={chunk_source})"
                        )
                        fine_grained_pbar.set_description(f"Fine-Grained Compressing Func {i}")
                        
                        # Calculate target tokens for this function
                        original_func_tokens = self.get_token_length(chunk)
                        fine_unit_tokens = sum(self.get_token_length(block) for block in chunk_lines)
                        target_base_tokens = fine_unit_tokens if chunk_source == "split_code2_semantic" else original_func_tokens
                        target_func_tokens = max(1, int(target_base_tokens * effective_fine_ratio))
                        logger.info(
                            f"[MO-KNAPSACK][BUDGET] func={i} original_tokens={original_func_tokens} "
                            f"fine_unit_tokens={fine_unit_tokens} target_tokens={target_func_tokens} "
                            f"assigned_ratio={individual_fine_ratio:.4f} effective_ratio={effective_fine_ratio:.4f} "
                            f"block_source={chunk_source} blocks={len(chunk_lines)}"
                        )
                        
                        # Calculate importance for each block based on the chosen method
                        block_importances = []
                        importance_calculation_start = time.time()

                        if fine_grained_importance_method == "conditional_ppl":
                            # Calculate conditional PPL importance for each block
                            if not query or not query.strip():
                                logger.warning(f"Query is empty for func {i}, cannot calculate conditional PPL. Assigning 0 importance.")
                                block_importances = [0.0] * len(chunk_lines)
                            else:
                                query_ppl_result = self.get_ppl(query, granularity="line")
                                query_ppl_without_context = query_ppl_result["ppl"]

                                if math.isinf(query_ppl_without_context):
                                    logger.warning(f"Base query PPL is infinite for func {i}. Assigning 0 importance to blocks.")
                                    block_importances = [0.0] * len(chunk_lines)
                                else:
                                    pbar_cond = tqdm(enumerate(chunk_lines), total=len(chunk_lines), desc=f"Func {i} Block CondPPL", leave=False)
                                    for block_idx, block in pbar_cond:
                                        if not block.strip():
                                            block_importances.append(-float('inf'))  # Low score for empty blocks
                                            continue

                                        conditional_text = block + "\n\n" + query
                                        prefix_len_text = block + "\n\n"
                                        prefix_len = self.get_token_length(prefix_len_text, add_special_tokens=True)

                                        cond_ppl_result = self.get_ppl(
                                            text=conditional_text,
                                            granularity="line",
                                            condition_mode="prefix",
                                            condition_pos_id=prefix_len
                                        )
                                        ppl_with_context = cond_ppl_result["ppl"]

                                        if math.isinf(ppl_with_context):
                                            ppl_change = -float('inf')
                                        else:
                                            ppl_change = query_ppl_without_context - ppl_with_context

                                        block_importances.append(ppl_change)
                                        pbar_cond.set_description(f"Func {i} Block CondPPL (B{block_idx}: {ppl_change:.2f})")

                        elif fine_grained_importance_method == "contrastive_perplexity":
                            # Calculate contrastive PPL importance for each block
                            fine_grained_pbar.set_description(f"Fine-Grained ContrastivePPL Func {i}")
                            
                            with torch.no_grad():
                                pbar = tqdm(enumerate(chunk_lines), total=len(chunk_lines), desc="Block Contrastive PPL", leave=False)
                                for block_idx, block in pbar:
                                    if not block.strip():
                                        block_importances.append(-float('inf'))
                                        continue

                                    # Build context from previous blocks
                                    prev_context = "\n\n".join(chunk_lines[:block_idx]) if block_idx > 0 else ""
                                    
                                    # 1. PPL(Block | prev_blocks)
                                    regular_ppl_condition = prev_context + "\n\n" if prev_context else None
                                    regular_ppl = self._calculate_perplexity_for_contrastive(block, condition_text=regular_ppl_condition)

                                    # 2. PPL(Block | query, prev_blocks)
                                    question_context_parts = [query]
                                    if prev_context:
                                        question_context_parts.append(prev_context)
                                    question_context = "\n\n".join(filter(None, question_context_parts))
                                    cond_ppl_condition = question_context + "\n\n"
                                    cond_ppl = self._calculate_perplexity_for_contrastive(block, condition_text=cond_ppl_condition)

                                    # 3. Importance = PPL(Block|prev) - PPL(Block|Q,prev)
                                    if math.isinf(regular_ppl) or math.isinf(cond_ppl):
                                        importance = -float('inf')
                                    else:
                                        importance = regular_ppl - cond_ppl

                                    block_importances.append(importance)
                                    pbar.set_description(f"Block {block_idx}: {importance:.2f}")

                        else:
                            raise ValueError(f"Unsupported fine_grained_importance_method: {fine_grained_importance_method}")

                        importance_calculation_end = time.time()
                        logger.debug(f"  - Block importance calculation took {importance_calculation_end - importance_calculation_start:.2f}s")

                        comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
                        preserved_block_indices = self._select_preserved_completion_blocks(
                            chunk_lines,
                            query=query,
                            language=language,
                        )
                        multiobjective_block_placeholder = self._build_hybrid_completion_block_scores(
                            blocks=chunk_lines,
                            block_importances=block_importances,
                            block_dependency_counts=block_dependency_counts,
                            query=query,
                            language=language,
                        )

                        # Choose selection method based on use_knapsack parameter
                        processing_start = time.time()
                        
                        if use_knapsack:
                            # Use knapsack algorithm to select blocks
                            logger.debug(f"  - Using weight-free Pareto multi-objective knapsack for block selection")
                            selected_block_indices, selection_info = self._hybrid_knapsack_block_selection(
                                blocks=chunk_lines,
                                block_importances=block_importances,
                                hybrid_scores=multiobjective_block_placeholder,
                                block_dependency_counts=block_dependency_counts,
                                target_tokens=target_func_tokens,
                                preserved_block_indices=preserved_block_indices,
                                language=language,
                            )
                            
                            # Build compressed chunk from selected blocks
                            compressed_blocks = []
                            
                            # Determine base indentation for omission markers
                            base_indentation = ""
                            for block in chunk_lines:
                                for line in block.split('\n'):
                                    if line.strip():
                                        match = re.match(r"^(\s*)", line)
                                        if match:
                                            base_indentation = match.group(1)
                                        break
                                if base_indentation:
                                    break
                            
                            omission_marker = f"{base_indentation}{comment_marker} ... "
                            
                            # Build output with omission markers for gaps
                            last_selected_idx = -1
                            for block_idx in sorted(selected_block_indices):
                                # Add omission marker if there's a gap
                                if last_selected_idx != -1 and block_idx > last_selected_idx + 1:
                                    if not compressed_blocks or compressed_blocks[-1] != omission_marker:
                                        compressed_blocks.append(omission_marker)
                                
                                compressed_blocks.append(chunk_lines[block_idx])
                                last_selected_idx = block_idx

                            # Handle trailing omission if needed
                            if last_selected_idx != -1 and last_selected_idx < len(chunk_lines) - 1:
                                if not compressed_blocks or compressed_blocks[-1] != omission_marker:
                                    compressed_blocks.append(omission_marker)

                            # Entropy chunks use paragraph-like separators; split_code2
                            # blocks already carry exact statement indentation, so single
                            # newlines avoid adding many empty spacer lines.
                            compressed_chunk = block_joiner.join(compressed_blocks)
                            
                        else:
                            # Use original greedy line-by-line approach with smoothing
                            logger.debug(f"  - Using original greedy line-by-line approach")
                            
                            # Convert block importances to line importances for compatibility
                            lines = []
                            line_importances = []
                            line_indices = []
                            
                            for block_idx, (block, block_importance) in enumerate(zip(chunk_lines, block_importances)):
                                block_lines = block.split('\n')
                                for line_idx_in_block, line in enumerate(block_lines):
                                    global_line_idx = len(lines)
                                    lines.append(line)
                                    line_importances.append(block_importance)  # Use block importance for all lines in block
                                    line_indices.append(global_line_idx)
                            
                            # Apply original processing logic with smoothing
                            full_line_scores = [float('nan')] * len(lines)
                            for score_idx, original_line_idx in enumerate(line_indices):
                                if score_idx < len(line_importances):
                                    full_line_scores[original_line_idx] = line_importances[score_idx]

                            # Replace NaN/Inf with min valid score for consistent processing
                            valid_scores = [s for s in full_line_scores if not math.isnan(s) and not math.isinf(s)]
                            if valid_scores:
                                min_valid_score = min(valid_scores)
                                if min_valid_score == float('inf') or min_valid_score == -float('inf') or math.isnan(min_valid_score):
                                    min_replacement_score = 0.0
                                else:
                                    min_replacement_score = min_valid_score

                                processed_line_scores = []
                                for s in full_line_scores:
                                    if math.isnan(s) or s == -float('inf'):
                                        processed_line_scores.append(min_replacement_score)
                                    elif s == float('inf'):
                                        processed_line_scores.append(min_replacement_score)
                                    else:
                                        processed_line_scores.append(s)
                            else:
                                processed_line_scores = [0.0] * len(lines)

                            # Apply smoothing using moving average
                            smoothing_window = 5
                            smoothed_importances = processed_line_scores.copy()
                            num_processed_scores = len(processed_line_scores)
                            for j in range(num_processed_scores):
                                window_start = max(0, j - smoothing_window // 2)
                                window_end = min(num_processed_scores, j + smoothing_window // 2 + 1)
                                window = processed_line_scores[window_start:window_end]
                                valid_window_scores = [s for s in window if not math.isnan(s) and not math.isinf(s)]
                                if valid_window_scores:
                                    smoothed_importances[j] = sum(valid_window_scores) / len(valid_window_scores)

                            # Find preserved lines (convert block indices to line indices)
                            preserved_line_indices = set()
                            line_offset = 0
                            for block_idx, block in enumerate(chunk_lines):
                                block_lines = block.split('\n')
                                if block_idx in preserved_block_indices:
                                    for line_idx_in_block in range(len(block_lines)):
                                        preserved_line_indices.add(line_offset + line_idx_in_block)
                                line_offset += len(block_lines)

                            # Sort remaining lines by importance
                            sortable_lines = []
                            for idx in range(len(lines)):
                                if idx not in preserved_line_indices:
                                    if idx < len(line_indices) and idx < len(line_importances):
                                        original_score = line_importances[idx]
                                        if not math.isnan(original_score) and not math.isinf(original_score):
                                            smoothed_score = smoothed_importances[idx]
                                            sortable_lines.append((idx, smoothed_score))

                            # Sort descending by score
                            sorted_line_indices = sorted(sortable_lines, key=lambda x: -x[1])

                            # Calculate target number of lines to keep
                            total_lines = len(lines)
                            preserved_count = len(preserved_line_indices)
                            target_lines = max(preserved_count, int(total_lines * effective_fine_ratio))

                            # Select top lines by importance up to target
                            selected_lines = set(preserved_line_indices)
                            for idx, score in sorted_line_indices:
                                if len(selected_lines) >= target_lines:
                                    break
                                selected_lines.add(idx)

                            # Build compressed chunk from selected lines
                            compressed_chunks = []
                            base_indentation = ""
                            if lines:
                                for line in lines:
                                    if line.strip():
                                        match = re.match(r"^(\s*)", line)
                                        if match:
                                            base_indentation = match.group(1)
                                        break

                            omission_marker_line = f"{base_indentation}{comment_marker} ... "
                            
                            last_added_line_idx = -1
                            for j in range(len(lines)):
                                if j in selected_lines:
                                    if last_added_line_idx != -1 and j > last_added_line_idx + 1:
                                        if not compressed_chunks or compressed_chunks[-1] != omission_marker_line:
                                            compressed_chunks.append(omission_marker_line)
                                    compressed_chunks.append(lines[j])
                                    last_added_line_idx = j

                            if last_added_line_idx != -1 and last_added_line_idx < len(lines) - 1:
                                if not compressed_chunks or compressed_chunks[-1] != omission_marker_line:
                                    compressed_chunks.append(omission_marker_line)

                            compressed_chunk = "\n".join(compressed_chunks)
                            
                            # Create selection info for compatibility
                            selection_info = {
                                "method": "greedy_line_by_line",
                                "preserved_lines": len(preserved_line_indices),
                                "selected_lines": len(selected_lines),
                                "total_lines": len(lines),
                                "smoothing_applied": True,
                                "block_source": chunk_source,
                            }
                            selected_block_indices = preserved_block_indices  # For compatibility

                        processing_end = time.time()
                        if isinstance(selection_info, dict):
                            selection_info["block_source"] = chunk_source
                            selection_info["effective_fine_ratio"] = effective_fine_ratio
                            selection_info["target_func_tokens"] = target_func_tokens
                        method_name = "pareto_multiobjective_knapsack" if use_knapsack else "greedy"
                        logger.debug(f"  - {method_name} selection took {processing_end - processing_start:.2f}s")
                        
                        if use_knapsack:
                            logger.debug(f"  - Selected {len(selected_block_indices)}/{len(chunk_lines)} blocks")
                        else:
                            logger.debug(f"  - Selected {len(selected_lines)}/{len(lines)} lines")

                        # Update token count and store compression info
                        self._log_text_block(f"Func {i} - 细粒度压缩后的函数块", compressed_chunk)
                        fine_compressed_chunks.append(compressed_chunk)
                        compressed_chunk_tokens = self.get_token_length(compressed_chunk)
                        compressed_tokens += compressed_chunk_tokens

                        # Store compression info
                        actual_compression_ratio = compressed_chunk_tokens / original_func_tokens if original_func_tokens > 0 else 1.0
                        function_compressions[i] = {
                            "original_tokens": original_func_tokens,
                            "compressed_tokens": compressed_chunk_tokens,
                            "compression_ratio": actual_compression_ratio,
                            "individual_fine_ratio": individual_fine_ratio,
                            "effective_fine_ratio": effective_fine_ratio,
                            "block_source": chunk_source,
                            "fine_unit_tokens": fine_unit_tokens,
                            "target_func_tokens": target_func_tokens,
                            "preserved_blocks": list(preserved_block_indices),
                            "selected_blocks": list(selected_block_indices),
                            "selection_info": selection_info,
                            "importance_method": fine_grained_importance_method,
                            "selection_method": "pareto_multiobjective_knapsack" if use_knapsack else "greedy_line_by_line"
                        }
                        logger.info(
                            f"[MO-KNAPSACK][FUNC] func={i} original_tokens={original_func_tokens} "
                            f"compressed_tokens={compressed_chunk_tokens} keep_ratio={actual_compression_ratio:.4f} "
                            f"compression_ratio={original_func_tokens / max(1, compressed_chunk_tokens):.4f}"
                        )
                    else:
                         # This case should now be handled by the check at the beginning of the loop
                         logger.warning(f"Reached unexpected state for func {i}. Keeping chunk as is.")
                         fine_compressed_chunks.append(chunk)
                         chunk_tokens = self.get_token_length(chunk)
                         compressed_tokens += chunk_tokens
                         function_compressions[i] = {
                            "original_tokens": chunk_tokens,
                            "compressed_tokens": chunk_tokens,
                            "compression_ratio": 1.0,
                            "individual_fine_ratio": individual_fine_ratio,
                            "note": "Unexpected state, kept function.",
                            "importance_method": None
                         }

                else:
                    # This function was not selected during coarse-grained compression
                    # Add a placeholder
                    comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
                    omission_text = f"{comment_marker} ... "
                    fine_compressed_chunks.append(omission_text)
                    compressed_tokens += self.get_token_length(omission_text)
                    # Log skipped chunk
                    # logger.debug(f"Skipped Func {i} (not selected in coarse stage)")


            # Combine fine-grained compressed chunks
            compressed_code = "\n\n".join(fine_compressed_chunks)

            # --- Post-join cleanup for consecutive omission markers ---
            logger.debug("Cleaning up consecutive omission markers after joining...")
            lines = compressed_code.split("\n")
            cleaned_lines = []
            last_non_empty_line_was_omission = False
            comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
            omission_marker_content = f"{comment_marker} ...".strip() # Content to check against

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    # Keep empty lines
                    cleaned_lines.append(line)
                    # Don't reset the flag here, wait for a non-empty line
                elif stripped_line == omission_marker_content:
                    if last_non_empty_line_was_omission:
                        # Skip this consecutive omission marker line
                        logger.debug(f"Skipping line: '{line}' (consecutive omission)")
                        continue
                    else:
                        # Keep the first omission marker line
                        cleaned_lines.append(line)
                        last_non_empty_line_was_omission = True
                else:
                    # Regular code line
                    cleaned_lines.append(line)
                    last_non_empty_line_was_omission = False

            compressed_code = "\n".join(cleaned_lines)
            logger.debug("Cleanup finished.")
            self._log_text_block("整体细粒度压缩完后的结果", compressed_code)
            # --- End post-join cleanup ---


            # Ensure instruction/query parts are handled correctly, maybe use a template
            prompt_parts = []
            if instruction and instruction.strip():
                prompt_parts.append(instruction.strip())
            if compressed_code.strip():
                prompt_parts.append(compressed_code) # Already has newlines handled
            if query and query.strip():
                 # Add query, potentially repeating instruction based on original logic
                 prompt_parts.append(query.strip())
                 # Decide if instruction should be repeated after query based on original implementation's needs
                 # if instruction and instruction.strip(): # Repeat instruction if needed
                 #     prompt_parts.append(instruction.strip())

            output = "\n\n".join(prompt_parts) # Use double newline separation

            # Calculate final compressed tokens
            final_compressed_tokens = self.get_token_length(output)

            end_time = time.time()
            logger.debug(f"Fine-grained compression processing completed in {end_time - start_time:.2f} seconds")
            final_compression_ratio = compressed_tokens / total_tokens if total_tokens > 0 else 1.0
            logger.info(
                f"[MO-KNAPSACK][TOTAL] original_tokens={total_tokens} compressed_tokens={compressed_tokens} "
                f"keep_ratio={final_compression_ratio:.4f} compression_ratio={total_tokens / max(1, compressed_tokens):.4f}"
            )


            return {
                "original_code": code,
                "compressed_code": compressed_code,
                "compressed_prompt": output,
                "original_tokens": total_tokens,
                "compressed_tokens": compressed_tokens,
                "final_compressed_tokens": final_compressed_tokens,
                "compression_ratio": final_compression_ratio,
                "function_compressions": function_compressions,
                "selected_functions": selected_indices,
                "demonstrations_sort": demonstrations_sort,
                "compressed_chunks": fine_compressed_chunks,
                "fine_grained_method_used": fine_grained_importance_method,
            }
    

    def _log_text_block(self, title: str, text: str) -> None:
        """统一打印一段代码/文本，方便调试。"""
        logger.info("\n" + "=" * 100)
        logger.info(title)
        logger.info("\n" + (text.rstrip() if text and text.strip() else "[EMPTY]"))
        logger.info("=" * 100)

    def _log_semantic_chunks(
        self,
        func_idx: int,
        original_code: str,
        chunks: List[str],
        language: str,
        block_source: str = "entropy",
        block_metadata: Optional[List[Dict[str, Any]]] = None,
        block_dependency_counts: Optional[List[float]] = None,
    ) -> None:
        """打印函数原文和细粒度块划分结果。"""
        if language.lower() != "python":
            return

        original_tokens = self.get_token_length(original_code)
        spans = self._line_spans_for_chunks(chunks)
        logger.info("\n" + "=" * 100)
        logger.info(f"Func {func_idx} - 函数块原本样子：tokens={original_tokens}")
        logger.info("\n" + original_code.rstrip())
        logger.info("-" * 100)
        logger.info(f"Func {func_idx} - 细粒度块划分结果：source={block_source}, blocks={len(chunks)}")
        for idx, chunk in enumerate(chunks, 1):
            meta = block_metadata[idx - 1] if block_metadata and idx - 1 < len(block_metadata) else {}
            span = (
                int(meta.get("start_line", spans[idx - 1][0] if idx - 1 < len(spans) else -1) or -1),
                int(meta.get("end_line", spans[idx - 1][1] if idx - 1 < len(spans) else -1) or -1),
            )
            dep = 0.0
            if block_dependency_counts and idx - 1 < len(block_dependency_counts):
                dep = float(block_dependency_counts[idx - 1] or 0.0)
            token_count = self.get_token_length(chunk)
            block_type = meta.get("type") or self._block_type_label(chunk, language)
            logger.info(
                f"【Block {idx}】lines={span[0]}-{span[1]} tokens={token_count} "
                f"type={block_type} dep={dep:.4f}\n{chunk.rstrip()}"
            )
        logger.info("=" * 100)

    @staticmethod
    def _safe_minmax(values: List[float]) -> List[float]:
        clean = [0.0 if (v is None or math.isnan(float(v)) or math.isinf(float(v))) else float(v) for v in values]
        if not clean:
            return []
        min_v = min(clean)
        max_v = max(clean)
        if abs(max_v - min_v) < 1e-12:
            return [0.0 for _ in clean]
        return [(v - min_v) / (max_v - min_v) for v in clean]

    @staticmethod
    def _completion_identifiers(text: str) -> set:
        stop_words = {
            "and", "as", "assert", "async", "await", "break", "class", "continue",
            "def", "del", "elif", "else", "except", "false", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda", "none",
            "nonlocal", "not", "or", "pass", "raise", "return", "true", "try",
            "while", "with", "yield", "self", "cls", "str", "int", "len", "list",
            "dict", "set", "tuple", "object", "type", "range", "print",
        }
        tokens = {m.group(0).lower() for m in re.finditer(r"[A-Za-z_]\w*", text or "")}
        return {tok for tok in tokens if tok not in stop_words and len(tok) > 1}

    @staticmethod
    def _completion_attr_chains(text: str) -> set:
        return {m.group(0).lower() for m in re.finditer(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", text or "")}

    @staticmethod
    def _block_is_comment_only(block: str, language: str = "python") -> bool:
        comment_marker = "#" if language.lower() in ["python", "typescript", "rust"] else "//"
        useful = [line.strip() for line in (block or "").splitlines() if line.strip()]
        return bool(useful) and all(line.startswith(comment_marker) for line in useful)

    @staticmethod
    def _block_contains_docstring_marker(block: str) -> bool:
        text = block or ""
        return '"""' in text or "'''" in text or "For example::" in text or "Example:" in text

    def _block_type_label(self, block: str, language: str = "python") -> str:
        stripped_lines = [line.strip() for line in (block or "").splitlines() if line.strip()]
        if not stripped_lines:
            return "empty"
        first = stripped_lines[0]
        if self._block_is_comment_only(block, language):
            return "comment"
        if re.match(r"^(async\s+def|def)\b", first):
            return "signature"
        if re.match(r"^class\b", first):
            return "class"
        if self._is_control_or_definition_line(first, language):
            return "control"
        if any(re.match(r"^(return|yield)\b", line) for line in stripped_lines):
            return "return"
        if any(re.match(r"^raise\b", line) for line in stripped_lines):
            return "exception"
        if self._has_assignment_or_call_shape(block):
            return "assignment_call"
        if self._block_contains_docstring_marker(block):
            return "docstring"
        return "body"

    def _block_return_score(self, block: str) -> float:
        score = 0.0
        for line in (block or "").splitlines():
            stripped = line.strip()
            if re.match(r"^(return|yield)\b", stripped):
                score = max(score, 1.0)
            elif re.match(r"^raise\b", stripped):
                score = max(score, 0.7)
            elif re.match(r"^(break|continue)\b", stripped):
                score = max(score, 0.4)
        return score

    def _block_query_phrase_score(self, block: str, query: str) -> float:
        # Backward-compatible helper: pure unweighted symbol-match count.
        return float(len(self._block_overlap_symbols(block, query)))

    def _query_symbol_profile(self, query: str) -> Dict[str, set]:
        query = query or ""
        identifiers = self._completion_identifiers(query)
        attrs = self._completion_attr_chains(query)
        calls = {
            m.group(1).lower()
            for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", query)
            if m.group(1).lower() not in {"if", "for", "while", "return", "with"}
        }
        funcs = {
            m.group(1).lower()
            for m in re.finditer(r"\b(?:async\s+def|def)\s+([A-Za-z_]\w*)\s*\(", query)
        }
        classes = {
            m.group(1).lower()
            for m in re.finditer(r"\bclass\s+([A-Za-z_]\w*)\b", query)
        }
        strings = {
            m.group(2).lower()
            for m in re.finditer(r"([\"'])([^\"'\n]{2,80})\1", query)
            if not re.fullmatch(r"[-_=*/\\\s]+", m.group(2))
        }
        params = set()
        for m in re.finditer(r"\b(?:async\s+def|def)\s+[A-Za-z_]\w*\s*\(([^)]*)\)", query, re.S):
            for part in m.group(1).split(","):
                name = part.strip().split("=", 1)[0].strip().lstrip("*")
                if re.match(r"^[A-Za-z_]\w*$", name) and name not in {"self", "cls"}:
                    params.add(name.lower())
        return {
            "identifiers": identifiers,
            "attrs": attrs,
            "calls": calls,
            "functions": funcs,
            "classes": classes,
            "strings": strings,
            "params": params,
        }

    def _query_overlap_symbol_set(self, text: str) -> set:
        """Return one unweighted symbol set used by the overlap objective."""
        profile = self._query_symbol_profile(text or "")
        symbols = set()
        for key in ("identifiers", "attrs", "calls", "functions", "classes", "params", "strings"):
            symbols.update(str(s).lower() for s in profile.get(key, set()) if s)
        # Attribute tails make `self.foo` comparable with a nearby `foo` mention,
        # but each tail is still just one symbol, not a weighted feature.
        for attr in profile.get("attrs", set()):
            if "." in attr:
                symbols.add(attr.rsplit(".", 1)[-1])
        return symbols

    def _classify_completion_context(self, query: str) -> Dict[str, Any]:
        query = query or ""
        nonempty = [ln.strip() for ln in query.splitlines() if ln.strip()]
        first = nonempty[0] if nonempty else ""
        tail = "\n".join(nonempty[-8:]).lower()
        last = nonempty[-1] if nonempty else ""
        profile = self._query_symbol_profile(query)
        is_test = bool(re.search(r"\bdef\s+test_", query) or re.search(r"\b(self\.assert\w+|assertRaises|pytest\.raises|unittest|assert\s+)", query))
        is_class = bool(re.match(r"^class\b", first) or re.search(r"\bself\.[A-Za-z_]\w*\s*=", query))
        in_doc = self._query_is_inside_docstring(query) or bool(re.search(r"(^|\n)\s*(Args:|Returns:|Examples?:|Parameters\n|-{3,})", query))
        is_decorator = bool(first.startswith("@") or last.startswith("@"))
        is_api_schema_table = bool(
            re.search(r"\b(required|default|no_log|choices|argument_spec|module_args)\b", tail)
            and re.search(r"\b(type|str|int|bool|list|dict|root|host|password|username)\b", tail)
        )
        is_literal_table = bool(
            is_api_schema_table
            or re.search(r"([\"'])[A-Za-z_][\w\-]*\1\s*:", tail)
            or re.search(r"\b(fields\.|models\.|columns?\.|many2one|one2many|foreignkey|select\s+|join\s+|where\s+|group\s+by)\b", tail)
            or last.endswith(("{", "[", "(", ",", ":"))
            or re.match(r"^[}\]],?$", last)
        )
        recent_control = bool(re.search(r"\b(if|elif|else|for|while|try|except|finally|with)\b", tail))
        recent_loop = bool(re.search(r"\b(for|while)\b", tail))
        recent_call = bool(re.search(r"[A-Za-z_][\w\.]*\s*\([^()\n]*$", last) or re.search(r"\)\s*$", last))
        recent_assignment = bool(re.search(r"(?<![<>=!])=(?!=)", tail))
        if is_test:
            family = "test"
        elif in_doc:
            family = "doc_text"
        elif is_api_schema_table:
            family = "api_schema_table"
        elif is_literal_table:
            family = "literal_table"
        elif is_decorator:
            family = "decorator"
        elif is_class and not re.match(r"^(async\s+def|def)\b", first):
            family = "class_body"
        elif recent_control:
            family = "control_loop" if recent_loop else "control"
        elif recent_assignment or recent_call or profile.get("calls") or profile.get("attrs"):
            family = "assign_call"
        else:
            family = "general"
        return {"family": family, "is_test": is_test, "is_class": is_class, "in_doc": in_doc, "is_api_schema_table": is_api_schema_table, "is_literal_table": is_literal_table, "is_decorator": is_decorator, "recent_control": recent_control, "recent_loop": recent_loop, "recent_assignment": recent_assignment, "recent_call": recent_call, "profile": profile, "symbols": self._query_overlap_symbol_set(query)}

    def _block_overlap_symbols(self, block: str, query: str) -> set:
        query_symbols = self._query_overlap_symbol_set(query)
        if not query_symbols:
            return set()
        block_symbols = self._query_overlap_symbol_set(block)
        return query_symbols.intersection(block_symbols)

    def _block_symbol_overlap_score(self, block: str, query: str) -> float:
        # Unweighted count: every matched identifier/attribute/call/string key
        # contributes exactly one unit for clean ablations.
        return float(len(self._block_overlap_symbols(block, query)))

    def _block_signature_score(self, block: str, query: str, language: str = "python") -> float:
        block_type = self._block_type_label(block, language)
        if block_type not in {"signature", "class"}:
            return 0.0
        return 1.0 if self._block_overlap_symbols(block, query) else 0.0

    def _block_class_init_score(self, block: str, query: str) -> float:
        text = block or ""
        assigned_attrs = {
            m.group(1).lower()
            for m in re.finditer(r"\bself\.([A-Za-z_]\w*)\s*=", text)
        }
        if not assigned_attrs:
            return 0.0
        query_symbols = self._query_overlap_symbol_set(query)
        return 1.0 if assigned_attrs.intersection(query_symbols) or "__init__" in text else 0.0

    def _block_control_score(self, block: str, query: str, language: str = "python") -> float:
        for line in (block or "").splitlines():
            if self._is_control_or_definition_line(line.strip(), language=language):
                return 1.0 if self._block_overlap_symbols(block, query) else 0.0
        return 0.0

    @staticmethod
    def _comment_value_score(block: str, query: str = "") -> float:
        lines = [line.strip() for line in (block or "").splitlines() if line.strip()]
        comment_lines = [line for line in lines if line.startswith("#") or line.startswith('\"\"\"') or line.startswith("'''")]
        if not comment_lines:
            return 0.0
        text = " ".join(comment_lines).lower()
        if re.search(r"\b(license|copyright|permission|warranty|generated by|do not edit)\b", text):
            return 0.0
        if re.fullmatch(r"[#\s*=\-_/.*]+", text):
            return 0.0
        useful_terms = r"\b(return|returns|param|parameter|args|argument|todo|fixme|note|warning|unit|format|default|example|边界|返回|参数)\b"
        if re.search(useful_terms, text):
            return 1.0
        for ident in re.findall(r"[A-Za-z_]\w*", query or ""):
            if len(ident) >= 4 and ident.lower() in text:
                return 1.0
        return 0.0

    def _low_value_comment_penalty(self, block: str, language: str = "python") -> float:
        # Kept for compatibility with older code paths; ff8/ff9 do not use penalties.
        return 0.0

    @staticmethod
    def _defined_names_in_block(block: str) -> set:
        names = set()
        for line in (block or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for m in re.finditer(r"\b([A-Za-z_]\w*)\s*=", stripped):
                if not re.search(r"[=!<>]=", stripped[max(0, m.start() - 2):m.end() + 2]):
                    names.add(m.group(1).lower())
            for m in re.finditer(r"\bfor\s+([A-Za-z_]\w*)\s+in\b", stripped):
                names.add(m.group(1).lower())
            for m in re.finditer(r"\bas\s+([A-Za-z_]\w*)\b", stripped):
                names.add(m.group(1).lower())
        return names

    @staticmethod
    def _return_names_in_block(block: str) -> set:
        names = set()
        for line in (block or "").splitlines():
            stripped = line.strip()
            if re.match(r"^(return|yield|raise)\b", stripped):
                names.update(m.group(0).lower() for m in re.finditer(r"[A-Za-z_]\w*", stripped))
        return names - {"return", "yield", "raise", "none", "true", "false"}

    @staticmethod
    def _clean_score(value: Any) -> float:
        try:
            value = float(value)
        except Exception:
            return 0.0
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value

    def _block_preserve_priority(
        self,
        idx: int,
        block: str,
        query: str,
        dependency: float,
        language: str = "python",
    ) -> float:
        # Compatibility shim for old callers. ff8/ff9 do not rank hard blocks by
        # weighted priority; return the unweighted overlap count instead.
        return float(len(self._block_overlap_symbols(block, query)))

    @staticmethod
    def _query_is_inside_docstring(query: str) -> bool:
        query = query or ""
        return (query.count('"""') % 2 == 1) or (query.count("'''") % 2 == 1)

    def _semantic_chunks_are_usable(
        self,
        source: str,
        semantic_chunks: List[str],
        min_chunks: int = 2,
        min_token_coverage: float = 0.25,
    ) -> bool:
        nonempty = [chunk for chunk in (semantic_chunks or []) if chunk and chunk.strip()]
        if len(nonempty) < min_chunks:
            return False

        source_tokens = max(1, self.get_token_length(source))
        semantic_tokens = sum(self.get_token_length(chunk) for chunk in nonempty)
        if semantic_tokens <= 0:
            return False

        # split_code2 can attach only nearby useful comments/docstrings, but if the
        # remaining code is tiny, parsing probably failed and the entropy path
        # is safer for completion quality.
        if semantic_tokens / source_tokens < min_token_coverage:
            return False

        return True

    def _semantic_block_metadata(self, semantic_blocks: List[Any], chunks: List[str]) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        spans = self._line_spans_for_chunks(chunks)
        for idx, chunk in enumerate(chunks):
            block = semantic_blocks[idx] if semantic_blocks and idx < len(semantic_blocks) else None
            metadata.append(
                {
                    "source_indices": [idx],
                    "start_line": int(getattr(block, "start_line", spans[idx][0]) or spans[idx][0]),
                    "end_line": int(getattr(block, "end_line", spans[idx][1]) or spans[idx][1]),
                    "type": (
                        f"{getattr(block, 'kind', '')}:{getattr(block, 'ast_kind', '')}".strip(":")
                        if block is not None
                        else self._block_type_label(chunk)
                    ),
                    "comment_attached": bool(getattr(block, "split_code2_comment_attached", False)) if block is not None else False,
                    "comment_start_line": int(getattr(block, "split_code2_comment_start_line", 0) or 0) if block is not None else 0,
                    "split_code2_repairs": list(getattr(block, "split_code2_repairs", []) or []) if block is not None else [],
                }
            )
        return metadata

    def _postprocess_split_code2_blocks(
        self,
        func_idx: int,
        chunks: List[str],
        dependency_counts: List[float],
        semantic_blocks: List[Any],
        query: str,
        language: str = "python",
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Repair split_code2 fragments before scoring them as selectable units."""
        if not chunks:
            return chunks, dependency_counts, []

        deps = list(dependency_counts or [0.0] * len(chunks))
        if len(deps) < len(chunks):
            deps.extend([0.0] * (len(chunks) - len(deps)))
        elif len(deps) > len(chunks):
            deps = deps[:len(chunks)]

        metadata = self._semantic_block_metadata(semantic_blocks, chunks)
        for meta_idx, meta in enumerate(metadata):
            if meta.get("comment_attached"):
                logger.info(
                    f"[SPLIT_CODE2][POST] Func {func_idx}: attached useful comment to block {meta_idx} "
                    f"comment_start={meta.get('comment_start_line')} code_lines={meta.get('start_line')}-{meta.get('end_line')}"
                )
            for repair in meta.get("split_code2_repairs", []):
                logger.info(f"[SPLIT_CODE2][POST] Func {func_idx}: splitter repair block {meta_idx}: {repair}")

        merged_chunks: List[str] = []
        merged_deps: List[float] = []
        merged_meta: List[Dict[str, Any]] = []

        idx = 0
        while idx < len(chunks):
            chunk = chunks[idx]
            dep = deps[idx]
            meta = dict(metadata[idx])
            token_count = self.get_token_length(chunk)
            block_type = self._block_type_label(chunk, language)
            should_merge_next = False

            if idx + 1 < len(chunks):
                next_chunk = chunks[idx + 1]
                next_tokens = self.get_token_length(next_chunk)
                next_type = self._block_type_label(next_chunk, language)
                is_header = block_type in {"signature", "class", "control"}
                tiny_non_anchor = token_count <= 18 and self._block_query_overlap_score(chunk, query) == 0
                dangling_header = is_header and next_type not in {"signature", "class"}
                if (dangling_header and token_count + next_tokens <= 220) or (tiny_non_anchor and token_count + next_tokens <= 120):
                    should_merge_next = True

            if should_merge_next:
                next_chunk = chunks[idx + 1]
                next_meta = metadata[idx + 1]
                merged = chunk.rstrip() + "\n" + next_chunk.lstrip("\n")
                merged_dep = dep + deps[idx + 1]
                meta["source_indices"] = list(meta.get("source_indices", [])) + list(next_meta.get("source_indices", [idx + 1]))
                meta["end_line"] = max(int(meta.get("end_line", -1)), int(next_meta.get("end_line", -1)))
                meta["type"] = f"merged:{block_type}+{self._block_type_label(next_chunk, language)}"
                logger.info(
                    f"[SPLIT_CODE2][POST] Func {func_idx}: merge blocks {idx}->{idx + 1} "
                    f"type={meta['type']} tokens={token_count}+{self.get_token_length(next_chunk)}"
                )
                merged_chunks.append(merged)
                merged_deps.append(merged_dep)
                merged_meta.append(meta)
                idx += 2
                continue

            merged_chunks.append(chunk)
            merged_deps.append(dep)
            merged_meta.append(meta)
            idx += 1

        empty_fixed = 0
        filtered_chunks: List[str] = []
        filtered_deps: List[float] = []
        filtered_meta: List[Dict[str, Any]] = []
        for chunk, dep, meta in zip(merged_chunks, merged_deps, merged_meta):
            if not chunk or not chunk.strip():
                empty_fixed += 1
                logger.info(f"[SPLIT_CODE2][POST] Func {func_idx}: drop empty block meta={meta}")
                continue
            filtered_chunks.append(chunk)
            filtered_deps.append(dep)
            filtered_meta.append(meta)
        merged_chunks, merged_deps, merged_meta = filtered_chunks, filtered_deps, filtered_meta

        overlap_fixed = 0
        last_end = -1
        for meta in merged_meta:
            start = int(meta.get("start_line", -1) or -1)
            end = int(meta.get("end_line", -1) or -1)
            if start > 0 and last_end >= start:
                old_start = start
                meta["start_line"] = last_end + 1
                overlap_fixed += 1
                logger.info(
                    f"[SPLIT_CODE2][POST] Func {func_idx}: fix overlapping metadata "
                    f"start {old_start}->{meta['start_line']} end={end}"
                )
            last_end = max(last_end, end)

        # ff8/ff9 keep split_code2 block repair, but do not inject dependency
        # bonuses into neighboring blocks; dependency_count remains a raw count.
        anchor_indices = [
            idx
            for idx, chunk in enumerate(merged_chunks)
            if self._block_query_overlap_score(chunk, query) > 0
            or self._block_return_score(chunk) > 0
            or self._block_type_label(chunk, language) in {"signature", "control"}
        ]
        for anchor in anchor_indices:
            for nb in (anchor - 1, anchor + 1):
                if 0 <= nb < len(merged_deps):
                    logger.info(
                        f"[SPLIT_CODE2][POST] Func {func_idx}: neighbor observed without dep bonus "
                        f"anchor={anchor} neighbor={nb}"
                    )

        logger.info(
            f"[SPLIT_CODE2][POST] Func {func_idx}: blocks {len(chunks)} -> {len(merged_chunks)}, "
            f"empty_fixed={empty_fixed}, overlap_fixed={overlap_fixed}"
        )
        return merged_chunks, merged_deps, merged_meta

    def _block_query_overlap_score(self, block: str, query: str) -> float:
        # Same objective as symbol overlap: a pure unweighted match count.
        return float(len(self._block_overlap_symbols(block, query)))

    @staticmethod
    def _is_control_or_definition_line(line: str, language: str = "python") -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if language.lower() == "python":
            return re.match(
                r"^(async\s+def|def|class|if|elif|else|for|async\s+for|while|try|except|finally|with|async\s+with|match|case)\b",
                stripped,
            ) is not None
        return re.match(r"^(function|fn|func|class|if|else|for|while|switch|case|try|catch)\b", stripped) is not None

    @staticmethod
    def _has_completion_literal_pattern(block: str) -> bool:
        text = block or ""
        if "'" in text or '"' in text:
            return True
        return re.search(
            r"\b(select|from|join|where|on|group\s+by|order\s+by|insert|update|delete|request\.method|endswith|startswith)\b",
            text,
            re.IGNORECASE,
        ) is not None

    @staticmethod
    def _has_assignment_or_call_shape(block: str) -> bool:
        for line in (block or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.search(r"(?<![<>=!])=(?!=)", stripped):
                return True
            if re.search(r"\w+\s*\(", stripped):
                return True
        return False

    def _block_structural_score(self, block: str, language: str = "python") -> float:
        score = 0.0
        for line in (block or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_control_or_definition_line(stripped, language=language):
                score = max(score, 1.0)
            elif re.match(r"^(return|yield|raise|break|continue)\b", stripped):
                score = max(score, 0.8)
        return score

    def _select_preserved_completion_blocks(
        self,
        blocks: List[str],
        query: str = "",
        language: str = "python",
    ) -> set:
        """Minimal structural hard constraints; objectives stay in the knapsack.

        This is intentionally rule-based rather than weighted: keep the function
        entry skeleton, exact query-matched boundaries, and query-related returns.
        Everything else competes through ppl_change/dependency/overlap.
        """
        hard_preserved = set()
        soft_preserved = set()
        reasons: Dict[int, List[str]] = defaultdict(list)
        if not blocks:
            self._last_fine_preservation = {"hard": set(), "soft": set(), "reasons": {}, "query": query or ""}
            return hard_preserved

        hard_preserved.add(0)
        reasons[0].append("entry_block")
        query_attrs = {s for s in self._query_overlap_symbol_set(query) if s}
        ctx_info = self._classify_completion_context(query)
        ctx_family = ctx_info.get("family", "general")
        logger.info(f"[MO-KNAPSACK][QUERY-TYPE] family={ctx_family} details={ctx_info}")

        for block_idx, block in enumerate(blocks):
            block_type = self._block_type_label(block, language)
            overlap_symbols = self._block_overlap_symbols(block, query)
            has_overlap = bool(overlap_symbols)

            if block_type in {"signature", "class"}:
                hard_preserved.add(block_idx)
                reasons[block_idx].append("structural_boundary")
                if has_overlap:
                    reasons[block_idx].append("query_matched_boundary")

            if block_type == "return":
                if has_overlap:
                    hard_preserved.add(block_idx)
                    reasons[block_idx].append("query_related_return")
                else:
                    soft_preserved.add(block_idx)
                    reasons[block_idx].append("return_candidate")

            if block_type == "control":
                if has_overlap:
                    hard_preserved.add(block_idx)
                    reasons[block_idx].append("query_related_control")
                else:
                    soft_preserved.add(block_idx)
                    reasons[block_idx].append("control_candidate")

            assigned_attrs = {
                m.group(1).lower()
                for m in re.finditer(r"\bself\.([A-Za-z_]\w*)\s*=", block or "")
            }
            if assigned_attrs:
                if assigned_attrs.intersection(query_attrs) or "__init__" in (block or ""):
                    hard_preserved.add(block_idx)
                    reasons[block_idx].append("class_attr_init_constraint")
                else:
                    soft_preserved.add(block_idx)
                    reasons[block_idx].append("class_attr_init_candidate")

            if ctx_family == "assign_call" and has_overlap and self._has_assignment_or_call_shape(block):
                hard_preserved.add(block_idx)
                reasons[block_idx].append("type_assign_call_symbol_carrier")
            if ctx_family == "control" and block_type == "control":
                hard_preserved.add(block_idx)
                reasons[block_idx].append("type_control_boundary")
            if ctx_family == "test" and re.search(r"\b(self\.assert\w+|assertRaises|assertEqual|assertTrue|assertIn|pytest\.raises|posix\.|os\.)\b", block or ""):
                if has_overlap or block_idx in hard_preserved or self._block_control_score(block, query, language) > 0:
                    hard_preserved.add(block_idx)
                    reasons[block_idx].append("type_test_assert_api_query_linked")
                else:
                    soft_preserved.add(block_idx)
                    reasons[block_idx].append("type_test_assert_api_soft")
            if ctx_family == "class_body" and (assigned_attrs or re.match(r"^\s*[A-Z_][A-Z0-9_]*\s*=", block or "")):
                hard_preserved.add(block_idx)
                reasons[block_idx].append("type_class_attr_or_constant")

        neighbor_preserved = set()
        for block_idx in sorted(hard_preserved):
            for nb in (block_idx - 1, block_idx + 1):
                if 0 <= nb < len(blocks) and nb not in hard_preserved:
                    nb_tokens = self.get_token_length(blocks[nb])
                    nb_type = self._block_type_label(blocks[nb], language)
                    if nb_tokens <= 96 or nb_type in {"return", "control"}:
                        neighbor_preserved.add(nb)
                        reasons[nb].append(f"neighbor_of_hard_{block_idx}")
        hard_preserved.update(neighbor_preserved)

        self._last_fine_preservation = {
            "hard": set(hard_preserved),
            "soft": set(soft_preserved) - set(hard_preserved),
            "reasons": {idx: vals[:] for idx, vals in reasons.items()},
            "query": query or "",
        }
        logger.info(
            f"[MO-KNAPSACK][PRESERVE] hard={sorted(self._last_fine_preservation['hard'])}, "
            f"soft={sorted(self._last_fine_preservation['soft'])}"
        )
        for idx in sorted(set(hard_preserved) | set(soft_preserved)):
            logger.info(
                f"[MO-KNAPSACK][PRESERVE] block[{idx}] mode="
                f"{'hard' if idx in hard_preserved else 'soft'} reasons={';'.join(reasons.get(idx, []))}"
            )
        return hard_preserved

    def _line_spans_for_chunks(self, chunks: List[str]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        line_no = 1
        for chunk in chunks:
            line_count = max(1, len((chunk or "").splitlines()))
            spans.append((line_no, line_no + line_count - 1))
            line_no += line_count
        return spans

    def _project_semantic_dependencies_to_entropy_chunks(
        self,
        func_source: str,
        entropy_chunks: List[str],
        semantic_blocks: List[Any],
        semantic_dependency_counts: List[float],
    ) -> List[float]:
        if not entropy_chunks:
            return []
        dep_scores = [0.0 for _ in entropy_chunks]
        if not semantic_blocks or len(semantic_blocks) != len(semantic_dependency_counts):
            return dep_scores

        entropy_spans = self._line_spans_for_chunks(entropy_chunks)
        semantic_spans = []
        for block, dep in zip(semantic_blocks, semantic_dependency_counts):
            start_line = int(getattr(block, "start_line", 0) or 0)
            end_line = int(getattr(block, "end_line", start_line) or start_line)
            if start_line <= 0:
                continue
            semantic_spans.append((start_line, max(start_line, end_line), float(dep or 0.0)))

        for chunk_idx, (chunk_start, chunk_end) in enumerate(entropy_spans):
            total_dep = 0.0
            for sem_start, sem_end, dep in semantic_spans:
                if sem_end < chunk_start or sem_start > chunk_end:
                    continue
                overlap = min(chunk_end, sem_end) - max(chunk_start, sem_start) + 1
                if overlap > 0:
                    sem_len = max(1, sem_end - sem_start + 1)
                    total_dep += dep * (overlap / sem_len)
            dep_scores[chunk_idx] = total_dep
        return dep_scores

    def _build_hybrid_completion_block_scores(
        self,
        blocks: List[str],
        block_importances: List[float],
        block_dependency_counts: Optional[List[float]],
        query: str = "",
        language: str = "python",
    ) -> List[float]:
        """Log the three raw objectives and return a legacy placeholder list.

        ff8/ff9 no longer build a weighted scalar score. The selector consumes
        raw ppl_change, dependency_count, and unweighted query-symbol overlap.
        """
        deps = block_dependency_counts if block_dependency_counts is not None else [0.0] * len(blocks)
        if len(deps) != len(blocks):
            deps = [0.0] * len(blocks)

        clean_ppl: List[float] = []
        for value in block_importances:
            try:
                v = float(value)
            except Exception:
                v = 0.0
            if math.isnan(v) or math.isinf(v):
                v = 0.0
            clean_ppl.append(max(0.0, v))

        for idx, block in enumerate(blocks):
            symbols = sorted(self._block_overlap_symbols(block, query))
            logger.info(
                f"[MO-KNAPSACK][OBJECTIVE] block[{idx}] "
                f"type={self._block_type_label(block, language)} "
                f"tokens={self.get_token_length(block)} "
                f"ppl_change={clean_ppl[idx] if idx < len(clean_ppl) else 0.0:.6f} "
                f"dependency_count={float(deps[idx] or 0.0) if idx < len(deps) else 0.0:.6f} "
                f"overlap_count={len(symbols)} overlap_symbols={symbols}"
            )
        return clean_ppl

    def _expand_selected_neighbors(
        self,
        blocks: List[str],
        selected_indices: set,
        target_tokens: int,
        hybrid_scores: List[float],
        budget_slack: float = 1.0,
    ) -> set:
        return set(selected_indices)

    def _repair_selected_dependencies(
        self,
        blocks: List[str],
        selected_indices: set,
        target_tokens: int,
        hybrid_scores: List[float],
        query: str,
        language: str = "python",
        budget_slack: float = 1.0,
    ) -> Tuple[set, Dict[int, str]]:
        return set(selected_indices), {}

    def _trim_selected_to_budget(
        self,
        blocks: List[str],
        selected_indices: set,
        hard_indices: set,
        target_tokens: int,
        hybrid_scores: List[float],
        query: str,
        language: str = "python",
        budget_slack: float = 1.0,
    ) -> Tuple[set, Dict[int, str]]:
        return set(selected_indices), {}

    def _hybrid_knapsack_block_selection(
        self,
        blocks: List[str],
        block_importances: List[float],
        hybrid_scores: List[float],
        block_dependency_counts: Optional[List[float]],
        target_tokens: int,
        preserved_block_indices: set = None,
        language: str = "python",
    ) -> Tuple[set, Dict]:
        selector = "maximin"
        logger.info("[MO-KNAPSACK] ==================================================")
        logger.info(
            f"[MO-KNAPSACK] start selection: selector={selector}, "
            f"objectives=(ppl_change, dependency_count, overlap_count), "
            f"target_tokens={target_tokens}, blocks={len(blocks)}"
        )

        if preserved_block_indices is None:
            preserved_block_indices = set()
        if not blocks:
            return set(), {"method": f"pareto_multiobjective_knapsack_{selector}", "empty": True}

        preservation = getattr(self, "_last_fine_preservation", {})
        preserve_reasons: Dict[int, List[str]] = preservation.get("reasons", {}) or {}
        query = str(preservation.get("query", "") or "")
        query_symbols = self._query_overlap_symbol_set(query)

        weights = [int(max(1, self.get_token_length(block))) for block in blocks]
        deps = block_dependency_counts if block_dependency_counts is not None else [0.0] * len(blocks)
        if len(deps) != len(blocks):
            logger.warning("[MO-KNAPSACK] dependency count length mismatch, fallback to zeros")
            deps = [0.0] * len(blocks)

        clean_ppl: List[float] = []
        clean_dep: List[float] = []
        block_symbols: List[set] = []
        for idx, block in enumerate(blocks):
            try:
                ppl = float(block_importances[idx]) if idx < len(block_importances) else 0.0
            except Exception:
                ppl = 0.0
            if math.isnan(ppl) or math.isinf(ppl):
                ppl = 0.0
            try:
                dep = float(deps[idx] or 0.0)
            except Exception:
                dep = 0.0
            if math.isnan(dep) or math.isinf(dep):
                dep = 0.0
            clean_ppl.append(max(0.0, ppl))
            clean_dep.append(max(0.0, dep))
            block_symbols.append(set(self._block_overlap_symbols(block, query)))
            logger.info(
                f"[MO-KNAPSACK][BLOCK] block[{idx}] tokens={weights[idx]} "
                f"ppl_change={clean_ppl[-1]:.6f} dependency_count={clean_dep[-1]:.6f} "
                f"overlap_count={len(block_symbols[-1])} overlap_symbols={sorted(block_symbols[-1])} "
                f"preserved={'Y' if idx in preserved_block_indices else 'N'} "
                f"type={self._block_type_label(block, language)}"
            )

        preserved_block_indices = {i for i in preserved_block_indices if 0 <= i < len(blocks)}
        preserved_tokens = sum(weights[i] for i in preserved_block_indices)
        preserved_ppl = sum(clean_ppl[i] for i in preserved_block_indices)
        preserved_dep = sum(clean_dep[i] for i in preserved_block_indices)
        preserved_symbols = set()
        for i in preserved_block_indices:
            preserved_symbols.update(block_symbols[i])
        remaining_budget = max(0, int(target_tokens - preserved_tokens))

        logger.info(
            f"[MO-KNAPSACK] preserved_tokens={preserved_tokens}, remaining_budget={remaining_budget}, "
            f"preserved_ppl={preserved_ppl:.6f}, preserved_dep={preserved_dep:.6f}, "
            f"preserved_overlap={len(preserved_symbols)}/{len(query_symbols)}"
        )

        selectable = []
        for idx in range(len(blocks)):
            if idx in preserved_block_indices:
                continue
            if self._block_is_comment_only(blocks[idx], language) and not block_symbols[idx]:
                logger.info(f"[MO-KNAPSACK][CANDIDATE] skip block[{idx}] reason=comment_without_overlap")
                continue
            selectable.append(idx)

        def make_solution(extra_indices: set) -> Dict[str, Any]:
            extra = {i for i in extra_indices if i in selectable}
            full = set(preserved_block_indices).union(extra)
            symbols = set(preserved_symbols)
            for i in extra:
                symbols.update(block_symbols[i])
            return {
                "selected_indices": extra,
                "full_selected_indices": full,
                "weight": sum(weights[i] for i in full),
                "ppl_change": preserved_ppl + sum(clean_ppl[i] for i in extra),
                "dependency": preserved_dep + sum(clean_dep[i] for i in extra),
                "overlap": float(len(symbols)),
                "overlap_symbols": symbols,
            }

        def feasible(extra_indices: set) -> bool:
            return sum(weights[i] for i in extra_indices) <= remaining_budget

        def obj_tuple(sol: Dict[str, Any]) -> Tuple[float, float, float]:
            return (
                float(sol.get("ppl_change", 0.0)),
                float(sol.get("dependency", 0.0)),
                float(sol.get("overlap", 0.0)),
            )

        def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            av = obj_tuple(a)
            bv = obj_tuple(b)
            return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))

        def dedupe(solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            best_by_selection: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for sol in solutions:
                if sol["weight"] > max(target_tokens, preserved_tokens):
                    continue
                key = tuple(sorted(int(x) for x in sol.get("selected_indices", set())))
                cur = best_by_selection.get(key)
                if cur is None or (obj_tuple(sol), -int(sol["weight"])) > (obj_tuple(cur), -int(cur["weight"])):
                    best_by_selection[key] = sol
            return list(best_by_selection.values())

        def pareto_front(solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            unique = dedupe(solutions)
            front: List[Dict[str, Any]] = []
            for sol in unique:
                if any(dominates(other, sol) for other in unique if other is not sol):
                    continue
                front.append(sol)
            front.sort(key=lambda s: (obj_tuple(s), -int(s["weight"]), tuple(sorted(s["selected_indices"]))), reverse=True)
            return front

        def normalize_archive(solutions: List[Dict[str, Any]]) -> List[Tuple[float, float, float]]:
            matrix = [obj_tuple(sol) for sol in solutions]
            if not matrix:
                return []
            cols = list(zip(*matrix))
            mins = [min(col) for col in cols]
            maxs = [max(col) for col in cols]
            norm = []
            for row in matrix:
                vals = []
                for value, lo, hi in zip(row, mins, maxs):
                    vals.append(1.0 if abs(hi - lo) <= 1e-12 else (value - lo) / (hi - lo))
                norm.append(tuple(vals))
            return norm

        def select_maximin(candidates: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            norm = normalize_archive(candidates)
            best_idx = 0
            best_key = None
            for idx, (sol, row) in enumerate(zip(candidates, norm)):
                key = (
                    min(row),
                    float(sol.get("overlap", 0.0)),
                    float(sol.get("ppl_change", 0.0)),
                    float(sol.get("dependency", 0.0)),
                    -int(sol.get("weight", 0)),
                    -len(sol.get("selected_indices", set())),
                )
                if best_key is None or key > best_key:
                    best_idx = idx
                    best_key = key
            return candidates[best_idx], {
                "selector": "maximin_balance",
                "normalized_objectives": [list(row) for row in norm],
                "selected_index": best_idx,
                "balance_score": float(min(norm[best_idx])) if norm else 0.0,
            }

        def select_copeland(candidates: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            n = len(candidates)
            scores = [0 for _ in range(n)]
            pairwise = []
            eps = 1e-12
            for i in range(n):
                for j in range(i + 1, n):
                    ai = obj_tuple(candidates[i])
                    bj = obj_tuple(candidates[j])
                    i_better = sum(1 for a, b in zip(ai, bj) if a > b + eps)
                    j_better = sum(1 for a, b in zip(ai, bj) if b > a + eps)
                    if i_better > j_better:
                        scores[i] += 1
                        scores[j] -= 1
                        winner = i
                    elif j_better > i_better:
                        scores[j] += 1
                        scores[i] -= 1
                        winner = j
                    else:
                        winner = -1
                    pairwise.append({"a": i, "b": j, "winner": winner, "a_better_dims": i_better, "b_better_dims": j_better})
            norm = normalize_archive(candidates)
            best_idx = 0
            best_key = None
            for idx, sol in enumerate(candidates):
                row = norm[idx] if idx < len(norm) else (0.0, 0.0, 0.0)
                best_dims = sum(
                    1 for dim in range(3)
                    if all(obj_tuple(sol)[dim] >= obj_tuple(other)[dim] - eps for other in candidates)
                )
                key = (
                    scores[idx],
                    best_dims,
                    min(row),
                    float(sol.get("overlap", 0.0)),
                    float(sol.get("ppl_change", 0.0)),
                    float(sol.get("dependency", 0.0)),
                    -int(sol.get("weight", 0)),
                    -len(sol.get("selected_indices", set())),
                )
                if best_key is None or key > best_key:
                    best_idx = idx
                    best_key = key
            return candidates[best_idx], {
                "selector": "copeland_majority_then_maximin",
                "copeland_scores": scores,
                "pairwise": pairwise[:200],
                "normalized_objectives": [list(row) for row in norm],
                "selected_index": best_idx,
            }

        solutions: List[Dict[str, Any]] = [make_solution(set())]
        n_items = len(selectable)
        exact_mode = n_items <= 12 or (1 << min(n_items, 20)) <= 4096
        logger.info(f"[MO-KNAPSACK] selectable={n_items}, exact_mode={exact_mode}")

        if exact_mode:
            for mask in range(1 << n_items):
                chosen = {selectable[pos] for pos in range(n_items) if (mask >> pos) & 1}
                if feasible(chosen):
                    solutions.append(make_solution(chosen))
        else:
            def greedy(order: List[int]) -> set:
                chosen = set()
                used = 0
                covered = set()
                for idx in order:
                    w = weights[idx]
                    if used + w <= remaining_budget:
                        chosen.add(idx)
                        used += w
                        covered.update(block_symbols[idx])
                return chosen

            orders = []
            orders.append(sorted(selectable, key=lambda i: (clean_ppl[i], clean_dep[i], len(block_symbols[i]), -weights[i]), reverse=True))
            orders.append(sorted(selectable, key=lambda i: (clean_dep[i], len(block_symbols[i]), clean_ppl[i], -weights[i]), reverse=True))
            orders.append(sorted(selectable, key=lambda i: (len(block_symbols[i]), clean_ppl[i], clean_dep[i], -weights[i]), reverse=True))
            orders.append(sorted(selectable, key=lambda i: (clean_ppl[i] / max(weights[i], 1), clean_ppl[i], -weights[i]), reverse=True))
            orders.append(sorted(selectable, key=lambda i: (clean_dep[i] / max(weights[i], 1), clean_dep[i], -weights[i]), reverse=True))
            orders.append(sorted(selectable, key=lambda i: (len(block_symbols[i]) / max(weights[i], 1), len(block_symbols[i]), -weights[i]), reverse=True))
            for order in orders:
                sol_set = greedy(order)
                solutions.append(make_solution(sol_set))
                # One-step completion: if any remaining block fits and improves at least one objective, keep that variant too.
                for idx in order:
                    if idx not in sol_set and feasible(sol_set | {idx}):
                        solutions.append(make_solution(sol_set | {idx}))

            seed_material = "|".join([
                selector,
                str(target_tokens),
                str(len(blocks)),
                str(sorted(preserved_block_indices)),
                ",".join(f"{x:.6f}" for x in clean_ppl),
                ",".join(f"{x:.6f}" for x in clean_dep),
                ",".join(str(len(s)) for s in block_symbols),
                ",".join(str(x) for x in weights),
            ])
            seed = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(seed)
            attempts = max(120, min(600, n_items * 24))
            for _ in range(attempts):
                order = selectable[:]
                rng.shuffle(order)
                chosen = set()
                used = 0
                for idx in order:
                    if used + weights[idx] <= remaining_budget and rng.random() < 0.5:
                        chosen.add(idx)
                        used += weights[idx]
                if not chosen:
                    feasible_items = [idx for idx in selectable if weights[idx] <= remaining_budget]
                    if feasible_items:
                        chosen.add(rng.choice(feasible_items))
                solutions.append(make_solution(chosen))

        all_solutions = dedupe(solutions)
        pareto = pareto_front(all_solutions)
        if not pareto:
            pareto = [make_solution(set())]

        max_archive = 512
        if len(pareto) > max_archive:
            # Deterministic thinning without weighted scalarization: keep objective extrema,
            # then cycle through lexicographic objective orders.
            keep: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for dim_name in ("ppl_change", "dependency", "overlap"):
                sol = max(pareto, key=lambda s: (float(s[dim_name]), -int(s["weight"]), tuple(sorted(s["selected_indices"]))))
                keep[tuple(sorted(sol["selected_indices"]))] = sol
            orders = [
                ("ppl_change", "dependency", "overlap"),
                ("ppl_change", "overlap", "dependency"),
                ("dependency", "ppl_change", "overlap"),
                ("dependency", "overlap", "ppl_change"),
                ("overlap", "ppl_change", "dependency"),
                ("overlap", "dependency", "ppl_change"),
            ]
            cursor = 0
            while len(keep) < max_archive and cursor < len(pareto) * len(orders):
                order = orders[cursor % len(orders)]
                rank = cursor // len(orders)
                sorted_pool = sorted(
                    pareto,
                    key=lambda s: (float(s[order[0]]), float(s[order[1]]), float(s[order[2]]), -int(s["weight"])),
                    reverse=True,
                )
                if rank < len(sorted_pool):
                    sol = sorted_pool[rank]
                    keep[tuple(sorted(sol["selected_indices"]))] = sol
                cursor += 1
            pareto = list(keep.values())[:max_archive]

        if selector == "copeland":
            best, selector_info = select_copeland(pareto)
        else:
            best, selector_info = select_maximin(pareto)

        final_selection = set(best.get("full_selected_indices", set(preserved_block_indices).union(best.get("selected_indices", set()))))
        # Case451-like repair: when a large helper is represented by several
        # selected islands, small gaps between them often contain the exact
        # setup/control line the model needs. Fill affordable interior gaps as
        # intervals, still without scoring weights.
        gap_budget = max(0, int(target_tokens * 1.08) - sum(weights[i] for i in final_selection))
        ordered_selected = sorted(final_selection)
        for left, right in zip(ordered_selected, ordered_selected[1:]):
            if right <= left + 1:
                continue
            gap = list(range(left + 1, right))
            gap_tokens = sum(weights[i] for i in gap)
            if gap_tokens <= gap_budget and gap_tokens <= 220:
                final_selection.update(gap)
                gap_budget -= gap_tokens
                logger.info(
                    f"[MO-KNAPSACK][INTERVAL] fill gap {left + 1}-{right - 1} "
                    f"tokens={gap_tokens} remaining_gap_budget={gap_budget}"
                )
        total_weight = sum(weights[i] for i in final_selection)
        total_ppl = sum(clean_ppl[i] for i in final_selection)
        total_dep = sum(clean_dep[i] for i in final_selection)
        total_symbols = set()
        for i in final_selection:
            total_symbols.update(block_symbols[i])

        for cand_idx, sol in enumerate(pareto[: min(20, len(pareto))]):
            logger.info(
                f"[MO-KNAPSACK][CAND {cand_idx}] weight={sol['weight']} "
                f"ppl_change={sol['ppl_change']:.6f} dependency={sol['dependency']:.6f} "
                f"overlap={sol['overlap']:.0f}/{len(query_symbols)} selected={sorted(sol['full_selected_indices'])}"
            )

        for idx in range(len(blocks)):
            if idx in final_selection:
                if idx in preserved_block_indices:
                    reason = "hard_structural_constraint"
                elif idx in best.get("selected_indices", set()):
                    reason = f"pareto_{selector}"
                else:
                    reason = "selected"
                logger.info(
                    f"[MO-KNAPSACK][SELECT] block[{idx}] selected=Y reason={reason} "
                    f"tokens={weights[idx]} ppl_change={clean_ppl[idx]:.6f} "
                    f"dependency_count={clean_dep[idx]:.6f} overlap_count={len(block_symbols[idx])} "
                    f"type={self._block_type_label(blocks[idx], language)}"
                )
            else:
                logger.info(
                    f"[MO-KNAPSACK][SELECT] block[{idx}] selected=N reason=not_in_selected_pareto_solution "
                    f"tokens={weights[idx]} ppl_change={clean_ppl[idx]:.6f} "
                    f"dependency_count={clean_dep[idx]:.6f} overlap_count={len(block_symbols[idx])} "
                    f"type={self._block_type_label(blocks[idx], language)}"
                )

        info = {
            "method": f"pareto_multiobjective_knapsack_{selector}",
            "selector": selector,
            "objective_names": ["ppl_change", "dependency_count", "overlap_count"],
            "preserved_blocks": len(preserved_block_indices),
            "selected_blocks": len(best.get("selected_indices", set())),
            "total_blocks": len(final_selection),
            "total_ppl_change": total_ppl,
            "total_dependency_score": total_dep,
            "total_overlap_count": float(len(total_symbols)),
            "query_symbol_count": len(query_symbols),
            "total_weight": total_weight,
            "target_weight": target_tokens,
            "solution_count": len(all_solutions),
            "pareto_count": len(pareto),
            "selector_info": selector_info,
        }
        logger.info(
            f"[MO-KNAPSACK] final selection={len(final_selection)}/{len(blocks)} blocks, "
            f"ppl_change={total_ppl:.6f}, dependency_count={total_dep:.6f}, "
            f"overlap={len(total_symbols)}/{len(query_symbols)}, weight={total_weight}/{target_tokens}, "
            f"solutions={len(all_solutions)}, pareto={len(pareto)}"
        )
        logger.info("[MO-KNAPSACK] ==================================================")
        return final_selection, info


    @staticmethod
    def _is_docstring_expr(stmt: ast.AST) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    def _is_normal_python_function_chunk(self, chunk: str) -> bool:
        """
        判断一个 Python 语义块是否是函数块（兼容 Python2）
        """
        if not chunk or not chunk.strip():
            return False

        import textwrap
        import ast
        import re

        dedented = textwrap.dedent(chunk)
        lines = dedented.splitlines()

        i = 0
        n = len(lines)

        # 1️⃣ 跳过空行 + 注释
        while i < n:
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
            else:
                break

        # 2️⃣ 跳过 decorator
        while i < n:
            line = lines[i].strip()
            if line.startswith("@"):
                i += 1
            else:
                break

        if i >= n:
            return False

        # 3️⃣ 判断 def（核心）
        first_sig_line = lines[i].lstrip()

        if not re.match(r"^(async\s+def|def)\s+[A-Za-z_]\w*\s*\(", first_sig_line):
            return False

        # =========================
        # ⭐ 关键修改：AST 容错
        # =========================
        try:
            tree = ast.parse(dedented)
            body = [n for n in tree.body if not self._is_docstring_expr(n)]

            return len(body) == 1 and isinstance(
                body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
            )

        except SyntaxError:
            # 🚀 Python2 / 非标准代码 fallback
            # 只要 def 命中，就认为是函数块
            return True

    @staticmethod
    def _first_significant_python_line(chunk: str) -> str:
        dedented = textwrap.dedent(chunk).strip("\n")
        for line in dedented.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("@"):
                continue
            return s
        return ""

    @staticmethod
    def _leading_indent_width(text: str) -> int:
        for line in text.splitlines():
            if line.strip():
                m = re.match(r"^\s*", line)
                return len(m.group(0)) if m else 0
        return 0

    def _extract_python_class_signature(self, chunk: str) -> Dict[str, str]:
        """Extract class name/signature for a Python class chunk."""
        dedented = textwrap.dedent(chunk).strip("\n")
        signature_text = ""
        class_name = ""

        try:
            tree = ast.parse(dedented)
            class_node = None
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_node = node
                    break
            if class_node is not None:
                class_name = getattr(class_node, "name", "") or ""
                lines = dedented.splitlines()
                for line in lines:
                    s = line.strip()
                    if not s or s.startswith("#") or s.startswith("@"):
                        continue
                    if re.match(r"^class\s+[A-Za-z_]\w*", s):
                        signature_text = s
                        break
        except Exception:
            pass

        if not class_name or not signature_text:
            lines = dedented.splitlines()
            for line in lines:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("@"):
                    continue
                m = re.match(r"^class\s+([A-Za-z_]\w*)(.*)$", s)
                if m:
                    class_name = m.group(1)
                    signature_text = s
                    break

        return {
            "class_name": class_name,
            "signature_text": signature_text,
        }


    def _extract_python_function_signature(self, chunk: str) -> Dict[str, Any]:
        """Extract function name/signature for a Python function chunk."""
        dedented = textwrap.dedent(chunk).strip("\n")
        signature_text = ""
        function_name = ""
        param_names: List[str] = []
        positional_only_param_names: List[str] = []
        kwonly_param_names: List[str] = []
        has_vararg = False
        has_varkw = False
        vararg_name = ""
        varkw_name = ""
        defaults_count = 0
        required_positional_count = 0
        max_positional_count = 0
        normalized_signature = ""
        signature_key = ""

        try:
            tree = ast.parse(dedented)
            func_node = None
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_node = node
                    break
            if func_node is not None:
                function_name = getattr(func_node, "name", "") or ""
                lines = dedented.splitlines()
                for line in lines:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.startswith("@"):
                        continue
                    if re.match(r"^(async\s+def|def)\s+[A-Za-z_]\w*\s*\(", s):
                        signature_text = s
                        break

                args = getattr(func_node, "args", None)
                if args is not None:
                    positional_only_param_names = [a.arg for a in getattr(args, "posonlyargs", [])]
                    positional_param_names = [a.arg for a in getattr(args, "args", [])]
                    param_names = positional_only_param_names + positional_param_names
                    kwonly_param_names = [a.arg for a in getattr(args, "kwonlyargs", [])]
                    has_vararg = getattr(args, "vararg", None) is not None
                    has_varkw = getattr(args, "kwarg", None) is not None
                    vararg_name = getattr(getattr(args, "vararg", None), "arg", "") or ""
                    varkw_name = getattr(getattr(args, "kwarg", None), "arg", "") or ""
                    defaults_count = len(getattr(args, "defaults", []) or [])
                    max_positional_count = len(param_names)
                    required_positional_count = max(0, max_positional_count - defaults_count)

                    sig_parts: List[str] = []
                    for n in positional_only_param_names:
                        sig_parts.append(n)
                    if positional_only_param_names:
                        sig_parts.append("/")
                    for n in positional_param_names:
                        sig_parts.append(n)
                    if has_vararg:
                        sig_parts.append(f"*{vararg_name}" if vararg_name else "*")
                    elif kwonly_param_names:
                        sig_parts.append("*")
                    sig_parts.extend(kwonly_param_names)
                    if has_varkw:
                        sig_parts.append(f"**{varkw_name}" if varkw_name else "**")
                    normalized_signature = ", ".join(sig_parts)
                    signature_key = f"{function_name}({normalized_signature})" if function_name else normalized_signature
        except Exception:
            pass

        if not function_name or not signature_text:
            lines = dedented.splitlines()
            for line in lines:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("@"): 
                    continue
                m = re.match(r"^(async\s+def|def)\s+([A-Za-z_]\w*)\s*\((.*)$", s)
                if m:
                    function_name = m.group(2)
                    signature_text = s
                    break

        return {
            "function_name": function_name,
            "signature_text": signature_text,
            "param_names": param_names,
            "positional_only_param_names": positional_only_param_names,
            "kwonly_param_names": kwonly_param_names,
            "has_vararg": has_vararg,
            "has_varkw": has_varkw,
            "vararg_name": vararg_name,
            "varkw_name": varkw_name,
            "defaults_count": defaults_count,
            "required_positional_count": required_positional_count,
            "max_positional_count": max_positional_count,
            "normalized_signature": normalized_signature,
            "signature_key": signature_key,
        }

    def _extract_called_function_names(
        self,
        chunk: str,
        current_class_path: str = "",
        known_class_names: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Collect call records inside one Python function chunk."""
        dedented = textwrap.dedent(chunk).strip("\n")
        if not dedented.strip():
            return []

        try:
            tree = ast.parse(dedented)
        except SyntaxError:
            return []

        func_node = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_node = node
                break
        if func_node is None:
            return []

        known_class_names = set(known_class_names or [])

        class _CallVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.calls: List[Dict[str, Any]] = []

            def _source_text(self, node: ast.AST) -> str:
                try:
                    seg = ast.get_source_segment(dedented, node)
                    if seg:
                        return seg.strip()
                except Exception:
                    pass
                try:
                    return ast.unparse(node).strip()  # type: ignore[attr-defined]
                except Exception:
                    return ""

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
                return

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
                return

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
                return

            def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
                func = node.func
                call_name = ""
                base_name = ""
                base_kind = "other"
                qualified_hints: List[str] = []

                positional_arg_count = 0
                has_starargs = False
                for arg in getattr(node, "args", []):
                    if isinstance(arg, ast.Starred):
                        has_starargs = True
                    else:
                        positional_arg_count += 1

                keyword_names = [kw.arg for kw in getattr(node, "keywords", []) if kw.arg is not None]
                has_kwargs = any(kw.arg is None for kw in getattr(node, "keywords", []))

                if isinstance(func, ast.Name):
                    call_name = func.id
                    base_name = func.id
                    base_kind = "bare"
                    qualified_hints.append(func.id)
                elif isinstance(func, ast.Attribute):
                    call_name = func.attr
                    base = func.value
                    if isinstance(base, ast.Name):
                        base_name = base.id
                        if current_class_path and base.id in {"self", "cls"}:
                            base_kind = base.id
                            qualified_hints.append(f"{current_class_path}.{call_name}")
                            qualified_hints.append(call_name)
                        elif base.id in known_class_names:
                            base_kind = "class_name"
                            qualified_hints.append(f"{base.id}.{call_name}")
                            qualified_hints.append(call_name)
                        else:
                            base_kind = "attr"
                            qualified_hints.append(call_name)
                    elif isinstance(base, ast.Call):
                        if isinstance(base.func, ast.Name) and base.func.id == "super":
                            base_kind = "super"
                            if current_class_path:
                                qualified_hints.append(f"{current_class_path}.{call_name}")
                        else:
                            qualified_hints.append(call_name)
                    else:
                        qualified_hints.append(call_name)
                else:
                    try:
                        call_name = self._source_text(node)
                    except Exception:
                        call_name = ""

                if not call_name:
                    self.generic_visit(node)
                    return

                record = {
                    "call_name": call_name,
                    "base_name": base_name,
                    "base_kind": base_kind,
                    "qualified_hints": list(dict.fromkeys([h for h in qualified_hints if h])),
                    "current_class_path": current_class_path,
                    "positional_arg_count": positional_arg_count,
                    "keyword_names": keyword_names,
                    "has_starargs": has_starargs,
                    "has_kwargs": has_kwargs,
                    "call_text": self._source_text(node) or call_name,
                }
                self.calls.append(record)
                self.generic_visit(node)

        visitor = _CallVisitor()
        for stmt in getattr(func_node, "body", []):
            visitor.visit(stmt)
        return visitor.calls

    def _build_function_call_stats(self, code_chunks: List[str], language: str = "python") -> List[Dict[str, Any]]:
        """Build per-chunk function signature and call statistics."""
        chunk_stats: List[Dict[str, Any]] = []
        qualified_name_to_indices: Dict[str, List[int]] = defaultdict(list)
        bare_name_to_indices: Dict[str, List[int]] = defaultdict(list)
        class_function_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        signature_to_indices: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        known_class_names: set = set()
        class_stack: List[Dict[str, Any]] = []

        def _class_path_to_str(class_path: Union[str, List[str], Tuple[str, ...]]) -> str:
            if isinstance(class_path, str):
                return class_path
            return ".".join([p for p in class_path if p])

        def _effective_signature_info(info: Dict[str, Any]) -> Dict[str, Any]:
            param_names = list(info.get("param_names", []) or [])
            kwonly_param_names = list(info.get("kwonly_param_names", []) or [])
            has_vararg = bool(info.get("has_vararg", False))
            has_varkw = bool(info.get("has_varkw", False))
            defaults_count = int(info.get("defaults_count", 0) or 0)
            required_positional_count = int(info.get("required_positional_count", 0) or 0)
            max_positional_count = int(info.get("max_positional_count", 0) or 0)
            class_path_str = _class_path_to_str(info.get("class_path", []))

            if class_path_str and param_names and param_names[0] in {"self", "cls"}:
                param_names = param_names[1:]
                if required_positional_count > 0:
                    required_positional_count -= 1
                if max_positional_count > 0:
                    max_positional_count -= 1
                if defaults_count > 0:
                    defaults_count = max(0, defaults_count - 1)

            keyword_param_names = list(param_names) + kwonly_param_names
            return {
                "param_names": param_names,
                "keyword_param_names": keyword_param_names,
                "kwonly_param_names": kwonly_param_names,
                "has_vararg": has_vararg,
                "has_varkw": has_varkw,
                "required_positional_count": required_positional_count,
                "max_positional_count": max_positional_count,
                "defaults_count": defaults_count,
                "class_path_str": class_path_str,
            }

        def _display_name(idx: int) -> str:
            info = chunk_stats[idx]
            class_path_str = _class_path_to_str(info.get("class_path", []))
            qual = info.get("qualified_name") or ""
            signature = info.get("signature_text") or ""
            if qual:
                if signature and signature != qual:
                    return f"{qual} | {signature}"
                return qual
            return signature or info.get("function_name") or (code_chunks[idx].splitlines()[0].strip() if code_chunks[idx].splitlines() else f"chunk_{idx}")

        def _signature_compatibility(call_rec: Dict[str, Any], candidate_info: Dict[str, Any]) -> Tuple[int, str]:
            sig = _effective_signature_info(candidate_info)
            if not sig["param_names"] and not sig["has_vararg"] and not sig["has_varkw"]:
                return 0, "no_signature"

            positional_count = int(call_rec.get("positional_arg_count", 0) or 0)
            keyword_names = [k for k in (call_rec.get("keyword_names", []) or []) if k]
            has_starargs = bool(call_rec.get("has_starargs", False))
            has_kwargs = bool(call_rec.get("has_kwargs", False))

            score = 0
            if has_starargs:
                score += 1
            if has_kwargs:
                score += 1

            if not has_starargs:
                if positional_count < sig["required_positional_count"]:
                    return -10**9, "too_few_positional"
                if (not sig["has_vararg"]) and positional_count > sig["max_positional_count"]:
                    return -10**9, "too_many_positional"
                expected = sig["max_positional_count"]
                score += max(0, 12 - abs(positional_count - expected) * 3)
            else:
                score += 2

            if keyword_names:
                candidate_kw_params = set(sig["keyword_param_names"])
                if not sig["has_varkw"]:
                    missing = [k for k in keyword_names if k not in candidate_kw_params]
                    if missing:
                        return -10**9, f"keyword_mismatch:{missing}"
                    score += 10 - len(keyword_names)
                else:
                    score += 5
            else:
                score += 2

            return score, "ok"

        def _resolve_call_target_index(call_rec: Dict[str, Any]) -> Optional[int]:
            call_name = call_rec.get("call_name", "") or ""
            if not call_name:
                return None

            current_class_path = call_rec.get("current_class_path", "") or ""
            base_kind = call_rec.get("base_kind", "") or ""
            qualified_hints = [h for h in call_rec.get("qualified_hints", []) or [] if h]

            candidates: List[int] = []
            seen: set = set()

            def _add_candidate(idx: int) -> None:
                if idx not in seen:
                    candidates.append(idx)
                    seen.add(idx)

            for hint in qualified_hints:
                if hint in qualified_name_to_indices:
                    for idx in qualified_name_to_indices[hint]:
                        _add_candidate(idx)

            # 1) current class first for self/cls calls and bare calls inside a class
            if current_class_path and base_kind in {"self", "cls", "super", "bare"}:
                key = (current_class_path, call_name)
                for idx in class_function_to_indices.get(key, []):
                    _add_candidate(idx)

            # 2) explicit class base like ClassName.method()
            if base_kind == "class_name":
                for idx in class_function_to_indices.get((call_rec.get("base_name", ""), call_name), []):
                    _add_candidate(idx)

            # 3) if the call is already fully qualified and matched, keep those only
            if candidates:
                return _best_match_for_call(call_rec, candidates)

            # 4) exact bare unique function
            if call_name in bare_name_to_indices and len(bare_name_to_indices[call_name]) == 1:
                return bare_name_to_indices[call_name][0]

            # 5) if current class has same-named methods, try them even if not qualified yet
            if current_class_path:
                local_candidates = class_function_to_indices.get((current_class_path, call_name), [])
                if local_candidates:
                    return _best_match_for_call(call_rec, local_candidates)

            # 6) top-level unique bare function or any candidate with best signature match
            if call_name in bare_name_to_indices:
                return _best_match_for_call(call_rec, bare_name_to_indices[call_name])

            # 7) fallback: try matching by suffix in qualified names, but only if unambiguous by signature
            suffix_matches = []
            for qname, idxs in qualified_name_to_indices.items():
                if qname.endswith(f".{call_name}"):
                    suffix_matches.extend(idxs)
            if suffix_matches:
                return _best_match_for_call(call_rec, suffix_matches)

            return None

        def _best_match_for_call(call_rec: Dict[str, Any], candidate_indices: List[int]) -> Optional[int]:
            if not candidate_indices:
                return None
            ranked: List[Tuple[int, int, int, int, int]] = []
            current_class_path = call_rec.get("current_class_path", "") or ""
            call_name = call_rec.get("call_name", "") or ""
            for idx in candidate_indices:
                info = chunk_stats[idx]
                score, _reason = _signature_compatibility(call_rec, info)
                if score <= -10**8:
                    continue
                class_path_str = _class_path_to_str(info.get("class_path", []))
                qual = info.get("qualified_name", "") or ""
                signature_key = info.get("signature_key", "") or ""
                base_score = score
                if qual and qual in call_rec.get("qualified_hints", []):
                    base_score += 120
                if current_class_path and class_path_str == current_class_path:
                    base_score += 100
                if call_rec.get("base_kind", "") == "class_name" and class_path_str == call_rec.get("base_name", ""):
                    base_score += 80
                if call_rec.get("base_kind", "") in {"self", "cls", "super"} and class_path_str == current_class_path:
                    base_score += 60
                if info.get("function_name", "") == call_name:
                    base_score += 20
                if info.get("signature_key", ""):
                    base_score += 1
                # Prefer earlier code order only as final tiebreaker.
                ranked.append((base_score, len(info.get("class_path", []) or []), len(info.get("param_names", []) or []), -int(info.get("index", idx)), idx))
            if not ranked:
                return None
            ranked.sort(reverse=True)
            return ranked[0][-1]

        for idx, chunk in enumerate(code_chunks):
            indent_level = self._leading_indent_width(chunk)
            first_sig = self._first_significant_python_line(chunk)
            info = {
                "index": idx,
                "chunk": chunk,
                "is_function": False,
                "is_class": False,
                "chunk_kind": "other",
                "class_name": "",
                "class_path": [],
                "qualified_name": "",
                "function_name": "",
                "signature_text": "",
                "signature_key": "",
                "param_names": [],
                "positional_only_param_names": [],
                "kwonly_param_names": [],
                "has_vararg": False,
                "has_varkw": False,
                "vararg_name": "",
                "varkw_name": "",
                "defaults_count": 0,
                "required_positional_count": 0,
                "max_positional_count": 0,
                "normalized_signature": "",
                "call_count": 0,
                "called_by_count": 0,
                "calls": set(),
                "called_by": set(),
                "calls_text": [],
                "called_by_text": [],
                "ppl": 0.0,
                "tokens": 0,
                "indent_level": indent_level,
            }

            if language.lower() == "python":
                is_class_chunk = bool(re.match(r"^class\s+[A-Za-z_]\w*", first_sig))
                is_function_chunk = self._is_normal_python_function_chunk(chunk)

                if is_class_chunk:
                    class_sig = self._extract_python_class_signature(chunk)
                    class_name = class_sig.get("class_name", "") or ""
                    while class_stack and indent_level <= class_stack[-1]["indent_level"]:
                        class_stack.pop()
                    parent_class_path = list(class_stack[-1]["class_path"]) if class_stack else []
                    class_path = parent_class_path + ([class_name] if class_name else [])
                    class_path_str = _class_path_to_str(class_path)
                    info.update({
                        "is_class": True,
                        "chunk_kind": "class",
                        "class_name": class_name,
                        "class_path": class_path,
                        "qualified_name": class_path_str or class_name,
                        "signature_text": class_sig.get("signature_text", ""),
                    })
                    if class_name:
                        known_class_names.add(class_name)
                    if class_name:
                        class_stack.append({"class_name": class_name, "class_path": class_path, "indent_level": indent_level})
                    chunk_stats.append(info)
                    continue

                if is_function_chunk:
                    sig = self._extract_python_function_signature(chunk)
                    function_name = sig.get("function_name", "") or ""
                    current_class_path: List[str] = []
                    for frame in reversed(class_stack):
                        if indent_level > frame["indent_level"]:
                            current_class_path = list(frame["class_path"])
                            break
                    current_class_path_str = _class_path_to_str(current_class_path)
                    qualified_name = f"{current_class_path_str}.{function_name}" if current_class_path_str and function_name else function_name
                    info.update({
                        "is_function": True,
                        "chunk_kind": "function",
                        "class_name": current_class_path_str,
                        "class_path": current_class_path,
                        "qualified_name": qualified_name,
                        "function_name": function_name,
                        "signature_text": sig.get("signature_text", ""),
                        "signature_key": sig.get("signature_key", ""),
                        "param_names": sig.get("param_names", []),
                        "positional_only_param_names": sig.get("positional_only_param_names", []),
                        "kwonly_param_names": sig.get("kwonly_param_names", []),
                        "has_vararg": sig.get("has_vararg", False),
                        "has_varkw": sig.get("has_varkw", False),
                        "vararg_name": sig.get("vararg_name", ""),
                        "varkw_name": sig.get("varkw_name", ""),
                        "defaults_count": sig.get("defaults_count", 0),
                        "required_positional_count": sig.get("required_positional_count", 0),
                        "max_positional_count": sig.get("max_positional_count", 0),
                        "normalized_signature": sig.get("normalized_signature", ""),
                    })
                    if qualified_name:
                        qualified_name_to_indices[qualified_name].append(idx)
                    if function_name:
                        bare_name_to_indices[function_name].append(idx)
                    if current_class_path_str and function_name:
                        class_function_to_indices[(current_class_path_str, function_name)].append(idx)
                    if current_class_path_str and function_name and sig.get("signature_key", ""):
                        signature_to_indices[(current_class_path_str, function_name, sig.get("signature_key", ""))].append(idx)
                    chunk_stats.append(info)
                    continue

            chunk_stats.append(info)

        ambiguous_bare_names = {name: idxs for name, idxs in bare_name_to_indices.items() if len(idxs) > 1}
        if ambiguous_bare_names:
            logger.debug(
                f"[CALL-GRAPH] ambiguous bare function names detected; resolving by class path / signature when possible: {ambiguous_bare_names}"
            )

        for caller_idx, info in enumerate(chunk_stats):
            if not info["is_function"]:
                continue
            chunk = info["chunk"]
            current_class_path_str = _class_path_to_str(info.get("class_path", []))
            call_records = (
                self._extract_called_function_names(
                    chunk,
                    current_class_path=current_class_path_str,
                    known_class_names=known_class_names,
                )
                if language.lower() == "python"
                else []
            )
            resolved_call_targets: List[int] = []
            for call_rec in call_records:
                callee_idx = _resolve_call_target_index(call_rec)
                if callee_idx is None:
                    continue
                resolved_call_targets.append(callee_idx)
                info["call_count"] += 1
                info["calls"].add(callee_idx)
                if _display_name(callee_idx) not in info["calls_text"]:
                    info["calls_text"].append(_display_name(callee_idx))
                if callee_idx == caller_idx:
                    continue
                chunk_stats[callee_idx]["called_by_count"] += 1
                chunk_stats[callee_idx]["called_by"].add(caller_idx)
                caller_name = _display_name(caller_idx)
                if caller_name not in chunk_stats[callee_idx]["called_by_text"]:
                    chunk_stats[callee_idx]["called_by_text"].append(caller_name)

            # keep stable order for debug logs
            info["calls_text"] = list(dict.fromkeys(info["calls_text"]))
            info["called_by_text"] = list(dict.fromkeys(info["called_by_text"]))

        for info in chunk_stats:
            info["calls"] = sorted(info["calls"])
            info["called_by"] = sorted(info["called_by"])
            info["calls_text"] = sorted(set(info["calls_text"]))
            info["called_by_text"] = sorted(set(info["called_by_text"]))
            info["tokens"] = self.get_token_length(info["chunk"])
            if info["is_function"]:
                try:
                    info["ppl"] = self._compute_chunk_ppl(info["chunk"])
                except Exception as e:
                    logger.warning(f"[CALL-GRAPH] failed to compute ppl for chunk {info['index']}: {e}")
                    info["ppl"] = 0.0
            else:
                info["ppl"] = 0.0

        return chunk_stats

    def _compute_chunk_ppl(self, chunk: str) -> float:
        """Compute a light-weight PPL score for one code chunk."""
        if not chunk or not chunk.strip():
            return 0.0

        cache = self.cache.setdefault("chunk_ppl", {})
        cache_key = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:16]
        if cache_key in cache:
            return float(cache[cache_key])

        try:
            ppl = float(self.entropy_chunking._compute_ppl("", chunk))
        except Exception:
            try:
                ppl = float(len(chunk.splitlines()))
            except Exception:
                ppl = 0.0

        cache[cache_key] = ppl
        self._manage_cache_size("chunk_ppl")
        return ppl

    @staticmethod
    def _coarse_solution_key(sol: Dict[str, Any]) -> Tuple[int, ...]:
        return tuple(sorted(int(x) for x in sol.get("selected_indices", [])))

    @staticmethod
    def _coarse_solution_rank_key(sol: Dict[str, Any]) -> Tuple[float, float, float, int, Tuple[int, ...]]:
        return (
            float(sol.get("ppl", 0.0)) + float(sol.get("calls", 0.0)),
            float(sol.get("ppl", 0.0)),
            float(sol.get("calls", 0.0)),
            -int(sol.get("weight", 0)),
            tuple(sorted(int(x) for x in sol.get("selected_indices", []))),
        )

    def _coarse_pareto_prune_solutions(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not solutions:
            return []

        unique_by_selection: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for sol in solutions:
            key = self._coarse_solution_key(sol)
            cur = unique_by_selection.get(key)
            if cur is None or self._coarse_solution_rank_key(sol) > self._coarse_solution_rank_key(cur):
                unique_by_selection[key] = sol

        unique_solutions = list(unique_by_selection.values())
        pruned: List[Dict[str, Any]] = []
        for sol in unique_solutions:
            dominated = False
            remove_idxs: List[int] = []
            for i, kept in enumerate(pruned):
                kept_better = kept["ppl"] >= sol["ppl"] and kept["calls"] >= sol["calls"]
                kept_strict = kept["ppl"] > sol["ppl"] or kept["calls"] > sol["calls"]
                if kept_better and kept_strict:
                    dominated = True
                    break
                sol_better = sol["ppl"] >= kept["ppl"] and sol["calls"] >= kept["calls"]
                sol_strict = sol["ppl"] > kept["ppl"] or sol["calls"] > kept["calls"]
                if sol_better and sol_strict:
                    remove_idxs.append(i)
            if dominated:
                continue
            for idx in reversed(remove_idxs):
                pruned.pop(idx)
            pruned.append(sol)

        pruned.sort(key=lambda s: self._coarse_solution_rank_key(s), reverse=True)
        return pruned

    def _coarse_minmax_normalize(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
        return (matrix - mins) / denom

    def _coarse_build_diverse_archive(
        self,
        solutions: List[Dict[str, Any]],
        max_archive: int = 512,
        epsilon: float = 0.05,
        preserve_pareto: bool = True,
    ) -> List[Dict[str, Any]]:
        if not solutions:
            return []

        unique_by_selection: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for sol in solutions:
            key = self._coarse_solution_key(sol)
            cur = unique_by_selection.get(key)
            if cur is None or self._coarse_solution_rank_key(sol) > self._coarse_solution_rank_key(cur):
                unique_by_selection[key] = sol

        unique = list(unique_by_selection.values())
        if not unique:
            return []

        if preserve_pareto:
            pareto = self._coarse_pareto_prune_solutions(unique)
            if len(pareto) >= 4:
                unique = pareto
            else:
                logger.info(f"[NSGA-COARSE] Pareto front too small ({len(pareto)}), keeping unique feasible pool ({len(unique)})")

        if len(unique) <= max_archive:
            return sorted(unique, key=self._coarse_solution_rank_key, reverse=True)

        matrix = np.array([[float(sol["ppl"]), float(sol["calls"])] for sol in unique], dtype=float)
        norm = self._coarse_minmax_normalize(matrix)
        eps = max(float(epsilon), 1e-6)
        buckets: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for sol, vec in zip(unique, norm):
            key = (int(min(vec[0] / eps, 1.0 / eps)), int(min(vec[1] / eps, 1.0 / eps)))
            cur = buckets.get(key)
            if cur is None or self._coarse_solution_rank_key(sol) > self._coarse_solution_rank_key(cur):
                buckets[key] = sol

        archive = list(buckets.values())
        if len(archive) > max_archive:
            archive.sort(key=self._coarse_solution_rank_key, reverse=True)
            archive = archive[:max_archive]

        if len(archive) < min(8, len(unique)):
            unique.sort(key=self._coarse_solution_rank_key, reverse=True)
            for sol in unique:
                if sol not in archive:
                    archive.append(sol)
                if len(archive) >= min(max_archive, len(unique)):
                    break

        archive.sort(key=self._coarse_solution_rank_key, reverse=True)
        return archive

    def _coarse_topsis_select_solution(self, solutions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not solutions:
            raise ValueError("No feasible solutions for TOPSIS.")

        deduped: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for sol in solutions:
            key = self._coarse_solution_key(sol)
            cur = deduped.get(key)
            if cur is None:
                deduped[key] = sol
            else:
                cur_score = (float(cur["ppl"]) + float(cur["calls"]), float(cur["ppl"]), float(cur["calls"]), -int(cur["weight"]))
                new_score = (float(sol["ppl"]) + float(sol["calls"]), float(sol["ppl"]), float(sol["calls"]), -int(sol["weight"]))
                if new_score > cur_score:
                    deduped[key] = sol

        solutions = list(deduped.values())
        if not solutions:
            raise ValueError("No feasible solutions for TOPSIS after deduplication.")

        logger.info(f"[TOPSIS] candidate_count={len(solutions)}")
        for i, sol in enumerate(solutions[: min(10, len(solutions))]):
            logger.info(
                f"[TOPSIS][cand {i}] weight={sol['weight']}, ppl={sol['ppl']:.6f}, calls={sol['calls']:.6f}, "
                f"selected={sorted(sol['selected_indices'])}"
            )

        matrix = np.array([[float(sol["ppl"]), float(sol["calls"])] for sol in solutions], dtype=float)
        logger.info("[TOPSIS] decision matrix (PPL, call_score):")
        for i, row in enumerate(matrix.tolist()):
            logger.info(f"  cand {i}: PPL={row[0]:.6f}, calls={row[1]:.6f}")

        if matrix.shape[0] == 1 or np.allclose(matrix, matrix[0:1, :], atol=1e-12, rtol=0.0):
            logger.warning("[TOPSIS] all candidates are identical in objective space; using tie-breaker")
            best_idx = max(
                range(len(solutions)),
                key=lambda i: (
                    float(solutions[i]["ppl"]) + float(solutions[i]["calls"]),
                    float(solutions[i]["ppl"]),
                    float(solutions[i]["calls"]),
                    -int(solutions[i]["weight"]),
                    -len(solutions[i]["selected_indices"]),
                ),
            )
            best = solutions[best_idx]
            norm = np.zeros_like(matrix, dtype=float)
            positive_ideal = np.array([1.0, 1.0], dtype=float)
            negative_ideal = np.array([0.0, 0.0], dtype=float)
            d_pos = np.linalg.norm(norm - positive_ideal, axis=1)
            d_neg = np.linalg.norm(norm - negative_ideal, axis=1)
            closeness = np.where((d_pos + d_neg) > 0, d_neg / (d_pos + d_neg + 1e-12), 0.5)
        else:
            norm = self._coarse_minmax_normalize(matrix)
            positive_ideal = np.array([1.0, 1.0], dtype=float)
            negative_ideal = np.array([0.0, 0.0], dtype=float)
            d_pos = np.linalg.norm(norm - positive_ideal, axis=1)
            d_neg = np.linalg.norm(norm - negative_ideal, axis=1)
            closeness = d_neg / (d_pos + d_neg + 1e-12)
            best_idx = int(np.argmax(closeness))
            best = solutions[best_idx]

        logger.info("[TOPSIS] normalized matrix:")
        for i, row in enumerate(norm.tolist()):
            logger.info(f"  cand {i}: n_PPL={row[0]:.6f}, n_calls={row[1]:.6f}")
        logger.info(f"[TOPSIS] positive ideal={positive_ideal.tolist()}")
        logger.info(f"[TOPSIS] negative ideal={negative_ideal.tolist()}")

        for i in range(len(solutions)):
            logger.info(
                f"[TOPSIS][cand {i}] d+={float(d_pos[i]):.6f}, d-={float(d_neg[i]):.6f}, closeness={float(closeness[i]):.6f}"
            )

        info = {
            "normalized_matrix": norm.tolist(),
            "positive_ideal": positive_ideal.tolist(),
            "negative_ideal": negative_ideal.tolist(),
            "distance_to_ideal": float(d_pos[best_idx]),
            "distance_to_negative_ideal": float(d_neg[best_idx]),
            "closeness": float(closeness[best_idx]),
            "all_distances_to_ideal": [float(x) for x in d_pos.tolist()],
            "all_distances_to_negative_ideal": [float(x) for x in d_neg.tolist()],
            "all_closeness": [float(x) for x in closeness.tolist()],
            "candidate_count": len(solutions),
            "selected_index": best_idx,
            "selected_solution": {
                "weight": best["weight"],
                "ppl": best["ppl"],
                "calls": best["calls"],
                "selected_indices": sorted(best["selected_indices"]),
            },
        }

        logger.info(
            f"[TOPSIS] selected cand {best_idx}: weight={best['weight']}, PPL={best['ppl']:.6f}, "
            f"calls={best['calls']:.6f}, closeness={info['closeness']:.6f}, d+={info['distance_to_ideal']:.6f}, d-={info['distance_to_negative_ideal']:.6f}"
        )
        return best, info


    def _nsga_function_selection(
        self,
        items: List[Dict[str, Any]],
        target_tokens: int,
        preserved_indices: Optional[set] = None,
        max_archive: int = 512,
        language: str = "python",
        expansion_ratio: float = 1.5,
        context_budget: str = "+100",
    ) -> Tuple[set, Dict[str, Any]]:
        """Two-stage ranking selector for function chunks.

        Stage 1:
            Use the same greedy rule as the original coarse selector:
            pick the next highest-PPL chunk, then stop only after the budget is exceeded.
            The number of selected chunks is p.

        Stage 2:
            Expand the stage-1 set to k = ceil(r * p), where r is expansion_ratio,
            then stable-sort those k candidates by call score descending (tie -> ppl_change),
            and apply the same greedy rule as Stage 1.

        Notes:
            - context_budget keeps the original extra-token behavior (default +100).
            - Python's sorted()/list.sort() are stable.
            - We use the original chunk index as a deterministic tie-breaker.
        """
        logger.info("[TWO-STAGE] ==================================================")
        logger.info(
            f"[TWO-STAGE] start selection: target_tokens={target_tokens}, "
            f"context_budget={context_budget}, language={language}, expansion_ratio={expansion_ratio}"
        )
        logger.info(f"[TWO-STAGE] items={len(items)}, preserved_indices={sorted(preserved_indices or [])}")

        if not items:
            logger.warning("[TWO-STAGE] empty items, return empty selection")
            return set(), {
                "method": "two_stage_pplchange_then_calls",
                "empty": True,
                "stable_sort": True,
                "stage1": {},
                "stage2": {},
            }

        block_weights = [int(item.get("tokens", self.get_token_length(item.get("chunk", "")))) for item in items]

        clean_ppl: List[float] = []
        clean_calls: List[float] = []
        for i, item in enumerate(items):
            ppl = float(item.get("ppl", 0.0) or 0.0)
            calls = float(item.get("call_score", 0.0) or 0.0)
            if math.isnan(ppl) or math.isinf(ppl):
                ppl = 0.0
            if math.isnan(calls) or math.isinf(calls):
                calls = 0.0
            clean_ppl.append(ppl)
            clean_calls.append(calls)
            logger.debug(
                f"[TWO-STAGE] item[{i}] orig_idx={item['orig_index']}, weight={block_weights[i]}, "
                f"PPL={ppl:.6f}, calls={calls:.6f}"
            )

        try:
            selection_budget = int(eval(f"target_tokens{context_budget}"))
        except Exception as e:
            logger.warning(f"[TWO-STAGE] failed to evaluate context_budget={context_budget!r}: {e}; using target_tokens directly")
            selection_budget = int(target_tokens)

        selection_budget = max(0, selection_budget)
        logger.info(f"[TWO-STAGE] selection_budget={selection_budget} (from target_tokens={target_tokens}, context_budget={context_budget})")

        item_indices = list(range(len(items)))
        selectable_positions = item_indices

        def explain_item(pos: int) -> str:
            item = items[pos]
            name = item.get("function_name") or item.get("signature_text") or f"chunk_{item['orig_index']}"
            return (
                f"idx={item['orig_index']} name={name!r} tokens={block_weights[pos]} "
                f"ppl_change={clean_ppl[pos]:.6f} calls={clean_calls[pos]:.6f}"
            )

        stage1_order = sorted(
            selectable_positions,
            key=lambda pos: (-float(clean_ppl[pos]), int(items[pos]["orig_index"])),
        )

        logger.info("[TWO-STAGE][stage1] ranking by ppl_change (stable, tie -> original index):")
        for rank, pos in enumerate(stage1_order, start=1):
            logger.info(f"  rank={rank:03d} {explain_item(pos)}")

        stage1_selected: List[int] = []
        used_budget = 0
        logger.info("[TWO-STAGE][stage1] greedy selection under budget:")
        for rank, pos in enumerate(stage1_order, start=1):
            w = block_weights[pos]
            before_budget = used_budget
            stage1_selected.append(pos)
            used_budget += w
            logger.info(
                f"  rank={rank:03d} action=TAKE used={before_budget}/{selection_budget} "
                f"next_weight={w} -> {explain_item(pos)}"
            )
            if used_budget > selection_budget:
                break

        p = len(stage1_selected)
        if p <= 0:
            logger.warning("[TWO-STAGE] stage1 selected nothing; return empty selection")
            selection_info = {
                "method": "two_stage_pplchange_then_calls",
                "stable_sort": True,
                "total_ppl_change": 0.0,
                "total_calls": 0.0,
                "total_weight": 0,
                "target_weight": target_tokens,
                "selection_budget": selection_budget,
                "solution_count": 0,
                "stage1": {
                    "p": 0,
                    "k": 0,
                    "selected_positions": [],
                    "selected_indices": [],
                    "order_positions": [int(x) for x in stage1_order],
                    "order_indices": [int(items[pos]["orig_index"]) for pos in stage1_order],
                },
                "stage2": {
                    "q": 0,
                    "selected_positions": [],
                    "selected_indices": [],
                    "order_positions": [],
                    "order_indices": [],
                },
            }
            logger.info("[TWO-STAGE] ==================================================")
            return set(), selection_info

        n = len(selectable_positions)
        raw_k = float(expansion_ratio) * float(p)
        k = int(math.ceil(raw_k))
        k = max(1, min(n, k))
        stage1_topk = stage1_order[:k]

        logger.info(
            f"[TWO-STAGE] stage1 summary: n={n}, p={p}, r={expansion_ratio}, r*p={raw_k:.6f}, k=ceil(r*p)={k}"
        )
        logger.info(
            "[TWO-STAGE][stage1] selected by PPL under budget: "
            + str([items[pos]["orig_index"] for pos in stage1_selected])
        )
        logger.info(
            "[TWO-STAGE][stage1] top-k candidates kept for stage2: "
            + str([items[pos]["orig_index"] for pos in stage1_topk])
        )

        stage2_order = sorted(
            stage1_topk,
            key=lambda pos: (
                -float(clean_calls[pos]),
                -float(clean_ppl[pos]),
                int(items[pos]["orig_index"]),
            ),
        )

        logger.info("[TWO-STAGE][stage2] ranking by call score, tie -> ppl_change, tie -> original index:")
        for rank, pos in enumerate(stage2_order, start=1):
            logger.info(f"  rank={rank:03d} {explain_item(pos)}")

        # Stage 2 uses the same greedy rule as Stage 1:
        # take the current item first, then stop only after the budget becomes negative.
        stage2_selected: List[int] = []
        used_budget = 0
        logger.info("[TWO-STAGE][stage2] greedy selection under budget:")
        for rank, pos in enumerate(stage2_order, start=1):
            w = block_weights[pos]
            before_budget = used_budget
            stage2_selected.append(pos)
            used_budget += w
            logger.info(
                f"  rank={rank:03d} action=TAKE used={before_budget}/{selection_budget} "
                f"next_weight={w} -> {explain_item(pos)}"
            )
            if used_budget > selection_budget:
                break

        q = len(stage2_selected)
        logger.info(
            f"[TWO-STAGE] stage2 summary: q={q}, budget_used={used_budget}/{selection_budget}, "
            f"selected_indices={[items[pos]['orig_index'] for pos in stage2_selected]}"
        )

        best_positions = stage2_selected
        best_indices = [int(items[pos]["orig_index"]) for pos in best_positions]
        total_ppl = sum(clean_ppl[pos] for pos in best_positions)
        total_calls = sum(clean_calls[pos] for pos in best_positions)
        total_weight = sum(block_weights[pos] for pos in best_positions)

        selection_info = {
            "method": "two_stage_pplchange_then_calls",
            "stable_sort": True,
            "total_ppl_change": float(total_ppl),
            "total_calls": float(total_calls),
            "total_weight": int(total_weight),
            "target_weight": target_tokens,
            "selection_budget": selection_budget,
            "candidate_count": len(items),
            "selectable_count": len(selectable_positions),
            "stage1": {
                "n": n,
                "p": p,
                "k": k,
                "raw_k": float(raw_k),
                "expansion_ratio": float(expansion_ratio),
                "selected_positions": [int(x) for x in stage1_selected],
                "selected_indices": [int(items[pos]["orig_index"]) for pos in stage1_selected],
                "order_positions": [int(x) for x in stage1_order],
                "order_indices": [int(items[pos]["orig_index"]) for pos in stage1_order],
                "topk_positions": [int(x) for x in stage1_topk],
                "topk_indices": [int(items[pos]["orig_index"]) for pos in stage1_topk],
            },
            "stage2": {
                "q": q,
                "selected_positions": [int(x) for x in stage2_selected],
                "selected_indices": [int(items[pos]["orig_index"]) for pos in stage2_selected],
                "order_positions": [int(x) for x in stage2_order],
                "order_indices": [int(items[pos]["orig_index"]) for pos in stage2_order],
            },
        }

        final_selection = set(best_indices)
        logger.info(
            f"[TWO-STAGE] final selection={len(final_selection)}/{len(items)} chunks, "
            f"ppl_change={total_ppl:.6f}, calls={total_calls:.6f}, weight={total_weight}/{selection_budget}"
        )
        logger.info("[TWO-STAGE] ==================================================")
        return final_selection, selection_info

    def hoist_nested_functions_to_outer_scope(self, source: str) -> str:
        """
        文本级 hoist：
        - 支持 Python2 / Python3
        - 不使用 AST
        - 只把嵌套函数提到最外层/外层函数前
        - 外层函数其余内容保持原缩进，不重排
        - 被提出来的函数块整体缩进对齐到外层函数同级
        - 函数内部相对缩进保持不变
        - 不插入 pass
        - class 内函数不越界
        """

        import re

        TABSIZE = 4

        def leading_ws(line: str) -> str:
            m = re.match(r"^[ \t]*", line)
            return m.group(0) if m else ""

        def indent_width(line: str) -> int:
            """
            用于比较缩进层级：把 tab 按 4 个空格计算。
            """
            return len(leading_ws(line).expandtabs(TABSIZE))

        def is_blank(line: str) -> bool:
            return not line.strip()

        def is_comment(line: str) -> bool:
            return line.lstrip().startswith("#")

        def is_decorator(line: str) -> bool:
            return line.lstrip().startswith("@")

        def is_def_header(line: str) -> bool:
            return re.match(r"\s*(async\s+def|def)\b", line) is not None

        def is_class_header(line: str) -> bool:
            return re.match(r"\s*class\b", line) is not None

        def is_def_or_class(line: str) -> bool:
            return is_def_header(line) or is_class_header(line)

        def next_code(lines, i, end):
            while i < end and (is_blank(lines[i]) or is_comment(lines[i])):
                i += 1
            return i if i < end else None

        def find_block_end(lines, start, base_width, end):
            """
            从 start 开始找代码块结束。
            规则：遇到缩进 <= base_width 的非空非注释行，就认为块结束。
            """
            i = start + 1
            while i < end:
                if is_blank(lines[i]) or is_comment(lines[i]):
                    i += 1
                    continue
                if indent_width(lines[i]) <= base_width:
                    break
                i += 1
            return i

        def capture_block(lines, i, end):
            """
            返回 (start, header, block_end, kind)
            - start：decorator 起点或 def/class 起点
            - header：真正的 def/class 行
            - block_end：块结束（开区间）
            - kind：function / class
            """
            if i >= end:
                return None

            # decorator + def/class
            if is_decorator(lines[i]):
                base_width = indent_width(lines[i])
                j = i
                while j < end and is_decorator(lines[j]) and indent_width(lines[j]) == base_width:
                    j += 1

                header = next_code(lines, j, end)
                if header is None:
                    return None

                if indent_width(lines[header]) != base_width:
                    return None

                if not is_def_or_class(lines[header]):
                    return None

                kind = "function" if is_def_header(lines[header]) else "class"
                block_end = find_block_end(lines, header, base_width, end)
                return i, header, block_end, kind

            # def/class
            if is_def_or_class(lines[i]):
                base_width = indent_width(lines[i])
                kind = "function" if is_def_header(lines[i]) else "class"
                block_end = find_block_end(lines, i, base_width, end)
                return i, i, block_end, kind

            return None

        def shift_block_to_ws(block, from_ws: str, to_ws: str):
            """
            把 block 的整体缩进从 from_ws 平移到 to_ws。
            只改前导缩进，不改正文内容。
            """
            if from_ws == to_ws:
                return block[:]

            out = []
            for line in block:
                if is_blank(line):
                    out.append(line)
                    continue

                if line.startswith(from_ws):
                    out.append(to_ws + line[len(from_ws):])
                else:
                    # 兜底：如果前缀不完全匹配，就退化为去掉所有前导空白再加目标缩进
                    out.append(to_ws + line.lstrip())

            return out

        def process_function(lines, start, header, end, target_ws: str):
            """
            处理一个 function block：
            - 保留外层函数其余内容原样
            - 只移除嵌套函数，并把它们 hoist 出去
            - 返回值：hoisted_items + current_function_block
            """
            header_ws = leading_ws(lines[header])
            body_start = next_code(lines, header + 1, end)

            # 空函数 / 异常函数体：原样或按需移动
            if body_start is None:
                original = lines[start:end]
                return original if target_ws == header_ws else shift_block_to_ws(original, header_ws, target_ws)

            # prefix：decorator、def 行、以及 def 后面紧跟着的空行/注释（如果有）
            # body：真正的函数体内容
            prefix = lines[start:body_start]
            kept_body = []
            hoisted = []
            changed = False

            i = body_start
            while i < end:
                if is_blank(lines[i]) or is_comment(lines[i]):
                    kept_body.append(lines[i])
                    i += 1
                    continue

                blk = capture_block(lines, i, end)
                if blk:
                    s, h, e, kind = blk

                    if kind == "function":
                        # 递归处理嵌套函数：先把它内部的更深层嵌套也 hoist 好，再整体提出来
                        nested_items = process_function(lines, s, h, e, target_ws)
                        hoisted.extend(nested_items)
                        changed = True
                        i = e
                        continue

                    if kind == "class":
                        # class 不越界：保持在当前函数体内
                        cls = process_class(lines, s, h, e)
                        kept_body.extend(cls)
                        if cls != lines[s:e]:
                            changed = True
                        i = e
                        continue

                kept_body.append(lines[i])
                i += 1

            # 没有任何变化：完全原样返回
            if not changed:
                original = lines[start:end]
                return original if target_ws == header_ws else shift_block_to_ws(original, header_ws, target_ws)

            # 当前函数本体：只去掉被 hoist 的 nested function，其余内容保持原缩进
            current_block = prefix + kept_body

            # 如果这是被提到外层的函数块，则整体缩进对齐到 target_ws
            if target_ws != header_ws:
                current_block = shift_block_to_ws(current_block, header_ws, target_ws)

            return hoisted + current_block

        def process_class(lines, start, header, end):
            """
            处理 class block：
            - class 内的 method 里的嵌套函数可以 hoist 到 method 前
            - 但不会越过 class 边界
            - class 自身不做整体重排
            """
            class_ws = leading_ws(lines[header])
            body_start = next_code(lines, header + 1, end)

            if body_start is None:
                return lines[start:end]

            prefix = lines[start:body_start]
            body = []
            changed = False

            i = body_start
            while i < end:
                if is_blank(lines[i]) or is_comment(lines[i]):
                    body.append(lines[i])
                    i += 1
                    continue

                blk = capture_block(lines, i, end)
                if blk:
                    s, h, e, kind = blk

                    if kind == "function":
                        # method 内的嵌套函数：hoist 到 method 同级
                        method_target_ws = leading_ws(lines[body_start])
                        items = process_function(lines, s, h, e, method_target_ws)
                        body.extend(items)
                        if items != lines[s:e]:
                            changed = True
                        i = e
                        continue

                    if kind == "class":
                        nested_cls = process_class(lines, s, h, e)
                        body.extend(nested_cls)
                        if nested_cls != lines[s:e]:
                            changed = True
                        i = e
                        continue

                body.append(lines[i])
                i += 1

            if not changed:
                return lines[start:end]

            # class 本身不越界，不整体重缩进，只重组内容
            return prefix + body

        # ===== 主逻辑 =====
        source = source.replace("\r\n", "\n").replace("\r", "\n")
        lines = source.split("\n")
        n = len(lines)

        out = []
        i = 0

        while i < n:
            if is_blank(lines[i]) or is_comment(lines[i]):
                out.append(lines[i])
                i += 1
                continue

            blk = capture_block(lines, i, n)
            if blk:
                s, h, e, kind = blk

                if kind == "function":
                    items = process_function(lines, s, h, e, leading_ws(lines[h]))
                    out.extend(items)
                    i = e
                    continue

                if kind == "class":
                    cls = process_class(lines, s, h, e)
                    out.extend(cls)
                    i = e
                    continue

            out.append(lines[i])
            i += 1

        return "\n".join(out)

    def split_code_by_functions(self, code: str, language: str = "python", custom_separator: str = "# --CHUNK_SEPARATOR-- #") -> List[str]:
        """
        Split code into chunks based on function and class definitions for various languages.
        Also splits on custom separator if provided.
        
        Args:
            code: The code to split
            language: Programming language of the code (python, cpp, java, typescript, rust, go)
            custom_separator: Optional custom separator string to also split on
            
        Returns:
            List of code chunks, each containing a function, class, or class method
        """
        logger.debug(f"Splitting code by functions and classes for language: {language}")
        start_time = time.time()
        
        # Define regex patterns for different languages
        patterns = {
            # Python: Simplified to match 'def' or 'class' followed by content until the next def/class or end
            "python": r'(^|\n)(\s*)(def|class)\s+[^\n]+(\n(?!\s*(?:def|class)\s)[^\n]*)*',
            # C++: Improved to better handle multi-line declarations
            "cpp": r'(^|\n)(\s*)(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*[^{]*)?|(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
            # Java: Improved for multi-line method declarations
            "java": r'(^|\n)(\s*)(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:<.*>)?(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*throws\s+[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
            # TypeScript: Enhanced to handle multi-line methods and arrow functions
            "typescript": r'(^|\n)(\s*)(?:(?:public|private|protected|static|abstract)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:(?:public|private|protected|static|async)\s+)*(?:function\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<.*>)?\s*\([^{;]*\)\s*(?::\s*[^{;]*\s*)?(?:=>)?)\s*(?:{[^}]*}|[^;]*;)?',
            # Rust: Improved for multi-line function declarations
            "rust": r'(^|\n)(\s*)(?:pub\s+)?(?:struct\s+[a-zA-Z_][a-zA-Z0-9_]*|impl(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+for\s+[a-zA-Z_][a-zA-Z0-9_]*)?|(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<.*>)?\s*\([^{;]*\)(?:\s*->\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
            # Go: Improved for multi-line function declarations
            "go": r'(^|\n)(\s*)(?:type\s+[a-zA-Z_][a-zA-Z0-9_]*\s+struct|func\s+(?:\([^)]*\)\s*)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
        }
        
        # Use default Python pattern if language not supported
        if language.lower() not in patterns:
            language = "python"
        
        # First check if we need to split by custom separator
        separator_chunks = []
        if custom_separator and custom_separator in code:
            logger.debug(f"Custom separator '{custom_separator}' found, first splitting by separator")
            separator_chunks = [chunk for chunk in code.split(custom_separator) if chunk.strip()]
        else:
            separator_chunks = [code]  # Just one chunk - the entire code

        # Function to split a single chunk by functions/classes
        def split_chunk_by_pattern(chunk_code):
            function_pattern = re.compile(patterns[language.lower()], re.MULTILINE)
            matches = list(function_pattern.finditer(chunk_code))
            
            if not matches:
                return [chunk_code]  # No matches, return whole chunk
                
            result_chunks = []
            
            # Add code before first match
            if matches[0].start() > 0:
                result_chunks.append(chunk_code[:matches[0].start()])
            
            # Process each match
            for i, match in enumerate(matches):
                start = match.start()
                
                # End is either start of next match or end of code
                if i < len(matches) - 1:
                    end = matches[i + 1].start()
                else:
                    end = len(chunk_code)
                
                result_chunks.append(chunk_code[start:end])
            
            return result_chunks
        
        # Now apply function/class splitting to each separator chunk
        final_chunks = []
        for chunk in separator_chunks:
            function_chunks = split_chunk_by_pattern(chunk)
            final_chunks.extend(function_chunks)
        
        end_time = time.time()
        logger.debug(f"Code splitting completed in {time.time() - start_time:.2f} seconds")
        logger.debug(f"Split code into {len(final_chunks)} chunks (using both separator and patterns)")
        
        return final_chunks

    def _calculate_perplexity_for_contrastive(self, text, condition_text=None):
        """Helper to calculate perplexity of text, optionally conditioned on condition_text"""
        if condition_text:
            full_text = condition_text + text
            inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(self.device) # Use add_special_tokens=True for consistency
            
            condition_input_ids = self.tokenizer(condition_text, return_tensors="pt", add_special_tokens=True).input_ids
            condition_length = condition_input_ids.size(1)

            # Handle potential edge case where condition length might exceed max length or input length
            if condition_length >= inputs.input_ids.size(1):
                    logger.warning(f"Condition length ({condition_length}) >= input length ({inputs.input_ids.size(1)}). Cannot calculate conditional PPL.")
                    return float('inf')

            with torch.no_grad():
                outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask) # Pass attention_mask

            # Logits for the 'text' part, labels are the 'text' part shifted
            logits = outputs.logits[0, condition_length-1:-1]
            labels = inputs.input_ids[0, condition_length:]

            if logits.size(0) == 0 or labels.size(0) == 0 or logits.size(0) != labels.size(0):
                logger.warning(f"Logits/Labels shape mismatch or empty in _calculate_perplexity_for_contrastive (cond). Logits: {logits.shape}, Labels: {labels.shape}. Returning inf.")
                return float('inf') # Return inf if shapes mismatch or empty

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            mean_loss = loss.mean().item()
            perplexity = math.exp(mean_loss) if not math.isnan(mean_loss) and not math.isinf(mean_loss) else float('inf')

        else:
            # Calculate unconditional perplexity
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device) # Use add_special_tokens=True
            with torch.no_grad():
                outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask) # Pass attention_mask

            # Logits for all tokens except last, labels are all tokens except first
            logits = outputs.logits[0, :-1]
            labels = inputs.input_ids[0, 1:]

            if logits.size(0) == 0 or labels.size(0) == 0 or logits.size(0) != labels.size(0):
                logger.warning(f"Logits/Labels shape mismatch or empty in _calculate_perplexity_for_contrastive (uncond). Logits: {logits.shape}, Labels: {labels.shape}. Returning inf.")
                return float('inf') # Return inf if shapes mismatch or empty

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            mean_loss = loss.mean().item()
            perplexity = math.exp(mean_loss) if not math.isnan(mean_loss) and not math.isinf(mean_loss) else float('inf')

        return perplexity

    def _calculate_contrastive_perplexity(self, code_lines: List[str], question: str):
        """
        Calculate contrastive perplexity-based importance for each line of code.
        s_i = perplexity(x_i | x_{<i}) - perplexity(x_i | x^{que}, x_{<i})
        Higher score means the question helps predict the line more.

        Args:
            code_lines: List of code lines to analyze
            question: The query/question text

        Returns:
            Tuple of (line_scores, scored_indices)
        """
        logger.debug("Calculating contrastive perplexity-based line importance...")
        line_scores = []
        scored_indices = []

        with torch.no_grad():
            # Use tqdm.auto for better compatibility
            pbar = tqdm(enumerate(code_lines), total=len(code_lines), desc="Contrastive PPL", leave=False)
            for i, line in pbar:
                if not line.strip():
                    continue  # Skip empty lines

                # Ensure line has content before proceeding
                if not line:
                    logger.debug(f"Skipping empty line {i}")
                    continue

                # 1. PPL(L_i | L_<i)
                prev_context = "\n".join(code_lines[:i])
                # Add newline only if previous context exists
                regular_ppl_condition = prev_context + "\n" if prev_context else None
                regular_ppl = self._calculate_perplexity_for_contrastive(line, condition_text=regular_ppl_condition)


                # 2. PPL(L_i | Q, L_<i)
                # Combine question and previous context carefully
                question_context_parts = [question]
                if prev_context:
                    question_context_parts.append(prev_context)
                # Join with double newline between Q and prev_context if both exist
                question_context = "\n\n".join(filter(None, question_context_parts))
                # Add trailing newline before the target line
                cond_ppl_condition = question_context + "\n"
                cond_ppl = self._calculate_perplexity_for_contrastive(line, condition_text=cond_ppl_condition)

                # 3. Importance = PPL(L|prev) - PPL(L|Q,prev)
                if math.isinf(regular_ppl) or math.isinf(cond_ppl):
                    # If either is infinite, the difference isn't well-defined for ranking.
                    # Assign a very low score, potentially based on which one is inf.
                    # If regular_ppl is inf, question might still help (cond_ppl could be finite).
                    # If cond_ppl is inf, question made it worse or impossible to predict.
                    # Let's assign -inf for simplicity, meaning "least important".
                    importance = -float('inf')
                    logger.debug(f"Line {i}: Inf PPL detected. Regular: {regular_ppl}, Conditional: {cond_ppl}. Importance set to -inf")
                else:
                    importance = regular_ppl - cond_ppl
                    logger.debug(f"Line {i}: PPL(L|prev)={regular_ppl:.4f}, PPL(L|Q,prev)={cond_ppl:.4f}, Importance={importance:.4f}")

                line_scores.append(importance)
                scored_indices.append(i)
                # Update tqdm description if needed, e.g., with last score
                # pbar.set_description(f"Contrastive PPL (L{i}: {importance:.2f})")

        logger.debug(f"Finished calculating contrastive PPL for {len(line_scores)} lines.")
        return line_scores, scored_indices


    def _is_code_line(self, text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return False
        if s.startswith("#") or s.startswith("//"):
            return False
        return True

    def _collect_block_candidate_lines(self, source_lines: List[str], start_line: int, end_line: int) -> List[int]:
        line_nos: List[int] = []
        start_line = max(1, int(start_line))
        end_line = max(start_line, int(end_line))
        end_line = min(end_line, len(source_lines))
        for ln in range(start_line, end_line + 1):
            if 1 <= ln <= len(source_lines) and self._is_code_line(source_lines[ln - 1]):
                line_nos.append(ln)
        return line_nos

    def _build_line_to_block_map(self, blocks: List[Any], source_lines: List[str]) -> Dict[int, int]:
        candidates: Dict[int, List[Tuple[int, int, int]]] = {}
        for blk in blocks:
            bid = int(getattr(blk, "block_id", 0) or 0)
            start_line = int(getattr(blk, "start_line", 0) or 0)
            end_line = int(getattr(blk, "end_line", start_line) or start_line)
            depth = int(getattr(blk, "depth", 0) or 0)
            line_nos = self._collect_block_candidate_lines(source_lines, start_line, end_line)
            if not line_nos:
                continue
            span_len = max(1, max(line_nos) - min(line_nos) + 1)
            for ln in line_nos:
                candidates.setdefault(ln, []).append((span_len, depth, bid))

        line_to_block: Dict[int, int] = {}
        for ln, items in candidates.items():
            items.sort(key=lambda x: (x[0], x[1], x[2]))
            line_to_block[ln] = items[0][2]
        return line_to_block

    def _compute_block_dependency_counts(self, blocks: List[Any], source_lines: List[str], dependency_line_graph: Any) -> List[int]:
        """
        Count how many dependency edges touch each block.

        Objective dependency_count = used_by_total + depends_on_total
        - used_by_total: other blocks depend on this block
        - depends_on_total: this block depends on other blocks
        """
        line_to_block = self._build_line_to_block_map(blocks, source_lines)
        dep_counts: List[int] = []

        for blk in blocks:
            bid = int(getattr(blk, "block_id", 0) or 0)
            start_line = int(getattr(blk, "start_line", 0) or 0)
            end_line = int(getattr(blk, "end_line", start_line) or start_line)
            owned_lines = self._collect_block_candidate_lines(source_lines, start_line, end_line)

            used_by_total = 0
            depends_on_total = 0

            for ln in owned_lines:
                src_id = f"L{ln}"
                if src_id not in dependency_line_graph:
                    continue
                for pred, _, data in dependency_line_graph.in_edges(src_id, data=True):
                    cnt = int(data.get("count", 1) or 1)
                    pred_line = int(dependency_line_graph.nodes[pred].get("line_no", 0) or 0)
                    if line_to_block.get(pred_line) == bid:
                        continue
                    used_by_total += cnt
                for _, succ, data in dependency_line_graph.out_edges(src_id, data=True):
                    cnt = int(data.get("count", 1) or 1)
                    succ_line = int(dependency_line_graph.nodes[succ].get("line_no", 0) or 0)
                    if line_to_block.get(succ_line) == bid:
                        continue
                    depends_on_total += cnt

            dep_counts.append(float(used_by_total) + float(depends_on_total))

        return dep_counts

    @staticmethod
    def _solution_selected_key(sol: Dict[str, Any]) -> Tuple[int, ...]:
        return tuple(sorted(int(x) for x in sol.get("selected_indices", [])))

    @staticmethod
    def _solution_obj_key(sol: Dict[str, Any]) -> Tuple[float, float]:
        return (round(float(sol.get("ami", 0.0)), 6), round(float(sol.get("dependency", 0.0)), 6))

    @staticmethod
    def _solution_rank_key(sol: Dict[str, Any]) -> Tuple[float, float, float, int, Tuple[int, ...]]:
        return (
            float(sol.get("ami", 0.0)) + float(sol.get("dependency", 0.0)),
            float(sol.get("ami", 0.0)),
            float(sol.get("dependency", 0.0)),
            -int(sol.get("weight", 0)),
            tuple(sorted(int(x) for x in sol.get("selected_indices", []))),
        )

    def _pareto_prune_solutions(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only Pareto-nondominated solutions, but never collapse identical selections prematurely."""
        if not solutions:
            return []

        unique_by_selection: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for sol in solutions:
            key = self._solution_selected_key(sol)
            cur = unique_by_selection.get(key)
            if cur is None or self._solution_rank_key(sol) > self._solution_rank_key(cur):
                unique_by_selection[key] = sol

        unique_solutions = list(unique_by_selection.values())
        pruned: List[Dict[str, Any]] = []
        for sol in unique_solutions:
            dominated = False
            remove_idxs: List[int] = []
            for i, kept in enumerate(pruned):
                kept_better = kept["ami"] >= sol["ami"] and kept["dependency"] >= sol["dependency"]
                kept_strict = kept["ami"] > sol["ami"] or kept["dependency"] > sol["dependency"]
                if kept_better and kept_strict:
                    dominated = True
                    break
                sol_better = sol["ami"] >= kept["ami"] and sol["dependency"] >= kept["dependency"]
                sol_strict = sol["ami"] > kept["ami"] or sol["dependency"] > kept["dependency"]
                if sol_better and sol_strict:
                    remove_idxs.append(i)
            if dominated:
                continue
            for idx in reversed(remove_idxs):
                pruned.pop(idx)
            pruned.append(sol)

        pruned.sort(key=lambda s: self._solution_rank_key(s), reverse=True)
        return pruned

    def _build_diverse_archive(
        self,
        solutions: List[Dict[str, Any]],
        max_archive: int = 256,
        epsilon: float = 0.05,
        preserve_pareto: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build a diverse archive for TOPSIS.

        For tiny subproblems, a strict Pareto front can collapse to a single point
        because both objectives are positively correlated.  This archive keeps a
        diverse set of unique selections and only falls back to Pareto pruning when
        it still leaves enough variety.
        """
        if not solutions:
            return []

        unique_by_selection: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for sol in solutions:
            key = self._solution_selected_key(sol)
            cur = unique_by_selection.get(key)
            if cur is None or self._solution_rank_key(sol) > self._solution_rank_key(cur):
                unique_by_selection[key] = sol

        unique = list(unique_by_selection.values())
        if not unique:
            return []

        if preserve_pareto:
            pareto = self._pareto_prune_solutions(unique)
            # If Pareto collapses too much, keep the wider unique pool instead.
            if len(pareto) >= 4:
                unique = pareto
            else:
                logger.info(
                    f"[NSGA-II] Pareto front too small ({len(pareto)}), keeping unique feasible pool ({len(unique)}) for diversity"
                )

        if len(unique) <= max_archive:
            return sorted(unique, key=self._solution_rank_key, reverse=True)

        matrix = np.array([[float(sol["ami"]), float(sol["dependency"])] for sol in unique], dtype=float)
        norm = self._minmax_normalize(matrix)
        eps = max(float(epsilon), 1e-6)
        buckets: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for sol, vec in zip(unique, norm):
            key = (int(min(vec[0] / eps, 1.0 / eps)), int(min(vec[1] / eps, 1.0 / eps)))
            cur = buckets.get(key)
            if cur is None or self._solution_rank_key(sol) > self._solution_rank_key(cur):
                buckets[key] = sol

        archive = list(buckets.values())
        if len(archive) > max_archive:
            archive.sort(key=self._solution_rank_key, reverse=True)
            archive = archive[:max_archive]

        if len(archive) < min(8, len(unique)):
            # Keep at least a small diverse pool if epsilon buckets were too coarse.
            unique.sort(key=self._solution_rank_key, reverse=True)
            archive = unique[:max_archive]

        return archive

    @staticmethod
    def _minmax_normalize(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
        return (matrix - mins) / denom


    def _topsis_select_solution(self, solutions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not solutions:
            raise ValueError("No feasible solutions for TOPSIS.")

        deduped: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for sol in solutions:
            key = tuple(sorted(int(x) for x in sol.get("selected_indices", [])))
            cur = deduped.get(key)
            if cur is None:
                deduped[key] = sol
            else:
                cur_score = (float(cur["ami"]) + float(cur["dependency"]), float(cur["ami"]), float(cur["dependency"]), -int(cur["weight"]))
                new_score = (float(sol["ami"]) + float(sol["dependency"]), float(sol["ami"]), float(sol["dependency"]), -int(sol["weight"]))
                if new_score > cur_score:
                    deduped[key] = sol

        solutions = list(deduped.values())
        if not solutions:
            raise ValueError("No feasible solutions for TOPSIS after deduplication.")

        logger.info(f"[TOPSIS] candidate_count={len(solutions)}")
        for i, sol in enumerate(solutions[: min(10, len(solutions))]):
            logger.info(
                f"[TOPSIS][cand {i}] weight={sol['weight']}, AMI={sol['ami']:.6f}, dep_score={sol['dependency']:.6f}, "
                f"selected={sorted(sol['selected_indices'])}"
            )

        matrix = np.array([[float(sol["ami"]), float(sol["dependency"])] for sol in solutions], dtype=float)
        logger.info("[TOPSIS] decision matrix (AMI, dependency):")
        for i, row in enumerate(matrix.tolist()):
            logger.info(f"  cand {i}: AMI={row[0]:.6f}, dep={row[1]:.6f}")

        if matrix.shape[0] == 1 or np.allclose(matrix, matrix[0:1, :], atol=1e-12, rtol=0.0):
            logger.warning("[TOPSIS] all candidates are identical in objective space; using tie-breaker")
            best_idx = max(
                range(len(solutions)),
                key=lambda i: (
                    float(solutions[i]["ami"]) + float(solutions[i]["dependency"]),
                    float(solutions[i]["ami"]),
                    float(solutions[i]["dependency"]),
                    -int(solutions[i]["weight"]),
                    -len(solutions[i]["selected_indices"]),
                ),
            )
            best = solutions[best_idx]
            norm = np.zeros_like(matrix, dtype=float)
            positive_ideal = np.array([1.0, 1.0], dtype=float)
            negative_ideal = np.array([0.0, 0.0], dtype=float)
            d_pos = np.linalg.norm(norm - positive_ideal, axis=1)
            d_neg = np.linalg.norm(norm - negative_ideal, axis=1)
            closeness = np.where((d_pos + d_neg) > 0, d_neg / (d_pos + d_neg + 1e-12), 0.5)
        else:
            norm = self._minmax_normalize(matrix)
            positive_ideal = np.array([1.0, 1.0], dtype=float)
            negative_ideal = np.array([0.0, 0.0], dtype=float)
            d_pos = np.linalg.norm(norm - positive_ideal, axis=1)
            d_neg = np.linalg.norm(norm - negative_ideal, axis=1)
            closeness = d_neg / (d_pos + d_neg + 1e-12)
            best_idx = int(np.argmax(closeness))
            best = solutions[best_idx]

        logger.info("[TOPSIS] normalized matrix:")
        for i, row in enumerate(norm.tolist()):
            logger.info(f"  cand {i}: n_AMI={row[0]:.6f}, n_dep={row[1]:.6f}")
        logger.info(f"[TOPSIS] positive ideal={positive_ideal.tolist()}")
        logger.info(f"[TOPSIS] negative ideal={negative_ideal.tolist()}")

        for i in range(len(solutions)):
            logger.info(
                f"[TOPSIS][cand {i}] d+={float(d_pos[i]):.6f}, d-={float(d_neg[i]):.6f}, closeness={float(closeness[i]):.6f}"
            )

        info = {
            "normalized_matrix": norm.tolist(),
            "positive_ideal": positive_ideal.tolist(),
            "negative_ideal": negative_ideal.tolist(),
            "distance_to_ideal": float(d_pos[best_idx]),
            "distance_to_negative_ideal": float(d_neg[best_idx]),
            "closeness": float(closeness[best_idx]),
            "all_distances_to_ideal": [float(x) for x in d_pos.tolist()],
            "all_distances_to_negative_ideal": [float(x) for x in d_neg.tolist()],
            "all_closeness": [float(x) for x in closeness.tolist()],
            "candidate_count": len(solutions),
            "selected_index": best_idx,
            "selected_solution": {
                "weight": best["weight"],
                "ami": best["ami"],
                "dependency": best["dependency"],
                "selected_indices": sorted(best["selected_indices"]),
            },
        }

        logger.info(
            f"[TOPSIS] selected cand {best_idx}: weight={best['weight']}, AMI={best['ami']:.6f}, "
            f"dep={best['dependency']:.6f}, closeness={info['closeness']:.6f}, d+={info['distance_to_ideal']:.6f}, d-={info['distance_to_negative_ideal']:.6f}"
        )
        return best, info

    def _get_semantic_blocks_and_dependency_counts(
        self,
        func_source: str,
        language: str = "python",
    ) -> Tuple[List[str], List[int], List[Any]]:
        """
        Return semantic chunks together with per-block dependency counts.

        This version prefers graph.py's raw-dot dependency statistics so that:
        - semantic block formatting stays exactly as in the original compression path
        - used_by counts match the visualizer's block-level dependency definition
        - AMI is computed on the same semantic block boundaries
        """
        if language.lower() != "python":
            chunks = self.semantic_chunker._fallback_line_chunks(func_source) if hasattr(self.semantic_chunker, "_fallback_line_chunks") else ([func_source] if func_source.strip() else [])
            return chunks, [0] * len(chunks), []

        raw_func_source = func_source if func_source.endswith("\n") else func_source + "\n"
        analysis_source = textwrap.dedent(raw_func_source)
        if not analysis_source.endswith("\n"):
            analysis_source += "\n"

        # Fast fallback if graph.py or split_code.py is unavailable.
        if semantic_splitter is None or semantic_graph_viz is None:
            chunks = self.semantic_chunker._fallback_line_chunks(func_source) if hasattr(self.semantic_chunker, "_fallback_line_chunks") else ([func_source] if func_source.strip() else [])
            return chunks, [0] * len(chunks), []

        cache_key = hashlib.md5(raw_func_source.encode("utf-8")).hexdigest()[:16]
        if cache_key in self._graph_cache:
            cached = self._graph_cache[cache_key]
            # cached value is the dependency list only; semantic blocks are recomputed below for safety
        try:
            tree = semantic_splitter.parse_python_ast(analysis_source)
            target = None
            for node in getattr(tree, "body", []):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    target = node
                    break
            if target is None:
                chunks = self.semantic_chunker._fallback_line_chunks(func_source) if hasattr(self.semantic_chunker, "_fallback_line_chunks") else ([func_source] if func_source.strip() else [])
                return chunks, [0] * len(chunks), []

            work_dir = self.semantic_work_root / f"semantic_{cache_key}"
            work_dir.mkdir(parents=True, exist_ok=True)
            source_file = work_dir / "snippet.py"
            source_file.write_text(analysis_source, encoding="utf-8")

            pdg_dir = Path(semantic_splitter.generate_pdg_with_joern(source_file, work_dir, self.joern_home))
            func_start_line, func_end_line = semantic_splitter.function_span(target)
            candidate_graphs = semantic_splitter.choose_candidate_graphs(pdg_dir, target.name, func_start_line, func_end_line)
            if not candidate_graphs:
                chunks = self.semantic_chunker._fallback_line_chunks(func_source) if hasattr(self.semantic_chunker, "_fallback_line_chunks") else ([func_source] if func_source.strip() else [])
                return chunks, [0] * len(chunks), []

            parsed_graphs: List[nx.MultiDiGraph] = []
            for g in candidate_graphs:
                dot_file = Path(g.graph.get("dot_file", "")) if g.graph.get("dot_file") else None
                if dot_file and dot_file.exists():
                    try:
                        parsed_graphs.append(semantic_graph_viz.parse_dot_file(dot_file))
                    except Exception:
                        continue
            if not parsed_graphs:
                # fall back to the candidate graph file list, if any
                dot_files = sorted({Path(g.graph.get("dot_file", "")) for g in candidate_graphs if g.graph.get("dot_file")})
                for dot_file in dot_files:
                    if dot_file.exists():
                        try:
                            parsed_graphs.append(semantic_graph_viz.parse_dot_file(dot_file))
                        except Exception:
                            continue

            if not parsed_graphs:
                chunks = self.semantic_chunker._fallback_line_chunks(func_source) if hasattr(self.semantic_chunker, "_fallback_line_chunks") else ([func_source] if func_source.strip() else [])
                return chunks, [0] * len(chunks), []

            merged_raw = semantic_graph_viz.merge_raw_graphs(parsed_graphs)
            raw_lines = raw_func_source.splitlines()

            stmt_infos, stmt_by_ast_id, span_by_line = semantic_splitter.build_stmt_infos_for_function(
                source_text=analysis_source,
                source_lines=raw_lines,
                func_node=target,
            )

            splitter_line_graph = semantic_splitter.build_line_graph_from_merged_pdg(
                merged_raw=merged_raw,
                stmt_infos=stmt_infos,
                span_by_line=span_by_line,
                func_start_line=func_start_line,
                func_end_line=func_end_line,
            )

            blocks = semantic_splitter.build_semantic_blocks(
                func_node=target,
                source_text=analysis_source,
                source_lines=raw_lines,
                line_graph=splitter_line_graph,
                stmt_by_ast_id=stmt_by_ast_id,
            )
            blocks = [blk for blk in blocks if getattr(blk, "code", None) is not None]
            if not blocks:
                chunks = self.semantic_chunker._fallback_line_chunks(func_source) if hasattr(self.semantic_chunker, "_fallback_line_chunks") else ([func_source] if func_source.strip() else [])
                return chunks, [0] * len(chunks), []

            # Build the dependency graph using the exact raw-dot workflow from graph.py.
            dependency_line_graph, raw_node_to_line, diagnostics = semantic_graph_viz.build_dependency_line_graph_from_raw_pdg(
                merged_raw=merged_raw,
                source_lines=raw_lines,
                func_start_line=func_start_line,
                func_end_line=func_end_line,
            )

            block_stats_by_id, _, _ = semantic_graph_viz.compute_block_stats(
                blocks=blocks,
                source_lines=raw_lines,
                line_graph=dependency_line_graph,
                ppl_list=[0.0] * len(blocks),
            )

            chunks = [getattr(blk, "code", "").rstrip() for blk in blocks]
            dep_counts = []
            for blk in blocks:
                bid = int(getattr(blk, "block_id", 0) or 0)
                stats = block_stats_by_id.get(bid, {})
                used_by_total = int(stats.get("used_by_total", 0) or 0)
                depends_on_total = int(stats.get("depends_on_total", 0) or 0)
                dep_counts.append(float(used_by_total) + float(depends_on_total))

            # cache counts only; block formatting always comes from the blocks above
            self._graph_cache[cache_key] = list(dep_counts)

            if len(dep_counts) != len(chunks):
                logger.warning(
                    f"Dependency count length mismatch: chunks={len(chunks)}, dep_counts={len(dep_counts)}; padding/truncating to keep alignment."
                )
                if len(dep_counts) < len(chunks):
                    dep_counts = dep_counts + [0] * (len(chunks) - len(dep_counts))
                else:
                    dep_counts = dep_counts[:len(chunks)]

            logger.debug(
                f"[graph.py] raw nodes={merged_raw.number_of_nodes()}, raw edges={merged_raw.number_of_edges()}, "
                f"dep_graph nodes={dependency_line_graph.number_of_nodes()}, dep_graph edges={dependency_line_graph.number_of_edges()}, "
                f"mapped_line_nodes={diagnostics.get('mapped_line_nodes', 0)}, edge_roles={diagnostics.get('edge_role_counter', {})}"
            )

            return chunks, dep_counts, blocks

        except Exception as e:
            logger.warning(f"Failed to compute semantic blocks with dependency counts; fallback to line chunks. Error: {e}")
            chunks = self.semantic_chunker._fallback_line_chunks(func_source) if hasattr(self.semantic_chunker, "_fallback_line_chunks") else ([func_source] if func_source.strip() else [])
            return chunks, [0] * len(chunks), []

    def _knapsack_block_selection(
        self,
        blocks: List[str],
        block_importances: List[float],
        block_dependency_counts: Optional[List[float]],
        target_tokens: int,
        preserved_block_indices: set = None,
        language: str = "python"
    ) -> Tuple[set, Dict]:
        """Compatibility wrapper: ff8/ff9 use the same three-objective Pareto selector."""
        placeholder = self._build_hybrid_completion_block_scores(
            blocks=blocks,
            block_importances=block_importances,
            block_dependency_counts=block_dependency_counts,
            query=getattr(self, "_last_fine_preservation", {}).get("query", ""),
            language=language,
        )
        return self._hybrid_knapsack_block_selection(
            blocks=blocks,
            block_importances=block_importances,
            hybrid_scores=placeholder,
            block_dependency_counts=block_dependency_counts,
            target_tokens=target_tokens,
            preserved_block_indices=preserved_block_indices,
            language=language,
        )

    def _solve_knapsack_dp(self, items: List[Tuple[int, int, float]], capacity: int) -> set:
        """
        Solve knapsack problem using dynamic programming.

        Args:
            items: List of (index, weight, value) tuples
            capacity: Maximum weight capacity

        Returns:
            Set of selected item indices
        """
        n = len(items)
        if n == 0 or capacity <= 0:
            return set()

        # DP table: dp[i][w] = maximum value using first i items with weight limit w
        dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        # Fill DP table
        for i in range(1, n + 1):
            idx, weight, value = items[i - 1]
            for w in range(capacity + 1):
                # Don't take item i
                dp[i][w] = dp[i - 1][w]

                # Take item i if it fits
                if weight <= w:
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - weight] + value)

        # Backtrack to find selected items
        selected = set()
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                idx, weight, value = items[i - 1]
                selected.add(idx)
                w -= weight

        return selected

    def _solve_knapsack_greedy(self, items: List[Tuple[int, int, float]], capacity: int) -> set:
        """
        Solve knapsack problem using greedy approximation (by value/weight ratio).

        Args:
            items: List of (index, weight, value) tuples (should be pre-sorted by ratio)
            capacity: Maximum weight capacity

        Returns:
            Set of selected item indices
        """
        selected = set()
        current_weight = 0

        for idx, weight, value in items:
            if current_weight + weight <= capacity:
                selected.add(idx)
                current_weight += weight

        return selected

if __name__ == "__main__":
    # Load real examples from the dataset
    # with open("exp-cur50lines-bg5000tokens/results/deepseek-coder-6.7b-instruct/method_code_compressor_t2048_rankonly/deepseek-ai_slash_deepseek-coder-6.7b-instruct.jsonl", "r") as f:
    with open("exp-cur50lines-bg5000tokens-500examples/results/mistral-7b-instruct/method_code_compressor_t512_rankonly/mistralai_slash_Mistral-7B-Instruct-v0.3.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    
    example = data[190]
    # print(example.keys()) # dict_keys(['id', 'gt', 'original_background_context', 'original_current_function_context', 'language', 'prompt', 'output', 'es', 'em'])

    context = example["original_background_context"]
    question = example["original_current_function_context"]
    ground_truth = example["gt"]

    # Initialize compressor
    logger.info("Initializing compressor...")
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    compressor = CodeCompressor(model_name=model_name)
    
    # Test function-based code file compression with query
    logger.info("\nTesting function-based code file compression with query...")

    original_tokens = len(compressor.tokenizer.encode(context))
    target_token = 512
    target_ratio = min(1.0, max(0.0, target_token / original_tokens))
    logger.info(f"CodeCompressor: Original tokens={original_tokens}, Target tokens={target_token}, Calculated ratio={target_ratio:.4f}")

    result = compressor.compress_code_file(
        code=context,
        query=question, # Using current function context as query focus
        instruction="Complete the following code function given the context.",
        rate=target_ratio,
        rank_only=False, # Test fine-grained compression
        fine_grained_importance_method="contrastive_perplexity", # Explicitly test default
        min_lines_for_fine_grained=5, # New parameter
        importance_beta=0.5, # Sensitivity to importance score
        use_knapsack=True,
    )

    # show the compressed code
    logger.info(f"Compressed code (using {result['fine_grained_method_used']}): \n{result['compressed_code']}")
    logger.info(f"Current function context: \n{question}")
    # final prompt
    final_prompt = result['compressed_prompt']
    # get the completion
    try:
        tokenized_prompt = compressor.tokenizer(final_prompt, return_tensors="pt").to(compressor.device)
        # Increase max_new_tokens for potentially longer completions
        completion_ids = compressor.model.generate(**tokenized_prompt, max_new_tokens=128, pad_token_id=compressor.tokenizer.eos_token_id)
        # Decode only the generated part, skipping special tokens
        completion = compressor.tokenizer.decode(completion_ids[0][len(tokenized_prompt.input_ids[0]):], skip_special_tokens=True)

        # Basic cleanup: remove leading/trailing whitespace and potentially stop words if needed
        completion = completion.strip()
        # More robust cleanup: Find the first meaningful line if generation includes noise
        completion_lines = [line for line in completion.split("\n") if line.strip() and not line.strip().startswith(("#", "//"))] # Simple comment removal
        cleaned_completion = completion_lines[0] if completion_lines else completion # Take first non-comment line or original if none found

    except Exception as e:
        logger.error(f"Error during generation or decoding: {e}")
        cleaned_completion = "[ERROR DURING GENERATION]"

    logger.info(f"Cleaned Completion: {cleaned_completion}")
    logger.info(f"Ground truth: {ground_truth}")

    # Optional: Test with conditional_ppl method
    logger.info("\nTesting fine-grained compression with conditional_ppl...")
    result_cond = compressor.compress_code_file(
        code=context,
        query=question,
        instruction="Complete the following code function given the context.",
        rate=target_ratio,
        rank_only=False,
        fine_grained_importance_method="conditional_ppl",
        min_lines_for_fine_grained=5,
        importance_beta=0.5
    )
    logger.info(f"Compressed code (using {result_cond['fine_grained_method_used']}): \n{result_cond['compressed_code']}")
