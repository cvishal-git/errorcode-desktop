#!/usr/bin/env python3
"""
Query Engine for Offline RAG-based Error Code QA System.

Implements:
- Intent detection (error code, category, symptom, how_to, location, wiring)
- Hybrid search over ChromaDB (exact code, category+vector, pure vector)
- Media filtering by intent (wiring/location/how_to)
- Context building for RAG
- Answer generation using Qwen 2.5-3B (llama-cpp)

Run:
    python src/query_engine.py

Requires prior phases:
- Models downloaded (embedding + GGUF)
- Index built with create_index.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import yaml

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None  # type: ignore


logger = logging.getLogger("query_engine")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class SearchResult:
    code: str
    title: str
    similarity: float
    document: str
    metadata: Dict[str, Any]


class QueryEngine:
    """Hybrid search + LLM answer generator for error code QA.

    Methods:
        - extract_error_code
        - extract_category
        - detect_intent
        - search
        - filter_media_by_intent
        - build_context
        - generate_answer
        - query
    """

    def __init__(self, project_root: Optional[Path] = None, preset_name: str = "balanced") -> None:
        """Initialize models, DB clients, and indexes.

        - Loads config.yaml
        - Loads SentenceTransformer embedding model (local dir)
        - Connects to ChromaDB persistent store
        - Loads media_index.json
        - Loads Qwen 2.5-3B using llama-cpp (GGUF file path from config)
        """
        t0 = time.time()
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.cfg = self._load_config(self.project_root)
        self.preset_name = preset_name
        
        # Apply preset configuration
        self._apply_preset()

        # Paths
        self.models_root = Path(self.cfg["paths"]["models"])  # absolute
        self.db_root = Path(self.cfg["paths"]["database"]) / "chromadb"
        self.processed_root = Path(self.cfg["paths"]["processed_json"])  # data/processed

        # Embedding model
        embed_dir_name = self.cfg["models"]["embedding_dir"]
        embed_dir = self.models_root / embed_dir_name
        logger.info("Loading embedding model...")
        self.embed_model = SentenceTransformer(str(embed_dir), trust_remote_code=True)
        self.embed_dim: int = int(getattr(self.embed_model, "get_sentence_embedding_dimension", lambda: 768)())
        logger.info(f"Embedding model ready. dim={self.embed_dim}")

        # ChromaDB
        logger.info("Connecting to ChromaDB...")
        self.chroma = chromadb.PersistentClient(path=str(self.db_root), settings=Settings(anonymized_telemetry=False))
        self.collection = self.chroma.get_or_create_collection(name="error_codes", metadata={"hnsw:space": "cosine"})

        # Media index
        media_index_path = self.processed_root / "media_index.json"
        self.media_index: Dict[str, List[Dict[str, Any]]] = {}
        if media_index_path.exists():
            try:
                self.media_index = json.loads(media_index_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not load media_index.json: {e}")
        logger.info(f"Media index loaded ({len(self.media_index)} codes)")

        # LLM (config-driven directory)
        llm_dir_name = self.cfg["models"]["llm_dir"]
        self.llm_path = self.models_root / llm_dir_name / self.cfg["models"]["llm_gguf_filename"]
        self.llm: Optional[Llama] = None
        if Llama is not None and self.llm_path.exists():
            try:
                logger.info(f"Loading LLM via llama-cpp: {self.llm_path}")
                # Use llm settings from config if available
                llm_cfg = self.cfg.get("llm", {}) or {}
                n_ctx = int(llm_cfg.get("n_ctx", 512))
                n_threads = int(llm_cfg.get("n_threads", 8))
                n_batch = int(llm_cfg.get("n_batch", 128))
                self.llm = Llama(
                    model_path=str(self.llm_path), 
                    n_ctx=n_ctx, 
                    n_threads=n_threads,
                    n_batch=n_batch
                )
            except Exception as e:
                logger.warning(f"Failed to load GGUF LLM: {e}")
        else:
            logger.warning("llama-cpp not available or model file missing; answering will be skipped.")

        logger.info(f"QueryEngine initialized in {time.time() - t0:.2f}s")

    def _apply_preset(self) -> None:
        """Apply preset configuration to override default settings."""
        if "presets" in self.cfg and self.preset_name in self.cfg["presets"]:
            preset = self.cfg["presets"][self.preset_name]
            logger.info(f"Applying preset: {preset['name']} - {preset['description']}")
            
            # Override LLM settings
            if "llm" in preset:
                self.cfg["llm"].update(preset["llm"])
            
            # Override RAG settings
            if "rag" in preset:
                self.cfg["rag"].update(preset["rag"])
        else:
            logger.warning(f"Preset '{self.preset_name}' not found, using default configuration")

    @staticmethod
    def _load_config(project_root: Path) -> Dict[str, Any]:
        cfg_path = project_root / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def extract_error_code(self, query: str) -> Optional[str]:
        """Detects error codes like BMCR01, BMMJ18, PRC3, 103, etc., and normalizes.

        Rules:
          - Pattern: (BMCR|BMMJ|BMMI|BMAT|PRC)?(\d{2,3})
          - If category is missing and 2-3 digit number is present, return number as-is (e.g., "103")
        """
        pattern = re.compile(r"\b(?:(BMCR|BMMJ|BMMI|BMAT|PRC)\s*)?(\d{2,3})\b", re.IGNORECASE)
        m = pattern.search(query)
        if not m:
            return None
        cat = m.group(1)
        num = m.group(2)
        if cat:
            code = f"{cat.upper()}{num.zfill(2)}"
        else:
            # If just digits, return as-is (e.g., 103 or 03)
            code = num
        return code

    def extract_category(self, query: str) -> Optional[str]:
        """Maps human terms to category codes.

        critical -> BMCR
        mechanical -> BMMJ
        minor -> BMMI
        attachment/tooling -> BMAT
        procedure/setup -> PRC
        """
        q = query.lower()
        if any(k in q for k in ["critical", "emergency"]):
            return "BMCR"
        if any(k in q for k in ["mechanical", "mechanic", "motion"]):
            return "BMMJ"
        if "minor" in q:
            return "BMMI"
        if any(k in q for k in ["attachment", "tool", "tooling"]):
            return "BMAT"
        if any(k in q for k in ["procedure", "setup", "parameter", "prc"]):
            return "PRC"
        return None

    def detect_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent and extract helpful signals.

        Returns dict like {"type": "exact_code", "code": "BMCR01", "keywords": [..]}
        """
        code = self.extract_error_code(query)
        category = self.extract_category(query)
        q = query.lower()
        intent_type = "symptom"
        if code:
            intent_type = "exact_code"
        elif category:
            intent_type = "category"
        if any(k in q for k in ["how", "fix", "reset", "resolve"]):
            intent_type = "how_to"
        if any(k in q for k in ["where", "location", "locate", "position"]):
            intent_type = "location"
        if any(k in q for k in ["wiring", "electrical", "circuit", "connector"]):
            intent_type = "wiring"
        keywords = re.findall(r"[a-zA-Z]{3,}", q)
        out = {"type": intent_type, "code": code, "category": category, "keywords": keywords}
        logger.info(f"Intent: {out}")
        return out

    def _query_chroma(self, query_emb: List[float], n: int, where: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        res = self.collection.query(query_embeddings=[query_emb], n_results=n, where=where or {})
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        results: List[SearchResult] = []
        for rid, rdoc, dist, meta in zip(ids, docs, dists, metas):
            title = (meta or {}).get("title", "")
            results.append(
                SearchResult(
                    code=rid,
                    title=title,
                    similarity=max(0.0, 1 - float(dist)),
                    document=rdoc,
                    metadata=meta or {},
                )
            )
        return results

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Three-tier hybrid search: exact match, category-filtered, semantic.

        Returns {"search_type": str, "results": [SearchResult as dict...]}
        """
        intent = self.detect_intent(query)

        # Tier 1: exact code match
        if intent.get("code"):
            code = intent["code"]
            try:
                res = self.collection.get(ids=[code])
                ids = res.get("ids", [])
                if ids:
                    doc = res.get("documents", [""])[0]
                    meta = res.get("metadatas", [{}])[0]
                    title = (meta or {}).get("title", "")
                    return {
                        "search_type": "exact_match",
                        "results": [
                            {
                                "code": code,
                                "title": title,
                                "similarity": 1.0,
                                "document": doc,
                                "metadata": meta or {},
                            }
                        ],
                        "intent": intent,
                    }
            except Exception as e:
                logger.warning(f"Exact match fetch failed: {e}")

        # Prepare embedding for vector search
        try:
            q_emb = self.embed_model.encode(query)
            q_emb_list = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return {"search_type": "error", "results": [], "intent": intent}

        # Tier 2: category-filtered vector search
        if intent.get("category"):
            try:
                results = self._query_chroma(q_emb_list, n_results, where={"category": intent["category"]})
                if results:
                    return {
                        "search_type": "category_semantic",
                        "results": [r.__dict__ for r in results],
                        "intent": intent,
                    }
            except Exception as e:
                logger.warning(f"Category semantic search failed: {e}")

        # Tier 3: pure semantic
        try:
            results = self._query_chroma(q_emb_list, n_results)
            return {
                "search_type": "semantic",
                "results": [r.__dict__ for r in results],
                "intent": intent,
            }
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"search_type": "error", "results": [], "intent": intent}

    def filter_media_by_intent(self, media_list: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Order media based on query intent.

        - wiring -> prefer items tagged with wiring/electrical
        - location -> prefer items tagged with location/position
        - how_to -> prefer GIFs (animated)
        Fallback: original order
        """
        if not media_list:
            return []

        t = intent.get("type")
        def score(m: Dict[str, Any]) -> int:
            s = 0
            tags = [t.lower() for t in (m.get("tags") or [])]
            path = (m.get("path") or m.get("organized_path") or "").lower()
            if t == "wiring":
                if any(k in tags for k in ["wiring", "electrical", "diagram", "circuit"]):
                    s += 5
            if t == "location":
                if any(k in tags for k in ["location", "position", "panel", "button"]):
                    s += 5
            if t == "how_to":
                if path.endswith(".gif"):
                    s += 4
            return s

        return sorted(media_list, key=score, reverse=True)

    def build_context(self, search_results: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """Build the RAG context string from top search results.

        Includes code, title, relevant sections, and media captions. Aim for <= 2000 tokens.
        """
        results = search_results.get("results", [])
        lines: List[str] = []
        
        for r in results:
            meta = r.get("metadata", {})
            code = meta.get("code") or r.get("code") or ""
            title = meta.get("title") or r.get("title") or ""
            
            # Parse the document to extract structured information
            doc = r.get("document", "")
            
            lines.append(f"=== ERROR CODE: {code} ===")
            lines.append(f"Title: {title}")
            
            # Try to extract alarm message, description, and remedies from the document
            if "Alarm Message" in doc:
                alarm_start = doc.find("Alarm Message") + len("Alarm Message")
                alarm_end = doc.find("\n", alarm_start)
                alarm_msg = doc[alarm_start:alarm_end].strip()
                lines.append(f"Alarm Message: {alarm_msg}")
            
            if "Remedies:" in doc:
                remedy_start = doc.find("Remedies:") + len("Remedies:")
                remedy_end = doc.find("\nVisual references", remedy_start)
                if remedy_end == -1:
                    remedy_end = len(doc)
                remedies = doc[remedy_start:remedy_end].strip()
                lines.append(f"Remedies: {remedies}")
            
            if "Visual references:" in doc:
                visual_start = doc.find("Visual references:") + len("Visual references:")
                visual_end = len(doc)
                visuals = doc[visual_start:visual_end].strip()
                lines.append(f"Visual References: {visuals}")
            
            lines.append("")  # Empty line between results
        
        context = "\n".join(lines).strip()
        # Trim hard if somehow too long
        return context[:8000]

    def generate_answer(self, query: str, context: str) -> str:
        """Generate a concise technical answer using Qwen (llama-cpp).

        If LLM is unavailable, return an offline fallback message.
        """
        # Log context for debugging
        logger.info(f"Context length: {len(context)} chars")
        logger.info(f"Context preview: {context[:200]}...")
        
        prompt = (
            "You are a technical support assistant for machine error codes.\n\n"
            "IMPORTANT: Use ONLY the information provided in the context below. Do not make up or guess information.\n\n"
            "Context from knowledge base:\n"
            f"{context}\n\n"
            f"User question: {query}\n\n"
            "Based on the context above, provide a clear answer that includes:\n"
            "1. The exact error code name from the context\n"
            "2. The exact alarm message from the context\n"
            "3. The exact reasons and remedies from the context\n"
            "4. Any relevant notes from the context\n\n"
            "If the context doesn't contain the requested information, say so clearly.\n"
            "Keep answer under 200 words. Be specific and actionable.\n\n"
            "Answer:"
        )

        if not self.llm:
            logger.warning("LLM not available; returning fallback answer.")
            return "Model unavailable offline. Use the context above to assist the user."

        try:
            llm_cfg = self.cfg.get("llm", {}) or {}
            out = self.llm.create_completion(
                prompt=prompt,
                max_tokens=int(llm_cfg.get("max_tokens", 100)),
                temperature=float(llm_cfg.get("temperature", 0.1)),
                top_p=float(llm_cfg.get("top_p", 0.9)),
                stop=["<|im_end|>", "\n\nUser:", "##", "<|im_start|>"]
            )
            text = (out.get("choices", [{}])[0].get("text") or "").strip()
            logger.info(f"LLM response: {text[:200]}...")
            return text
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return "Unable to generate answer at this time."

    def query(self, user_question: str, n_results: int = 3) -> Dict[str, Any]:
        """Full query pipeline.

        1) Detect intent
        2) Search via hybrid strategy
        3) Select and filter media by intent
        4) Build context
        5) Generate answer
        6) Return structured response with timing
        """
        t0 = time.time()
        intent = self.detect_intent(user_question)
        search_results = self.search(user_question, n_results=n_results)

        # collect media for the best result code if any
        media: List[Dict[str, Any]] = []
        codes: List[str] = []
        for r in search_results.get("results", []):
            c = r.get("metadata", {}).get("code") or r.get("code")
            if c:
                codes.append(c)
        if codes:
            media = self.media_index.get(codes[0], [])
        media_filtered = self.filter_media_by_intent(media, intent)

        context = self.build_context(search_results, intent)
        answer = self.generate_answer(user_question, context)

        # Estimate confidence as average similarity if available
        sims = [float(r.get("similarity", 0.0)) for r in search_results.get("results", [])]
        confidence = float(sum(sims) / len(sims)) if sims else 0.0

        resp = {
            "question": user_question,
            "answer": answer,
            "error_codes": codes[:3],
            "search_type": search_results.get("search_type"),
            "confidence": round(confidence, 4),
            "media": [
                {
                    "path": m.get("organized_path") or m.get("path"),
                    "type": m.get("type"),
                    "caption": m.get("caption"),
                    "thumbnail": str(Path(self.cfg["paths"]["media"]) / "thumbnails" / ((Path(m.get("path") or Path(m.get("organized_path") or "")).stem) + "_thumb" + (Path(m.get("path") or Path(m.get("organized_path") or "")).suffix)))
                }
                for m in media_filtered[:5]
            ],
            "response_time": round(time.time() - t0, 3),
        }
        return resp


def _print_result(name: str, result: Dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    print(f"Q: {result['question']}")
    print(f"Type: {result.get('search_type')} | Codes: {result.get('error_codes')} | Confidence: {result.get('confidence')}")
    print(f"Answer: {result['answer'][:400]}")


if __name__ == "__main__":
    try:
        engine = QueryEngine()
        tests = [
            ("exact_code", "BMCR01 alarm description and fix"),
            ("category", "critical alarm emergency stop issue"),
            ("symptom", "servo not in virtual mode, not homing"),
            ("how_to", "how to reset BMMJ04 error"),
            ("wiring", "wiring diagram for emergency stop button location"),
        ]
        for name, q in tests:
            res = engine.query(q, n_results=3)
            _print_result(name, res)
    except Exception as e:
        logger.error(f"QueryEngine failed: {e}")
        sys.exit(1)
