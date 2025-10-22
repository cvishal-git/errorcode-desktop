#!/usr/bin/env python3
"""
Enhanced Core Engine for Offline RAG-based Error Code QA System.

Features:
- Better prompting for technical answers
- Proper preset switching with LLM reload
- Error code detection and direct lookup
- Media display with proper paths
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Local imports
from .query_engine import QueryEngine

# Configure logging
logger = logging.getLogger("app")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class AppState:
    """Application state management."""
    engine: Optional[QueryEngine] = None
    status: str = "Initializing..."
    model_info: Dict[str, Any] = field(default_factory=dict)
    current_preset: str = "balanced"
    error_code_cache: Dict[str, Dict] = field(default_factory=dict)
    project_root: Optional[Path] = None


STATE = AppState()


def load_config(project_root: Path) -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    cfg_path = project_root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_error_code_cache(project_root: Path) -> Dict[str, Dict]:
    """Load all error codes into memory for fast lookup."""
    cache = {}
    processed_dir = project_root / "data" / "processed"

    if not processed_dir.exists():
        return cache

    json_files = list(processed_dir.glob("*.json"))
    for json_file in json_files:
        if json_file.name in ["media_index.json", "processing_log.json", "index.json"]:
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "code" in data:
                    cache[data["code"].upper()] = data
        except Exception as e:
            logger.warning(f"Failed to load {json_file.name}: {e}")

    logger.info(f"Loaded {len(cache)} error codes into cache")
    return cache


def extract_error_codes(query: str) -> List[str]:
    """Extract error codes from query (e.g., BMCR01, BMMJ05)."""
    pattern = r'\b([A-Z]{4}\d{2})\b'
    codes = re.findall(pattern, query.upper())
    return list(set(codes))


def build_enhanced_prompt(query: str, context: str, error_code: Optional[str] = None) -> str:
    """Build optimized prompt for technical troubleshooting."""

    if error_code:
        prompt = f"""You are a technical support assistant for industrial machinery.

CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Give a clear, step-by-step answer
- Start with what the error means
- List the most likely cause first
- Use numbered steps for procedures
- Mention if images are available
- Be concise and practical

ANSWER:"""
    else:
        prompt = f"""You are a technical support assistant for industrial machinery.

CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer clearly and directly
- Use technical language appropriately
- Reference diagrams if mentioned in context
- Be practical and actionable

ANSWER:"""

    return prompt


def check_and_download_models(project_root: Path, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if models exist."""
    models_root = Path(cfg["paths"]["models"])
    messages: List[str] = []

    # Check embedding model
    embed_dir_name = cfg["models"]["embedding_dir"]
    embed_dir = models_root / embed_dir_name
    if not embed_dir.exists():
        messages.append(f"⚠️ Missing embedding model at {embed_dir}")
        messages.append("Run: python src/download_models.py")
        return False, "; ".join(messages)
    else:
        messages.append(f"✅ Embedding model: {embed_dir.name}")

    # Check LLM model
    llm_dir_name = cfg["models"]["llm_dir"]
    gguf = models_root / llm_dir_name / cfg["models"]["llm_gguf_filename"]
    if not gguf.exists():
        messages.append(f"⚠️ Missing LLM model at {gguf}")
        messages.append("Run: python src/download_models.py")
        return False, "; ".join(messages)
    else:
        messages.append(f"✅ LLM model: {gguf.name}")

    return True, "; ".join(messages)


def startup_checks(project_root: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """Perform comprehensive startup checks."""
    cfg = load_config(project_root)
    models_root = Path(cfg["paths"]["models"])
    db_root = Path(cfg["paths"]["database"]) / "chromadb"
    processed_root = Path(cfg["paths"]["processed_json"])

    ok = True
    messages: List[str] = []

    # Check models
    models_ok, model_msg = check_and_download_models(project_root, cfg)
    if not models_ok:
        ok = False
        messages.append(model_msg)
    else:
        messages.append(model_msg)

    # Check database (optional for basic functionality)
    if not db_root.exists():
        messages.append(f"⚠️ Missing ChromaDB at {db_root} (RAG will be unavailable)")
        # Don't set ok = False - we can still work with cached error codes
    else:
        messages.append(f"✅ Database: {db_root.name}")

    # Check media index
    media_index = processed_root / "media_index.json"
    if media_index.exists():
        messages.append(f"✅ Media index: {media_index.name}")

    # Build model info
    embed_dir_name = cfg["models"]["embedding_dir"]
    embed_dir = models_root / embed_dir_name
    llm_dir_name = cfg["models"]["llm_dir"]
    gguf = models_root / llm_dir_name / cfg["models"]["llm_gguf_filename"]

    model_info: Dict[str, Any] = {
        "embedding_model": {
            "name": cfg["models"]["embedding_model"],
            "path": str(embed_dir),
            "dimensions": cfg["models"]["embedding_dimensions"],
        },
        "llm_model": {
            "name": cfg["models"]["llm_name"],
            "path": str(gguf),
            "parameters": cfg["models"]["llm_parameters"],
            "quantization": cfg["models"]["llm_quantization"],
        }
    }

    status = "; ".join(messages) if messages else "All systems ready"
    return ok, status, model_info


def init_engine(preset_name: str = "balanced") -> str:
    """Initialize the QueryEngine (optional - only for RAG queries)."""
    try:
        if not STATE.project_root:
            STATE.project_root = Path(__file__).resolve().parents[2]

        logger.info(f"Initializing QueryEngine with preset: {preset_name}")

        # Perform startup checks on first init
        if not STATE.error_code_cache:
            ok, status, model_info = startup_checks(STATE.project_root)
            STATE.model_info = model_info
            STATE.error_code_cache = load_error_code_cache(STATE.project_root)

            if not ok:
                STATE.status = f"Startup checks failed: {status}"
                logger.warning(STATE.status)
                # Don't fail completely - we can still work with cached error codes
                return f"⚠️ Limited mode - RAG unavailable: {status}"

        # Try to create engine with specified preset (optional)
        try:
            STATE.engine = QueryEngine(project_root=STATE.project_root, preset_name=preset_name)
            STATE.status = "Ready"
            STATE.current_preset = preset_name
            logger.info(f"QueryEngine ready with {preset_name} preset")
            return f"✅ System Ready ({preset_name})"
        except Exception as e:
            logger.warning(f"QueryEngine initialization failed: {e}")
            STATE.status = "Limited mode - cached responses only"
            STATE.current_preset = preset_name
            return f"⚠️ Limited mode - cached responses only: {e}"

    except Exception as e:
        STATE.status = f"Engine init error: {e}"
        logger.exception("Engine initialization failed")
        return f"❌ Initialization Error: {e}"


def switch_preset(preset_name: str) -> str:
    """Switch to a different preset configuration."""
    if preset_name == STATE.current_preset:
        return f"Already using {preset_name} preset"

    logger.info(f"Switching preset from {STATE.current_preset} to {preset_name}")
    status = init_engine(preset_name)
    return status


def format_template_response(error_code: str, error_data: Dict[str, Any], query: str = "") -> str:
    """Generate natural language template response from error data without LLM."""
    parts = []

    # Header with error code and alarm
    alarm = error_data.get('alarm_message', 'Unknown alarm')
    category = error_data.get('category', '').upper()

    # Natural language intro based on error type (concise, not redundant)
    if 'BMCR' in category or 'critical' in alarm.lower():
        intro = f"This is a **critical alarm**."
    elif 'emergency' in alarm.lower():
        intro = f"This is an **emergency alarm**."
    elif 'BMMJ' in category or 'mechanical' in alarm.lower():
        intro = f"This is a **mechanical issue**."
    elif 'BMMI' in category or 'minor' in alarm.lower():
        intro = f"This is a **minor alarm**."
    else:
        intro = f"**Error Alert:**"

    parts.append(intro)

    # Add description if available with natural language
    desc = error_data.get('description', '').strip()
    if desc:
        parts.append(f"\n{desc}")

    # Troubleshooting steps with natural language
    steps = error_data.get('reasons_remedies', [])
    if steps:
        # Filter out reference markers like (Emergency Location)
        filtered_steps = [s.strip() for s in steps if s.strip() and not (s.strip().startswith('(') and s.strip().endswith(')'))]

        if not filtered_steps:
            pass  # No valid steps
        else:
            # Add natural transition
            if len(filtered_steps) > 5:
                parts.append(f"\n**Common causes and how to fix them:**")
            elif len(filtered_steps) > 2:
                parts.append(f"\n**Main causes and solutions:**")
            else:
                parts.append(f"\n**How to fix:**")

            # Group steps into cause-solution pairs intelligently
            i = 0
            step_num = 1
            while i < len(filtered_steps):
                step = filtered_steps[i]

                # Check if this looks like a cause (problem description)
                is_cause = any(word in step.lower() for word in [
                    'is pressed', 'is loose', 'is damaged', 'not functioning',
                    'loose connection', 'is not', 'faulty', 'error', 'problem',
                    'button', 'contact', 'wiring'
                ])

                if is_cause and i + 1 < len(filtered_steps):
                    # Next step might be the solution
                    next_step = filtered_steps[i + 1]
                    is_solution = any(word in next_step.lower() for word in [
                        'turn', 'check', 'replace', 'reset', 'refer', 'adjust',
                        'set', 'ensure', 'verify', 'contact', 'reconnect', 'take'
                    ])

                    if is_solution:
                        # Format as natural cause → solution
                        parts.append(f"\n**{step_num}. Problem:** {step}")
                        parts.append(f"   **→ Fix:** {next_step}")
                        i += 2
                        step_num += 1
                        continue

                # Default: list as action item
                parts.append(f"\n**{step_num}.** {step}")
                step_num += 1
                i += 1

    # Media references with natural language (de-duplicated)
    media = error_data.get('media', [])
    if media:
        if len(media) == 1:
            parts.append(f"\n**📸 Visual Guide:**")
        else:
            parts.append(f"\n**📸 Visual Guides ({len(media)} diagrams):**")

        # Collect and de-duplicate captions
        seen_captions = set()
        unique_captions = []
        for m in media[:10]:  # Process up to 10
            caption = m.get('caption', '').strip()
            if caption and caption.lower() not in seen_captions:
                seen_captions.add(caption.lower())
                unique_captions.append(caption)

        if unique_captions:
            for caption in unique_captions:
                parts.append(f"  • {caption}")
        else:
            parts.append(f"  • Wiring diagrams, component locations, and troubleshooting steps")

    # Notes with natural intro
    notes = error_data.get('notes', '').strip()
    if notes:
        parts.append(f"\n**💡 Important Note:**")
        parts.append(notes)

    # No footer here - it will be added by the caller
    return "\n".join(parts)


def is_direct_error_query(query: str) -> bool:
    """Determine if query is a direct error code lookup vs complex question."""
    query_lower = query.lower()

    # Direct lookup patterns
    direct_patterns = [
        r'^what is (error )?[A-Z]{4}\d{2}',
        r'^(error |code )?[A-Z]{4}\d{2}$',
        r'^explain [A-Z]{4}\d{2}',
        r'^tell me about [A-Z]{4}\d{2}',
        r'^info (on|about) [A-Z]{4}\d{2}',
    ]

    for pattern in direct_patterns:
        if re.match(pattern, query, re.IGNORECASE):
            return True

    # Complex query indicators (need LLM)
    complex_keywords = [
        'why', 'how come', 'difference between', 'compare',
        'better than', 'similar to', 'related to',
        'troubleshoot', 'diagnose', 'still not working',
        'tried', 'already', 'but', 'however'
    ]

    if any(keyword in query_lower for keyword in complex_keywords):
        return False

    # Default: if it mentions just one error code, treat as direct
    error_codes = extract_error_codes(query)
    return len(error_codes) == 1


def process_query(user_question: str, preset_name: str = "balanced") -> Tuple[str, List[Tuple[str, str]]]:
    """Process user query with enhanced error code handling."""
    if not user_question.strip():
        return ("Please enter a question.", [])

    # Handle hybrid mode specially (no engine switch needed)
    use_hybrid = (preset_name == "hybrid")

    # Switch preset if needed (skip for hybrid mode)
    if not use_hybrid and STATE.current_preset != preset_name:
        switch_status = switch_preset(preset_name)
        if "Error" in switch_status or "failed" in switch_status:
            return (f"❌ {switch_status}", [])

    if not STATE.engine and not use_hybrid:
        return ("❌ Engine not initialized. Please restart.", [])

    try:
        # Extract error codes from query
        error_codes = extract_error_codes(user_question)
        
        # Handle multiple error codes in the query
        if len(error_codes) > 1:
            logger.info(f"Multiple error codes detected: {error_codes}")
            # Build combined response for all codes
            all_answers = []
            all_media = []
            
            for code in error_codes:
                if code in STATE.error_code_cache:
                    error_data = STATE.error_code_cache[code]
                    alarm = error_data.get('alarm_message', '')
                    
                    # Use instant template for each code
                    answer = format_template_response(code, error_data, user_question)
                    all_answers.append(f"## {code}: {alarm}\n\n{answer}")
                    
                    # Collect media
                    media_items = error_data.get('media', [])
                    for m in media_items:
                        mpath = m.get('path', '')
                        if mpath:
                            full_path = str(STATE.project_root / mpath)
                            caption = m.get('caption', '')
                            all_media.append((full_path, f"{code}: {caption}"))
            
            if all_answers:
                combined_answer = "\n\n---\n\n".join(all_answers)
                combined_answer += "\n\n---\n⚡ *Instant response for multiple error codes*"
                return (combined_answer, all_media)
            else:
                return (f"❌ Error codes {', '.join(error_codes)} not found in database.", [])
        
        # Single error code handling
        error_code = error_codes[0] if error_codes else None

        # Get direct error code data if available
        error_data = None
        if error_code and error_code in STATE.error_code_cache:
            error_data = STATE.error_code_cache[error_code]
            logger.info(f"Direct cache hit for {error_code}")

        # HYBRID MODE: Decide between template (instant) or LLM (intelligent)
        if error_data:
            alarm = error_data.get('alarm_message', '')

            # Check if we should use template (fast) or LLM (intelligent)
            if use_hybrid and is_direct_error_query(user_question):
                # INSTANT TEMPLATE RESPONSE
                logger.info(f"Hybrid mode: Using instant template for {error_code}")
                answer = format_template_response(error_code, error_data, user_question)
                answer = f"## {error_code}: {alarm}\n\n{answer}\n\n---\n⚡ *Instant response - answered in < 1 second*"

            else:
                # LLM RESPONSE (for complex queries or non-hybrid mode)
                if use_hybrid:
                    logger.info(f"Hybrid mode: Using LLM for complex query about {error_code}")

                # Build context from error data
                context_parts = []

                # Basic info
                desc = error_data.get('description', '')

                context_parts.append(f"Error Code: {error_code}")
                context_parts.append(f"Alarm Message: {alarm}")
                if desc:
                    context_parts.append(f"Description: {desc}")

                # Troubleshooting steps
                steps = error_data.get('reasons_remedies', [])
                if steps:
                    context_parts.append("\nReasons and Remedies:")
                    for i, step in enumerate(steps, 1):
                        context_parts.append(f"{i}. {step}")

                # Media information
                media = error_data.get('media', [])
                if media:
                    context_parts.append(f"\nAvailable Visual References ({len(media)} files):")
                    for m in media:
                        caption = m.get('caption', '')
                        mtype = m.get('type', '')
                        if caption:
                            context_parts.append(f"- {caption} ({mtype})")

                # Notes
                notes = error_data.get('notes', '')
                if notes:
                    context_parts.append(f"\nNotes: {notes}")

                context = "\n".join(context_parts)

                # Build better prompt for detailed response
                prompt = f"""You are a technical support assistant for industrial machinery.

CONTEXT:
{context}

USER QUESTION: {user_question}

INSTRUCTIONS:
1. Start by clearly stating what error {error_code} means (use the alarm message)
2. List ALL the reasons (causes) from the context - don't skip any
3. For each reason, provide the corresponding remedy/solution
4. Include all steps in proper order
5. Mention that diagrams and images are available
6. Be detailed and thorough - include ALL information from the context

ANSWER FORMAT:
**Error Description:**
[Explain the error and alarm message]

**Causes and Solutions:**
[List all reasons and remedies with numbers]

**Visual Aids:**
[Mention available images/diagrams]

ANSWER:"""

                # Generate answer using the engine's LLM directly for better control
                if STATE.engine and STATE.engine.llm:
                    try:
                        llm_cfg = STATE.engine.cfg.get("llm", {}) or {}
                        out = STATE.engine.llm.create_completion(
                            prompt=prompt,
                            max_tokens=int(llm_cfg.get("max_tokens", 150)),
                            temperature=float(llm_cfg.get("temperature", 0.15)),
                            top_p=float(llm_cfg.get("top_p", 0.9)),
                            stop=["<|im_end|>", "\n\nUser:", "<|im_start|>"]
                        )
                        answer = (out.get("choices", [{}])[0].get("text") or "").strip()
                    except Exception as e:
                        logger.warning(f"Direct LLM generation failed: {e}, using fallback")
                        answer = STATE.engine.generate_answer(user_question, context)
                else:
                    answer = STATE.engine.generate_answer(user_question, context) if STATE.engine else "Engine not available"

                # Add header
                answer = f"## {error_code}: {alarm}\n\n{answer}"

        else:
            # No direct match, use RAG if available
            if STATE.engine:
                resp = STATE.engine.query(user_question)
                answer = resp.get("answer", "No answer generated.")
            else:
                answer = "RAG system unavailable. Please ask about specific error codes (e.g., BMCR01, BMMJ05) for instant responses."

        # Format media gallery
        media_items = []

        # Get media from error data
        if error_data and error_data.get('media'):
            media_items.extend(error_data['media'])
            logger.info(f"Found {len(media_items)} media items in error data for {error_code}")

        gallery = []
        for item in media_items:
            path = item.get("path", "")
            if path and any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                # Try multiple path resolutions
                abs_path = None

                if path.startswith('/'):
                    # Already absolute
                    abs_path = Path(path)
                else:
                    # Try organized path from media index first
                    if error_code and error_code in STATE.error_code_cache:
                        media_index_path = STATE.project_root / "data" / "processed" / "media_index.json"
                        if media_index_path.exists():
                            try:
                                import json
                                with open(media_index_path, 'r') as f:
                                    media_idx = json.load(f)
                                    if error_code in media_idx:
                                        for m in media_idx[error_code]:
                                            if m.get("original_path", "") == path or m.get("path", "") == path:
                                                org_path = m.get("organized_path", "")
                                                if org_path:
                                                    abs_path = STATE.project_root / org_path
                                                    if abs_path.exists():
                                                        break
                            except:
                                pass

                    # Try path relative to project root (ErrorCodeQA/)
                    if not abs_path or not abs_path.exists():
                        abs_path = STATE.project_root / path

                    # Try path relative to parent directory (one level up)
                    if not abs_path.exists():
                        abs_path = STATE.project_root.parent / path

                if abs_path and abs_path.exists():
                    caption = item.get("caption", "")
                    gallery.append((str(abs_path), caption))
                else:
                    logger.warning(f"Media file not found: {path}")

        # Log image loading (no footer needed for hybrid template mode)
        if gallery:
            logger.info(f"Successfully loaded {len(gallery)} images for display")
        else:
            logger.warning(f"No images found for error code {error_code}")

        return (answer, gallery)

    except Exception as e:
        logger.exception("Query processing failed")
        error_msg = f"❌ **Error Processing Query**\n\n{str(e)}\n\nPlease try rephrasing your question."
        return (error_msg, [])


def handle_chat_query(query: str, preset: str = "balanced") -> Tuple[str, List[Dict[str, Any]]]:
    """Wrapper function for API integration.

    Takes a query string and preset, calls process_query,
    and returns (answer_text, media_list) in API-friendly format.
    """
    answer, gallery = process_query(query, preset)

    # Convert gallery format to API-friendly media list with proper URLs
    media_list = []
    for path, caption in gallery:
        # Convert file system path to URL path
        # Path format: /full/path/to/data/media/CATEGORY/FILE.ext
        # Convert to: http://127.0.0.1:8000/media/CATEGORY/FILE.ext
        
        if '/data/media/' in path:
            # Extract relative path after 'data/media/'
            relative_path = path.split('/data/media/')[-1]
            url_path = f'http://127.0.0.1:8000/media/{relative_path}'
        else:
            # Fallback to original path
            url_path = path
            
        media_list.append({
            'path': url_path,
            'caption': caption,
            'type': 'image' if path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) else 'unknown'
        })

    return answer, media_list
