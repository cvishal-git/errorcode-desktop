from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.core import STATE, init_engine, switch_preset, extract_error_codes, handle_chat_query

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    preset: str = 'balanced'

class QueryResponse(BaseModel):
    answer: str
    media: List[Dict[str, Any]] = []
    processing_time: float = 0.0
    intent: str = ''

@router.post('/api/query', response_model=QueryResponse)
async def query_error_code(request: QueryRequest):
    '''Main chat query endpoint'''
    try:
        # Always use 'hybrid' mode for optimal performance:
        # - Instant template response for direct error code queries
        # - LLM response for complex questions
        preset = 'hybrid'

        # Check if engine is ready (hybrid mode can work without engine for cached queries)
        # Process query using handle_chat_query wrapper with hybrid preset
        answer, media = handle_chat_query(request.query, preset)

        return QueryResponse(
            answer=answer,
            media=media,
            intent='query'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/health')
async def health_check():
    '''Check backend health'''
    return {
        'status': 'ok' if STATE.engine else 'not_ready',
        'engine_status': STATE.status,
        'preset': STATE.current_preset
    }

@router.get('/api/config')
async def get_config():
    '''Get available presets and model info'''
    return {
        'presets': ['instant', 'fast', 'balanced', 'quality', 'hybrid'],
        'current_preset': STATE.current_preset,
        'model': STATE.model_info.get('llm_model', {}).get('name', 'Unknown')
    }

@router.post('/api/preset')
async def change_preset(preset: str):
    '''Switch preset'''
    status = switch_preset(preset)
    return {'status': status, 'current_preset': STATE.current_preset}
