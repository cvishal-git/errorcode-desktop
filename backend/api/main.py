from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.core import init_engine, STATE
from api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Startup: Initialize engine'''
    print('🚀 Starting ErrorCodeQA backend...')

    # Set project root (3 levels up from api/main.py)
    project_root = Path(__file__).parent.parent.parent
    STATE.project_root = project_root
    
    # Mount media files for serving images
    media_path = project_root / 'data' / 'media'
    if media_path.exists():
        app.mount('/media', StaticFiles(directory=str(media_path)), name='media')
        print(f'📁 Serving media from: {media_path}')

    # Initialize engine with balanced preset
    status = init_engine('balanced')
    print(f'Engine status: {status}')

    yield

    print('🛑 Shutting down...')

app = FastAPI(
    title='ErrorCodeQA API',
    description='Offline error code lookup system',
    lifespan=lifespan
)

# CORS for Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:*', 'http://127.0.0.1:*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Include routes
app.include_router(router)

@app.get('/')
async def root():
    return {
        'message': 'ErrorCodeQA API',
        'status': STATE.status,
        'preset': STATE.current_preset
    }
