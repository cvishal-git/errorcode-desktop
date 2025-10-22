# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for ErrorCodeQA Backend

This bundles the FastAPI backend with all dependencies into a standalone executable.
Supports Windows packaging for distribution.
"""

import sys
from pathlib import Path

block_cipher = None

# Collect all backend files
backend_files = [
    ('api/*.py', 'api'),
    ('engine/*.py', 'engine'),
    ('utils/*.py', 'utils'),
]

# Hidden imports for ML libraries that PyInstaller might miss
hidden_imports = [
    # FastAPI and web framework
    'uvicorn',
    'fastapi',
    'starlette',
    'pydantic',
    'multipart',
    
    # ML and AI
    'llama_cpp',
    'sentence_transformers',
    'transformers',
    'torch',
    'chromadb',
    
    # ChromaDB dependencies
    'chromadb.config',
    'chromadb.api',
    'chromadb.db',
    'onnxruntime',
    'posthog',
    
    # Sentence transformers
    'tokenizers',
    'huggingface_hub',
    
    # Utilities
    'yaml',
    'pyyaml',
    'einops',
    'numpy',
    'PIL',
    'tqdm',
]

a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=[],
    datas=backend_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'notebook',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Show console for debugging; set False for production
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend',
)

