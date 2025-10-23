#!/usr/bin/env python3
"""
ErrorCodeQA Backend Server Runner

Runs the FastAPI application with command line arguments.
"""

import argparse
import sys
import io
import os
from pathlib import Path
import uvicorn


def main():
    # Disable all network calls for offline operation
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    if getattr(sys, 'frozen', False):
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
    
    parser = argparse.ArgumentParser(description='ErrorCodeQA Backend Server')
    parser.add_argument('--port', type=int, default=48127, help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind server to')

    args = parser.parse_args()

    print(f'Starting ErrorCodeQA backend on {args.host}:{args.port}')

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["use_colors"] = False
    log_config["formatters"]["access"]["use_colors"] = False

    uvicorn.run(
        'api.main:app',
        host=args.host,
        port=args.port,
        log_level='info',
        reload=False,
        log_config=log_config
    )


if __name__ == '__main__':
    main()
