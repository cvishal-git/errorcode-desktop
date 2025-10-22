#!/usr/bin/env python3
"""
ErrorCodeQA Backend Server Runner

Runs the FastAPI application with command line arguments.
"""

import argparse
from pathlib import Path
import uvicorn


def main():
    parser = argparse.ArgumentParser(description='ErrorCodeQA Backend Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind server to')

    args = parser.parse_args()

    print(f'Starting ErrorCodeQA backend on {args.host}:{args.port}')

    uvicorn.run(
        'api.main:app',
        host=args.host,
        port=args.port,
        log_level='info',
        reload=False
    )


if __name__ == '__main__':
    main()
