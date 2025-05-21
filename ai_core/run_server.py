#!/usr/bin/env python3
"""
Script to run the FastAPI server locally.
This assumes that a Qdrant server is already running on localhost:6333.
"""

import argparse
import logging
import uvicorn
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run the FastAPI server locally."""
    parser = argparse.ArgumentParser(description="Run the AI Core API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port to bind to (default: 8000)")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333", 
                        help="URL of the Qdrant server (default: http://localhost:6333)")
    parser.add_argument("--reload", action="store_true", 
                        help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set Qdrant URL as environment variable so it can be accessed by the API
    os.environ["QDRANT_URL"] = args.qdrant_url
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Using Qdrant server at {args.qdrant_url}")
    logger.info("Make sure the Qdrant server is running!")
    
    # Start the FastAPI server
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()