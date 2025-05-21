"""
Command-line entry point for the AI Core package.
"""

import argparse
import logging
import uvicorn
import sys
import os
from typing import Optional

from ai_core.database import QdrantDB
from ai_core.models import EmbeddingModel
from ai_core.processors import load_qa_into_qdrant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.
    
    Args:
        host (str): Host to bind to
        port (int): Port to bind to
        reload (bool): Whether to enable auto-reload
    """
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "ai_core.api.app:app",
        host=host,
        port=port,
        reload=reload
    )


def load_data(
    file_path: str,
    collection_name: str = "qa_collection",
    db_url: Optional[str] = None
):
    """
    Load data from a JSON file into Qdrant.
    
    Args:
        file_path (str): Path to the JSON file
        collection_name (str): Name of the collection
        db_url (Optional[str]): URL of the Qdrant server
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
        
    # Check if file is a JSON file
    if not file_path.endswith('.json'):
        logger.error(f"File is not a JSON file: {file_path}")
        sys.exit(1)
        
    # Initialize database client
    db = QdrantDB(url=db_url) if db_url else QdrantDB()
    
    # Initialize embedding model
    model = EmbeddingModel()
    
    # Load data into Qdrant
    try:
        count, info = load_qa_into_qdrant(
            json_file_path=file_path,
            db=db,
            model=model,
            collection_name=collection_name
        )
        
        logger.info(f"Successfully loaded {count} QA pairs into collection {collection_name}")
        logger.info(f"Collection info: {info}")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point for the AI Core package."""
    parser = argparse.ArgumentParser(description="AI Core CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load data from a JSON file into Qdrant")
    load_parser.add_argument("file", type=str, help="Path to the JSON file")
    load_parser.add_argument("--collection", type=str, default="qa_collection", help="Name of the collection")
    load_parser.add_argument("--db-url", type=str, help="URL of the Qdrant server")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "api":
        start_api(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "load":
        load_data(file_path=args.file, collection_name=args.collection, db_url=args.db_url)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()