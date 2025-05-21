"""
Qdrant vector database client implementation.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QdrantDB:
    """Class to interact with Qdrant vector database."""
    
    def __init__(self, url="http://localhost:6333"):
        """
        Initialize the Qdrant client.
        
        Args:
            url (str): URL of the Qdrant server
        """
        self.client = QdrantClient(url=url)
        
    def create_collection(self, collection_name: str, vector_size: int = 384):
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name (str): Name of the collection
            vector_size (int): Size of the vectors to be stored
        """
        # Check if collection exists and delete if it does
        try:
            self.client.get_collection(collection_name)
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass
        
        # Create new collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Created collection: {collection_name}")
        
    def upload_batch(self, collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]], start_id: int = 0):
        """
        Upload a batch of vectors and payloads to Qdrant.
        
        Args:
            collection_name (str): Name of the collection
            vectors (List[List[float]]): List of embedding vectors
            payloads (List[Dict[str, Any]]): List of payloads (metadata)
            start_id (int): Starting ID for the batch
        """
        ids = list(range(start_id, start_id + len(vectors)))
        self.client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            )
        )
        
    def search(self, collection_name: str, query_vector: List[float], limit: int = 3):
        """
        Search for similar vectors in the collection.
        
        Args:
            collection_name (str): Name of the collection
            query_vector (List[float]): Query embedding vector
            limit (int): Maximum number of results to return
            
        Returns:
            List: List of search results
        """
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
    def get_collection_info(self, collection_name: str):
        """
        Get information about a collection.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            Dict: Collection information
        """
        return self.client.get_collection(collection_name)