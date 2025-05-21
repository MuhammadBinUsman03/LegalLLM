"""
Data processing utilities for QA data.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from models import EmbeddingModel
from database import QdrantDB

logger = logging.getLogger(__name__)


def load_qa_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load and process QA data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Dict[str, str]]: List of processed QA pairs
    """
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    
    # Process the nested format into a simpler structure
    processed_data = []
    for item in raw_data:
        # Extract question (prompt) from user content
        question = ""
        for prompt_item in item.get("prompt", []):
            if prompt_item.get("role") == "user":
                question = prompt_item.get("content", "")
                break
        
        # Extract answer (completion) from assistant content
        answer = ""
        for completion_item in item.get("completion", []):
            if completion_item.get("role") == "assistant":
                answer = completion_item.get("content", "")
                break
        
        # Add to processed data if both question and answer exist
        if question and answer:
            processed_data.append({
                "question": question,
                "answer": answer
            })
    
    logger.info(f"Processed {len(processed_data)} valid QA pairs from {len(raw_data)} raw records")
    return processed_data


def process_and_upload_data(
    db: QdrantDB, 
    data: List[Dict[str, str]], 
    model: EmbeddingModel, 
    collection_name: str,
    batch_size: int = 100,
    show_progress: bool = True
) -> int:
    """
    Process QA data and upload to Qdrant.
    
    Args:
        db (QdrantDB): Qdrant database client
        data (List[Dict[str, str]]): List of QA pairs
        model (EmbeddingModel): Embedding model
        collection_name (str): Name of the collection
        batch_size (int): Size of batches for uploading
        show_progress (bool): Whether to show progress bar
        
    Returns:
        int: Number of QA pairs uploaded
    """
    vectors = []
    payloads = []
    total_uploaded = 0
    
    logger.info("Processing and embedding QA pairs...")
    
    # Use tqdm for progress tracking if requested
    data_iterator = tqdm(data) if show_progress else data
    
    for i, qa in enumerate(data_iterator):
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        
        # Text to embed - using only the question for search efficiency
        text_to_embed = question
        
        # Create embedding
        embedding = model.get_embedding(text_to_embed)
        
        # Create payload with metadata
        payload = {
            "question": question,
            "answer": answer,
            "id": i + total_uploaded
        }
        
        vectors.append(embedding)
        payloads.append(payload)
        
        # Upload in batches to avoid memory issues
        if len(vectors) >= batch_size:
            db.upload_batch(collection_name, vectors, payloads, start_id=total_uploaded)
            total_uploaded += len(vectors)
            vectors = []
            payloads = []
    
    # Upload any remaining items
    if vectors:
        db.upload_batch(collection_name, vectors, payloads, start_id=total_uploaded)
        total_uploaded += len(vectors)
    
    logger.info(f"Uploaded {total_uploaded} QA pairs to Qdrant.")
    return total_uploaded


def load_qa_into_qdrant(
    json_file_path: str, 
    db: QdrantDB,
    model: Optional[EmbeddingModel] = None,
    collection_name: str = "qa_collection",
    vector_size: int = 384,
    show_progress: bool = True
) -> Tuple[int, Dict]:
    """
    Load QA data into Qdrant.
    
    Args:
        json_file_path (str): Path to the JSON file
        db (QdrantDB): Qdrant database client
        model (Optional[EmbeddingModel]): Embedding model, created if None
        collection_name (str): Name of the collection
        vector_size (int): Size of the embedding vectors
        show_progress (bool): Whether to show progress bar
        
    Returns:
        Tuple[int, Dict]: Number of QA pairs uploaded and collection info
    """
    # Load data
    data = load_qa_data(json_file_path)
    logger.info(f"Loaded {len(data)} QA pairs from {json_file_path}")
    
    # Initialize embedding model if not provided
    if model is None:
        logger.info("Initializing embedding model...")
        model = EmbeddingModel()
    
    # Create collection
    db.create_collection(collection_name, vector_size=vector_size)
    
    # Process and upload data
    total_uploaded = process_and_upload_data(
        db, data, model, collection_name, show_progress=show_progress
    )
    
    # Get collection info
    collection_info = db.get_collection_info(collection_name)
    
    return total_uploaded, collection_info