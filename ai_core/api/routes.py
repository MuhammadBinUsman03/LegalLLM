"""
API routes for the AI Core package.
"""

import os
import uuid
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any

from database import QdrantDB
from models import EmbeddingModel
from processors import load_qa_into_qdrant

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1")

# Initialize embedding model and database client
import os
embedding_model = EmbeddingModel()
qdrant_db = QdrantDB(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

# Track background tasks
background_tasks_status = {}


async def process_file_in_background(
    file_path: str,
    collection_name: str,
    task_id: str,
    db_url: Optional[str] = None
):
    """
    Process a file in the background and update task status.
    
    Args:
        file_path (str): Path to the uploaded file
        collection_name (str): Name of the Qdrant collection
        task_id (str): ID of the background task
        db_url (Optional[str]): URL of the Qdrant server
    """
    try:
        # Update status to processing
        background_tasks_status[task_id] = {"status": "processing", "progress": 0}
        
        # Initialize database client if URL provided
        db = QdrantDB(url=db_url) if db_url else qdrant_db
        
        # Process the file
        count, info = load_qa_into_qdrant(
            json_file_path=file_path,
            db=db,
            model=embedding_model,
            collection_name=collection_name
        )
        
        # Update status to completed
        background_tasks_status[task_id] = {
            "status": "completed",
            "vectors_count": count,
            "collection_info": {
                "name": info.name,
                "vectors_count": info.vectors_count
            }
        }
        
        # Delete temporary file
        os.unlink(file_path)
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        background_tasks_status[task_id] = {"status": "failed", "error": str(e)}
        # Delete temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.post("/vectordb/load")
async def load_json_to_vectordb(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form("qa_collection"),
    db_url: Optional[str] = Form(None)
):
    """
    Upload a JSON file and load QA pairs into Qdrant vector database.
    
    Args:
        background_tasks: BackgroundTasks for async processing
        file: The uploaded JSON file
        collection_name: Name for the Qdrant collection
        db_url: Optional URL for the Qdrant server
        
    Returns:
        JSONResponse: Task ID and status
    """
    # Validate file type
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Write uploaded file content to the temporary file
        with open(temp_file_path, 'wb') as f:
            f.write(await file.read())
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Start background task
        background_tasks.add_task(
            process_file_in_background,
            temp_file_path,
            collection_name,
            task_id,
            db_url
        )
        
        # Set initial task status
        background_tasks_status[task_id] = {"status": "queued"}
        
        return JSONResponse(
            status_code=202,
            content={
                "task_id": task_id,
                "status": "queued",
                "message": "File upload successful. Processing started in the background."
            }
        )
        
    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/vectordb/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a background task.
    
    Args:
        task_id (str): ID of the task
        
    Returns:
        Dict[str, Any]: Task status and details
    """
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found")
    
    return background_tasks_status[task_id]


@router.get("/vectordb/collections")
async def list_collections(db_url: Optional[str] = None):
    """
    List all collections in the Qdrant database.
    
    Args:
        db_url (Optional[str]): URL of the Qdrant server
        
    Returns:
        Dict[str, Any]: List of collections
    """
    try:
        # Use the provided DB URL or the default one
        db = QdrantDB(url=db_url) if db_url else qdrant_db
        
        # Get collections
        collections = db.client.get_collections().collections
        
        return {
            "collections": [
                {"name": collection.name} for collection in collections
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@router.get("/vectordb/collection/{collection_name}")
async def get_collection_info(collection_name: str, db_url: Optional[str] = None):
    """
    Get information about a specific collection.
    
    Args:
        collection_name (str): Name of the collection
        db_url (Optional[str]): URL of the Qdrant server
        
    Returns:
        Dict[str, Any]: Collection information
    """
    try:
        # Use the provided DB URL or the default one
        db = QdrantDB(url=db_url) if db_url else qdrant_db
        
        # Get collection info
        info = db.get_collection_info(collection_name)
        
        return {
            "name": info.name,
            "vectors_count": info.vectors_count,
            "status": info.status
        }
        
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")


@router.post("/vectordb/search/{collection_name}")
async def search_collection(
    collection_name: str,
    query: Dict[str, Any],
    db_url: Optional[str] = None
):
    """
    Search for similar vectors in a collection.
    
    Args:
        collection_name (str): Name of the collection
        query (Dict[str, Any]): Query containing the text to search for
        db_url (Optional[str]): URL of the Qdrant server
        
    Returns:
        Dict[str, Any]: Search results
    """
    try:
        # Check if query text is provided
        if "text" not in query:
            raise HTTPException(status_code=400, detail="Query text is required")
        
        # Get query parameters
        text = query["text"]
        limit = query.get("limit", 3)
        
        # Use the provided DB URL or the default one
        db = QdrantDB(url=db_url) if db_url else qdrant_db
        
        # Generate embedding for the query text
        query_vector = embedding_model.get_embedding(text)
        
        # Search the collection
        results = db.search(collection_name, query_vector, limit=limit)
        
        # Format the response
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "score": hit.score,
                "question": hit.payload.get("question"),
                "answer": hit.payload.get("answer"),
                "id": hit.payload.get("id")
            })
        
        return {"results": formatted_results}
        
    except Exception as e:
        logger.error(f"Error searching collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching collection: {str(e)}")