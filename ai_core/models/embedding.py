"""
Embedding model implementation using Hugging Face transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    """Class to create embeddings using a pre-trained model."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Hugging Face model name
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def get_embedding(self, text):
        """
        Generate embedding for a given text.
        
        Args:
            text (str): The text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling to get sentence embedding
        attention_mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        
        # Convert to numpy and return
        return mean_pooled[0].cpu().numpy()