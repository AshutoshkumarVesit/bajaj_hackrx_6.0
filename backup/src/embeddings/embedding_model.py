# src/embeddings/embedding_model.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingModel:
    _instance = None
    MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, fast and efficient
    # Alternative models:
    # "all-mpnet-base-v2" = 768 dimensions, better quality
    # "multi-qa-MiniLM-L6-cos-v1" = 384 dimensions, optimized for Q&A

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            print(f"Loading SentenceTransformer model: {cls.MODEL_NAME}")
            try:
                cls._instance.model = SentenceTransformer(cls.MODEL_NAME)
                print(f"✅ SentenceTransformer model loaded successfully")
                print(f"� Model embedding dimension: {cls._instance.model.get_sentence_embedding_dimension()}")
            except Exception as e:
                print(f"❌ Failed to load SentenceTransformer model: {e}")
                raise
        return cls._instance

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings using SentenceTransformer with optimized batch processing.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each as a list of floats)
        """
        if not texts:
            return []
        
        try:
            # For large batches, process in smaller chunks for memory efficiency
            batch_size = 32 if len(texts) > 100 else len(texts)
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Generate embeddings for this batch
                embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,  # Return as numpy arrays
                    normalize_embeddings=True,  # Normalize for better similarity computation
                    show_progress_bar=False,  # Disable progress bar for speed
                    batch_size=16  # Smaller internal batch size for stability
                )
                
                # Convert numpy arrays to lists
                if isinstance(embeddings, np.ndarray):
                    all_embeddings.extend(embeddings.tolist())
                else:
                    all_embeddings.extend([emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings])
            
            return all_embeddings
                
        except Exception as e:
            print(f"Error generating embeddings with SentenceTransformer: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        return self.model.get_sentence_embedding_dimension()
