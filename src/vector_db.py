import os
import logging
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeProductDB:
    def __init__(self, api_key: str, cloud: str = "aws", region: str = "us-east-1", index_name: str = "product-recommendations"):
        """
        Initialize the Pinecone vector database wrapper.
        """
        self.index_name = index_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # Loads the embedding model

        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=api_key)
            self._ensure_index_exists(cloud, region)
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"❌ Pinecone connection failed: {e}")
            raise e

    def _ensure_index_exists(self, cloud: str, region: str):
        """Checks if index exists, creates it if not."""
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            logger.info(f"⚙️ Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )

    def search(self, query: str, top_k: int = 5):
        """
        Converts query to vector and searches Pinecone.
        """
        try:
            # 1. Convert text to vector
            query_vector = self.model.encode([query]).tolist()[0]
            
            # 2. Query Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {"matches": []}