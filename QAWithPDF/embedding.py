from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("Initializing Gemini embedding model")

        # Configure settings instead of ServiceContext
        Settings.llm = model
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        logging.info("Creating VectorStoreIndex")
        index = VectorStoreIndex.from_documents(document)

        # Persist storage context
        index.storage_context.persist()

        logging.info("Indexing completed, setting up query engine")
        query_engine = index.as_query_engine()
        return query_engine

    except Exception as e:
        raise customexception(e, sys)
