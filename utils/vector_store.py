"""
Vector store manager voor Gemini embeddings met ChromaDB/Pinecone
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# LangChain imports
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

# ChromaDB
import chromadb
from chromadb.config import Settings

# Gemini
import google.generativeai as genai

# Config
from config import GeminiConfig, VectorStoreConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEmbeddings(Embeddings):
    """
    Custom Gemini embeddings class voor LangChain compatibility
    """
    
    def __init__(self, model_name: str = None):
        genai.configure(api_key=GeminiConfig.API_KEY)
        self.model_name = model_name or GeminiConfig.EMBEDDING_MODEL
        logger.info(f"Initialized Gemini embeddings with model: {self.model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Rate limiting - Gemini has limits
                if i > 0:
                    time.sleep(0.1)  # Small delay between requests
                
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Embedded {i + 1}/{len(texts)} documents")
            
            except Exception as e:
                logger.error(f"Error embedding document {i}: {str(e)}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 768)  # Gemini embedding dimension
        
        logger.info(f"Successfully embedded {len(embeddings)} documents")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            return [0.0] * 768  # Fallback zero vector

class VectorStoreManager:
    """
    Beheer vector databases voor contract embeddings met Gemini
    """
    
    def __init__(self, use_pinecone: bool = False, collection_name: str = None):
        self.use_pinecone = use_pinecone
        self.collection_name = collection_name or VectorStoreConfig.CHROMA_COLLECTION_NAME
        
        # Initialize embeddings
        self.embeddings = GeminiEmbeddings()
        
        # Initialize vector store
        if use_pinecone:
            self._init_pinecone()
        else:
            self._init_chroma()
        
        logger.info(f"VectorStore initialized ({'Pinecone' if use_pinecone else 'ChromaDB'})")
    
    def _init_chroma(self):
        """
        Initialize ChromaDB (local vector store)
        """
        # Ensure directory exists
        chroma_path = Path(VectorStoreConfig.CHROMA_DB_PATH)
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(chroma_path)
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
        
        # Initialize LangChain Chroma wrapper
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
    
    def _init_pinecone(self):
        """
        Initialize Pinecone (cloud vector store)
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError("Pinecone not installed. Run: pip install pinecone-client")
        
        if not VectorStoreConfig.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY required for Pinecone usage")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=VectorStoreConfig.PINECONE_API_KEY,
            environment=VectorStoreConfig.PINECONE_ENVIRONMENT
        )
        
        index_name = VectorStoreConfig.PINECONE_INDEX_NAME
        
        # Create index if not exists
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=VectorStoreConfig.EMBEDDING_DIMENSION,
                metric="cosine"
            )
            logger.info(f"Created Pinecone index: {index_name}")
        
        self.index = pinecone.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """
        Voeg documenten toe aan vector store in batches
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        if self.use_pinecone:
            return self._add_documents_pinecone(documents, batch_size)
        else:
            return self._add_documents_chroma(documents, batch_size)
    
    def _add_documents_chroma(self, documents: List[Document], batch_size: int) -> List[str]:
        """
        Add documents to ChromaDB
        """
        added_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Extract texts and metadatas
                texts = [doc.page_content for doc in batch]
                metadatas = []
                
                # Fix metadata format for ChromaDB
                for doc in batch:
                    metadata = doc.metadata.copy()
                    
                    # Convert lists to strings (ChromaDB doesn't support lists)
                    if 'semantic_tags' in metadata and isinstance(metadata['semantic_tags'], list):
                        metadata['semantic_tags'] = ','.join(metadata['semantic_tags'])
                    
                    # Ensure all values are ChromaDB compatible types
                    cleaned_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            cleaned_metadata[key] = value
                        else:
                            cleaned_metadata[key] = str(value)
                    
                    metadatas.append(cleaned_metadata)
                
                # Generate embeddings
                logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                embeddings = self.embeddings.embed_documents(texts)
                
                # Generate IDs
                batch_ids = [f"doc_{i + j}_{int(time.time())}" for j in range(len(batch))]
                
                # Add to ChromaDB
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=batch_ids
                )
                
                added_ids.extend(batch_ids)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                continue
        
        logger.info(f"Successfully added {len(added_ids)} documents to ChromaDB")
        return added_ids
    
    def _add_documents_pinecone(self, documents: List[Document], batch_size: int) -> List[str]:
        """
        Add documents to Pinecone
        """
        added_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                vectors = []
                batch_ids = []
                
                for j, doc in enumerate(batch):
                    doc_id = f"doc_{i + j}_{int(time.time())}"
                    
                    # Generate embedding
                    embedding = self.embeddings.embed_query(doc.page_content)
                    
                    # Prepare vector
                    vector = {
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            **doc.metadata,
                            "text": doc.page_content[:1000]  # Pinecone metadata size limit
                        }
                    }
                    
                    vectors.append(vector)
                    batch_ids.append(doc_id)
                
                # Upsert to Pinecone
                self.index.upsert(vectors=vectors)
                added_ids.extend(batch_ids)
                
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents to Pinecone")
            
            except Exception as e:
                logger.error(f"Error adding batch to Pinecone: {str(e)}")
                continue
        
        logger.info(f"Successfully added {len(added_ids)} documents to Pinecone")
        return added_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        filter: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Zoek relevante documenten op basis van similarity
        """
        k = k or VectorStoreConfig.DEFAULT_K
        k = min(k, VectorStoreConfig.MAX_K)
        
        logger.debug(f"Similarity search: query='{query[:50]}...', k={k}")
        
        if self.use_pinecone:
            return self._similarity_search_pinecone(query, k, filter, score_threshold)
        else:
            return self._similarity_search_chroma(query, k, filter, score_threshold)
    
    def _similarity_search_chroma(
        self, 
        query: str, 
        k: int, 
        filter: Optional[Dict],
        score_threshold: Optional[float]
    ) -> List[Document]:
        """
        ChromaDB similarity search
        """
        try:
            # Use LangChain wrapper for easier interface
            if filter:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
            else:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )
            
            # Apply score threshold if specified
            if score_threshold:
                results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*2)
                results = [doc for doc, score in results_with_scores if score >= score_threshold]
                results = results[:k]  # Limit to k results
            
            logger.debug(f"Found {len(results)} similar documents")
            return results
        
        except Exception as e:
            logger.error(f"ChromaDB search error: {str(e)}")
            return []
    
    def _similarity_search_pinecone(
        self, 
        query: str, 
        k: int, 
        filter: Optional[Dict],
        score_threshold: Optional[float]
    ) -> List[Document]:
        """
        Pinecone similarity search
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter
            )
            
            documents = []
            for match in results['matches']:
                # Apply score threshold
                if score_threshold and match['score'] < score_threshold:
                    continue
                
                # Reconstruct document
                metadata = match['metadata'].copy()
                text = metadata.pop('text', '')
                
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.debug(f"Pinecone found {len(documents)} similar documents")
            return documents
        
        except Exception as e:
            logger.error(f"Pinecone search error: {str(e)}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        k: int = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """
        Combineer semantic en keyword search voor betere resultaten
        """
        k = k or VectorStoreConfig.DEFAULT_K
        
        logger.debug(f"Hybrid search: semantic_weight={semantic_weight}, keyword_weight={keyword_weight}")
        
        # Semantic search
        semantic_results = self.similarity_search(query, k=k*2)
        
        # Keyword search (simple implementation)
        keyword_results = self._keyword_search(query, k=k*2)
        
        # Combine and rank results
        combined_results = self._combine_search_results(
            semantic_results, 
            keyword_results, 
            semantic_weight, 
            keyword_weight
        )
        
        return combined_results[:k]
    
    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """
        Simple keyword search implementation
        """
        query_terms = query.lower().split()
        all_docs = self._get_all_documents()
        
        scored_docs = []
        for doc in all_docs:
            text_lower = doc.page_content.lower()
            
            # Calculate keyword score
            score = 0
            for term in query_terms:
                score += text_lower.count(term)
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
    
    def _get_all_documents(self) -> List[Document]:
        """
        Get all documents from vector store (for keyword search)
        """
        if self.use_pinecone:
            # For Pinecone, this would require scanning all vectors
            # Simplified implementation
            logger.warning("Full document retrieval not implemented for Pinecone")
            return []
        else:
            try:
                # Get all documents from ChromaDB
                results = self.collection.get()
                documents = []
                
                for i, doc_text in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    documents.append(doc)
                
                return documents
            except Exception as e:
                logger.error(f"Error retrieving all documents: {str(e)}")
                return []
    
    def _combine_search_results(
        self,
        semantic_results: List[Document],
        keyword_results: List[Document],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Tuple[Document, float]]:
        """
        Combine semantic and keyword search results with weighted scoring
        """
        combined_scores = {}
        
        # Add semantic results
        for i, doc in enumerate(semantic_results):
            doc_key = doc.page_content[:100]  # Use first 100 chars as key
            score = semantic_weight * (len(semantic_results) - i) / len(semantic_results)
            combined_scores[doc_key] = {'doc': doc, 'score': score}
        
        # Add keyword results
        for i, doc in enumerate(keyword_results):
            doc_key = doc.page_content[:100]
            keyword_score = keyword_weight * (len(keyword_results) - i) / len(keyword_results)
            
            if doc_key in combined_scores:
                combined_scores[doc_key]['score'] += keyword_score
            else:
                combined_scores[doc_key] = {'doc': doc, 'score': keyword_score}
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [(result['doc'], result['score']) for result in sorted_results]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        """
        try:
            if self.use_pinecone:
                stats = self.index.describe_index_stats()
                return {
                    "total_vectors": stats['total_vector_count'],
                    "index_fullness": stats.get('index_fullness', 0),
                    "dimension": stats['dimension']
                }
            else:
                count = self.collection.count()
                return {
                    "total_documents": count,
                    "collection_name": self.collection_name,
                    "embedding_dimension": VectorStoreConfig.EMBEDDING_DIMENSION
                }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_documents(self, filter: Dict[str, Any]) -> int:
        """
        Delete documents matching filter
        """
        try:
            if self.use_pinecone:
                # Pinecone deletion by filter
                delete_response = self.index.delete(filter=filter)
                logger.info(f"Deleted documents from Pinecone with filter: {filter}")
                return 1  # Pinecone doesn't return count
            else:
                # ChromaDB deletion
                # First get IDs to delete
                results = self.collection.get(where=filter)
                if results['ids']:
                    self.collection.delete(ids=results['ids'])
                    deleted_count = len(results['ids'])
                    logger.info(f"Deleted {deleted_count} documents from ChromaDB")
                    return deleted_count
                else:
                    return 0
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return 0
    
    def reset_collection(self):
        """
        Reset/clear the entire collection
        """
        try:
            if self.use_pinecone:
                self.index.delete(delete_all=True)
                logger.info("Reset Pinecone index")
            else:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Reset ChromaDB collection")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")

# Advanced search utilities
class AdvancedSearch:
    """
    Advanced search capabilities for contract analysis
    """
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
    
    def search_by_section(self, query: str, section: str, k: int = 5) -> List[Document]:
        """
        Search within specific contract sections
        """
        filter_dict = {"section": section}
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def search_by_contract_type(self, query: str, contract_type: str, k: int = 5) -> List[Document]:
        """
        Search within specific contract types
        """
        filter_dict = {"contract_type": contract_type}
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def search_high_quality_chunks(self, query: str, min_quality: float = 0.7, k: int = 5) -> List[Document]:
        """
        Search only high-quality chunks
        """
        # First get more results
        initial_results = self.vector_store.similarity_search(query, k=k*3)
        
        # Filter by quality score
        quality_results = [
            doc for doc in initial_results 
            if doc.metadata.get("quality_score", 0) >= min_quality
        ]
        
        return quality_results[:k]
    
    def search_with_semantic_tags(self, query: str, required_tags: List[str], k: int = 5) -> List[Document]:
        """
        Search documents that contain specific semantic tags
        """
        initial_results = self.vector_store.similarity_search(query, k=k*2)
        
        # Filter by semantic tags
        tagged_results = []
        for doc in initial_results:
            doc_tags = doc.metadata.get("semantic_tags", [])
            if any(tag in doc_tags for tag in required_tags):
                tagged_results.append(doc)
        
        return tagged_results[:k]

# Example usage and testing
if __name__ == "__main__":
    # Test the vector store
    logger.info("Testing VectorStoreManager with Gemini embeddings...")
    
    try:
        # Initialize vector store
        vector_store = VectorStoreManager(use_pinecone=False)
        
        # Test documents
        test_docs = [
            Document(
                page_content="Dit is een koopovereenkomst voor een woning in Amsterdam. De koopsom bedraagt €500.000.",
                metadata={"section": "prijs", "contract_type": "koopovereenkomst", "semantic_tags": ["financial"]}
            ),
            Document(
                page_content="De verkoper garandeert dat de woning vrij is van gebreken en in goede staat verkeert.",
                metadata={"section": "garanties", "contract_type": "koopovereenkomst", "semantic_tags": ["legal", "risk"]}
            ),
            Document(
                page_content="De overeenkomst kan worden ontbonden indien de financiering niet wordt verkregen binnen 30 dagen.",
                metadata={"section": "ontbinding", "contract_type": "koopovereenkomst", "semantic_tags": ["conditional", "temporal"]}
            )
        ]
        
        # Add documents
        logger.info("Adding test documents...")
        ids = vector_store.add_documents(test_docs)
        logger.info(f"Added {len(ids)} documents")
        
        # Test search
        logger.info("Testing similarity search...")
        results = vector_store.similarity_search("koopsom prijs", k=2)
        
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc.page_content[:100]}...")
            logger.info(f"  Section: {doc.metadata.get('section')}")
            logger.info(f"  Tags: {doc.metadata.get('semantic_tags')}")
        
        # Test advanced search
        advanced = AdvancedSearch(vector_store)
        
        # Search by section
        section_results = advanced.search_by_section("garanties gebreken", "garanties", k=1)
        logger.info(f"Section search found {len(section_results)} results")
        
        # Get stats
        stats = vector_store.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
        
        logger.info("✅ Vector store test completed successfully!")
    
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {str(e)}")
        raise