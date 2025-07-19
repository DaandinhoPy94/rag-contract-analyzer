"""
Configuration settings voor RAG Contract Analyzer met Gemini API
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
CONTRACTS_DIR = PROJECT_ROOT / "contracts"
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Create directories if they don't exist
CONTRACTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
(CONTRACTS_DIR / "uploaded").mkdir(exist_ok=True)
(CONTRACTS_DIR / "samples").mkdir(exist_ok=True)
(DATA_DIR / "chroma_db").mkdir(exist_ok=True)
(DATA_DIR / "processed_contracts").mkdir(exist_ok=True)

class GeminiConfig:
    """Gemini API configuratie"""
    
    # API Configuration
    API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    
    # Model parameters
    TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
    TOP_P: float = float(os.getenv("GEMINI_TOP_P", "0.8"))
    TOP_K: int = int(os.getenv("GEMINI_TOP_K", "40"))
    
    # Safety settings
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    @classmethod
    def validate(cls) -> bool:
        """Valideer of alle vereiste configuratie aanwezig is"""
        if not cls.API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        return True

class VectorStoreConfig:
    """Vector store configuratie"""
    
    # ChromaDB (local development)
    CHROMA_DB_PATH: str = str(DATA_DIR / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "contracts"
    
    # Pinecone (production)
    USE_PINECONE: bool = os.getenv("USE_PINECONE", "false").lower() == "true"
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = "contract-analyzer"
    
    # Embedding settings
    EMBEDDING_DIMENSION: int = 768  # Gemini embedding dimension
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Search settings
    DEFAULT_K: int = int(os.getenv("DEFAULT_K", "5"))
    MAX_K: int = int(os.getenv("MAX_K", "20"))

class DocumentProcessingConfig:
    """Document processing configuratie"""
    
    # File settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt", ".docx"]
    
    # Text processing
    MIN_CHUNK_LENGTH: int = int(os.getenv("MIN_CHUNK_LENGTH", "100"))
    MAX_CHUNK_LENGTH: int = int(os.getenv("MAX_CHUNK_LENGTH", "2000"))
    
    # Language settings
    DEFAULT_LANGUAGE: str = "nl"  # Nederlands
    SUPPORTED_LANGUAGES: list = ["nl", "en", "de"]

class StreamlitConfig:
    """Streamlit app configuratie"""
    
    PAGE_TITLE: str = "RAG Contract Analyzer"
    PAGE_ICON: str = "📄"
    LAYOUT: str = "wide"
    
    # UI settings
    SIDEBAR_STATE: str = "expanded"
    THEME_PRIMARY_COLOR: str = "#FF6B6B"
    THEME_BACKGROUND_COLOR: str = "#FFFFFF"
    THEME_SECONDARY_BACKGROUND_COLOR: str = "#F0F2F6"

class LoggingConfig:
    """Logging configuratie"""
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = str(PROJECT_ROOT / "logs" / "app.log")
    
    # Maak logs directory
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

class SecurityConfig:
    """Security en privacy configuratie"""
    
    # Data retention
    RETENTION_DAYS: int = int(os.getenv("RETENTION_DAYS", "30"))
    
    # Encryption (voor productie)
    ENCRYPT_AT_REST: bool = os.getenv("ENCRYPT_AT_REST", "false").lower() == "true"
    ENCRYPTION_KEY: Optional[str] = os.getenv("ENCRYPTION_KEY")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# Compliance regels voor Nederlandse contracten
COMPLIANCE_RULES = {
    "koopovereenkomst": [
        {
            "id": "bedenktijd",
            "description": "Wettelijke bedenktijd",
            "requirement": "Contract moet 3 dagen bedenktijd voor consument bevatten",
            "search_query": "bedenktijd drie dagen consument",
            "mandatory": True
        },
        {
            "id": "lijst_van_zaken",
            "description": "Lijst van zaken",
            "requirement": "Lijst van zaken moet zijn bijgevoegd",
            "search_query": "lijst van zaken roerende",
            "mandatory": True
        },
        {
            "id": "energielabel",
            "description": "Energielabel",
            "requirement": "Energielabel moet worden overhandigd",
            "search_query": "energielabel energie prestatie",
            "mandatory": True
        },
        {
            "id": "ouderdomsclausule",
            "description": "Ouderdomsclausule",
            "requirement": "Bij woningen ouder dan 20 jaar moet ouderdomsclausule zijn opgenomen",
            "search_query": "ouderdom bouwjaar staat onderhoud",
            "mandatory": False
        }
    ],
    "huurovereenkomst": [
        {
            "id": "huurprijs",
            "description": "Maximale huurprijs",
            "requirement": "Huurprijs mag niet boven de maximale huurprijs liggen",
            "search_query": "huurprijs maximum punten",
            "mandatory": True
        },
        {
            "id": "servicekosten",
            "description": "Servicekosten specificatie",
            "requirement": "Servicekosten moeten gespecificeerd zijn",
            "search_query": "servicekosten specificatie overzicht",
            "mandatory": True
        }
    ]
}

# Development/Production environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"

# Export all configs
__all__ = [
    "GeminiConfig",
    "VectorStoreConfig", 
    "DocumentProcessingConfig",
    "StreamlitConfig",
    "LoggingConfig",
    "SecurityConfig",
    "COMPLIANCE_RULES",
    "PROJECT_ROOT",
    "CONTRACTS_DIR",
    "DATA_DIR",
    "EMBEDDINGS_DIR",
    "DEBUG"
]