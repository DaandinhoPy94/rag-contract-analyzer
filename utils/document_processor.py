"""
Document processor voor contracten - geoptimaliseerd voor Gemini API
"""
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import google.generativeai as genai
from config import GeminiConfig, VectorStoreConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContractProcessor:
    """
    Verwerkt vastgoedcontracten voor RAG met Gemini API
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        # Initialize Gemini
        genai.configure(api_key=GeminiConfig.API_KEY)
        GeminiConfig.validate()
        
        # Text splitting configuration
        self.chunk_size = chunk_size or VectorStoreConfig.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or VectorStoreConfig.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )
        
        # Contract section patterns (Nederlandse contracten)
        self.section_patterns = {
            "partijen": [
                r"(?i)(partijen|ondergetekenden|contractanten|verkoper|koper)",
                r"(?i)(naam|adres|woonplaats).*?:",
                r"(?i)(hierna te noemen|verder genoemd)"
            ],
            "object": [
                r"(?i)(object|onroerende zaak|pand|woning|appartement)",
                r"(?i)(gelegen te|gevestigd te|staande te)",
                r"(?i)(kadastraal bekend|sectie|nummer)"
            ],
            "prijs": [
                r"(?i)(koopsom|koopprijs|prijs|bedrag)",
                r"(?i)(euro|€|\d+\.\d+)",
                r"(?i)(inclusief|exclusief|btw)"
            ],
            "voorwaarden": [
                r"(?i)(voorwaarden|condities|bepalingen)",
                r"(?i)(ontbindende voorwaarden|opschortende voorwaarden)",
                r"(?i)(financiering|hypotheek|lening)"
            ],
            "ontbinding": [
                r"(?i)(ontbinding|ontbindende voorwaarden|nietigheid)",
                r"(?i)(niet nakoming|wanprestatie|in gebreke)",
                r"(?i)(schadevergoeding|boete|dwangsom)"
            ],
            "garanties": [
                r"(?i)(garanties|waarborgen|staat|onderhoud)",
                r"(?i)(gebreken|mankementen|verborgen gebreken)",
                r"(?i)(as is|where is|zoals het er bij ligt)"
            ],
            "termijnen": [
                r"(?i)(termijn|datum|deadline|uiterlijk)",
                r"(?i)(levering|overdracht|transport)",
                r"(?i)(dagen|weken|maanden|voor)"
            ]
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text uit PDF contract met metadata"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            metadata = {
                "filename": Path(pdf_path).name,
                "total_pages": len(reader.pages),
                "file_size": os.path.getsize(pdf_path),
                "extraction_method": "pypdf"
            }
            
            logger.info(f"Processing PDF: {metadata['filename']} ({metadata['total_pages']} pages)")
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    page_text = self._clean_text(page_text)
                    
                    if page_text.strip():
                        text += f"\n--- Pagina {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
            
            metadata.update({
                "total_characters": len(text),
                "total_words": len(text.split()),
                "estimated_chunks": len(text) // self.chunk_size + 1
            })
            
            logger.info(f"Extracted {metadata['total_words']} words from {metadata['total_pages']} pages")
            
            return {"text": text, "metadata": metadata}
        
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean up extracted text voor betere processing"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors in Dutch text
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\€\%\&\@\+\=\[\]]', '', text)
        
        # Fix common Dutch character issues
        replacements = {
            'ë': 'e', 'ï': 'i', 'ö': 'o', 'é': 'e', 'è': 'e', 
            'ê': 'e', 'á': 'a', 'à': 'a', 'â': 'a'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def identify_contract_sections(self, text: str) -> Dict[str, str]:
        """Identificeer belangrijke secties in Nederlandse contracten"""
        sections = {}
        
        for section_name, patterns in self.section_patterns.items():
            section_text = ""
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    start = max(0, match.start() - 500)
                    end = min(len(text), match.end() + 500)
                    context = text[start:end]
                    
                    if context not in section_text:
                        section_text += f"\n{context}\n"
            
            if section_text.strip():
                sections[section_name] = section_text.strip()
                logger.debug(f"Section '{section_name}': {len(section_text)} characters found")
        
        return sections
    
    def extract_semantic_tags(self, text: str) -> List[str]:
        """Extract semantic tags voor betere search en categorisatie"""
        tags = []
        text_lower = text.lower()
        
        tag_patterns = {
            "financial": ["euro", "€", "koopsom", "prijs", "betaling", "bedrag", "geld", "kosten"],
            "legal": ["recht", "plicht", "aansprakelijk", "juridisch", "wet", "artikel", "clausule"],
            "temporal": ["datum", "termijn", "deadline", "uiterlijk", "voor", "binnen", "dagen", "weken"],
            "conditional": ["voorwaarde", "indien", "mits", "tenzij", "behoudens", "onder voorbehoud"],
            "risk": ["risico", "gevaar", "waarschuwing", "let op", "gebreken", "schade", "aansprakelijkheid"],
            "property": ["vastgoed", "onroerend", "woning", "appartement", "grond", "bouw"]
        }
        
        for tag, terms in tag_patterns.items():
            if any(term in text_lower for term in terms):
                tags.append(tag)
        
        # Contract type detection
        if any(term in text_lower for term in ["koop", "verkoop", "koopovereenkomst"]):
            tags.append("purchase_contract")
        elif any(term in text_lower for term in ["huur", "verhuur", "huurovereenkomst"]):
            tags.append("rental_contract")
        
        return list(set(tags))
    
    def create_chunks_with_metadata(self, text: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Maak intelligent chunks met uitgebreide metadata"""
        all_chunks = []
        
        # First identify sections
        sections = self.identify_contract_sections(text)
        logger.info(f"Identified {len(sections)} contract sections")
        
        # Process each section separately
        for section_name, section_text in sections.items():
            if section_text and len(section_text.strip()) > 100:
                section_chunks = self.text_splitter.split_text(section_text)
                
                for i, chunk in enumerate(section_chunks):
                    if len(chunk.strip()) < 100:
                        continue
                    
                    chunk_metadata = {
                        **base_metadata,
                        "section": section_name,
                        "chunk_index": i,
                        "total_chunks_in_section": len(section_chunks),
                        "chunk_length": len(chunk),
                        "chunk_type": "section_specific"
                    }
                    
                    chunk_metadata["semantic_tags"] = self.extract_semantic_tags(chunk)
                    chunk_metadata["quality_score"] = self._calculate_chunk_quality(chunk)
                    
                    doc = Document(page_content=chunk, metadata=chunk_metadata)
                    all_chunks.append(doc)
        
        # Also process full text
        full_text_chunks = self.text_splitter.split_text(text)
        for i, chunk in enumerate(full_text_chunks):
            if len(chunk.strip()) < 100:
                continue
            
            chunk_metadata = {
                **base_metadata,
                "section": "full_document",
                "chunk_index": i,
                "total_chunks": len(full_text_chunks),
                "chunk_length": len(chunk),
                "chunk_type": "full_document"
            }
            
            chunk_metadata["semantic_tags"] = self.extract_semantic_tags(chunk)
            chunk_metadata["quality_score"] = self._calculate_chunk_quality(chunk)
            
            doc = Document(page_content=chunk, metadata=chunk_metadata)
            all_chunks.append(doc)
        
        logger.info(f"Created {len(all_chunks)} total chunks")
        return all_chunks
    
    def _calculate_chunk_quality(self, chunk: str) -> float:
        """Bereken kwaliteitsscore voor een chunk (0-1)"""
        score = 0.5
        
        # Length score
        length = len(chunk)
        if 500 <= length <= 1500:
            score += 0.2
        elif 200 <= length <= 2000:
            score += 0.1
        
        # Content richness
        sentences = chunk.count('.')
        if sentences >= 3:
            score += 0.1
        
        # Legal terminology
        legal_terms = ["artikel", "clausule", "bepaling", "voorwaarde", "overeenkomst"]
        legal_count = sum(1 for term in legal_terms if term.lower() in chunk.lower())
        score += min(legal_count * 0.05, 0.2)
        
        # Avoid repetition
        words = chunk.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness = len(unique_words) / len(words)
            if uniqueness < 0.3:
                score -= 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def process_contract(self, file_path: str, contract_type: str = "unknown") -> Dict[str, Any]:
        """Complete contract processing pipeline"""
        logger.info(f"Starting contract processing: {file_path}")
        
        # Extract text
        extraction_result = self.extract_text_from_pdf(file_path)
        text = extraction_result["text"]
        file_metadata = extraction_result["metadata"]
        
        # Create base metadata
        base_metadata = {
            **file_metadata,
            "contract_type": contract_type,
            "processing_timestamp": str(os.times()),
            "processor_version": "1.0.0"
        }
        
        # Create chunks
        chunks = self.create_chunks_with_metadata(text, base_metadata)
        
        # Identify sections
        sections = self.identify_contract_sections(text)
        
        # Calculate statistics
        stats = {
            "total_chunks": len(chunks),
            "sections_found": list(sections.keys()),
            "avg_chunk_length": sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0,
            "quality_scores": [chunk.metadata.get("quality_score", 0) for chunk in chunks]
        }
        
        logger.info(f"Processing complete: {stats['total_chunks']} chunks, {len(sections)} sections")
        
        return {
            "chunks": chunks,
            "sections": sections,
            "metadata": base_metadata,
            "statistics": stats,
            "raw_text": text
        }

def validate_pdf_file(file_path: str) -> bool:
    """Valideer of PDF bestand geldig is"""
    try:
        if not os.path.exists(file_path):
            return False
        
        if not file_path.lower().endswith('.pdf'):
            return False
        
        # Check file size (50MB max)
        file_size = os.path.getsize(file_path)
        max_size = 50 * 1024 * 1024
        if file_size > max_size:
            logger.warning(f"File too large: {file_size / 1024 / 1024:.1f}MB > 50MB")
            return False
        
        # Try to open with PyPDF
        reader = PdfReader(file_path)
        if len(reader.pages) == 0:
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"PDF validation failed: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    processor = ContractProcessor()
    print("✅ ContractProcessor initialized successfully")