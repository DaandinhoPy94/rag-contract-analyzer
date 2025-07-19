"""
Test script voor het complete RAG Contract Analyzer systeem
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from config import GeminiConfig, VectorStoreConfig, DocumentProcessingConfig
from utils.document_processor import ContractProcessor, validate_pdf_file
from utils.vector_store import VectorStoreManager, GeminiEmbeddings
from chains.contract_analyzer_chain import ContractAnalyzerChain, ContractConversation, validate_gemini_setup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_contract_text() -> str:
    """
    Maak een sample contract voor testing
    """
    return """
KOOPOVEREENKOMST ONROERENDE ZAAK

Partijen:
Verkoper: De heer Johannes Petrus van der Berg, geboren op 15 maart 1975 te Amsterdam, 
wonende te (1012 AB) Amsterdam, Damrak 100, ingeschreven in de Basisregistratie Personen 
van de gemeente Amsterdam.

Koper: Mevrouw Sarah Elisabeth Jansen, geboren op 22 juli 1985 te Utrecht,
wonende te (3511 AB) Utrecht, Oudegracht 250, ingeschreven in de Basisregistratie Personen
van de gemeente Utrecht.

ARTIKEL 1 - OBJECT
Het object van deze overeenkomst betreft de verkoop en koop van:
Een woning gelegen aan de Prinsengracht 456, 1017 KG Amsterdam, kadastraal bekend 
gemeente Amsterdam, sectie AA, nummer 12345, groot 125 m², hierna te noemen "de woning".

De woning omvat:
- Begane grond: hal, woonkamer, keuken, toilet
- Eerste verdieping: 2 slaapkamers, badkamer
- Tweede verdieping: bergzolder
- Achtertuin van circa 25 m²

ARTIKEL 2 - KOOPSOM EN BETALING
De koopsom bedraagt € 850.000 (achthonderdvijftigduizend euro), te betalen als volgt:
- Bij ondertekening: € 85.000 (10% aanbetaling)
- Bij transport: € 765.000

ARTIKEL 3 - ONTBINDENDE VOORWAARDEN
Deze overeenkomst wordt aangegaan onder de navolgende ontbindende voorwaarden:

3.1 Financieringsvoorbehoud
Koper heeft het recht deze overeenkomst te ontbinden indien zij er niet in slaagt 
vóór 31 maart 2024 een hypothecaire lening te verkrijgen van ten minste € 680.000 
tegen een rentepercentage van maximaal 4,5% per jaar.

3.2 Bouwkundige keuring
Koper heeft het recht om binnen 14 dagen na ondertekening een bouwkundige keuring 
te laten uitvoeren. Indien uit deze keuring gebreken blijken die de waarde van 
de woning met meer dan € 15.000 doen dalen, kan koper de overeenkomst ontbinden.

3.3 Energielabel
Verkoper zal uiterlijk bij de levering een geldig energielabel overhandigen.

ARTIKEL 4 - GARANTIES EN AANSPRAKELIJKHEID
4.1 Verkoper garandeert dat de woning vrij is van verborgen gebreken die de 
bewoonbaarheid aantasten of de waarde significant beïnvloeden.

4.2 De woning wordt verkocht in de staat waarin zij zich bevindt ("as is, where is"), 
behoudens de garanties genoemd in artikel 4.1.

4.3 Verkoper is aansprakelijk voor gebreken die hij heeft verzwegen of 
waarvan hij wist ten tijde van de verkoop.

ARTIKEL 5 - LEVERING EN TRANSPORT
5.1 Levering geschiedt op 15 april 2024 of zoveel eerder of later als 
partijen nader overeenkomen.

5.2 De kosten van transport komen voor rekening van koper.

5.3 Bij levering zal verkoper de woning ontruimd en bezemschoon opleveren.

ARTIKEL 6 - BOETECLAUSULE
Indien een partij toerekenbaar tekortschiet in de nakoming van haar 
verplichtingen, is zij een onmiddellijk opeisbare boete verschuldigd 
van € 50.000, onverminderd het recht van de wederpartij op nakoming 
en/of schadevergoeding.

ARTIKEL 7 - BIJZONDERE BEPALINGEN
7.1 In de koopsom zijn begrepen de in de woning aanwezige vaste zaken 
en toebehoren zoals vermeld in de lijst van zaken (bijlage A).

7.2 De erfpacht eindigt op 31 december 2050. De jaarlijkse erfpachtcanon 
bedraagt € 2.400 per jaar.

7.3 De woning is gebouwd in 1920 en verkoper wijst uitdrukkelijk op de 
ouderdom van het pand.

Aldus overeengekomen en getekend te Amsterdam op 15 februari 2024.

Verkoper: J.P. van der Berg
Koper: S.E. Jansen
"""

def test_environment_setup():
    """
    Test of de environment correct is geconfigureerd
    """
    logger.info("🔧 Testing environment setup...")
    
    # Check API key
    if not GeminiConfig.API_KEY:
        logger.error("❌ GEMINI_API_KEY not configured")
        return False
    
    logger.info(f"✅ Gemini API key configured (ends with: ...{GeminiConfig.API_KEY[-4:]})")
    
    # Check directories
    required_dirs = ['contracts', 'data', 'data/chroma_db']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {dir_name}")
        else:
            logger.info(f"✅ Directory exists: {dir_name}")
    
    # Validate Gemini API
    if not validate_gemini_setup():
        logger.error("❌ Gemini API validation failed")
        return False
    
    logger.info("✅ Environment setup complete")
    return True

def test_document_processor():
    """
    Test document processing functionality
    """
    logger.info("📄 Testing document processor...")
    
    try:
        # Create processor
        processor = ContractProcessor()
        
        # Create a sample text file to test with
        sample_text = create_sample_contract_text()
        
        # Test text cleaning
        cleaned = processor._clean_text(sample_text)
        logger.info(f"✅ Text cleaning: {len(sample_text)} → {len(cleaned)} chars")
        
        # Test section identification
        sections = processor.identify_contract_sections(sample_text)
        logger.info(f"✅ Section identification: found {len(sections)} sections")
        for section_name in sections.keys():
            logger.info(f"   - {section_name}")
        
        # Test semantic tagging
        tags = processor.extract_semantic_tags(sample_text)
        logger.info(f"✅ Semantic tagging: {tags}")
        
        # Test chunking
        chunks = processor.create_chunks_with_metadata(
            sample_text,
            {"contract_type": "koopovereenkomst", "test": True}
        )
        logger.info(f"✅ Chunking: created {len(chunks)} chunks")
        
        # Show sample chunk
        if chunks:
            sample_chunk = chunks[0]
            logger.info(f"   Sample chunk: {sample_chunk.page_content[:100]}...")
            logger.info(f"   Metadata: {sample_chunk.metadata}")
        
        logger.info("✅ Document processor test complete")
        return chunks
    
    except Exception as e:
        logger.error(f"❌ Document processor test failed: {str(e)}")
        return None

def test_vector_store(chunks):
    """
    Test vector store functionality
    """
    logger.info("🗄️ Testing vector store...")
    
    try:
        # Initialize vector store
        vector_store = VectorStoreManager(use_pinecone=False)
        logger.info("✅ Vector store initialized")
        
        # Test embeddings
        embeddings = vector_store.embeddings
        test_embedding = embeddings.embed_query("test query")
        logger.info(f"✅ Embeddings: dimension {len(test_embedding)}")
        
        # Add documents
        if chunks:
            ids = vector_store.add_documents(chunks[:3])  # Test with first 3 chunks
            logger.info(f"✅ Added {len(ids)} documents")
        
        # Test search
        search_results = vector_store.similarity_search("koopsom prijs", k=2)
        logger.info(f"✅ Search: found {len(search_results)} results")
        
        for i, doc in enumerate(search_results):
            logger.info(f"   Result {i+1}: {doc.page_content[:80]}...")
        
        # Test stats
        stats = vector_store.get_collection_stats()
        logger.info(f"✅ Collection stats: {stats}")
        
        logger.info("✅ Vector store test complete")
        return vector_store
    
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {str(e)}")
        return None

def test_analyzer_chain(vector_store):
    """
    Test contract analyzer chain
    """
    logger.info("🤖 Testing analyzer chain...")
    
    try:
        # Initialize analyzer
        analyzer = ContractAnalyzerChain(vector_store)
        logger.info("✅ Analyzer chain initialized")
        
        # Test Q&A
        logger.info("Testing Q&A...")
        qa_result = analyzer.ask_question("Wat is de koopsom van de woning?")
        logger.info(f"✅ Q&A response: {qa_result['answer'][:100]}...")
        
        # Test risk analysis
        logger.info("Testing risk analysis...")
        risk_result = analyzer.analyze_contract_risks("koopovereenkomst", "koper")
        logger.info(f"✅ Risk analysis complete")
        if 'risk_score' in risk_result and risk_result['risk_score']:
            logger.info(f"   Risk score: {risk_result['risk_score']}/10")
        
        # Test summary
        logger.info("Testing summary generation...")
        summary_result = analyzer.generate_summary()
        logger.info(f"✅ Summary generated")
        if 'contract_score' in summary_result and summary_result['contract_score']:
            logger.info(f"   Contract score: {summary_result['contract_score']}/10")
        
        # Test compliance
        logger.info("Testing compliance check...")
        compliance_result = analyzer.check_compliance("koopovereenkomst")
        logger.info(f"✅ Compliance check complete")
        if 'overall_compliance' in compliance_result:
            logger.info(f"   Overall compliance: {compliance_result['overall_compliance']:.1f}%")
        
        # Test conversation
        logger.info("Testing conversational interface...")
        conversation = ContractConversation(analyzer)
        
        test_messages = [
            "Wat zijn de belangrijkste risico's?",
            "Wanneer moet de financiering rond zijn?",
            "Geef me een samenvatting van het contract"
        ]
        
        for msg in test_messages:
            result = conversation.chat(msg)
            logger.info(f"   Message: '{msg[:30]}...' → Intent: {result.get('intent')}")
        
        conv_summary = conversation.get_conversation_summary()
        logger.info(f"✅ Conversation: {conv_summary['total_messages']} messages")
        
        logger.info("✅ Analyzer chain test complete")
        return analyzer
    
    except Exception as e:
        logger.error(f"❌ Analyzer chain test failed: {str(e)}")
        return None

def test_end_to_end_workflow():
    """
    Test complete end-to-end workflow
    """
    logger.info("🚀 Starting end-to-end workflow test...")
    
    try:
        # Step 1: Environment
        if not test_environment_setup():
            return False
        
        # Step 2: Document processing
        chunks = test_document_processor()
        if not chunks:
            return False
        
        # Step 3: Vector store
        vector_store = test_vector_store(chunks)
        if not vector_store:
            return False
        
        # Step 4: Analyzer chain
        analyzer = test_analyzer_chain(vector_store)
        if not analyzer:
            return False
        
        logger.info("🎉 End-to-end workflow test SUCCESSFUL!")
        
        # Show summary
        logger.info("\n" + "="*50)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*50)
        logger.info(f"✅ Document chunks created: {len(chunks)}")
        logger.info(f"✅ Vector store initialized: ChromaDB")
        logger.info(f"✅ Analyzer chain ready: Gemini-powered")
        logger.info(f"✅ All major components working")
        logger.info("="*50)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {str(e)}")
        return False

def interactive_test():
    """
    Interactive test mode
    """
    logger.info("🎮 Starting interactive test mode...")
    
    # Run full workflow first
    if not test_end_to_end_workflow():
        logger.error("❌ End-to-end test failed, cannot start interactive mode")
        return
    
    print("\n" + "="*60)
    print("🤖 INTERACTIVE CONTRACT ANALYZER TEST")
    print("="*60)
    print("The system is ready! You can now ask questions about the sample contract.")
    print("Sample questions:")
    print("- Wat is de koopsom?")
    print("- Welke ontbindende voorwaarden zijn er?")
    print("- Analyseer de risico's")
    print("- Geef een samenvatting")
    print("\nType 'quit' to exit")
    print("="*60)
    
    # Create minimal system for interaction
    sample_text = create_sample_contract_text()
    processor = ContractProcessor()
    chunks = processor.create_chunks_with_metadata(
        sample_text,
        {"contract_type": "koopovereenkomst", "interactive_test": True}
    )
    
    vector_store = VectorStoreManager(use_pinecone=False)
    vector_store.add_documents(chunks)
    
    analyzer = ContractAnalyzerChain(vector_store)
    conversation = ContractConversation(analyzer)
    
    while True:
        try:
            user_input = input("\n💬 Jouw vraag: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'stop']:
                print("👋 Tot ziens!")
                break
            
            if not user_input:
                continue
            
            print("🤔 Bezig met analyseren...")
            result = conversation.chat(user_input)
            
            print(f"\n🤖 Antwoord ({result.get('intent', 'unknown')} intent):")
            print("-" * 40)
            print(result.get('answer', result.get('analysis', result.get('summary', 'Geen antwoord beschikbaar'))))
            print("-" * 40)
        
        except KeyboardInterrupt:
            print("\n👋 Tot ziens!")
            break
        except Exception as e:
            print(f"❌ Fout: {str(e)}")

if __name__ == "__main__":
    print("🚀 RAG Contract Analyzer - System Test")
    print("=" * 50)
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        # Run comprehensive test
        success = test_end_to_end_workflow()
        
        if success:
            print("\n🎉 All tests passed! System is ready.")
            print("\nNext steps:")
            print("1. Run 'python test_system.py interactive' for interactive testing")
            print("2. Run 'streamlit run frontend/app.py' to start the web interface")
            print("3. Upload your own PDF contracts to test with real documents")
        else:
            print("\n❌ Tests failed. Please check the logs above.")
            sys.exit(1)