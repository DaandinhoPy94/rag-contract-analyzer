"""
RAG Chain voor contract analyse met Gemini API
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

# LangChain imports
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

# Gemini
import google.generativeai as genai

# Local imports
from config import GeminiConfig, COMPLIANCE_RULES
from utils.vector_store import VectorStoreManager, AdvancedSearch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM:
    """
    Custom Gemini LLM wrapper voor LangChain compatibility
    """
    
    def __init__(self, model_name: str = None, temperature: float = None):
        genai.configure(api_key=GeminiConfig.API_KEY)
        self.model_name = model_name or GeminiConfig.MODEL_NAME
        self.temperature = temperature or GeminiConfig.TEMPERATURE
        
        # Configure the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=GeminiConfig.MAX_OUTPUT_TOKENS,
                top_p=GeminiConfig.TOP_P,
                top_k=GeminiConfig.TOP_K,
            ),
            safety_settings=GeminiConfig.SAFETY_SETTINGS
        )
        
        logger.info(f"Initialized Gemini LLM: {self.model_name}")
    
    def invoke(self, prompt: str) -> str:
        """
        Generate response from Gemini
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def __call__(self, prompt: str) -> str:
        """
        Make class callable
        """
        return self.invoke(prompt)

class ContractAnalyzerChain:
    """
    RAG chain voor intelligente contract analyse met Gemini
    """
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        model_name: str = None,
        temperature: float = 0.1
    ):
        self.vector_store = vector_store
        self.advanced_search = AdvancedSearch(vector_store)
        
        # Initialize Gemini LLM
        self.llm = GeminiLLM(model_name, temperature)
        
        # Memory for conversations
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create prompts
        self.qa_prompt = self._create_qa_prompt()
        self.risk_prompt = self._create_risk_analysis_prompt()
        self.summary_prompt = self._create_summary_prompt()
        self.compliance_prompt = self._create_compliance_prompt()
        
        logger.info("ContractAnalyzerChain initialized")
    
    def _create_qa_prompt(self) -> PromptTemplate:
        """
        Prompt voor Q&A over contracten
        """
        template = """Je bent een expert juridisch adviseur gespecialiseerd in Nederlandse vastgoedcontracten.

Gebruik ALLEEN de volgende context uit het contract om de vraag te beantwoorden:

{context}

BELANGRIJKE INSTRUCTIES:
1. Baseer je antwoord UITSLUITEND op de gegeven context
2. Als het antwoord niet in de context staat, zeg dat expliciet
3. Citeer relevante passages tussen aanhalingstekens
4. Wees precies en juridisch correct
5. Gebruik heldere Nederlandse juridische terminologie
6. Vermeld altijd de bron van je informatie

Conversatie geschiedenis:
{chat_history}

Huidige vraag: {question}

Gestructureerd antwoord:
**Antwoord:** [Je directe antwoord]
**Bronverwijzing:** [Relevante passage uit de context]
**Toelichting:** [Extra uitleg indien nodig]
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question", "chat_history"]
        )
    
    def _create_risk_analysis_prompt(self) -> PromptTemplate:
        """
        Prompt voor uitgebreide risico analyse
        """
        template = """Je bent een senior risk analyst voor vastgoedtransacties met 20+ jaar ervaring.

Analyseer de volgende contractclausules grondig op risico's:

{context}

CONTRACT DETAILS:
- Type: {contract_type}
- Cliënt rol: {client_role}
- Analyse datum: {analysis_date}

Voer een SYSTEMATISCHE risicoanalyse uit:

**🔴 HOGE RISICO'S (Onmiddellijke actie vereist)**
Voor elk hoog risico:
- Risico beschrijving
- Financiële impact (€ schatting)
- Waarschijnlijkheid (%)
- Juridische gevolgen
- Concrete mitigatie stappen

**🟡 MEDIUM RISICO'S (Monitoring vereist)**
Voor elk medium risico:
- Risico beschrijving
- Potentiële impact
- Aanbevolen actie
- Timing

**🔵 AANDACHTSPUNTEN (Preventief)**
- Wat moet gemonitord worden
- Waarom relevant
- Preventieve maatregelen

**❌ ONTBREKENDE CLAUSULES**
- Standaard bescherming die ontbreekt
- Impact van afwezigheid
- Aanbevolen toevoegingen

**📋 ONDERHANDELINGS STRATEGIE**
- Top 3 prioriteiten
- Concrete voorstellen
- Argumentatie

**RISICO SCORE: X/10** (Geef overall score)

Wees specifiek, actionable en use Nederlandse juridische terminologie."""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "contract_type", "client_role", "analysis_date"]
        )
    
    def _create_summary_prompt(self) -> PromptTemplate:
        """
        Prompt voor executive summary
        """
        template = """Maak een professionele executive summary van dit vastgoedcontract.

Contract inhoud:
{context}

Maak een GESTRUCTUREERDE samenvatting:

**📋 KERNGEGEVENS**
- Contract type: 
- Partijen: [Verkoper] → [Koper]
- Object: [Adres en beschrijving]
- Transactiewaarde: €[bedrag]
- Sleuteldatum: [belangrijkste deadline]

**⚖️ HOOFDVOORWAARDEN**
1. [Belangrijkste voorwaarde 1]
2. [Belangrijkste voorwaarde 2]  
3. [Belangrijkste voorwaarde 3]
4. [Belangrijkste voorwaarde 4]
5. [Belangrijkste voorwaarde 5]

**💰 FINANCIËLE STRUCTUUR**
- Koopsom: €[bedrag]
- Betalingsregeling: [wanneer wat]
- Zekerheden: [welke waarborgen]
- Boetes/kosten: [mogelijke extra kosten]

**⚠️ KRITIEKE RISICO'S**
1. 🔴 [Hoogste risico]
2. 🟡 [Medium risico]  
3. 🔵 [Aandachtspunt]

**✅ ACTIEPUNTEN**
- [ ] [Actie 1] - deadline: [datum]
- [ ] [Actie 2] - deadline: [datum]
- [ ] [Actie 3] - deadline: [datum]

**📊 CONTRACT SCORE: X/10**
[Korte beoordeling waarom deze score]

Wees beknopt maar volledig. Gebruik bullets en duidelijke structuur."""
        
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )
    
    def _create_compliance_prompt(self) -> PromptTemplate:
        """
        Prompt voor compliance check
        """
        template = """Je bent een compliance officer gespecialiseerd in Nederlandse vastgoedwetgeving.

Controleer of het contract voldoet aan deze specifieke regel:

REGEL DETAILS:
- Beschrijving: {rule_description}
- Vereiste: {rule_requirement}
- Verplicht: {rule_mandatory}

CONTRACT CONTEXT:
{context}

Voer een GRONDIGE compliance check uit:

**✅ COMPLIANCE STATUS: [VOLDOET/VOLDOET NIET/ONDUIDELIJK]**

**🔍 BEVINDINGEN:**
- Relevante clausule gevonden: [Ja/Nee]
- Exacte tekst: "[citeer relevante passage]"
- Voldoet aan eis: [Ja/Nee/Gedeeltelijk]

**📋 ANALYSE:**
- Wat is aanwezig in contract
- Wat ontbreekt (indien van toepassing)
- Mate van compliance (%)

**⚠️ RISICO BEOORDELING:**
- Juridisch risico: [Hoog/Medium/Laag]
- Financieel risico: [schatting impact]
- Reputatie risico: [beoordeling]

**🎯 AANBEVOLEN ACTIES:**
1. [Concrete actie 1]
2. [Concrete actie 2]
3. [Concrete actie 3]

**📄 DOCUMENTATIE VEREIST:**
- [Welke documenten nodig]
- [Welke bewijsstukken]

Wees juridisch precies en actionable in je aanbevelingen."""
        
        return PromptTemplate(
            template=template,
            input_variables=["rule_description", "rule_requirement", "rule_mandatory", "context"]
        )
    
    def ask_question(self, question: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Beantwoord vragen over het contract met RAG
        """
        logger.info(f"Processing question: {question[:50]}...")
        
        try:
            # Get relevant context
            relevant_docs = self.vector_store.similarity_search(question, k=5)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Get chat history
            chat_history = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history[-4:]])  # Last 4 messages
            
            # Generate prompt
            prompt = self.qa_prompt.format(
                context=context,
                question=question,
                chat_history=history_text
            )
            
            # Get response from Gemini
            response = self.llm.invoke(prompt)
            
            # Update memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(response)
            
            return {
                "answer": response,
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in relevant_docs
                ],
                "context_used": context,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": f"Er is een fout opgetreden bij het verwerken van je vraag: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def analyze_contract_risks(
        self,
        contract_type: str,
        client_role: str = "koper"
    ) -> Dict[str, Any]:
        """
        Uitgebreide risico analyse van het contract
        """
        logger.info(f"Starting risk analysis for {contract_type} (client: {client_role})")
        
        try:
            # Get relevant sections for risk analysis
            risk_queries = [
                "ontbinding ontbindende voorwaarden nietigheid",
                "garanties waarborgen staat onderhoud gebreken",
                "boetes schadevergoeding dwangsom aansprakelijkheid",
                "termijnen deadlines levering transport",
                "financiering hypotheek lening voorwaarden",
                "kosten bijkomende kosten notaris makelaar"
            ]
            
            all_contexts = []
            for query in risk_queries:
                docs = self.vector_store.similarity_search(query, k=3)
                all_contexts.extend([doc.page_content for doc in docs])
            
            # Remove duplicates and combine
            unique_contexts = list(set(all_contexts))
            context = "\n\n".join(unique_contexts)
            
            # Generate risk analysis
            prompt = self.risk_prompt.format(
                context=context,
                contract_type=contract_type,
                client_role=client_role,
                analysis_date=datetime.now().strftime("%d-%m-%Y")
            )
            
            analysis = self.llm.invoke(prompt)
            
            # Extract risk score using regex
            risk_score_match = re.search(r'RISICO SCORE:\s*(\d+(?:\.\d+)?)/10', analysis)
            risk_score = float(risk_score_match.group(1)) if risk_score_match else None
            
            return {
                "analysis": analysis,
                "risk_score": risk_score,
                "contract_type": contract_type,
                "client_role": client_role,
                "analysis_date": datetime.now().isoformat(),
                "contexts_analyzed": len(unique_contexts),
                "sources": unique_contexts
            }
        
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            return {
                "analysis": f"Fout bij risicoanalyse: {str(e)}",
                "error": str(e)
            }
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Genereer executive summary van het contract
        """
        logger.info("Generating contract summary")
        
        try:
            # Get key sections for summary
            summary_queries = [
                "partijen verkoper koper naam adres",
                "object woning appartement adres gelegen",
                "koopsom prijs bedrag euro",
                "voorwaarden hoofdvoorwaarden bepalingen",
                "levering transport datum termijn"
            ]
            
            all_contexts = []
            for query in summary_queries:
                docs = self.vector_store.similarity_search(query, k=2)
                all_contexts.extend([doc.page_content for doc in docs])
            
            context = "\n\n".join(set(all_contexts))  # Remove duplicates
            
            # Generate summary
            prompt = self.summary_prompt.format(context=context)
            summary = self.llm.invoke(prompt)
            
            # Extract contract score
            score_match = re.search(r'CONTRACT SCORE:\s*(\d+(?:\.\d+)?)/10', summary)
            contract_score = float(score_match.group(1)) if score_match else None
            
            return {
                "summary": summary,
                "contract_score": contract_score,
                "generation_date": datetime.now().isoformat(),
                "sources": all_contexts
            }
        
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "summary": f"Fout bij het genereren van samenvatting: {str(e)}",
                "error": str(e)
            }
    
    def check_compliance(
        self,
        contract_type: str = "koopovereenkomst"
    ) -> Dict[str, Any]:
        """
        Check contract compliance tegen Nederlandse wetgeving
        """
        logger.info(f"Checking compliance for {contract_type}")
        
        try:
            # Get compliance rules for contract type
            rules = COMPLIANCE_RULES.get(contract_type, [])
            if not rules:
                return {
                    "error": f"Geen compliance regels gevonden voor {contract_type}",
                    "available_types": list(COMPLIANCE_RULES.keys())
                }
            
            compliance_results = []
            overall_compliance = 0
            
            for rule in rules:
                # Search for relevant context
                docs = self.vector_store.similarity_search(rule["search_query"], k=3)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Check compliance for this rule
                prompt = self.compliance_prompt.format(
                    rule_description=rule["description"],
                    rule_requirement=rule["requirement"],
                    rule_mandatory="Ja" if rule["mandatory"] else "Nee",
                    context=context
                )
                
                result = self.llm.invoke(prompt)
                
                # Determine compliance status
                compliance_status = "ONDUIDELIJK"
                if "VOLDOET NIET" in result.upper():
                    compliance_status = "VOLDOET NIET"
                    compliance_score = 0
                elif "VOLDOET" in result.upper():
                    compliance_status = "VOLDOET"
                    compliance_score = 1
                else:
                    compliance_score = 0.5
                
                overall_compliance += compliance_score
                
                compliance_results.append({
                    "rule_id": rule["id"],
                    "rule_description": rule["description"],
                    "compliance_status": compliance_status,
                    "compliance_score": compliance_score,
                    "mandatory": rule["mandatory"],
                    "analysis": result,
                    "sources": [doc.page_content for doc in docs]
                })
            
            # Calculate overall compliance percentage
            overall_percentage = (overall_compliance / len(rules)) * 100 if rules else 0
            
            return {
                "contract_type": contract_type,
                "overall_compliance": overall_percentage,
                "total_rules_checked": len(rules),
                "compliant_rules": sum(1 for r in compliance_results if r["compliance_score"] == 1),
                "non_compliant_rules": sum(1 for r in compliance_results if r["compliance_score"] == 0),
                "results": compliance_results,
                "analysis_date": datetime.now().isoformat(),
                "recommendations": self._generate_compliance_recommendations(compliance_results)
            }
        
        except Exception as e:
            logger.error(f"Error in compliance check: {str(e)}")
            return {
                "error": f"Fout bij compliance check: {str(e)}",
                "contract_type": contract_type
            }
    
    def _generate_compliance_recommendations(self, compliance_results: List[Dict]) -> List[str]:
        """
        Genereer aanbevelingen gebaseerd op compliance resultaten
        """
        recommendations = []
        
        for result in compliance_results:
            if result["compliance_score"] < 1 and result["mandatory"]:
                recommendations.append(
                    f"🔴 URGENT: {result['rule_description']} - vereist onmiddellijke actie"
                )
            elif result["compliance_score"] < 1:
                recommendations.append(
                    f"🟡 AANBEVOLEN: {result['rule_description']} - overweeg aanpassing"
                )
        
        if not recommendations:
            recommendations.append("✅ Contract voldoet aan alle gecontroleerde compliance regels")
        
        return recommendations
    
    def intelligent_search(
        self,
        query: str,
        search_type: str = "hybrid",
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """
        Intelligente zoekfunctie met verschillende strategieën
        """
        logger.debug(f"Intelligent search: '{query}' (type: {search_type})")
        
        if search_type == "semantic":
            return self.vector_store.similarity_search(query, k=5, filter=filters)
        
        elif search_type == "hybrid":
            results = self.vector_store.hybrid_search(query, k=5)
            return [doc for doc, score in results]
        
        elif search_type == "section_specific":
            # Determine most relevant section based on query
            section = self._determine_relevant_section(query)
            return self.advanced_search.search_by_section(query, section, k=5)
        
        elif search_type == "high_quality":
            return self.advanced_search.search_high_quality_chunks(query, min_quality=0.7, k=5)
        
        else:
            return self.vector_store.similarity_search(query, k=5)
    
    def _determine_relevant_section(self, query: str) -> str:
        """
        Bepaal meest relevante contractsectie voor een query
        """
        query_lower = query.lower()
        
        section_keywords = {
            "prijs": ["koopsom", "prijs", "bedrag", "euro", "betaling", "kosten"],
            "partijen": ["verkoper", "koper", "partijen", "naam", "adres"],
            "object": ["woning", "appartement", "object", "onroerend", "gelegen"],
            "voorwaarden": ["voorwaarden", "bepalingen", "clausule", "artikel"],
            "ontbinding": ["ontbinding", "nietigheid", "beëindiging", "opzegging"],
            "garanties": ["garanties", "waarborgen", "staat", "onderhoud", "gebreken"],
            "termijnen": ["termijn", "datum", "deadline", "levering", "transport"]
        }
        
        best_section = "full_document"
        max_matches = 0
        
        for section, keywords in section_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                best_section = section
        
        return best_section
    
    def get_contract_insights(self) -> Dict[str, Any]:
        """
        Krijg algemene insights over het contract
        """
        try:
            stats = self.vector_store.get_collection_stats()
            
            # Get sample of documents to analyze
            sample_docs = self.vector_store.similarity_search("contract overeenkomst", k=10)
            
            # Analyze contract characteristics
            contract_types = set()
            sections_found = set()
            semantic_tags = set()
            
            for doc in sample_docs:
                metadata = doc.metadata
                contract_types.add(metadata.get("contract_type", "unknown"))
                sections_found.add(metadata.get("section", "unknown"))
                tags = metadata.get("semantic_tags", [])
                semantic_tags.update(tags)
            
            return {
                "collection_stats": stats,
                "contract_types": list(contract_types),
                "sections_available": list(sections_found),
                "semantic_tags": list(semantic_tags),
                "total_documents_analyzed": len(sample_docs),
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting contract insights: {str(e)}")
            return {"error": str(e)}

class ContractConversation:
    """
    Beheer conversaties over contracten met context awareness
    """
    
    def __init__(self, analyzer_chain: ContractAnalyzerChain):
        self.analyzer = analyzer_chain
        self.conversation_history = []
        self.context_memory = {}
    
    def chat(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Conversational interface met context awareness
        """
        # Analyze message intent
        intent = self._analyze_intent(message)
        
        # Route to appropriate handler
        if intent == "question":
            response = self.analyzer.ask_question(message)
        elif intent == "risk_analysis":
            response = self._handle_risk_request(message)
        elif intent == "summary":
            response = self.analyzer.generate_summary()
        elif intent == "compliance":
            response = self._handle_compliance_request(message)
        else:
            response = self.analyzer.ask_question(message)  # Default to Q&A
        
        # Update conversation history
        self.conversation_history.append({
            "user_message": message,
            "response": response,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            **response,
            "intent": intent,
            "conversation_id": len(self.conversation_history)
        }
    
    def _analyze_intent(self, message: str) -> str:
        """
        Bepaal de intent van een gebruikersbericht
        """
        message_lower = message.lower()
        
        # Risk analysis intent
        if any(word in message_lower for word in ["risico", "gevaar", "analyseer", "bedreigingen"]):
            return "risk_analysis"
        
        # Summary intent
        if any(word in message_lower for word in ["samenvatting", "overzicht", "kernpunten", "summary"]):
            return "summary"
        
        # Compliance intent
        if any(word in message_lower for word in ["compliance", "wetgeving", "voldoet", "regels"]):
            return "compliance"
        
        # Default to question
        return "question"
    
    def _handle_risk_request(self, message: str) -> Dict[str, Any]:
        """
        Handle risk analysis requests
        """
        # Extract contract type and client role from message or use defaults
        contract_type = "koopovereenkomst"  # Default
        client_role = "koper"  # Default
        
        if "huur" in message.lower():
            contract_type = "huurovereenkomst"
            client_role = "huurder"
        elif "verkoper" in message.lower():
            client_role = "verkoper"
        
        return self.analyzer.analyze_contract_risks(contract_type, client_role)
    
    def _handle_compliance_request(self, message: str) -> Dict[str, Any]:
        """
        Handle compliance check requests
        """
        contract_type = "koopovereenkomst"  # Default
        
        if "huur" in message.lower():
            contract_type = "huurovereenkomst"
        
        return self.analyzer.check_compliance(contract_type)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Krijg samenvatting van de conversatie
        """
        if not self.conversation_history:
            return {"message": "Geen conversatie geschiedenis beschikbaar"}
        
        intents = [conv["intent"] for conv in self.conversation_history]
        intent_counts = {intent: intents.count(intent) for intent in set(intents)}
        
        return {
            "total_messages": len(self.conversation_history),
            "intent_breakdown": intent_counts,
            "first_message_time": self.conversation_history[0]["timestamp"],
            "last_message_time": self.conversation_history[-1]["timestamp"],
            "conversation_flow": [conv["intent"] for conv in self.conversation_history]
        }

# Utility functions
def create_analyzer_from_documents(documents: List[Document], use_pinecone: bool = False) -> ContractAnalyzerChain:
    """
    Create een complete analyzer chain van een lijst documenten
    """
    from utils.vector_store import VectorStoreManager
    
    # Initialize vector store
    vector_store = VectorStoreManager(use_pinecone=use_pinecone)
    
    # Add documents
    vector_store.add_documents(documents)
    
    # Create analyzer
    analyzer = ContractAnalyzerChain(vector_store)
    
    return analyzer

def validate_gemini_setup() -> bool:
    """
    Valideer of Gemini API correct is geconfigureerd
    """
    try:
        genai.configure(api_key=GeminiConfig.API_KEY)
        
        # Test API call with correct model name
        model = genai.GenerativeModel(GeminiConfig.MODEL_NAME)  # Use config model name
        response = model.generate_content("Test")
        
        logger.info("✅ Gemini API setup validated successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ Gemini API validation failed: {str(e)}")
        return False

# Example usage and testing
if __name__ == "__main__":
    # Test the analyzer chain
    logger.info("Testing ContractAnalyzerChain with Gemini...")
    
    try:
        # Validate Gemini setup first
        if not validate_gemini_setup():
            raise Exception("Gemini API setup validation failed")
        
        # Create test documents
        test_docs = [
            Document(
                page_content="""
                KOOPOVEREENKOMST
                
                Partijen:
                Verkoper: Jan de Vries, wonende te Amsterdam
                Koper: Maria Jansen, wonende te Utrecht
                
                Object: Woning gelegen aan de Damrak 123, Amsterdam
                Koopsom: € 750.000 (zevenhonderdvijftigduizend euro)
                
                Ontbindende voorwaarden:
                1. Financieringsvoorbehoud tot 31 maart 2024
                2. Bouwkundige keuring binnen 14 dagen
                
                De verkoper garandeert dat de woning vrij is van verborgen gebreken.
                """,
                metadata={
                    "contract_type": "koopovereenkomst",
                    "section": "full_document",
                    "semantic_tags": ["financial", "legal", "conditional"]
                }
            )
        ]
        
        # Create analyzer
        analyzer = create_analyzer_from_documents(test_docs, use_pinecone=False)
        
        # Test Q&A
        logger.info("Testing Q&A functionality...")
        qa_result = analyzer.ask_question("Wat is de koopsom van de woning?")
        logger.info(f"Q&A Response: {qa_result['answer'][:100]}...")
        
        # Test risk analysis
        logger.info("Testing risk analysis...")
        risk_result = analyzer.analyze_contract_risks("koopovereenkomst", "koper")
        logger.info(f"Risk analysis completed. Score: {risk_result.get('risk_score', 'N/A')}")
        
        # Test summary
        logger.info("Testing summary generation...")
        summary_result = analyzer.generate_summary()
        logger.info(f"Summary generated. Score: {summary_result.get('contract_score', 'N/A')}")
        
        # Test compliance
        logger.info("Testing compliance check...")
        compliance_result = analyzer.check_compliance("koopovereenkomst")
        logger.info(f"Compliance: {compliance_result.get('overall_compliance', 'N/A')}%")
        
        # Test conversation
        logger.info("Testing conversational interface...")
        conversation = ContractConversation(analyzer)
        chat_result = conversation.chat("Vertel me over de risico's van dit contract")
        logger.info(f"Chat response intent: {chat_result.get('intent')}")
        
        logger.info("✅ All ContractAnalyzerChain tests passed!")
    
    except Exception as e:
        logger.error(f"❌ ContractAnalyzerChain test failed: {str(e)}")
        raise