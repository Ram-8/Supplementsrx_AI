# src/rag/unified_pipeline.py

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from neo4j import GraphDatabase
from dotenv import load_dotenv

from src.embeddings.semantic_search import SemanticSearcher

# Load environment variables
load_dotenv()

# Use Google Gemini API for LLM curation
USE_GEMINI = False

# Initialize logger (try loguru first, fallback to standard logging)
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Initialize Gemini API
try:
    import google.generativeai as genai
    
    # Get API key from environment variable or use provided key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDhskcx0LJA2GQoEAsq5xpz4k6S-Snd6w4")
    
    if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIza"):
        genai.configure(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
        logger.info("Gemini API configured successfully")
    else:
        logger.warning("Invalid Gemini API key. Please set GEMINI_API_KEY environment variable.")
except ImportError:
    logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")
except Exception as e:
    logger.warning(f"Error configuring Gemini API: {e}")

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "supplements_pass")

@dataclass
class SourceInfo:
    """Information about a source"""
    source_type: str  # "vector_embedding" or "neo4j_kg"
    supplement_name: Optional[str]
    section: Optional[str]
    text: str
    source_url: Optional[str]
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedRAGResult:
    """Unified result from both sources"""
    question: str
    answer: str
    reasoning: str
    sources: List[SourceInfo]
    used_llm_knowledge: bool
    precaution_notice: str


class UnifiedRAGPipeline:
    """
    Unified RAG pipeline that combines:
    1. Vector embeddings (semantic search)
    2. Neo4j knowledge graph
    3. Google Gemini API for LLM curation
    """

    def __init__(
        self,
        searcher: Optional[SemanticSearcher] = None,
        llm_model: str = "gemini-2.0-flash",  # Gemini model name
        top_k_embeddings: int = 5,
        top_k_kg: int = 10,
    ):
        self.searcher = searcher or SemanticSearcher()
        self.top_k_embeddings = top_k_embeddings
        self.top_k_kg = top_k_kg
        
        # Initialize Neo4j driver
        try:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            logger.info("Neo4j connection established")
            
            # Test connection and get available supplements
            try:
                with self.neo4j_driver.session() as session:
                    result = session.run("MATCH (s:Supplement) RETURN count(s) as count")
                    count = result.single()["count"]
                    logger.info(f"Neo4j database contains {count} Supplement nodes")
                    if count == 0:
                        logger.warning("⚠️  No supplements found in Neo4j! Make sure to run load_kg.py to populate the database.")
            except Exception as e:
                logger.warning(f"Could not verify Neo4j data: {e}")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            self.neo4j_driver = None
        
        # Initialize Gemini API LLM
        self.llm = None
        self.llm_model = llm_model
        
        if USE_GEMINI:
            try:
                self.model = genai.GenerativeModel(llm_model)
                self.llm = "gemini"
                logger.info(f"Using Gemini API with model: {llm_model}")
            except Exception as e:
                logger.error(f"Error initializing Gemini model '{llm_model}': {e}")
                # Try default model
                try:
                    self.llm_model = "gemini-pro"
                    self.model = genai.GenerativeModel(self.llm_model)
                    self.llm = "gemini"
                    logger.info(f"Using fallback Gemini model: {self.llm_model}")
                except Exception as e2:
                    logger.error(f"Error initializing fallback Gemini model: {e2}")
        
        if not self.llm:
            logger.warning("Gemini API not available. Will use template-based responses.")

    def _query_neo4j(self, question: str) -> List[Dict[str, Any]]:
        """
        Query Neo4j knowledge graph for relevant information.
        Uses a flexible query that searches for supplements, conditions, and relationships.
        """
        if not self.neo4j_driver:
            logger.warning("Neo4j driver not available, skipping KG query")
            return []

        # Extract key terms from question (improved approach)
        question_lower = question.lower()
        
        # Try to identify if question is about a specific supplement
        supplements = []
        conditions = []
        
        # Expanded supplement list with variations
        supplement_keywords = [
            "berberine", "chromium", "magnesium", "vitamin d", "vitamin b12", "vitamin b 12",
            "zinc", "fish oil", "turmeric", "curcumin", "green tea", "ginger",
            "fenugreek", "gymnema", "bitter melon", "milk thistle", "resveratrol",
            "alpha-lipoic acid", "alpha lipoic acid", "cinnamon", "vanadium",
            "inositol", "banaba", "amla", "indian gooseberry", "l-arginine",
            "l-arginine", "l-carnitine", "taurine", "vitamin e", "biotin", "niacin",
            "alpha lipoic", "vitamin b-12"
        ]
        
        # Also try to get supplement names from Neo4j dynamically (for better matching)
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (s:Supplement) RETURN toLower(s.name) as name LIMIT 50")
                db_supplements = [record["name"] for record in result if record.get("name")]
                # Check if any database supplement names are in the question
                for db_supp in db_supplements:
                    if db_supp and db_supp in question_lower and db_supp not in supplements:
                        supplements.append(db_supp)
                        logger.debug(f"Found supplement '{db_supp}' in question from Neo4j database")
        except Exception as e:
            logger.debug(f"Could not fetch supplements from Neo4j for matching: {e}")
        
        # Check hardcoded keywords
        for supp in supplement_keywords:
            if supp in question_lower and supp not in supplements:
                supplements.append(supp)
        
        # Common conditions
        condition_keywords = [
            "diabetes", "diabetic", "blood sugar", "glucose", "hypertension",
            "high blood pressure", "cholesterol", "hyperlipidemia", "hba1c",
            "a1c", "glycated hemoglobin", "insulin", "metformin", "prediabetes"
        ]
        
        for cond in condition_keywords:
            if cond in question_lower:
                conditions.append(cond)

        results = []
        
        logger.info(f"Querying Neo4j for question: {question}")
        logger.info(f"Detected supplements: {supplements}, conditions: {conditions}")
        
        with self.neo4j_driver.session() as session:
            # Query 1: If supplement mentioned, get supplement details
            if supplements:
                for supp in supplements:
                    # Try multiple query strategies for better matching
                    cypher_queries = [
                        # Exact match (case-insensitive)
                        """
                        MATCH (s:Supplement)
                        WHERE toLower(s.name) = $supplement
                        OPTIONAL MATCH (s)-[r:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(c:Condition)
                        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
                        WITH s, collect(DISTINCT {
                            condition: c.name,
                            relationType: type(r),
                            dosage: r.dosage,
                            frequency: r.frequency,
                            duration: r.duration,
                            notes: r.notes
                        }) AS conditions,
                        collect(DISTINCT {
                            drug: d.name,
                            severity: i.severity,
                            description: i.description
                        }) AS interactions
                        RETURN s.name AS supplement,
                               s.overview_text AS overview,
                               s.effectiveness_text AS effectiveness,
                               s.safety_text AS safety,
                               s.dosing_text AS dosing,
                               s.interactions_text AS interactions_text,
                               s.mechanism_text AS mechanism,
                               conditions,
                               interactions,
                               s.source_url AS source_url
                        LIMIT 1
                        """,
                        # Contains match (fallback)
                        """
                        MATCH (s:Supplement)
                        WHERE toLower(s.name) CONTAINS $supplement
                        OPTIONAL MATCH (s)-[r:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(c:Condition)
                        OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
                        WITH s, collect(DISTINCT {
                            condition: c.name,
                            relationType: type(r),
                            dosage: r.dosage,
                            frequency: r.frequency,
                            duration: r.duration,
                            notes: r.notes
                        }) AS conditions,
                        collect(DISTINCT {
                            drug: d.name,
                            severity: i.severity,
                            description: i.description
                        }) AS interactions
                        RETURN s.name AS supplement,
                               s.overview_text AS overview,
                               s.effectiveness_text AS effectiveness,
                               s.safety_text AS safety,
                               s.dosing_text AS dosing,
                               s.interactions_text AS interactions_text,
                               s.mechanism_text AS mechanism,
                               conditions,
                               interactions,
                               s.source_url AS source_url
                        LIMIT 1
                        """
                    ]
                    
                    for cypher in cypher_queries:
                        try:
                            result = session.run(cypher, supplement=supp.lower())
                            record = result.single()
                            if record:
                                # Format the result
                                kg_data = {
                                    "supplement": record.get("supplement"),
                                    "overview": record.get("overview", ""),
                                    "effectiveness": record.get("effectiveness", ""),
                                    "safety": record.get("safety", ""),
                                    "dosing": record.get("dosing", ""),
                                    "interactions_text": record.get("interactions_text", ""),
                                    "mechanism": record.get("mechanism", ""),
                                    "conditions": record.get("conditions", []),
                                    "interactions": record.get("interactions", []),
                                    "source_url": record.get("source_url"),
                                }
                                results.append(kg_data)
                                logger.info(f"Found supplement in Neo4j: {kg_data.get('supplement')}")
                                break  # Found a match, no need to try other queries
                        except Exception as e:
                            logger.warning(f"Error querying Neo4j for supplement '{supp}': {e}")
                            continue

            # Query 2: If condition mentioned, get supplements for that condition
            if conditions:
                for cond in conditions:
                    cypher = """
                    MATCH (c:Condition)
                    WHERE toLower(c.name) CONTAINS $condition
                    MATCH (s:Supplement)-[r:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(c)
                    OPTIONAL MATCH (s)-[dg:DOSAGE_GUIDELINE_FOR]->(c)
                    OPTIONAL MATCH (s)-[i:INTERACTS_WITH]->(d:Drug)
                    WITH c, s, r,
                         collect(DISTINCT {
                             dosage: dg.dosage,
                             frequency: dg.frequency,
                             duration: dg.duration,
                             notes: dg.notes
                         }) AS dosage_guidelines,
                         collect(DISTINCT {
                             drug: d.name,
                             severity: i.severity,
                             description: i.description
                         }) AS interactions
                    RETURN c.name AS condition,
                           s.name AS supplement,
                           type(r) AS relationType,
                           dosage_guidelines,
                           interactions,
                           s.source_url AS source_url,
                           s.overview_text AS overview,
                           s.effectiveness_text AS effectiveness,
                           s.safety_text AS safety,
                           s.dosing_text AS dosing
                    LIMIT $limit
                    """
                    try:
                        result = session.run(cypher, condition=cond.lower(), limit=self.top_k_kg)
                        count = 0
                        for record in result:
                            kg_data = {
                                "condition": record.get("condition"),
                                "supplement": record.get("supplement"),
                                "relationType": record.get("relationType"),
                                "dosage_guidelines": record.get("dosage_guidelines", []),
                                "interactions": record.get("interactions", []),
                                "source_url": record.get("source_url"),
                                "overview": record.get("overview", ""),
                                "effectiveness": record.get("effectiveness", ""),
                                "safety": record.get("safety", ""),
                                "dosing": record.get("dosing", ""),
                            }
                            results.append(kg_data)
                            count += 1
                        if count > 0:
                            logger.info(f"Found {count} supplements for condition '{cond}' in Neo4j")
                    except Exception as e:
                        logger.warning(f"Error querying Neo4j for condition '{cond}': {e}")

            # Query 3: General search if no specific supplement/condition found
            # Also try general search even if we found supplements/conditions to get more context
            if not supplements and not conditions:
                # Extract key words from question (longer words are usually more meaningful)
                query_words = " ".join([w for w in question.split() if len(w) > 3][:5])
                if query_words:
                    cypher = """
                        MATCH (s:Supplement)
                        WHERE toLower(s.overview_text) CONTAINS toLower($query)
                             OR toLower(s.effectiveness_text) CONTAINS toLower($query)
                             OR toLower(s.safety_text) CONTAINS toLower($query)
                             OR toLower(s.dosing_text) CONTAINS toLower($query)
                             OR toLower(s.name) CONTAINS toLower($query)
                        RETURN s.name AS supplement,
                               s.overview_text AS overview,
                               s.effectiveness_text AS effectiveness,
                               s.safety_text AS safety,
                               s.dosing_text AS dosing,
                               s.source_url AS source_url
                        LIMIT $limit
                        """
                    try:
                        result = session.run(cypher, query=query_words, limit=self.top_k_kg)
                        count = 0
                        for record in result:
                            kg_data = {
                                "supplement": record.get("supplement"),
                                "overview": record.get("overview", ""),
                                "effectiveness": record.get("effectiveness", ""),
                                "safety": record.get("safety", ""),
                                "dosing": record.get("dosing", ""),
                                "source_url": record.get("source_url"),
                            }
                            results.append(kg_data)
                            count += 1
                        if count > 0:
                            logger.info(f"Found {count} supplements via general search in Neo4j")
                    except Exception as e:
                        logger.warning(f"Error in general Neo4j search: {e}")
            
            # Query 4: If we still have no results, try to get any supplements related to diabetes
            if not results and ("diabetes" in question_lower or "diabetic" in question_lower):
                cypher = """
                    MATCH (c:Condition)
                    WHERE toLower(c.name) CONTAINS 'diabetes' OR toLower(c.name) CONTAINS 'diabetic'
                    MATCH (s:Supplement)-[r:MENTIONED_FOR|DOSAGE_GUIDELINE_FOR]->(c)
                    RETURN DISTINCT s.name AS supplement,
                           s.overview_text AS overview,
                           s.effectiveness_text AS effectiveness,
                           s.safety_text AS safety,
                           s.dosing_text AS dosing,
                           s.source_url AS source_url,
                           c.name AS condition
                    LIMIT $limit
                """
                try:
                    result = session.run(cypher, limit=self.top_k_kg)
                    count = 0
                    for record in result:
                        kg_data = {
                            "supplement": record.get("supplement"),
                            "condition": record.get("condition"),
                            "overview": record.get("overview", ""),
                            "effectiveness": record.get("effectiveness", ""),
                            "safety": record.get("safety", ""),
                            "dosing": record.get("dosing", ""),
                            "source_url": record.get("source_url"),
                        }
                        results.append(kg_data)
                        count += 1
                    if count > 0:
                        logger.info(f"Found {count} diabetes-related supplements via fallback query in Neo4j")
                except Exception as e:
                    logger.warning(f"Error in fallback Neo4j query: {e}")

        if not results:
            logger.warning(f"No results found in Neo4j for question: {question}")
        else:
            logger.info(f"Found {len(results)} result(s) in Neo4j")

        return results

    def _format_kg_context(self, kg_results: List[Dict[str, Any]]) -> str:
        """Format Neo4j results into readable context"""
        if not kg_results:
            logger.warning("No KG results to format")
            return "No information found in knowledge graph."

        blocks = []
        for i, result in enumerate(kg_results, 1):
            block = f"[Knowledge Graph Result {i}]\n"
            
            if result.get("supplement"):
                block += f"Supplement: {result['supplement']}\n"
            
            if result.get("condition"):
                block += f"Condition: {result['condition']}\n"
            
            # Include text fields (these are important for LLM)
            if result.get("overview"):
                overview_text = result['overview'].strip()
                if overview_text:
                    block += f"Overview: {overview_text[:800]}{'...' if len(overview_text) > 800 else ''}\n"
            
            if result.get("effectiveness"):
                eff_text = result['effectiveness'].strip()
                if eff_text:
                    block += f"Effectiveness: {eff_text[:800]}{'...' if len(eff_text) > 800 else ''}\n"
            
            if result.get("safety"):
                safety_text = result['safety'].strip()
                if safety_text:
                    block += f"Safety: {safety_text[:800]}{'...' if len(safety_text) > 800 else ''}\n"
            
            if result.get("dosing"):
                dosing_text = result['dosing'].strip()
                if dosing_text:
                    block += f"Dosing: {dosing_text[:500]}{'...' if len(dosing_text) > 500 else ''}\n"
            
            if result.get("interactions_text"):
                interactions_text = result['interactions_text'].strip()
                if interactions_text:
                    block += f"Drug Interactions Text: {interactions_text[:500]}{'...' if len(interactions_text) > 500 else ''}\n"
            
            if result.get("mechanism"):
                mechanism_text = result['mechanism'].strip()
                if mechanism_text:
                    block += f"Mechanism: {mechanism_text[:500]}{'...' if len(mechanism_text) > 500 else ''}\n"
            
            # Structured data
            if result.get("dosage_guidelines") and isinstance(result['dosage_guidelines'], list) and len(result['dosage_guidelines']) > 0:
                # Filter out None/empty entries
                valid_guidelines = [dg for dg in result['dosage_guidelines'] if dg and isinstance(dg, dict) and any(dg.values())]
                if valid_guidelines:
                    block += f"Dosage Guidelines: {json.dumps(valid_guidelines, indent=2)}\n"
            
            if result.get("interactions") and isinstance(result['interactions'], list) and len(result['interactions']) > 0:
                # Filter out None/empty entries
                valid_interactions = [inter for inter in result['interactions'] if inter and isinstance(inter, dict) and any(inter.values())]
                if valid_interactions:
                    block += f"Drug Interactions (Structured): {json.dumps(valid_interactions, indent=2)}\n"
            
            if result.get("conditions") and isinstance(result['conditions'], list) and len(result['conditions']) > 0:
                # Filter out None/empty entries
                valid_conditions = [c for c in result['conditions'] if c and isinstance(c, dict) and any(c.values())]
                if valid_conditions:
                    block += f"Related Conditions: {json.dumps(valid_conditions, indent=2)}\n"
            
            if result.get("source_url"):
                block += f"Source: {result['source_url']}\n"
            
            blocks.append(block)
        
        formatted = "\n\n---\n\n".join(blocks)
        logger.info(f"Formatted KG context: {len(formatted)} characters")
        return formatted

    def _format_embedding_context(self, embedding_hits: List[Dict[str, Any]]) -> str:
        """Format vector embedding results into readable context"""
        if not embedding_hits:
            return "No information found in vector embeddings."

        blocks = []
        for i, hit in enumerate(embedding_hits, 1):
            name = hit.get("supplement_name") or "Unknown supplement"
            section = hit.get("section", "unknown")
            text = hit.get("text", "")
            src = hit.get("source_url") or "N/A"
            score = hit.get("score", 0.0)

            blocks.append(
                f"[Vector Embedding Result {i}] (Relevance Score: {score:.3f})\n"
                f"Supplement: {name}\n"
                f"Section: {section}\n"
                f"Source: {src}\n"
                f"Text: {text[:500]}...\n"
            )
        
        return "\n\n---\n\n".join(blocks)

    def _is_yes_no_question(self, question: str) -> bool:
        """
        Detect if a question is a yes/no type question.
        Checks for common yes/no question patterns.
        """
        question_lower = question.lower().strip()
        
        # Common yes/no question starters
        yes_no_starters = [
            "is ", "are ", "was ", "were ", "do ", "does ", "did ", 
            "can ", "could ", "should ", "would ", "will ", "has ", 
            "have ", "had ", "may ", "might ", "must "
        ]
        
        # Check if question starts with a yes/no starter
        for starter in yes_no_starters:
            if question_lower.startswith(starter):
                return True
        
        # Check for question patterns with "?" that suggest yes/no
        if "?" in question:
            # Questions asking about existence, possibility, or truth
            yes_no_patterns = [
                "is there", "are there", "does it", "do they",
                "can you", "should i", "will it", "would it"
            ]
            for pattern in yes_no_patterns:
                if pattern in question_lower:
                    return True
        
        return False

    def _call_llm(self, question: str, embedding_context: str, kg_context: str, has_embedding_info: bool = None, has_kg_info: bool = None) -> Dict[str, str]:
        """
        Call LLM to curate and combine information from both sources.
        Returns dict with 'answer', 'reasoning', 'used_llm_knowledge', 'sources_used'
        """
        
        # Determine if we have sufficient information (if not provided)
        if has_embedding_info is None:
            has_embedding_info = len(embedding_context) > 100 and "No information" not in embedding_context
        if has_kg_info is None:
            has_kg_info = len(kg_context) > 100 and "No information" not in kg_context
        
        # Check if this is a yes/no question
        is_yes_no = self._is_yes_no_question(question)
        
        system_prompt = """You are an expert medical assistant that answers diabetes-related questions based on the results provided from the vector embeddings and neo4j knowledge graph methods.

IMPORTANT: You will receive information from TWO sources:
1. Vector Embeddings (Semantic Search) - labeled as "Information from Vector Embeddings"
2. Knowledge Graph (Structured Relationships) - labeled as "Information from Knowledge Graph"

You MUST use information from BOTH sources when available. If one source says "No information found", you should still use the other source. 

Your role:
1. Synthesize information from BOTH vector embeddings (semantic search) AND knowledge graph (structured relationships) when both are available
2. If only one source has information, use that source. If both have information, combine them intelligently.
3. First directly mention the fact, then mention the context behind that fact based on these 2 method's results
4. CRITICAL: You must explicitly mention exactly what information is taken from vector embeddings and what is taken from knowledge graph. For each piece of information in your answer, clearly indicate its source using inline citations [VE] for vector embeddings and [KG] for knowledge graph
5. Also cite all sources at the end with detailed breakdown, exactly mentioning which specific information came from knowledge graph and which came from vector embeddings
6. Provide clear, accurate, and evidence-based answers (tone should be similar to how a human says it)
7. Always include a precaution notice about consulting healthcare providers
8. Use LLM knowledge only when the information from BOTH sources are insufficient to answer the query
9. The reasoning should be in third person context tone, like "this information...." and not "i did this" or "i did that"
10. If you see "No information found in knowledge graph" or "No information found in vector embeddings", that means that particular source had no relevant data - use the other source if available

Response format:
- Provide a clear, structured answer that first states the fact, then provides context
- For EVERY piece of information, explicitly mark its source: [VE] for vector embeddings, [KG] for knowledge graph
- At the end, provide a detailed breakdown: "Information from Vector Embeddings: [list exactly what came from VE]" and "Information from Knowledge Graph: [list exactly what came from KG]"
- Explain your reasoning in third person tone
- End with a precaution notice
- CRITICAL: If the question is a yes/no question (starts with Is, Are, Do, Does, Can, Should, etc.), your answer MUST start with either Yes or No followed by a comma, then provide the explanation. For example: Yes, [explanation] or No, [explanation]"""

        # Ensure both contexts are included even if one is empty
        if not has_kg_info:
            logger.warning("No knowledge graph information available - LLM will rely on vector embeddings only")
        if not has_embedding_info:
            logger.warning("No vector embedding information available - LLM will rely on knowledge graph only")
        
        # Always include both contexts in the prompt, even if empty
        # This ensures the LLM knows what sources were checked
        user_prompt = f"""User Question: {question}

Information from Vector Embeddings (Semantic Search):
{embedding_context}

Information from Knowledge Graph (Structured Relationships):
{kg_context}

Please provide:
1. A comprehensive answer to the user's question - first state the fact directly, then provide context from the sources
2. Your reasoning for how you combined the information (in third person tone)
3. Which sources you primarily used (vector embeddings, knowledge graph, or general knowledge)
4. A precaution notice

CRITICAL REQUIREMENTS:
- Use information from BOTH sources when available. If one source says "No information found", use the other source.
- Only use your general knowledge if BOTH sources are insufficient (i.e., both say "No information found")
- In your answer, cite sources inline using brackets: [KG] for knowledge graph, [VE] for vector embeddings
- You MUST explicitly mention exactly what information came from vector embeddings and what came from knowledge graph
- If a source says "No information found", do NOT cite it - only cite sources that actually provided information
- At the end of your answer, provide a clear breakdown section that lists:
  * "Information from Vector Embeddings: [exact list of what information came from VE, or 'None' if no VE info was available]"
  * "Information from Knowledge Graph: [exact list of what information came from KG, or 'None' if no KG info was available]"
- Be specific and detailed about which facts, data points, or statements originated from which source
{f"- CRITICAL FOR YES/NO QUESTIONS: This is a yes/no question. Your answer MUST start with either 'Yes' or 'No' followed by a comma, then provide the explanation. Example: 'Yes, [explanation with citations]' or 'No, [explanation with citations]'. Do NOT start with any other word." if is_yes_no else ""}

Format your response as JSON with these keys:
- "answer": The main answer (with inline citations using [KG] and [VE] brackets throughout, and a detailed breakdown at the end showing exactly what came from each source)
- "reasoning": How you combined the sources (in third person context tone), explicitly mentioning what information was taken from vector embeddings and what from knowledge graph
- "sources_used": List of sources used (e.g., ["vector_embeddings", "knowledge_graph"])
- "precaution_notice": A safety notice"""

        if self.llm == "gemini":
            try:
                # Combine system prompt and user prompt for Gemini
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                # Call Gemini API
                response = self.model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.3,
                        "top_p": 0.9,
                    }
                )
                
                response_text = response.text if response.text else ""
                
                # Try to parse JSON from response
                try:
                    # Extract JSON if wrapped in markdown code blocks
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()
                    elif "```" in response_text:
                        json_start = response_text.find("```") + 3
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()
                    
                    result = json.loads(response_text)
                    
                    # Post-process: Ensure yes/no questions start with Yes/No
                    if is_yes_no and "answer" in result:
                        answer = result["answer"].strip()
                        answer_lower = answer.lower()
                        
                        # Check if answer doesn't start with Yes or No
                        if not (answer_lower.startswith("yes") or answer_lower.startswith("no")):
                            # Try to extract Yes/No from the answer or determine from context
                            # For now, we'll prepend based on positive/negative indicators
                            if any(word in answer_lower[:50] for word in ["helpful", "effective", "beneficial", "recommended", "safe", "good", "works", "can help"]):
                                if not answer_lower.startswith("yes"):
                                    result["answer"] = "Yes, " + answer
                            elif any(word in answer_lower[:50] for word in ["not helpful", "not effective", "not recommended", "unsafe", "harmful", "should not", "cannot", "does not"]):
                                if not answer_lower.startswith("no"):
                                    result["answer"] = "No, " + answer
                            else:
                                # Default: if we can't determine, start with a neutral response
                                # But since it's a yes/no question, we should still try to determine
                                # For safety, we'll add a note but keep the original answer
                                pass
                
                except json.JSONDecodeError:
                    # Fallback: parse manually if JSON parsing fails
                    # Try to extract structured information from text response
                    sources_used = []
                    if has_embedding_info:
                        sources_used.append("vector_embeddings")
                    if has_kg_info:
                        sources_used.append("knowledge_graph")
                    
                    answer_text = response_text
                    
                    # Post-process: Ensure yes/no questions start with Yes/No
                    if is_yes_no:
                        answer_lower = answer_text.lower().strip()
                        if not (answer_lower.startswith("yes") or answer_lower.startswith("no")):
                            # Try to determine from content
                            if any(word in answer_lower[:100] for word in ["helpful", "effective", "beneficial", "recommended", "safe", "good", "works", "can help"]):
                                answer_text = "Yes, " + answer_text
                            elif any(word in answer_lower[:100] for word in ["not helpful", "not effective", "not recommended", "unsafe", "harmful", "should not", "cannot", "does not"]):
                                answer_text = "No, " + answer_text
                    
                    result = {
                        "answer": answer_text,
                        "reasoning": "This information was combined from both vector embeddings and knowledge graph sources using the Gemini API",
                        "sources_used": sources_used if sources_used else ["general_knowledge"],
                        "precaution_notice": "⚠️ IMPORTANT: This information is for educational purposes only. Always consult with your healthcare provider before taking any dietary supplements, especially if you have medical conditions, are pregnant, breastfeeding, or taking medications."
                    }
                    
                    # Add optional fields if they exist in response
                    if "natmed_sections" in response_text or "used_llm_knowledge" in response_text:
                        # Try to extract if mentioned in text
                        pass
                
                # Final check: Ensure yes/no questions start with Yes/No
                if is_yes_no and "answer" in result:
                    answer = result["answer"].strip()
                    answer_lower = answer.lower()
                    if not (answer_lower.startswith("yes") or answer_lower.startswith("no")):
                        # If still doesn't start with Yes/No, try to prepend based on content
                        if any(word in answer_lower[:100] for word in ["helpful", "effective", "beneficial", "recommended", "safe", "good", "works", "can help", "may help", "might help"]):
                            result["answer"] = "Yes, " + answer
                        elif any(word in answer_lower[:100] for word in ["not helpful", "not effective", "not recommended", "unsafe", "harmful", "should not", "cannot", "does not", "not safe"]):
                            result["answer"] = "No, " + answer
                
                return result
            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
                # Fallback response
                return self._fallback_response(question, embedding_context, kg_context, has_embedding_info, has_kg_info, is_yes_no)
        else:
            # Fallback: simple template-based response
            logger.warning("Gemini API not available, using template-based response")
            return self._fallback_response(question, embedding_context, kg_context, has_embedding_info, has_kg_info, is_yes_no)

    def _fallback_response(
        self, 
        question: str, 
        embedding_context: str, 
        kg_context: str,
        has_embedding_info: bool,
        has_kg_info: bool,
        is_yes_no: bool = False
    ) -> Dict[str, Any]:
        """Fallback response when LLM is not available"""
        sources_used = []
        if has_embedding_info:
            sources_used.append("vector_embeddings")
        if has_kg_info:
            sources_used.append("knowledge_graph")
        
        combined_context = ""
        if has_embedding_info:
            combined_context += f"From Vector Embeddings:\n{embedding_context[:1000]}\n\n"
        if has_kg_info:
            combined_context += f"From Knowledge Graph:\n{kg_context[:1000]}\n\n"
        
        answer = f"Based on the available information:\n\n{combined_context}\n\n"
        if not sources_used:
            answer += "Note: Limited information available. Please consult with a healthcare provider for personalized advice."
            used_llm_knowledge = True
        else:
            used_llm_knowledge = False
        
        # For yes/no questions, try to start with Yes/No
        if is_yes_no:
            answer_lower = answer.lower()
            if not (answer_lower.startswith("yes") or answer_lower.startswith("no")):
                # Try to determine from context
                if any(word in answer_lower[:200] for word in ["helpful", "effective", "beneficial", "recommended", "safe", "good", "works", "can help", "may help"]):
                    answer = "Yes, " + answer
                elif any(word in answer_lower[:200] for word in ["not helpful", "not effective", "not recommended", "unsafe", "harmful", "should not", "cannot", "does not", "not safe"]):
                    answer = "No, " + answer
                else:
                    # Default to a neutral response but still try to format as yes/no
                    answer = "Based on available information, " + answer
        
        return {
            "answer": answer,
            "reasoning": f"This information was combined from: {', '.join(sources_used) if sources_used else 'general knowledge'}. The sources were merged to provide a comprehensive answer.",
            "sources_used": sources_used if sources_used else ["general_knowledge"],
            "precaution_notice": "⚠️ IMPORTANT: This information is for educational purposes only. Always consult with your healthcare provider before taking any dietary supplements, especially if you have medical conditions, are pregnant, breastfeeding, or taking medications."
        }

    def run(self, question: str) -> UnifiedRAGResult:
        """
        Main pipeline: retrieve from both sources and curate with LLM
        """
        # 1. Get vector embeddings
        embedding_bundle = self.searcher.best_answer(question, top_k=self.top_k_embeddings)
        embedding_hits = embedding_bundle.get("hits", [])
        
        # 2. Get Neo4j knowledge graph results
        kg_results = self._query_neo4j(question)
        
        # 3. Format contexts
        embedding_context = self._format_embedding_context(embedding_hits)
        kg_context = self._format_kg_context(kg_results)
        
        # Log context availability
        has_embedding_info = len(embedding_context) > 100 and "No information" not in embedding_context
        has_kg_info = len(kg_context) > 100 and "No information" not in kg_context
        logger.info(f"Context availability - Embeddings: {has_embedding_info}, KG: {has_kg_info}")
        logger.info(f"Embedding context length: {len(embedding_context)}, KG context length: {len(kg_context)}")
        
        # 4. Call LLM to curate (pass context availability flags)
        llm_result = self._call_llm(question, embedding_context, kg_context, has_embedding_info, has_kg_info)
        
        # 5. Build source list
        sources = []
        
        # Add embedding sources
        for hit in embedding_hits:
            sources.append(SourceInfo(
                source_type="vector_embedding",
                supplement_name=hit.get("supplement_name"),
                section=hit.get("section"),
                text=hit.get("text", "")[:200],
                source_url=hit.get("source_url"),
                score=hit.get("score"),
                metadata={"chunk_index": hit.get("chunk_index")}
            ))
        
        # Add KG sources
        for kg_result in kg_results:
            # Extract text from KG result
            kg_text = ""
            if kg_result.get("overview"):
                kg_text += f"Overview: {kg_result['overview'][:200]}... "
            if kg_result.get("effectiveness"):
                kg_text += f"Effectiveness: {kg_result['effectiveness'][:200]}... "
            if kg_result.get("safety"):
                kg_text += f"Safety: {kg_result['safety'][:200]}... "
            
            sources.append(SourceInfo(
                source_type="neo4j_kg",
                supplement_name=kg_result.get("supplement"),
                section=None,
                text=kg_text or "Knowledge graph relationship data",
                source_url=kg_result.get("source_url"),
                score=None,
                metadata=kg_result
            ))
        
        # 6. Build result
        # Determine used_llm_knowledge from sources_used if not explicitly provided
        sources_used = llm_result.get("sources_used", [])
        used_llm_knowledge = llm_result.get("used_llm_knowledge", 
                                           "general_knowledge" in sources_used or len(sources_used) == 0)
        
        return UnifiedRAGResult(
            question=question,
            answer=llm_result.get("answer", ""),
            reasoning=llm_result.get("reasoning", ""),
            sources=sources,
            used_llm_knowledge=used_llm_knowledge,
            precaution_notice=llm_result.get("precaution_notice", "Please consult with your healthcare provider.")
        )

    def run_as_dict(self, question: str) -> Dict[str, Any]:
        """Run pipeline and return as dictionary"""
        result = self.run(question)
        return {
            "answer": result.answer,
            "reasoning": result.reasoning,
            "sources": [asdict(s) for s in result.sources],
            "used_llm_knowledge": result.used_llm_knowledge,
            "precaution_notice": result.precaution_notice,
        }

    def close(self):
        """Close Neo4j connection"""
        if self.neo4j_driver:
            self.neo4j_driver.close()

if __name__ == "__main__":
    pipeline = UnifiedRAGPipeline()
    
    try:
        question = "Is berberine helpful for type 2 diabetes?"
        result = pipeline.run(question)
        print("\n" + "="*80)
        print("QUESTION:", result.question)
        print("\nANSWER:\n", result.answer)
        print("\nREASONING:\n", result.reasoning)
        print("\nSOURCES:", len(result.sources))
        print("\nPRECAUTION NOTICE:\n", result.precaution_notice)
    finally:
        pipeline.close()
