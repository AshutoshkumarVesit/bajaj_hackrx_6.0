# src/utils/multilingual_query_handler.py
import re
import logging
from typing import List, Dict, Any, Tuple
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from .puzzle_handler import PuzzleHandler
from .puzzle_solver import PuzzleSolver

logger = logging.getLogger(__name__)

class MultilingualQueryHandler:
    """
    Core multilingual query processing handler
    Supports Malayalam + English mixed content processing + Puzzle documents
    """
    
    def __init__(self, llm_client=None, faiss_manager=None, enhanced_processor=None, embedding_generator=None):
        """Initialize multilingual query handler"""
        self.llm_client = llm_client
        self.faiss_manager = faiss_manager
        self.enhanced_processor = enhanced_processor
        self.embedding_generator = embedding_generator
        self.puzzle_handler = PuzzleHandler()
        self.puzzle_solver = PuzzleSolver()
        
        # Malayalam Unicode range pattern
        self.malayalam_pattern = re.compile(r'[\u0d00-\u0d7f]')
        
        logger.info("✅ Multilingual query handler initialized with puzzle support")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of the text
        Priority: Malayalam Unicode detection -> langdetect -> English default
        """
        try:
            # Check for Malayalam characters first (most reliable)
            if self.malayalam_pattern.search(text):
                return "malayalam"
            
            # Use langdetect for other languages
            try:
                detected = detect(text)
                if detected in ['en', 'english']:
                    return "english"
                elif detected in ['ml', 'malayalam']:
                    return "malayalam"
                else:
                    return "english"  # Default to English for unknown languages
            except LangDetectException:
                return "english"
                
        except Exception as e:
            logger.warning(f"⚠️ Language detection error: {e}")
            return "english"
    
    def extract_questions(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Extract and analyze questions from queries
        Returns structured question data with language, category, difficulty
        """
        processed_questions = []
        
        for i, query in enumerate(queries):
            try:
                # Detect language
                language = self.detect_language(query)
                
                # Initialize defaults
                category = "general"
                difficulty = "intermediate"
                processed_query = query.strip()
                
                # Use enhanced processor if available
                if self.enhanced_processor:
                    category = self.enhanced_processor.categorize_question(query, language)
                    difficulty = self.enhanced_processor.assess_difficulty(query, category)
                    processed_query = self.enhanced_processor.preprocess_question(query, language)
                
                question_data = {
                    'id': i + 1,
                    'question': processed_query,
                    'original_question': query,
                    'language': language,
                    'category': category,
                    'difficulty': difficulty
                }
                
                processed_questions.append(question_data)
                logger.info(f"📝 Q{i+1}: {language} | {category} | {difficulty}")
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i+1}: {e}")
                # Add basic question data on error
                processed_questions.append({
                    'id': i + 1,
                    'question': query.strip(),
                    'original_question': query,
                    'language': 'english',
                    'category': 'general',
                    'difficulty': 'intermediate'
                })
        
        return processed_questions
    
    def retrieve_context(self, question: str, language: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant context for question using FAISS
        """
        try:
            if not self.faiss_manager:
                logger.warning("⚠️ No FAISS manager available")
                return []
                
            if not self.embedding_generator:
                logger.warning("⚠️ No embedding generator available")
                return []
            
            # Generate embedding for the question
            question_embeddings = self.embedding_generator.get_embeddings([question])
            if not question_embeddings:
                logger.warning(f"⚠️ Failed to generate embeddings for: {question[:50]}...")
                return []
            
            # Use the first (and only) embedding
            query_embedding = question_embeddings[0]
            
            # Check FAISS index status
            total_docs = getattr(self.faiss_manager, 'get_total_documents', lambda: 0)()
            logger.info(f"🔍 FAISS has {total_docs} documents, searching for: {question[:50]}...")
            
            # Search for relevant chunks
            results = self.faiss_manager.search(query_embedding, k=k)
            
            logger.info(f"📊 FAISS search returned {len(results) if results else 0} results")
            
            if not results:
                logger.warning(f"⚠️ No relevant context found for: {question[:50]}...")
                return []
            
            # Extract context text
            contexts = []
            for result in results:
                if isinstance(result, dict) and 'text' in result:
                    contexts.append(result['text'])
                elif isinstance(result, str):
                    contexts.append(result)
            
            logger.info(f"📚 Retrieved {len(contexts)} context chunks for {language} question")
            return contexts
            
        except Exception as e:
            logger.error(f"❌ Error retrieving context: {e}")
            return []
    
    def generate_answer(self, question: str, context: List[str], language: str, category: str = "general") -> str:
        """
        Generate answer using LLM with language-appropriate prompt
        Includes special handling for puzzle documents
        """
        try:
            if not self.llm_client:
                return "LLM client not available"
            
            # Prepare context
            context_text = "\n\n".join(context) if context else "No relevant context found in the document."
            
            # Check if this is a puzzle document  
            if self.puzzle_handler.is_puzzle_document(context_text):
                logger.info("🧩 Detected puzzle document - using puzzle handler")
                
                # Check if it's an interactive puzzle requiring API calls
                if self.puzzle_solver.is_puzzle_document(context_text):
                    logger.info("🔧 Interactive puzzle detected - executing solution")
                    puzzle_response = self.puzzle_solver.generate_puzzle_solution(question)
                    if puzzle_response:
                        return puzzle_response
                
                # Fallback to regular puzzle handler
                puzzle_response = self.puzzle_handler.generate_puzzle_response(question, context_text)
                if puzzle_response:
                    return puzzle_response
            
            # Create language and category-specific prompt
            if language == "malayalam":
                prompt = self._create_malayalam_prompt(question, context_text, category)
            else:
                prompt = self._create_english_prompt(question, context_text, category)
            
            # Generate response
            response = self.llm_client.generate_response(prompt)
            
            if not response or response.strip() == "":
                return "I couldn't generate a proper response for this question based on the provided document."
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _create_english_prompt(self, question: str, context: str, category: str) -> str:
        """Create English language prompt with strict document-only instructions"""
        category_guidance = {
            'insurance_coverage': "Focus on coverage details, limits, and benefits mentioned in the document.",
            'insurance_procedures': "Explain the procedures and requirements as stated in the document.",
            'medical_conditions': "Provide information about medical conditions as covered in the document.",
            'general': "Answer based strictly on the information provided in the document."
        }
        
        guidance = category_guidance.get(category, category_guidance['general'])
        
        return f"""You are an expert document analysis assistant. Answer questions based ONLY on the provided document content.

STRICT RULES:
1. Answer ONLY using information from the provided document context
2. If the document doesn't contain relevant information, say "I don't have enough information in the provided document to answer this question"
3. Be precise, factual, and comprehensive
4. {guidance}
5. Do not add external knowledge or make assumptions
6. Quote specific details from the document when possible

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER (based strictly on the document):"""
    
    def _create_malayalam_prompt(self, question: str, context: str, category: str) -> str:
        """Create Malayalam-aware prompt that can handle Malayalam questions"""
        return f"""You are an expert document analysis assistant that understands both Malayalam and English.

STRICT RULES:
1. Answer ONLY based on the provided document context
2. If the question is in Malayalam, try to provide relevant information (you can respond in English if the document content is in English)
3. If the document doesn't contain relevant information, say "I don't have enough information in the provided document to answer this question"
4. Be precise, factual, and comprehensive
5. Do not add external knowledge or assumptions
6. Focus on providing accurate information from the document

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER (based strictly on the document):"""
    
    def process_multilingual_queries(self, queries: List[str], document_path: str = None) -> Dict[str, Any]:
        """
        Main function to process multiple queries in different languages
        This is the core function that ties everything together
        """
        try:
            logger.info(f"🌐 Processing {len(queries)} multilingual queries")
            
            # Step 1: Extract and analyze questions
            questions = self.extract_questions(queries)
            
            # Step 2: Batch process contexts for all questions (faster)
            logger.info(f"🔍 Retrieving contexts for all {len(questions)} questions...")
            contexts = {}
            for question_data in questions:
                question = question_data['question']
                language = question_data['language']
                q_id = question_data['id']
                
                # Retrieve context (this is the main bottleneck)
                context = self.retrieve_context(question, language, k=5)
                contexts[q_id] = context
            
            # Step 3: Process answers (can be optimized further with batch LLM calls)
            results = []
            
            for question_data in questions:
                question = question_data['question']
                language = question_data['language']
                category = question_data['category']
                q_id = question_data['id']
                
                # Use pre-retrieved context
                context = contexts[q_id]
                
                # Generate answer
                answer = self.generate_answer(question, context, language, category)
                
                # Store result
                result = {
                    'question_id': q_id,
                    'question': question,
                    'original_question': question_data['original_question'],
                    'language': language,
                    'category': category,
                    'difficulty': question_data['difficulty'],
                    'answer': answer,
                    'context_found': len(context) > 0,
                    'context_chunks': len(context)
                }
                
                results.append(result)
            
            # Step 4: Create summary
            languages_used = list(set([r['language'] for r in results]))
            categories_found = list(set([r['category'] for r in results]))
            
            response = {
                'total_questions': len(results),
                'languages_detected': languages_used,
                'categories_found': categories_found,
                'document_path': document_path,
                'results': results,
                'success': True
            }
            
            logger.info(f"🎯 Successfully processed {len(results)} questions")
            logger.info(f"🌐 Languages: {', '.join(languages_used)}")
            logger.info(f"📊 Categories: {', '.join(categories_found)}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing multilingual queries: {e}")
            return {
                'total_questions': len(queries) if queries else 0,
                'error': str(e),
                'success': False,
                'results': []
            }
