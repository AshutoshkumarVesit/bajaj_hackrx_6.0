# src/utils/multilingual_query_handler.py
import re
import asyncio
from typing import List, Dict, Any

class MultilingualQueryHandler:
    def __init__(self):
        self.malayalam_pattern = re.compile(r'[\u0d00-\u0d7f]+')
        self.translation_cache = {}
        
    def detect_multilingual_context(self, questions: List[str]) -> Dict[str, Any]:
        """Analyze if the question set contains multilingual content"""
        language_stats = {
            'malayalam_count': 0,
            'english_count': 0,
            'mixed_count': 0,
            'question_languages': [],
            'is_multilingual': False
        }
        
        for question in questions:
            lang = self._detect_question_language(question)
            language_stats['question_languages'].append(lang)
            
            if lang == 'malayalam':
                language_stats['malayalam_count'] += 1
            elif lang == 'english':
                language_stats['english_count'] += 1
            else:
                language_stats['mixed_count'] += 1
        
        # Mark as multilingual if we have more than one language
        language_stats['is_multilingual'] = (
            language_stats['malayalam_count'] > 0 and 
            language_stats['english_count'] > 0
        )
        
        return language_stats
    
    def _detect_question_language(self, question: str) -> str:
        """Detect language of individual question"""
        malayalam_chars = len(self.malayalam_pattern.findall(question))
        total_chars = len(re.findall(r'[a-zA-Z\u0d00-\u0d7f]', question))
        
        if total_chars == 0:
            return 'english'
            
        malayalam_ratio = malayalam_chars / total_chars
        
        # More sensitive Malayalam detection for better language recognition
        if malayalam_ratio > 0.1:  # Lowered threshold for better Malayalam detection
            return 'malayalam'
        elif malayalam_ratio > 0:
            return 'mixed'
        else:
            return 'english'
    
    async def process_multilingual_questions(self, questions: List[str], 
                                           chunk_embeddings: List, 
                                           text_chunks: List[str],
                                           faiss_manager, 
                                           embedding_generator,
                                           llm_client) -> List[str]:
        """Process questions with language-aware retrieval and response"""
        
        multilingual_context = self.detect_multilingual_context(questions)
        
        if not multilingual_context['is_multilingual']:
            # Use standard processing for single-language queries
            return await self._process_standard_questions(
                questions, chunk_embeddings, text_chunks, faiss_manager, embedding_generator, llm_client
            )
        
        print(f"🌐 Detected multilingual context: {multilingual_context['malayalam_count']} Malayalam, {multilingual_context['english_count']} English")
        
        # Process each question with language awareness
        answers = []
        for i, question in enumerate(questions):
            question_lang = multilingual_context['question_languages'][i]
            
            answer = await self._process_single_multilingual_question(
                question, question_lang, faiss_manager, embedding_generator, llm_client
            )
            answers.append(answer)
            
        return answers
    
    async def _process_single_multilingual_question(self, question: str, 
                                                   question_lang: str,
                                                   faiss_manager,
                                                   embedding_generator,
                                                   llm_client) -> str:
        """Process individual question with language-specific handling"""
        
        # Generate embedding for the original question
        question_embedding = embedding_generator.get_embeddings([question])[0]

        # Search for relevant contexts with appropriate threshold
        search_results = faiss_manager.search(question_embedding, k=8, distance_threshold=2.0)

        if not search_results:
            if question_lang == 'malayalam':
                return "ഡോക്യുമെന്റിൽ ഈ ചോദ്യത്തിന് ഉത്തരം ലഭ്യമല്ല."
            else:
                return "The document does not contain this information."

        # Build context from retrieved chunks
        retrieved_contexts = [result["text"] for result in search_results]

        # Create language-appropriate prompt
        prompt = self._create_multilingual_prompt(question, retrieved_contexts, question_lang)
        
        # Generate response using the existing LLM client method
        try:
            # Use the existing generate_answer method if available
            if hasattr(llm_client, 'generate_answer'):
                response = await llm_client.generate_answer(question, retrieved_contexts)
            else:
                # Fallback to a direct prompt method
                response = await llm_client.generate_response(prompt)
                
            # Post-process response based on language  
            return response
            
        except Exception as e:
            print(f"❌ Error generating multilingual response: {e}")
            if question_lang == 'malayalam':
                return f"ഉത്തരം സൃഷ്ടിക്കുന്നതിൽ പിശക്: {str(e)}"
            else:
                return f"Error generating answer: {str(e)}"
    
    def _contains_malayalam(self, text: str) -> bool:
        """Check if text contains Malayalam characters"""
        return bool(self.malayalam_pattern.search(text))
    
    def _create_multilingual_prompt(self, question: str, contexts: List[str], question_lang: str) -> str:
        """Create language-appropriate prompts for better responses"""
        
        context_text = "\n\n".join(contexts)
        
        if question_lang == 'malayalam':
            prompt = f"""നിങ്ങൾ ഒരു സഹായകരമായ AI അസിസ്റ്റന്റാണ്. ഇനിപ്പറയുന്ന സന്ദർഭത്തെ അടിസ്ഥാനമാക്കി ചോദ്യത്തിന് മലയാളത്തിൽ കൃത്യവും വിശദവുമായ ഉത്തരം നൽകുക.

സന്ദർഭം:
{context_text}

ചോദ്യം: {question}

നിർദ്ദേശങ്ങൾ:
- മലയാളത്തിൽ മാത്രം ഉത്തരം നൽകുക
- സന്ദർഭത്തിൽ നിന്നുള്ള വിവരങ്ങൾ മാത്രം ഉപയോഗിക്കുക
- കൃത്യവും വ്യക്തവുമായ ഉത്തരം നൽകുക

ഉത്തരം:"""
        else:
            prompt = f"""You are a helpful AI assistant. Based on the following context, provide a precise and detailed answer to the question.

Context:
{context_text}

Question: {question}

Instructions:
- Answer in English only
- Use only information from the provided context
- Be precise and clear in your response

Answer:"""
        
        return prompt
    
    async def _process_standard_questions(self, questions: List[str], 
                                        chunk_embeddings: List,
                                        text_chunks: List[str],
                                        faiss_manager,
                                        embedding_generator,
                                        llm_client) -> List[str]:
        """Fallback to standard processing for single-language queries"""
        
        # Generate embeddings for all questions
        question_embeddings = embedding_generator.get_embeddings(questions)
        
        answers = []
        for i, question in enumerate(questions):
            search_results = faiss_manager.search(question_embeddings[i], k=8, distance_threshold=1.5)
            
            if search_results:
                retrieved_contexts = [result["text"] for result in search_results]
                answer = await llm_client.generate_answer(question, retrieved_contexts)
            else:
                answer = "No relevant context found for this question in the document."
                
            answers.append(answer)
            
        return answers
