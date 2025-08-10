# src/utils/enhanced_query_processor.py
"""
Enhanced query processing with metadata support for improved queries.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from src.utils.multilingual_processor import MultilingualProcessor
from src.utils.api_interaction_handler import APIInteractionHandler

class EnhancedQueryProcessor:
    """Processes queries with enhanced metadata and context awareness."""
    
    def __init__(self):
        self.multilingual_processor = MultilingualProcessor()
        self.difficulty_weights = {
            'basic': 1.0,
            'intermediate': 1.2,
            'advanced': 1.5
        }
        
    def load_enhanced_queries(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load enhanced queries from JSON file.
        
        Args:
            file_path: Path to enhanced queries JSON file
            
        Returns:
            List of enhanced query documents
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading enhanced queries: {e}")
            return []
    
    def categorize_question(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Categorize a question based on content and context.
        
        Args:
            question: The question text
            context: Additional context from document
            
        Returns:
            Question metadata with category, difficulty, etc.
        """
        question_lower = question.lower()
        
        # Determine category based on keywords
        categories = {
            'coverage': ['cover', 'coverage', 'benefit', 'included', 'exclude'],
            'claim_process': ['claim', 'submit', 'document', 'procedure', 'process'],
            'policy_terms': ['term', 'condition', 'waiting period', 'period'],
            'technical': ['specification', 'engine', 'maintenance', 'repair'],
            'legal': ['law', 'legal', 'right', 'constitution', 'article'],
            'api_interaction': ['flight number', 'api', 'real-time', 'status'],
            'multilingual': ['malayalam', 'hindi', 'tamil']
        }
        
        detected_category = 'general'
        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_category = category
                break
        
        # Determine difficulty
        complexity_indicators = {
            'basic': ['what is', 'how much', 'is it', 'yes or no'],
            'intermediate': ['explain', 'how does', 'what are', 'describe'],
            'advanced': ['analyze', 'compare', 'evaluate', 'multiple', 'complex']
        }
        
        detected_difficulty = 'basic'
        for difficulty, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                detected_difficulty = difficulty
        
        # Detect language
        detected_language = self.multilingual_processor.detect_language(question)
        
        return {
            'category': detected_category,
            'difficulty': detected_difficulty,
            'language': detected_language,
            'requires_api': 'api' in question_lower or 'real-time' in question_lower,
            'is_multilingual': detected_language != 'en'
        }
    
    def enhance_retrieval_for_question(self, question: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance retrieval parameters based on question metadata.
        
        Args:
            question: Original question
            metadata: Question metadata
            
        Returns:
            Enhanced retrieval parameters
        """
        retrieval_params = {
            'top_k': 8,  # Default
            'boost_factor': 1.0,
            'search_keywords': [],
            'filter_criteria': []
        }
        
        # Adjust top_k based on difficulty
        difficulty_multipliers = {
            'basic': 0.75,      # 6 chunks
            'intermediate': 1.0, # 8 chunks  
            'advanced': 1.5     # 12 chunks
        }
        
        retrieval_params['top_k'] = int(
            retrieval_params['top_k'] * difficulty_multipliers.get(metadata['difficulty'], 1.0)
        )
        
        # Add category-specific keywords
        category_keywords = {
            'coverage': ['benefit', 'include', 'exclude', 'cover'],
            'claim_process': ['claim', 'submit', 'document', 'procedure'],
            'policy_terms': ['term', 'condition', 'waiting', 'period'],
            'technical': ['specification', 'maintenance', 'repair'],
            'legal': ['article', 'section', 'law', 'constitution']
        }
        
        if metadata['category'] in category_keywords:
            retrieval_params['search_keywords'] = category_keywords[metadata['category']]
        
        # Boost factor for complex queries
        if metadata['difficulty'] == 'advanced':
            retrieval_params['boost_factor'] = 1.3
        
        return retrieval_params
    
    def process_enhanced_query_batch(self, enhanced_queries: List[Dict[str, Any]], 
                                   target_document: str) -> List[str]:
        """
        Process a batch of enhanced queries for a specific document.
        
        Args:
            enhanced_queries: List of enhanced query documents
            target_document: Document ID to filter queries for
            
        Returns:
            List of questions for the target document
        """
        questions = []
        
        for doc_queries in enhanced_queries:
            if doc_queries.get('document_id') == target_document:
                for question_data in doc_queries.get('questions', []):
                    questions.append(question_data['question'])
                break
        
        return questions
    
    def get_document_metadata(self, enhanced_queries: List[Dict[str, Any]], 
                            document_url: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata from enhanced queries.
        
        Args:
            enhanced_queries: List of enhanced query documents
            document_url: Document URL to match
            
        Returns:
            Document metadata if found
        """
        for doc_queries in enhanced_queries:
            if doc_queries.get('documents') == document_url:
                return {
                    'document_id': doc_queries.get('document_id'),
                    'document_title': doc_queries.get('document_title'),
                    'document_type': doc_queries.get('document_type'),
                    'language': doc_queries.get('language'),
                    'format': doc_queries.get('format'),
                    'special_requirements': doc_queries.get('special_requirements', {}),
                    'question_count': len(doc_queries.get('questions', []))
                }
        
        return None
    
    def create_context_enhanced_prompt(self, question: str, contexts: List[str], 
                                     question_metadata: Dict[str, Any],
                                     document_metadata: Dict[str, Any]) -> str:
        """
        Create an enhanced prompt with metadata context.
        
        Args:
            question: Original question
            contexts: Retrieved contexts
            question_metadata: Question categorization metadata
            document_metadata: Document metadata
            
        Returns:
            Enhanced prompt for LLM
        """
        prompt_parts = []
        
        # Add document context
        prompt_parts.append(f"Document: {document_metadata.get('document_title', 'Unknown')}")
        prompt_parts.append(f"Type: {document_metadata.get('document_type', 'unknown')}")
        
        # Add question context
        if question_metadata['is_multilingual']:
            prompt_parts.append(f"Language: {question_metadata['language']}")
            prompt_parts.append("Note: This question is in a non-English language. Provide response in the same language when appropriate.")
        
        prompt_parts.append(f"Question Category: {question_metadata['category']}")
        prompt_parts.append(f"Difficulty Level: {question_metadata['difficulty']}")
        
        # Add special instructions
        if question_metadata['requires_api']:
            prompt_parts.append("Note: This question may require real-time data. Indicate where API calls would be needed.")
        
        # Add contexts
        prompt_parts.append("\nRelevant Document Sections:")
        for i, context in enumerate(contexts, 1):
            prompt_parts.append(f"{i}. {context}")
        
        # Add question
        prompt_parts.append(f"\nQuestion: {question}")
        
        # Add response guidance
        if question_metadata['difficulty'] == 'advanced':
            prompt_parts.append("\nProvide a comprehensive, detailed response with examples where applicable.")
        elif question_metadata['difficulty'] == 'basic':
            prompt_parts.append("\nProvide a clear, concise response.")
        
        return "\n".join(prompt_parts)
