# src/utils/enhanced_query_processor.py
import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class EnhancedQueryProcessor:
    """
    Enhanced query processor for categorization and difficulty assessment
    Optimized for insurance, medical, and general knowledge questions
    """
    
    def __init__(self):
        """Initialize enhanced query processor with category patterns"""
        self.category_patterns = {
            'insurance_coverage': [
                'covered', 'coverage', 'expenses', 'benefits', 'limits', 'hospitalization',
                'claim', 'policy', 'premium', 'reimbursement', 'cashless', 'deductible',
                'copay', 'maximum', 'annual', 'sum insured', 'room rent', 'ICU'
            ],
            'insurance_procedures': [
                'claim process', 'documents', 'submission', 'settlement', 'pre-authorization',
                'waiting period', 'exclusions', 'nomination', 'how to claim', 'procedure',
                'steps', 'requirements', 'forms', 'application'
            ],
            'medical_conditions': [
                'disease', 'illness', 'surgery', 'treatment', 'diagnosis', 'psychiatric',
                'maternity', 'dental', 'cancer', 'heart', 'accident', 'emergency',
                'chronic', 'pre-existing', 'condition', 'therapy', 'medicine'
            ],
            'telemedicine': [
                'telemedicine', 'teleconsultation', 'online consultation', 'digital health',
                'remote consultation', 'virtual doctor'
            ],
            'ambulance': [
                'ambulance', 'emergency transport', 'medical transport', 'emergency vehicle'
            ],
            'domiciliary': [
                'domiciliary', 'home treatment', 'at home', 'home care', 'in-house treatment'
            ],
            'waiting_period': [
                'waiting period', 'waiting time', 'pre-existing', 'specified diseases',
                'initial waiting', 'cooling period'
            ],
            'general_knowledge': [
                'capital', 'dinosaurs', 'clouds', 'plants', 'lungs', 'galaxy', 'human body',
                'geography', 'science', 'history', 'general'
            ],
            'puzzle_interactive': [
                'flight number', 'puzzle', 'mission', 'step', 'decode', 'landmark', 'mapping',
                'api endpoint', 'call this endpoint', 'interactive', 'challenge', 'solution'
            ]
        }
        
        logger.info("✅ Enhanced query processor initialized")
    
    def categorize_question(self, question: str, language: str = "english") -> str:
        """
        Categorize question based on content patterns
        Returns the most relevant category
        """
        try:
            question_lower = question.lower()
            
            # Find matching categories with scores
            category_scores = {}
            for category, patterns in self.category_patterns.items():
                score = 0
                for pattern in patterns:
                    if pattern.lower() in question_lower:
                        # Give higher score for exact matches
                        if pattern.lower() == question_lower.strip():
                            score += 5
                        else:
                            score += 1
                
                if score > 0:
                    category_scores[category] = score
            
            # Return category with highest score
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])[0]
                return best_category
            else:
                # Fallback categorization for Malayalam or unmatched English
                if language == "malayalam":
                    # Basic Malayalam keyword detection
                    malayalam_insurance_keywords = [
                        'ഇൻഷുറൻസ്', 'ക്ലെയിം', 'പോളിസി', 'കവറേജ്', 'ചികിത്സ'
                    ]
                    for keyword in malayalam_insurance_keywords:
                        if keyword in question:
                            return 'insurance_coverage'
                
                return 'general'
                
        except Exception as e:
            logger.error(f"❌ Error categorizing question: {e}")
            return 'general'
    
    def assess_difficulty(self, question: str, category: str = "") -> str:
        """
        Assess question difficulty based on complexity indicators
        Returns: 'basic', 'intermediate', or 'advanced'
        """
        try:
            question_lower = question.lower()
            
            # Advanced complexity indicators
            advanced_indicators = [
                'explain in detail', 'comprehensive', 'analyze', 'compare and contrast',
                'evaluate', 'justify', 'elaborate', 'describe the process',
                'what are the implications', 'how does it affect', 'pros and cons'
            ]
            
            # Intermediate complexity indicators
            intermediate_indicators = [
                'explain', 'describe', 'how', 'why', 'process', 'procedure',
                'requirements', 'steps', 'difference between', 'types of'
            ]
            
            # Basic complexity indicators
            basic_indicators = [
                'what is', 'who is', 'where', 'when', 'which', 'list',
                'name', 'is there', 'does', 'can', 'will', 'yes or no'
            ]
            
            # Count indicators
            advanced_count = sum(1 for indicator in advanced_indicators if indicator in question_lower)
            intermediate_count = sum(1 for indicator in intermediate_indicators if indicator in question_lower)
            basic_count = sum(1 for indicator in basic_indicators if indicator in question_lower)
            
            # Question length factor
            word_count = len(question.split())
            
            # Determine difficulty
            if advanced_count > 0 or word_count > 25:
                return 'advanced'
            elif intermediate_count > 0 and basic_count == 0:
                return 'intermediate'
            elif basic_count > 0 and intermediate_count == 0:
                return 'basic'
            elif word_count < 8:
                return 'basic'
            else:
                return 'intermediate'
                
        except Exception as e:
            logger.error(f"❌ Error assessing difficulty: {e}")
            return 'intermediate'
    
    def preprocess_question(self, question: str, language: str = "english") -> str:
        """
        Preprocess question for better retrieval
        Cleans and normalizes the question text
        """
        try:
            # Basic cleaning
            question = question.strip()
            
            # Language-specific preprocessing
            if language == "malayalam":
                # Keep Malayalam characters intact
                question = re.sub(r'[^\w\s\u0d00-\u0d7f?.,!]', ' ', question)
            else:
                # Standard preprocessing for English
                question = re.sub(r'[^\w\s?.,!]', ' ', question)
            
            # Remove extra whitespace
            question = re.sub(r'\s+', ' ', question)
            
            # Ensure question ends with proper punctuation
            question = question.strip()
            if question and not question[-1] in '?.!':
                question += '?'
            
            return question
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing question: {e}")
            return question
    
    def get_category_description(self, category: str) -> str:
        """Get human-readable description of category"""
        descriptions = {
            'insurance_coverage': 'Insurance Coverage & Benefits',
            'insurance_procedures': 'Insurance Claims & Procedures',
            'medical_conditions': 'Medical Conditions & Treatments',
            'telemedicine': 'Telemedicine Services',
            'ambulance': 'Ambulance Services',
            'domiciliary': 'Domiciliary Treatment',
            'waiting_period': 'Waiting Periods',
            'general_knowledge': 'General Knowledge',
            'flight_puzzle': 'Flight Information',
            'general': 'General Query'
        }
        return descriptions.get(category, 'Unknown Category')
    
    def get_processing_hints(self, category: str, difficulty: str) -> Dict[str, str]:
        """Get processing hints based on category and difficulty"""
        return {
            'category': category,
            'difficulty': difficulty,
            'description': self.get_category_description(category),
            'retrieval_focus': self._get_retrieval_focus(category),
            'answer_style': self._get_answer_style(difficulty)
        }
    
    def _get_retrieval_focus(self, category: str) -> str:
        """Get retrieval focus for different categories"""
        focus_map = {
            'insurance_coverage': 'Policy terms, coverage limits, and benefit details',
            'insurance_procedures': 'Claims process, documentation, and requirements',
            'medical_conditions': 'Medical terms, treatments, and coverage specifics',
            'telemedicine': 'Digital health services and consultation processes',
            'ambulance': 'Emergency transport services and coverage',
            'domiciliary': 'Home treatment policies and conditions',
            'waiting_period': 'Time-based restrictions and conditions',
            'general': 'Comprehensive document search'
        }
        return focus_map.get(category, 'General document content')
    
    def _get_answer_style(self, difficulty: str) -> str:
        """Get answer style for different difficulty levels"""
        style_map = {
            'basic': 'Direct, concise answers with key facts',
            'intermediate': 'Detailed explanations with context',
            'advanced': 'Comprehensive analysis with examples and implications'
        }
        return style_map.get(difficulty, 'Balanced explanation with context')
