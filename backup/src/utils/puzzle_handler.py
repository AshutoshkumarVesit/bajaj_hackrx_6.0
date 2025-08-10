# src/utils/puzzle_handler.py
import re
import logging
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class PuzzleHandler:
    """
    Handles puzzle/interactive documents that require procedural steps
    and API calls to solve problems.
    """
    
    def __init__(self):
        """Initialize puzzle handler"""
        self.puzzle_patterns = {
            'api_endpoints': re.compile(r'GET\s+(https?://[^\s]+)', re.IGNORECASE),
            'step_by_step': re.compile(r'Step\s+\d+[:\-]', re.IGNORECASE),
            'mission_objective': re.compile(r'mission\s+objective|your\s+mission', re.IGNORECASE),
            'flight_number': re.compile(r'flight\s+number', re.IGNORECASE),
            'landmark_mapping': re.compile(r'(\w+(?:\s+\w+)*)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.MULTILINE),
            'conditional_instructions': re.compile(r'If\s+.*?call[:\s]*\n?\s*GET\s+(https?://[^\s]+)', re.IGNORECASE | re.DOTALL)
        }
        
        logger.info("✅ Puzzle handler initialized")
    
    def is_puzzle_document(self, text: str) -> bool:
        """
        Detect if the document is a puzzle/interactive document
        Requires multiple specific indicators to avoid false positives
        """
        # Specific puzzle indicators that need to co-occur
        primary_indicators = [
            'step-by-step guide',
            'mission objective',
            'api endpoint',
            'call this endpoint',
            'follow these instructions'
        ]
        
        # Secondary indicators that support puzzle detection
        secondary_indicators = [
            'decode',
            'puzzle',
            'interactive',
            'solve',
            'GET https://',
            'endpoint'
        ]
        
        text_lower = text.lower()
        
        # Count primary indicators (more specific)
        primary_count = sum(1 for indicator in primary_indicators if indicator in text_lower)
        
        # Count secondary indicators (broader)
        secondary_count = sum(1 for indicator in secondary_indicators if indicator in text_lower)
        
        # Require at least 2 primary indicators OR 1 primary + 3 secondary
        is_puzzle = (primary_count >= 2) or (primary_count >= 1 and secondary_count >= 3)
        
        if is_puzzle:
            logger.info("🧩 Document detected as puzzle/interactive type")
        
        return is_puzzle
    
    def extract_api_endpoints(self, text: str) -> List[str]:
        """
        Extract all API endpoints mentioned in the document
        """
        endpoints = self.puzzle_patterns['api_endpoints'].findall(text)
        return list(set(endpoints))  # Remove duplicates
    
    def extract_step_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract the step-by-step structure from the document
        """
        steps = []
        lines = text.split('\n')
        
        current_step = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a step header
            step_match = self.puzzle_patterns['step_by_step'].search(line)
            if step_match:
                # Save previous step if exists
                if current_step:
                    steps.append({
                        'step': current_step,
                        'content': '\n'.join(current_content),
                        'apis': self.puzzle_patterns['api_endpoints'].findall('\n'.join(current_content))
                    })
                
                # Start new step
                current_step = line
                current_content = []
            else:
                if current_step:
                    current_content.append(line)
        
        # Add the last step
        if current_step:
            steps.append({
                'step': current_step,
                'content': '\n'.join(current_content),
                'apis': self.puzzle_patterns['api_endpoints'].findall('\n'.join(current_content))
            })
        
        return steps
    
    def extract_landmark_mappings(self, text: str) -> Dict[str, str]:
        """
        Extract landmark to location mappings from tables
        """
        mappings = {}
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or 'Landmark' in line or 'Current Location' in line:
                continue
            
            # Try to match landmark and location patterns
            parts = line.split()
            if len(parts) >= 2:
                # Handle multi-word landmarks and locations
                potential_landmark = []
                potential_location = []
                
                # Simple heuristic: assume the last word is the location
                if len(parts) >= 2:
                    potential_location = [parts[-1]]
                    potential_landmark = parts[:-1]
                    
                    landmark = ' '.join(potential_landmark)
                    location = ' '.join(potential_location)
                    
                    if landmark and location:
                        mappings[location] = landmark
        
        return mappings
    
    def extract_conditional_logic(self, text: str) -> List[Dict[str, str]]:
        """
        Extract conditional instructions (if X then call Y)
        """
        conditionals = []
        
        # Look for patterns like "If landmark is X, call Y"
        patterns = [
            r'If\s+landmark.*?"([^"]+)".*?call[:\s]*\n?\s*GET\s+(https?://[^\s]+)',
            r'If.*?favourite\s+city.*?"([^"]+)".*?call[:\s]*\n?\s*GET\s+(https?://[^\s]+)',
            r'For\s+all\s+other\s+landmarks.*?call[:\s]*\n?\s*GET\s+(https?://[^\s]+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.groups()) == 2:
                    conditionals.append({
                        'condition': match.group(1),
                        'action': match.group(2),
                        'type': 'specific'
                    })
                else:
                    conditionals.append({
                        'condition': 'default',
                        'action': match.group(1),
                        'type': 'fallback'
                    })
        
        return conditionals
    
    def generate_puzzle_response(self, question: str, document_content: str) -> str:
        """
        Generate a structured response for puzzle documents
        """
        if not self.is_puzzle_document(document_content):
            return None
        
        # Extract puzzle components
        steps = self.extract_step_structure(document_content)
        endpoints = self.extract_api_endpoints(document_content)
        mappings = self.extract_landmark_mappings(document_content)
        conditionals = self.extract_conditional_logic(document_content)
        
        # Build structured response
        response = []
        
        if 'flight number' in question.lower():
            response.append("To find your flight number, you need to follow this puzzle solution:")
            response.append("")
            
            # Add steps
            if steps:
                for i, step in enumerate(steps, 1):
                    response.append(f"**{step['step']}**")
                    response.append(step['content'])
                    if step['apis']:
                        response.append(f"API to call: `{step['apis'][0]}`")
                    response.append("")
            
            # Add landmark mappings if found
            if mappings:
                response.append("**Landmark Mappings (City → Landmark):**")
                for city, landmark in sorted(mappings.items()):
                    response.append(f"- {city}: {landmark}")
                response.append("")
            
            # Add conditional logic
            if conditionals:
                response.append("**Flight Path Selection:**")
                for cond in conditionals:
                    if cond['type'] == 'specific':
                        response.append(f"- If landmark is '{cond['condition']}': `{cond['action']}`")
                    else:
                        response.append(f"- For all other landmarks: `{cond['action']}`")
                response.append("")
            
            response.append("**Important Note:** This is an interactive puzzle that requires:")
            response.append("1. Making API calls to get your city")
            response.append("2. Looking up the landmark for that city")
            response.append("3. Calling the appropriate flight endpoint")
            response.append("4. The actual flight number can only be obtained by executing these steps")
        
        return '\n'.join(response) if response else None
    
    def solve_flight_puzzle(self, document_content: str) -> Optional[str]:
        """
        Attempt to solve the flight puzzle automatically (if possible)
        Note: This would require actual API access
        """
        try:
            # Extract the first API endpoint
            endpoints = self.extract_api_endpoints(document_content)
            if not endpoints:
                return None
            
            city_endpoint = None
            for endpoint in endpoints:
                if 'myFavouriteCity' in endpoint:
                    city_endpoint = endpoint
                    break
            
            if not city_endpoint:
                return None
            
            # In a real implementation, you would make the API call
            # For now, return instructions
            return f"To solve this puzzle, call: {city_endpoint}"
            
        except Exception as e:
            logger.error(f"Error solving flight puzzle: {e}")
            return None
