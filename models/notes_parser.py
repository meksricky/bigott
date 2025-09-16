# models/notes_parser.py
# Enhanced version with advanced parsing capabilities

import re
import logging
from odoo import models, fields, api
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)


class NotesParser(models.Model):
    _name = 'notes.parser'
    _description = 'Client Notes Parser with Strict Rules'
    
    def parse_client_notes(self, notes):
        """Parse notes and extract MANDATORY requirements with advanced capabilities"""
        
        if not notes:
            return {'use_default': True}
        
        parsed = {
            'use_default': False,
            'mandatory_count': None,
            'specific_products': [],
            'categories_required': {},
            'categories_excluded': [],
            'budget_flexibility': None,
            'preferences': {},
            # New advanced features
            'budget_override': None,
            'override_form': {},
            'composition_type': None,
            'dietary': [],
            'special_instructions': []
        }
        
        notes_lower = notes.lower()
        
        # 1. ADVANCED BUDGET PARSING WITH OVERRIDE DETECTION
        budget_info = self._parse_budget_advanced(notes, notes_lower)
        if budget_info['found']:
            parsed['budget_override'] = budget_info['amount']
            parsed['budget_flexibility'] = budget_info['flexibility']
            parsed['override_form']['budget'] = True
            _logger.info(f"Budget override detected: €{budget_info['amount']} ±{budget_info['flexibility']}%")
        else:
            # Use existing budget flexibility parsing
            if 'flexible budget' in notes_lower or 'budget is flexible' in notes_lower:
                parsed['budget_flexibility'] = 0.20  # Allow 20% variance
            elif 'strict budget' in notes_lower:
                parsed['budget_flexibility'] = 0.05  # Only 5% variance
            else:
                parsed['budget_flexibility'] = 0.10  # Default 10%
        
        # 2. MANDATORY PRODUCT COUNT (Enhanced patterns)
        count_patterns = [
            r'want\s+(\d+)\s+products?',
            r'exactly\s+(\d+)\s+items?',
            r'only\s+(\d+)\s+products?',
            r'give\s+me\s+(\d+)',
            r'(\d+)\s+products?\s+only',
            r'need\s+(\d+)\s+items?',
            # New patterns
            r'(\d+)\s+productos?',  # Spanish
            r'must\s+have\s+(\d+)\s+products?',
            r'include\s+(\d+)\s+items?',
            r'total\s+of\s+(\d+)',
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, notes_lower)
            if match:
                parsed['mandatory_count'] = int(match.group(1))
                parsed['override_form']['product_count'] = True
                break
        
        # 3. ENHANCED CATEGORY REQUIREMENTS (with Spanish support)
        category_patterns = {
            'wines': r'(\d+)\s+(?:wines?|vinos?)|(?:wines?|vinos?)\s*:\s*(\d+)',
            'champagne': r'(\d+)\s+(?:champagnes?|cavas?)|(?:champagnes?|cavas?)\s*:\s*(\d+)',
            'sweets': r'(\d+)\s+(?:sweets?|dulces?|chocolates?)|(?:sweets?|dulces?)\s*:\s*(\d+)',
            'cheese': r'(\d+)\s+(?:cheese|cheeses|queso|quesos)|(?:cheese|quesos?)\s*:\s*(\d+)',
            'ibericos': r'(\d+)\s+(?:ibericos?|jam[oó]n|hams?)|(?:ibericos?|jamón)\s*:\s*(\d+)',
            'experiences': r'(\d+)\s+experiences?|experiences?\s*:\s*(\d+)',
            'foie': r'(\d+)\s+foie|foie\s*:\s*(\d+)',
        }
        
        for category, pattern in category_patterns.items():
            match = re.search(pattern, notes_lower)
            if match:
                count = int(match.group(1) or match.group(2))
                parsed['categories_required'][category] = count
        
        # 4. DIETARY RESTRICTIONS (Comprehensive)
        parsed['dietary'] = self._parse_dietary_comprehensive(notes_lower)
        if parsed['dietary']:
            parsed['override_form']['dietary'] = True
        
        # 5. COMPOSITION TYPE
        if 'hybrid' in notes_lower:
            parsed['composition_type'] = 'hybrid'
            parsed['override_form']['composition_type'] = True
        elif 'experience' in notes_lower and ('based' in notes_lower or 'package' in notes_lower):
            parsed['composition_type'] = 'experience'
            parsed['override_form']['composition_type'] = True
        elif 'custom' in notes_lower:
            parsed['composition_type'] = 'custom'
            parsed['override_form']['composition_type'] = True
        
        # 6. SPECIFIC PRODUCT REQUESTS (Enhanced)
        product_patterns = [
            r'include\s+([^,\.]+)',
            r'must\s+have\s+([^,\.]+)',
            r'add\s+([^,\.]+)',
            r'want\s+([^,\.]+)',
            # Product codes
            r'([A-Z]+-[A-Z0-9-]+)',
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, notes_lower)
            for match in matches:
                product = match.strip()
                if len(product) > 3 and 'product' not in product and product not in parsed['specific_products']:
                    parsed['specific_products'].append(product)
        
        # Check for specific luxury products
        luxury_products = {
            'bellota': 'BELLOTA_HAM',
            'gran reserva': 'GRAN_RESERVA_WINE',
            'dom perignon': 'DOM_PERIGNON',
            'caviar': 'CAVIAR',
            'truffle': 'TRUFFLE_PRODUCT',
            'foie gras': 'FOIE_GRAS',
        }
        
        for keyword, product_code in luxury_products.items():
            if keyword in notes_lower and product_code not in parsed['specific_products']:
                parsed['specific_products'].append(product_code)
        
        # 7. EXCLUSIONS (Enhanced)
        exclude_patterns = [
            r'no\s+([^,\.]+)',
            r'exclude\s+([^,\.]+)',
            r'without\s+([^,\.]+)',
            r'avoid\s+([^,\.]+)',
            r'sin\s+([^,\.]+)',  # Spanish
        ]
        
        for pattern in exclude_patterns:
            matches = re.findall(pattern, notes_lower)
            for match in matches:
                excluded = match.strip()
                if len(excluded) > 2 and excluded not in parsed['categories_excluded']:
                    parsed['categories_excluded'].append(excluded)
        
        # 8. SPECIAL INSTRUCTIONS
        parsed['special_instructions'] = self._extract_special_instructions(notes)
        
        # 9. PREFERENCES (Enhanced)
        parsed['preferences'] = {
            'loves_sweets': 'loves sweet' in notes_lower or 'sweet tooth' in notes_lower or 'dulces' in notes_lower,
            'premium': 'premium' in notes_lower or 'luxury' in notes_lower or 'best' in notes_lower or 'finest' in notes_lower,
            'traditional': 'traditional' in notes_lower or 'classic' in notes_lower or 'authentic' in notes_lower,
            'modern': 'modern' in notes_lower or 'innovative' in notes_lower or 'contemporary' in notes_lower,
            'healthy': 'healthy' in notes_lower or 'organic' in notes_lower or 'natural' in notes_lower,
            'local': 'local' in notes_lower or 'spanish' in notes_lower or 'regional' in notes_lower,
            'international': 'international' in notes_lower or 'imported' in notes_lower or 'foreign' in notes_lower,
            'variety': 'variety' in notes_lower or 'diverse' in notes_lower or 'different' in notes_lower,
        }
        
        return parsed
    
    def _parse_budget_advanced(self, notes, notes_lower):
        """Advanced budget parsing with override detection and flexibility"""
        
        result = {'found': False, 'amount': None, 'flexibility': 10}
        
        # Advanced budget patterns
        budget_patterns = [
            # Standard patterns
            (r'budget\s+(?:of\s+)?(?:around\s+)?(?:€|\$)?([0-9,]+)', 'normal'),
            (r'(?:€|\$)([0-9,]+)\s+budget', 'normal'),
            (r'([0-9,]+)\s+(?:euro|euros|dollar|dollars|€|\$)\s+budget', 'normal'),
            
            # With flexibility indicators
            (r'(?:around|approximately|roughly|about)\s+(?:€|\$)?([0-9,]+)', 'flexible'),
            (r'budget\s+(?:of\s+)?(?:€|\$)?([0-9,]+)\s*(?:\+/?-|plus.?minus)\s*(\d+)%?', 'percentage'),
            
            # Range patterns (take midpoint)
            (r'budget\s+(?:between\s+)?(?:€|\$)?([0-9,]+)\s*(?:to|-)\s*(?:€|\$)?([0-9,]+)', 'range'),
            (r'(?:€|\$)?([0-9,]+)\s*(?:to|-)\s*(?:€|\$)?([0-9,]+)\s+budget', 'range'),
            
            # Explicit override patterns
            (r'use\s+(?:a\s+)?budget\s+(?:of\s+)?(?:€|\$)?([0-9,]+)', 'override'),
            (r'change\s+budget\s+to\s+(?:€|\$)?([0-9,]+)', 'override'),
            (r'set\s+budget\s+(?:at\s+)?(?:€|\$)?([0-9,]+)', 'override'),
            
            # Spanish patterns
            (r'presupuesto\s+(?:de\s+)?(?:€|\$)?([0-9,]+)', 'normal'),
        ]
        
        for pattern, pattern_type in budget_patterns:
            match = re.search(pattern, notes_lower)
            if match:
                if pattern_type == 'range':
                    # Take the midpoint of the range
                    low = float(match.group(1).replace(',', ''))
                    high = float(match.group(2).replace(',', ''))
                    result['amount'] = (low + high) / 2
                    result['flexibility'] = ((high - low) / 2 / result['amount']) * 100
                elif pattern_type == 'percentage':
                    result['amount'] = float(match.group(1).replace(',', ''))
                    result['flexibility'] = float(match.group(2)) if len(match.groups()) > 1 else 10
                else:
                    result['amount'] = float(match.group(1).replace(',', ''))
                    
                    # Set flexibility based on pattern type
                    if pattern_type == 'flexible':
                        result['flexibility'] = 15
                    elif pattern_type == 'override':
                        result['flexibility'] = 5  # Strict when explicitly overriding
                    else:
                        result['flexibility'] = 10  # Default
                
                result['found'] = True
                
                # Check for explicit flexibility mentions
                if 'flexible' in notes_lower or 'flexibility' in notes_lower:
                    result['flexibility'] = max(result['flexibility'], 20)
                elif 'strict' in notes_lower or 'exact' in notes_lower or 'exactly' in notes_lower:
                    result['flexibility'] = min(result['flexibility'], 5)
                
                break
        
        return result
    
    def _parse_dietary_comprehensive(self, notes_lower):
        """Parse dietary restrictions comprehensively including multiple languages"""
        
        dietary = []
        
        # Comprehensive dietary keywords with variations
        dietary_keywords = {
            'halal': ['halal', 'muslim', 'islamic', 'no pork', 'no alcohol', 'sin cerdo'],
            'vegan': ['vegan', 'plant-based', 'no animal', 'vegano', 'vegana'],
            'vegetarian': ['vegetarian', 'no meat', 'vegetariano', 'vegetariana', 'sin carne'],
            'gluten_free': ['gluten free', 'gluten-free', 'celiac', 'coeliac', 'sin gluten', 'celiaco'],
            'non_alcoholic': ['no alcohol', 'non-alcoholic', 'alcohol free', 'sin alcohol', 'alcohol-free'],
            'kosher': ['kosher', 'jewish dietary', 'koscher'],
            'diabetic': ['diabetic', 'no sugar', 'sugar free', 'diabetes', 'sin azúcar'],
            'lactose_free': ['lactose free', 'lactose-free', 'no dairy', 'sin lactosa'],
            'nut_free': ['nut free', 'nut-free', 'no nuts', 'nut allergy', 'sin frutos secos'],
        }
        
        for restriction, keywords in dietary_keywords.items():
            for keyword in keywords:
                if keyword in notes_lower:
                    if restriction not in dietary:
                        dietary.append(restriction)
                    break
        
        return dietary
    
    def _extract_special_instructions(self, notes):
        """Extract special instructions and important preferences"""
        
        instructions = []
        
        # Keywords that indicate special instructions
        instruction_keywords = [
            'prefer', 'like', 'favorite', 'love',
            'important', 'priority', 'focus', 'emphasis',
            'make sure', 'ensure', 'guarantee', 'must',
            'please', 'would like', 'request',
            'avoid', 'do not', "don't", 
            'special', 'specifically', 'especially'
        ]
        
        # Split notes into sentences
        sentences = re.split(r'[.!?]+', notes)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(keyword in sentence.lower() for keyword in instruction_keywords):
                if sentence not in instructions:
                    instructions.append(sentence)
        
        return instructions