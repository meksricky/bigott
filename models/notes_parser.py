# models/notes_parser.py
import re
from odoo import models, fields, api
from odoo.exceptions import UserError

class NotesParser(models.Model):
    _name = 'notes.parser'
    _description = 'Client Notes Parser with Strict Rules'
    
    def parse_client_notes(self, notes):
        """Parse notes and extract MANDATORY requirements"""
        
        if not notes:
            return {'use_default': True}
        
        parsed = {
            'use_default': False,
            'mandatory_count': None,
            'specific_products': [],
            'categories_required': {},
            'categories_excluded': [],
            'budget_flexibility': None,
            'preferences': {}
        }
        
        notes_lower = notes.lower()
        
        # 1. MANDATORY PRODUCT COUNT
        count_patterns = [
            r'want\s+(\d+)\s+products?',
            r'exactly\s+(\d+)\s+items?',
            r'only\s+(\d+)\s+products?',
            r'give\s+me\s+(\d+)',
            r'(\d+)\s+products?\s+only',
            r'need\s+(\d+)\s+items?'
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, notes_lower)
            if match:
                parsed['mandatory_count'] = int(match.group(1))
                break
        
        # 2. SPECIFIC CATEGORY REQUIREMENTS
        category_patterns = {
            'wines': r'(\d+)\s+wines?|wines?\s*:\s*(\d+)',
            'champagne': r'(\d+)\s+champagnes?|champagnes?\s*:\s*(\d+)',
            'sweets': r'(\d+)\s+sweets?|sweets?\s*:\s*(\d+)',
            'cheese': r'(\d+)\s+cheese?|cheese?\s*:\s*(\d+)',
            'ibericos': r'(\d+)\s+ibericos?|ibericos?\s*:\s*(\d+)',
            'experiences': r'(\d+)\s+experiences?|experiences?\s*:\s*(\d+)'
        }
        
        for category, pattern in category_patterns.items():
            match = re.search(pattern, notes_lower)
            if match:
                count = int(match.group(1) or match.group(2))
                parsed['categories_required'][category] = count
        
        # 3. SPECIFIC PRODUCT REQUESTS
        product_patterns = [
            r'include\s+([^,\.]+)',
            r'must\s+have\s+([^,\.]+)',
            r'add\s+([^,\.]+)',
            r'want\s+([^,\.]+)'
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, notes_lower)
            for match in matches:
                # Clean and add specific product requests
                product = match.strip()
                if len(product) > 3 and 'product' not in product:
                    parsed['specific_products'].append(product)
        
        # 4. EXCLUSIONS
        exclude_patterns = [
            r'no\s+([^,\.]+)',
            r'exclude\s+([^,\.]+)',
            r'without\s+([^,\.]+)',
            r'avoid\s+([^,\.]+)'
        ]
        
        for pattern in exclude_patterns:
            matches = re.findall(pattern, notes_lower)
            for match in matches:
                excluded = match.strip()
                parsed['categories_excluded'].append(excluded)
        
        # 5. BUDGET FLEXIBILITY
        if 'flexible budget' in notes_lower or 'budget is flexible' in notes_lower:
            parsed['budget_flexibility'] = 0.20  # Allow 20% variance
        elif 'strict budget' in notes_lower:
            parsed['budget_flexibility'] = 0.05  # Only 5% variance
        else:
            parsed['budget_flexibility'] = 0.10  # Default 10%
        
        # 6. PREFERENCES
        parsed['preferences'] = {
            'loves_sweets': 'loves sweet' in notes_lower or 'sweet tooth' in notes_lower,
            'premium': 'premium' in notes_lower or 'luxury' in notes_lower or 'best' in notes_lower,
            'traditional': 'traditional' in notes_lower or 'classic' in notes_lower,
            'modern': 'modern' in notes_lower or 'innovative' in notes_lower,
            'healthy': 'healthy' in notes_lower or 'organic' in notes_lower,
            'local': 'local' in notes_lower or 'spanish' in notes_lower,
            'international': 'international' in notes_lower or 'imported' in notes_lower
        }
        
        return parsed