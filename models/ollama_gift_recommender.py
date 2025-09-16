# models/ollama_gift_recommender.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import json
import logging
import requests
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import random

_logger = logging.getLogger(__name__)

class OllamaGiftRecommender(models.Model):
    _name = 'ollama.gift.recommender'
    _description = 'Ollama-Powered Gift Recommendation Engine'
    _rec_name = 'name'
    
    # Basic Configuration
    name = fields.Char(string="Recommender Name", default="Ollama Gift Recommender", required=True)
    active = fields.Boolean(string="Active", default=True)
    
    # Ollama Configuration
    ollama_enabled = fields.Boolean(string="Ollama Enabled", default=True)
    ollama_base_url = fields.Char(string="Ollama Base URL", default="http://localhost:11434")
    ollama_model = fields.Char(string="Ollama Model", default="llama3.2:3b")
    ollama_timeout = fields.Integer(string="Timeout (seconds)", default=30)
    
    # Recommendation Settings
    max_products = fields.Integer(string="Max Products per Recommendation", default=15)
    budget_flexibility = fields.Float(string="Budget Flexibility (%)", default=5.0)
    min_confidence_score = fields.Float(string="Minimum Confidence Score", default=0.6)
    
    # Performance Tracking
    total_recommendations = fields.Integer(string="Total Recommendations Made", default=0)
    successful_recommendations = fields.Integer(string="Successful Recommendations", default=0)
    avg_response_time = fields.Float(string="Average Response Time (seconds)", default=0.0)
    last_recommendation_date = fields.Datetime(string="Last Recommendation Date")
    
    # Computed Fields
    success_rate = fields.Float(string="Success Rate (%)", compute='_compute_success_rate', store=True)
    
    @api.depends('total_recommendations', 'successful_recommendations')
    def _compute_success_rate(self):
        for record in self:
            if record.total_recommendations > 0:
                record.success_rate = (record.successful_recommendations / record.total_recommendations) * 100
            else:
                record.success_rate = 0.0

    @api.model
    def get_or_create_recommender(self):
        """Get existing recommender or create new one"""
        recommender = self.search([('active', '=', True)], limit=1)
        if not recommender:
            recommender = self.create({
                'name': 'Default Ollama Gift Recommender',
                'ollama_enabled': False  # Start with fallback mode until Ollama is configured
            })
        return recommender
    
    def test_ollama_connection(self):
        """Test connection to Ollama service"""
        self.ensure_one()
        
        if not self.ollama_enabled:
            return {
                'success': False,
                'message': 'Ollama is disabled. Enable it to use AI recommendations.'
            }
        
        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                
                if self.ollama_model not in models:
                    return {
                        'success': False,
                        'message': f'Model {self.ollama_model} not found. Available: {", ".join(models)}'
                    }
                
                return {
                    'success': True,
                    'message': f'✅ Connected! Model {self.ollama_model} is ready.'
                }
            else:
                return {
                    'success': False,
                    'message': f'Connection failed: HTTP {response.status_code}'
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'message': '❌ Cannot connect to Ollama. Is it running on ' + self.ollama_base_url + '?'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
    
    def _parse_notes_inline(self, notes):
        """Parse notes to extract mandatory requirements"""
        if not notes:
            return {'use_default': True}
        
        parsed = {
            'use_default': False,
            'mandatory_count': None,
            'categories_required': {},
            'categories_excluded': [],
            'budget_flexibility': 0.05,
            'preferences': {}
        }
        
        notes_lower = notes.lower()
        
        # Extract mandatory product count
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
        
        # Extract category requirements
        category_patterns = {
            'wines': r'(\d+)\s+wines?|wines?\s*:\s*(\d+)',
            'champagne': r'(\d+)\s+champagnes?|champagnes?\s*:\s*(\d+)',
            'sweets': r'(\d+)\s+sweets?|sweets?\s*:\s*(\d+)',
            'cheese': r'(\d+)\s+cheese?|cheese?\s*:\s*(\d+)',
            'ibericos': r'(\d+)\s+ibericos?|ibericos?\s*:\s*(\d+)',
        }
        
        for category, pattern in category_patterns.items():
            match = re.search(pattern, notes_lower)
            if match:
                count = int(match.group(1) or match.group(2))
                parsed['categories_required'][category] = count
        
        # Budget flexibility
        if 'flexible budget' in notes_lower:
            parsed['budget_flexibility'] = 0.20
        elif 'strict budget' in notes_lower:
            parsed['budget_flexibility'] = 0.05
        
        # Preferences
        parsed['preferences'] = {
            'loves_sweets': 'loves sweet' in notes_lower or 'sweet tooth' in notes_lower,
            'premium': 'premium' in notes_lower or 'luxury' in notes_lower,
            'traditional': 'traditional' in notes_lower or 'classic' in notes_lower
        }
        
        return parsed
    
    def generate_gift_recommendations(self, partner_id, target_budget, 
                                     client_notes='', dietary_restrictions=None):
        """Generate recommendations with notes enforcement"""
        
        # Parse notes FIRST
        notes_requirements = self._parse_notes_inline(client_notes)
        
        _logger.info(f"Generating for partner {partner_id}, budget: {target_budget}")
        _logger.info(f"Parsed notes requirements: {notes_requirements}")
        
        # Update tracking
        self.total_recommendations += 1
        self.last_recommendation_date = fields.Datetime.now()
        
        # If notes specify exact requirements, enforce them
        if notes_requirements.get('mandatory_count'):
            result = self._generate_with_mandatory_count(
                partner_id, target_budget, notes_requirements, 
                dietary_restrictions, client_notes
            )
            if result.get('success'):
                self.successful_recommendations += 1
            return result
        
        # Otherwise use standard flow
        partner = self.env['res.partner'].browse(partner_id)
        
        # Try Ollama if enabled
        if self.ollama_enabled:
            result = self._generate_with_ollama(
                partner, target_budget, client_notes, dietary_restrictions
            )
            if result and result.get('success'):
                self.successful_recommendations += 1
                return result
        
        # Fallback to rule-based
        result = self._generate_fallback_recommendation(
            partner, target_budget, client_notes, dietary_restrictions
        )
        if result.get('success'):
            self.successful_recommendations += 1
        return result
    
    def _generate_with_mandatory_count(self, partner_id, target_budget, 
                                       notes_requirements, dietary_restrictions, notes):
        """Generate EXACTLY the number of products specified in notes"""
        
        mandatory_count = notes_requirements['mandatory_count']
        _logger.info(f"Generating EXACTLY {mandatory_count} products as specified")
        
        # Get available products
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return {
                'success': False,
                'error': 'No products available matching criteria'
            }
        
        # Calculate target price per item
        avg_price_per_item = target_budget / mandatory_count
        
        selected = []
        
        # First, fulfill specific category requirements
        for category, count in notes_requirements.get('categories_required', {}).items():
            cat_products = [
                p for p in products 
                if category.lower() in p.name.lower()
            ]
            
            # Sort by how close they are to ideal price
            cat_products.sort(key=lambda p: abs(p.list_price - avg_price_per_item))
            
            # Take exactly the required count
            for i in range(min(count, len(cat_products))):
                if cat_products[i] not in selected:
                    selected.append(cat_products[i])
        
        # Fill remaining slots
        remaining_count = mandatory_count - len(selected)
        
        if remaining_count > 0:
            # Get products not yet selected
            available = [p for p in products if p not in selected]
            
            # If user loves sweets, prioritize them
            if notes_requirements.get('preferences', {}).get('loves_sweets'):
                sweet_products = [
                    p for p in available 
                    if 'sweet' in p.name.lower() or 'chocolate' in p.name.lower()
                ]
                available = sweet_products + [p for p in available if p not in sweet_products]
            
            # Sort by price proximity to ideal
            available.sort(key=lambda p: abs(p.list_price - avg_price_per_item))
            
            # Add products to reach exact count
            selected.extend(available[:remaining_count])
        
        # CRITICAL: Ensure EXACTLY the right count
        if len(selected) > mandatory_count:
            selected = selected[:mandatory_count]
        elif len(selected) < mandatory_count:
            # Add any products to reach count
            for p in products:
                if p not in selected:
                    selected.append(p)
                    if len(selected) >= mandatory_count:
                        break
        
        # Create composition
        total_cost = sum(p.list_price for p in selected)
        
        try:
            composition = self.env['gift.composition'].sudo().create({
                'partner_id': partner_id,
                'target_budget': target_budget,
                'target_year': fields.Date.today().year,
                'composition_type': 'custom',
                'product_ids': [(6, 0, [p.id for p in selected])],
                'state': 'draft',
                'client_notes': notes,  # Use client_notes field
                'ai_reasoning': f"Generated exactly {len(selected)} products as requested in notes: {notes}"
            })
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.8,
                'message': f'Generated exactly {len(selected)} products as requested'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {
                'success': False,
                'error': f'Failed to create composition: {str(e)}'
            }
    
    def _generate_with_ollama(self, partner, target_budget, client_notes, dietary_restrictions):
        """Generate using Ollama with proper notes enforcement"""
        
        # Parse notes
        notes_requirements = self._parse_notes_inline(client_notes)
        
        # Get products
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return None
        
        # Build prompt with notes enforcement
        prompt = self._build_ollama_prompt_with_notes(
            partner, target_budget, client_notes, notes_requirements,
            dietary_restrictions, products
        )
        
        # Call Ollama
        response = self._call_ollama(prompt, format_json=True)
        if not response:
            return None
        
        # Parse response
        return self._process_ollama_response(
            response, partner, target_budget, client_notes, 
            dietary_restrictions, notes_requirements
        )
    
    def _build_ollama_prompt_with_notes(self, partner, target_budget, client_notes, 
                                        notes_requirements, dietary_restrictions, products):
        """Build Ollama prompt that enforces notes requirements"""
        
        dietary_str = ', '.join(dietary_restrictions) if dietary_restrictions else 'None'
        
        # Format available products
        product_list = ""
        for i, p in enumerate(products[:50], 1):
            product_list += f"{i}. ID:{p.id} | {p.name} | €{p.list_price:.2f}\n"
        
        prompt = f"""You are an expert gift curator for Le Bigott. Follow these STRICT requirements:

CLIENT: {partner.name}
BUDGET: €{target_budget} (MUST be within 95-105% = €{target_budget*0.95:.2f} to €{target_budget*1.05:.2f})

MANDATORY REQUIREMENTS FROM NOTES:"""
        
        if notes_requirements.get('mandatory_count'):
            prompt += f"\n- EXACTLY {notes_requirements['mandatory_count']} products (no more, no less)"
        
        for category, count in notes_requirements.get('categories_required', {}).items():
            prompt += f"\n- {category}: EXACTLY {count} items"
        
        prompt += f"""

CLIENT NOTES: {client_notes}
DIETARY RESTRICTIONS: {dietary_str}

AVAILABLE PRODUCTS (select from these ONLY):
{product_list}

CRITICAL RULES:
1. {"Select EXACTLY " + str(notes_requirements['mandatory_count']) + " products" if notes_requirements.get('mandatory_count') else "Select 10-15 products"}
2. Total must be €{target_budget*0.95:.2f} to €{target_budget*1.05:.2f}
3. Honor all dietary restrictions
4. Follow category requirements from notes

Return ONLY valid JSON:
{{
    "selected_products": [
        {{"id": <product_id>, "reason": "<why selected>"}},
        ...
    ],
    "total_cost": <sum of prices>,
    "product_count": <number of products>,
    "reasoning": "<explanation>"
}}"""
        
        return prompt
    
    def _process_ollama_response(self, response, partner, target_budget, client_notes, 
                                 dietary_restrictions, notes_requirements):
        """Process Ollama response with notes validation"""
        
        try:
            # Parse JSON
            try:
                recommendation = json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    recommendation = json.loads(json_match.group())
                else:
                    return None
            
            # Extract product IDs
            product_ids = []
            for p in recommendation.get('selected_products', []):
                if isinstance(p, dict) and 'id' in p:
                    product_ids.append(p['id'])
            
            # Validate count if mandatory
            if notes_requirements.get('mandatory_count'):
                if len(product_ids) != notes_requirements['mandatory_count']:
                    _logger.warning(f"Ollama returned {len(product_ids)} products, expected {notes_requirements['mandatory_count']}")
                    product_ids = product_ids[:notes_requirements['mandatory_count']]
            
            # Get products
            selected_products = self.env['product.template'].sudo().browse(product_ids).exists()
            
            if not selected_products:
                return None
            
            # Create composition
            total_cost = sum(selected_products.mapped('list_price'))
            
            composition = self.env['gift.composition'].sudo().create({
                'partner_id': partner.id,
                'target_budget': target_budget,
                'target_year': fields.Date.today().year,
                'composition_type': 'custom',
                'product_ids': [(6, 0, selected_products.ids)],
                'state': 'draft',
                'client_notes': client_notes,  # Use client_notes field
                'ai_reasoning': recommendation.get('ai_reasoning', 'AI-generated selection')
            })
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected_products,
                'total_cost': total_cost,
                'confidence_score': 0.9,
                'message': f'AI generated {len(selected_products)} products'
            }
            
        except Exception as e:
            _logger.error(f"Failed to process Ollama response: {e}")
            return None
    
    def _call_ollama(self, prompt, format_json=False):
        """Make a call to Ollama API"""
        if not self.ollama_enabled:
            return None
        
        try:
            url = f"{self.ollama_base_url}/api/generate"
            
            payload = {
                'model': self.ollama_model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 2000
                }
            }
            
            if format_json:
                payload['format'] = 'json'
            
            response = requests.post(url, json=payload, timeout=self.ollama_timeout)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '').strip()
            else:
                _logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            _logger.error(f"Ollama request failed: {str(e)}")
            return None
    
    def _generate_fallback_recommendation(self, partner, target_budget, client_notes, dietary_restrictions):
        """Fallback with proper budget compliance and notes enforcement"""
        
        # Parse notes
        notes_requirements = self._parse_notes_inline(client_notes)
        
        # Get products
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return {'success': False, 'error': 'No products available'}
        
        # Select products
        if notes_requirements.get('mandatory_count'):
            selected = self._select_exact_count(
                products, notes_requirements['mandatory_count'], target_budget
            )
        else:
            selected = self._select_products_intelligently(products, target_budget)
        
        if not selected:
            return {'success': False, 'error': 'Could not select products'}
        
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        try:
            composition = self.env['gift.composition'].sudo().create({
                'partner_id': partner.id,
                'target_budget': target_budget,
                'target_year': fields.Date.today().year,
                'composition_type': 'custom',
                'product_ids': [(6, 0, [p.id for p in selected])],
                'state': 'draft',
                'client_notes': client_notes,  # Use client_notes field - THIS IS THE FIX
                'ai_reasoning': f"Rule-based selection: {len(selected)} products, €{total_cost:.2f}"
            })
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'confidence_score': 0.7,
                'message': f'Generated {len(selected)} products (€{total_cost:.2f})'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {
                'success': False,
                'error': f'Failed to create composition: {str(e)}'
            }
    
    def _select_exact_count(self, products, count, target_budget):
        """Select exactly 'count' products optimizing for budget"""
        
        avg_price = target_budget / count
        
        # Sort by distance from ideal price
        products_sorted = sorted(products, key=lambda p: abs(p.list_price - avg_price))
        
        # Take exactly 'count' products
        selected = products_sorted[:count]
        
        # Optimize for budget
        total = sum(p.list_price for p in selected)
        min_target = target_budget * 0.95
        max_target = target_budget * 1.05
        
        # If outside range, try to swap products
        if total < min_target or total > max_target:
            remaining = [p for p in products if p not in selected]
            
            for i in range(len(selected)):
                for r in remaining:
                    new_total = total - selected[i].list_price + r.list_price
                    if min_target <= new_total <= max_target:
                        selected[i] = r
                        return selected
        
        return selected
    
    def _select_products_intelligently(self, products, target_budget):
        """Select products to achieve 95-105% of budget"""
        
        if not products:
            return []
        
        # Target 95-105% of budget
        min_target = target_budget * 0.95
        max_target = target_budget * 1.05
        
        # Filter reasonable products
        min_price = max(10, target_budget * 0.02)
        suitable = [p for p in products if p.list_price >= min_price]
        
        if not suitable:
            suitable = products
        
        # Try multiple combinations
        best_combination = []
        best_total = 0
        
        for attempt in range(10):
            combo = []
            total = 0
            products_copy = suitable.copy()
            random.shuffle(products_copy)
            
            for product in products_copy:
                if total + product.list_price <= max_target:
                    combo.append(product)
                    total += product.list_price
                    
                    if min_target <= total <= max_target and len(combo) >= 5:
                        return combo
            
            if min_target <= total <= max_target:
                if abs(total - target_budget) < abs(best_total - target_budget):
                    best_combination = combo
                    best_total = total
        
        # If no perfect match, build greedily
        if not best_combination or best_total < min_target:
            sorted_products = sorted(suitable, key=lambda p: p.list_price, reverse=True)
            combo = []
            total = 0
            
            for product in sorted_products:
                if total + product.list_price <= max_target:
                    combo.append(product)
                    total += product.list_price
                    
                    if total >= min_target:
                        return combo
            
            best_combination = combo
        
        return best_combination
    
    def _get_available_products(self, target_budget, dietary_restrictions):
        """Get available products with proper filtering"""
        
        min_price = max(10, target_budget * 0.02)
        max_price = target_budget * 0.40
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
        ]
        
        # Add dietary filters if needed
        if dietary_restrictions:
            if 'halal' in dietary_restrictions:
                # Check if these fields exist before adding to domain
                if 'is_halal_compatible' in self.env['product.template']._fields:
                    domain.append(('is_halal_compatible', '!=', False))
                if 'contains_pork' in self.env['product.template']._fields:
                    domain.append(('contains_pork', '=', False))
                if 'contains_alcohol' in self.env['product.template']._fields:
                    domain.append(('contains_alcohol', '=', False))
        
        products = self.env['product.template'].sudo().search(domain, limit=500)
        
        # Filter by stock if available
        available = []
        for product in products:
            if hasattr(product, 'qty_available'):
                if product.qty_available > 0:
                    available.append(product)
            else:
                available.append(product)
        
        _logger.info(f"Found {len(available)} available products")
        return available
    
    def action_view_recommendations(self):
        """View all recommendations"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'All Recommendations',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('composition_type', '=', 'custom')],
            'context': {'search_default_partner_id': True}
        }
    
    def action_view_successful_recommendations(self):
        """View successful recommendations"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'Successful Recommendations',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [
                ('composition_type', '=', 'custom'),
                ('state', 'in', ['confirmed', 'approved', 'delivered'])
            ]
        }

    def trigger_learning(self):
        """Placeholder for future machine learning functionality"""
        self.ensure_one()
        
        # For now, just show a notification that this feature is coming soon
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': '🧠 Learning Module',
                'message': 'Machine learning features are coming soon! This will analyze past recommendations to improve future suggestions.',
                'type': 'info',
                'sticky': False,
            }
        }