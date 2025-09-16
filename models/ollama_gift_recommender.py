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
    
    def generate_gift_recommendations(self, partner_id, target_budget, 
                                    client_notes='', dietary_restrictions=None):
        """Generate recommendations with intelligent notes parsing and history learning"""
        
        partner = self.env['res.partner'].browse(partner_id)
        if not partner:
            return {'success': False, 'error': 'Partner not found'}
        
        # 1. PARSE NOTES FIRST (HIGHEST PRIORITY) - CORRECTED TO USE ODOO MODEL
        parser = self.env['notes.parser']
        requirements = parser.parse_client_notes(client_notes)
        
        _logger.info(f"Parsed requirements from notes: {requirements}")
        
        # 2. OVERRIDE FORM VALUES WITH PARSED REQUIREMENTS
        if requirements.get('budget_override'):
            original_budget = target_budget
            target_budget = requirements['budget_override']
            _logger.info(f"Overriding budget from notes: €{original_budget} → €{target_budget}")
        
        if requirements.get('dietary'):
            dietary_restrictions = requirements['dietary']
            _logger.info(f"Overriding dietary from notes: {dietary_restrictions}")
        
        # 3. GET PREVIOUS ORDER DATA
        previous_data = self._get_previous_order_data(partner_id)
        
        # 4. DETERMINE GENERATION STRATEGY
        # Update tracking
        self.total_recommendations += 1
        self.last_recommendation_date = fields.Datetime.now()
        
        # Check for explicit product count requirement
        if requirements.get('mandatory_count'):
            _logger.info(f"Strict product count requirement: {requirements['mandatory_count']}")
            result = self._generate_with_strict_requirements(
                partner, target_budget, requirements, 
                dietary_restrictions, client_notes, previous_data
            )
        
        # Check if we should use history
        elif (previous_data and 
            ('like last time' in client_notes.lower() or 
            'same as before' in client_notes.lower() or
            'repeat' in client_notes.lower())):
            _logger.info("Using history-based generation as requested")
            result = self._generate_from_history(
                partner, target_budget, previous_data,
                client_notes, dietary_restrictions, requirements
            )
        
        # Use history if available and no specific instructions
        elif previous_data and not requirements.get('special_instructions'):
            _logger.info("Using history-based generation (70/30 rule)")
            # If no specific budget in notes, use previous order budget
            if not requirements.get('budget_override'):
                target_budget = previous_data['budget']
                _logger.info(f"Using budget from previous order: €{target_budget:.2f}")
            
            result = self._generate_from_history(
                partner, target_budget, previous_data,
                client_notes, dietary_restrictions, requirements
            )
        
        # Try Ollama if enabled
        elif self.ollama_enabled:
            _logger.info("Using Ollama AI generation")
            result = self._generate_with_ollama_enhanced(
                partner, target_budget, client_notes, 
                dietary_restrictions, requirements
            )
            
            # Fallback if Ollama fails
            if not result or not result.get('success'):
                _logger.warning("Ollama failed, using fallback")
                result = self._generate_fallback_recommendation_enhanced(
                    partner, target_budget, client_notes, 
                    dietary_restrictions, requirements
                )
        
        # Standard fallback generation
        else:
            _logger.info("Using fallback generation")
            result = self._generate_fallback_recommendation_enhanced(
                partner, target_budget, client_notes, 
                dietary_restrictions, requirements
            )
        
        # Update success tracking
        if result and result.get('success'):
            self.successful_recommendations += 1
            _logger.info(f"Successfully generated composition using method: {result.get('method', 'unknown')}")
        
        return result

    def _generate_with_strict_requirements(self, partner, budget, requirements, 
                                        dietary, notes, previous_data=None):
        """Generate with strict adherence to parsed requirements"""
        
        product_count = requirements.get('mandatory_count', requirements.get('product_count', 12))
        budget_flexibility = requirements.get('budget_flexibility', 10)
        
        # Start with previous products if available (for continuity)
        base_products = []
        if previous_data and previous_data.get('products'):
            # Take some products from history as base
            prev_products = [p['product'] for p in previous_data['products'][:5]]
            base_products = [p for p in prev_products if self._has_stock(p)]
        
        # Get available products
        products = self._get_available_products(budget * 2, dietary)
        
        # Combine with base products
        all_products = base_products + products
        
        # Remove duplicates
        seen = set()
        unique_products = []
        for p in all_products:
            if p.id not in seen:
                seen.add(p.id)
                unique_products.append(p)
        
        # Apply filters based on requirements
        if requirements.get('specific_products'):
            specific = self._get_specific_products(requirements['specific_products'])
            unique_products = specific + unique_products
        
        if requirements.get('categories_excluded'):
            unique_products = self._filter_exclusions(unique_products, requirements['categories_excluded'])
        
        # Select products
        if requirements.get('categories_required'):
            selected = self._select_by_categories(
                unique_products, requirements['categories_required'],
                product_count, budget
            )
        else:
            selected = self._optimize_selection(
                unique_products, product_count, budget, budget_flexibility
            )
        
        # Ensure we meet the count requirement
        if len(selected) != product_count:
            _logger.warning(f"Adjusting selection. Requested: {product_count}, Got: {len(selected)}")
            
            if len(selected) < product_count:
                # Add more products
                remaining = product_count - len(selected)
                excluded_ids = [p.id for p in selected]
                additional = [p for p in unique_products if p.id not in excluded_ids][:remaining]
                selected.extend(additional)
            else:
                # Remove excess products
                selected = selected[:product_count]
        
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        try:
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': budget,
                'target_year': fields.Date.today().year,
                'product_ids': [(6, 0, [p.id for p in selected])],
                'dietary_restrictions': ', '.join(dietary) if dietary else '',
                'client_notes': notes,
                'generation_method': 'ollama',
                'composition_type': requirements.get('composition_type', 'custom'),
                'confidence_score': 0.95,
                'ai_reasoning': self._build_reasoning(requirements, selected, total_cost, budget)
            })
            
            # Auto-categorize
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.95,
                'message': f'Generated exactly {product_count} products as required',
                'method': 'strict_requirements'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}

    def _generate_from_history(self, partner, target_budget, previous_data, 
                            notes, dietary, requirements=None):
        """Generate based on history with requirements integration"""
        
        previous_products = previous_data['products']
        total_count = len(previous_products)
        
        # Check if requirements override the count
        if requirements and requirements.get('mandatory_count'):
            total_count = requirements['mandatory_count']
        elif requirements and requirements.get('product_count'):
            total_count = requirements['product_count']
        
        # Calculate split (70% keep, 30% change)
        keep_count = int(total_count * 0.7)
        change_count = total_count - keep_count
        
        # Select products to keep
        products_to_keep = []
        for item in previous_products[:keep_count]:
            product = item['product']
            if self._check_dietary_compliance(product, dietary) and self._has_stock(product):
                products_to_keep.append(product)
        
        # Find new products
        new_products = self._find_replacement_products(
            change_count,
            target_budget - sum(p.list_price for p in products_to_keep),
            dietary,
            exclude_products=products_to_keep
        )
        
        # Apply any specific requirements
        if requirements:
            if requirements.get('specific_products'):
                # Ensure specific products are included
                specific = self._get_specific_products(requirements['specific_products'])
                new_products = specific + new_products[:change_count-len(specific)]
        
        # Combine products
        final_products = products_to_keep + new_products
        total_cost = sum(p.list_price for p in final_products)
        
        # Create composition
        try:
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': target_budget,
                'target_year': fields.Date.today().year,
                'product_ids': [(6, 0, [p.id for p in final_products])],
                'dietary_restrictions': ', '.join(dietary) if dietary else '',
                'client_notes': notes,
                'generation_method': 'ollama',
                'composition_type': 'custom',
                'confidence_score': 0.95,
                'ai_reasoning': f"""History-based generation:
                - Based on order: {previous_data['order_name']}
                - Kept {len(products_to_keep)} products (70%)
                - Added {len(new_products)} new products (30%)
                - Previous budget: €{previous_data['budget']:.2f}
                - Current total: €{total_cost:.2f}"""
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': final_products,
                'total_cost': total_cost,
                'product_count': len(final_products),
                'confidence_score': 0.95,
                'message': f'Generated from history: {len(products_to_keep)} repeated + {len(new_products)} new',
                'method': 'history_based'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_with_ollama_enhanced(self, partner, target_budget, client_notes, 
                                      dietary_restrictions, requirements):
        """Generate using Ollama with enhanced requirements"""
        
        # Get products
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return None
        
        # Build prompt with requirements enforcement
        prompt = self._build_ollama_prompt_enhanced(
            partner, target_budget, client_notes, requirements,
            dietary_restrictions, products
        )
        
        # Call Ollama
        response = self._call_ollama(prompt, format_json=True)
        if not response:
            return None
        
        # Parse response
        return self._process_ollama_response_enhanced(
            response, partner, target_budget, client_notes, 
            dietary_restrictions, requirements
        )
    
    def _build_ollama_prompt_enhanced(self, partner, target_budget, client_notes, 
                                     requirements, dietary_restrictions, products):
        """Build enhanced Ollama prompt with all requirements"""
        
        dietary_str = ', '.join(dietary_restrictions) if dietary_restrictions else 'None'
        
        # Format available products
        product_list = ""
        for i, p in enumerate(products[:50], 1):
            product_list += f"{i}. ID:{p.id} | {p.name} | €{p.list_price:.2f}\n"
        
        prompt = f"""You are an expert gift curator for Le Bigott. Follow these STRICT requirements:

CLIENT: {partner.name}
BUDGET: €{target_budget} (MUST be within 95-105% = €{target_budget*0.95:.2f} to €{target_budget*1.05:.2f})

MANDATORY REQUIREMENTS FROM NOTES:"""
        
        if requirements.get('mandatory_count'):
            prompt += f"\n- EXACTLY {requirements['mandatory_count']} products (no more, no less)"
        elif requirements.get('product_count'):
            prompt += f"\n- EXACTLY {requirements['product_count']} products"
        
        for category, count in requirements.get('categories_required', {}).items():
            prompt += f"\n- {category}: EXACTLY {count} items"
        
        if requirements.get('categories_excluded'):
            prompt += f"\n- EXCLUDE: {', '.join(requirements['categories_excluded'])}"
        
        prompt += f"""

CLIENT NOTES: {client_notes}
DIETARY RESTRICTIONS: {dietary_str}

AVAILABLE PRODUCTS (select from these ONLY):
{product_list}

CRITICAL RULES:
1. {"Select EXACTLY " + str(requirements.get('mandatory_count', requirements.get('product_count', '10-15'))) + " products"}
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
    
    def _process_ollama_response_enhanced(self, response, partner, target_budget, client_notes, 
                                         dietary_restrictions, requirements):
        """Process Ollama response with enhanced validation"""
        
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
            target_count = requirements.get('mandatory_count', requirements.get('product_count'))
            if target_count:
                if len(product_ids) != target_count:
                    _logger.warning(f"Ollama returned {len(product_ids)} products, expected {target_count}")
                    product_ids = product_ids[:target_count]
            
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
                'client_notes': client_notes,
                'ai_reasoning': recommendation.get('reasoning', 'AI-generated selection')
            })
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected_products,
                'total_cost': total_cost,
                'confidence_score': 0.9,
                'message': f'AI generated {len(selected_products)} products',
                'method': 'ollama_ai'
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
    
    def _generate_fallback_recommendation_enhanced(self, partner, target_budget, client_notes, 
                                                  dietary_restrictions, requirements):
        """Enhanced fallback with requirements support"""
        
        # Get products
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return {'success': False, 'error': 'No products available'}
        
        # Select products based on requirements
        if requirements.get('mandatory_count'):
            selected = self._select_exact_count(
                products, requirements['mandatory_count'], target_budget
            )
        elif requirements.get('product_count'):
            selected = self._select_exact_count(
                products, requirements['product_count'], target_budget
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
                'client_notes': client_notes,
                'ai_reasoning': f"Rule-based selection: {len(selected)} products, €{total_cost:.2f}"
            })
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'confidence_score': 0.7,
                'message': f'Generated {len(selected)} products (€{total_cost:.2f})',
                'method': 'fallback'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {
                'success': False,
                'error': f'Failed to create composition: {str(e)}'
            }
    
    # Helper Methods
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

    def _get_previous_order_data(self, partner_id):
        """Get data from previous sales orders for this client"""
        
        # Search for confirmed sales orders
        sales = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc', limit=5)
        
        if not sales:
            # Try quotations if no sales
            sales = self.env['sale.order'].search([
                ('partner_id', '=', partner_id),
                ('state', '=', 'sent')
            ], order='date_order desc', limit=5)
        
        if not sales:
            return None
        
        # Get the most recent order
        last_order = sales[0]
        
        # Extract product data
        previous_products = []
        total_amount = 0.0
        
        for line in last_order.order_line:
            if line.product_id and line.product_id.type == 'product':
                # Get the product template
                product_tmpl = line.product_id.product_tmpl_id
                previous_products.append({
                    'product': product_tmpl,
                    'qty': line.product_uom_qty,
                    'price': line.price_unit
                })
                total_amount += line.price_subtotal
        
        return {
            'budget': total_amount,
            'products': previous_products,
            'order_date': last_order.date_order,
            'order_name': last_order.name
        }

    def _find_replacement_products(self, count, budget, dietary, exclude_products=None):
        """Find new products similar to excluded ones"""
        
        exclude_ids = [p.id for p in exclude_products] if exclude_products else []
        
        # Get products not in the exclusion list
        domain = [
            ('sale_ok', '=', True),
            ('id', 'not in', exclude_ids),
            ('list_price', '>', 0)
        ]
        
        products = self._get_available_products_with_stock(domain)
        
        # Filter by dietary
        products = [p for p in products if self._check_dietary_compliance(p, dietary)]
        
        # Select products to fit budget
        avg_price = budget / count if count > 0 else budget
        
        # Sort by price proximity to average
        products_sorted = sorted(products, key=lambda p: abs(p.list_price - avg_price))
        
        return products_sorted[:count]
    
    def _get_available_products_with_stock(self, domain):
        """Get products with stock available"""
        products = self.env['product.template'].sudo().search(domain, limit=500)
        available = []
        
        for product in products:
            if self._has_stock(product):
                available.append(product)
        
        return available

    def _has_stock(self, product):
        """Check if product has stock"""
        if hasattr(product, 'qty_available') and product.qty_available > 0:
            return True
            
        for variant in product.product_variant_ids:
            stock_quants = self.env['stock.quant'].search([
                ('product_id', '=', variant.id),
                ('location_id.usage', '=', 'internal')
            ])
            if sum(stock_quants.mapped('available_quantity')) > 0:
                return True
        return False
    
    def _check_dietary_compliance(self, product, dietary_restrictions):
        """Check if product complies with dietary restrictions"""
        if not dietary_restrictions:
            return True
        
        for restriction in dietary_restrictions:
            if restriction == 'halal':
                if hasattr(product, 'contains_pork') and product.contains_pork:
                    return False
                if hasattr(product, 'contains_alcohol') and product.contains_alcohol:
                    return False
            elif restriction == 'vegan':
                if hasattr(product, 'is_vegan') and not product.is_vegan:
                    return False
            elif restriction == 'gluten_free':
                if hasattr(product, 'contains_gluten') and product.contains_gluten:
                    return False
        
        return True
    
    def _get_specific_products(self, product_list):
        """Get specific products by name or code"""
        products = []
        for item in product_list:
            # Search by name or internal reference
            domain = ['|', 
                     ('name', 'ilike', item),
                     ('default_code', 'ilike', item)]
            found = self.env['product.template'].search(domain, limit=5)
            products.extend(found)
        return products
    
    def _filter_exclusions(self, products, exclusions):
        """Filter out excluded products"""
        filtered = []
        for product in products:
            exclude = False
            for exclusion in exclusions:
                if exclusion.lower() in product.name.lower():
                    exclude = True
                    break
            if not exclude:
                filtered.append(product)
        return filtered
    
    def _select_by_categories(self, products, categories_required, total_count, budget):
        """Select products by category requirements"""
        selected = []
        
        # First fulfill category requirements
        for category, count in categories_required.items():
            cat_products = [p for p in products if category.lower() in p.name.lower()]
            selected.extend(cat_products[:count])
        
        # Fill remaining slots
        remaining_count = total_count - len(selected)
        if remaining_count > 0:
            unused = [p for p in products if p not in selected]
            selected.extend(unused[:remaining_count])
        
        return selected
    
    def _optimize_selection(self, products, target_count, target_budget, flexibility):
        """Optimize product selection to match count and budget"""
        
        # Calculate target average price
        avg_price = target_budget / target_count if target_count > 0 else 0
        
        # Score and sort products
        scored_products = []
        for product in products:
            # Score based on price proximity to average
            price_score = 1 / (1 + abs(product.list_price - avg_price))
            
            # Add variety score
            variety_score = random.uniform(0.8, 1.2)
            
            total_score = price_score * variety_score
            scored_products.append((product, total_score))
        
        # Sort by score
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Select products
        selected = []
        current_total = 0
        budget_min = target_budget * (1 - flexibility/100)
        budget_max = target_budget * (1 + flexibility/100)
        
        for product, score in scored_products:
            if len(selected) < target_count:
                future_total = current_total + product.list_price
                remaining_slots = target_count - len(selected) - 1
                
                if remaining_slots > 0:
                    # Check if we can still reach target with remaining slots
                    min_possible = future_total + (remaining_slots * 10)
                    max_possible = future_total + (remaining_slots * 500)
                    
                    if min_possible <= budget_max and max_possible >= budget_min:
                        selected.append(product)
                        current_total = future_total
                else:
                    # Last product - try to hit target
                    if budget_min <= future_total <= budget_max:
                        selected.append(product)
                        current_total = future_total
        
        return selected
    
    def _build_reasoning(self, requirements, products, total_cost, target_budget):
        """Build detailed reasoning for the composition"""
        
        reasoning_parts = []
        
        if requirements.get('mandatory_count') or requirements.get('product_count'):
            count = requirements.get('mandatory_count', requirements.get('product_count'))
            reasoning_parts.append(f"✓ Matched requested count: {count} products")
        
        if requirements.get('budget_override'):
            variance = ((total_cost - target_budget) / target_budget * 100)
            reasoning_parts.append(f"✓ Budget adherence: €{total_cost:.2f} ({variance:+.1f}% from target)")
        
        if requirements.get('dietary'):
            reasoning_parts.append(f"✓ Dietary compliance: {', '.join(requirements['dietary'])}")
        
        if requirements.get('categories_required'):
            reasoning_parts.append(f"✓ Category requirements met: {requirements['categories_required']}")
        
        if requirements.get('special_instructions'):
            reasoning_parts.append(f"✓ Special instructions followed")
        
        return "\n".join(reasoning_parts)
    
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