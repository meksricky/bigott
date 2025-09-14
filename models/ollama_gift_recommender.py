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
    max_products = fields.Integer(string="Max Products per Recommendation", default=8)
    budget_flexibility = fields.Float(string="Budget Flexibility (%)", default=15.0)
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
            # Try to list available models
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
            
            start_time = datetime.now()
            response = requests.post(url, json=payload, timeout=self.ollama_timeout)
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update average response time
            self._update_response_time(response_time)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '').strip()
            else:
                _logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            _logger.error(f"Ollama request failed: {str(e)}")
            return None
    
    def _update_response_time(self, new_time):
        """Update average response time"""
        if self.total_recommendations > 0:
            self.avg_response_time = ((self.avg_response_time * self.total_recommendations) + new_time) / (self.total_recommendations + 1)
        else:
            self.avg_response_time = new_time
    
    def generate_gift_recommendations(self, partner_id, target_budget, client_notes='', dietary_restrictions=None):
        """Main method to generate gift recommendations"""
        self.ensure_one()
        
        try:
            partner = self.env['res.partner'].sudo().browse(partner_id)
            if not partner.exists():
                return {'success': False, 'error': 'Invalid partner'}
            
            _logger.info(f"Generating for {partner.name} - Budget: €{target_budget}")
            
            # Try Ollama first if enabled
            if self.ollama_enabled:
                result = self._generate_with_ollama(
                    partner, target_budget, client_notes, dietary_restrictions
                )
                if result and result.get('success'):
                    return result
                else:
                    _logger.info("Ollama failed, falling back to rule-based system")
            
            # Fallback to rule-based system
            return self._generate_fallback_recommendation(
                partner, target_budget, client_notes, dietary_restrictions
            )
            
        except Exception as e:
            _logger.error(f"Gift recommendation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': f'Recommendation failed: {str(e)}'
            }
    
    def _generate_with_ollama(self, partner, target_budget, client_notes, dietary_restrictions):
        """Generate recommendations using Ollama AI"""
        
        # Get available products
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return None
        
        # Build context
        history_context = self._get_client_context(partner.id)
        product_catalog = self._prepare_product_catalog(products, target_budget)
        
        # Create the prompt
        prompt = self._build_ollama_prompt(
            partner, target_budget, client_notes, 
            dietary_restrictions, history_context, product_catalog
        )
        
        # Call Ollama
        response = self._call_ollama(prompt, format_json=True)
        if not response:
            return None
        
        # Parse and process response
        return self._process_ollama_response(
            response, partner, target_budget, client_notes, dietary_restrictions
        )
    
    def _build_ollama_prompt(self, partner, target_budget, client_notes, 
                             dietary_restrictions, history_context, product_catalog):
        """Build the prompt for Ollama"""
        
        dietary_str = ', '.join(dietary_restrictions) if dietary_restrictions else 'None'
        
        return f"""You are an expert gift curator for Señor Bigott, a premium gift company.
Create a personalized gift composition for a client.

CLIENT INFORMATION:
- Name: {partner.name}
- Type: {'Company' if partner.is_company else 'Individual'}
- Budget: €{target_budget}
- Notes: {client_notes or 'None'}
- Dietary Restrictions: {dietary_str}

{history_context}

AVAILABLE PRODUCTS (select from these):
{product_catalog}

RULES:
1. Select 3-5 products that total 85-95% of the budget
2. Ensure variety in categories (no more than 2 from same category)
3. Honor all dietary restrictions
4. Create a cohesive gift experience with a clear theme
5. Balance premium and accessible items

Return ONLY valid JSON with this exact structure:
{{
    "selected_products": [
        {{"id": <product_id>, "reason": "<why this product>"}},
        ...
    ],
    "theme": "<overall gift theme>",
    "reasoning": "<detailed explanation of choices>",
    "confidence": <0.0-1.0>
}}"""
    
    def _process_ollama_response(self, response, partner, target_budget, client_notes, dietary_restrictions):
        """Process Ollama's response and create composition"""
        
        try:
            # Parse JSON response
            try:
                recommendation = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
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
            
            if not product_ids:
                return None
            
            # Get the actual products
            selected_products = self.env['product.template'].sudo().browse(product_ids).exists()
            if not selected_products:
                return None
            
            # Calculate total
            total_cost = sum(selected_products.mapped('list_price'))
            
            # Build reasoning HTML
            reasoning_html = self._build_reasoning_html(recommendation, total_cost, target_budget)
            
            # Create composition
            composition = self.env['gift.composition'].sudo().create({
                'partner_id': partner.id,
                'target_budget': target_budget,
                'target_year': fields.Date.today().year,
                'composition_type': 'ai_generated',
                'product_ids': [(6, 0, selected_products.ids)],
                'state': 'draft',
                'client_notes': client_notes,
                'confidence_score': recommendation.get('confidence', 0.8),
                'dietary_restrictions': ', '.join(dietary_restrictions) if dietary_restrictions else None,
                'reasoning': reasoning_html
            })
            
            # Update tracking
            self.sudo().write({
                'total_recommendations': self.total_recommendations + 1,
                'successful_recommendations': self.successful_recommendations + 1,
                'last_recommendation_date': fields.Datetime.now()
            })
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected_products,
                'total_cost': total_cost,
                'confidence_score': recommendation.get('confidence', 0.8),
                'message': f'AI generated {len(selected_products)} products for €{total_cost:.2f}'
            }
            
        except Exception as e:
            _logger.error(f"Failed to process Ollama response: {e}")
            return None
    
    def _build_reasoning_html(self, recommendation, total_cost, target_budget):
        """Build HTML reasoning from Ollama recommendation"""
        
        html = f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3>🤖 AI Gift Recommendation</h3>
            
            <div style="margin: 15px 0;">
                <strong>Theme:</strong> {recommendation.get('theme', 'Curated Gift Collection')}
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Budget Usage:</strong> €{total_cost:.2f} of €{target_budget:.2f} 
                ({(total_cost/target_budget*100):.1f}%)
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Confidence:</strong> {recommendation.get('confidence', 0.8)*100:.0f}%
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Reasoning:</strong>
                <p>{recommendation.get('reasoning', 'Products selected for optimal variety and value.')}</p>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Product Selection:</strong>
                <ul>
        """
        
        for product in recommendation.get('selected_products', []):
            html += f"<li>{product.get('reason', 'Selected for quality and value')}</li>"
        
        html += """
                </ul>
            </div>
        </div>
        """
        
        return html
    
    def _generate_fallback_recommendation(self, partner, target_budget, client_notes, dietary_restrictions):
        """Fallback recommendation when Ollama is not available"""
        
        # Get products
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return {'success': False, 'error': 'No products available'}
        
        # Smart selection algorithm
        selected_products = self._select_products_intelligently(products, target_budget)
        
        if not selected_products:
            return {'success': False, 'error': 'Could not select appropriate products'}
        
        total_cost = sum(p.list_price for p in selected_products)
        confidence = min(0.75, 1 - abs(target_budget - total_cost) / target_budget)
        
        # Create composition
        composition = self.env['gift.composition'].sudo().create({
            'partner_id': partner.id,
            'target_budget': target_budget,
            'target_year': fields.Date.today().year,
            'composition_type': 'ai_generated',
            'product_ids': [(6, 0, [p.id for p in selected_products])],
            'state': 'draft',
            'client_notes': f"Auto-generated for {partner.name}. {client_notes}",
            'confidence_score': confidence,
            'dietary_restrictions': ', '.join(dietary_restrictions) if dietary_restrictions else None,
            'reasoning': self._build_fallback_reasoning(selected_products, total_cost, target_budget)
        })
        
        # Update tracking
        self.sudo().write({
            'total_recommendations': self.total_recommendations + 1,
            'successful_recommendations': self.successful_recommendations + 1,
            'last_recommendation_date': fields.Datetime.now()
        })
        
        return {
            'success': True,
            'composition_id': composition.id,
            'products': selected_products,
            'total_cost': total_cost,
            'confidence_score': confidence,
            'message': f'Generated {len(selected_products)} products for €{total_cost:.2f} (Rule-based)'
        }
    
    def _get_available_products(self, target_budget, dietary_restrictions):
        """Get available products based on budget and restrictions"""
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>', 0),
            ('list_price', '<=', target_budget * 0.5)  # No single product over 50% of budget
        ]
        
        # Add dietary filters if needed
        if dietary_restrictions:
            # This would need custom fields on products
            # For now, filter by name/description
            pass
        
        products = self.env['product.template'].sudo().search(domain, limit=200)
        
        # Filter by stock availability
        available = []
        for product in products:
            if product.qty_available > 0:
                available.append(product)
        
        return available
    
    def _select_products_intelligently(self, products, target_budget):
        """Intelligent product selection algorithm"""
        
        # Group products by price range
        price_ranges = {
            'premium': [],    # > 40% of budget
            'mid': [],        # 20-40% of budget
            'standard': [],   # 10-20% of budget
            'small': []       # < 10% of budget
        }
        
        for product in products:
            price_ratio = product.list_price / target_budget
            if price_ratio > 0.4:
                price_ranges['premium'].append(product)
            elif price_ratio > 0.2:
                price_ranges['mid'].append(product)
            elif price_ratio > 0.1:
                price_ranges['standard'].append(product)
            else:
                price_ranges['small'].append(product)
        
        selected = []
        total = 0
        target = target_budget * 0.9  # Aim for 90% of budget
        
        # Strategy: 1 premium/mid + 2-3 standard + 1-2 small
        
        # Try to add one premium or mid-range item
        if price_ranges['premium'] and total + price_ranges['premium'][0].list_price <= target:
            item = random.choice(price_ranges['premium'][:5])
            selected.append(item)
            total += item.list_price
        elif price_ranges['mid']:
            item = random.choice(price_ranges['mid'][:5])
            if total + item.list_price <= target:
                selected.append(item)
                total += item.list_price
        
        # Add standard items
        for item in sorted(price_ranges['standard'], key=lambda x: x.list_price, reverse=True)[:10]:
            if total + item.list_price <= target and item not in selected:
                selected.append(item)
                total += item.list_price
                if len(selected) >= 3 and total >= target_budget * 0.75:
                    break
        
        # Fill with small items if needed
        for item in price_ranges['small'][:10]:
            if total + item.list_price <= target_budget * 1.05 and item not in selected:
                selected.append(item)
                total += item.list_price
                if len(selected) >= 5 or total >= target_budget * 0.85:
                    break
        
        return selected
    
    def _build_fallback_reasoning(self, products, total_cost, target_budget):
        """Build reasoning HTML for fallback recommendations"""
        
        categories = {}
        for p in products:
            cat = getattr(p, 'lebiggot_category', 'General')
            categories[cat] = categories.get(cat, 0) + 1
        
        return f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3>📦 Smart Recommendation (Rule-based)</h3>
            
            <div style="margin: 15px 0;">
                <strong>Selection Method:</strong> Intelligent rule-based algorithm
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Products:</strong> {len(products)} items selected
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Total Cost:</strong> €{total_cost:.2f} of €{target_budget:.2f} 
                ({(total_cost/target_budget*100):.1f}% budget usage)
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Categories:</strong> {', '.join([f"{cat} ({count})" for cat, count in categories.items()])}
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Strategy:</strong> Balanced selection with premium and accessible items for optimal gift experience.
            </div>
        </div>
        """
    
    def _get_client_context(self, partner_id):
        """Get client history context"""
        try:
            history = self.env['client.order.history'].sudo().search([
                ('partner_id', '=', partner_id)
            ], limit=3, order='order_year desc')
            
            if not history:
                return "HISTORY: New client - no previous orders"
            
            context = "HISTORY:\n"
            for h in history:
                context += f"- {h.order_year}: €{h.total_budget:.0f} budget, {h.total_products} products\n"
            
            return context
        except:
            return "HISTORY: Not available"
    
    def _prepare_product_catalog(self, products, target_budget):
        """Prepare product catalog for AI"""
        
        catalog = []
        for product in products[:50]:  # Limit to top 50 to avoid huge prompts
            catalog.append({
                'id': product.id,
                'name': product.name[:50],  # Truncate long names
                'price': product.list_price,
                'category': getattr(product, 'lebiggot_category', 'general'),
                'stock': product.qty_available
            })
        
        return json.dumps(catalog, indent=2)
    
    # Action methods for buttons
    def action_view_recommendations(self):
        """View all recommendations"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'All Recommendations',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('composition_type', '=', 'ai_generated')],
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
                ('composition_type', '=', 'ai_generated'),
                ('state', 'in', ['confirmed', 'approved', 'delivered'])
            ]
        }