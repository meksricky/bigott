# models/ollama_gift_recommender.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import json
import logging
import requests
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re

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
                'ollama_enabled': True
            })
        return recommender
    
    def test_ollama_connection(self):
        """Test connection to Ollama service"""
        if not self.ollama_enabled:
            raise UserError("Ollama is not enabled for this recommender.")
        
        try:
            url = f"{self.ollama_base_url.rstrip('/')}/api/tags"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                
                if self.ollama_model in models:
                    return {
                        'success': True,
                        'message': f"✅ Successfully connected to Ollama. Model '{self.ollama_model}' is available.",
                        'available_models': models
                    }
                else:
                    return {
                        'success': False,
                        'message': f"❌ Model '{self.ollama_model}' not found. Available models: {', '.join(models)}",
                        'available_models': models
                    }
            else:
                return {
                    'success': False,
                    'message': f"❌ Ollama service returned status {response.status_code}",
                    'available_models': []
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'message': "❌ Cannot connect to Ollama service. Please check if Ollama is running and the URL is correct.",
                'available_models': []
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Error testing Ollama connection: {str(e)}",
                'available_models': []
            }
    
    def _call_ollama(self, prompt, system_prompt=None):
        """Make a call to Ollama API"""
        if not self.ollama_enabled:
            _logger.warning("Ollama not enabled - skipping AI recommendation")
            return None
        
        try:
            url = f"{self.ollama_base_url.rstrip('/')}/api/generate"
            
            # Build the full prompt with system instructions if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            payload = {
                'model': self.ollama_model,
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 1500
                }
            }
            
            start_time = datetime.now()
            response = requests.post(url, json=payload, timeout=self.ollama_timeout)
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update average response time
            if self.total_recommendations > 0:
                self.avg_response_time = ((self.avg_response_time * self.total_recommendations) + response_time) / (self.total_recommendations + 1)
            else:
                self.avg_response_time = response_time
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '').strip()
            else:
                _logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            _logger.error(f"Ollama request timed out after {self.ollama_timeout} seconds")
            return None
        except Exception as e:
            _logger.error(f"Ollama request failed: {str(e)}")
            return None
    
    def generate_gift_recommendations(self, partner_id, target_budget, client_notes=None, dietary_restrictions=None):
        """
        Main method to generate gift recommendations using Ollama
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if not partner_id or target_budget <= 0:
                raise UserError("Partner ID and valid budget are required for recommendations.")
            
            # Get client information
            partner = self.env['res.partner'].browse(partner_id)
            if not partner.exists():
                raise UserError("Client not found.")
            
            # Gather client data
            client_data = self._gather_client_data(partner)
            
            # Get available products with stock and internal reference
            available_products = self._get_available_products(dietary_restrictions)
            
            if not available_products:
                raise UserError("No products available that meet the criteria (stock, internal reference, dietary restrictions).")
            
            # Build Ollama prompt
            prompt = self._build_recommendation_prompt(
                client_data, target_budget, client_notes, 
                dietary_restrictions, available_products
            )
            
            # Get Ollama recommendation
            system_prompt = self._get_system_prompt()
            ollama_response = self._call_ollama(prompt, system_prompt)
            
            if not ollama_response:
                # Fallback to rule-based recommendation
                _logger.warning("Ollama failed - using fallback recommendation")
                return self._fallback_recommendation(partner, target_budget, available_products, dietary_restrictions)
            
            # Parse Ollama response
            recommendation_result = self._parse_ollama_response(ollama_response, available_products, target_budget)
            
            # Create gift composition record
            composition = self._create_gift_composition(
                partner, recommendation_result, target_budget, 
                client_notes, dietary_restrictions, ollama_response
            )
            
            # Update statistics
            self.total_recommendations += 1
            self.successful_recommendations += 1
            self.last_recommendation_date = fields.Datetime.now()
            
            _logger.info(f"Successfully generated Ollama recommendation for {partner.name} in {(datetime.now() - start_time).total_seconds():.2f} seconds")
            
            return {
                'success': True,
                'composition_id': composition.id,
                'message': f"Successfully generated gift recommendation for {partner.name}",
                'products': recommendation_result['products'],
                'total_cost': sum(p.list_price for p in recommendation_result['products']),
                'reasoning': recommendation_result.get('reasoning', ''),
                'confidence_score': recommendation_result.get('confidence_score', 0.5)
            }
            
        except Exception as e:
            self.total_recommendations += 1  # Count failed attempts too
            _logger.error(f"Gift recommendation failed: {str(e)}")
            return {
                'success': False,
                'message': f"Recommendation failed: {str(e)}",
                'error': str(e)
            }
    
    def _gather_client_data(self, partner):
        """Gather comprehensive client data for analysis"""
        
        # Get historical sales data
        sales_orders = self.env['sale.order'].search([
            ('partner_id', '=', partner.id),
            ('state', 'in', ['sale', 'done']),
            ('amount_total', '>', 0)
        ], order='date_order desc', limit=10)
        
        # Get historical gift compositions
        gift_compositions = self.env['gift.composition'].search([
            ('partner_id', '=', partner.id)
        ], order='create_date desc', limit=5)
        
        # Get client order history
        client_history = self.env['client.order.history'].search([
            ('partner_id', '=', partner.id)
        ], order='order_year desc', limit=3)
        
        return {
            'partner': partner,
            'sales_orders': sales_orders,
            'gift_compositions': gift_compositions,
            'client_history': client_history,
            'total_past_orders': len(sales_orders),
            'avg_order_value': sum(sales_orders.mapped('amount_total')) / len(sales_orders) if sales_orders else 0,
            'preferred_categories': self._analyze_preferred_categories(sales_orders, gift_compositions),
            'past_budgets': sales_orders.mapped('amount_total')[:5],  # Last 5 budgets
        }
    
    def _analyze_preferred_categories(self, sales_orders, gift_compositions):
        """Analyze client's preferred product categories"""
        categories = []
        
        # From sales orders
        for order in sales_orders:
            for line in order.order_line:
                if line.product_id.product_tmpl_id.lebiggot_category:
                    categories.append(line.product_id.product_tmpl_id.lebiggot_category)
        
        # From gift compositions
        for comp in gift_compositions:
            for product in comp.product_ids:
                if product.lebiggot_category:
                    categories.append(product.lebiggot_category)
        
        # Count occurrences and return most common
        if categories:
            category_counts = Counter(categories)
            return [cat for cat, count in category_counts.most_common(5)]
        
        return []
    
    def _get_available_products(self, dietary_restrictions=None):
        """Get products that are available (stock + internal reference) and meet dietary restrictions"""
        
        # Base domain for available products
        domain = [
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('list_price', '>', 0),
            ('default_code', '!=', False),  # Must have internal reference
            ('lebiggot_category', '!=', False),  # Must have category
        ]
        
        products = self.env['product.template'].search(domain)
        
        # Filter by stock availability
        available_products = []
        for product in products:
            if self._check_product_stock(product):
                available_products.append(product)
        
        # Filter by dietary restrictions
        if dietary_restrictions:
            filtered_products = []
            for product in available_products:
                if self._check_dietary_compliance(product, dietary_restrictions):
                    filtered_products.append(product)
            available_products = filtered_products
        
        return available_products
    
    def _check_product_stock(self, product):
        """Check if product has available stock"""
        if product.type != 'product':
            return True  # Services and consumables are always "available"
        
        stock_quants = self.env['stock.quant'].search([
            ('product_id', 'in', product.product_variant_ids.ids),
            ('location_id.usage', '=', 'internal')
        ])
        
        available_qty = sum(stock_quants.mapped('available_quantity'))
        return available_qty > 0
    
    def _check_dietary_compliance(self, product, dietary_restrictions):
        """Check if product meets dietary restrictions"""
        
        for restriction in dietary_restrictions:
            restriction = restriction.lower().strip()
            
            if restriction == 'vegan':
                # Check product fields
                if hasattr(product, 'is_vegan') and not product.is_vegan:
                    return False
                # Keyword check in product name/description
                non_vegan_keywords = ['meat', 'fish', 'dairy', 'cheese', 'ham', 'salmon', 'beef', 'pork', 'chicken', 'milk']
                if any(keyword in product.name.lower() for keyword in non_vegan_keywords):
                    return False
            
            elif restriction == 'halal':
                # Check product fields
                if hasattr(product, 'is_halal') and not product.is_halal:
                    return False
                # Keyword check
                non_halal_keywords = ['pork', 'wine', 'alcohol', 'champagne', 'beer', 'whiskey', 'brandy']
                if any(keyword in product.name.lower() for keyword in non_halal_keywords):
                    return False
            
            elif restriction in ['non_alcoholic', 'no_alcohol']:
                # Check product fields
                if hasattr(product, 'contains_alcohol') and product.contains_alcohol:
                    return False
                # Keyword check
                alcohol_keywords = ['wine', 'champagne', 'alcohol', 'beer', 'whiskey', 'brandy', 'vodka', 'rum', 'gin']
                if any(keyword in product.name.lower() for keyword in alcohol_keywords):
                    return False
            
            elif restriction == 'gluten_free':
                # Check product fields
                if hasattr(product, 'is_gluten_free') and not product.is_gluten_free:
                    return False
                # Keyword check
                gluten_keywords = ['wheat', 'bread', 'pasta', 'flour', 'barley', 'oats']
                if any(keyword in product.name.lower() for keyword in gluten_keywords):
                    return False
        
        return True
    
    def _get_system_prompt(self):
        """Get the system prompt for Ollama"""
        return """You are a luxury gourmet gift recommendation specialist for Señor Bigott, an exclusive Spanish luxury food company. Your role is to analyze client data and recommend the perfect gift composition.

CRITICAL REQUIREMENTS:
1. Only recommend products that are explicitly listed in the available products
2. Respect the client's budget (within 15% flexibility)
3. Honor all dietary restrictions
4. Consider the client's purchase history and preferences
5. Ensure variety and balance in the gift composition
6. Focus on luxury and premium quality

RESPONSE FORMAT:
Provide your response in this exact JSON format:
{
    "recommended_products": ["product_code_1", "product_code_2", "..."],
    "reasoning": "Detailed explanation of why these products were chosen",
    "confidence_score": 0.85,
    "total_estimated_cost": 150.50,
    "composition_notes": "Brief notes about the overall composition"
}

Always respond with valid JSON only. Do not include any text outside the JSON structure."""
    
    def _build_recommendation_prompt(self, client_data, target_budget, client_notes, dietary_restrictions, available_products):
        """Build the recommendation prompt for Ollama"""
        
        # Build client profile
        partner = client_data['partner']
        client_profile = f"""
CLIENT PROFILE:
- Name: {partner.name}
- Total Past Orders: {client_data['total_past_orders']}
- Average Order Value: €{client_data['avg_order_value']:.2f}
- Preferred Categories: {', '.join(client_data['preferred_categories']) if client_data['preferred_categories'] else 'None identified'}
- Recent Budget History: {[f"€{budget:.2f}" for budget in client_data['past_budgets']]}
"""
        
        # Build dietary restrictions section
        dietary_info = ""
        if dietary_restrictions:
            dietary_info = f"\nDIETARY RESTRICTIONS: {', '.join(dietary_restrictions)}"
        
        # Build client notes section
        notes_info = ""
        if client_notes:
            notes_info = f"\nCLIENT NOTES: {client_notes}"
        
        # Build available products section (limit to reasonable number for prompt)
        products_info = "\nAVAILABLE PRODUCTS:\n"
        for product in available_products[:50]:  # Limit to first 50 to avoid token limits
            products_info += f"- Code: {product.default_code}, Name: {product.name}, Price: €{product.list_price:.2f}, Category: {product.lebiggot_category or 'N/A'}, Grade: {product.product_grade or 'N/A'}\n"
        
        if len(available_products) > 50:
            products_info += f"... and {len(available_products) - 50} more products available.\n"
        
        # Build the full prompt
        prompt = f"""
Please recommend luxury gourmet products for this client with a budget of €{target_budget:.2f}.

{client_profile}
{dietary_info}
{notes_info}
{products_info}

TASK: Select 6-8 products from the available list that would create an exceptional luxury gift composition for this client. Consider their history, preferences, budget, and any restrictions. Ensure the total cost is close to the target budget (within €{target_budget * 0.15:.2f}).

Focus on creating a balanced composition with variety across categories while maintaining the luxury standard expected from Señor Bigott.
"""
        
        return prompt
    
    def _parse_ollama_response(self, ollama_response, available_products, target_budget):
        """Parse Ollama response and validate recommendations"""
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', ollama_response, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Validate required fields
            if 'recommended_products' not in response_data:
                raise ValueError("Missing recommended_products in response")
            
            # Get product objects by code
            recommended_codes = response_data['recommended_products']
            recommended_products = []
            
            # Create mapping of available products by code
            products_by_code = {p.default_code: p for p in available_products if p.default_code}
            
            for code in recommended_codes:
                if code in products_by_code:
                    recommended_products.append(products_by_code[code])
                else:
                    _logger.warning(f"Product code {code} not found in available products")
            
            if not recommended_products:
                raise ValueError("No valid products found in recommendations")
            
            # Calculate total cost
            total_cost = sum(p.list_price for p in recommended_products)
            
            # Validate budget
            budget_variance = abs(total_cost - target_budget) / target_budget
            if budget_variance > 0.3:  # 30% tolerance for parsing
                _logger.warning(f"Budget variance high: {budget_variance:.1%}")
            
            return {
                'products': recommended_products,
                'reasoning': response_data.get('reasoning', 'AI-generated recommendation'),
                'confidence_score': min(1.0, max(0.0, response_data.get('confidence_score', 0.7))),
                'total_cost': total_cost,
                'composition_notes': response_data.get('composition_notes', ''),
                'budget_variance': budget_variance
            }
            
        except Exception as e:
            _logger.error(f"Failed to parse Ollama response: {str(e)}")
            _logger.debug(f"Ollama response was: {ollama_response}")
            
            # Return fallback result
            return self._create_fallback_result(available_products, target_budget)
    
    def _create_fallback_result(self, available_products, target_budget):
        """Create a fallback recommendation when Ollama parsing fails"""
        
        # Simple budget-based selection
        selected_products = []
        remaining_budget = target_budget
        
        # Sort products by category diversity
        categories_used = set()
        
        for product in sorted(available_products, key=lambda p: p.list_price):
            if len(selected_products) >= self.max_products:
                break
            
            if product.list_price <= remaining_budget:
                # Prefer products from unused categories
                if product.lebiggot_category not in categories_used or len(selected_products) < 4:
                    selected_products.append(product)
                    remaining_budget -= product.list_price
                    if product.lebiggot_category:
                        categories_used.add(product.lebiggot_category)
        
        return {
            'products': selected_products,
            'reasoning': 'Fallback recommendation based on budget and category diversity',
            'confidence_score': 0.5,
            'total_cost': sum(p.list_price for p in selected_products),
            'composition_notes': 'Generated using fallback algorithm',
            'budget_variance': abs(sum(p.list_price for p in selected_products) - target_budget) / target_budget
        }
    
    def _fallback_recommendation(self, partner, target_budget, available_products, dietary_restrictions):
        """Fallback recommendation when Ollama is not available"""
        
        fallback_result = self._create_fallback_result(available_products, target_budget)
        
        # Create gift composition
        composition = self._create_gift_composition(
            partner, fallback_result, target_budget, 
            None, dietary_restrictions, "Fallback recommendation (Ollama unavailable)"
        )
        
        return {
            'success': True,
            'composition_id': composition.id,
            'message': f"Generated fallback recommendation for {partner.name} (Ollama unavailable)",
            'products': fallback_result['products'],
            'total_cost': fallback_result['total_cost'],
            'reasoning': fallback_result['reasoning'],
            'confidence_score': fallback_result['confidence_score']
        }
    
    def _create_gift_composition(self, partner, recommendation_result, target_budget, client_notes, dietary_restrictions, ollama_response):
        """Create gift composition record"""
        
        reasoning_html = f"""
        <div class="ollama-recommendation">
            <h4>🤖 Ollama AI Recommendation</h4>
            <p><strong>Reasoning:</strong> {recommendation_result['reasoning']}</p>
            <p><strong>Confidence Score:</strong> {recommendation_result['confidence_score']:.1%}</p>
            <p><strong>Budget Variance:</strong> {recommendation_result.get('budget_variance', 0):.1%}</p>
            {f"<p><strong>Composition Notes:</strong> {recommendation_result['composition_notes']}</p>" if recommendation_result.get('composition_notes') else ""}
            
            <details>
                <summary>View Raw AI Response</summary>
                <pre style="background: #f5f5f5; padding: 10px; margin: 10px 0; white-space: pre-wrap;">{ollama_response}</pre>
            </details>
        </div>
        """
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner.id,
            'target_year': datetime.now().year,
            'target_budget': target_budget,
            'composition_type': 'custom',
            'product_ids': [(6, 0, [p.id for p in recommendation_result['products']])],
            'dietary_restrictions': ','.join(dietary_restrictions or []),
            'reasoning': reasoning_html,
            'confidence_score': recommendation_result['confidence_score'],
            'novelty_score': 0.8,  # Default for Ollama recommendations
            'actual_cost': recommendation_result['total_cost'],
            'state': 'draft'
        })
        
        return composition

    def action_view_recommendations(self):
        """View all gift compositions created by this recommender"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'Gift Recommendations',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('reasoning', 'ilike', 'Ollama AI Recommendation')],
            'context': {'search_default_partner_id': True}
        }

    def action_view_successful_recommendations(self):
        """View successful recommendations only"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'Successful Recommendations',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [
                ('reasoning', 'ilike', 'Ollama AI Recommendation'),
                ('state', 'in', ['approved', 'delivered'])
            ],
            'context': {'search_default_partner_id': True}
        }