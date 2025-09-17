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
    _description = 'Ollama-Powered Gift Recommendation Engine with Advanced Learning'
    _rec_name = 'name'
    
    # ================== FIELD DEFINITIONS ==================
    
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
    
    # Learning Cache
    learning_cache = fields.Text(string="Learning Cache JSON")
    cache_expiry = fields.Datetime(string="Cache Expiry")
    
    # Computed Fields
    success_rate = fields.Float(string="Success Rate (%)", compute='_compute_success_rate', store=True)
    
    # ================== COMPUTED METHODS ==================
    
    @api.depends('total_recommendations', 'successful_recommendations')
    def _compute_success_rate(self):
        for record in self:
            if record.total_recommendations > 0:
                record.success_rate = (record.successful_recommendations / record.total_recommendations) * 100
            else:
                record.success_rate = 0.0

    # ================== INITIALIZATION METHODS ==================
    
    @api.model
    def get_or_create_recommender(self):
        """Get existing recommender or create new one"""
        recommender = self.search([('active', '=', True)], limit=1)
        if not recommender:
            recommender = self.create({
                'name': 'Default Ollama Gift Recommender',
                'ollama_enabled': False
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
    
    # ================== MAIN RECOMMENDATION METHOD ==================
    
    def generate_gift_recommendations(self, partner_id, target_budget, 
                                    client_notes='', dietary_restrictions=None,
                                    composition_type=None):
        """Main generation method with intelligent multi-source merging"""
        
        partner = self.env['res.partner'].browse(partner_id)
        if not partner:
            return {'success': False, 'error': 'Partner not found'}
        
        # 1. COLLECT DATA FROM ALL SOURCES
        _logger.info("="*60)
        _logger.info(f"STARTING GENERATION FOR: {partner.name}")
        _logger.info("="*60)
        
        # Form data
        form_data = {
            'budget': target_budget if target_budget and target_budget > 0 else None,
            'dietary': dietary_restrictions or [],
            'composition_type': composition_type
        }
        _logger.info(f"📋 FORM DATA: Budget={form_data['budget']}, Dietary={form_data['dietary']}, Type={form_data['composition_type']}")
        
        # Parse notes
        notes_data = self._parse_notes_with_ollama(client_notes, form_data) if client_notes else {}
        _logger.info(f"📝 NOTES DATA: {notes_data}")
        
        # Get historical patterns
        learning_data = self._get_or_update_learning_cache(partner_id)
        patterns = learning_data.get('patterns', {})
        seasonal = learning_data.get('seasonal', {})
        _logger.info(f"📊 HISTORY DATA: {patterns.get('total_orders', 0)} orders, Avg €{patterns.get('avg_order_value', 0):.2f}")
        
        # Get previous sales
        previous_sales = self._get_all_previous_sales_data(partner_id)
        
        # 2. MERGE REQUIREMENTS
        final_requirements = self._merge_all_requirements(
            notes_data, form_data, patterns, seasonal
        )
        
        # 3. LOG MERGED REQUIREMENTS
        self._log_final_requirements(final_requirements)
        
        # 4. UPDATE TRACKING
        self.total_recommendations += 1
        self.last_recommendation_date = fields.Datetime.now()
        
        # 5. CREATE GENERATION CONTEXT
        generation_context = {
            'patterns': patterns,
            'seasonal': seasonal,
            'similar_clients': learning_data.get('similar_clients'),
            'previous_sales': previous_sales,
            'requirements_merged': True
        }
        
        # 6. DETERMINE STRATEGY
        strategy = self._determine_generation_strategy(
            previous_sales, patterns, final_requirements, client_notes
        )
        _logger.info(f"🎯 GENERATION STRATEGY: {strategy}")
        
        # 7. EXECUTE GENERATION
        result = None
        if strategy == '8020_rule':
            result = self._generate_with_8020_rule(
                partner, final_requirements, client_notes, generation_context
            )
        elif strategy == 'similar_clients':
            result = self._generate_from_similar_clients(
                partner, final_requirements, client_notes, generation_context
            )
        elif strategy == 'pattern_based':
            result = self._generate_from_patterns_enhanced(
                partner, final_requirements, client_notes, generation_context
            )
        else:
            result = self._generate_with_universal_enforcement(
                partner, final_requirements, client_notes, generation_context
            )
        
        # 8. VALIDATE RESULT
        if result and result.get('success'):
            self.successful_recommendations += 1
            self._validate_and_log_result(result, final_requirements)
        
        return result
    
    # ================== REQUIREMENT MERGING METHODS ==================
    
    def _merge_all_requirements(self, notes_data, form_data, patterns, seasonal):
        """Intelligently merge requirements from all sources"""
        
        merged = {
            'budget': None,
            'budget_source': 'none',
            'budget_flexibility': 10,
            'product_count': None,
            'count_source': 'none',
            'enforce_count': False,
            'dietary': [],
            'dietary_source': 'none',
            'composition_type': 'custom',
            'type_source': 'default',
            'categories_required': {},
            'categories_excluded': [],
            'specific_products': [],
            'special_instructions': [],
        }
        
        # MERGE BUDGET
        if notes_data.get('budget_override'):
            merged['budget'] = notes_data['budget_override']
            merged['budget_source'] = 'notes'
        elif form_data.get('budget'):
            merged['budget'] = form_data['budget']
            merged['budget_source'] = 'form'
        elif patterns and patterns.get('avg_order_value'):
            historical_budget = patterns['avg_order_value']
            if patterns.get('budget_trend') == 'increasing':
                historical_budget *= 1.1
            elif patterns.get('budget_trend') == 'decreasing':
                historical_budget *= 0.95
            merged['budget'] = historical_budget
            merged['budget_source'] = f"history ({patterns.get('budget_trend', 'stable')} trend)"
        else:
            merged['budget'] = 1000.0
            merged['budget_source'] = 'default'
        
        # MERGE PRODUCT COUNT
        if notes_data.get('product_count'):
            merged['product_count'] = notes_data['product_count']
            merged['enforce_count'] = True
            merged['count_source'] = 'notes (strict)'
        elif patterns and patterns.get('avg_product_count'):
            merged['product_count'] = int(round(patterns['avg_product_count']))
            merged['enforce_count'] = False
            merged['count_source'] = 'history (flexible)'
        else:
            avg_price = 80
            if patterns and patterns.get('preferred_price_range'):
                avg_price = patterns['preferred_price_range'].get('avg', 80)
            merged['product_count'] = max(8, min(20, int(merged['budget'] / avg_price)))
            merged['enforce_count'] = False
            merged['count_source'] = 'estimated'
        
        # MERGE DIETARY
        dietary_set = set()
        if notes_data.get('dietary'):
            dietary_set.update(notes_data['dietary'])
            merged['dietary_source'] = 'notes'
        if form_data.get('dietary'):
            dietary_set.update(form_data['dietary'])
            if merged['dietary_source'] == 'notes':
                merged['dietary_source'] = 'notes+form'
            else:
                merged['dietary_source'] = 'form'
        merged['dietary'] = list(dietary_set)
        
        # MERGE COMPOSITION TYPE
        if notes_data.get('composition_type'):
            merged['composition_type'] = notes_data['composition_type']
            merged['type_source'] = 'notes'
        elif form_data.get('composition_type'):
            merged['composition_type'] = form_data['composition_type']
            merged['type_source'] = 'form'
        else:
            merged['composition_type'] = 'custom'
            merged['type_source'] = 'default'
        
        # MERGE OTHER REQUIREMENTS
        if notes_data.get('budget_flexibility'):
            merged['budget_flexibility'] = notes_data['budget_flexibility']
        if notes_data.get('categories_required'):
            merged['categories_required'] = notes_data['categories_required']
        if notes_data.get('categories_excluded'):
            merged['categories_excluded'] = notes_data['categories_excluded']
        if notes_data.get('specific_products'):
            merged['specific_products'] = notes_data['specific_products']
        if notes_data.get('special_instructions'):
            merged['special_instructions'] = notes_data['special_instructions']
        
        return merged
    
    def _log_final_requirements(self, requirements):
        """Log the final merged requirements"""
        _logger.info("="*60)
        _logger.info("FINAL MERGED REQUIREMENTS:")
        _logger.info(f"  💰 Budget: €{requirements['budget']:.2f} (source: {requirements['budget_source']})")
        _logger.info(f"  📦 Products: {requirements['product_count']} (source: {requirements['count_source']}, strict={requirements['enforce_count']})")
        _logger.info(f"  🥗 Dietary: {requirements['dietary']} (source: {requirements['dietary_source']})")
        _logger.info(f"  🎁 Type: {requirements['composition_type']} (source: {requirements['type_source']})")
        _logger.info(f"  📐 Flexibility: {requirements['budget_flexibility']}%")
        _logger.info("="*60)
    
    # ================== STRATEGY DETERMINATION ==================
    
    def _determine_generation_strategy(self, previous_sales, patterns, requirements, notes):
        """Determine the best generation strategy"""
        notes_lower = notes.lower() if notes else ""
        
        if previous_sales and len(previous_sales) > 0:
            if 'all new' not in notes_lower and 'completely different' not in notes_lower:
                return '8020_rule'
        
        if patterns and patterns.get('total_orders', 0) >= 3:
            return 'pattern_based'
        
        if not previous_sales:
            return 'similar_clients'
        
        return 'universal'
    
    # ================== GENERATION METHODS ==================
    
    def _generate_with_8020_rule(self, partner, requirements, notes, context):
        """Generate with 80/20 rule"""
        previous_sales = context.get('previous_sales', [])
        if not previous_sales:
            return self._generate_with_universal_enforcement(partner, requirements, notes, context)
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        
        keep_count = int(product_count * 0.8)
        new_count = product_count - keep_count
        
        _logger.info(f"🔄 80/20 Split: Keep {keep_count}, add {new_count}")
        
        # Score previous products
        product_scores = {}
        for sale_data in previous_sales:
            for prod_data in sale_data['products']:
                product = prod_data['product']
                if product.id not in product_scores:
                    recency_score = 1.0 / (len(previous_sales) - previous_sales.index(sale_data) + 1)
                    frequency_score = prod_data['times_ordered']
                    product_scores[product.id] = {
                        'product': product,
                        'score': frequency_score * 2 + recency_score,
                        'frequency': prod_data['times_ordered']
                    }
        
        # Select products to keep
        scored_products = sorted(product_scores.values(), key=lambda x: x['score'], reverse=True)
        products_to_keep = []
        keep_budget = budget * 0.8
        current_keep_cost = 0
        
        for item in scored_products:
            product = item['product']
            if not self._has_stock(product) or not self._check_dietary_compliance(product, dietary):
                continue
            
            if current_keep_cost + product.list_price <= keep_budget and len(products_to_keep) < keep_count:
                products_to_keep.append(product)
                current_keep_cost += product.list_price
                
                if len(products_to_keep) >= keep_count:
                    break
        
        # Find new products
        new_budget = budget - current_keep_cost
        exclude_ids = [p.id for p in products_to_keep]
        new_products = self._find_complementary_products(
            new_count, new_budget, dietary, exclude_ids, context
        )
        
        # Combine and create composition
        final_products = products_to_keep + new_products
        total_cost = sum(p.list_price for p in final_products)
        
        try:
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': budget,
                'target_year': fields.Date.today().year,
                'product_ids': [(6, 0, [p.id for p in final_products])],
                'dietary_restrictions': ', '.join(dietary) if dietary else '',
                'client_notes': notes,
                'generation_method': 'ollama',
                'composition_type': requirements.get('composition_type', 'custom'),
                'confidence_score': 0.92,
                'ai_reasoning': f"80/20 Rule: Kept {len(products_to_keep)}/{keep_count}, Added {len(new_products)}/{new_count}"
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': final_products,
                'total_cost': total_cost,
                'product_count': len(final_products),
                'confidence_score': 0.92,
                'message': f'80/20 rule: {len(products_to_keep)} kept + {len(new_products)} new',
                'method': '8020_rule'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_with_universal_enforcement(self, partner, requirements, notes, context):
        """Universal generation with enforcement"""
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        flexibility = requirements['budget_flexibility']
        
        min_budget = budget * (1 - flexibility/100)
        max_budget = budget * (1 + flexibility/100)
        
        # Get product pool
        products = self._get_smart_product_pool_enhanced(
            budget, dietary, requirements.get('composition_type', 'custom'), context
        )
        
        if not products:
            return {'success': False, 'error': 'No products available'}
        
        # Select products
        selected = self._smart_optimize_selection(
            products, product_count, budget, flexibility,
            requirements.get('enforce_count', False), context
        )
        
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
                'confidence_score': 0.85,
                'ai_reasoning': f"Generated {len(selected)} products, €{total_cost:.2f}"
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.85,
                'message': f'Generated {len(selected)} products, €{total_cost:.2f}',
                'method': 'universal'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_from_similar_clients(self, partner, requirements, notes, context):
        """Generate based on similar clients"""
        similar_clients = context.get('similar_clients', [])
        if not similar_clients:
            return self._generate_with_universal_enforcement(partner, requirements, notes, context)
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        
        # Aggregate products from similar clients
        product_popularity = {}
        for similar in similar_clients:
            similarity_weight = similar['similarity']
            patterns = similar['patterns']
            
            for prod_id in patterns.get('favorite_products', []):
                if prod_id not in product_popularity:
                    product_popularity[prod_id] = 0
                product_popularity[prod_id] += similarity_weight
        
        # Get popular products
        popular_product_ids = sorted(
            product_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:product_count * 2]
        
        # Select products
        products = []
        for prod_id, score in popular_product_ids:
            product = self.env['product.template'].browse(prod_id)
            if product.exists() and self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                products.append(product)
        
        # Optimize selection
        selected = products[:product_count] if products else []
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
                'confidence_score': 0.85,
                'ai_reasoning': f"Based on {len(similar_clients)} similar clients"
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.85,
                'message': f'Similar clients: {len(selected)} products',
                'method': 'similar_clients'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_from_patterns_enhanced(self, partner, requirements, notes, context):
        """Generate from patterns"""
        patterns = context.get('patterns', {})
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        
        # Get products
        products = self._get_smart_product_pool_enhanced(
            budget, dietary, requirements.get('composition_type', 'custom'), context
        )
        
        if not products:
            return {'success': False, 'error': 'No products available'}
        
        # Score products
        scored_products = []
        for product in products:
            score = 1.0
            if patterns:
                if product.id in patterns.get('favorite_products', []):
                    score += 5.0
                if product.id in patterns.get('never_repeated_products', []):
                    score -= 2.0
            scored_products.append((product, score))
        
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Select products
        selected = []
        for product, score in scored_products[:product_count]:
            selected.append(product)
        
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
                'confidence_score': 0.88,
                'ai_reasoning': f"Pattern-based: {patterns.get('total_orders', 0)} historical orders"
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.88,
                'message': f'Pattern-based: {len(selected)} products',
                'method': 'pattern_based'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    # ================== PRODUCT SELECTION HELPERS ==================
    
    def _get_smart_product_pool_enhanced(self, budget, dietary, composition_type, context):
        """Get intelligent product pool"""
        patterns = context.get('patterns', {})
        
        # Price range based on type
        if composition_type == 'hybrid':
            min_price = max(15, budget * 0.03)
            max_price = budget * 0.4
        elif patterns and patterns.get('preferred_price_range'):
            min_price = patterns['preferred_price_range'].get('min', budget * 0.01)
            max_price = patterns['preferred_price_range'].get('max', budget * 0.4)
        else:
            min_price = max(5, budget * 0.01)
            max_price = budget * 0.4
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
        ]
        
        # Apply dietary filters
        if dietary:
            if 'halal' in dietary:
                if 'is_halal_compatible' in self.env['product.template']._fields:
                    domain.append(('is_halal_compatible', '!=', False))
                if 'contains_pork' in self.env['product.template']._fields:
                    domain.append(('contains_pork', '=', False))
                if 'contains_alcohol' in self.env['product.template']._fields:
                    domain.append(('contains_alcohol', '=', False))
        
        products = self.env['product.template'].sudo().search(domain, limit=1000)
        
        # Filter by stock
        available = [p for p in products if self._has_stock(p)]
        
        _logger.info(f"📦 Product pool: {len(available)} products (€{min_price:.2f}-€{max_price:.2f})")
        return available
    
    def _smart_optimize_selection(self, products, target_count, budget, flexibility, enforce_strict, context):
        """Smart product selection"""
        if not target_count:
            target_count = max(5, int(budget / 50))
        
        min_budget = budget * (1 - flexibility/100)
        max_budget = budget * (1 + flexibility/100)
        avg_price_target = budget / target_count
        
        # Score products
        scored_products = []
        for product in products:
            price_diff = abs(product.list_price - avg_price_target)
            price_score = 1 / (1 + price_diff/avg_price_target)
            scored_products.append((product, price_score))
        
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Select products
        selected = []
        current_total = 0
        
        for product, score in scored_products:
            if len(selected) >= target_count:
                break
            
            future_total = current_total + product.list_price
            remaining_slots = target_count - len(selected) - 1
            
            if remaining_slots > 0:
                min_remaining = remaining_slots * 5
                if future_total + min_remaining <= max_budget:
                    selected.append(product)
                    current_total = future_total
            else:
                if min_budget <= future_total <= max_budget:
                    selected.append(product)
                    current_total = future_total
        
        # Enforce count if required
        if enforce_strict and len(selected) != target_count:
            if len(selected) < target_count:
                remaining = [p for p, _ in scored_products if p not in selected]
                selected.extend(remaining[:target_count - len(selected)])
            else:
                selected = selected[:target_count]
        
        return selected
    
    def _find_complementary_products(self, count, budget, dietary, exclude_ids, context):
        """Find complementary new products"""
        avg_price = budget / count if count > 0 else 50
        min_price = max(5, avg_price * 0.3)
        max_price = avg_price * 2
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
            ('id', 'not in', exclude_ids)
        ]
        
        # Apply dietary
        if dietary and 'halal' in dietary:
            if 'is_halal_compatible' in self.env['product.template']._fields:
                domain.append(('is_halal_compatible', '!=', False))
        
        products = self.env['product.template'].sudo().search(domain, limit=500)
        available = [p for p in products if self._has_stock(p)]
        
        # Sort by price fitness
        available.sort(key=lambda p: abs(p.list_price - avg_price))
        
        return available[:count]
    
    # ================== LEARNING & ANALYSIS METHODS ==================
    
    def _get_or_update_learning_cache(self, partner_id):
        """Get or update learning cache"""
        if self.cache_expiry and self.cache_expiry > datetime.now():
            if self.learning_cache:
                try:
                    cache = json.loads(self.learning_cache)
                    if str(partner_id) in cache:
                        return cache[str(partner_id)]
                except:
                    pass
        
        # Generate fresh analysis
        learning_data = {
            'patterns': self._analyze_client_purchase_patterns(partner_id),
            'seasonal': self._analyze_seasonal_preferences(partner_id),
            'similar_clients': self._find_similar_clients(partner_id),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update cache
        try:
            cache = json.loads(self.learning_cache) if self.learning_cache else {}
        except:
            cache = {}
        
        cache[str(partner_id)] = learning_data
        self.learning_cache = json.dumps(cache)
        self.cache_expiry = datetime.now() + timedelta(hours=24)
        
        return learning_data
    
    def _analyze_client_purchase_patterns(self, partner_id):
        """Analyze client purchase patterns"""
        all_orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc')
        
        if not all_orders:
            return None
        
        pattern_analysis = {
            'total_orders': len(all_orders),
            'avg_order_value': 0,
            'preferred_categories': {},
            'product_frequency': {},
            'budget_trend': 'stable',
            'avg_product_count': 0,
            'favorite_products': [],
            'never_repeated_products': [],
            'preferred_price_range': {'min': 0, 'max': 0, 'avg': 0}
        }
        
        total_value = 0
        total_products = 0
        product_counter = Counter()
        category_counter = Counter()
        all_product_prices = []
        
        for order in all_orders:
            total_value += order.amount_untaxed
            
            for line in order.order_line:
                if line.product_id:
                    product_tmpl = line.product_id.product_tmpl_id
                    product_counter[product_tmpl.id] += 1
                    all_product_prices.append(line.price_unit)
                    
                    if hasattr(product_tmpl, 'categ_id'):
                        category_counter[product_tmpl.categ_id.name] += 1
                    total_products += 1
        
        # Calculate insights
        pattern_analysis['avg_order_value'] = total_value / len(all_orders) if all_orders else 0
        pattern_analysis['avg_product_count'] = total_products / len(all_orders) if all_orders else 0
        pattern_analysis['favorite_products'] = [
            prod_id for prod_id, count in product_counter.items() if count >= 2
        ]
        pattern_analysis['never_repeated_products'] = [
            prod_id for prod_id, count in product_counter.items() 
            if count == 1 and len(all_orders) > 2
        ]
        pattern_analysis['preferred_categories'] = dict(category_counter.most_common(5))
        
        if all_product_prices:
            pattern_analysis['preferred_price_range'] = {
                'min': min(all_product_prices),
                'max': max(all_product_prices),
                'avg': sum(all_product_prices) / len(all_product_prices)
            }
        
        # Budget trend
        if len(all_orders) >= 3:
            recent_avg = sum(o.amount_untaxed for o in all_orders[:3]) / 3
            older_avg = sum(o.amount_untaxed for o in all_orders[-3:]) / 3
            if recent_avg > older_avg * 1.2:
                pattern_analysis['budget_trend'] = 'increasing'
            elif recent_avg < older_avg * 0.8:
                pattern_analysis['budget_trend'] = 'decreasing'
        
        return pattern_analysis
    
    def _analyze_seasonal_preferences(self, partner_id):
        """Analyze seasonal preferences"""
        orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ])
        
        if not orders:
            return None
        
        seasonal_data = {
            'spring': {'products': [], 'categories': []},
            'summer': {'products': [], 'categories': []},
            'autumn': {'products': [], 'categories': []},
            'winter': {'products': [], 'categories': []},
            'christmas': {'products': [], 'categories': []}
        }
        
        for order in orders:
            month = order.date_order.month
            season = self._get_current_season(month)
            
            for line in order.order_line:
                if line.product_id:
                    seasonal_data[season]['products'].append(line.product_id.product_tmpl_id.id)
                    if hasattr(line.product_id.product_tmpl_id, 'categ_id'):
                        seasonal_data[season]['categories'].append(line.product_id.product_tmpl_id.categ_id.name)
        
        current_season = self._get_current_season(datetime.now().month)
        
        return {
            'current_season': current_season,
            'seasonal_data': seasonal_data,
            'seasonal_favorites': seasonal_data[current_season]['products'] if current_season in seasonal_data else []
        }
    
    def _get_current_season(self, month):
        """Get season from month"""
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        elif month == 12:
            return 'christmas'
        else:
            return 'winter'
    
    def _find_similar_clients(self, partner_id, limit=5):
        """Find similar clients"""
        target_patterns = self._analyze_client_purchase_patterns(partner_id)
        if not target_patterns:
            return []
        
        all_clients = self.env['sale.order'].read_group(
            [('state', 'in', ['sale', 'done'])],
            ['partner_id'],
            ['partner_id']
        )
        
        similar_clients = []
        
        for client_data in all_clients[:20]:
            other_partner_id = client_data['partner_id'][0]
            if other_partner_id == partner_id:
                continue
            
            other_patterns = self._analyze_client_purchase_patterns(other_partner_id)
            if not other_patterns:
                continue
            
            similarity_score = self._calculate_similarity(target_patterns, other_patterns)
            
            if similarity_score > 0.6:
                similar_clients.append({
                    'partner_id': other_partner_id,
                    'similarity': similarity_score,
                    'patterns': other_patterns
                })
        
        similar_clients.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_clients[:limit]
    
    def _calculate_similarity(self, patterns1, patterns2):
        """Calculate similarity between patterns"""
        score = 0
        factors = 0
        
        # Budget similarity
        avg1 = patterns1.get('avg_order_value', 0)
        avg2 = patterns2.get('avg_order_value', 0)
        if avg1 and avg2:
            budget_diff = abs(avg1 - avg2) / max(avg1, avg2)
            if budget_diff < 0.3:
                score += (1 - budget_diff)
            factors += 1
        
        # Product count similarity
        count1 = patterns1.get('avg_product_count', 0)
        count2 = patterns2.get('avg_product_count', 0)
        if count1 and count2:
            count_diff = abs(count1 - count2) / max(count1, count2)
            if count_diff < 0.3:
                score += (1 - count_diff)
            factors += 1
        
        # Category overlap
        cats1 = set(patterns1.get('preferred_categories', {}).keys())
        cats2 = set(patterns2.get('preferred_categories', {}).keys())
        if cats1 and cats2:
            overlap = len(cats1.intersection(cats2)) / len(cats1.union(cats2))
            score += overlap
            factors += 1
        
        return score / factors if factors > 0 else 0
    
    def _get_all_previous_sales_data(self, partner_id):
        """Get all previous sales data"""
        sales = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc')
        
        all_sales_data = []
        
        for sale in sales:
            products = []
            for line in sale.order_line:
                if line.product_id and line.product_id.type == 'product':
                    product_tmpl = line.product_id.product_tmpl_id
                    products.append({
                        'product': product_tmpl,
                        'qty': line.product_uom_qty,
                        'price': line.price_unit,
                        'times_ordered': 0
                    })
            
            if products:
                all_sales_data.append({
                    'order': sale,
                    'order_date': sale.date_order,
                    'products': products,
                    'total': sale.amount_untaxed
                })
        
        # Calculate frequency
        product_frequency = {}
        for sale_data in all_sales_data:
            for prod_data in sale_data['products']:
                prod_id = prod_data['product'].id
                product_frequency[prod_id] = product_frequency.get(prod_id, 0) + 1
        
        for sale_data in all_sales_data:
            for prod_data in sale_data['products']:
                prod_data['times_ordered'] = product_frequency.get(prod_data['product'].id, 0)
        
        return all_sales_data
    
    def _get_previous_order_data(self, partner_id):
        """Get last order data for compatibility"""
        sales = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc', limit=1)
        
        if not sales:
            return None
        
        last_order = sales[0]
        previous_products = []
        total_amount = 0.0
        
        for line in last_order.order_line:
            if line.product_id and line.product_id.type == 'product':
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
    
    # ================== NOTES PARSING ==================
    
    def _parse_notes_with_ollama(self, notes, form_data=None):
        """Parse notes using Ollama"""
        if not notes:
            return {'use_default': False}
        
        if not self.ollama_enabled:
            return self._parse_notes_basic_fallback(notes)
        
        prompt = f"""Extract requirements from these notes: "{notes}"

Return ONLY valid JSON:
{{
    "product_count": <number or null>,
    "budget": <number or null>,
    "budget_flexibility": <5-20>,
    "dietary_restrictions": [],
    "composition_type": "custom|hybrid|experience|null",
    "categories_required": {{}},
    "specific_products": [],
    "confidence": <0-100>
}}"""
        
        try:
            response = self._call_ollama(prompt, format_json=True)
            if response:
                try:
                    extracted = json.loads(response)
                    requirements = {
                        'use_default': False,
                        'product_count': extracted.get('product_count'),
                        'budget_override': extracted.get('budget'),
                        'budget_flexibility': extracted.get('budget_flexibility', 10),
                        'dietary': extracted.get('dietary_restrictions', []),
                        'composition_type': extracted.get('composition_type'),
                        'categories_required': extracted.get('categories_required', {}),
                        'specific_products': extracted.get('specific_products', [])
                    }
                    return requirements
                except json.JSONDecodeError:
                    return self._parse_notes_basic_fallback(notes)
            else:
                return self._parse_notes_basic_fallback(notes)
        except Exception as e:
            _logger.error(f"Ollama parsing error: {e}")
            return self._parse_notes_basic_fallback(notes)
    
    def _parse_notes_basic_fallback(self, notes):
        """Basic fallback parser"""
        parsed = {
            'use_default': False,
            'product_count': None,
            'budget_override': None,
            'budget_flexibility': 10,
            'dietary': [],
            'composition_type': None
        }
        
        notes_lower = notes.lower()
        
        # Extract numbers
        numbers = re.findall(r'\b(\d+)\b', notes)
        for num in numbers:
            num_int = int(num)
            if 1 <= num_int <= 100:
                if 'product' in notes_lower or 'item' in notes_lower:
                    parsed['product_count'] = num_int
            elif 100 <= num_int <= 10000:
                if 'budget' in notes_lower or '€' in notes or '$' in notes:
                    parsed['budget_override'] = float(num_int)
        
        # Dietary
        if 'halal' in notes_lower:
            parsed['dietary'].append('halal')
        if 'vegan' in notes_lower:
            parsed['dietary'].append('vegan')
        
        # Type
        if 'hybrid' in notes_lower:
            parsed['composition_type'] = 'hybrid'
        elif 'experience' in notes_lower:
            parsed['composition_type'] = 'experience'
        
        return parsed
    
    def _call_ollama(self, prompt, format_json=False):
        """Call Ollama API"""
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
                return None
                
        except Exception as e:
            _logger.error(f"Ollama request failed: {str(e)}")
            return None
    
    # ================== UTILITY METHODS ==================
    
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
        """Check dietary compliance"""
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
    
    def _validate_and_log_result(self, result, requirements):
        """Validate and log result"""
        actual_count = result.get('product_count', 0)
        actual_cost = result.get('total_cost', 0)
        expected_count = requirements.get('product_count')
        expected_budget = requirements['budget']
        flexibility = requirements['budget_flexibility']
        
        if requirements.get('enforce_count') and expected_count:
            if actual_count == expected_count:
                _logger.info(f"✅ Count MET: {actual_count}")
            else:
                _logger.error(f"❌ Count FAILED: {actual_count} != {expected_count}")
        
        min_budget = expected_budget * (1 - flexibility/100)
        max_budget = expected_budget * (1 + flexibility/100)
        
        if min_budget <= actual_cost <= max_budget:
            _logger.info(f"✅ Budget MET: €{actual_cost:.2f}")
        else:
            _logger.error(f"❌ Budget FAILED: €{actual_cost:.2f}")
        
        return result.get('compliant', False)
    
    # ================== ACTION METHODS ==================
    
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
    
    def trigger_learning(self):
        """Trigger learning analysis"""
        self.ensure_one()
        
        # Force cache refresh
        self.cache_expiry = datetime.now() - timedelta(days=1)
        
        # Analyze clients
        all_clients = self.env['sale.order'].read_group(
            [('state', 'in', ['sale', 'done'])],
            ['partner_id'],
            ['partner_id']
        )
        
        analyzed_count = 0
        for client_data in all_clients[:50]:
            partner_id = client_data['partner_id'][0]
            self._get_or_update_learning_cache(partner_id)
            analyzed_count += 1
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': '🧠 Learning Analysis Complete',
                'message': f'Analyzed {analyzed_count} clients',
                'type': 'success',
                'sticky': False,
            }
        }