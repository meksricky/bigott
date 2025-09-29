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
        
        # FIX: Check if patterns is None or empty before accessing
        if patterns:
            _logger.info(f"📊 HISTORY DATA: {patterns.get('total_orders', 0)} orders, Avg €{patterns.get('avg_order_value', 0):.2f}")
        else:
            _logger.info("📊 HISTORY DATA: No patterns available - using defaults")
            patterns = self._get_default_patterns()
        
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
    
    # ================== DEFAULT PATTERNS METHOD ==================
    
    def _get_default_patterns(self):
        """Return default pattern structure when no patterns are available"""
        return {
            'total_orders': 0,
            'avg_order_value': 0,
            'preferred_categories': {},
            'seasonal_patterns': {},
            'product_frequency': {},
            'budget_trend': 'stable',
            'avg_product_count': 0,
            'favorite_products': [],
            'never_repeated_products': [],
            'category_evolution': {},
            'order_intervals': [],
            'preferred_price_range': {'min': 0, 'max': 0, 'avg': 0}
        }
    
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
            # FIX: Ensure avg_price is never 0 to avoid division by zero
            avg_price = 80.0  # Default price
            if patterns and patterns.get('preferred_price_range'):
                price_range = patterns['preferred_price_range']
                # Check if avg exists and is greater than 0
                if price_range.get('avg') and price_range['avg'] > 0:
                    avg_price = float(price_range['avg'])
                # Fallback to calculating from min/max if avg is 0
                elif price_range.get('min') and price_range.get('max'):
                    min_p = float(price_range.get('min', 50))
                    max_p = float(price_range.get('max', 150))
                    if min_p > 0 and max_p > 0:
                        avg_price = (min_p + max_p) / 2
            
            # Ensure avg_price is never less than a minimum threshold
            avg_price = max(10.0, avg_price)
            
            # Safe division with fallback
            if avg_price > 0:
                calculated_count = int(merged['budget'] / avg_price)
            else:
                calculated_count = 12  # Default fallback count
            
            merged['product_count'] = max(8, min(20, calculated_count))
            merged['enforce_count'] = False
            merged['count_source'] = f'estimated (€{merged["budget"]:.0f}/€{avg_price:.0f})'
        
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
        """Generate with 80/20 rule - keep 80% from history, change 20%"""
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
        
        _logger.info(f"✅ Selected {len(products_to_keep)} products from history (target: {keep_count})")
        
        # Find new products
        new_budget = budget - current_keep_cost
        exclude_ids = [p.id for p in products_to_keep]
        new_products = self._find_complementary_products(
            new_count, new_budget, dietary, exclude_ids, context
        )
        
        _logger.info(f"✅ Added {len(new_products)} new products")
        
        # Combine and create composition
        final_products = products_to_keep + new_products
        total_cost = sum(p.list_price for p in final_products)
        
        try:
            reasoning = f"""80/20 Rule Applied:
- Kept {len(products_to_keep)} products from purchase history ({keep_count} target)
- Added {len(new_products)} new products for variety ({new_count} target)
- Total: {len(final_products)} products = €{total_cost:.2f}
- Budget target: €{budget:.2f} (variance: {((total_cost-budget)/budget)*100:+.1f}%)
- Based on {len(previous_sales)} previous orders"""
            
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
                'ai_reasoning': reasoning
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
        """Universal generation with strict enforcement of ALL requirements"""
        
        budget = requirements['budget']
        flexibility = requirements['budget_flexibility']
        product_count = requirements['product_count']
        enforce_count = requirements['enforce_count']
        dietary = requirements['dietary']
        
        # Calculate budget bounds
        min_budget = budget * (1 - flexibility/100)
        max_budget = budget * (1 + flexibility/100)
        
        _logger.info(f"🎯 Generating: {'EXACTLY' if enforce_count else 'APPROXIMATELY'} {product_count} products, €{min_budget:.2f}-€{max_budget:.2f}")
        
        # Get product pool
        products = self._get_smart_product_pool(budget, dietary, context)
        
        if not products:
            return {'success': False, 'error': 'No products available matching criteria'}
        
        # Apply category requirements if any
        if requirements.get('categories_required'):
            selected = self._select_with_category_requirements(
                products, requirements['categories_required'], 
                product_count, budget
            )
        else:
            # Use smart optimization
            selected = self._smart_optimize_selection(
                products, product_count, budget, flexibility,
                enforce_count, context
            )
        
        # STRICT ENFORCEMENT
        if enforce_count and product_count:
            selected = self._enforce_exact_count(selected, products, product_count, budget)
        
        # Calculate total
        total_cost = sum(p.list_price for p in selected)
        
        # Check compliance
        count_ok = (not enforce_count) or (len(selected) == product_count)
        budget_ok = min_budget <= total_cost <= max_budget
        
        if not count_ok:
            _logger.error(f"❌ Count violation: {len(selected)} != {product_count}")
        if not budget_ok:
            _logger.warning(f"⚠️ Budget variance: €{total_cost:.2f} not in €{min_budget:.2f}-€{max_budget:.2f}")
        
        # Create composition
        try:
            reasoning = self._build_comprehensive_reasoning(
                requirements, selected, total_cost, budget, context
            )
            
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': budget,
                'target_year': fields.Date.today().year,
                'product_ids': [(6, 0, [p.id for p in selected])],
                'dietary_restrictions': ', '.join(dietary) if dietary else '',
                'client_notes': notes,
                'generation_method': 'ollama',
                'composition_type': requirements.get('composition_type', 'custom'),
                'confidence_score': 0.95 if (count_ok and budget_ok) else 0.7,
                'ai_reasoning': reasoning
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.95 if (count_ok and budget_ok) else 0.7,
                'message': f"{'✅' if (count_ok and budget_ok) else '⚠️'} Generated {len(selected)} products, €{total_cost:.2f}",
                'method': 'universal_enforcement',
                'compliant': count_ok and budget_ok
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_from_similar_clients(self, partner, requirements, notes, context):
        """Generate based on similar clients when no direct history exists"""
        similar_clients = context.get('similar_clients', [])
        if not similar_clients:
            return self._generate_with_universal_enforcement(partner, requirements, notes, context)
        
        _logger.info(f"👥 Learning from {len(similar_clients)} similar clients")
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        
        # Aggregate popular products from similar clients
        product_popularity = {}
        for similar in similar_clients:
            similarity_weight = similar['similarity']
            patterns = similar['patterns']
            
            # FIX: Check patterns before accessing
            if patterns and patterns.get('favorite_products'):
                for prod_id in patterns['favorite_products']:
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
        if len(products) < product_count:
            # Need more products - get from general pool
            additional = self._get_available_products(budget, dietary)
            products.extend(additional[:product_count - len(products)])
        
        selected = products[:product_count] if products else []
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        try:
            top_similar = similar_clients[0] if similar_clients else None
            reasoning = f"""Similar Client Pattern Generation:
- Based on {len(similar_clients)} similar clients
- Top match: {top_similar['similarity']*100:.0f}% similarity
- Popular products from similar clients: {len(product_popularity)} identified
- Selected {len(selected)} products = €{total_cost:.2f}
- Budget target: €{budget:.2f} (variance: {((total_cost-budget)/budget)*100:+.1f}%)"""
            
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
                'ai_reasoning': reasoning
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.85,
                'message': f'Similar clients: {len(selected)} products = €{total_cost:.2f}',
                'method': 'similar_clients'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_from_patterns_enhanced(self, partner, requirements, notes, context):
        """Enhanced pattern-based generation using merged requirements"""
        
        patterns = context.get('patterns', {})
        seasonal = context.get('seasonal', {})
        
        # FIX: Ensure patterns is not None
        if not patterns:
            patterns = self._get_default_patterns()
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        composition_type = requirements['composition_type']
        
        _logger.info(f"📊 Generating from patterns: {product_count} products, €{budget:.2f}, type={composition_type}")
        
        # Get product pool
        products = self._get_smart_product_pool_enhanced(
            budget, dietary, composition_type, context
        )
        
        if not products:
            return {'success': False, 'error': 'No products available matching criteria'}
        
        # Score products based on patterns
        scored_products = []
        for product in products:
            score = self._score_product_by_patterns(
                product, patterns, seasonal, composition_type
            )
            scored_products.append((product, score))
        
        # Sort by score
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Select products
        selected = self._optimize_selection_enhanced(
            scored_products, product_count, budget, 
            requirements['budget_flexibility'],
            requirements['enforce_count']
        )
        
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        try:
            reasoning = self._build_enhanced_reasoning(
                requirements, selected, total_cost, budget, context
            )
            
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': budget,
                'target_year': fields.Date.today().year,
                'product_ids': [(6, 0, [p.id for p in selected])],
                'dietary_restrictions': ', '.join(dietary) if dietary else '',
                'client_notes': notes,
                'generation_method': 'ollama',
                'composition_type': composition_type,
                'confidence_score': 0.88,
                'ai_reasoning': reasoning
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.88,
                'message': f'Pattern-based: {len(selected)} products, €{total_cost:.2f}',
                'method': 'pattern_based_enhanced'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    # ================== PRODUCT SELECTION HELPERS ==================
    
    def _get_smart_product_pool(self, budget, dietary, context):
        """Get intelligently filtered product pool based on context"""
        
        patterns = context.get('patterns', {})
        
        # FIX: Ensure patterns is not None before accessing
        if not patterns:
            patterns = self._get_default_patterns()
        
        # Determine price range based on patterns or defaults
        if patterns.get('preferred_price_range'):
            min_price = patterns['preferred_price_range'].get('min', budget * 0.01)
            max_price = patterns['preferred_price_range'].get('max', budget * 0.4)
        else:
            min_price = max(5, budget * 0.01)  # At least €5
            max_price = budget * 0.4  # Max 40% of budget for single item
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
        ]
        
        # Add dietary filters
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
        available = []
        for product in products:
            if self._has_stock(product):
                available.append(product)
        
        # Score products based on patterns
        if patterns.get('favorite_products'):
            scored = []
            for product in available:
                score = 10 if product.id in patterns['favorite_products'] else 1
                scored.append((product, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            available = [p for p, s in scored]
        
        _logger.info(f"📦 Product pool: {len(available)} products (€{min_price:.2f}-€{max_price:.2f})")
        return available
    
    def _get_smart_product_pool_enhanced(self, budget, dietary, composition_type, context):
        """Get product pool considering composition type"""
        
        patterns = context.get('patterns', {})
        
        # FIX: Ensure patterns is not None
        if not patterns:
            patterns = self._get_default_patterns()
        
        # Adjust price range based on composition type
        if composition_type == 'experience':
            min_price = max(10, budget * 0.02)
            max_price = budget * 0.5
        elif composition_type == 'hybrid':
            min_price = max(15, budget * 0.03)
            max_price = budget * 0.4
        elif patterns.get('preferred_price_range'):
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
        
        # Add composition type filters
        if composition_type == 'hybrid':
            # For hybrid, prioritize wines and gourmet foods
            domain_hybrid = domain + [
                '|', '|',
                ('categ_id.name', 'ilike', 'wine'),
                ('categ_id.name', 'ilike', 'champagne'),
                ('categ_id.name', 'ilike', 'gourmet')
            ]
            products = self.env['product.template'].sudo().search(domain_hybrid, limit=500)
            
            # If not enough, add from general pool
            if len(products) < 100:
                general = self.env['product.template'].sudo().search(domain, limit=500)
                products = products | general
        else:
            products = self.env['product.template'].sudo().search(domain, limit=1000)
        
        # Apply dietary filters
        if dietary:
            filtered = []
            for product in products:
                if self._check_dietary_compliance(product, dietary) and self._has_stock(product):
                    filtered.append(product)
            products = filtered
        else:
            products = [p for p in products if self._has_stock(p)]
        
        _logger.info(f"📦 Product pool: {len(products)} products (€{min_price:.2f}-€{max_price:.2f}) for {composition_type}")
        return products
    
    def _smart_optimize_selection(self, products, target_count, budget, flexibility, enforce_strict, context):
        """Smart optimization considering all constraints and context"""
        
        # FIX: Ensure patterns exists before accessing
        patterns = context.get('patterns', {})
        if not patterns:
            patterns = self._get_default_patterns()
        
        if not target_count:
            # Estimate count based on budget
            avg_price = 50  # Default assumption
            if patterns.get('preferred_price_range'):
                avg_price = patterns['preferred_price_range'].get('avg', 50)
            target_count = max(5, int(budget / avg_price))
        
        min_budget = budget * (1 - flexibility/100)
        max_budget = budget * (1 + flexibility/100)
        # FIX: Avoid division by zero
        avg_price_target = budget / target_count if target_count > 0 else budget / 10
        
        # Score products
        scored_products = []
        for product in products:
            # Price fitness score
            price_diff = abs(product.list_price - avg_price_target)
            # FIX: Avoid division by zero in price score calculation
            if avg_price_target > 0:
                price_score = 1 / (1 + price_diff/avg_price_target)
            else:
                price_score = 0.5  # Default score when avg_price_target is 0
            
            # Pattern bonus
            pattern_score = 0
            if patterns.get('favorite_products') and product.id in patterns['favorite_products']:
                pattern_score = 2
            
            total_score = price_score + pattern_score
            scored_products.append((product, total_score))
        
        # Sort by score
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection with budget awareness
        selected = []
        current_total = 0
        
        for product, score in scored_products:
            if len(selected) >= target_count:
                break
            
            future_total = current_total + product.list_price
            remaining_slots = target_count - len(selected) - 1
            
            # Check if we can still meet budget constraints
            if remaining_slots > 0:
                # Estimate if we can fill remaining slots within budget
                min_remaining = remaining_slots * 5  # Assume min €5 per product
                max_remaining = remaining_slots * (max_budget - future_total) / remaining_slots if remaining_slots > 0 else 0
                
                if future_total + min_remaining <= max_budget:
                    selected.append(product)
                    current_total = future_total
            else:
                # Last product - check if total is within range
                if min_budget <= future_total <= max_budget:
                    selected.append(product)
                    current_total = future_total
        
        # If we don't have enough, add more
        if enforce_strict and len(selected) < target_count:
            remaining = [p for p in products if p not in selected]
            remaining.sort(key=lambda p: p.list_price)  # Add cheapest
            selected.extend(remaining[:target_count - len(selected)])
        
        return selected
    
    def _enforce_exact_count(self, selected, all_products, exact_count, budget):
        """Enforce exact product count no matter what"""
        
        if len(selected) == exact_count:
            return selected
        
        if len(selected) < exact_count:
            # Add products
            remaining_needed = exact_count - len(selected)
            available = [p for p in all_products if p not in selected]
            
            # Sort by price to add cheapest
            available.sort(key=lambda p: p.list_price)
            selected.extend(available[:remaining_needed])
            
            _logger.info(f"➕ Added {remaining_needed} products to meet count requirement")
        
        elif len(selected) > exact_count:
            # Remove products
            excess = len(selected) - exact_count
            
            # Remove most expensive to get closer to budget
            selected.sort(key=lambda p: p.list_price, reverse=True)
            selected = selected[excess:]  # Remove the most expensive ones
            
            _logger.info(f"➖ Removed {excess} products to meet count requirement")
        
        return selected[:exact_count]  # Final safety check
    
    def _select_with_category_requirements(self, products, categories_required, total_count, budget):
        """Select products meeting specific category requirements"""
        
        selected = []
        
        # First fulfill category requirements
        for category, count in categories_required.items():
            cat_products = [p for p in products if category.lower() in p.name.lower()]
            cat_products.sort(key=lambda p: abs(p.list_price - (budget/total_count if total_count else 50)))
            selected.extend(cat_products[:count])
            _logger.info(f"📂 Added {min(count, len(cat_products))} {category} products")
        
        # Fill remaining slots
        if total_count:
            remaining_count = total_count - len(selected)
            if remaining_count > 0:
                available = [p for p in products if p not in selected]
                available.sort(key=lambda p: abs(p.list_price - (budget/total_count)))
                selected.extend(available[:remaining_count])
        
        return selected
    
    def _score_product_by_patterns(self, product, patterns, seasonal, composition_type):
        """Score product based on patterns and composition type"""
        
        score = 1.0
        
        # FIX: Check patterns before accessing
        if patterns:
            # Favorite products
            if product.id in patterns.get('favorite_products', []):
                score += 5.0
            
            # Never repeated (penalty)
            if product.id in patterns.get('never_repeated_products', []):
                score -= 2.0
            
            # Category preference
            if hasattr(product, 'categ_id'):
                cat_name = product.categ_id.name
                if cat_name in patterns.get('preferred_categories', {}):
                    score += patterns['preferred_categories'][cat_name] * 0.5
        
        # Seasonal scoring
        if seasonal and product.id in seasonal.get('seasonal_favorites', []):
            score += 2.0
        
        # Composition type scoring
        if composition_type == 'hybrid' and hasattr(product, 'categ_id'):
            cat_name = product.categ_id.name.lower()
            if 'wine' in cat_name or 'champagne' in cat_name:
                score += 3.0
            elif 'cheese' in cat_name or 'gourmet' in cat_name:
                score += 2.0
        elif composition_type == 'experience':
            # Prefer unique/experiential products
            if hasattr(product, 'name'):
                if any(word in product.name.lower() for word in ['experience', 'tour', 'tasting', 'workshop']):
                    score += 3.0
        
        return score
    
    def _optimize_selection_enhanced(self, scored_products, target_count, budget, flexibility, enforce_strict):
        """Enhanced selection optimization"""
        
        min_budget = budget * (1 - flexibility/100)
        max_budget = budget * (1 + flexibility/100)
        # FIX: Avoid division by zero
        avg_price_target = budget / target_count if target_count > 0 else 50
        
        selected = []
        current_total = 0
        
        for product, score in scored_products:
            if enforce_strict and len(selected) >= target_count:
                break
            
            future_total = current_total + product.list_price
            remaining_slots = target_count - len(selected) - 1
            
            # Check if we can still meet constraints
            if remaining_slots > 0:
                min_remaining = remaining_slots * 5  # Minimum €5 per product
                if future_total + min_remaining <= max_budget:
                    selected.append(product)
                    current_total = future_total
            else:
                # Last slot - check total is within range
                if min_budget <= future_total <= max_budget:
                    selected.append(product)
                    current_total = future_total
                    break
        
        # Enforce exact count if required
        if enforce_strict:
            if len(selected) < target_count:
                # Add more products
                remaining = [p for p, _ in scored_products if p not in selected]
                selected.extend(remaining[:target_count - len(selected)])
            elif len(selected) > target_count:
                # Remove excess
                selected = selected[:target_count]
        
        return selected
    
    def _find_complementary_products(self, count, budget, dietary, exclude_ids, context):
        """Find complementary new products"""
        
        patterns = context.get('patterns', {})
        seasonal = context.get('seasonal', {})
        
        # FIX: Ensure patterns is not None
        if not patterns:
            patterns = self._get_default_patterns()
        
        # Determine search criteria
        # FIX: Avoid division by zero
        avg_price = budget / count if count > 0 else 50
        min_price = max(5, avg_price * 0.3)
        max_price = avg_price * 2
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
            ('id', 'not in', exclude_ids)
        ]
        
        # Add dietary filters
        if dietary:
            if 'halal' in dietary:
                if 'is_halal_compatible' in self.env['product.template']._fields:
                    domain.append(('is_halal_compatible', '!=', False))
                if 'contains_pork' in self.env['product.template']._fields:
                    domain.append(('contains_pork', '=', False))
                if 'contains_alcohol' in self.env['product.template']._fields:
                    domain.append(('contains_alcohol', '=', False))
        
        products = self.env['product.template'].sudo().search(domain, limit=500)
        
        # Filter for stock
        available = [p for p in products if self._has_stock(p)]
        
        # Score products for complementarity
        scored = []
        for product in available:
            score = 1.0
            
            # Prefer products from different categories than existing
            if hasattr(product, 'categ_id'):
                cat_name = product.categ_id.name
                # Higher score for underrepresented categories
                if cat_name not in patterns.get('preferred_categories', {}):
                    score += 0.5
            
            # Seasonal bonus
            if seasonal and product.id in seasonal.get('seasonal_favorites', []):
                score += 1.0
            
            # Price fitness
            price_diff = abs(product.list_price - avg_price)
            # FIX: Avoid division by zero in price score calculation
            if avg_price > 0:
                price_score = 1 / (1 + price_diff/avg_price)
            else:
                price_score = 0.5  # Default score when avg_price is 0
            score += price_score
            
            scored.append((product, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Select products
        selected = []
        current_total = 0
        
        for product, score in scored:
            if len(selected) >= count:
                break
            
            if current_total + product.list_price <= budget * 1.1:  # Allow 10% flexibility
                selected.append(product)
                current_total += product.list_price
        
        return selected
    
    def _build_comprehensive_reasoning(self, requirements, products, total_cost, budget, context):
        """Build detailed reasoning for the composition"""
        
        reasoning_parts = []
        
        # Basic stats
        reasoning_parts.append(f"📊 Generated {len(products)} products totaling €{total_cost:.2f}")
        
        # Budget compliance
        # FIX: Avoid division by zero in percentage calculation
        if budget > 0:
            variance = ((total_cost - budget) / budget) * 100
            reasoning_parts.append(f"💰 Budget variance: {variance:+.1f}%")
        else:
            reasoning_parts.append("💰 Budget: No budget specified")
        
        # Count compliance
        if requirements.get('enforce_count') and requirements.get('product_count'):
            if len(products) == requirements['product_count']:
                reasoning_parts.append(f"✅ Met exact count requirement: {requirements['product_count']}")
            else:
                reasoning_parts.append(f"⚠️ Count mismatch: {len(products)} vs {requirements['product_count']} required")
        
        # Dietary compliance
        if requirements.get('dietary'):
            reasoning_parts.append(f"🥗 Dietary restrictions applied: {', '.join(requirements['dietary'])}")
        
        # Historical insights used
        patterns = context.get('patterns')
        if patterns:
            if patterns.get('favorite_products'):
                favorites_included = sum(1 for p in products if p.id in patterns['favorite_products'])
                if favorites_included > 0:
                    reasoning_parts.append(f"⭐ Included {favorites_included} favorite products from history")
            
            if patterns.get('budget_trend'):
                reasoning_parts.append(f"📈 Budget trend: {patterns['budget_trend']}")
        
        # Category requirements
        if requirements.get('categories_required'):
            reasoning_parts.append(f"📂 Category requirements: {requirements['categories_required']}")
        
        return "\n".join(reasoning_parts)
    
    def _build_enhanced_reasoning(self, requirements, products, total_cost, budget, context):
        """Build comprehensive reasoning including sources used"""
        
        reasoning_parts = []
        
        # Data sources
        sources = []
        if requirements['budget_source'] != 'none':
            sources.append(f"Budget from {requirements['budget_source']}")
        if requirements['count_source'] != 'none':
            sources.append(f"Count from {requirements['count_source']}")
        if requirements['dietary_source'] != 'none':
            sources.append(f"Dietary from {requirements['dietary_source']}")
        if requirements['type_source'] != 'default':
            sources.append(f"Type from {requirements['type_source']}")
        
        reasoning_parts.append(f"📊 Data Sources: {', '.join(sources)}")
        
        # Results
        reasoning_parts.append(f"📦 Generated {len(products)} products = €{total_cost:.2f}")
        
        # Budget compliance
        # FIX: Avoid division by zero in percentage calculation
        if budget > 0:
            variance = ((total_cost - budget) / budget) * 100
            reasoning_parts.append(f"💰 Budget variance: {variance:+.1f}%")
        else:
            reasoning_parts.append("💰 Budget: No budget specified")
        
        # Requirements met
        if requirements.get('enforce_count'):
            if len(products) == requirements['product_count']:
                reasoning_parts.append(f"✅ Met exact count: {requirements['product_count']}")
            else:
                reasoning_parts.append(f"⚠️ Count mismatch: {len(products)} vs {requirements['product_count']}")
        
        if requirements.get('dietary'):
            reasoning_parts.append(f"🥗 Dietary applied: {', '.join(requirements['dietary'])}")
        
        if requirements.get('composition_type') != 'custom':
            reasoning_parts.append(f"🎁 Composition type: {requirements['composition_type']}")
        
        # Historical insights
        patterns = context.get('patterns')
        if patterns and patterns.get('favorite_products'):
            favorites_included = sum(1 for p in products if p.id in patterns['favorite_products'])
            if favorites_included > 0:
                reasoning_parts.append(f"⭐ Included {favorites_included} favorite products")
        
        return "\n".join(reasoning_parts)
    
    # ================== DATA RETRIEVAL METHODS ==================
    
    def _get_all_previous_sales_data(self, partner_id):
        """Get ALL previous sales data for 80/20 rule application"""
        
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
                        'times_ordered': 0  # Will be calculated later
                    })
            
            if products:
                all_sales_data.append({
                    'order': sale,
                    'order_date': sale.date_order,
                    'products': products,
                    'total': sale.amount_untaxed
                })
        
        # Calculate how many times each product was ordered
        product_frequency = {}
        for sale_data in all_sales_data:
            for prod_data in sale_data['products']:
                prod_id = prod_data['product'].id
                product_frequency[prod_id] = product_frequency.get(prod_id, 0) + 1
        
        # Update frequency in the data
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
    
    def _get_available_products(self, target_budget, dietary_restrictions):
        """Get available products with proper filtering"""
        
        min_price = max(10, target_budget * 0.02)
        max_price = target_budget * 0.40
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
        ]
        
        if dietary_restrictions:
            if 'halal' in dietary_restrictions:
                if 'is_halal_compatible' in self.env['product.template']._fields:
                    domain.append(('is_halal_compatible', '!=', False))
                if 'contains_pork' in self.env['product.template']._fields:
                    domain.append(('contains_pork', '=', False))
                if 'contains_alcohol' in self.env['product.template']._fields:
                    domain.append(('contains_alcohol', '=', False))
        
        products = self.env['product.template'].sudo().search(domain, limit=500)
        
        available = []
        for product in products:
            if self._has_stock(product):
                available.append(product)
        
        _logger.info(f"Found {len(available)} available products")
        return available
    
    # ================== LEARNING & ANALYSIS METHODS ==================
    
    def _get_or_update_learning_cache(self, partner_id):
        """Get cached learning data or update if expired"""
        
        # Check if cache is valid (24 hours)
        if self.cache_expiry and self.cache_expiry > datetime.now():
            if self.learning_cache:
                try:
                    cache = json.loads(self.learning_cache)
                    if str(partner_id) in cache:
                        _logger.info("Using cached learning data")
                        return cache[str(partner_id)]
                except:
                    pass
        
        # Generate new analysis
        _logger.info("Generating fresh learning analysis")
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
        """Analyze patterns across all client orders"""
        
        all_orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc')
        
        if not all_orders:
            _logger.info(f"No historical orders found for partner {partner_id}")
            return self._get_default_patterns()  # FIX: Return default patterns instead of None
        
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
        
        # Analyze each order
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
        
        # Find favorite products (ordered multiple times)
        pattern_analysis['favorite_products'] = [
            prod_id for prod_id, count in product_counter.items() 
            if count >= 2
        ]
        
        # Find never-repeated products
        pattern_analysis['never_repeated_products'] = [
            prod_id for prod_id, count in product_counter.items() 
            if count == 1 and len(all_orders) > 2
        ]
        
        # Preferred categories
        pattern_analysis['preferred_categories'] = dict(category_counter.most_common(5))
        
        # Price range preference
        if all_product_prices:
            pattern_analysis['preferred_price_range'] = {
                'min': min(all_product_prices),
                'max': max(all_product_prices),
                'avg': sum(all_product_prices) / len(all_product_prices)
            }
        
        # Budget trend analysis
        if len(all_orders) >= 3:
            recent_orders = all_orders[:3]
            older_orders = all_orders[-3:]
            recent_avg = sum(o.amount_untaxed for o in recent_orders) / 3
            older_avg = sum(o.amount_untaxed for o in older_orders) / 3
            
            if recent_avg > older_avg * 1.2:
                pattern_analysis['budget_trend'] = 'increasing'
            elif recent_avg < older_avg * 0.8:
                pattern_analysis['budget_trend'] = 'decreasing'
        
        return pattern_analysis
    
    def _analyze_seasonal_preferences(self, partner_id):
        """Identify seasonal patterns in purchases"""
        
        orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ])
        
        if not orders:
            return {}  # FIX: Return empty dict instead of None
        
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
        """Get current season based on month"""
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
        """Find clients with similar purchase patterns"""
        
        target_patterns = self._analyze_client_purchase_patterns(partner_id)
        
        if not target_patterns or target_patterns.get('total_orders', 0) == 0:
            return []
        
        # Find other clients with orders
        all_clients = self.env['sale.order'].read_group(
            [('state', 'in', ['sale', 'done'])],
            ['partner_id'],
            ['partner_id']
        )
        
        similar_clients = []
        
        for client_data in all_clients[:20]:  # Limit to 20 for performance
            other_partner_id = client_data['partner_id'][0]
            if other_partner_id == partner_id:
                continue
            
            other_patterns = self._analyze_client_purchase_patterns(other_partner_id)
            if not other_patterns or other_patterns.get('total_orders', 0) == 0:
                continue
            
            # Calculate similarity score
            similarity_score = self._calculate_similarity(target_patterns, other_patterns)
            
            if similarity_score > 0.6:  # 60% similarity threshold
                similar_clients.append({
                    'partner_id': other_partner_id,
                    'similarity': similarity_score,
                    'patterns': other_patterns
                })
        
        # Sort by similarity and return top matches
        similar_clients.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_clients[:limit]
    
    def _calculate_similarity(self, patterns1, patterns2):
        """Calculate similarity score between two client patterns"""
        
        score = 0
        factors = 0
        
        # Budget similarity (within 30%)
        avg1 = patterns1.get('avg_order_value', 0)
        avg2 = patterns2.get('avg_order_value', 0)
        if avg1 and avg2:
            # FIX: Avoid division by zero with max()
            max_avg = max(avg1, avg2)
            if max_avg > 0:
                budget_diff = abs(avg1 - avg2) / max_avg
                if budget_diff < 0.3:
                    score += (1 - budget_diff)
                factors += 1
        
        # Product count similarity
        count1 = patterns1.get('avg_product_count', 0)
        count2 = patterns2.get('avg_product_count', 0)
        if count1 and count2:
            # FIX: Avoid division by zero with max()
            max_count = max(count1, count2)
            if max_count > 0:
                count_diff = abs(count1 - count2) / max_count
                if count_diff < 0.3:
                    score += (1 - count_diff)
                factors += 1
        
        # Category overlap
        cats1 = set(patterns1.get('preferred_categories', {}).keys())
        cats2 = set(patterns2.get('preferred_categories', {}).keys())
        if cats1 and cats2:
            union_size = len(cats1.union(cats2))
            if union_size > 0:  # FIX: Avoid division by zero
                overlap = len(cats1.intersection(cats2)) / union_size
                score += overlap
                factors += 1
        
        # Budget trend similarity
        if patterns1.get('budget_trend') == patterns2.get('budget_trend'):
            score += 0.5
            factors += 0.5
        
        # FIX: Avoid division by zero when calculating final score
        return score / factors if factors > 0 else 0
    
    # ================== NOTES PARSING METHODS ==================
    
    def _parse_notes_with_ollama(self, notes, form_data=None):
        """Use Ollama to intelligently parse notes and extract requirements"""
        
        if not notes:
            return {'use_default': True}
        
        if not self.ollama_enabled:
            return self._parse_notes_basic_fallback(notes)
        
        prompt = f"""You are an expert at understanding customer requirements for luxury gift compositions.

Analyze the following customer notes and extract ALL requirements mentioned.

CUSTOMER NOTES: "{notes}"

Extract and return ONLY a valid JSON object with these fields:
{{
    "product_count": <number or null>,
    "budget": <number or null>,
    "budget_flexibility": <5 for strict, 10 for normal, 15 for flexible, 20 for very flexible>,
    "dietary_restrictions": ["halal", "vegan", "vegetarian", "gluten_free", "non_alcoholic"],
    "composition_type": "hybrid|experience|custom|null",
    "categories_required": {{"wines": 2, "cheese": 3}},
    "specific_products": ["product names"],
    "exclude_products": ["things to avoid"],
    "special_instructions": ["any special requests"],
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
                        'specific_products': extracted.get('specific_products', []),
                        'categories_excluded': extracted.get('exclude_products', []),
                        'special_instructions': extracted.get('special_instructions', [])
                    }
                    
                    _logger.info(f"Ollama parsed requirements (confidence: {extracted.get('confidence', 0)}%): {requirements}")
                    
                    return requirements
                    
                except json.JSONDecodeError as e:
                    _logger.error(f"Failed to parse Ollama response as JSON: {e}")
                    return self._parse_notes_basic_fallback(notes)
            else:
                return self._parse_notes_basic_fallback(notes)
                
        except Exception as e:
            _logger.error(f"Error in Ollama parsing: {e}")
            return self._parse_notes_basic_fallback(notes)
    
    def _parse_notes_basic_fallback(self, notes):
        """Basic fallback parser for when Ollama is not available"""
        
        parsed = {
            'use_default': False,
            'product_count': None,
            'budget_override': None,
            'budget_flexibility': 10,
            'dietary': [],
            'composition_type': None,
            'categories_required': {},
            'categories_excluded': [],
            'special_instructions': []
        }
        
        notes_lower = notes.lower()
        
        # Basic number extraction
        numbers = re.findall(r'\b(\d+)\b', notes)
        for num in numbers:
            num_int = int(num)
            if 1 <= num_int <= 100:
                if 'product' in notes_lower or 'item' in notes_lower:
                    parsed['product_count'] = num_int
            elif 100 <= num_int <= 10000:
                if 'budget' in notes_lower or '€' in notes or '$' in notes:
                    parsed['budget_override'] = float(num_int)
        
        # Basic dietary detection
        if 'halal' in notes_lower:
            parsed['dietary'].append('halal')
        if 'vegan' in notes_lower:
            parsed['dietary'].append('vegan')
        
        # Composition type
        if 'hybrid' in notes_lower:
            parsed['composition_type'] = 'hybrid'
        elif 'experience' in notes_lower:
            parsed['composition_type'] = 'experience'
        
        return parsed
    
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
    
    def _validate_and_log_result(self, result, requirements):
        """Validate and log the result against requirements"""
        
        actual_count = result.get('product_count', 0)
        actual_cost = result.get('total_cost', 0)
        expected_count = requirements.get('product_count')
        expected_budget = requirements['budget']
        flexibility = requirements['budget_flexibility']
        
        # Count validation
        if requirements.get('enforce_count') and expected_count:
            if actual_count == expected_count:
                _logger.info(f"✅ Count requirement MET: {actual_count} products")
            else:
                _logger.error(f"❌ Count requirement FAILED: {actual_count} != {expected_count}")
        
        # Budget validation
        min_budget = expected_budget * (1 - flexibility/100)
        max_budget = expected_budget * (1 + flexibility/100)
        
        if min_budget <= actual_cost <= max_budget:
            # FIX: Avoid division by zero in percentage calculation
            if expected_budget > 0:
                variance = ((actual_cost - expected_budget) / expected_budget) * 100
                _logger.info(f"✅ Budget requirement MET: €{actual_cost:.2f} ({variance:+.1f}% variance)")
            else:
                _logger.info(f"✅ Budget requirement MET: €{actual_cost:.2f}")
        else:
            # FIX: Avoid division by zero in percentage calculation
            if expected_budget > 0:
                variance = ((actual_cost - expected_budget) / expected_budget) * 100
                _logger.error(f"❌ Budget requirement FAILED: €{actual_cost:.2f} ({variance:+.1f}% variance)")
            else:
                _logger.error(f"❌ Budget requirement FAILED: €{actual_cost:.2f} (no expected budget)")
        
        return result.get('compliant', False)
    
    # ================== ACTION METHODS ==================
    
    def action_view_recommendations(self):
        """View all recommendations"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'All Recommendations',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('generation_method', '=', 'ollama')],
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
                ('generation_method', '=', 'ollama'),
                ('confidence_score', '>=', 0.8)
            ],
            'context': {'search_default_partner_id': True}
        }
    
    def trigger_learning(self):
        """Trigger comprehensive learning analysis"""
        self.ensure_one()
        
        # Force cache refresh
        self.cache_expiry = datetime.now() - timedelta(days=1)
        
        # Analyze all clients
        all_clients = self.env['sale.order'].read_group(
            [('state', 'in', ['sale', 'done'])],
            ['partner_id'],
            ['partner_id']
        )
        
        analyzed_count = 0
        for client_data in all_clients[:50]:  # Limit for performance
            partner_id = client_data['partner_id'][0]
            self._get_or_update_learning_cache(partner_id)
            analyzed_count += 1
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': '🧠 Learning Analysis Complete',
                'message': f'Analyzed {analyzed_count} clients. Cache updated for 24 hours.',
                'type': 'success',
                'sticky': False,
            }
        }

    def generate_gift_recommendations_v2(self, partner_id, target_budget, 
                                        client_notes='', dietary_restrictions=None,
                                        composition_type=None):
        """
        Enhanced version that uses the composition engine for business rules
        """
        
        # Check if we should use the composition engine
        if self._should_use_composition_engine(partner_id, composition_type):
            engine = self.env['gift.composition.engine'].search([('active', '=', True)], limit=1)
            if not engine:
                engine = self.env['gift.composition.engine'].create({
                    'name': 'Master Engine',
                    'ollama_recommender_id': self.id
                })
            
            return engine.generate_complete_composition(
                partner_id=partner_id,
                target_budget=target_budget,
                client_notes=client_notes,
                dietary_restrictions=dietary_restrictions,
                composition_type=composition_type,
                wizard_data={'recommender': self}
            )
        
        # Otherwise use existing logic
        return self.generate_gift_recommendations(
            partner_id, target_budget, client_notes, 
            dietary_restrictions, composition_type
        )

    def _should_use_composition_engine(self, partner_id, composition_type):
        """Determine when to use the composition engine"""
        
        # Use composition engine for:
        # 1. Experience-based compositions
        if composition_type == 'experience':
            return True
        
        # 2. Clients with last year's data (for 80/20 rule)
        last_year_order = self._get_last_year_order(partner_id)
        if last_year_order:
            return True
        
        # 3. Explicit business rules request
        if self.env.context.get('use_business_rules'):
            return True
        
        return False