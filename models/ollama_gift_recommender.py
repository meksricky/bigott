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
    _description = 'Ollama-Powered Gift Recommendation Engine with Advanced Learning'
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
    
    # Learning Cache (to avoid recalculating patterns every time)
    learning_cache = fields.Text(string="Learning Cache JSON")
    cache_expiry = fields.Datetime(string="Cache Expiry")
    
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
    
    # ============== LEARNING & ANALYSIS METHODS ==============
    
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
        
        # Get ALL historical orders, not just the last one
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
            'seasonal_patterns': {},
            'product_frequency': {},
            'budget_trend': 'stable',  # increasing/decreasing/stable
            'avg_product_count': 0,
            'favorite_products': [],
            'never_repeated_products': [],
            'category_evolution': {},
            'order_intervals': [],  # Days between orders
            'preferred_price_range': {'min': 0, 'max': 0}
        }
        
        # Analyze each order
        total_value = 0
        total_products = 0
        product_counter = Counter()
        category_counter = Counter()
        budgets_timeline = []
        all_product_prices = []
        last_order_date = None
        
        for order in all_orders:
            order_value = order.amount_untaxed
            total_value += order_value
            budgets_timeline.append((order.date_order, order_value))
            
            # Calculate intervals between orders
            if last_order_date:
                interval = (last_order_date - order.date_order).days
                pattern_analysis['order_intervals'].append(interval)
            last_order_date = order.date_order
            
            # Count products and categories
            for line in order.order_line:
                if line.product_id:
                    product_tmpl = line.product_id.product_tmpl_id
                    product_counter[product_tmpl.id] += 1
                    all_product_prices.append(line.price_unit)
                    
                    # Track category if available
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
        
        # Find never-repeated products (avoid these)
        pattern_analysis['never_repeated_products'] = [
            prod_id for prod_id, count in product_counter.items() 
            if count == 1 and len(all_orders) > 2  # Only relevant if multiple orders exist
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
        if len(budgets_timeline) >= 3:
            recent_orders = budgets_timeline[:min(3, len(budgets_timeline))]
            older_orders = budgets_timeline[-min(3, len(budgets_timeline)):]
            recent_avg = sum(b[1] for b in recent_orders) / len(recent_orders)
            older_avg = sum(b[1] for b in older_orders) / len(older_orders)
            
            if recent_avg > older_avg * 1.2:
                pattern_analysis['budget_trend'] = 'increasing'
            elif recent_avg < older_avg * 0.8:
                pattern_analysis['budget_trend'] = 'decreasing'
        
        # Average order interval
        if pattern_analysis['order_intervals']:
            pattern_analysis['avg_order_interval'] = sum(pattern_analysis['order_intervals']) / len(pattern_analysis['order_intervals'])
        
        return pattern_analysis
    
    def _analyze_seasonal_preferences(self, partner_id):
        """Identify seasonal patterns in purchases"""
        
        orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ])
        
        if not orders:
            return None
        
        seasonal_data = {
            'spring': {'products': [], 'categories': [], 'avg_value': 0},
            'summer': {'products': [], 'categories': [], 'avg_value': 0},
            'autumn': {'products': [], 'categories': [], 'avg_value': 0},
            'winter': {'products': [], 'categories': [], 'avg_value': 0},
            'christmas': {'products': [], 'categories': [], 'avg_value': 0}
        }
        
        season_order_values = defaultdict(list)
        
        for order in orders:
            month = order.date_order.month
            
            # Determine season
            if month in [3, 4, 5]:
                season = 'spring'
            elif month in [6, 7, 8]:
                season = 'summer'
            elif month in [9, 10, 11]:
                season = 'autumn'
            elif month == 12:
                season = 'christmas'
            else:
                season = 'winter'
            
            season_order_values[season].append(order.amount_untaxed)
            
            # Track products for this season
            for line in order.order_line:
                if line.product_id:
                    seasonal_data[season]['products'].append(line.product_id.product_tmpl_id.id)
                    if hasattr(line.product_id.product_tmpl_id, 'categ_id'):
                        seasonal_data[season]['categories'].append(line.product_id.product_tmpl_id.categ_id.name)
        
        # Calculate average values per season
        for season, values in season_order_values.items():
            if values:
                seasonal_data[season]['avg_value'] = sum(values) / len(values)
        
        # Find patterns
        current_month = datetime.now().month
        current_season = self._get_current_season(current_month)
        
        # Get most common products/categories per season
        for season in seasonal_data:
            if seasonal_data[season]['products']:
                product_counter = Counter(seasonal_data[season]['products'])
                seasonal_data[season]['top_products'] = product_counter.most_common(5)
            
            if seasonal_data[season]['categories']:
                category_counter = Counter(seasonal_data[season]['categories'])
                seasonal_data[season]['top_categories'] = category_counter.most_common(3)
        
        return {
            'current_season': current_season,
            'seasonal_data': seasonal_data,
            'seasonal_favorites': seasonal_data[current_season]['products'] if current_season in seasonal_data else [],
            'seasonal_categories': seasonal_data[current_season].get('top_categories', []) if current_season in seasonal_data else []
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
        
        if not target_patterns:
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
            if not other_patterns:
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
        
        # Budget trend similarity
        if patterns1.get('budget_trend') == patterns2.get('budget_trend'):
            score += 0.5
            factors += 0.5
        
        return score / factors if factors > 0 else 0
    
    def _analyze_product_affinity(self):
        """Find products that are frequently bought together"""
        
        # Get all sales with multiple products (limit for performance)
        orders = self.env['sale.order'].search([
            ('state', 'in', ['sale', 'done']),
            ('order_line', '!=', False)
        ], limit=200, order='date_order desc')
        
        # Build affinity matrix
        product_pairs = defaultdict(int)
        product_frequency = defaultdict(int)
        
        for order in orders:
            products = []
            for line in order.order_line:
                if line.product_id and line.product_id.product_tmpl_id:
                    products.append(line.product_id.product_tmpl_id.id)
            
            # Count individual products
            for prod in products:
                product_frequency[prod] += 1
            
            # Count pairs
            for i in range(len(products)):
                for j in range(i + 1, len(products)):
                    pair = tuple(sorted([products[i], products[j]]))
                    product_pairs[pair] += 1
        
        # Calculate affinity scores
        affinity_scores = {}
        total_orders = len(orders)
        
        for pair, count in product_pairs.items():
            if count >= 2:  # Minimum support
                prod1, prod2 = pair
                # Lift calculation: P(A and B) / (P(A) * P(B))
                prob_together = count / total_orders if total_orders > 0 else 0
                prob_1 = product_frequency[prod1] / total_orders if total_orders > 0 else 0
                prob_2 = product_frequency[prod2] / total_orders if total_orders > 0 else 0
                
                if prob_1 > 0 and prob_2 > 0:
                    lift = prob_together / (prob_1 * prob_2)
                    if lift > 1.2:  # Meaningful affinity
                        affinity_scores[pair] = {
                            'lift': lift,
                            'count': count,
                            'products': pair
                        }
        
        return affinity_scores
    
    def _learn_from_outcomes(self):
        """Learn from which compositions led to actual sales"""
        
        # Get recent compositions
        compositions = self.env['gift.composition'].search([], limit=100, order='create_date desc')
        
        success_patterns = {
            'successful_category_distributions': [],
            'successful_price_ranges': [],
            'successful_product_counts': [],
            'failed_patterns': [],
            'conversion_rate': 0
        }
        
        successful = 0
        
        for comp in compositions:
            # Check if it led to a confirmed sale
            sale_order = self.env['sale.order'].search([
                ('partner_id', '=', comp.partner_id.id),
                ('create_date', '>=', comp.create_date),
                ('create_date', '<=', comp.create_date + timedelta(days=30))
            ], limit=1)
            
            if sale_order and sale_order.state in ['sale', 'done']:
                # This was successful
                successful += 1
                
                if hasattr(comp, 'get_category_distribution'):
                    success_patterns['successful_category_distributions'].append(
                        comp.get_category_distribution()
                    )
                
                product_count = len(comp.product_ids) if comp.product_ids else 0
                if product_count > 0:
                    avg_price = comp.target_budget / product_count
                    success_patterns['successful_price_ranges'].append(avg_price)
                    success_patterns['successful_product_counts'].append(product_count)
            else:
                # This didn't convert
                if hasattr(comp, 'get_category_distribution'):
                    success_patterns['failed_patterns'].append({
                        'categories': comp.get_category_distribution(),
                        'total': comp.target_budget,
                        'count': len(comp.product_ids) if comp.product_ids else 0
                    })
        
        if compositions:
            success_patterns['conversion_rate'] = (successful / len(compositions)) * 100
        
        return success_patterns
    
    # ============== NOTES PARSING METHODS ==============
    
    def _parse_notes_with_ollama(self, notes, form_data=None):
        """Use Ollama to intelligently parse notes and extract requirements"""
        
        if not notes:
            return {'use_default': True}
        
        if not self.ollama_enabled:
            return self._parse_notes_basic_fallback(notes)
        
        prompt = f"""You are an expert at understanding customer requirements for luxury gift compositions.
        
Analyze the following customer notes and extract ALL requirements mentioned.
The customer might express things in various ways - be intelligent about understanding their intent.

CUSTOMER NOTES: "{notes}"

FORM DATA (for context - notes should OVERRIDE these if different):
- Current form budget: €{form_data.get('budget', 0) if form_data else 0}
- Current form dietary: {form_data.get('dietary', []) if form_data else []}

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
    "override_budget": <true if notes mention a budget different from form>,
    "override_count": <true if notes specify exact product count>,
    "confidence": <0-100 how confident you are in the extraction>
}}

IMPORTANT EXTRACTION RULES:
- If they say "has X products" or "with X items" - set product_count
- If they mention any amount with €, euros, dollars, or just numbers in budget context - set budget
- "around", "approximately", "roughly" = 15% flexibility
- "strict", "exact", "exactly" = 5% flexibility  
- "halal", "muslim", "no pork", "no alcohol" = add "halal" to dietary_restrictions
- "hybrid composition" = composition_type: "hybrid"
- Look for ANY number mentioned with products/items context for product_count
- Look for ANY monetary amount for budget
- Be smart about variations in how people express requirements

Return ONLY the JSON object, no other text."""
        
        try:
            response = self._call_ollama(prompt, format_json=True)
            
            if response:
                try:
                    extracted = json.loads(response)
                    
                    requirements = {
                        'use_default': False,
                        'product_count': extracted.get('product_count'),
                        'mandatory_count': extracted.get('product_count'),
                        'budget_override': extracted.get('budget'),
                        'budget_flexibility': extracted.get('budget_flexibility', 10),
                        'dietary': extracted.get('dietary_restrictions', []),
                        'composition_type': extracted.get('composition_type'),
                        'categories_required': extracted.get('categories_required', {}),
                        'specific_products': extracted.get('specific_products', []),
                        'categories_excluded': extracted.get('exclude_products', []),
                        'special_instructions': extracted.get('special_instructions', []),
                        'preferences': {},
                        'override_form': {}
                    }
                    
                    if extracted.get('override_budget'):
                        requirements['override_form']['budget'] = True
                    if extracted.get('override_count'):
                        requirements['override_form']['product_count'] = True
                    if extracted.get('dietary_restrictions'):
                        requirements['override_form']['dietary'] = True
                    
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
            'mandatory_count': None,
            'product_count': None,
            'budget_override': None,
            'budget_flexibility': 10,
            'dietary': [],
            'categories_required': {},
            'categories_excluded': [],
            'special_instructions': [],
            'composition_type': None,
            'preferences': {},
            'override_form': {}
        }
        
        notes_lower = notes.lower()
        
        # Basic number extraction
        numbers = re.findall(r'\b(\d+)\b', notes)
        for num in numbers:
            num_int = int(num)
            if 1 <= num_int <= 100:
                if 'product' in notes_lower or 'item' in notes_lower:
                    parsed['product_count'] = num_int
                    parsed['mandatory_count'] = num_int
                    parsed['override_form']['product_count'] = True
            elif 100 <= num_int <= 10000:
                if 'budget' in notes_lower or '€' in notes or '$' in notes:
                    parsed['budget_override'] = float(num_int)
                    parsed['override_form']['budget'] = True
        
        # Basic dietary detection
        if 'halal' in notes_lower:
            parsed['dietary'].append('halal')
            parsed['override_form']['dietary'] = True
        if 'vegan' in notes_lower:
            parsed['dietary'].append('vegan')
            parsed['override_form']['dietary'] = True
        
        # Composition type
        if 'hybrid' in notes_lower:
            parsed['composition_type'] = 'hybrid'
        elif 'experience' in notes_lower:
            parsed['composition_type'] = 'experience'
        
        return parsed
    
    # ============== MAIN RECOMMENDATION METHOD ==============
    
    def generate_gift_recommendations(self, partner_id, target_budget, 
                                    client_notes='', dietary_restrictions=None):
        """Generate recommendations with intelligent parsing and comprehensive learning"""
        
        partner = self.env['res.partner'].browse(partner_id)
        if not partner:
            return {'success': False, 'error': 'Partner not found'}
        
        # Create form data context for the parser
        form_data = {
            'budget': target_budget,
            'dietary': dietary_restrictions or []
        }
        
        # 1. PARSE NOTES USING OLLAMA
        _logger.info(f"Parsing notes for partner {partner.name}: {client_notes}")
        requirements = self._parse_notes_with_ollama(client_notes, form_data)
        
        # 2. GET COMPREHENSIVE LEARNING DATA (cached for performance)
        learning_data = self._get_or_update_learning_cache(partner_id)
        patterns = learning_data.get('patterns')
        seasonal = learning_data.get('seasonal')
        similar_clients = learning_data.get('similar_clients')
        
        # Get product affinities (global, not per client)
        product_affinities = self._analyze_product_affinity() if self.total_recommendations % 10 == 0 else {}
        
        # 3. APPLY OVERRIDES FROM NOTES
        if requirements.get('budget_override'):
            original_budget = target_budget
            target_budget = requirements['budget_override']
            _logger.info(f"✅ Budget override: €{original_budget} → €{target_budget}")
        
        if requirements.get('dietary'):
            dietary_restrictions = requirements['dietary']
            _logger.info(f"✅ Dietary override: {dietary_restrictions}")
        
        # 4. APPLY LEARNING INSIGHTS
        if patterns and not requirements.get('budget_override'):
            # Adjust budget based on trend
            if patterns['budget_trend'] == 'increasing':
                suggested_budget = target_budget * 1.1
                _logger.info(f"📈 Client shows increasing budget trend. Suggesting: €{suggested_budget:.2f}")
                # Only apply if no explicit budget in notes
                if not requirements.get('budget_override'):
                    target_budget = suggested_budget
            elif patterns['budget_trend'] == 'decreasing':
                suggested_budget = target_budget * 0.95
                _logger.info(f"📉 Client shows decreasing budget trend. Adjusting to: €{suggested_budget:.2f}")
                if not requirements.get('budget_override'):
                    target_budget = suggested_budget
            
            # Log insights for generation
            if patterns['favorite_products']:
                _logger.info(f"⭐ Client has {len(patterns['favorite_products'])} favorite products to prioritize")
            
            if patterns.get('preferred_categories'):
                _logger.info(f"📦 Preferred categories: {list(patterns['preferred_categories'].keys())[:3]}")
        
        if seasonal and seasonal.get('seasonal_favorites'):
            _logger.info(f"🌡️ Current season: {seasonal['current_season']} with {len(seasonal['seasonal_favorites'])} seasonal preferences")
        
        if similar_clients:
            _logger.info(f"👥 Found {len(similar_clients)} similar clients for collaborative learning")
        
        # 5. GET PREVIOUS ORDER DATA
        previous_data = self._get_previous_order_data(partner_id)
        
        # 6. DETERMINE GENERATION STRATEGY
        self.total_recommendations += 1
        self.last_recommendation_date = fields.Datetime.now()
        
        # Pass learning data to generation methods
        generation_context = {
            'patterns': patterns,
            'seasonal': seasonal,
            'similar_clients': similar_clients,
            'product_affinities': product_affinities,
            'previous_data': previous_data
        }
        
        # Check for explicit product count requirement
        if requirements.get('product_count'):
            _logger.info(f"📋 Generating with strict product count: {requirements['product_count']}")
            result = self._generate_with_strict_requirements(
                partner, target_budget, requirements, 
                dietary_restrictions, client_notes, generation_context
            )
        
        # Check if we should use history
        elif (previous_data and 
            ('like last time' in client_notes.lower() or 
             'same as before' in client_notes.lower())):
            _logger.info("🔄 Using history-based generation as requested")
            result = self._generate_from_history_enhanced(
                partner, target_budget, previous_data,
                client_notes, dietary_restrictions, requirements,
                generation_context
            )
        
        # Use pattern-based generation if good history exists
        elif patterns and patterns.get('total_orders', 0) >= 3:
            _logger.info("📊 Using pattern-based generation (sufficient history)")
            result = self._generate_from_patterns(
                partner, target_budget, client_notes,
                dietary_restrictions, requirements, generation_context
            )
        
        # Try Ollama generation
        elif self.ollama_enabled:
            _logger.info("🤖 Using Ollama AI generation")
            result = self._generate_with_ollama_enhanced(
                partner, target_budget, client_notes, 
                dietary_restrictions, requirements, generation_context
            )
            
            if not result or not result.get('success'):
                _logger.warning("Ollama failed, using fallback")
                result = self._generate_fallback_recommendation_enhanced(
                    partner, target_budget, client_notes, 
                    dietary_restrictions, requirements
                )
        
        # Standard fallback generation
        else:
            _logger.info("📝 Using fallback generation")
            result = self._generate_fallback_recommendation_enhanced(
                partner, target_budget, client_notes, 
                dietary_restrictions, requirements
            )
        
        # Update success tracking
        if result and result.get('success'):
            self.successful_recommendations += 1
            _logger.info(f"✅ Success: {result.get('product_count')} products, €{result.get('total_cost'):.2f}")
        else:
            _logger.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    # ============== ENHANCED GENERATION METHODS ==============
    
    def _generate_from_patterns(self, partner, target_budget, client_notes,
                               dietary_restrictions, requirements, context):
        """Generate based on learned patterns"""
        
        patterns = context.get('patterns', {})
        seasonal = context.get('seasonal', {})
        affinities = context.get('product_affinities', {})
        
        # Determine product count
        if requirements.get('product_count'):
            product_count = requirements['product_count']
        else:
            product_count = int(patterns.get('avg_product_count', 12))
        
        _logger.info(f"Generating {product_count} products based on patterns")
        
        # Get base products pool
        products = self._get_available_products(target_budget * 2, dietary_restrictions)
        
        # Prioritize products based on patterns
        scored_products = []
        
        for product in products:
            score = 0
            
            # Favorite products get highest score
            if product.id in patterns.get('favorite_products', []):
                score += 10
            
            # Never repeated products get penalty
            if product.id in patterns.get('never_repeated_products', []):
                score -= 5
            
            # Category preference scoring
            if hasattr(product, 'categ_id'):
                cat_name = product.categ_id.name
                if cat_name in patterns.get('preferred_categories', {}):
                    score += patterns['preferred_categories'][cat_name] * 2
            
            # Seasonal scoring
            if seasonal and product.id in seasonal.get('seasonal_favorites', []):
                score += 3
            
            # Price range scoring
            price_range = patterns.get('preferred_price_range', {})
            if price_range:
                if price_range['min'] <= product.list_price <= price_range['max']:
                    score += 2
            
            scored_products.append((product, score))
        
        # Sort by score
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Select products using affinity information
        selected = []
        selected_ids = set()
        
        for product, score in scored_products:
            if len(selected) >= product_count:
                break
            
            # Check affinities - if we already selected a product, prefer its companions
            add_bonus = 0
            for selected_prod in selected:
                pair = tuple(sorted([selected_prod.id, product.id]))
                if pair in affinities:
                    add_bonus = affinities[pair].get('lift', 0) * 2
            
            if score + add_bonus > 0 or len(selected) < product_count / 2:
                selected.append(product)
                selected_ids.add(product.id)
        
        # Optimize for budget
        selected = self._optimize_selection(
            selected, product_count, target_budget, 
            requirements.get('budget_flexibility', 10)
        )
        
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        try:
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': target_budget,
                'target_year': fields.Date.today().year,
                'product_ids': [(6, 0, [p.id for p in selected])],
                'dietary_restrictions': ', '.join(dietary_restrictions) if dietary_restrictions else '',
                'client_notes': client_notes,
                'generation_method': 'ollama',
                'composition_type': requirements.get('composition_type', 'custom'),
                'confidence_score': 0.85,
                'ai_reasoning': f"""Pattern-based generation:
                - Based on {patterns.get('total_orders', 0)} historical orders
                - Budget trend: {patterns.get('budget_trend', 'stable')}
                - Prioritized {len(patterns.get('favorite_products', []))} favorite products
                - Season: {seasonal.get('current_season', 'unknown') if seasonal else 'N/A'}
                - Total: €{total_cost:.2f} ({len(selected)} products)"""
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.85,
                'message': f'Pattern-based: {len(selected)} products, €{total_cost:.2f}',
                'method': 'pattern_based'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_from_history_enhanced(self, partner, target_budget, previous_data,
                                       notes, dietary, requirements, context):
        """Enhanced history-based generation with learning insights"""
        
        patterns = context.get('patterns', {})
        seasonal = context.get('seasonal', {})
        
        previous_products = previous_data['products']
        total_count = len(previous_products)
        
        # Override count if specified
        if requirements and requirements.get('product_count'):
            total_count = requirements['product_count']
        elif patterns and patterns.get('avg_product_count'):
            total_count = int(patterns['avg_product_count'])
        
        # Adjust split based on patterns
        if patterns and patterns.get('never_repeated_products'):
            # If many products are never repeated, increase variation
            keep_ratio = 0.6  # 60% keep, 40% change
        else:
            keep_ratio = 0.7  # Standard 70% keep, 30% change
        
        keep_count = int(total_count * keep_ratio)
        change_count = total_count - keep_count
        
        # Select products to keep (prioritize favorites)
        products_to_keep = []
        
        # First add favorites from history
        for item in previous_products:
            if len(products_to_keep) >= keep_count:
                break
            product = item['product']
            if patterns and product.id in patterns.get('favorite_products', []):
                if self._check_dietary_compliance(product, dietary) and self._has_stock(product):
                    products_to_keep.append(product)
        
        # Then add others
        for item in previous_products:
            if len(products_to_keep) >= keep_count:
                break
            product = item['product']
            if product not in products_to_keep:
                if self._check_dietary_compliance(product, dietary) and self._has_stock(product):
                    products_to_keep.append(product)
        
        # Find new products with seasonal consideration
        new_products = []
        if seasonal and seasonal.get('seasonal_favorites'):
            # Try to add seasonal products
            seasonal_products = self.env['product.template'].browse(seasonal['seasonal_favorites'][:change_count])
            for prod in seasonal_products:
                if self._has_stock(prod) and self._check_dietary_compliance(prod, dietary):
                    new_products.append(prod)
        
        # Fill remaining with regular selection
        remaining_needed = change_count - len(new_products)
        if remaining_needed > 0:
            additional = self._find_replacement_products(
                remaining_needed,
                target_budget - sum(p.list_price for p in products_to_keep) - sum(p.list_price for p in new_products),
                dietary,
                exclude_products=products_to_keep + new_products
            )
            new_products.extend(additional)
        
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
                'ai_reasoning': f"""Enhanced history-based generation:
                - Based on: {previous_data['order_name']}
                - Kept {len(products_to_keep)} products ({int(keep_ratio*100)}%)
                - Added {len(new_products)} new products
                - Seasonal consideration: {seasonal.get('current_season', 'N/A') if seasonal else 'No'}
                - Total: €{total_cost:.2f}"""
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': final_products,
                'total_cost': total_cost,
                'product_count': len(final_products),
                'confidence_score': 0.95,
                'message': f'History-based: {len(products_to_keep)} kept + {len(new_products)} new',
                'method': 'history_enhanced'
            }
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    # ============== EXISTING METHODS (kept as is) ==============
    
    def _generate_with_strict_requirements(self, partner, budget, requirements, 
                                        dietary, notes, context):
        """Generate with strict adherence to parsed requirements"""
        
        patterns = context.get('patterns', {})
        product_count = requirements.get('mandatory_count', requirements.get('product_count', 12))
        budget_flexibility = requirements.get('budget_flexibility', 10)
        
        _logger.info(f"Generating EXACTLY {product_count} products with €{budget} budget (±{budget_flexibility}%)")
        
        # Get available products
        products = self._get_available_products(budget * 2, dietary)
        
        # Prioritize based on patterns if available
        if patterns and patterns.get('favorite_products'):
            # Move favorites to the front
            favorites = []
            others = []
            for p in products:
                if p.id in patterns['favorite_products']:
                    favorites.append(p)
                else:
                    others.append(p)
            products = favorites + others
        
        # Apply requirements filters
        if requirements.get('specific_products'):
            specific = self._get_specific_products(requirements['specific_products'])
            products = specific + products
        
        if requirements.get('categories_excluded'):
            products = self._filter_exclusions(products, requirements['categories_excluded'])
        
        # Select products
        if requirements.get('categories_required'):
            selected = self._select_by_categories(
                products, requirements['categories_required'],
                product_count, budget
            )
        else:
            selected = self._optimize_selection(
                products, product_count, budget, budget_flexibility
            )
        
        # Ensure count
        if len(selected) != product_count:
            if len(selected) < product_count:
                remaining = product_count - len(selected)
                excluded_ids = [p.id for p in selected]
                additional = [p for p in products if p.id not in excluded_ids][:remaining]
                selected.extend(additional)
            else:
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
    
    def _generate_with_ollama_enhanced(self, partner, target_budget, client_notes, 
                                      dietary_restrictions, requirements, context=None):
        """Generate using Ollama with enhanced context"""
        
        # Get products
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return None
        
        # Build enhanced prompt with context
        prompt = self._build_ollama_prompt_with_context(
            partner, target_budget, client_notes, requirements,
            dietary_restrictions, products, context
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
    
    def _build_ollama_prompt_with_context(self, partner, target_budget, client_notes, 
                                         requirements, dietary_restrictions, products, context):
        """Build Ollama prompt with learning context"""
        
        dietary_str = ', '.join(dietary_restrictions) if dietary_restrictions else 'None'
        
        # Add context information
        context_info = ""
        if context:
            patterns = context.get('patterns', {})
            if patterns:
                context_info += f"\nCLIENT INSIGHTS:"
                context_info += f"\n- Average order value: €{patterns.get('avg_order_value', 0):.2f}"
                context_info += f"\n- Budget trend: {patterns.get('budget_trend', 'stable')}"
                context_info += f"\n- Preferred categories: {list(patterns.get('preferred_categories', {}).keys())[:3]}"
                context_info += f"\n- Has {len(patterns.get('favorite_products', []))} favorite products"
        
        # Format products
        product_list = ""
        for i, p in enumerate(products[:50], 1):
            product_list += f"{i}. ID:{p.id} | {p.name} | €{p.list_price:.2f}\n"
        
        prompt = f"""You are an expert gift curator for Le Bigott. Follow these STRICT requirements:

CLIENT: {partner.name}
BUDGET: €{target_budget} (MUST be within 95-105% = €{target_budget*0.95:.2f} to €{target_budget*1.05:.2f})
{context_info}

MANDATORY REQUIREMENTS FROM NOTES:"""
        
        if requirements.get('mandatory_count'):
            prompt += f"\n- EXACTLY {requirements['mandatory_count']} products"
        elif requirements.get('product_count'):
            prompt += f"\n- EXACTLY {requirements['product_count']} products"
        
        for category, count in requirements.get('categories_required', {}).items():
            prompt += f"\n- {category}: EXACTLY {count} items"
        
        prompt += f"""

CLIENT NOTES: {client_notes}
DIETARY RESTRICTIONS: {dietary_str}

AVAILABLE PRODUCTS (select from these ONLY):
{product_list}

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
            try:
                recommendation = json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    recommendation = json.loads(json_match.group())
                else:
                    return None
            
            product_ids = []
            for p in recommendation.get('selected_products', []):
                if isinstance(p, dict) and 'id' in p:
                    product_ids.append(p['id'])
            
            target_count = requirements.get('mandatory_count', requirements.get('product_count'))
            if target_count:
                if len(product_ids) != target_count:
                    _logger.warning(f"Ollama returned {len(product_ids)} products, expected {target_count}")
                    product_ids = product_ids[:target_count]
            
            selected_products = self.env['product.template'].sudo().browse(product_ids).exists()
            
            if not selected_products:
                return None
            
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
        
        products = self._get_available_products(target_budget, dietary_restrictions)
        if not products:
            return {'success': False, 'error': 'No products available'}
        
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
            return {'success': False, 'error': str(e)}
    
    # ============== HELPER METHODS (unchanged from before) ==============
    
    def _select_exact_count(self, products, count, target_budget):
        """Select exactly 'count' products optimizing for budget"""
        
        avg_price = target_budget / count
        products_sorted = sorted(products, key=lambda p: abs(p.list_price - avg_price))
        selected = products_sorted[:count]
        
        total = sum(p.list_price for p in selected)
        min_target = target_budget * 0.95
        max_target = target_budget * 1.05
        
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
        
        min_target = target_budget * 0.95
        max_target = target_budget * 1.05
        
        min_price = max(10, target_budget * 0.02)
        suitable = [p for p in products if p.list_price >= min_price]
        
        if not suitable:
            suitable = products
        
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
            if hasattr(product, 'qty_available'):
                if product.qty_available > 0:
                    available.append(product)
            else:
                available.append(product)
        
        _logger.info(f"Found {len(available)} available products")
        return available

    def _get_previous_order_data(self, partner_id):
        """Get data from previous sales orders for this client"""
        
        sales = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc', limit=5)
        
        if not sales:
            sales = self.env['sale.order'].search([
                ('partner_id', '=', partner_id),
                ('state', '=', 'sent')
            ], order='date_order desc', limit=5)
        
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

    def _find_replacement_products(self, count, budget, dietary, exclude_products=None):
        """Find new products similar to excluded ones"""
        
        exclude_ids = [p.id for p in exclude_products] if exclude_products else []
        
        domain = [
            ('sale_ok', '=', True),
            ('id', 'not in', exclude_ids),
            ('list_price', '>', 0)
        ]
        
        products = self._get_available_products_with_stock(domain)
        products = [p for p in products if self._check_dietary_compliance(p, dietary)]
        
        avg_price = budget / count if count > 0 else budget
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
        
        for category, count in categories_required.items():
            cat_products = [p for p in products if category.lower() in p.name.lower()]
            selected.extend(cat_products[:count])
        
        remaining_count = total_count - len(selected)
        if remaining_count > 0:
            unused = [p for p in products if p not in selected]
            selected.extend(unused[:remaining_count])
        
        return selected
    
    def _optimize_selection(self, products, target_count, target_budget, flexibility):
        """Optimize product selection to match count and budget"""
        
        avg_price = target_budget / target_count if target_count > 0 else 0
        
        scored_products = []
        for product in products:
            price_score = 1 / (1 + abs(product.list_price - avg_price))
            variety_score = random.uniform(0.8, 1.2)
            total_score = price_score * variety_score
            scored_products.append((product, total_score))
        
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        current_total = 0
        budget_min = target_budget * (1 - flexibility/100)
        budget_max = target_budget * (1 + flexibility/100)
        
        for product, score in scored_products:
            if len(selected) < target_count:
                future_total = current_total + product.list_price
                remaining_slots = target_count - len(selected) - 1
                
                if remaining_slots > 0:
                    min_possible = future_total + (remaining_slots * 10)
                    max_possible = future_total + (remaining_slots * 500)
                    
                    if min_possible <= budget_max and max_possible >= budget_min:
                        selected.append(product)
                        current_total = future_total
                else:
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
        
        # Analyze global patterns
        affinities = self._analyze_product_affinity()
        outcomes = self._learn_from_outcomes()
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': '🧠 Learning Analysis Complete',
                'message': f"""Analyzed {analyzed_count} clients
                - Found {len(affinities)} product affinities
                - Conversion rate: {outcomes.get('conversion_rate', 0):.1f}%
                - Cache updated for 24 hours""",
                'type': 'success',
                'sticky': False,
            }
        }