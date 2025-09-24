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
    _description = 'Ollama-Powered Gift Recommendation Engine with Business Rules and Advanced Learning'
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
        """Main generation method with intelligent multi-source merging and business rules"""
        
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
        patterns = learning_data.get('patterns') if learning_data else None
        seasonal = learning_data.get('seasonal') if learning_data else None
        
        # Ensure patterns is never None
        if not patterns:
            patterns = {
                'total_orders': 0,
                'avg_order_value': 0,
                'preferred_categories': {},
                'product_frequency': {},
                'budget_trend': 'stable',
                'avg_product_count': 0,
                'favorite_products': [],
                'never_repeated_products': [],
                'preferred_price_range': {'min': 0, 'max': 0, 'avg': 0}
            }
            _logger.info("📊 HISTORY DATA: No historical data available (new customer)")
        else:
            _logger.info(f"📊 HISTORY DATA: {patterns.get('total_orders', 0)} orders, Avg €{patterns.get('avg_order_value', 0):.2f}")
        
        if not seasonal:
            seasonal = {
                'current_season': self._get_current_season(datetime.now().month),
                'seasonal_data': {},
                'seasonal_favorites': []
            }
        
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
            'similar_clients': learning_data.get('similar_clients', []) if learning_data else [],
            'previous_sales': previous_sales,
            'requirements_merged': True
        }
        
        # 6. DETERMINE STRATEGY (Enhanced with business rules)
        strategy = self._determine_generation_strategy(
            previous_sales, patterns, final_requirements, client_notes
        )
        _logger.info(f"🎯 GENERATION STRATEGY: {strategy}")
        
        # 7. EXECUTE GENERATION
        result = None
        
        # Check for business rules applicability first
        last_year_products = self._get_last_year_products(partner_id)
        if last_year_products and strategy in ['8020_rule', 'pattern_based']:
            # Apply business rules THEN enforce requirements
            result = self._apply_business_rules_with_enforcement(
                partner, last_year_products, final_requirements, 
                client_notes, generation_context
            )
        elif strategy == '8020_rule':
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

    # ================== BUSINESS RULES INTEGRATION ==================
    
    def _apply_business_rules_with_enforcement(self, partner, last_products, 
                                              requirements, notes, context):
        """Apply business rules R1-R6 THEN enforce all requirements"""
        
        _logger.info("🔧 Applying Business Rules R1-R6 with requirement enforcement...")
        
        # 1. Apply business rules transformation
        try:
            rules_engine = self.env['business.rules.engine'].sudo()
            transformation = rules_engine.apply_composition_rules(
                partner.id,
                datetime.now().year,
                last_products
            )
        except:
            _logger.warning("Business rules engine not available, using alternative approach")
            transformation = self._apply_basic_transformation(last_products)
        
        if not transformation.get('products'):
            _logger.warning("⚠️ Business rules produced no products, falling back to pattern generation")
            return self._generate_from_patterns_enhanced(
                partner, requirements, notes, context
            )
        
        # Extract locked attributes from transformation (experience items, foie presence)
        locked_attributes = transformation.get('locked_attributes', {}) or {}
        experience_has_foie = bool(locked_attributes.get('experience_has_foie', False))
        
        # 2. Apply dietary filters
        filtered_products = self._filter_products_by_dietary(
            transformation['products'], requirements.get('dietary', [])
        )
        
        # 2a. Enforce Tokaji ↔ Foie pairing if needed (Tokaji requires Foie)
        filtered_products = self._ensure_tokaji_foie_pairing(
            filtered_products, experience_has_foie, requirements.get('dietary', [])
        )
        
        # 3. CRITICAL: Enforce requirements from notes
        if requirements.get('product_count') and requirements.get('enforce_count'):
            _logger.info(f"📋 Enforcing exact product count: {requirements['product_count']}")
            filtered_products = self._enforce_exact_product_count(
                filtered_products, requirements['product_count'], requirements['budget']
            )
        else:
            # If last year data exists, preserve same number of items by default
            if last_products:
                target_count = len(last_products)
                _logger.info(f"📋 Defaulting to last year's product count: {target_count}")
                filtered_products = self._enforce_exact_product_count(
                    filtered_products, target_count, requirements['budget']
                )
        
        # 4. Apply budget optimization
        optimized_products = self._smart_optimize_selection(
            filtered_products,
            requirements['product_count'],
            requirements['budget'],
            requirements['budget_flexibility'],
            requirements['enforce_count'],
            {**context, 'locked_attributes': locked_attributes}
        )
        
        # 4a. Enforce same category counts as last year, when data exists
        if last_products:
            target_counts = self._compute_category_counts(last_products)
            optimized_products = self._enforce_category_counts(
                optimized_products,
                target_counts,
                requirements['budget'],
                requirements.get('dietary', []),
                {**context, 'locked_attributes': locked_attributes}
            )

        # 4b. Strict budget guardrail ±5% at the end
        optimized_products = self._enforce_budget_guardrail(
            optimized_products,
            requirements['budget'],
            tolerance=0.05,
            dietary=requirements.get('dietary', []),
            context={**context, 'locked_attributes': locked_attributes}
        )

        # 5. Calculate total cost
        total_cost = sum(p.list_price for p in optimized_products)
        
        # 6. Create composition
        try:
            # Build comprehensive reasoning
            rule_summary = self._build_business_rules_summary(
                transformation.get('rule_applications', []),
                requirements,
                optimized_products,
                total_cost
            )
            
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': requirements['budget'],
                'target_year': fields.Date.today().year,
                'product_ids': [(6, 0, [p.id for p in optimized_products])],
                'dietary_restrictions': ', '.join(requirements.get('dietary', [])),
                'client_notes': notes,
                'generation_method': 'business_rules_enforced',
                'composition_type': requirements.get('composition_type', 'custom'),
                'confidence_score': 0.95,
                'ai_reasoning': rule_summary
            })
            
            composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': optimized_products,
                'total_cost': total_cost,
                'product_count': len(optimized_products),
                'rules_applied': transformation.get('rule_applications', []),
                'method': 'business_rules_with_enforcement',
                'message': f"Applied {len(transformation.get('rule_applications', []))} business rules + requirements"
            }
            
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _apply_basic_transformation(self, last_products):
        """Basic transformation when business rules engine unavailable"""
        transformed = []
        
        for product in last_products:
            # Basic rule: Keep products with good history
            if self._has_stock(product):
                transformed.append(product)
        
        return {'products': transformed, 'rule_applications': []}

    # ===================== ENFORCEMENT HELPERS =====================
    def _compute_category_counts(self, products):
        counts = {
            'beverage': 0,
            'aperitif': 0,
            'foie': 0,
            'canned': 0,
            'charcuterie': 0,
            'sweet': 0,
            'other': 0,
        }
        for p in products:
            cat = (getattr(p, 'lebiggot_category', '') or '').lower()
            bevfam = getattr(p, 'beverage_family', '') or ''
            name = (getattr(p, 'name', '') or '').lower()
            if bevfam in ['cava', 'champagne', 'vermouth', 'tokaj', 'tokaji', 'wine', 'red_wine', 'white_wine', 'rose_wine', 'beer', 'spirits_high']:
                counts['beverage'] += 1
            elif cat == 'foie_gras' or 'foie' in name:
                counts['foie'] += 1
            elif cat in ['preserves'] or any(k in name for k in ['conserva', 'lata', 'anchoa', 'bonito', 'sardina', 'mejillón', 'ventresca']):
                counts['canned'] += 1
            elif cat in ['charcuterie', 'cheese']:
                counts['charcuterie'] += 1
            elif cat in ['sweets', 'chocolates']:
                counts['sweet'] += 1
            elif bevfam in ['spirits_high'] or any(k in name for k in ['vermouth', 'vermut', 'tokaji', 'beer', 'cerveza', 'whisky', 'gin', 'vodka', 'brandy', 'cognac', 'licor']):
                counts['aperitif'] += 1
            else:
                counts['other'] += 1
        return counts

    def _enforce_category_counts(self, products, target_counts, budget, dietary, context):
        if not products:
            return products
        locked_ids = set((context.get('locked_attributes') or {}).get('experience_item_ids', set()) or [])
        current_counts = self._compute_category_counts(products)
        if current_counts == target_counts:
            return products
        exclude_ids = [p.id for p in products] + list(locked_ids)
        pool = self._get_smart_product_pool(budget, dietary, {**context, 'exclude_ids': exclude_ids})
        def cat_of(p):
            tmp = self._compute_category_counts([p])
            for k in ['beverage', 'aperitif', 'foie', 'canned', 'charcuterie', 'sweet']:
                if tmp.get(k, 0) == 1:
                    return k
            return 'other'
        products_mut = list(products)
        for cat in ['beverage','aperitif','foie','canned','charcuterie','sweet']:
            deficit = max(0, target_counts.get(cat, 0) - current_counts.get(cat, 0))
            while deficit > 0:
                candidate = None
                for p in pool:
                    if p.id in exclude_ids:
                        continue
                    if cat_of(p) == cat and self._check_dietary_compliance(p, dietary):
                        candidate = p
                        break
                if not candidate:
                    break
                replace_idx = None
                for i, existing in enumerate(products_mut):
                    if existing.id in locked_ids:
                        continue
                    ex_cat = cat_of(existing)
                    if current_counts.get(ex_cat, 0) > target_counts.get(ex_cat, 0):
                        replace_idx = i
                        current_counts[ex_cat] -= 1
                        break
                if replace_idx is None:
                    break
                products_mut[replace_idx] = candidate
                exclude_ids.append(candidate.id)
                current_counts[cat] = current_counts.get(cat, 0) + 1
                deficit -= 1
        return products_mut

    def _enforce_budget_guardrail(self, products, budget, tolerance, dietary, context):
        if not products or budget <= 0:
            return products
        min_budget = budget * (1 - tolerance)
        max_budget = budget * (1 + tolerance)
        # Remove any zero or negative price items before enforcing
        products = [p for p in products if float(getattr(p, 'list_price', 0) or 0) > 0]
        total = sum(float(p.list_price) for p in products)
        if min_budget <= total <= max_budget:
            return products
        locked_ids = set((context.get('locked_attributes') or {}).get('experience_item_ids', set()) or [])
        exclude_ids = [p.id for p in products] + list(locked_ids)
        pool = self._get_smart_product_pool(budget, dietary, {**context, 'exclude_ids': exclude_ids})
        products_mut = list(products)
        pool_sorted_low = sorted(pool, key=lambda p: float(p.list_price))
        pool_sorted_high = list(reversed(pool_sorted_low))
        max_iters = 20
        it = 0
        while (total < min_budget or total > max_budget) and it < max_iters:
            it += 1
            replace_idx = None
            if total > max_budget:
                sorted_existing = sorted([(i, p) for i,p in enumerate(products_mut) if p.id not in locked_ids], key=lambda ip: float(ip[1].list_price), reverse=True)
                if not sorted_existing:
                    break
                replace_idx, to_replace = sorted_existing[0]
                candidate = next((p for p in pool_sorted_low if float(p.list_price) < float(to_replace.list_price) and self._check_dietary_compliance(p, dietary)), None)
            else:
                sorted_existing = sorted([(i, p) for i,p in enumerate(products_mut) if p.id not in locked_ids], key=lambda ip: float(ip[1].list_price))
                if not sorted_existing:
                    break
                replace_idx, to_replace = sorted_existing[0]
                candidate = next((p for p in pool_sorted_high if float(p.list_price) > float(to_replace.list_price) and self._check_dietary_compliance(p, dietary)), None)
            if not candidate:
                break
            total -= float(products_mut[replace_idx].list_price)
            products_mut[replace_idx] = candidate
            total += float(candidate.list_price)
            exclude_ids.append(candidate.id)
        return products_mut

    def _ensure_tokaji_foie_pairing(self, products, experience_has_foie, dietary):
        if experience_has_foie:
            return products
        has_tokaji = any(getattr(p, 'beverage_family', '') in ['tokaj', 'tokaji'] for p in products)
        has_foie = any((getattr(p, 'lebiggot_category', '') or '') == 'foie_gras' for p in products)
        if not has_tokaji or has_foie:
            return products
        tokaji_grades = [getattr(p, 'product_grade', None) for p in products if getattr(p, 'beverage_family', '') in ['tokaj','tokaji']]
        preferred_grade = tokaji_grades[0] if tokaji_grades and tokaji_grades[0] else None
        domain = [('lebiggot_category', '=', 'foie_gras'), ('active', '=', True), ('sale_ok', '=', True)]
        if preferred_grade:
            domain.append(('product_grade', '=', preferred_grade))
        foie_candidates = self.env['product.template'].sudo().search(domain)
        for foie in foie_candidates:
            if self._check_dietary_compliance(foie, dietary):
                _logger.info("🦆 Added Foie pairing for Tokaji rule")
                return products + [foie]
        return products
    
    def _enforce_exact_product_count(self, products, target_count, budget):
        """Enforce exact product count requirement"""
        
        if len(products) == target_count:
            return products
        
        if len(products) < target_count:
            # Need more products - find additional
            needed = target_count - len(products)
            additional = self._find_additional_products(
                needed, budget, exclude_ids=[p.id for p in products]
            )
            products.extend(additional)
        else:
            # Too many - intelligently reduce
            products = self._reduce_to_target_count(products, target_count, budget)
        
        return products[:target_count]  # Final safety
    
    def _find_additional_products(self, count_needed, budget, exclude_ids=None):
        """Find additional products to meet count requirement"""
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>', 0),
            ('list_price', '<=', budget * 0.3)
        ]
        
        if exclude_ids:
            domain.append(('id', 'not in', exclude_ids))
        
        products = self.env['product.template'].search(domain, limit=count_needed * 2)
        available = [p for p in products if self._has_stock(p)]
        
        return available[:count_needed]
    
    def _reduce_to_target_count(self, products, target_count, budget):
        """Intelligently reduce products to target count"""
        
        # Calculate ideal average price
        ideal_avg = budget / target_count
        
        # Score products by how close they are to ideal
        scored = []
        for product in products:
            score = 1 / (1 + abs(product.list_price - ideal_avg))
            scored.append((product, score))
        
        # Sort by score and take top products
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:target_count]]
    
    def _filter_products_by_dietary(self, products, dietary_restrictions):
        """Apply dietary restriction filters"""
        
        if not dietary_restrictions:
            return products
        
        filtered = []
        for product in products:
            if self._check_dietary_compliance(product, dietary_restrictions):
                filtered.append(product)
        
        _logger.info(f"🥗 Dietary filter: {len(products)} → {len(filtered)} products")
        return filtered
    
    def _build_business_rules_summary(self, rule_applications, requirements, 
                                     products, total_cost):
        """Build comprehensive summary including business rules and requirements"""
        
        summary_parts = []
        
        # Business rules section
        if rule_applications:
            summary_parts.append("📋 BUSINESS RULES APPLIED:")
            
            rules_grouped = defaultdict(list)
            for app in rule_applications:
                rules_grouped[app.get('rule', 'unknown')].append(app)
            
            rule_descriptions = {
                'R1': '🔁 R1 - Exact Repeats',
                'R2': '🔄 R2 - Wine Rotation',
                'R3': '🎁 R3 - Experience Swaps',
                'R4': '🥓 R4 - Charcuterie Repeats',
                'R5': '🦆 R5 - Foie Gras Rotation',
                'R6': '🍬 R6 - Sweets Rules'
            }
            
            for rule, apps in rules_grouped.items():
                if rule in rule_descriptions:
                    summary_parts.append(f"{rule_descriptions[rule]}: {len(apps)} applications")
        
        # Requirements enforcement section
        summary_parts.append("\n📊 REQUIREMENTS ENFORCEMENT:")
        summary_parts.append(f"- Products: {len(products)} (target: {requirements.get('product_count', 'flexible')})")
        summary_parts.append(f"- Budget: €{total_cost:.2f} (target: €{requirements['budget']:.2f})")
        
        variance = ((total_cost - requirements['budget']) / requirements['budget']) * 100
        summary_parts.append(f"- Variance: {variance:+.1f}%")
        
        if requirements.get('dietary'):
            summary_parts.append(f"- Dietary: {', '.join(requirements['dietary'])} ✓")
        
        if requirements.get('enforce_count'):
            if len(products) == requirements['product_count']:
                summary_parts.append(f"- ✅ Exact count requirement met")
            else:
                summary_parts.append(f"- ⚠️ Count variance: {len(products)} vs {requirements['product_count']}")
        
        return "\n".join(summary_parts)
    
    # ================== EXISTING ADVANCED METHODS (PRESERVED) ==================
    
    def _analyze_client_purchase_patterns(self, partner_id):
        """Analyze patterns across all client orders"""
        
        default_pattern = {
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
        
        all_orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc')
        
        if not all_orders:
            _logger.info(f"No historical orders found for partner {partner_id}")
            return default_pattern
        
        pattern_analysis = default_pattern.copy()
        pattern_analysis['total_orders'] = len(all_orders)
        
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
            
            if last_order_date:
                interval = (last_order_date - order.date_order).days
                pattern_analysis['order_intervals'].append(interval)
            last_order_date = order.date_order
            
            for line in order.order_line:
                if line.product_id:
                    product_tmpl = line.product_id.product_tmpl_id
                    product_counter[product_tmpl.id] += 1
                    all_product_prices.append(line.price_unit)
                    
                    if hasattr(product_tmpl, 'categ_id'):
                        category_counter[product_tmpl.categ_id.name] += 1
                    total_products += 1
        
        pattern_analysis['avg_order_value'] = total_value / len(all_orders) if all_orders else 0
        pattern_analysis['avg_product_count'] = total_products / len(all_orders) if all_orders else 0
        
        pattern_analysis['favorite_products'] = [
            prod_id for prod_id, count in product_counter.items() 
            if count >= 2
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
        
        if len(budgets_timeline) >= 3:
            recent_orders = budgets_timeline[:min(3, len(budgets_timeline))]
            older_orders = budgets_timeline[-min(3, len(budgets_timeline)):]
            recent_avg = sum(b[1] for b in recent_orders) / len(recent_orders)
            older_avg = sum(b[1] for b in older_orders) / len(older_orders)
            
            if recent_avg > older_avg * 1.2:
                pattern_analysis['budget_trend'] = 'increasing'
            elif recent_avg < older_avg * 0.8:
                pattern_analysis['budget_trend'] = 'decreasing'
        
        if pattern_analysis['order_intervals']:
            pattern_analysis['avg_order_interval'] = sum(pattern_analysis['order_intervals']) / len(pattern_analysis['order_intervals'])
        
        return pattern_analysis

    def _analyze_seasonal_preferences(self, partner_id):
        """Identify seasonal patterns in purchases"""
        
        default_seasonal = {
            'current_season': self._get_current_season(datetime.now().month),
            'seasonal_data': {
                'spring': {'products': [], 'categories': [], 'avg_value': 0},
                'summer': {'products': [], 'categories': [], 'avg_value': 0},
                'autumn': {'products': [], 'categories': [], 'avg_value': 0},
                'winter': {'products': [], 'categories': [], 'avg_value': 0},
                'christmas': {'products': [], 'categories': [], 'avg_value': 0}
            },
            'seasonal_favorites': [],
            'seasonal_categories': []
        }
        
        orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ])
        
        if not orders:
            _logger.info(f"No orders found for seasonal analysis - partner {partner_id}")
            return default_seasonal
        
        seasonal_data = default_seasonal['seasonal_data'].copy()
        season_order_values = defaultdict(list)
        
        for order in orders:
            month = order.date_order.month
            
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
            
            for line in order.order_line:
                if line.product_id:
                    seasonal_data[season]['products'].append(line.product_id.product_tmpl_id.id)
                    if hasattr(line.product_id.product_tmpl_id, 'categ_id'):
                        seasonal_data[season]['categories'].append(line.product_id.product_tmpl_id.categ_id.name)
        
        for season, values in season_order_values.items():
            if values:
                seasonal_data[season]['avg_value'] = sum(values) / len(values)
        
        current_month = datetime.now().month
        current_season = self._get_current_season(current_month)
        
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
    
    def _merge_all_requirements(self, notes_data, form_data, patterns, seasonal):
        """Intelligently merge requirements from all sources with proper error handling"""
        
        merged = {
            'budget': 100.0,
            'budget_source': 'default',
            'budget_flexibility': 15,
            'product_count': 5,
            'count_source': 'default',
            'enforce_count': False,
            'dietary': [],
            'dietary_source': 'none',
            'composition_type': 'custom',
            'type_source': 'default',
            'categories_required': {},
            'categories_excluded': [],
            'specific_products': [],
            'special_instructions': [],
            'seasonal_hint': None,
            'preferred_price_range': None
        }
        
        # 1. MERGE BUDGET (Priority: Notes > Form > History > Default)
        if notes_data and notes_data.get('budget_override') and notes_data['budget_override'] > 0:
            merged['budget'] = float(notes_data['budget_override'])
            merged['budget_source'] = 'notes (override)'
            _logger.info(f"💰 Budget from NOTES: €{merged['budget']:.2f}")
        elif form_data and form_data.get('budget') and form_data['budget'] > 0:
            merged['budget'] = float(form_data['budget'])
            merged['budget_source'] = 'form'
            _logger.info(f"💰 Budget from FORM: €{merged['budget']:.2f}")
        elif patterns and patterns.get('avg_order_value') and patterns['avg_order_value'] > 0:
            historical_budget = float(patterns['avg_order_value'])
            trend = patterns.get('budget_trend', 'stable')
            if trend == 'increasing':
                historical_budget *= 1.1
                merged['budget_source'] = 'history (increasing trend +10%)'
            elif trend == 'decreasing':
                historical_budget *= 0.95
                merged['budget_source'] = 'history (decreasing trend -5%)'
            else:
                merged['budget_source'] = 'history (stable trend)'
            merged['budget'] = max(100.0, historical_budget)
            _logger.info(f"💰 Budget from HISTORY: €{merged['budget']:.2f} ({trend} trend)")
        else:
            merged['budget'] = 1000.0
            merged['budget_source'] = 'default'
            _logger.info(f"💰 Using DEFAULT budget: €{merged['budget']:.2f}")
        
        merged['budget'] = max(100.0, float(merged['budget']))
        
        # 2. MERGE PRODUCT COUNT (Priority: Notes > History > Calculated)
        if notes_data and notes_data.get('product_count') and notes_data['product_count'] > 0:
            merged['product_count'] = int(notes_data['product_count'])
            merged['enforce_count'] = True
            merged['count_source'] = 'notes (strict enforcement)'
            _logger.info(f"📦 Product count from NOTES: {merged['product_count']} (STRICT)")
        elif notes_data and notes_data.get('mandatory_count') and notes_data['mandatory_count'] > 0:
            merged['product_count'] = int(notes_data['mandatory_count'])
            merged['enforce_count'] = True
            merged['count_source'] = 'notes (mandatory)'
            _logger.info(f"📦 Mandatory count from NOTES: {merged['product_count']}")
        elif patterns and patterns.get('avg_product_count') and patterns['avg_product_count'] > 0:
            merged['product_count'] = max(1, int(round(patterns['avg_product_count'])))
            merged['enforce_count'] = False
            merged['count_source'] = 'history (flexible)'
            _logger.info(f"📦 Product count from HISTORY: {merged['product_count']} (flexible)")
        else:
            avg_price = 80.0
            if patterns and patterns.get('preferred_price_range'):
                price_range = patterns['preferred_price_range']
                if price_range.get('avg') and price_range['avg'] > 0:
                    avg_price = float(price_range['avg'])
                elif price_range.get('min') and price_range.get('max'):
                    min_p = float(price_range.get('min', 50))
                    max_p = float(price_range.get('max', 150))
                    if min_p > 0 and max_p > 0:
                        avg_price = (min_p + max_p) / 2
            avg_price = max(10.0, avg_price)
            calculated_count = int(merged['budget'] / avg_price) if avg_price > 0 else 12
            merged['product_count'] = max(5, min(25, calculated_count))
            merged['enforce_count'] = False
            merged['count_source'] = f'calculated (€{merged["budget"]:.0f}/€{avg_price:.0f})'
            _logger.info(f"📦 Product count CALCULATED: {merged['product_count']} (flexible)")
        
        merged['product_count'] = max(1, int(merged['product_count']))
        
        # 3. MERGE DIETARY RESTRICTIONS (Union of all sources)
        dietary_set = set()
        
        if notes_data and notes_data.get('dietary'):
            dietary_items = notes_data['dietary']
            if isinstance(dietary_items, list):
                dietary_set.update(dietary_items)
            elif isinstance(dietary_items, str):
                dietary_set.add(dietary_items)
            merged['dietary_source'] = 'notes'
            _logger.info(f"🥗 Dietary from NOTES: {notes_data['dietary']}")
        
        if form_data and form_data.get('dietary'):
            dietary_items = form_data['dietary']
            if isinstance(dietary_items, list):
                dietary_set.update(dietary_items)
            elif isinstance(dietary_items, str):
                dietary_set.add(dietary_items)
            if merged['dietary_source'] == 'notes':
                merged['dietary_source'] = 'notes+form (combined)'
            else:
                merged['dietary_source'] = 'form'
            _logger.info(f"🥗 Dietary from FORM: {form_data['dietary']}")
        
        merged['dietary'] = list(dietary_set)
        if not merged['dietary']:
            merged['dietary_source'] = 'none'
            _logger.info("🥗 No dietary restrictions")
        
        # 4. MERGE COMPOSITION TYPE
        if notes_data and notes_data.get('composition_type'):
            merged['composition_type'] = notes_data['composition_type']
            merged['type_source'] = 'notes'
            _logger.info(f"🎁 Composition type from NOTES: {merged['composition_type']}")
        elif form_data and form_data.get('composition_type'):
            merged['composition_type'] = form_data['composition_type']
            merged['type_source'] = 'form'
            _logger.info(f"🎁 Composition type from FORM: {merged['composition_type']}")
        elif patterns and patterns.get('total_orders', 0) >= 3:
            if patterns.get('preferred_categories'):
                top_categories = list(patterns['preferred_categories'].keys())
                top_categories_str = ' '.join(str(cat) for cat in top_categories).lower()
                if any(word in top_categories_str for word in ['wine', 'vino', 'champagne', 'alcohol']):
                    merged['composition_type'] = 'hybrid'
                    merged['type_source'] = 'history (wine preference detected)'
                elif any(word in top_categories_str for word in ['experience', 'experiencia']):
                    merged['composition_type'] = 'experience'
                    merged['type_source'] = 'history (experience preference detected)'
                else:
                    merged['composition_type'] = 'custom'
                    merged['type_source'] = 'history (general products)'
            else:
                merged['composition_type'] = 'custom'
                merged['type_source'] = 'default'
        else:
            merged['composition_type'] = 'custom'
            merged['type_source'] = 'default'
        
        # 5. MERGE BUDGET FLEXIBILITY
        if notes_data and notes_data.get('budget_flexibility'):
            try:
                flex = float(notes_data['budget_flexibility'])
                merged['budget_flexibility'] = max(5, min(30, flex))
                _logger.info(f"📐 Flexibility from NOTES: {merged['budget_flexibility']}%")
            except (ValueError, TypeError):
                merged['budget_flexibility'] = 15
        else:
            merged['budget_flexibility'] = 15
        
        # 6. MERGE CATEGORY REQUIREMENTS
        if notes_data:
            if notes_data.get('categories_required'):
                merged['categories_required'] = notes_data['categories_required']
                _logger.info(f"📂 Categories required: {merged['categories_required']}")
            if notes_data.get('categories_excluded'):
                merged['categories_excluded'] = notes_data['categories_excluded']
                _logger.info(f"🚫 Categories excluded: {merged['categories_excluded']}")
            if notes_data.get('specific_products'):
                merged['specific_products'] = notes_data['specific_products']
                _logger.info(f"⭐ Specific products requested: {len(merged['specific_products'])} items")
            if notes_data.get('special_instructions'):
                merged['special_instructions'] = notes_data['special_instructions']
                _logger.info(f"📋 Special instructions: {len(merged['special_instructions'])} notes")
        
        # 7. ADD SEASONAL PREFERENCES
        if seasonal and not merged.get('categories_required'):
            current_season = seasonal.get('current_season')
            if current_season and seasonal.get('seasonal_data', {}).get(current_season):
                season_data = seasonal['seasonal_data'][current_season]
                if season_data.get('top_categories'):
                    merged['seasonal_hint'] = season_data['top_categories']
                    _logger.info(f"🌡️ Seasonal hint ({current_season}): {merged['seasonal_hint'][:3] if merged['seasonal_hint'] else 'None'}")
                if season_data.get('top_products'):
                    merged['seasonal_products'] = [p[0] for p in season_data['top_products'][:5]]
                    _logger.info(f"🌡️ Seasonal products to consider: {len(merged.get('seasonal_products', []))} items")
        
        # 8. ADD PRICE RANGE PREFERENCE
        if patterns and patterns.get('preferred_price_range'):
            price_range = patterns['preferred_price_range']
            if price_range.get('min') and price_range.get('max'):
                merged['preferred_price_range'] = price_range
                _logger.info(f"💰 Price range preference: €{price_range.get('min', 0):.2f} - €{price_range.get('max', 0):.2f}")
        
        # 9. CALCULATE FINAL BUDGET BOUNDS
        flexibility = float(merged['budget_flexibility'])
        budget = float(merged['budget'])
        merged['min_budget'] = budget * (1 - flexibility/100)
        merged['max_budget'] = budget * (1 + flexibility/100)
        
        # 10. LOG SUMMARY
        _logger.info("="*60)
        _logger.info("FINAL MERGED REQUIREMENTS SUMMARY:")
        _logger.info(f"  💰 Budget: €{merged['budget']:.2f} (±{flexibility}%)")
        _logger.info(f"     Range: €{merged['min_budget']:.2f} - €{merged['max_budget']:.2f}")
        _logger.info(f"     Source: {merged['budget_source']}")
        _logger.info(f"  📦 Products: {merged['product_count']} (enforce: {merged['enforce_count']})")
        _logger.info(f"     Source: {merged['count_source']}")
        _logger.info(f"  🎁 Type: {merged['composition_type']} (from: {merged['type_source']})")
        if merged['dietary']:
            _logger.info(f"  🥗 Dietary: {', '.join(merged['dietary'])} (from: {merged['dietary_source']})")
        _logger.info("="*60)
        
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
    
    def _determine_generation_strategy(self, previous_sales, patterns, requirements, notes):
        """Determine the best generation strategy"""
        notes_lower = notes.lower() if notes else ""
        
        # Check if business rules should apply (when we have last year data)
        if previous_sales and len(previous_sales) > 0:
            # Unless explicitly asked for all new products
            if 'all new' not in notes_lower and 'completely different' not in notes_lower:
                return '8020_rule'  # Will trigger business rules first
        
        if patterns and patterns.get('total_orders', 0) >= 3:
            return 'pattern_based'  # Will also check for business rules
        
        if not previous_sales:
            return 'similar_clients'
        
        return 'universal'
    
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
    
    def _smart_optimize_selection(self, products, target_count, budget, flexibility, enforce_strict, context):
        """FIXED: Ensure budget compliance with no zero-price products"""
        
        if not products:
            _logger.error("❌ No products available for selection")
            return []
        
        # CRITICAL FIX: Strictly filter out invalid prices
        valid_products = []
        for p in products:
            try:
                price = float(p.list_price)
                if price >= 5.0:  # Minimum €5 per product
                    valid_products.append(p)
            except:
                continue
        
        if not valid_products:
            _logger.error("❌ No valid priced products available after filtering")
            return []
        
        products = valid_products
        
        # Set realistic flexibility bounds
        flexibility = flexibility if flexibility else 15
        min_budget = budget * 0.85  # 85% of target
        max_budget = budget * 1.15  # 115% of target
        
        # Calculate realistic target count
        if not target_count or target_count <= 0:
            avg_price = budget / 12  # Assume average 12 products
            target_count = max(5, min(25, int(budget / avg_price)))
        
        _logger.info("="*70)
        _logger.info("💰 BUDGET OPTIMIZATION - FIXED VERSION")
        _logger.info(f"   🎯 Target Budget: €{budget:.2f}")
        _logger.info(f"   📊 Acceptable Range: €{min_budget:.2f} - €{max_budget:.2f}")
        _logger.info(f"   📦 Target Products: {target_count}")
        _logger.info(f"   🛒 Valid Products Available: {len(products)}")
        
        # Calculate what's possible
        products_by_price = sorted(products, key=lambda p: float(p.list_price))
        
        # STRATEGY: Build a balanced selection
        selected = []
        selected_ids = set()
        current_total = 0
        
        # Calculate ideal price per product
        ideal_price_per_product = budget / target_count
        
        # Group products by price ranges
        low_range = [p for p in products if float(p.list_price) <= ideal_price_per_product * 0.7]
        mid_range = [p for p in products if ideal_price_per_product * 0.7 < float(p.list_price) <= ideal_price_per_product * 1.3]
        high_range = [p for p in products if float(p.list_price) > ideal_price_per_product * 1.3]
        
        _logger.info(f"   Product distribution: Low={len(low_range)}, Mid={len(mid_range)}, High={len(high_range)}")
        
        # Build selection with balanced approach
        # 1. Start with some high-value items (30% of count)
        high_count = max(1, int(target_count * 0.3))
        for p in high_range[:high_count]:
            if p.id not in selected_ids and current_total + float(p.list_price) <= max_budget:
                selected.append(p)
                selected_ids.add(p.id)
                current_total += float(p.list_price)
                _logger.info(f"   💎 High: {p.name[:40]} | €{p.list_price:.2f}")
        
        # 2. Fill with mid-range products (50% of count)
        mid_count = max(2, int(target_count * 0.5))
        for p in mid_range[:mid_count]:
            if p.id not in selected_ids and current_total + float(p.list_price) <= max_budget:
                selected.append(p)
                selected_ids.add(p.id)
                current_total += float(p.list_price)
                _logger.info(f"   ➕ Mid: {p.name[:40]} | €{p.list_price:.2f}")
        
        # 3. Add low-range to reach count (20% of count)
        remaining_count = target_count - len(selected)
        for p in low_range[:remaining_count]:
            if p.id not in selected_ids and current_total + float(p.list_price) <= max_budget:
                selected.append(p)
                selected_ids.add(p.id)
                current_total += float(p.list_price)
                _logger.info(f"   ➕ Low: {p.name[:40]} | €{p.list_price:.2f}")
        
        # 4. If still under budget, add more expensive items
        if current_total < min_budget:
            _logger.warning(f"⚠️ Under budget: €{current_total:.2f} < €{min_budget:.2f}")
            
            # Sort all products by price descending
            all_products_sorted = sorted(products, key=lambda p: float(p.list_price), reverse=True)
            
            for p in all_products_sorted:
                if current_total >= min_budget:
                    break
                
                price = float(p.list_price)
                if price > 0 and current_total + price <= max_budget:
                    selected.append(p)
                    current_total += price
                    _logger.info(f"   🔄 Adding: {p.name[:40]} | €{price:.2f} → €{current_total:.2f}")
        
        # 5. Final adjustment if still not meeting budget
        if current_total < min_budget:
            _logger.error(f"🚨 CRITICAL: Cannot meet minimum budget with available products")
            _logger.error(f"   Current: €{current_total:.2f}, Required: €{min_budget:.2f}")
            
            # Emergency: Allow duplicates of expensive items
            expensive_items = sorted(selected, key=lambda p: float(p.list_price), reverse=True)[:5]
            
            for p in expensive_items:
                if current_total >= min_budget:
                    break
                price = float(p.list_price)
                if price > 0 and current_total + price <= max_budget:
                    selected.append(p)  # Allow duplicate
                    current_total += price
                    _logger.warning(f"   🚨 DUPLICATE: {p.name[:40]} | €{price:.2f} → €{current_total:.2f}")
        
        # Final validation
        final_total = sum(float(p.list_price) for p in selected)
        variance_pct = ((final_total / budget) - 1) * 100
        
        _logger.info("="*70)
        _logger.info("📊 FINAL RESULT:")
        _logger.info(f"   📦 Products: {len(selected)}")
        _logger.info(f"   💰 Total: €{final_total:.2f}")
        _logger.info(f"   📈 Variance: {variance_pct:+.1f}%")
        
        if min_budget <= final_total <= max_budget:
            _logger.info("✅ SUCCESS: Budget requirement ACHIEVED!")
        else:
            _logger.error("❌ FAILURE: Budget requirement NOT MET!")
            
            # Log details about what went wrong
            _logger.error(f"   Available products price range: €{min(float(p.list_price) for p in products):.2f} - €{max(float(p.list_price) for p in products):.2f}")
            _logger.error(f"   Selected products: {[f'{p.name[:20]} (€{p.list_price:.2f})' for p in selected[:5]]}")
        
        _logger.info("="*70)
        
        return selected
    
    # ================== NOTES PARSING WITH OLLAMA ==================
    
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
    
    # [Continue with remaining methods: _get_smart_product_pool, _check_dietary_compliance, etc.]
    # All methods remain the same as in the original implementation
    
    def _get_smart_product_pool(self, budget, dietary, context):
        """Get intelligently filtered product pool - STRICTLY EXCLUDING ZERO-PRICE ITEMS"""
        
        patterns = context.get('patterns', {})
        
        # CRITICAL FIX: Set meaningful minimum price threshold
        min_price = max(10.0, budget * 0.02)  # At least €10, never less
        max_price = min(budget * 0.4, 500.0)  # Cap at €500 or 40% of budget
        
        # If patterns have price range, use it but enforce minimums
        if patterns and patterns.get('preferred_price_range'):
            pattern_min = patterns['preferred_price_range'].get('min', min_price)
            pattern_max = patterns['preferred_price_range'].get('max', max_price)
            min_price = max(10.0, pattern_min)  # NEVER allow below €10
            max_price = min(budget * 0.4, pattern_max)
        
        domain = [
            ('sale_ok', '=', True),
            ('active', '=', True),
            ('list_price', '>=', min_price),  # Use >= with meaningful minimum
            ('list_price', '<=', max_price),
            ('default_code', '!=', False),
            ('qty_available', '>', 0),  # Add stock check in domain
        ]
        
        # Add dietary filters
        if dietary and 'halal' in dietary:
            domain.extend([
                '|', '|',
                ('categ_id.complete_name', 'not ilike', 'IBERICOS'),
                ('categ_id.complete_name', 'not ilike', 'ALCOHOL'),
                ('name', 'not ilike', 'pork')
            ])
        
        # Exclude explicitly provided ids
        exclude_ids = list(set((context or {}).get('exclude_ids', []) or []))
        locked_attrs = (context or {}).get('locked_attributes') or {}
        exp_item_ids = list(set(locked_attrs.get('experience_item_ids', set()) or []))
        exclude_ids.extend(exp_item_ids)
        if exclude_ids:
            domain.append(('id', 'not in', exclude_ids))
        
        products = self.env['product.template'].sudo().search(domain, limit=1000)
        
        # CRITICAL: Triple-check price validation
        valid_products = []
        for p in products:
            try:
                price = float(p.list_price)
                if price >= min_price and price <= max_price and self._has_stock(p):
                    valid_products.append(p)
            except (ValueError, TypeError):
                continue
        
        # Convert back to recordset
        valid_products = self.env['product.template'].browse([p.id for p in valid_products])
        
        # Add randomization
        if len(valid_products) > 20:
            import random
            product_list = list(valid_products)
            random.shuffle(product_list)
            valid_products = self.env['product.template'].browse([p.id for p in product_list])
        
        _logger.info(f"Smart pool: {len(valid_products)} VALID products (€{min_price:.2f} - €{max_price:.2f})")
        
        # CRITICAL: If no products found in range, expand search
        if len(valid_products) < 10:
            _logger.warning(f"⚠️ Only {len(valid_products)} products found, expanding search...")
            
            expanded_domain = [
                ('sale_ok', '=', True),
                ('active', '=', True),
                ('list_price', '>=', 5.0),  # Lower minimum to €5
                ('list_price', '<=', budget * 0.6),  # Increase to 60% of budget
                ('qty_available', '>', 0),
            ]
            
            if exclude_ids:
                expanded_domain.append(('id', 'not in', exclude_ids))
            
            expanded_products = self.env['product.template'].sudo().search(expanded_domain, limit=500)
            
            for p in expanded_products:
                if p not in valid_products:
                    try:
                        price = float(p.list_price)
                        if price >= 5.0 and self._has_stock(p):
                            valid_products |= p
                    except:
                        continue
            
            _logger.info(f"Expanded pool: {len(valid_products)} total products")
        
        return valid_products
    
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
                if hasattr(product, 'is_iberian_product') and product.is_iberian_product:
                    return False
            elif restriction == 'vegan':
                if hasattr(product, 'is_vegan') and not product.is_vegan:
                    return False
            elif restriction == 'vegetarian':
                # Basic heuristic: allow if vegan or not meat/foie/charcuterie
                cat = (getattr(product, 'lebiggot_category', '') or '').lower()
                if cat in ['charcuterie', 'foie_gras']:
                    return False
            elif restriction == 'gluten_free':
                if hasattr(product, 'contains_gluten') and product.contains_gluten:
                    return False
            elif restriction in ['non_alcoholic', 'no_alcohol']:
                if hasattr(product, 'contains_alcohol') and product.contains_alcohol:
                    return False
            elif restriction == 'no_pork':
                if hasattr(product, 'contains_pork') and product.contains_pork:
                    return False
            elif restriction == 'no_iberian':
                if hasattr(product, 'is_iberian_product') and product.is_iberian_product:
                    return False
        
        return True
    
    def _has_stock(self, product):
        """Enhanced stock check - verify product actually has stock"""
        try:
            # Method 1: Check qty_available field directly
            if hasattr(product, 'qty_available'):
                qty = float(product.qty_available)
                if qty > 0:
                    return True
            
            # Method 2: Check virtual_available (includes incoming)
            if hasattr(product, 'virtual_available'):
                qty = float(product.virtual_available)
                if qty > 0:
                    return True
            
            # Method 3: Check stock quants
            for variant in product.product_variant_ids:
                stock_quants = self.env['stock.quant'].sudo().search([
                    ('product_id', '=', variant.id),
                    ('location_id.usage', '=', 'internal'),
                    ('quantity', '>', 0)
                ])
                if stock_quants:
                    return True
            
            return False
            
        except Exception as e:
            _logger.warning(f"Stock check error for {product.name}: {e}")
            return False
    
    def _enforce_exact_count(self, selected, all_products, exact_count, budget):
        """Enforce exact product count no matter what"""
        
        if len(selected) == exact_count:
            return selected
        
        if len(selected) < exact_count:
            # Add products
            remaining_needed = exact_count - len(selected)
            available = [p for p in all_products if p not in selected]
            available.sort(key=lambda p: p.list_price)
            selected.extend(available[:remaining_needed])
            _logger.info(f"➕ Added {remaining_needed} products to meet count requirement")
        
        elif len(selected) > exact_count:
            # Remove products
            excess = len(selected) - exact_count
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
    
    def _find_complementary_products(self, count, budget, dietary, exclude_ids, context):
        """Find complementary new products"""
        
        patterns = context.get('patterns', {})
        seasonal = context.get('seasonal', {})
        
        avg_price = budget / count if count > 0 else 50
        min_price = max(5, avg_price * 0.3)
        max_price = avg_price * 2
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
            ('id', 'not in', exclude_ids)
        ]
        
        if dietary:
            if 'halal' in dietary:
                if 'is_halal_compatible' in self.env['product.template']._fields:
                    domain.append(('is_halal_compatible', '!=', False))
                if 'contains_pork' in self.env['product.template']._fields:
                    domain.append(('contains_pork', '=', False))
                if 'contains_alcohol' in self.env['product.template']._fields:
                    domain.append(('contains_alcohol', '=', False))
        
        products = self.env['product.template'].sudo().search(domain, limit=500)
        available = [p for p in products if self._has_stock(p)]
        
        scored = []
        for product in available:
            score = 1.0
            if hasattr(product, 'categ_id'):
                cat_name = product.categ_id.name
                if patterns and cat_name not in patterns.get('preferred_categories', {}):
                    score += 0.5
            if seasonal and product.id in seasonal.get('seasonal_favorites', []):
                score += 1.0
            price_diff = abs(product.list_price - avg_price)
            price_score = 1 / (1 + price_diff/avg_price)
            score += price_score
            scored.append((product, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        current_total = 0
        
        for product, score in scored:
            if len(selected) >= count:
                break
            
            if current_total + product.list_price <= budget * 1.1:  # Allow 10% flexibility
                selected.append(product)
                current_total += product.list_price
        
        return selected
    
    # [Continue with remaining methods exactly as they were]
    # All methods remain exactly the same as the original
    
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
    
    def _get_last_year_products(self, partner_id):
        """Get products from last year's composition or orders"""
        
        # Try gift compositions first
        last_composition = self.env['gift.composition'].search([
            ('partner_id', '=', partner_id),
            ('target_year', '=', datetime.now().year - 1)
        ], limit=1)
        
        if last_composition:
            return last_composition.product_ids
        
        # Try sales orders from last year
        last_year = datetime.now().year - 1
        orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done']),
            ('date_order', '>=', f'{last_year}-01-01'),
            ('date_order', '<=', f'{last_year}-12-31')
        ])
        
        products = []
        for order in orders:
            for line in order.order_line:
                if line.product_id and line.product_id.product_tmpl_id:
                    products.append(line.product_id.product_tmpl_id)
        
        return products
    
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
    
    def _find_similar_clients(self, partner_id, limit=5):
        """Find clients with similar purchase patterns"""
        
        target_patterns = self._analyze_client_purchase_patterns(partner_id)
        
        if not target_patterns:
            return []
        
        all_clients = self.env['sale.order'].read_group(
            [('state', 'in', ['sale', 'done'])],
            ['partner_id'],
            ['partner_id']
        )
        
        similar_clients = []
        
        for client_data in all_clients[:20]:  # Limit for performance
            other_partner_id = client_data['partner_id'][0]
            if other_partner_id == partner_id:
                continue
            
            other_patterns = self._analyze_client_purchase_patterns(other_partner_id)
            if not other_patterns:
                continue
            
            similarity_score = self._calculate_similarity(target_patterns, other_patterns)
            
            if similarity_score > 0.6:  # 60% similarity threshold
                similar_clients.append({
                    'partner_id': other_partner_id,
                    'similarity': similarity_score,
                    'patterns': other_patterns
                })
        
        similar_clients.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_clients[:limit]
    
    def _calculate_similarity(self, patterns1, patterns2):
        """Calculate similarity score between two client patterns"""
        
        score = 0
        factors = 0
        
        avg1 = patterns1.get('avg_order_value', 0)
        avg2 = patterns2.get('avg_order_value', 0)
        if avg1 and avg2:
            budget_diff = abs(avg1 - avg2) / max(avg1, avg2)
            if budget_diff < 0.3:
                score += (1 - budget_diff)
            factors += 1
        
        count1 = patterns1.get('avg_product_count', 0)
        count2 = patterns2.get('avg_product_count', 0)
        if count1 and count2:
            count_diff = abs(count1 - count2) / max(count1, count2)
            if count_diff < 0.3:
                score += (1 - count_diff)
            factors += 1
        
        cats1 = set(patterns1.get('preferred_categories', {}).keys())
        cats2 = set(patterns2.get('preferred_categories', {}).keys())
        if cats1 and cats2:
            overlap = len(cats1.intersection(cats2)) / len(cats1.union(cats2))
            score += overlap
            factors += 1
        
        if patterns1.get('budget_trend') == patterns2.get('budget_trend'):
            score += 0.5
            factors += 0.5
        
        return score / factors if factors > 0 else 0
    
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
    
    def _build_comprehensive_reasoning(self, requirements, products, total_cost, budget, context):
        """Build detailed reasoning for the composition"""
        
        reasoning_parts = []
        reasoning_parts.append(f"📊 Generated {len(products)} products totaling €{total_cost:.2f}")
        
        variance = ((total_cost - budget) / budget) * 100
        reasoning_parts.append(f"💰 Budget variance: {variance:+.1f}%")
        
        if requirements.get('enforce_count') and requirements.get('product_count'):
            if len(products) == requirements['product_count']:
                reasoning_parts.append(f"✅ Met exact count requirement: {requirements['product_count']}")
            else:
                reasoning_parts.append(f"⚠️ Count mismatch: {len(products)} vs {requirements['product_count']} required")
        
        if requirements.get('dietary'):
            reasoning_parts.append(f"🥗 Dietary restrictions applied: {', '.join(requirements['dietary'])}")
        
        if context.get('patterns'):
            patterns = context['patterns']
            if patterns.get('favorite_products'):
                favorites_included = sum(1 for p in products if p.id in patterns['favorite_products'])
                if favorites_included > 0:
                    reasoning_parts.append(f"⭐ Included {favorites_included} favorite products from history")
            
            if patterns.get('budget_trend'):
                reasoning_parts.append(f"📈 Budget trend: {patterns['budget_trend']}")
        
        if requirements.get('categories_required'):
            reasoning_parts.append(f"📂 Category requirements: {requirements['categories_required']}")
        
        return "\n".join(reasoning_parts)
    
    def _validate_and_log_result(self, result, requirements):
        """Validate and log the result against requirements"""
        
        actual_count = result.get('product_count', 0)
        actual_cost = result.get('total_cost', 0)
        expected_count = requirements.get('product_count')
        expected_budget = requirements['budget']
        flexibility = requirements['budget_flexibility']
        
        if requirements.get('enforce_count') and expected_count:
            if actual_count == expected_count:
                _logger.info(f"✅ Count requirement MET: {actual_count} products")
            else:
                _logger.error(f"❌ Count requirement FAILED: {actual_count} != {expected_count}")
        
        min_budget = expected_budget * (1 - flexibility/100)
        max_budget = expected_budget * (1 + flexibility/100)
        
        if min_budget <= actual_cost <= max_budget:
            variance = ((actual_cost - expected_budget) / expected_budget) * 100
            _logger.info(f"✅ Budget requirement MET: €{actual_cost:.2f} ({variance:+.1f}% variance)")
        else:
            variance = ((actual_cost - expected_budget) / expected_budget) * 100
            _logger.error(f"❌ Budget requirement FAILED: €{actual_cost:.2f} ({variance:+.1f}% variance)")
        
        return result.get('compliant', False)
    
    # ================== ALTERNATIVE GENERATION METHODS ==================
    
    def _generate_from_similar_clients(self, partner, requirements, notes, context):
        """Generate based on similar clients when no direct history exists"""
        similar_clients = context.get('similar_clients', [])
        if not similar_clients:
            return self._generate_with_universal_enforcement(partner, requirements, notes, context)
        
        _logger.info(f"👥 Learning from {len(similar_clients)} similar clients")
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        
        product_popularity = {}
        for similar in similar_clients:
            similarity_weight = similar['similarity']
            patterns = similar['patterns']
            
            for prod_id in patterns.get('favorite_products', []):
                if prod_id not in product_popularity:
                    product_popularity[prod_id] = 0
                product_popularity[prod_id] += similarity_weight
        
        popular_product_ids = sorted(
            product_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:product_count * 2]
        
        products = []
        for prod_id, score in popular_product_ids:
            product = self.env['product.template'].browse(prod_id)
            if product.exists() and self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                products.append(product)
        
        if len(products) < product_count:
            additional = self._get_smart_product_pool(budget, dietary, context)
            products.extend(additional[:product_count - len(products)])
        
        selected = products[:product_count] if products else []
        total_cost = sum(p.list_price for p in selected)
        
        try:
            top_similar = similar_clients[0] if similar_clients else None
            reasoning = f"""Similar Client Pattern Generation:
- Based on {len(similar_clients)} similar clients
- Top match: {top_similar['similarity']*100:.0f}% similarity
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
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        composition_type = requirements['composition_type']
        
        _logger.info(f"📊 Generating from patterns: {product_count} products, €{budget:.2f}, type={composition_type}")
        
        products = self._get_smart_product_pool(budget, dietary, context)
        
        if not products:
            return {'success': False, 'error': 'No products available matching criteria'}
        
        selected = self._smart_optimize_selection(
            products, product_count, budget, 
            requirements['budget_flexibility'],
            requirements['enforce_count'],
            context
        )
        
        total_cost = sum(p.list_price for p in selected)
        
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
        
        self.cache_expiry = datetime.now() - timedelta(days=1)
        
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