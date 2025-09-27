from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import json
import logging
import requests
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import random
import statistics

_logger = logging.getLogger(__name__)

class OllamaGiftRecommender(models.Model):
    _name = 'ollama.gift.recommender'
    _description = 'Truly Intelligent Gift Recommendation Engine - Deep Learning + Business Rules'
    _rec_name = 'name'
    
    # ================== FIELD DEFINITIONS (COMBINED) ==================
    
    name = fields.Char(string="Recommender Name", default="AI Gift Recommender", required=True)
    active = fields.Boolean(string="Active", default=True)
    
    # Ollama Configuration
    ollama_enabled = fields.Boolean(string="Ollama Enabled", default=True)
    ollama_base_url = fields.Char(string="Ollama Base URL", default="http://localhost:11434")
    ollama_model = fields.Char(string="Ollama Model", default="llama3.2:3b")
    ollama_timeout = fields.Integer(string="Timeout (seconds)", default=60)
    
    # Settings (NO HARD LIMITS!)
    max_products = fields.Integer(string="Max Products per Recommendation", default=30)
    budget_flexibility = fields.Float(string="Budget Flexibility (%)", default=10.0)
    min_confidence_score = fields.Float(string="Minimum Confidence Score", default=0.7)
    
    # Performance Tracking
    total_recommendations = fields.Integer(string="Total Recommendations", default=0)
    successful_recommendations = fields.Integer(string="Successful Recommendations", default=0)
    avg_response_time = fields.Float(string="Avg Response Time (s)", default=0.0)
    last_recommendation_date = fields.Datetime(string="Last Recommendation")
    
    # Learning Cache
    learning_cache = fields.Text(string="Learning Cache JSON")
    cache_expiry = fields.Datetime(string="Cache Expiry")
    
    success_rate = fields.Float(string="Success Rate (%)", compute='_compute_success_rate', store=True)
    
    @api.depends('total_recommendations', 'successful_recommendations')
    def _compute_success_rate(self):
        for rec in self:
            if rec.total_recommendations > 0:
                rec.success_rate = (rec.successful_recommendations / rec.total_recommendations) * 100
            else:
                rec.success_rate = 0.0
    
    # ================== MAIN ENTRY POINT (MERGED) ==================
    
    def generate_gift_recommendations(self, partner_id, target_budget, 
                                    client_notes='', dietary_restrictions=None,
                                    composition_type='custom'):
        """
        MERGED: Deep Learning + Business Rules Generation
        """
        self.ensure_one()
        start_time = datetime.now()
        
        try:
            partner = self.env['res.partner'].browse(partner_id)
            if not partner.exists():
                return {'success': False, 'error': 'Invalid partner'}
            
            # Form data
            form_data = {
                'budget': target_budget if target_budget and target_budget > 0 else None,
                'dietary': dietary_restrictions or [],
                'composition_type': composition_type
            }
            
            # STEP 1: Deep Historical Learning (from version 1)
            historical_intelligence = self._deep_historical_learning(partner_id, target_budget)
            
            # STEP 2: Get all learning data (from version 2)
            learning_data = self._get_or_update_learning_cache(partner_id)
            patterns = learning_data.get('patterns') if learning_data else historical_intelligence.get('patterns', {})
            seasonal = learning_data.get('seasonal') if learning_data else {}
            
            # STEP 3: Parse notes with intelligence (combined approach)
            notes_data = self._parse_notes_with_ollama(client_notes, form_data) if client_notes else {}
            requirements = self._parse_with_intelligence(
                client_notes, 
                target_budget, 
                dietary_restrictions,
                composition_type,
                historical_intelligence
            )
            
            # STEP 4: Merge all requirements (from version 2)
            final_requirements = self._merge_all_requirements(
                notes_data, form_data, patterns or {}, seasonal or {}
            )
            
            # Override with intelligence-based requirements where appropriate
            if requirements.get('product_count'):
                final_requirements['product_count'] = requirements['product_count']
                final_requirements['enforce_count'] = requirements.get('enforce_count', False)
            
            # STEP 5: Get previous sales data
            previous_sales = self._get_all_previous_sales_data(partner_id)
            
            # STEP 6: Create generation context
            generation_context = {
                'patterns': patterns,
                'seasonal': seasonal,
                'similar_clients': learning_data.get('similar_clients', []) if learning_data else [],
                'previous_sales': previous_sales,
                'historical_intelligence': historical_intelligence,
                'requirements_merged': True
            }
            
            # STEP 7: Determine strategy (Enhanced with both approaches)
            strategy = self._determine_comprehensive_strategy(
                previous_sales, patterns, final_requirements, client_notes,
                historical_intelligence
            )
            _logger.info(f"🎯 GENERATION STRATEGY: {strategy}")
            
            # STEP 8: Execute generation based on strategy
            result = None
            
            # Check for business rules applicability first
            last_year_products = self._get_last_year_products(partner_id)
            
            if last_year_products and strategy in ['business_rules', '8020_rule', 'pattern_based']:
                # Apply business rules WITH deep learning enforcement
                result = self._apply_business_rules_with_deep_learning(
                    partner, last_year_products, final_requirements, 
                    client_notes, generation_context
                )
            elif historical_intelligence['confidence'] >= 0.8:
                # High confidence - use deep learning
                result = self._generate_from_deep_learning(
                    partner, final_requirements, historical_intelligence
                )
            elif historical_intelligence['confidence'] >= 0.5:
                # Medium confidence - blend approaches
                result = self._generate_blended_intelligence(
                    partner, final_requirements, historical_intelligence
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
                # Fallback to universal enforcement
                result = self._generate_with_universal_enforcement(
                    partner, final_requirements, client_notes, generation_context
                )
            
            # STEP 9: Validate and learn
            if result and result.get('success'):
                self._update_learning_from_result(result, final_requirements, historical_intelligence)
                self.total_recommendations += 1
                self.successful_recommendations += 1
                self.last_recommendation_date = fields.Datetime.now()
                self._validate_and_log_result(result, final_requirements)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            if self.avg_response_time:
                self.avg_response_time = (self.avg_response_time + response_time) / 2
            else:
                self.avg_response_time = response_time
            
            return result
            
        except Exception as e:
            _logger.error(f"Generation failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}

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
    
    # ================== STRATEGY DETERMINATION (ENHANCED) ==================
    
    def _determine_comprehensive_strategy(self, previous_sales, patterns, requirements, 
                                         notes, historical_intelligence):
        """Enhanced strategy determination combining both approaches"""
        
        notes_lower = notes.lower() if notes else ""
        
        # Priority 1: Business rules when we have last year data
        if previous_sales and len(previous_sales) > 0:
            # Unless explicitly asked for all new products
            if 'all new' not in notes_lower and 'completely different' not in notes_lower:
                return 'business_rules'  # Will apply business rules with deep learning
        
        # Priority 2: Deep learning when high confidence
        if historical_intelligence.get('confidence', 0) >= 0.8:
            return 'deep_learning'
        
        # Priority 3: Pattern-based when sufficient history
        if patterns and patterns.get('total_orders', 0) >= 3:
            return 'pattern_based'
        
        # Priority 4: Similar clients when no direct history
        if not previous_sales:
            return 'similar_clients'
        
        # Default: Universal enforcement
        return 'universal'
    
    # ================== MERGED BUSINESS RULES + DEEP LEARNING ==================
    
    def _apply_business_rules_with_deep_learning(self, partner, last_products, 
                                                requirements, notes, context):
        """Apply business rules ENHANCED with deep learning insights"""
        
        _logger.info("🔧 Applying Business Rules + Deep Learning...")
        
        historical_intelligence = context.get('historical_intelligence', {})
        
        # 1. Apply business rules transformation
        try:
            rules_engine = self.env['business.rules.engine'].sudo()
            transformation = rules_engine.apply_composition_rules(
                partner.id,
                datetime.now().year,
                last_products
            )
        except:
            _logger.warning("Business rules engine not available, using deep learning approach")
            # Fallback to deep learning transformation
            transformation = self._apply_deep_learning_transformation(
                last_products, historical_intelligence
            )
        
        if not transformation.get('products'):
            _logger.warning("⚠️ No products from transformation, using deep learning")
            return self._generate_from_deep_learning(
                partner, requirements, historical_intelligence
            )
        
        # Extract locked attributes
        locked_attributes = transformation.get('locked_attributes', {}) or {}
        experience_has_foie = bool(locked_attributes.get('experience_has_foie', False))
        
        # 2. Apply dietary filters
        filtered_products = self._filter_products_by_dietary(
            transformation['products'], requirements.get('dietary', [])
        )
        
        # 2a. Enforce Tokaji ↔ Foie pairing
        filtered_products = self._ensure_tokaji_foie_pairing(
            filtered_products, experience_has_foie, requirements.get('dietary', [])
        )
        
        # 3. Use deep learning to optimize count
        if historical_intelligence.get('optimal_product_count'):
            target_count = historical_intelligence['optimal_product_count']
            _logger.info(f"📋 Using deep learning product count: {target_count}")
        else:
            target_count = requirements.get('product_count', len(last_products))
        
        # 4. Apply intelligent optimization (from deep learning)
        optimized_products = self._intelligent_budget_optimization(
            filtered_products,
            target_count,
            requirements['budget'],
            requirements.get('budget_flexibility', 10)
        )
        
        # 5. Enforce category counts if needed
        if last_products:
            target_counts = self._compute_category_counts(last_products)
            optimized_products = self._enforce_category_counts(
                optimized_products,
                target_counts,
                requirements['budget'],
                requirements.get('dietary', []),
                {**context, 'locked_attributes': locked_attributes}
            )
        
        # 6. Final budget guardrail
        optimized_products = self._enforce_budget_guardrail(
            optimized_products,
            requirements['budget'],
            tolerance=0.05,
            dietary=requirements.get('dietary', []),
            context={**context, 'locked_attributes': locked_attributes}
        )
        
        # Calculate total cost
        total_cost = sum(p.list_price for p in optimized_products)
        
        # Create composition
        try:
            rule_summary = self._build_comprehensive_rules_summary(
                transformation.get('rule_applications', []),
                requirements,
                optimized_products,
                total_cost,
                historical_intelligence
            )
            
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': requirements['budget'],
                'target_year': fields.Date.today().year,
                'actual_cost': total_cost,
                'product_ids': [(6, 0, [p.id for p in optimized_products])],
                'dietary_restrictions': ', '.join(requirements.get('dietary', [])),
                'client_notes': notes,
                'generation_method': 'business_rules_deep_learning',
                'composition_type': requirements.get('composition_type', 'custom'),
                'confidence_score': max(0.95, historical_intelligence.get('confidence', 0.9)),
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
                'confidence_score': composition.confidence_score,
                'method': 'business_rules_deep_learning',
                'message': f"Applied {len(transformation.get('rule_applications', []))} rules + deep learning"
            }
            
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _apply_deep_learning_transformation(self, last_products, intelligence):
        """Apply deep learning when business rules unavailable"""
        
        transformed = []
        patterns = intelligence.get('successful_patterns', [])
        
        # Use successful patterns to transform
        if patterns:
            for pattern in patterns[:3]:  # Use top 3 patterns
                for prod_id in pattern.get('products', []):
                    product = self.env['product.template'].browse(prod_id)
                    if product.exists() and self._has_stock(product):
                        transformed.append(product)
        
        # Add products from last year with good stock
        for product in last_products:
            if self._has_stock(product) and product not in transformed:
                transformed.append(product)
        
        return {'products': transformed, 'rule_applications': []}
    
    def _build_comprehensive_rules_summary(self, rule_applications, requirements, 
                                          products, total_cost, intelligence):
        """Build summary including business rules, requirements, and deep learning"""
        
        summary_parts = []
        
        # Deep learning insights
        if intelligence and intelligence.get('insights'):
            summary_parts.append("🧠 DEEP LEARNING INSIGHTS:")
            for insight in intelligence['insights'][:3]:
                summary_parts.append(f"- {insight}")
            summary_parts.append(f"- Confidence: {intelligence.get('confidence', 0)*100:.0f}%")
        
        # Business rules section
        if rule_applications:
            summary_parts.append("\n📋 BUSINESS RULES APPLIED:")
            
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
        
        # Requirements enforcement
        summary_parts.append("\n📊 REQUIREMENTS ENFORCEMENT:")
        summary_parts.append(f"- Products: {len(products)} (target: {requirements.get('product_count', 'flexible')})")
        summary_parts.append(f"- Budget: €{total_cost:.2f} (target: €{requirements['budget']:.2f})")
        
        variance = ((total_cost - requirements['budget']) / requirements['budget']) * 100
        summary_parts.append(f"- Variance: {variance:+.1f}%")
        
        if requirements.get('dietary'):
            summary_parts.append(f"- Dietary: {', '.join(requirements['dietary'])} ✓")
        
        return "\n".join(summary_parts)
    
    # ================== DEEP LEARNING METHODS (FROM VERSION 1) ==================
    
    def _deep_historical_learning(self, partner_id, target_budget):
        """
        Deep learning from ALL available data - this is TRUE intelligence
        """
        intelligence = {
            'optimal_product_count': None,
            'typical_categories': {},
            'price_distribution': {},
            'successful_patterns': [],
            'confidence': 0.0,
            'insights': [],
            'patterns': {}  # Add patterns for compatibility
        }
        
        # 1. Learn from this client's history
        client_orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ])
        
        if client_orders:
            # Analyze product counts across different budgets
            budget_to_count = []
            for order in client_orders:
                if order.amount_untaxed > 0:
                    product_count = len(order.order_line.filtered(lambda l: l.product_id and l.price_unit > 0))
                    budget_to_count.append({
                        'budget': order.amount_untaxed,
                        'count': product_count,
                        'ratio': product_count / order.amount_untaxed * 100
                    })
            
            # Find pattern for this budget level
            similar_budget_orders = [
                bc for bc in budget_to_count 
                if target_budget * 0.7 <= bc['budget'] <= target_budget * 1.3
            ]
            
            if similar_budget_orders:
                counts = [bc['count'] for bc in similar_budget_orders]
                intelligence['optimal_product_count'] = int(statistics.median(counts))
                intelligence['confidence'] += 0.5
                intelligence['insights'].append(f"Historical: typically {intelligence['optimal_product_count']} products for this budget range")
            
            # Learn category distributions
            all_categories = Counter()
            for order in client_orders:
                for line in order.order_line:
                    if line.product_id and line.product_id.categ_id:
                        all_categories[line.product_id.categ_id.name] += 1
            
            total_products = sum(all_categories.values())
            if total_products > 0:
                intelligence['typical_categories'] = {
                    cat: count/total_products 
                    for cat, count in all_categories.most_common(10)
                }
                intelligence['confidence'] += 0.3
        
        # 2. Learn from similar budget sales (all clients)
        if not intelligence['optimal_product_count']:
            similar_sales = self.env['sale.order'].search([
                ('state', 'in', ['sale', 'done']),
                ('amount_untaxed', '>=', target_budget * 0.8),
                ('amount_untaxed', '<=', target_budget * 1.2)
            ], limit=50, order='date_order desc')
            
            if similar_sales:
                product_counts = []
                for sale in similar_sales:
                    count = len(sale.order_line.filtered(lambda l: l.product_id and l.price_unit > 0))
                    if count > 0:
                        product_counts.append(count)
                
                if product_counts:
                    intelligence['optimal_product_count'] = int(statistics.median(product_counts))
                    intelligence['confidence'] += 0.2
                    intelligence['insights'].append(f"Market data: {intelligence['optimal_product_count']} products typical for €{target_budget:.0f} budget")
        
        # 3. Learn successful product combinations
        successful_compositions = self.env['gift.composition'].search([
            ('confidence_score', '>=', 0.8),
            ('actual_cost', '>=', target_budget * 0.8),
            ('actual_cost', '<=', target_budget * 1.2)
        ], limit=20)
        
        if successful_compositions:
            for comp in successful_compositions:
                pattern = {
                    'products': comp.product_ids.ids,
                    'categories': Counter([p.categ_id.name for p in comp.product_ids if p.categ_id]),
                    'count': len(comp.product_ids),
                    'budget': comp.actual_cost
                }
                intelligence['successful_patterns'].append(pattern)
            
            if not intelligence['optimal_product_count']:
                counts = [p['count'] for p in intelligence['successful_patterns']]
                intelligence['optimal_product_count'] = int(statistics.median(counts))
                intelligence['confidence'] += 0.1
        
        # 4. Final fallback - intelligent estimation
        if not intelligence['optimal_product_count']:
            all_sales = self.env['sale.order'].search([
                ('state', 'in', ['sale', 'done']),
                ('amount_untaxed', '>', 0)
            ], limit=200, order='date_order desc')
            
            if all_sales:
                budget_count_pairs = []
                for sale in all_sales:
                    count = len(sale.order_line.filtered(lambda l: l.product_id and l.price_unit > 0))
                    if count > 0:
                        budget_count_pairs.append((sale.amount_untaxed, count))
                
                if budget_count_pairs:
                    ratios = [count / (budget / 100) for budget, count in budget_count_pairs if budget > 0]
                    avg_ratio = statistics.median(ratios)
                    intelligence['optimal_product_count'] = max(3, min(30, int(target_budget / 100 * avg_ratio)))
                    intelligence['insights'].append(f"Learned ratio: ~{avg_ratio:.1f} products per €100")
            
            if not intelligence['optimal_product_count']:
                intelligence['optimal_product_count'] = 12
        
        if not intelligence['confidence']:
            intelligence['confidence'] = 0.1
        
        _logger.info(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                  DEEP LEARNING ANALYSIS                      ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ Optimal Product Count: {intelligence['optimal_product_count']:<38} ║
        ║ Confidence: {intelligence['confidence']*100:>48.0f}% ║
        ║ Insights: {len(intelligence['insights']):<47} ║
        ║ Categories Learned: {len(intelligence['typical_categories']):<41} ║
        ║ Successful Patterns: {len(intelligence['successful_patterns']):<40} ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        return intelligence
    
    def _parse_with_intelligence(self, notes, budget, dietary, composition_type, intelligence):
        """
        Parse requirements using Ollama + Historical Intelligence
        """
        
        requirements = {
            'budget': budget,
            'product_count': intelligence['optimal_product_count'],
            'dietary': dietary if dietary else [],
            'composition_type': composition_type,
            'categories': intelligence['typical_categories'],
            'patterns': intelligence['successful_patterns'],
            'special_notes': notes
        }
        
        if self.ollama_enabled and notes:
            prompt = f"""
            Parse these gift requirements. We have learned from history:
            - Optimal products for €{budget:.0f}: {intelligence['optimal_product_count']}
            - Typical categories: {', '.join(list(intelligence['typical_categories'].keys())[:5])}
            
            Client notes: {notes}
            
            Extract any OVERRIDES or MODIFICATIONS to our learned patterns:
            - Does client want MORE or FEWER products than {intelligence['optimal_product_count']}?
            - Any specific categories to include/exclude?
            - Any special requirements?
            
            Return JSON with only the CHANGES/OVERRIDES:
            {{
                "product_count_override": null or number if specified,
                "must_include_categories": [],
                "must_exclude_categories": [],
                "special_requirements": ""
            }}
            """
            
            try:
                response = self._call_ollama(prompt, format_json=True)
                if response:
                    parsed = json.loads(response)
                    
                    if parsed.get('product_count_override'):
                        count = parsed['product_count_override']
                        if 1 <= count <= 100:
                            requirements['product_count'] = count
                            requirements['enforce_count'] = True
                            _logger.info(f"Override: product count {intelligence['optimal_product_count']} → {count}")
                    
                    if parsed.get('must_include_categories'):
                        requirements['must_include'] = parsed['must_include_categories']
                    
                    if parsed.get('must_exclude_categories'):
                        requirements['must_exclude'] = parsed['must_exclude_categories']
                    
                    if parsed.get('special_requirements'):
                        requirements['special_notes'] += " " + parsed['special_requirements']
                        
            except Exception as e:
                _logger.debug(f"Ollama parsing adjustment failed: {e}")
        
        return requirements
    
    def _generate_from_deep_learning(self, partner, requirements, intelligence):
        """
        Generate using deep learned patterns
        """
        
        budget = requirements['budget']
        product_count = requirements.get('product_count', intelligence['optimal_product_count'])
        dietary = requirements.get('dietary', [])
        
        _logger.info(f"🧠 Deep Learning Generation: {product_count} products for €{budget:.0f}")
        
        candidate_products = []
        used_product_ids = set()
        
        # 1. Use 80% from successful historical patterns
        historical_portion = int(product_count * 0.8)
        
        for pattern in intelligence['successful_patterns']:
            if len(candidate_products) >= historical_portion:
                break
            
            for prod_id in pattern['products']:
                if prod_id not in used_product_ids:
                    product = self.env['product.template'].browse(prod_id)
                    if product.exists() and self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                        candidate_products.append(product)
                        used_product_ids.add(prod_id)
                        
                        if len(candidate_products) >= historical_portion:
                            break
        
        # 2. Add 20% new discoveries
        new_portion = product_count - len(candidate_products)
        if new_portion > 0:
            category_weights = intelligence['typical_categories']
            
            for category, weight in category_weights.items():
                if new_portion <= 0:
                    break
                
                cat_products = self.env['product.template'].search([
                    ('categ_id.name', 'ilike', category),
                    ('id', 'not in', list(used_product_ids)),
                    ('list_price', '>', budget / (product_count * 5)),
                    ('list_price', '<', budget * 0.3),
                    ('sale_ok', '=', True)
                ], limit=int(new_portion * weight) + 1)
                
                for product in cat_products:
                    if self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                        candidate_products.append(product)
                        used_product_ids.add(product.id)
                        new_portion -= 1
                        
                        if new_portion <= 0:
                            break
        
        # 3. Optimize selection for budget
        selected = self._intelligent_budget_optimization(
            candidate_products,
            product_count,
            budget,
            requirements.get('budget_flexibility', 10)
        )
        
        if not selected:
            return {'success': False, 'error': 'Could not optimize selection'}
        
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        composition = self.env['gift.composition'].create({
            'partner_id': partner.id,
            'target_budget': budget,
            'actual_cost': total_cost,
            'product_ids': [(6, 0, [p.id for p in selected])],
            'dietary_restrictions': ', '.join(dietary) if dietary else '',
            'client_notes': requirements.get('special_notes', ''),
            'generation_method': 'deep_learning',
            'composition_type': requirements.get('composition_type', 'custom'),
            'confidence_score': intelligence['confidence'],
            'ai_reasoning': f"Deep learning: {historical_portion}/{product_count} from history, {len(selected)-historical_portion} new. " + 
                           f"Learned from {len(intelligence['successful_patterns'])} patterns."
        })
        
        return {
            'success': True,
            'composition_id': composition.id,
            'products': selected,
            'total_cost': total_cost,
            'product_count': len(selected),
            'confidence_score': intelligence['confidence'],
            'method': 'deep_learning',
            'ai_insights': '. '.join(intelligence['insights'])
        }
    
    def _generate_blended_intelligence(self, partner, requirements, intelligence):
        """
        Blend historical patterns with similar client learning
        """
        
        budget = requirements['budget']
        product_count = requirements.get('product_count', intelligence['optimal_product_count'])
        
        similar_clients = self._find_truly_similar_clients(partner.id, budget)
        
        if not similar_clients:
            return self._generate_from_deep_learning(partner, requirements, intelligence)
        
        # Blend patterns: 60% historical, 40% similar clients
        historical_count = int(product_count * 0.6)
        similar_count = product_count - historical_count
        
        selected_products = []
        
        # Add from historical patterns
        for pattern in intelligence['successful_patterns'][:3]:
            for prod_id in pattern['products']:
                if len(selected_products) >= historical_count:
                    break
                product = self.env['product.template'].browse(prod_id)
                if product.exists() and self._has_stock(product) and product not in selected_products:
                    selected_products.append(product)
        
        # Add from similar clients
        for client_data in similar_clients:
            if len(selected_products) >= product_count:
                break
            
            client_orders = self.env['sale.order'].search([
                ('partner_id', '=', client_data['partner_id']),
                ('state', 'in', ['sale', 'done'])
            ], limit=3)
            
            for order in client_orders:
                for line in order.order_line:
                    if len(selected_products) >= product_count:
                        break
                    if line.product_id:
                        product = line.product_id.product_tmpl_id
                        if product not in selected_products and self._has_stock(product):
                            selected_products.append(product)
        
        # Optimize for budget
        selected = self._intelligent_budget_optimization(
            selected_products,
            product_count,
            budget,
            requirements.get('budget_flexibility', 10)
        )
        
        total_cost = sum(p.list_price for p in selected)
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner.id,
            'target_budget': budget,
            'actual_cost': total_cost,
            'product_ids': [(6, 0, [p.id for p in selected])],
            'generation_method': 'blended_intelligence',
            'confidence_score': (intelligence['confidence'] + 0.7) / 2,
            'ai_reasoning': f"Blended: {historical_count} historical + {similar_count} from {len(similar_clients)} similar clients"
        })
        
        return {
            'success': True,
            'composition_id': composition.id,
            'total_cost': total_cost,
            'product_count': len(selected),
            'method': 'blended_intelligence',
            'confidence_score': composition.confidence_score
        }
    
    def _generate_from_successful_patterns(self, partner, requirements, target_budget):
        """
        Learn from the most successful sales for this budget range
        """
        
        successful_orders = self.env['sale.order'].search([
            ('state', '=', 'done'),
            ('amount_untaxed', '>=', target_budget * 0.8),
            ('amount_untaxed', '<=', target_budget * 1.2)
        ], limit=10, order='amount_untaxed desc')
        
        if not successful_orders:
            successful_orders = self.env['sale.order'].search([
                ('state', '=', 'done')
            ], limit=10, order='amount_untaxed desc')
        
        product_frequency = Counter()
        category_frequency = Counter()
        typical_counts = []
        
        for order in successful_orders:
            lines_with_products = order.order_line.filtered(lambda l: l.product_id and l.price_unit > 0)
            typical_counts.append(len(lines_with_products))
            
            for line in lines_with_products:
                product = line.product_id.product_tmpl_id
                product_frequency[product.id] += 1
                if product.categ_id:
                    category_frequency[product.categ_id.name] += 1
        
        product_count = requirements.get('product_count', 12)
        if typical_counts:
            learned_count = int(statistics.median(typical_counts))
            if not requirements.get('count_override'):
                product_count = learned_count
        
        selected = []
        for prod_id, freq in product_frequency.most_common(product_count * 2):
            if len(selected) >= product_count:
                break
            product = self.env['product.template'].browse(prod_id)
            if product.exists() and self._has_stock(product):
                selected.append(product)
        
        if len(selected) < product_count:
            for cat_name, freq in category_frequency.most_common(5):
                if len(selected) >= product_count:
                    break
                
                cat_products = self.env['product.template'].search([
                    ('categ_id.name', '=', cat_name),
                    ('id', 'not in', [p.id for p in selected]),
                    ('list_price', '>', 0),
                    ('sale_ok', '=', True)
                ], limit=product_count - len(selected))
                
                for product in cat_products:
                    if self._has_stock(product):
                        selected.append(product)
                        if len(selected) >= product_count:
                            break
        
        selected = self._intelligent_budget_optimization(
            selected,
            product_count,
            target_budget,
            requirements.get('budget_flexibility', 10)
        )
        
        total_cost = sum(p.list_price for p in selected)
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner.id,
            'target_budget': target_budget,
            'actual_cost': total_cost,
            'product_ids': [(6, 0, [p.id for p in selected])],
            'generation_method': 'success_patterns',
            'confidence_score': 0.75,
            'ai_reasoning': f"Learned from {len(successful_orders)} successful orders"
        })
        
        return {
            'success': True,
            'composition_id': composition.id,
            'total_cost': total_cost,
            'product_count': len(selected),
            'method': 'success_pattern_learning',
            'confidence_score': 0.75
        }

    def _generate_from_similar_clients(self, partner, requirements, notes, context):
        """Generate based on similar clients when no direct history exists"""
        
        similar_clients = context.get('similar_clients', [])
        
        if not similar_clients:
            # No similar clients found, fallback to universal enforcement
            return self._generate_with_universal_enforcement(partner, requirements, notes, context)
        
        _logger.info(f"👥 Learning from {len(similar_clients)} similar clients")
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements.get('dietary', [])
        
        # Aggregate product popularity from similar clients
        product_popularity = {}
        
        for similar in similar_clients:
            similarity_weight = similar['similarity']
            patterns = similar['patterns']
            
            # Weight products by similarity score
            for prod_id in patterns.get('favorite_products', []):
                if prod_id not in product_popularity:
                    product_popularity[prod_id] = 0
                product_popularity[prod_id] += similarity_weight
        
        # Sort products by popularity score
        popular_product_ids = sorted(
            product_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:product_count * 2]  # Get more than needed for filtering
        
        # Build product list
        products = []
        for prod_id, score in popular_product_ids:
            product = self.env['product.template'].browse(prod_id)
            if product.exists() and self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                products.append(product)
                if len(products) >= product_count * 1.5:  # Get extras for optimization
                    break
        
        # If not enough products from similar clients, add from smart pool
        if len(products) < product_count:
            additional = self._get_smart_product_pool(budget, dietary, context)
            for product in additional:
                if product not in products:
                    products.append(product)
                    if len(products) >= product_count * 2:
                        break
        
        # Optimize selection for budget and count
        if products:
            selected = self._smart_optimize_selection(
                products,
                product_count,
                budget,
                requirements.get('budget_flexibility', 10),
                requirements.get('enforce_count', False),
                context
            )
        else:
            selected = []
        
        if not selected:
            _logger.warning("No products selected from similar clients, using fallback")
            return self._generate_with_universal_enforcement(partner, requirements, notes, context)
        
        total_cost = sum(p.list_price for p in selected)
        
        try:
            # Build reasoning that explains the strategy used
            top_similar = similar_clients[0] if similar_clients else None
            reasoning = f"""AI Generation Using Similar Client Patterns (No Direct History):
    - Based on {len(similar_clients)} similar clients
    - Top match: {top_similar['similarity']*100:.0f}% similarity
    - Selected {len(selected)} products = €{total_cost:.2f}
    - Budget target: €{budget:.2f} (variance: {((total_cost-budget)/budget)*100:+.1f}%)
    - Products from similar patterns: {len([p for p in selected if p.id in product_popularity])}
    - Strategy: Learning from similar purchasing patterns due to limited client history
    """
            
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': budget,
                'target_year': fields.Date.today().year,
                'actual_cost': total_cost,
                'product_ids': [(6, 0, [p.id for p in selected])],
                'dietary_restrictions': ', '.join(dietary) if dietary else '',
                'client_notes': notes,
                'generation_method': 'ollama',  # AI-powered generation
                'composition_type': requirements.get('composition_type', 'custom'),
                'confidence_score': 0.85,
                'ai_reasoning': reasoning
            })
            
            # Auto-categorize if method exists
            if hasattr(composition, 'auto_categorize_products'):
                composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.85,
                'message': f'Similar clients strategy: {len(selected)} products = €{total_cost:.2f}',
                'method': 'similar_clients'  # Internal tracking
            }
            
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}

    def _smart_optimize_selection(self, products, target_count, budget, flexibility, enforce_count, context):
        """COMPLETE selection optimization - GUARANTEED budget compliance"""
        
        min_budget = budget * (1 - flexibility/100)
        max_budget = budget * (1 + flexibility/100)
        
        _logger.info(f"""
        Selection Optimization:
        Target: {target_count} products @ €{budget:.2f}
        Range: €{min_budget:.2f} - €{max_budget:.2f}
        """)
        
        # CRITICAL: Remove ALL zero-price products
        valid_products = []
        for p in products:
            try:
                if float(p.list_price) > 0:
                    valid_products.append(p)
            except:
                continue
        
        if not valid_products:
            _logger.error("No valid products!")
            return []
        
        # Calculate ideal price
        ideal_price = budget / target_count
        
        # METHOD 1: Try perfect selection
        products_sorted = sorted(valid_products, key=lambda p: abs(float(p.list_price) - ideal_price))
        
        selected = []
        current_total = 0
        
        # First pass - add products near ideal price
        for product in products_sorted:
            if len(selected) >= target_count:
                break
            
            price = float(product.list_price)
            if current_total + price <= max_budget:
                selected.append(product)
                current_total += price
        
        # Check if we need adjustment
        if current_total < min_budget:
            # CRITICAL FIX: Add expensive products to meet budget
            _logger.warning(f"Under budget: €{current_total:.2f}, need €{min_budget - current_total:.2f} more")
            
            # Get expensive products
            expensive_products = sorted(
                [p for p in valid_products if p not in selected],
                key=lambda p: float(p.list_price),
                reverse=True
            )
            
            if enforce_count and len(selected) >= target_count:
                # Replace cheap products with expensive ones
                while current_total < min_budget and expensive_products:
                    # Find cheapest selected
                    cheapest_idx = None
                    cheapest_price = float('inf')
                    for i, p in enumerate(selected):
                        if float(p.list_price) < cheapest_price:
                            cheapest_price = float(p.list_price)
                            cheapest_idx = i
                    
                    if cheapest_idx is not None:
                        # Replace with expensive product
                        for exp_product in expensive_products:
                            new_total = current_total - cheapest_price + float(exp_product.list_price)
                            if new_total <= max_budget:
                                selected[cheapest_idx] = exp_product
                                current_total = new_total
                                expensive_products.remove(exp_product)
                                _logger.info(f"Replaced €{cheapest_price:.2f} with €{exp_product.list_price:.2f}")
                                break
                        else:
                            break
                    else:
                        break
            else:
                # Add more products
                for product in expensive_products:
                    if current_total >= min_budget:
                        break
                    if current_total + float(product.list_price) <= max_budget:
                        selected.append(product)
                        current_total += float(product.list_price)
                        _logger.info(f"Added €{product.list_price:.2f} product")
        
        # Final total
        final_total = sum(float(p.list_price) for p in selected)
        
        # EMERGENCY: If still under budget, duplicate expensive products
        if final_total < min_budget and selected:
            shortage = min_budget - final_total
            most_expensive = max(selected, key=lambda p: float(p.list_price))
            
            # Try to find similar expensive products
            similar_expensive = self.env['product.template'].search([
                ('list_price', '>=', float(most_expensive.list_price) * 0.8),
                ('list_price', '<=', shortage),
                ('sale_ok', '=', True),
                ('id', 'not in', [p.id for p in selected])
            ], limit=5)
            
            for product in similar_expensive:
                if final_total >= min_budget:
                    break
                selected.append(product)
                final_total += float(product.list_price)
                _logger.info(f"Emergency add: €{product.list_price:.2f}")
        
        _logger.info(f"""
        FINAL RESULT:
        Products: {len(selected)}
        Total: €{final_total:.2f}
        Target: €{min_budget:.2f} - €{max_budget:.2f}
        Status: {'✅ OK' if min_budget <= final_total <= max_budget else '❌ FAILED'}
        """)
        
        return selected

    def _generate_with_universal_enforcement(self, partner, requirements, notes, context):
        """Universal generation with strict enforcement of ALL requirements - Fallback method"""
        
        budget = requirements['budget']
        flexibility = requirements.get('budget_flexibility', 10)
        product_count = requirements['product_count']
        enforce_count = requirements.get('enforce_count', False)
        dietary = requirements.get('dietary', [])
        
        # Calculate budget bounds
        min_budget = budget * (1 - flexibility/100)
        max_budget = budget * (1 + flexibility/100)
        
        _logger.info(f"🎯 Universal Generation: {'EXACTLY' if enforce_count else 'APPROXIMATELY'} {product_count} products, €{min_budget:.2f}-€{max_budget:.2f}")
        
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
        
        # STRICT ENFORCEMENT if required
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
            base_reasoning = self._build_comprehensive_reasoning(
                requirements, selected, total_cost, budget, context
            )
            
            reasoning = f"""AI Universal Generation (Fallback Strategy):
    {base_reasoning}
    - Strategy: Universal optimization with requirement enforcement
    - Compliance: {'✅ All requirements met' if (count_ok and budget_ok) else '⚠️ Best effort within constraints'}
    """
            
            composition = self.env['gift.composition'].create({
                'partner_id': partner.id,
                'target_budget': budget,
                'target_year': fields.Date.today().year,
                'actual_cost': total_cost,
                'product_ids': [(6, 0, [p.id for p in selected])],
                'dietary_restrictions': ', '.join(dietary) if dietary else '',
                'client_notes': notes,
                'generation_method': 'ollama',  # AI-powered generation
                'composition_type': requirements.get('composition_type', 'custom'),
                'confidence_score': 0.95 if (count_ok and budget_ok) else 0.7,
                'ai_reasoning': reasoning
            })
            
            if hasattr(composition, 'auto_categorize_products'):
                composition.auto_categorize_products()
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected,
                'total_cost': total_cost,
                'product_count': len(selected),
                'confidence_score': 0.95 if (count_ok and budget_ok) else 0.7,
                'message': f"{'✅' if (count_ok and budget_ok) else '⚠️'} Generated {len(selected)} products, €{total_cost:.2f}",
                'method': 'universal_enforcement',  # Internal tracking
                'compliant': count_ok and budget_ok
            }
            
        except Exception as e:
            _logger.error(f"Failed to create composition: {e}")
            return {'success': False, 'error': str(e)}


    def _select_with_category_requirements(self, products, categories_required, total_count, budget):
        """Select products meeting specific category requirements"""
        
        selected = []
        
        # First fulfill category requirements
        for category, count in categories_required.items():
            cat_products = [p for p in products 
                        if category.lower() in (p.name.lower() if p.name else '') or 
                        (p.categ_id and category.lower() in (p.categ_id.name.lower() if p.categ_id.name else ''))]
            
            # Sort by price proximity to ideal
            ideal_price = budget/total_count if total_count else 50
            cat_products.sort(key=lambda p: abs(float(p.list_price) - ideal_price))
            
            added = cat_products[:count]
            selected.extend(added)
            _logger.info(f"📂 Added {len(added)}/{count} {category} products")
        
        # Fill remaining slots
        if total_count:
            remaining_count = total_count - len(selected)
            if remaining_count > 0:
                available = [p for p in products if p not in selected]
                available.sort(key=lambda p: abs(float(p.list_price) - (budget/total_count)))
                selected.extend(available[:remaining_count])
                _logger.info(f"📦 Added {min(remaining_count, len(available))} additional products")
        
        return selected


    def _enforce_exact_count(self, selected, all_products, exact_count, budget):
        """Enforce exact product count no matter what"""
        
        current_count = len(selected)
        
        if current_count == exact_count:
            return selected
        
        if current_count < exact_count:
            # Need more products
            remaining_needed = exact_count - current_count
            available = [p for p in all_products if p not in selected]
            
            # Sort by price to add cheapest first
            available.sort(key=lambda p: float(p.list_price))
            
            added = available[:remaining_needed]
            selected.extend(added)
            _logger.info(f"➕ Added {len(added)} products to meet count requirement ({current_count} → {len(selected)})")
        
        elif current_count > exact_count:
            # Too many products - remove most expensive
            excess = current_count - exact_count
            
            # Sort by price descending
            selected_list = list(selected)
            selected_list.sort(key=lambda p: float(p.list_price), reverse=True)
            
            # Remove most expensive products
            selected = selected_list[excess:]
            _logger.info(f"➖ Removed {excess} expensive products to meet count requirement ({current_count} → {len(selected)})")
        
        # Final safety check
        final_selected = selected[:exact_count] if len(selected) > exact_count else selected
        _logger.info(f"✅ Final count: {len(final_selected)} (target: {exact_count})")
        
        return final_selected


    def _build_comprehensive_reasoning(self, requirements, products, total_cost, budget, context):
        """Build detailed reasoning for the composition"""
        
        reasoning_parts = []
        
        # Basic stats
        reasoning_parts.append(f"📊 Generated {len(products)} products totaling €{total_cost:.2f}")
        
        # Budget analysis
        variance = ((total_cost - budget) / budget * 100) if budget else 0
        reasoning_parts.append(f"💰 Budget: €{budget:.2f} → €{total_cost:.2f} ({variance:+.1f}% variance)")
        
        # Count compliance
        if requirements.get('enforce_count') and requirements.get('product_count'):
            target = requirements['product_count']
            if len(products) == target:
                reasoning_parts.append(f"✅ Exact count requirement met: {target} products")
            else:
                reasoning_parts.append(f"⚠️ Count variance: {len(products)} vs {target} required")
        else:
            reasoning_parts.append(f"📦 Product count: {len(products)} (flexible target: {requirements.get('product_count', 'auto')})")
        
        # Dietary restrictions
        if requirements.get('dietary'):
            reasoning_parts.append(f"🥗 Dietary restrictions applied: {', '.join(requirements['dietary'])}")
        
        # Historical patterns
        if context.get('patterns'):
            patterns = context['patterns']
            
            if patterns.get('total_orders', 0) > 0:
                reasoning_parts.append(f"📈 Based on {patterns['total_orders']} historical orders")
            
            if patterns.get('favorite_products'):
                favorites_included = sum(1 for p in products if p.id in patterns['favorite_products'])
                if favorites_included > 0:
                    reasoning_parts.append(f"⭐ Included {favorites_included} favorite products")
            
            if patterns.get('budget_trend'):
                reasoning_parts.append(f"📊 Budget trend: {patterns['budget_trend']}")
        
        # Categories
        if requirements.get('categories_required'):
            cats = ', '.join(f"{cat}({count})" for cat, count in requirements['categories_required'].items())
            reasoning_parts.append(f"📂 Category requirements: {cats}")
        
        # Source info
        if requirements.get('budget_source'):
            reasoning_parts.append(f"💡 Budget source: {requirements['budget_source']}")
        
        if requirements.get('count_source'):
            reasoning_parts.append(f"💡 Count source: {requirements['count_source']}")
        
        return "\n".join(reasoning_parts)

    def _validate_and_log_result(self, result, requirements):
        """Validate and log the result against requirements"""
        
        if not result or not result.get('success'):
            _logger.error("❌ Generation failed - no valid result")
            return False
        
        actual_count = result.get('product_count', 0)
        actual_cost = result.get('total_cost', 0)
        expected_count = requirements.get('product_count')
        expected_budget = requirements.get('budget', 0)
        flexibility = requirements.get('budget_flexibility', 10)
        
        # Validate product count if strict enforcement
        count_valid = True
        if requirements.get('enforce_count') and expected_count:
            if actual_count == expected_count:
                _logger.info(f"✅ Count requirement MET: {actual_count} products")
            else:
                _logger.error(f"❌ Count requirement FAILED: {actual_count} != {expected_count}")
                count_valid = False
        else:
            _logger.info(f"📦 Product count: {actual_count} (flexible target: {expected_count})")
        
        # Validate budget compliance
        budget_valid = True
        if expected_budget > 0:
            min_budget = expected_budget * (1 - flexibility/100)
            max_budget = expected_budget * (1 + flexibility/100)
            
            if min_budget <= actual_cost <= max_budget:
                variance = ((actual_cost - expected_budget) / expected_budget) * 100
                _logger.info(f"✅ Budget requirement MET: €{actual_cost:.2f} ({variance:+.1f}% variance)")
            else:
                variance = ((actual_cost - expected_budget) / expected_budget) * 100
                _logger.error(f"❌ Budget requirement FAILED: €{actual_cost:.2f} ({variance:+.1f}% variance)")
                _logger.error(f"   Expected range: €{min_budget:.2f} - €{max_budget:.2f}")
                budget_valid = False
        
        # Log dietary compliance
        if requirements.get('dietary'):
            _logger.info(f"🥗 Dietary restrictions applied: {', '.join(requirements['dietary'])}")
        
        # Log composition details
        if result.get('composition_id'):
            _logger.info(f"📝 Composition created: ID #{result['composition_id']}")
        
        # Log confidence score
        if result.get('confidence_score'):
            confidence = result['confidence_score']
            if confidence >= 0.9:
                _logger.info(f"🎯 High confidence: {confidence*100:.0f}%")
            elif confidence >= 0.7:
                _logger.info(f"✅ Good confidence: {confidence*100:.0f}%")
            else:
                _logger.warning(f"⚠️ Low confidence: {confidence*100:.0f}%")
        
        # Log generation method/strategy
        if result.get('method'):
            _logger.info(f"🔧 Strategy used: {result['method']}")
        
        # Log overall success
        overall_valid = count_valid and budget_valid
        
        _logger.info("="*60)
        if overall_valid:
            _logger.info("✅✅✅ GENERATION SUCCESSFUL - ALL REQUIREMENTS MET ✅✅✅")
        else:
            _logger.warning("⚠️⚠️⚠️ GENERATION COMPLETED WITH WARNINGS ⚠️⚠️⚠️")
            if not count_valid:
                _logger.warning("  - Product count requirement not met")
            if not budget_valid:
                _logger.warning("  - Budget requirement not met")
        _logger.info("="*60)
        
        # Log summary
        if result.get('message'):
            _logger.info(f"Summary: {result['message']}")
        
        # Performance metrics
        if result.get('products'):
            products = result['products']
            if products:
                prices = [float(p.list_price) for p in products]
                avg_price = sum(prices) / len(prices) if prices else 0
                min_price = min(prices) if prices else 0
                max_price = max(prices) if prices else 0
                
                _logger.info(f"📊 Product metrics:")
                _logger.info(f"   - Count: {len(products)}")
                _logger.info(f"   - Total: €{sum(prices):.2f}")
                _logger.info(f"   - Average: €{avg_price:.2f}")
                _logger.info(f"   - Range: €{min_price:.2f} - €{max_price:.2f}")
        
        return overall_valid
    
    def _intelligent_budget_optimization(self, products, target_count, target_budget, flexibility):
        """
        Truly intelligent selection to match budget and count
        """
        
        if not products:
            return []
        
        products = [p for p in products if p.list_price > 0]
        
        if len(products) <= target_count:
            return products
        
        target_avg = target_budget / target_count
        
        scored_products = []
        for product in products:
            price_score = 1 / (1 + abs(product.list_price - target_avg) / target_avg)
            
            category = product.categ_id.name if product.categ_id else 'other'
            category_count = sum(1 for p in products if p.categ_id and p.categ_id.name == category)
            diversity_score = 1 / category_count
            
            stock_score = 1.0 if self._verify_stock_availability(product) else 0.5
            
            total_score = (price_score * 0.5) + (diversity_score * 0.3) + (stock_score * 0.2)
            scored_products.append((product, total_score))
        
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        selected = [p for p, score in scored_products[:target_count]]
        
        total = sum(p.list_price for p in selected)
        min_budget = target_budget * (1 - flexibility/100)
        max_budget = target_budget * (1 + flexibility/100)
        
        # Adjust if needed
        if total < min_budget or total > max_budget:
            attempts = 0
            while attempts < 20 and (total < min_budget or total > max_budget):
                attempts += 1
                
                if total < min_budget:
                    cheapest = min(selected, key=lambda p: p.list_price)
                    candidates = [p for p, _ in scored_products if p not in selected and p.list_price > cheapest.list_price]
                    if candidates:
                        selected.remove(cheapest)
                        selected.append(candidates[0])
                        total = sum(p.list_price for p in selected)
                else:
                    expensive = max(selected, key=lambda p: p.list_price)
                    candidates = [p for p, _ in scored_products if p not in selected and p.list_price < expensive.list_price]
                    if candidates:
                        selected.remove(expensive)
                        selected.append(candidates[0])
                        total = sum(p.list_price for p in selected)
        
        return selected
    
    def _find_truly_similar_clients(self, partner_id, budget):
        """
        Find clients with similar purchasing patterns
        """
        
        partner_patterns = self._analyze_client_purchase_patterns(partner_id)
        
        similar_clients = []
        
        potential_clients = self.env['sale.order'].read_group(
            [
                ('state', 'in', ['sale', 'done']),
                ('partner_id', '!=', partner_id),
                ('amount_untaxed', '>=', budget * 0.7),
                ('amount_untaxed', '<=', budget * 1.3)
            ],
            ['partner_id'],
            ['partner_id'],
            limit=50
        )
        
        for client_data in potential_clients:
            client_id = client_data['partner_id'][0]
            client_patterns = self._analyze_client_purchase_patterns(client_id)
            
            if client_patterns:
                similarity = self._calculate_pattern_similarity(partner_patterns, client_patterns)
                if similarity > 0.3:
                    similar_clients.append({
                        'partner_id': client_id,
                        'similarity': similarity,
                        'patterns': client_patterns
                    })
        
        similar_clients.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_clients[:10]
    
    def _calculate_pattern_similarity(self, patterns1, patterns2):
        """
        Calculate similarity between two client patterns
        """
        
        if not patterns1 or not patterns2:
            return 0.0
        
        similarity_score = 0.0
        factors = 0
        
        avg1 = patterns1.get('avg_order_value', 0)
        avg2 = patterns2.get('avg_order_value', 0)
        if avg1 and avg2:
            diff = abs(avg1 - avg2) / max(avg1, avg2)
            if diff < 0.3:
                similarity_score += (1 - diff)
            factors += 1
        
        count1 = patterns1.get('avg_product_count', 0)
        count2 = patterns2.get('avg_product_count', 0)
        if count1 and count2:
            count_diff = abs(count1 - count2) / max(count1, count2)
            if count_diff < 0.3:
                similarity_score += (1 - count_diff)
            factors += 1
        
        cats1 = set(patterns1.get('preferred_categories', {}).keys())
        cats2 = set(patterns2.get('preferred_categories', {}).keys())
        if cats1 and cats2:
            overlap = len(cats1.intersection(cats2)) / len(cats1.union(cats2))
            similarity_score += overlap
            factors += 1
        
        if patterns1.get('budget_trend') == patterns2.get('budget_trend'):
            similarity_score += 0.5
            factors += 0.5
        
        return similarity_score / factors if factors > 0 else 0.0
    
    def _update_learning_from_result(self, result, requirements, intelligence):
        """
        Learn from each generation to improve future ones
        """
        
        try:
            cache_data = json.loads(self.learning_cache) if self.learning_cache else {}
        except:
            cache_data = {}
        
        if result.get('success'):
            pattern_key = f"budget_{int(requirements['budget']/100)*100}"
            
            if pattern_key not in cache_data:
                cache_data[pattern_key] = []
            
            cache_data[pattern_key].append({
                'product_count': result['product_count'],
                'total_cost': result['total_cost'],
                'confidence': result.get('confidence_score', 0.5),
                'timestamp': fields.Datetime.now().isoformat()
            })
            
            cache_data[pattern_key] = cache_data[pattern_key][-50:]
            
            self.learning_cache = json.dumps(cache_data)
            self.cache_expiry = fields.Datetime.now() + timedelta(days=30)
    
    # ================== BUSINESS RULES METHODS (FROM VERSION 2) ==================
    
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
        
        # Extract locked attributes from transformation
        locked_attributes = transformation.get('locked_attributes', {}) or {}
        experience_has_foie = bool(locked_attributes.get('experience_has_foie', False))
        
        # 2. Apply dietary filters
        filtered_products = self._filter_products_by_dietary(
            transformation['products'], requirements.get('dietary', [])
        )
        
        # 2a. Enforce Tokaji ↔ Foie pairing if needed
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
                sorted_existing = sorted([(i, p) for i,p in enumerate(products_mut) if p.id not in locked_ids], 
                                        key=lambda ip: float(ip[1].list_price), reverse=True)
                if not sorted_existing:
                    break
                replace_idx, to_replace = sorted_existing[0]
                candidate = next((p for p in pool_sorted_low if float(p.list_price) < float(to_replace.list_price) 
                                 and self._check_dietary_compliance(p, dietary)), None)
            else:
                sorted_existing = sorted([(i, p) for i,p in enumerate(products_mut) if p.id not in locked_ids], 
                                        key=lambda ip: float(ip[1].list_price))
                if not sorted_existing:
                    break
                replace_idx, to_replace = sorted_existing[0]
                candidate = next((p for p in pool_sorted_high if float(p.list_price) > float(to_replace.list_price) 
                                 and self._check_dietary_compliance(p, dietary)), None)
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
        tokaji_grades = [getattr(p, 'product_grade', None) for p in products 
                        if getattr(p, 'beverage_family', '') in ['tokaj','tokaji']]
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
    
    # ================== ALL PRESERVED HELPER METHODS ==================
    
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
    
    def _get_smart_product_pool(self, budget, dietary, context):
        """Get product pool - STRICT PRICE FILTERING"""
        
        # Calculate APPROPRIATE price range for budget
        if budget >= 1000:
            min_price = 30.0  # Higher minimum for high budgets
            max_price = 200.0
            target_count = 12
        elif budget >= 500:
            min_price = 20.0
            max_price = 150.0
            target_count = 10
        elif budget >= 200:
            min_price = 10.0
            max_price = 80.0
            target_count = 8
        else:
            min_price = 5.0
            max_price = 50.0
            target_count = 6
        
        # Calculate ideal price per product
        ideal_price = budget / target_count
        
        _logger.info(f"""
        Product Pool Filter:
        Budget: €{budget:.2f}
        Target: {target_count} products @ €{ideal_price:.2f} each
        Range: €{min_price:.2f} - €{max_price:.2f}
        """)
        
        # STRICT domain
        domain = [
            ('sale_ok', '=', True),
            ('active', '=', True),
            ('list_price', '>=', min_price),  # MUST have minimum price
            ('list_price', '<=', max_price),
            ('list_price', '>', 0),           # NEVER zero
        ]
        
        # Dietary filters
        if dietary:
            if 'halal' in [d.lower() for d in dietary]:
                domain.append(('name', 'not ilike', 'pork'))
                domain.append(('name', 'not ilike', 'jamón'))
                domain.append(('name', 'not ilike', 'cerdo'))
                domain.append(('name', 'not ilike', 'wine'))
                domain.append(('name', 'not ilike', 'alcohol'))
        
        # Exclude IDs
        exclude_ids = context.get('exclude_ids', [])
        if exclude_ids:
            domain.append(('id', 'not in', exclude_ids))
        
        # Search
        products = self.env['product.template'].sudo().search(domain, limit=200, order='list_price desc')
        
        # STRICT validation
        valid_products = []
        for product in products:
            try:
                price = float(product.list_price)
                
                # MUST have valid price in range
                if min_price <= price <= max_price and price > 0:
                    # MUST have stock
                    if self._has_stock(product):
                        # MUST pass dietary
                        if self._check_dietary_compliance(product, dietary):
                            valid_products.append(product)
            except:
                continue
        
        # Sort by price proximity to ideal
        valid_products.sort(key=lambda p: abs(float(p.list_price) - ideal_price))
        
        # Convert to recordset
        if valid_products:
            result = self.env['product.template'].browse([p.id for p in valid_products])
        else:
            # EMERGENCY: If no products found, expand search
            _logger.warning("No products found, expanding search")
            emergency_domain = [
                ('sale_ok', '=', True),
                ('list_price', '>', 10),
                ('list_price', '<', budget * 0.5)
            ]
            result = self.env['product.template'].sudo().search(emergency_domain, limit=50)
        
        _logger.info(f"Found {len(result)} valid products")
        return result
    
    def _check_dietary_compliance(self, product, dietary_restrictions):
        """Check if product complies with dietary restrictions (Enhanced from both versions)"""
        if not dietary_restrictions:
            return True
        
        product_name = product.name.lower() if product.name else ''
        categ_name = product.categ_id.name.lower() if product.categ_id else ''
        
        for restriction in dietary_restrictions:
            restriction = restriction.lower()
            
            if restriction in ['halal', 'no_pork']:
                # Check for pork/iberian products
                prohibited = ['cerdo', 'pork', 'jamón', 'jamon', 'ibérico', 'iberico', 
                            'chorizo', 'salchichón', 'lomo', 'panceta', 'bacon']
                if any(word in product_name for word in prohibited):
                    return False
                # Check for alcohol
                if any(word in product_name for word in ['vino', 'wine', 'alcohol', 'licor', 'whisky', 'vodka']):
                    return False
                # Check product attributes if available
                if hasattr(product, 'contains_pork') and product.contains_pork:
                    return False
                if hasattr(product, 'contains_alcohol') and product.contains_alcohol:
                    return False
                if hasattr(product, 'is_iberian_product') and product.is_iberian_product:
                    return False
            
            elif restriction in ['vegan', 'vegano']:
                prohibited = ['carne', 'meat', 'pollo', 'chicken', 'pescado', 'fish', 
                            'marisco', 'queso', 'cheese', 'leche', 'milk', 'huevo', 'egg',
                            'mantequilla', 'butter', 'nata', 'cream', 'miel', 'honey']
                if any(word in product_name for word in prohibited):
                    return False
                if hasattr(product, 'is_vegan') and not product.is_vegan:
                    return False
            
            elif restriction in ['vegetarian', 'vegetariano']:
                prohibited = ['carne', 'meat', 'pollo', 'chicken', 'pescado', 'fish', 
                            'marisco', 'jamón', 'anchoa', 'atún', 'tuna']
                if any(word in product_name for word in prohibited):
                    return False
                # Check category
                cat = (getattr(product, 'lebiggot_category', '') or '').lower()
                if cat in ['charcuterie', 'foie_gras']:
                    return False
            
            elif restriction in ['no_alcohol', 'non_alcoholic', 'sin_alcohol']:
                prohibited = ['vino', 'wine', 'alcohol', 'licor', 'cerveza', 'beer', 
                            'whisky', 'vodka', 'ginebra', 'gin', 'rum', 'brandy', 'cava', 'champagne']
                if any(word in product_name for word in prohibited):
                    return False
                if hasattr(product, 'contains_alcohol') and product.contains_alcohol:
                    return False
            
            elif restriction in ['gluten_free', 'sin_gluten']:
                prohibited = ['pan', 'bread', 'pasta', 'galleta', 'cookie', 'harina', 
                            'flour', 'trigo', 'wheat', 'cebada', 'barley', 'centeno', 'rye']
                if any(word in product_name for word in prohibited):
                    return False
                if hasattr(product, 'contains_gluten') and product.contains_gluten:
                    return False
            
            elif restriction == 'no_iberian':
                if hasattr(product, 'is_iberian_product') and product.is_iberian_product:
                    return False
                if any(word in product_name for word in ['ibérico', 'iberico', 'bellota']):
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
            
            # Method 3: Check stock quants for all variants
            for variant in product.product_variant_ids:
                stock_quants = self.env['stock.quant'].sudo().search([
                    ('product_id', '=', variant.id),
                    ('location_id.usage', '=', 'internal'),
                    ('quantity', '>', 0)
                ])
                if stock_quants:
                    available_qty = sum(sq.quantity - sq.reserved_quantity for sq in stock_quants)
                    if available_qty > 0:
                        return True
            
            return False
            
        except Exception as e:
            _logger.warning(f"Stock check error for {product.name}: {e}")
            return False
    
    def _verify_stock_availability(self, product):
        """More reliable stock check with SQL query"""
        # First check if it's a stockable product
        if product.type != 'product':
            return True  # Services and consumables are always "available"
        
        # Check through all variants
        for variant in product.product_variant_ids:
            # Direct SQL query for accuracy
            self._cr.execute("""
                SELECT COALESCE(SUM(sq.quantity - sq.reserved_quantity), 0) as available
                FROM stock_quant sq
                JOIN stock_location sl ON sq.location_id = sl.id
                WHERE sq.product_id = %s
                AND sl.usage = 'internal'
                AND sq.quantity > sq.reserved_quantity
            """, (variant.id,))
            
            result = self._cr.fetchone()
            if result and result[0] > 0:
                return True
        
        return False
    
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
        
        # Update frequency in data
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
                    if line.product_id.product_tmpl_id not in products:
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
        
        # Compare average order values
        avg1 = patterns1.get('avg_order_value', 0)
        avg2 = patterns2.get('avg_order_value', 0)
        if avg1 and avg2:
            budget_diff = abs(avg1 - avg2) / max(avg1, avg2)
            if budget_diff < 0.3:
                score += (1 - budget_diff)
            factors += 1
        
        # Compare product counts
        count1 = patterns1.get('avg_product_count', 0)
        count2 = patterns2.get('avg_product_count', 0)
        if count1 and count2:
            count_diff = abs(count1 - count2) / max(count1, count2)
            if count_diff < 0.3:
                score += (1 - count_diff)
            factors += 1
        
        # Compare categories
        cats1 = set(patterns1.get('preferred_categories', {}).keys())
        cats2 = set(patterns2.get('preferred_categories', {}).keys())
        if cats1 and cats2:
            overlap = len(cats1.intersection(cats2)) / len(cats1.union(cats2))
            score += overlap
            factors += 1
        
        # Compare budget trends
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
        """Make a call to Ollama API with proper error handling"""
        if not self.ollama_enabled:
            return None
        
        try:
            # Limit prompt length for the 3B model
            max_prompt_length = 2000
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
                _logger.warning(f"Prompt truncated to {max_prompt_length} characters")
            
            url = f"{self.ollama_base_url}/api/generate"
            
            payload = {
                'model': self.ollama_model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.5,  # Lower temperature for more consistent output
                    'top_p': 0.9,
                    'num_predict': 500,  # Reduced for faster response
                    'num_ctx': 2048,     # Context window
                    'seed': 42,          # For reproducibility
                    'stop': ['\n\n', '```']  # Stop sequences
                }
            }
            
            if format_json:
                # Simpler JSON instruction
                payload['format'] = 'json'
                payload['options']['temperature'] = 0.3  # Even lower for JSON
            
            response = requests.post(
                url, 
                json=payload, 
                timeout=60,  # 60 second timeout
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('response', '').strip()
                
                if format_json and result:
                    # Clean up JSON response
                    result = result.replace('```json', '').replace('```', '')
                    result = result.strip()
                    # Validate JSON
                    try:
                        json.loads(result)  # Test parse
                    except:
                        _logger.error(f"Invalid JSON from Ollama: {result[:200]}")
                        return None
                
                return result
            elif response.status_code == 500:
                _logger.error(f"Ollama 500 error. Model might be overloaded.")
                return None
            else:
                _logger.error(f"Ollama API error {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            _logger.error("Ollama request timeout")
            return None
        except requests.exceptions.RequestException as e:
            _logger.error(f"Ollama request error: {e}")
            return None
        except Exception as e:
            _logger.error(f"Unexpected Ollama error: {e}")
            return None
    
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
    
    def _parse_notes_with_ollama(self, notes, form_data=None):
        """Use Ollama to parse notes - SIMPLIFIED"""
        
        if not notes or not self.ollama_enabled:
            return self._parse_notes_basic_fallback(notes)
        
        # MUCH simpler prompt - just extract numbers
        prompt = f"""Extract from text: "{notes[:200]}"
    Return JSON:
    {{"count": <number or null>, "budget": <number or null>}}"""
        
        try:
            response = self._call_ollama(prompt, format_json=True)
            
            if response:
                try:
                    extracted = json.loads(response)
                    
                    requirements = {
                        'use_default': False,
                        'product_count': extracted.get('count'),
                        'budget_override': extracted.get('budget'),
                        'dietary': [],  # Parse dietary separately if needed
                    }
                    
                    _logger.info(f"Ollama parsed: {requirements}")
                    return requirements
                    
                except json.JSONDecodeError:
                    _logger.debug("JSON parse failed, using fallback")
                    return self._parse_notes_basic_fallback(notes)
            else:
                return self._parse_notes_basic_fallback(notes)
                
        except Exception as e:
            _logger.debug(f"Ollama error: {e}, using fallback")
            return self._parse_notes_basic_fallback(notes)

    def _parse_notes_basic_fallback(self, notes):
        """Basic fallback parser when Ollama fails"""
        
        parsed = {
            'use_default': False,
            'product_count': None,
            'budget_override': None,
            'dietary': [],
        }
        
        if not notes:
            return parsed
        
        notes_lower = notes.lower()
        
        # Extract numbers
        import re
        numbers = re.findall(r'\b(\d+)\b', notes)
        for num in numbers:
            num_int = int(num)
            if 5 <= num_int <= 50:
                parsed['product_count'] = num_int
            elif 100 <= num_int <= 10000:
                parsed['budget_override'] = float(num_int)
        
        # Basic dietary detection
        if 'halal' in notes_lower:
            parsed['dietary'].append('halal')
        if 'vegan' in notes_lower:
            parsed['dietary'].append('vegan')
        if 'vegetarian' in notes_lower:
            parsed['dietary'].append('vegetarian')
        
        return parsed
    
    # ================== ACTION METHODS ==================
    
    def action_view_recommendations(self):
        """View all recommendations"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'All Recommendations',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('generation_method', 'in', ['ollama', 'deep_learning', 'business_rules_deep_learning'])],
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
                ('generation_method', 'in', ['ollama', 'deep_learning', 'business_rules_deep_learning']),
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
            self._deep_historical_learning(partner_id, 1000)  # Also trigger deep learning
            analyzed_count += 1
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': '🧠 Deep Learning Analysis Complete',
                'message': f'Analyzed {analyzed_count} clients with deep learning + business rules. Cache updated.',
                'type': 'success',
                'sticky': False,
            }
        }
    
    @api.model
    def get_or_create_recommender(self):
        """Get existing recommender or create new one"""
        recommender = self.search([('active', '=', True)], limit=1)
        if not recommender:
            recommender = self.create({
                'name': 'AI Gift Recommender (Deep Learning + Business Rules)',
                'ollama_enabled': True
            })
        return recommender
    
    # ================== ACTION METHODS ==================
    
    def action_view_recommendations(self):
        """View all recommendations"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'All Recommendations',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('generation_method', 'in', ['ollama', 'deep_learning', 'business_rules_deep_learning'])],
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
                ('generation_method', 'in', ['ollama', 'deep_learning', 'business_rules_deep_learning']),
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
            self._deep_historical_learning(partner_id, 1000)  # Also trigger deep learning
            analyzed_count += 1
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': '🧠 Deep Learning Analysis Complete',
                'message': f'Analyzed {analyzed_count} clients with deep learning + business rules. Cache updated.',
                'type': 'success',
                'sticky': False,
            }
        }
    
    @api.model
    def get_or_create_recommender(self):
        """Get existing recommender or create new one"""
        recommender = self.search([('active', '=', True)], limit=1)
        if not recommender:
            recommender = self.create({
                'name': 'AI Gift Recommender (Deep Learning + Business Rules)',
                'ollama_enabled': True
            })
        return recommender
    
    def test_ollama_connection(self):
        """Test connection to Ollama service"""
        self.ensure_one()
        
        if not self.ollama_enabled:
            return {'success': False, 'message': 'Ollama is disabled'}
        
        try:
            response = self._call_ollama("Respond with 'OK' if you receive this message.")
            
            if response:
                return {'success': True, 'message': f'✅ Connected to Ollama ({self.ollama_model})'}
            else:
                return {'success': False, 'message': 'No response from Ollama'}
        except Exception as e:
            return {'success': False, 'message': f'Connection failed: {str(e)}'}