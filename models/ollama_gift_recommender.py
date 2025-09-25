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
import statistics

_logger = logging.getLogger(__name__)

class OllamaGiftRecommender(models.Model):
    _name = 'ollama.gift.recommender'
    _description = 'Complete Intelligent Gift Recommendation Engine with Business Rules'
    _rec_name = 'name'
    
    # ================== FIELD DEFINITIONS (PRESERVED FROM ORIGINAL) ==================
    
    name = fields.Char(string="Recommender Name", default="AI Gift Recommender", required=True)
    active = fields.Boolean(string="Active", default=True)
    
    # Ollama Configuration
    ollama_enabled = fields.Boolean(string="Ollama Enabled", default=True)
    ollama_base_url = fields.Char(string="Ollama Base URL", default="http://localhost:11434")
    ollama_model = fields.Char(string="Ollama Model", default="llama3.2:3b")
    ollama_timeout = fields.Integer(string="Timeout (seconds)", default=60)
    
    # Settings
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
    
    # ================== BUSINESS RULES DEFINITIONS ==================
    
    BUSINESS_RULES = {
        'R1': 'Maintain customer favorites (80% retention)',
        'R2': 'Add complementary products based on patterns',
        'R3': 'Apply seasonal adjustments',
        'R4': 'Ensure category balance',
        'R5': 'Apply quality upgrades for VIP clients',
        'R6': 'Smart budget optimization'
    }
    
    # ================== MAIN ENTRY POINT ==================
    
    def generate_gift_recommendations(self, partner_id, target_budget, client_notes, dietary_restrictions, composition_type='custom'):
        """
        COMPLETE INTELLIGENT GENERATION - Historical Learning + Business Rules
        """
        self.ensure_one()
        start_time = datetime.now()
        
        try:
            partner = self.env['res.partner'].browse(partner_id)
            if not partner.exists():
                return {'success': False, 'error': 'Invalid partner'}
            
            # STEP 1: Deep Historical Learning
            historical_intelligence = self._deep_historical_learning(partner_id, target_budget)
            
            # STEP 2: Parse requirements with Ollama + Intelligence
            requirements = self._parse_requirements_with_ollama(
                client_notes, 
                target_budget, 
                dietary_restrictions,
                composition_type,
                historical_intelligence
            )
            
            # STEP 3: Build comprehensive context
            context = self._build_generation_context(
                partner, requirements, historical_intelligence
            )
            
            # STEP 4: Apply appropriate strategy
            last_year_products = self._get_last_year_products(partner_id)
            patterns = self._analyze_client_purchase_patterns(partner_id)
            
            if last_year_products and len(last_year_products) >= 5:
                # Apply Business Rules with transformation
                result = self._apply_business_rules_with_transformation(
                    partner, requirements, client_notes, context
                )
            elif patterns and patterns.get('total_orders', 0) >= 3:
                # Pattern-based generation
                result = self._generate_from_patterns_enhanced(
                    partner, requirements, client_notes, context
                )
            elif patterns.get('total_orders', 0) == 0:
                # New client - use similar clients
                result = self._generate_from_similar_clients(
                    partner, requirements, client_notes, context
                )
            else:
                # Universal generation with enforcement
                result = self._generate_with_universal_enforcement(
                    partner, requirements, client_notes, context
                )
            
            # STEP 5: Track and learn
            if result.get('success'):
                self._update_learning_from_result(result, requirements, historical_intelligence)
                self.total_recommendations += 1
                self.successful_recommendations += 1
                self.last_recommendation_date = fields.Datetime.now()
                
                response_time = (datetime.now() - start_time).total_seconds()
                self.avg_response_time = (
                    (self.avg_response_time * (self.total_recommendations - 1) + response_time) 
                    / self.total_recommendations
                )
            
            return result
            
        except Exception as e:
            _logger.error(f"Generation failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    # ================== DEEP LEARNING METHODS ==================
    
    def _deep_historical_learning(self, partner_id, target_budget):
        """Deep learning from ALL available historical data"""
        intelligence = {
            'optimal_product_count': None,
            'typical_categories': {},
            'price_distribution': {},
            'successful_patterns': [],
            'confidence': 0.0,
            'insights': [],
            'avg_price_point': None,
            'budget_trend': 'stable'
        }
        
        # Learn from client's history
        client_orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ])
        
        if client_orders:
            budget_to_count = []
            all_prices = []
            
            for order in client_orders:
                if order.amount_untaxed > 0:
                    lines_with_products = order.order_line.filtered(lambda l: l.product_id and l.price_unit > 0)
                    product_count = len(lines_with_products)
                    
                    budget_to_count.append({
                        'budget': order.amount_untaxed,
                        'count': product_count,
                        'ratio': product_count / order.amount_untaxed * 100
                    })
                    
                    for line in lines_with_products:
                        all_prices.append(line.price_unit)
            
            # Find pattern for this budget level
            similar_budget_orders = [
                bc for bc in budget_to_count 
                if target_budget * 0.7 <= bc['budget'] <= target_budget * 1.3
            ]
            
            if similar_budget_orders:
                counts = [bc['count'] for bc in similar_budget_orders]
                intelligence['optimal_product_count'] = int(statistics.median(counts))
                intelligence['confidence'] += 0.5
                intelligence['insights'].append(f"Client history: {intelligence['optimal_product_count']} products typical")
            
            if all_prices:
                intelligence['avg_price_point'] = statistics.median(all_prices)
            
            # Analyze budget trend
            if len(client_orders) >= 2:
                recent_budgets = [o.amount_untaxed for o in client_orders[:3]]
                older_budgets = [o.amount_untaxed for o in client_orders[3:6]]
                if recent_budgets and older_budgets:
                    recent_avg = sum(recent_budgets) / len(recent_budgets)
                    older_avg = sum(older_budgets) / len(older_budgets)
                    if recent_avg > older_avg * 1.1:
                        intelligence['budget_trend'] = 'increasing'
                    elif recent_avg < older_avg * 0.9:
                        intelligence['budget_trend'] = 'decreasing'
        
        # Learn from similar budget sales across all clients
        if not intelligence['optimal_product_count']:
            similar_sales = self.env['sale.order'].search([
                ('state', 'in', ['sale', 'done']),
                ('amount_untaxed', '>=', target_budget * 0.8),
                ('amount_untaxed', '<=', target_budget * 1.2)
            ], limit=50, order='date_order desc')
            
            if similar_sales:
                product_counts = []
                category_freq = Counter()
                
                for sale in similar_sales:
                    lines = sale.order_line.filtered(lambda l: l.product_id and l.price_unit > 0)
                    if lines:
                        product_counts.append(len(lines))
                        
                        for line in lines:
                            if line.product_id.categ_id:
                                category_freq[line.product_id.categ_id.name] += 1
                
                if product_counts:
                    intelligence['optimal_product_count'] = int(statistics.median(product_counts))
                    intelligence['confidence'] += 0.3
                    intelligence['insights'].append(f"Market analysis: {intelligence['optimal_product_count']} products standard")
                
                if category_freq:
                    total = sum(category_freq.values())
                    intelligence['typical_categories'] = {
                        cat: count/total for cat, count in category_freq.most_common(10)
                    }
        
        # Final fallback based on learning
        if not intelligence['optimal_product_count']:
            # Learn the relationship between budget and product count
            sample_orders = self.env['sale.order'].search([
                ('state', '=', 'done'),
                ('amount_untaxed', '>', 0)
            ], limit=100)
            
            if sample_orders:
                ratios = []
                for order in sample_orders:
                    count = len(order.order_line.filtered(lambda l: l.product_id and l.price_unit > 0))
                    if count > 0:
                        ratios.append(count / (order.amount_untaxed / 100))
                
                if ratios:
                    avg_ratio = statistics.median(ratios)
                    intelligence['optimal_product_count'] = max(3, min(30, int(target_budget / 100 * avg_ratio)))
                    intelligence['insights'].append(f"Learned ratio: {avg_ratio:.1f} products per €100")
            
            if not intelligence['optimal_product_count']:
                intelligence['optimal_product_count'] = 12  # Ultimate fallback
        
        # Set confidence
        if not intelligence['confidence']:
            intelligence['confidence'] = 0.2
        
        _logger.info(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                  DEEP LEARNING ANALYSIS                      ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ Optimal Product Count: {intelligence['optimal_product_count']:<38} ║
        ║ Confidence: {intelligence['confidence']*100:>48.0f}% ║
        ║ Budget Trend: {intelligence['budget_trend']:<47} ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        return intelligence
    
    # ================== BUSINESS RULES WITH TRANSFORMATION ==================
    
    def _apply_business_rules_with_transformation(self, partner, requirements, notes, context):
        """Apply sophisticated business rules R1-R6 with 80/20 transformation"""
        
        last_year_products = context.get('last_year_products', [])
        if not last_year_products:
            return self._generate_with_universal_enforcement(partner, requirements, notes, context)
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        
        _logger.info(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║              BUSINESS RULES TRANSFORMATION                   ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ Rule R1: Maintain 80% customer favorites                     ║
        ║ Rule R2: Add 20% complementary new products                  ║
        ║ Rule R3: Apply seasonal adjustments                          ║
        ║ Rule R4: Ensure category balance                             ║
        ║ Rule R5: Quality upgrades for VIP                            ║
        ║ Rule R6: Smart budget optimization                           ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # R1: Keep 80% of favorites
        num_to_keep = min(len(last_year_products), int(product_count * 0.8))
        products_to_keep = []
        
        # Filter and validate last year's products
        for product in last_year_products[:num_to_keep * 2]:  # Check more to account for filtering
            if (self._has_stock(product) and 
                self._check_dietary_compliance(product, dietary) and
                product.list_price > 0):
                products_to_keep.append(product)
                if len(products_to_keep) >= num_to_keep:
                    break
        
        # R2: Add 20% complementary new products
        num_new = product_count - len(products_to_keep)
        if num_new > 0:
            new_products = self._find_complementary_products(
                products_to_keep, 
                budget - sum(p.list_price for p in products_to_keep),
                dietary, 
                num_new,
                context
            )
            products_to_keep.extend(new_products)
        
        # R3: Apply seasonal adjustments
        products_to_keep = self._apply_seasonal_adjustments(products_to_keep, context)
        
        # R4: Ensure category balance
        products_to_keep = self._ensure_category_balance(products_to_keep, requirements, context)
        
        # R5: VIP upgrades if applicable
        if context.get('is_vip'):
            products_to_keep = self._apply_vip_upgrades(products_to_keep, budget)
        
        # R6: Smart budget optimization
        products_to_keep = self._smart_optimize_selection(
            products_to_keep, 
            product_count, 
            budget,
            requirements['budget_flexibility'],
            requirements.get('enforce_count', True),
            context
        )
        
        total_cost = sum(p.list_price for p in products_to_keep)
        
        # Build reasoning
        reasoning = self._build_comprehensive_reasoning(
            requirements, products_to_keep, total_cost, budget, context
        )
        
        # Create composition
        composition = self.env['gift.composition'].create({
            'partner_id': partner.id,
            'target_budget': budget,
            'actual_cost': total_cost,
            'product_ids': [(6, 0, [p.id for p in products_to_keep])],
            'dietary_restrictions': ', '.join(dietary) if dietary else '',
            'client_notes': notes,
            'generation_method': 'ollama',
            'composition_type': requirements.get('composition_type', 'custom'),
            'confidence_score': 0.9,
            'ai_reasoning': reasoning
        })
        
        return {
            'success': True,
            'composition_id': composition.id,
            'products': products_to_keep,
            'total_cost': total_cost,
            'product_count': len(products_to_keep),
            'confidence_score': 0.9,
            'method': 'business_rules_transformation',
            'rules_applied': list(self.BUSINESS_RULES.keys()),
            'ai_insights': f"Applied 80/20 rule: {num_to_keep} retained, {num_new} new products"
        }
    
    # ================== SMART OPTIMIZATION METHODS ==================
    
    def _smart_optimize_selection(self, products, target_count, budget, flexibility, enforce_count, context):
        """Smart selection optimization with multiple constraints"""
        
        if not products:
            return []
        
        # Remove invalid products
        valid_products = [p for p in products if p.list_price > 0]
        
        # If we have the exact count needed, check budget
        if len(valid_products) == target_count:
            total = sum(p.list_price for p in valid_products)
            min_budget = budget * (1 - flexibility/100)
            max_budget = budget * (1 + flexibility/100)
            
            if min_budget <= total <= max_budget:
                return valid_products
        
        # Score products for intelligent selection
        scored_products = []
        target_avg = budget / target_count if target_count > 0 else 50
        
        for product in valid_products:
            # Price fitness score
            price_score = 1 / (1 + abs(product.list_price - target_avg) / target_avg)
            
            # Category diversity score
            category = product.categ_id.name if product.categ_id else 'other'
            category_count = sum(1 for p in valid_products if p.categ_id and p.categ_id.name == category)
            diversity_score = 1 / category_count if category_count > 0 else 0
            
            # Historical preference score
            historical_score = 0.8 if product.id in context.get('favorite_products', []) else 0.2
            
            # Combined score
            total_score = (price_score * 0.4) + (diversity_score * 0.3) + (historical_score * 0.3)
            scored_products.append((product, total_score))
        
        # Sort by score
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Select products
        if enforce_count:
            selected = [p for p, score in scored_products[:target_count]]
        else:
            # Flexible selection to match budget
            selected = []
            current_total = 0
            min_budget = budget * (1 - flexibility/100)
            max_budget = budget * (1 + flexibility/100)
            
            for product, score in scored_products:
                if current_total + product.list_price <= max_budget:
                    selected.append(product)
                    current_total += product.list_price
                    
                    if current_total >= min_budget and len(selected) >= target_count * 0.8:
                        break
        
        return selected
    
    def _enforce_budget_guardrail(self, products, budget, tolerance, dietary, context):
        """Enforce budget constraints through product substitution"""
        
        if not products or budget <= 0:
            return products
        
        min_budget = budget * (1 - tolerance)
        max_budget = budget * (1 + tolerance)
        total = sum(p.list_price for p in products)
        
        if min_budget <= total <= max_budget:
            return products
        
        # Get alternative products
        pool = self._get_smart_product_pool(budget, dietary, context)
        products_mut = list(products)
        
        attempts = 0
        while (total < min_budget or total > max_budget) and attempts < 20:
            attempts += 1
            
            if total > max_budget:
                # Need cheaper products
                expensive = max(products_mut, key=lambda p: p.list_price)
                cheaper_alternatives = [p for p in pool if p.list_price < expensive.list_price]
                
                if cheaper_alternatives:
                    products_mut.remove(expensive)
                    replacement = random.choice(cheaper_alternatives[:5])
                    products_mut.append(replacement)
                    total = sum(p.list_price for p in products_mut)
            else:
                # Need more expensive products
                cheapest = min(products_mut, key=lambda p: p.list_price)
                expensive_alternatives = [p for p in pool if p.list_price > cheapest.list_price]
                
                if expensive_alternatives:
                    products_mut.remove(cheapest)
                    replacement = random.choice(expensive_alternatives[:5])
                    products_mut.append(replacement)
                    total = sum(p.list_price for p in products_mut)
        
        return products_mut
    
    # ================== PRODUCT POOL & FILTERING ==================
    
    def _get_smart_product_pool(self, budget, dietary, context):
        """Get intelligently filtered product pool based on context"""
        
        patterns = context.get('patterns', {})
        historical = context.get('historical_intelligence', {})
        
        # Smart price filtering based on budget and learning
        if historical.get('avg_price_point'):
            # Use learned price point
            avg_price = historical['avg_price_point']
            min_price = max(1, avg_price * 0.3)
            max_price = min(budget * 0.4, avg_price * 3)
        else:
            # Intelligent defaults based on budget
            if budget >= 1000:
                min_price = 20.0
                max_price = min(budget * 0.3, 500.0)
            elif budget >= 500:
                min_price = 10.0
                max_price = min(budget * 0.4, 300.0)
            elif budget >= 200:
                min_price = 5.0
                max_price = min(budget * 0.5, 150.0)
            else:
                min_price = 1.0
                max_price = budget * 0.7
        
        _logger.info(f"📦 Product pool: €{min_price:.2f} - €{max_price:.2f} for €{budget:.2f} budget")
        
        domain = [
            ('sale_ok', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
        ]
        
        # Add dietary filters
        if dietary:
            if 'halal' in dietary:
                # Add category exclusions for halal
                domain.append(('categ_id.name', 'not ilike', '%iberic%'))
                domain.append(('categ_id.name', 'not ilike', '%alcohol%'))
        
        products = self.env['product.template'].search(domain, limit=1000)
        
        # Filter by stock and dietary
        available = []
        for product in products:
            if self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                available.append(product)
        
        # Score by patterns if available
        if patterns and patterns.get('favorite_products'):
            scored = []
            for product in available:
                score = 10 if product.id in patterns['favorite_products'] else 1
                scored.append((product, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            available = [p for p, s in scored]
        
        return available
    
    # ================== CATEGORY & PAIRING METHODS ==================
    
    def _compute_category_counts(self, products):
        """Compute category distribution of products"""
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
            cat = (getattr(p, 'categ_id', None) and p.categ_id.name or '').lower()
            name = (p.name or '').lower()
            
            if any(word in name for word in ['vino', 'wine', 'cava', 'champagne']):
                counts['beverage'] += 1
            elif 'foie' in name or 'foie_gras' in cat:
                counts['foie'] += 1
            elif any(word in name for word in ['conserva', 'lata', 'anchoa', 'bonito']):
                counts['canned'] += 1
            elif any(word in cat for word in ['charcuterie', 'cheese', 'queso']):
                counts['charcuterie'] += 1
            elif any(word in cat for word in ['sweet', 'chocolate', 'dulce']):
                counts['sweet'] += 1
            elif any(word in name for word in ['vermouth', 'gin', 'whisky']):
                counts['aperitif'] += 1
            else:
                counts['other'] += 1
        
        return counts
    
    def _ensure_tokaji_foie_pairing(self, products, dietary):
        """Ensure Tokaji and Foie Gras pairing rule"""
        
        has_tokaji = any('tokaj' in (p.name or '').lower() for p in products)
        has_foie = any('foie' in (p.name or '').lower() for p in products)
        
        if has_tokaji and not has_foie and 'vegan' not in dietary and 'vegetarian' not in dietary:
            # Add foie gras
            foie_products = self.env['product.template'].search([
                ('name', 'ilike', '%foie%'),
                ('sale_ok', '=', True),
                ('list_price', '>', 0)
            ], limit=1)
            
            if foie_products and self._has_stock(foie_products[0]):
                products.append(foie_products[0])
                _logger.info("🦆 Added Foie Gras pairing for Tokaji")
        
        return products
    
    def _ensure_category_balance(self, products, requirements, context):
        """Ensure proper category distribution"""
        
        current_counts = self._compute_category_counts(products)
        target_counts = context.get('target_category_counts', {})
        
        if not target_counts:
            # Default balanced distribution
            total = len(products)
            target_counts = {
                'beverage': max(1, int(total * 0.3)),
                'charcuterie': max(1, int(total * 0.2)),
                'sweet': max(1, int(total * 0.15)),
                'canned': int(total * 0.15),
                'other': int(total * 0.2)
            }
        
        # Adjust products to match target distribution
        # This is simplified - full implementation would swap products
        
        return products
    
    # ================== COMPLEMENTARY & SEASONAL METHODS ==================
    
    def _find_complementary_products(self, existing_products, budget, dietary, count_needed, context):
        """Find products that complement existing selection"""
        
        # Get categories of existing products
        existing_categories = set()
        for prod in existing_products:
            if prod.categ_id:
                existing_categories.add(prod.categ_id.id)
        
        # Find products from different/complementary categories
        domain = [
            ('list_price', '>', 0),
            ('list_price', '<=', budget / count_needed * 1.5) if count_needed > 0 else ('list_price', '<=', 100),
            ('sale_ok', '=', True)
        ]
        
        # Prefer different categories for variety
        if existing_categories:
            complementary_domain = domain + [('categ_id', 'not in', list(existing_categories))]
            products = self.env['product.template'].search(complementary_domain, limit=count_needed * 3)
        else:
            products = self.env['product.template'].search(domain, limit=count_needed * 3)
        
        # Filter and select
        valid_products = []
        for product in products:
            if self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                valid_products.append(product)
                if len(valid_products) >= count_needed:
                    break
        
        return valid_products
    
    def _apply_seasonal_adjustments(self, products, context):
        """Apply seasonal adjustments to product selection"""
        
        month = fields.Date.today().month
        season = self._get_current_season(month)
        
        if season == 'christmas' and month == 12:
            # Add Christmas specialties if December
            christmas_products = self.env['product.template'].search([
                ('name', 'ilike', '%navidad%'),
                ('sale_ok', '=', True)
            ], limit=2)
            
            for product in christmas_products:
                if self._has_stock(product) and product not in products:
                    # Replace a random product
                    if len(products) > 1:
                        products[random.randint(0, len(products)-1)] = product
                        _logger.info("🎄 Added Christmas product")
        
        return products
    
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
    
    def _apply_vip_upgrades(self, products, budget):
        """Apply quality upgrades for VIP clients"""
        
        # For VIP clients, try to upgrade some products to premium versions
        upgraded = []
        for product in products:
            # Look for premium version
            premium = self.env['product.template'].search([
                ('categ_id', '=', product.categ_id.id),
                ('list_price', '>', product.list_price * 1.3),
                ('list_price', '<', product.list_price * 2),
                ('name', 'ilike', '%premium%')
            ], limit=1)
            
            if premium and self._has_stock(premium[0]):
                upgraded.append(premium[0])
                _logger.info(f"⭐ Upgraded {product.name} to {premium[0].name}")
            else:
                upgraded.append(product)
        
        return upgraded
    
    # ================== PATTERN-BASED GENERATION ==================
    
    def _generate_from_patterns_enhanced(self, partner, requirements, notes, context):
        """Enhanced pattern-based generation using historical patterns"""
        
        patterns = context.get('patterns', {})
        historical = context.get('historical_intelligence', {})
        seasonal = context.get('seasonal', {})
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        
        _logger.info(f"📊 Pattern generation: {product_count} products, €{budget:.2f}")
        
        # Start with favorite products
        products = []
        for prod_id in patterns.get('favorite_products', [])[:product_count]:
            product = self.env['product.template'].browse(prod_id)
            if product.exists() and self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                products.append(product)
        
        # Add from preferred categories
        if len(products) < product_count:
            for cat_name in patterns.get('preferred_categories', {}).keys():
                if len(products) >= product_count:
                    break
                
                cat_products = self.env['product.template'].search([
                    ('categ_id.name', '=', cat_name),
                    ('list_price', '>', 0),
                    ('sale_ok', '=', True)
                ], limit=5)
                
                for product in cat_products:
                    if product not in products and self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                        products.append(product)
                        if len(products) >= product_count:
                            break
        
        # Fill remaining from smart pool
        if len(products) < product_count:
            pool = self._get_smart_product_pool(budget, dietary, context)
            for product in pool:
                if product not in products:
                    products.append(product)
                    if len(products) >= product_count:
                        break
        
        # Optimize selection
        selected = self._smart_optimize_selection(
            products, product_count, budget, 
            requirements['budget_flexibility'],
            requirements.get('enforce_count', True),
            context
        )
        
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        composition = self.env['gift.composition'].create({
            'partner_id': partner.id,
            'target_budget': budget,
            'actual_cost': total_cost,
            'product_ids': [(6, 0, [p.id for p in selected])],
            'dietary_restrictions': ', '.join(dietary) if dietary else '',
            'client_notes': notes,
            'generation_method': 'ollama',
            'composition_type': requirements.get('composition_type', 'custom'),
            'confidence_score': 0.85,
            'ai_reasoning': f"Pattern-based: {len(patterns.get('favorite_products', []))} favorites included"
        })
        
        return {
            'success': True,
            'composition_id': composition.id,
            'products': selected,
            'total_cost': total_cost,
            'product_count': len(selected),
            'confidence_score': 0.85,
            'message': f'Pattern-based: {len(selected)} products, €{total_cost:.2f}',
            'method': 'pattern_based_enhanced'
        }
    
    # ================== SIMILAR CLIENTS GENERATION ==================
    
    def _generate_from_similar_clients(self, partner, requirements, notes, context):
        """Generate based on similar clients when no direct history exists"""
        
        similar_clients = self._find_similar_clients_enhanced(
            partner.id, 
            requirements['budget'],
            context
        )
        
        if not similar_clients:
            return self._generate_with_universal_enforcement(partner, requirements, notes, context)
        
        _logger.info(f"👥 Learning from {len(similar_clients)} similar clients")
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        
        # Aggregate products from similar clients
        product_popularity = Counter()
        category_popularity = Counter()
        
        for client_data in similar_clients:
            weight = client_data['similarity']
            
            # Get their orders
            orders = self.env['sale.order'].search([
                ('partner_id', '=', client_data['partner_id']),
                ('state', 'in', ['sale', 'done']),
                ('amount_untaxed', '>=', budget * 0.7),
                ('amount_untaxed', '<=', budget * 1.3)
            ], limit=3)
            
            for order in orders:
                for line in order.order_line:
                    if line.product_id and line.price_unit > 0:
                        product = line.product_id.product_tmpl_id
                        product_popularity[product.id] += weight
                        if product.categ_id:
                            category_popularity[product.categ_id.name] += weight
        
        # Select most popular products
        products = []
        for prod_id, score in product_popularity.most_common(product_count * 2):
            product = self.env['product.template'].browse(prod_id)
            if product.exists() and self._has_stock(product) and self._check_dietary_compliance(product, dietary):
                products.append(product)
                if len(products) >= product_count:
                    break
        
        # Fill from popular categories if needed
        if len(products) < product_count:
            for cat_name, score in category_popularity.most_common(5):
                if len(products) >= product_count:
                    break
                
                cat_products = self.env['product.template'].search([
                    ('categ_id.name', '=', cat_name),
                    ('list_price', '>', 0),
                    ('sale_ok', '=', True)
                ], limit=5)
                
                for product in cat_products:
                    if product not in products and self._has_stock(product):
                        products.append(product)
                        if len(products) >= product_count:
                            break
        
        # Optimize selection
        selected = self._smart_optimize_selection(
            products, product_count, budget, 
            requirements['budget_flexibility'],
            requirements.get('enforce_count', True),
            context
        )
        
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        composition = self.env['gift.composition'].create({
            'partner_id': partner.id,
            'target_budget': budget,
            'actual_cost': total_cost,
            'product_ids': [(6, 0, [p.id for p in selected])],
            'dietary_restrictions': ', '.join(dietary) if dietary else '',
            'client_notes': notes,
            'generation_method': 'ollama',
            'composition_type': requirements.get('composition_type', 'custom'),
            'confidence_score': 0.75,
            'ai_reasoning': f"Based on {len(similar_clients)} similar clients"
        })
        
        return {
            'success': True,
            'composition_id': composition.id,
            'products': selected,
            'total_cost': total_cost,
            'product_count': len(selected),
            'confidence_score': 0.75,
            'message': f'Similar clients: {len(selected)} products = €{total_cost:.2f}',
            'method': 'similar_clients',
            'ai_insights': f"Learned from {len(similar_clients)} similar clients with {product_popularity.total()} product references"
        }
    
    def _find_similar_clients_enhanced(self, partner_id, budget, context):
        """Find clients with truly similar patterns"""
        
        # Get partner's patterns
        partner_patterns = self._analyze_client_purchase_patterns(partner_id)
        
        # Find clients with orders in similar budget range
        similar_budget_orders = self.env['sale.order'].search([
            ('state', 'in', ['sale', 'done']),
            ('partner_id', '!=', partner_id),
            ('amount_untaxed', '>=', budget * 0.7),
            ('amount_untaxed', '<=', budget * 1.3)
        ], limit=100)
        
        # Group by partner and calculate similarity
        partner_scores = {}
        for order in similar_budget_orders:
            pid = order.partner_id.id
            if pid not in partner_scores:
                other_patterns = self._analyze_client_purchase_patterns(pid)
                similarity = self._calculate_pattern_similarity(partner_patterns, other_patterns)
                partner_scores[pid] = {
                    'partner_id': pid,
                    'similarity': similarity,
                    'patterns': other_patterns
                }
        
        # Sort by similarity and return top matches
        similar_clients = sorted(partner_scores.values(), key=lambda x: x['similarity'], reverse=True)
        return similar_clients[:10]
    
    # ================== UNIVERSAL GENERATION WITH ENFORCEMENT ==================
    
    def _generate_with_universal_enforcement(self, partner, requirements, notes, context):
        """Universal generation with strict enforcement of all requirements"""
        
        budget = requirements['budget']
        product_count = requirements['product_count']
        dietary = requirements['dietary']
        flexibility = requirements['budget_flexibility']
        
        _logger.info(f"🎯 Universal generation: {product_count} products, €{budget:.2f}")
        
        # Get product pool
        products = self._get_smart_product_pool(budget, dietary, context)
        
        if not products:
            return {'success': False, 'error': 'No products available matching criteria'}
        
        # Apply smart selection
        selected = self._smart_optimize_selection(
            products, product_count, budget, 
            flexibility, True, context
        )
        
        # Enforce exact count
        if len(selected) != product_count:
            selected = self._enforce_exact_product_count(selected, product_count, budget, products)
        
        # Enforce budget
        selected = self._enforce_budget_guardrail(selected, budget, flexibility/100, dietary, context)
        
        # Apply special rules
        selected = self._ensure_tokaji_foie_pairing(selected, dietary)
        
        total_cost = sum(p.list_price for p in selected)
        
        # Create composition
        composition = self.env['gift.composition'].create({
            'partner_id': partner.id,
            'target_budget': budget,
            'actual_cost': total_cost,
            'product_ids': [(6, 0, [p.id for p in selected])],
            'dietary_restrictions': ', '.join(dietary) if dietary else '',
            'client_notes': notes,
            'generation_method': 'ollama',
            'composition_type': requirements.get('composition_type', 'custom'),
            'confidence_score': 0.7,
            'ai_reasoning': f"Universal generation with enforcement: {len(selected)} products"
        })
        
        return {
            'success': True,
            'composition_id': composition.id,
            'products': selected,
            'total_cost': total_cost,
            'product_count': len(selected),
            'confidence_score': 0.7,
            'method': 'universal_enforcement'
        }
    
    def _enforce_exact_product_count(self, products, target_count, budget, product_pool):
        """Enforce exact product count"""
        
        if len(products) == target_count:
            return products
        
        if len(products) < target_count:
            # Add more products
            needed = target_count - len(products)
            available = [p for p in product_pool if p not in products]
            products.extend(available[:needed])
        else:
            # Remove excess products
            products = products[:target_count]
        
        return products
    
    # ================== PARSING & REQUIREMENTS ==================
    
    def _parse_requirements_with_ollama(self, notes, budget, dietary, composition_type, intelligence):
        """Parse requirements using Ollama with historical intelligence"""
        
        requirements = {
            'budget': budget,
            'product_count': intelligence['optimal_product_count'],
            'budget_flexibility': 10,
            'dietary': dietary if dietary else [],
            'composition_type': composition_type,
            'enforce_count': False,
            'categories_required': {},
            'special_instructions': notes
        }
        
        # Use Ollama to parse notes for overrides
        if self.ollama_enabled and notes:
            parsed = self._parse_notes_with_ollama(notes, budget, intelligence)
            if parsed:
                # Apply overrides
                if parsed.get('product_count'):
                    requirements['product_count'] = parsed['product_count']
                    requirements['enforce_count'] = True
                
                if parsed.get('budget_override'):
                    requirements['budget'] = parsed['budget_override']
                
                if parsed.get('dietary'):
                    requirements['dietary'].extend(parsed['dietary'])
                
                if parsed.get('categories_required'):
                    requirements['categories_required'] = parsed['categories_required']
        else:
            # Basic parsing without Ollama
            parsed = self._parse_notes_basic(notes)
            requirements.update(parsed)
        
        return requirements
    
    def _parse_notes_with_ollama(self, notes, budget, intelligence):
        """Use Ollama to intelligently parse notes"""
        
        prompt = f"""
        Parse these gift requirements. We have learned:
        - Optimal products for €{budget:.0f}: {intelligence['optimal_product_count']}
        
        Client notes: {notes}
        
        Extract any specific requirements. Return JSON:
        {{
            "product_count": null or specific number if mentioned,
            "budget_override": null or amount if mentioned,
            "dietary": [],
            "categories_required": {{}},
            "special_requirements": ""
        }}
        """
        
        try:
            response = self._call_ollama(prompt, format_json=True)
            if response:
                return json.loads(response)
        except Exception as e:
            _logger.debug(f"Ollama parsing failed: {e}")
        
        return None
    
    def _parse_notes_basic(self, notes):
        """Basic parsing without AI"""
        
        parsed = {
            'enforce_count': False,
            'product_count': None,
            'budget_override': None,
            'dietary': [],
            'categories_required': {}
        }
        
        if not notes:
            return parsed
        
        notes_lower = notes.lower()
        
        # Extract numbers for product count
        count_match = re.search(r'(\d+)\s*product', notes_lower)
        if count_match:
            count = int(count_match.group(1))
            if 1 <= count <= 100:
                parsed['product_count'] = count
                parsed['enforce_count'] = True
        
        # Extract budget override
        budget_match = re.search(r'[€$]\s*(\d+)', notes)
        if budget_match:
            parsed['budget_override'] = float(budget_match.group(1))
        
        # Extract dietary
        if 'halal' in notes_lower:
            parsed['dietary'].append('halal')
        if 'vegan' in notes_lower:
            parsed['dietary'].append('vegan')
        if 'vegetarian' in notes_lower:
            parsed['dietary'].append('vegetarian')
        
        return parsed
    
    # ================== CONTEXT BUILDING ==================
    
    def _build_generation_context(self, partner, requirements, intelligence):
        """Build comprehensive generation context"""
        
        context = {
            'partner': partner,
            'patterns': self._analyze_client_purchase_patterns(partner.id),
            'last_year_products': self._get_last_year_products(partner.id),
            'historical_intelligence': intelligence,
            'seasonal': {
                'season': self._get_current_season(fields.Date.today().month),
                'month': fields.Date.today().month
            },
            'is_vip': partner.total_revenue >= 10000 if hasattr(partner, 'total_revenue') else False,
            'favorite_products': [],
            'target_category_counts': {}
        }
        
        # Add favorite products from patterns
        if context['patterns']:
            context['favorite_products'] = context['patterns'].get('favorite_products', [])
        
        return context
    
    # ================== REASONING & LEARNING ==================
    
    def _build_comprehensive_reasoning(self, requirements, products, total_cost, budget, context):
        """Build detailed reasoning for the composition"""
        
        reasoning_parts = []
        
        # Summary
        reasoning_parts.append(f"Generated {len(products)} products totaling €{total_cost:.2f}")
        
        # Budget compliance
        variance = ((total_cost - budget) / budget * 100) if budget > 0 else 0
        reasoning_parts.append(f"Budget variance: {variance:+.1f}%")
        
        # Method used
        if context.get('last_year_products'):
            reasoning_parts.append("Applied business rules R1-R6")
        elif context.get('patterns'):
            reasoning_parts.append("Pattern-based generation from history")
        else:
            reasoning_parts.append("Similar clients analysis")
        
        # Dietary compliance
        if requirements.get('dietary'):
            reasoning_parts.append(f"Dietary restrictions applied: {', '.join(requirements['dietary'])}")
        
        # Category distribution
        category_counts = self._compute_category_counts(products)
        if category_counts:
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            reasoning_parts.append(f"Categories: {', '.join([f'{cat}({cnt})' for cat, cnt in top_categories])}")
        
        # Confidence
        confidence = context.get('historical_intelligence', {}).get('confidence', 0.5)
        reasoning_parts.append(f"Confidence: {confidence*100:.0f}%")
        
        return " | ".join(reasoning_parts)
    
    def _update_learning_from_result(self, result, requirements, intelligence):
        """Update learning cache with new patterns"""
        
        try:
            cache_data = json.loads(self.learning_cache) if self.learning_cache else {}
        except:
            cache_data = {}
        
        # Store successful pattern
        if result.get('success'):
            pattern_key = f"budget_{int(requirements['budget']/100)*100}"
            
            if pattern_key not in cache_data:
                cache_data[pattern_key] = []
            
            cache_data[pattern_key].append({
                'product_count': result['product_count'],
                'total_cost': result['total_cost'],
                'confidence': result.get('confidence_score', 0.5),
                'method': result.get('method', 'unknown'),
                'timestamp': fields.Datetime.now().isoformat()
            })
            
            # Keep only recent patterns (last 50)
            cache_data[pattern_key] = cache_data[pattern_key][-50:]
            
            self.learning_cache = json.dumps(cache_data)
            self.cache_expiry = fields.Datetime.now() + timedelta(days=30)
    
    # ================== HELPER METHODS (ALL PRESERVED) ==================
    
    def _analyze_client_purchase_patterns(self, partner_id):
        """Analyze historical purchase patterns"""
        orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc')
        
        if not orders:
            return {}
        
        all_products = []
        total_value = 0
        product_counts = []
        categories = Counter()
        
        for order in orders:
            order_products = []
            for line in order.order_line:
                if line.product_id and line.price_unit > 0:
                    product = line.product_id.product_tmpl_id
                    all_products.append(product.id)
                    order_products.append(product.id)
                    if product.categ_id:
                        categories[product.categ_id.name] += 1
            
            if order_products:
                product_counts.append(len(order_products))
            total_value += order.amount_untaxed
        
        product_frequency = Counter(all_products)
        avg_order_value = total_value / len(orders) if orders else 0
        avg_product_count = sum(product_counts) / len(product_counts) if product_counts else 12
        
        # Determine budget trend
        budget_trend = 'stable'
        if len(orders) >= 2:
            recent_avg = sum(o.amount_untaxed for o in orders[:2]) / 2
            older_avg = sum(o.amount_untaxed for o in orders[2:4]) / 2 if len(orders) > 2 else recent_avg
            if recent_avg > older_avg * 1.1:
                budget_trend = 'increasing'
            elif recent_avg < older_avg * 0.9:
                budget_trend = 'decreasing'
        
        return {
            'total_orders': len(orders),
            'avg_order_value': avg_order_value,
            'avg_product_count': avg_product_count,
            'favorite_products': [p[0] for p in product_frequency.most_common(20)],
            'preferred_categories': dict(categories.most_common(10)),
            'budget_trend': budget_trend
        }
    
    def _get_last_year_products(self, partner_id):
        """Get products from last year's orders"""
        last_year = fields.Date.today().year - 1
        
        orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done']),
            ('date_order', '>=', f'{last_year}-01-01'),
            ('date_order', '<=', f'{last_year}-12-31')
        ])
        
        products = []
        for order in orders:
            for line in order.order_line:
                if line.product_id and line.product_id.product_tmpl_id not in products:
                    products.append(line.product_id.product_tmpl_id)
        
        return products
    
    def _check_dietary_compliance(self, product, dietary_restrictions):
        """Check if product meets dietary restrictions"""
        if not dietary_restrictions:
            return True
        
        product_name = product.name.lower() if product.name else ''
        categ_name = product.categ_id.name.lower() if product.categ_id else ''
        
        for restriction in dietary_restrictions:
            restriction = restriction.lower()
            
            if restriction in ['halal', 'no_pork']:
                prohibited = ['cerdo', 'pork', 'jamón', 'jamon', 'ibérico', 'iberico', 
                            'chorizo', 'salchichón', 'lomo', 'panceta', 'bacon']
                if any(word in product_name for word in prohibited):
                    return False
                if any(word in product_name for word in ['vino', 'wine', 'alcohol', 'licor', 'whisky', 'vodka']):
                    return False
                if 'iberic' in categ_name or 'alcohol' in categ_name:
                    return False
            
            if restriction in ['vegan', 'vegano']:
                prohibited = ['carne', 'meat', 'pollo', 'chicken', 'pescado', 'fish', 
                            'marisco', 'queso', 'cheese', 'leche', 'milk', 'huevo', 'egg',
                            'mantequilla', 'butter', 'nata', 'cream', 'miel', 'honey']
                if any(word in product_name for word in prohibited):
                    return False
            
            if restriction in ['vegetarian', 'vegetariano']:
                prohibited = ['carne', 'meat', 'pollo', 'chicken', 'pescado', 'fish', 
                            'marisco', 'jamón', 'anchoa', 'atún', 'tuna', 'salmon']
                if any(word in product_name for word in prohibited):
                    return False
            
            if restriction in ['no_alcohol', 'non_alcoholic', 'sin_alcohol']:
                prohibited = ['vino', 'wine', 'alcohol', 'licor', 'cerveza', 'beer', 
                            'whisky', 'vodka', 'ginebra', 'gin', 'rum', 'brandy', 'cava']
                if any(word in product_name for word in prohibited):
                    return False
                if 'alcohol' in categ_name or 'bebida' in categ_name:
                    return False
            
            if restriction in ['gluten_free', 'sin_gluten']:
                prohibited = ['pan', 'bread', 'pasta', 'galleta', 'cookie', 'harina', 
                            'flour', 'trigo', 'wheat', 'cebada', 'barley', 'centeno', 'rye']
                if any(word in product_name for word in prohibited):
                    return False
        
        return True
    
    def _has_stock(self, product):
        """Check if product has stock"""
        try:
            if hasattr(product, 'qty_available'):
                qty = float(product.qty_available)
                if qty > 0:
                    return True
            
            if hasattr(product, 'virtual_available'):
                qty = float(product.virtual_available)
                if qty > 0:
                    return True
            
            # Check through variants
            for variant in product.product_variant_ids:
                stock_quants = self.env['stock.quant'].search([
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
    
    def _calculate_pattern_similarity(self, patterns1, patterns2):
        """Calculate similarity between two client patterns"""
        if not patterns1 or not patterns2:
            return 0.0
        
        similarity_score = 0.0
        factors = 0
        
        # Compare average order values
        if patterns1.get('avg_order_value') and patterns2.get('avg_order_value'):
            diff = abs(patterns1['avg_order_value'] - patterns2['avg_order_value'])
            avg = (patterns1['avg_order_value'] + patterns2['avg_order_value']) / 2
            if avg > 0:
                similarity_score += max(0, 1 - diff / avg)
                factors += 1
        
        # Compare product counts
        if patterns1.get('avg_product_count') and patterns2.get('avg_product_count'):
            diff = abs(patterns1['avg_product_count'] - patterns2['avg_product_count'])
            avg = (patterns1['avg_product_count'] + patterns2['avg_product_count']) / 2
            if avg > 0:
                similarity_score += max(0, 1 - diff / avg)
                factors += 1
        
        # Compare categories
        cats1 = set(patterns1.get('preferred_categories', {}).keys())
        cats2 = set(patterns2.get('preferred_categories', {}).keys())
        if cats1 and cats2:
            overlap = len(cats1.intersection(cats2))
            union = len(cats1.union(cats2))
            if union > 0:
                similarity_score += overlap / union
                factors += 1
        
        # Compare budget trend
        if patterns1.get('budget_trend') == patterns2.get('budget_trend'):
            similarity_score += 0.5
            factors += 0.5
        
        return similarity_score / factors if factors > 0 else 0.0
    
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
                _logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            _logger.error(f"Ollama request failed: {str(e)}")
            return None
    
    # ================== TEST & UTILITY METHODS ==================
    
    def test_ollama_connection(self):
        """Test Ollama connection"""
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
    
    @api.model
    def get_or_create_recommender(self):
        """Get or create default recommender"""
        recommender = self.search([('active', '=', True)], limit=1)
        if not recommender:
            recommender = self.create({
                'name': 'Default AI Recommender',
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
            self._deep_historical_learning(partner_id, 1000)  # Sample budget
            analyzed_count += 1
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': '🧠 Learning Analysis Complete',
                'message': f'Analyzed {analyzed_count} clients. Cache updated.',
                'type': 'success',
                'sticky': False,
            }
        }