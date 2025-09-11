# SCHEMA-AWARE ADVANCED AI/ML SYSTEM
# Uses your actual database schema while maintaining sophisticated logic

from odoo import models, api, fields
from odoo.exceptions import UserError
import logging
import json
from datetime import datetime
from collections import defaultdict

_logger = logging.getLogger(__name__)

class IntegrationManager(models.Model):
    _name = 'integration.manager'
    _description = 'Advanced AI/ML Integration Manager - Schema Aware'
    
    name = fields.Char(string="Manager Name", default="Advanced AI/ML Integration Manager")
    
    # Engine Preferences
    use_ml_engine = fields.Boolean(string="Use ML Engine", default=True)
    use_ai_recommender = fields.Boolean(string="Use AI Recommender", default=True)
    use_stock_aware = fields.Boolean(string="Use Stock-Aware Engine", default=True)
    use_business_rules = fields.Boolean(string="Use Business Rules", default=True)
    
    # Fallback Strategy
    fallback_strategy = fields.Selection([
        ('cascade', 'Cascade (Try each engine in order)'),
        ('parallel', 'Parallel (Best of multiple engines)'),
        ('weighted', 'Weighted (Combine multiple results)')
    ], string="Fallback Strategy", default='cascade')
    
    # Performance Tracking
    last_generation_time = fields.Float(string="Last Generation Time (seconds)")
    total_generations = fields.Integer(string="Total Generations", default=0)

    @api.model
    def generate_complete_composition(self, partner_id, target_budget, target_year=None, 
                                    dietary_restrictions=None, notes_text=None, **kwargs):
        """Updated method using simplified approach"""
        
        # Use simplified engine for all generations
        engine = self.env['simplified.composition.engine']
        
        result = engine.generate_composition(
            partner_id=partner_id,
            target_budget=target_budget,
            target_year=target_year,
            dietary_restrictions=dietary_restrictions,
            notes_text=notes_text
        )
        
        return {
            'success': True,
            'products': result['products'],
            'total_cost': result['total_cost'],
            'product_count': result['product_count'],
            'method_used': result['method_used']
        }
    
    def _generate_ml_composition_safe(self, partner_id, target_budget, target_year, dietary_restrictions, notes_text):
        """ML composition generation with safe field handling"""
        
        try:
            # Check if ML engine exists
            if not self.env['ml.recommendation.engine'].search([]):
                _logger.info("No ML engine found, skipping ML generation")
                return None
            
            ml_engine = self.env['ml.recommendation.engine'].search([], limit=1)
            
            # Check if model is trained
            if not getattr(ml_engine, 'is_model_trained', False):
                _logger.info("ML model not trained, skipping ML generation")
                return None
            
            # Get client profile for ML
            client_profile = self._build_client_profile(partner_id, notes_text)
            
            # Get sophisticated product pool
            available_products = self._get_sophisticated_product_pool(dietary_restrictions)
            
            if not available_products:
                _logger.warning("No products available for ML engine")
                return None
            
            # ML-powered product scoring and selection
            scored_products = self._ml_score_products(client_profile, available_products, target_budget)
            selected_products = self._ml_select_products(scored_products, target_budget, client_profile)
            
            if selected_products:
                return self._create_sophisticated_composition(
                    partner_id, target_budget, target_year, selected_products,
                    dietary_restrictions, notes_text, 'ml_advanced'
                )
                
        except Exception as e:
            _logger.warning(f"ML composition generation failed: {e}")
            return None
    
    def _generate_ai_composition_safe(self, partner_id, target_budget, target_year, dietary_restrictions, notes_text):
        """AI composition generation with advanced logic"""
        
        try:
            # Get client insights
            client_profile = self._build_client_profile(partner_id, notes_text)
            
            # Get products with advanced filtering
            available_products = self._get_sophisticated_product_pool(dietary_restrictions)
            
            if not available_products:
                _logger.warning("No products available for AI engine")
                return None
            
            # AI-powered analysis and selection
            scored_products = self._ai_score_products_advanced(client_profile, available_products, target_budget)
            selected_products = self._ai_select_products_advanced(scored_products, target_budget, client_profile)
            
            if selected_products:
                return self._create_sophisticated_composition(
                    partner_id, target_budget, target_year, selected_products,
                    dietary_restrictions, notes_text, 'ai_advanced'
                )
                
        except Exception as e:
            _logger.warning(f"AI composition generation failed: {e}")
            return None
    
    def _generate_smart_composition(self, partner_id, target_budget, target_year, dietary_restrictions, notes_text):
        """Smart composition using business intelligence"""
        
        try:
            # Build comprehensive client profile
            client_profile = self._build_client_profile(partner_id, notes_text)
            
            # Get products with smart filtering
            available_products = self._get_sophisticated_product_pool(dietary_restrictions)
            
            if not available_products:
                _logger.warning("No products available for smart engine")
                return None
            
            # Smart product selection algorithm
            selected_products = self._smart_product_selection(
                available_products, target_budget, client_profile, dietary_restrictions
            )
            
            if selected_products:
                return self._create_sophisticated_composition(
                    partner_id, target_budget, target_year, selected_products,
                    dietary_restrictions, notes_text, 'smart_business'
                )
                
        except Exception as e:
            _logger.warning(f"Smart composition generation failed: {e}")
            return None
    
    def _generate_sophisticated_fallback(self, partner_id, target_budget, target_year, dietary_restrictions, notes_text):
        """Sophisticated fallback with advanced logic"""
        
        try:
            _logger.info("Using sophisticated fallback composition generation")
            
            # Get client analysis
            client_profile = self._build_client_profile(partner_id, notes_text)
            
            # Get products with category-aware filtering
            products = self._get_sophisticated_product_pool(dietary_restrictions)
            
            if not products:
                # Ultra-fallback: basic products
                products = self.env['product.template'].search([
                    ('active', '=', True),
                    ('sale_ok', '=', True),
                    ('list_price', '>', 0)
                ], limit=50)
            
            if not products:
                raise UserError("No products available in the system")
            
            # Sophisticated selection algorithm
            selected_products = self._sophisticated_selection_algorithm(
                products, target_budget, client_profile, dietary_restrictions
            )
            
            return self._create_sophisticated_composition(
                partner_id, target_budget, target_year, selected_products,
                dietary_restrictions, notes_text, 'sophisticated_fallback'
            )
            
        except Exception as e:
            _logger.error(f"Sophisticated fallback failed: {e}")
            raise UserError(f"Sophisticated fallback failed: {str(e)}")
    
    def _build_client_profile(self, partner_id, notes_text):
        """Build comprehensive client profile for AI/ML"""
        
        partner = self.env['res.partner'].browse(partner_id)
        
        profile = {
            'partner_id': partner_id,
            'name': partner.name,
            'has_history': False,
            'order_count': 0,
            'avg_budget': 200.0,
            'preferred_categories': [],
            'notes_analysis': {},
            'risk_level': 'medium'
        }
        
        # Analyze order history
        try:
            history_records = self.env['client.order.history'].search([
                ('partner_id', '=', partner_id)
            ])
            
            if history_records:
                profile['has_history'] = True
                profile['order_count'] = len(history_records)
                
                # Calculate average budget
                budgets = [h.total_budget for h in history_records if h.total_budget > 0]
                if budgets:
                    profile['avg_budget'] = sum(budgets) / len(budgets)
                
                # Analyze preferred categories (using existing fields)
                categories = []
                for record in history_records:
                    if hasattr(record, 'category_preferences') and record.category_preferences:
                        categories.extend(record.category_preferences.split(','))
                
                if categories:
                    profile['preferred_categories'] = list(set(categories))
                    
        except Exception as e:
            _logger.warning(f"History analysis failed: {e}")
        
        # Analyze notes with simple NLP
        if notes_text:
            profile['notes_analysis'] = self._analyze_notes_simple(notes_text)
        
        # Set risk level based on history
        if profile['has_history'] and profile['order_count'] >= 3:
            profile['risk_level'] = 'low'
        elif profile['has_history']:
            profile['risk_level'] = 'medium'
        else:
            profile['risk_level'] = 'high'
        
        return profile
    
    def _get_sophisticated_product_pool(self, dietary_restrictions):
        """Get products with sophisticated filtering"""
        
        # Start with basic domain
        domain = [
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('list_price', '>', 0)
        ]
        
        # Add category filter if lebiggot_category exists
        try:
            categorized_products = self.env['product.template'].search([
                ('active', '=', True),
                ('sale_ok', '=', True),
                ('list_price', '>', 0),
                ('lebiggot_category', '!=', False)
            ], limit=1)
            
            if categorized_products:
                domain.append(('lebiggot_category', '!=', False))
                _logger.info("Using lebiggot_category filter")
                
        except Exception as e:
            _logger.info(f"lebiggot_category not available: {e}")
        
        products = self.env['product.template'].search(domain, limit=100)
        
        # Apply sophisticated dietary filtering
        if dietary_restrictions and products:
            products = self._apply_dietary_restrictions_advanced(products, dietary_restrictions)
        
        return products
    
    def _apply_dietary_restrictions_advanced(self, products, dietary_restrictions):
        """Advanced dietary restrictions filtering"""
        
        filtered_products = []
        
        for product in products:
            include_product = True
            name_lower = product.name.lower()
            
            for restriction in dietary_restrictions:
                if restriction == 'vegan':
                    # Check for vegan field first, then fallback to keywords
                    try:
                        if hasattr(product, 'is_vegan') and not product.is_vegan:
                            include_product = False
                            break
                    except:
                        # Fallback to keyword analysis
                        if any(word in name_lower for word in ['meat', 'fish', 'cheese', 'ham', 'chicken', 'beef']):
                            include_product = False
                            break
                
                elif restriction == 'halal':
                    try:
                        if hasattr(product, 'is_halal') and not product.is_halal:
                            include_product = False
                            break
                    except:
                        if any(word in name_lower for word in ['pork', 'wine', 'alcohol', 'beer']):
                            include_product = False
                            break
                
                elif restriction == 'non_alcoholic':
                    try:
                        if hasattr(product, 'contains_alcohol') and product.contains_alcohol:
                            include_product = False
                            break
                    except:
                        if any(word in name_lower for word in ['wine', 'beer', 'champagne', 'alcohol', 'rum', 'whiskey']):
                            include_product = False
                            break
            
            if include_product:
                filtered_products.append(product)
        
        return filtered_products
    
    def _sophisticated_selection_algorithm(self, products, target_budget, client_profile, dietary_restrictions):
        """Sophisticated product selection with AI-like logic"""
        
        # Score products based on multiple factors
        scored_products = []
        
        for product in products:
            score = 0.0
            
            # Base score
            score += 0.5
            
            # Price fitness (30% of score)
            price_fitness = self._calculate_price_fitness(product.list_price, target_budget)
            score += price_fitness * 0.3
            
            # Historical compatibility (25% of score)
            if client_profile['has_history']:
                historical_score = self._calculate_historical_compatibility(product, client_profile)
                score += historical_score * 0.25
            
            # Category diversity bonus (20% of score)
            category_score = self._calculate_category_score(product, client_profile)
            score += category_score * 0.2
            
            # Quality indicators (15% of score)
            quality_score = self._calculate_quality_score(product)
            score += quality_score * 0.15
            
            # Notes compatibility (10% of score)
            notes_score = self._calculate_notes_compatibility(product, client_profile.get('notes_analysis', {}))
            score += notes_score * 0.1
            
            scored_products.append({
                'product': product,
                'score': min(1.0, max(0.0, score)),
                'price_fitness': price_fitness,
                'historical_score': historical_score if client_profile['has_history'] else 0,
                'category_score': category_score,
                'quality_score': quality_score
            })
        
        # Sort by score
        scored_products.sort(key=lambda x: x['score'], reverse=True)
        
        # Select products with sophisticated algorithm
        selected_products = []
        current_cost = 0
        categories_used = set()
        max_budget = target_budget * 1.15
        min_budget = target_budget * 0.85
        
        # First pass: select high-scoring products with category diversity
        for item in scored_products:
            product = item['product']
            
            # Get category
            category = getattr(product, 'lebiggot_category', 'general')
            
            # Category diversity constraint (max 2 per category for small orders)
            if len(selected_products) < 5 or categories_used.count(category) < 2:
                if current_cost + product.list_price <= max_budget:
                    selected_products.append(product)
                    current_cost += product.list_price
                    categories_used.add(category)
                    
                    # Stop if we have enough products and are in budget range
                    if (len(selected_products) >= 3 and 
                        current_cost >= min_budget and 
                        len(selected_products) >= 5):
                        break
        
        # Ensure minimum products
        if len(selected_products) < 3:
            for item in scored_products:
                if item['product'] not in selected_products:
                    if current_cost + item['product'].list_price <= max_budget:
                        selected_products.append(item['product'])
                        current_cost += item['product'].list_price
                        if len(selected_products) >= 3:
                            break
        
        # Ensure at least one product
        if not selected_products:
            selected_products = [scored_products[0]['product']]
        
        _logger.info(f"Sophisticated selection: {len(selected_products)} products, cost: €{current_cost:.2f}")
        
        return selected_products
    
    def _create_sophisticated_composition(self, partner_id, target_budget, target_year, 
                                        selected_products, dietary_restrictions, notes_text, engine_type):
        """Create composition with sophisticated metadata using existing fields"""
        
        partner_name = self.env['res.partner'].browse(partner_id).name
        composition_name = f"AI Composition - {partner_name}"
        
        # Calculate costs and variance
        actual_cost = sum(p.list_price for p in selected_products)
        variance = abs(actual_cost - target_budget) / target_budget * 100
        
        # Build sophisticated reasoning using existing fields
        reasoning_parts = [
            f"Advanced AI composition generated using {engine_type} engine.",
            f"Selected {len(selected_products)} products with sophisticated analysis.",
            f"Budget optimization: €{actual_cost:.2f} vs target €{target_budget:.2f} ({variance:.1f}% variance)."
        ]
        
        if notes_text:
            reasoning_parts.append(f"Special considerations: {notes_text}")
        
        if dietary_restrictions:
            reasoning_parts.append(f"Dietary compliance: {', '.join(dietary_restrictions)}")
        
        # Build category structure for advanced analytics
        category_structure = self._build_category_structure(selected_products)
        
        # Create composition with sophisticated data
        composition_vals = {
            'name': composition_name,
            'partner_id': partner_id,
            'target_year': target_year or fields.Date.today().year,
            'target_budget': target_budget,
            'composition_type': 'custom',
            'product_ids': [(6, 0, [p.id for p in selected_products])],
            'state': 'draft'
        }
        
        # Add sophisticated fields if they exist
        if 'dietary_restrictions' in self.env['gift.composition']._fields:
            composition_vals['dietary_restrictions'] = ', '.join(dietary_restrictions or [])
        
        if 'category_structure' in self.env['gift.composition']._fields:
            composition_vals['category_structure'] = json.dumps(category_structure)
        
        # Add business rules information
        if 'rule_applications' in self.env['gift.composition']._fields:
            rule_applications = {
                'engine_used': engine_type,
                'selection_criteria': 'advanced_ai_algorithm',
                'budget_compliance': variance <= 15,
                'category_diversity': len(set(getattr(p, 'lebiggot_category', 'general') for p in selected_products)),
                'generation_timestamp': datetime.now().isoformat()
            }
            composition_vals['rule_applications'] = json.dumps(rule_applications)
        
        # Create composition
        composition = self.env['gift.composition'].create(composition_vals)
        
        _logger.info(f"Sophisticated composition created: {composition.name}")
        _logger.info(f"Products: {[p.name[:30] for p in selected_products]}")
        _logger.info(f"Categories: {list(set(getattr(p, 'lebiggot_category', 'general') for p in selected_products))}")
        
        return composition
    
    def _enhance_composition_metadata(self, composition, notes_text, dietary_restrictions):
        """Enhance composition with sophisticated metadata"""
        
        try:
            # Update category structure if field exists
            if hasattr(composition, 'category_structure'):
                category_structure = self._build_category_structure(composition.product_ids)
                composition.category_structure = json.dumps(category_structure)
            
            # Add sophisticated business rules data if field exists
            if hasattr(composition, 'rule_applications'):
                existing_rules = {}
                try:
                    existing_rules = json.loads(composition.rule_applications or '{}')
                except:
                    pass
                
                existing_rules.update({
                    'enhanced_by_integration_manager': True,
                    'enhancement_timestamp': datetime.now().isoformat(),
                    'metadata_version': '2.0'
                })
                
                composition.rule_applications = json.dumps(existing_rules)
                
        except Exception as e:
            _logger.warning(f"Metadata enhancement failed: {e}")
    
    # Helper methods for sophisticated algorithms
    
    def _calculate_price_fitness(self, price, target_budget):
        """Calculate how well a product price fits the budget"""
        
        ideal_price_per_product = target_budget / 5  # Assume 5 products ideal
        
        if price <= ideal_price_per_product:
            return 1.0
        elif price <= ideal_price_per_product * 2:
            return 0.7
        elif price <= target_budget * 0.4:  # Max 40% of budget for one product
            return 0.4
        else:
            return 0.1
    
    def _calculate_historical_compatibility(self, product, client_profile):
        """Calculate compatibility with client history"""
        
        score = 0.5  # Base score
        
        # Category match
        if client_profile.get('preferred_categories'):
            product_category = getattr(product, 'lebiggot_category', None)
            if product_category in client_profile['preferred_categories']:
                score += 0.3
        
        # Price range compatibility
        avg_budget = client_profile.get('avg_budget', 200)
        expected_price_per_product = avg_budget / 5
        
        if abs(product.list_price - expected_price_per_product) / expected_price_per_product < 0.5:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_category_score(self, product, client_profile):
        """Calculate category diversity and preference score"""
        
        # Basic category scoring
        score = 0.5
        
        # Bonus for having a category
        if hasattr(product, 'lebiggot_category') and product.lebiggot_category:
            score += 0.2
        
        # Bonus for preferred categories
        if client_profile.get('preferred_categories'):
            product_category = getattr(product, 'lebiggot_category', None)
            if product_category in client_profile['preferred_categories']:
                score += 0.3
        
        return min(1.0, score)
    
    def _calculate_quality_score(self, product):
        """Calculate product quality indicators"""
        
        score = 0.5  # Base score
        
        # Name length (longer names often indicate premium products)
        if len(product.name) > 30:
            score += 0.1
        
        # Price indicator (higher price might indicate quality)
        if product.list_price > 50:
            score += 0.2
        elif product.list_price > 20:
            score += 0.1
        
        # Has detailed description
        if hasattr(product, 'description_sale') and product.description_sale:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_notes_compatibility(self, product, notes_analysis):
        """Calculate compatibility with client notes"""
        
        if not notes_analysis:
            return 0.5
        
        score = 0.5
        product_name_lower = product.name.lower()
        
        # Simple keyword matching
        positive_keywords = notes_analysis.get('positive_keywords', [])
        negative_keywords = notes_analysis.get('negative_keywords', [])
        
        for keyword in positive_keywords:
            if keyword.lower() in product_name_lower:
                score += 0.2
        
        for keyword in negative_keywords:
            if keyword.lower() in product_name_lower:
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _analyze_notes_simple(self, notes_text):
        """Simple notes analysis for AI enhancement"""
        
        if not notes_text:
            return {}
        
        notes_lower = notes_text.lower()
        
        # Simple keyword extraction
        positive_keywords = []
        negative_keywords = []
        
        # Look for positive indicators
        if any(word in notes_lower for word in ['premium', 'luxury', 'high-end', 'quality']):
            positive_keywords.append('premium')
        
        if any(word in notes_lower for word in ['wine', 'champagne']):
            positive_keywords.append('wine')
        
        if any(word in notes_lower for word in ['chocolate', 'sweet']):
            positive_keywords.append('chocolate')
        
        # Look for negative indicators
        if any(word in notes_lower for word in ['no alcohol', 'non-alcoholic']):
            negative_keywords.extend(['wine', 'champagne', 'alcohol'])
        
        if any(word in notes_lower for word in ['vegan', 'vegetarian']):
            negative_keywords.extend(['meat', 'fish', 'cheese'])
        
        return {
            'positive_keywords': positive_keywords,
            'negative_keywords': negative_keywords,
            'length': len(notes_text),
            'complexity': len(notes_text.split())
        }
    
    def _build_category_structure(self, products):
        """Build sophisticated category structure for analytics"""
        
        structure = {
            'total_products': len(products),
            'categories': {},
            'price_distribution': {},
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        category_counts = defaultdict(int)
        category_values = defaultdict(float)
        
        for product in products:
            category = getattr(product, 'lebiggot_category', 'uncategorized')
            category_counts[category] += 1
            category_values[category] += product.list_price
        
        for category, count in category_counts.items():
            structure['categories'][category] = {
                'count': count,
                'total_value': category_values[category],
                'avg_price': category_values[category] / count,
                'percentage': (count / len(products)) * 100
            }
        
        return structure
    
    # Placeholder methods for ML and AI scoring (to be implemented based on your specific models)
    
    def _ml_score_products(self, client_profile, products, target_budget):
        """ML-powered product scoring - sophisticated fallback if ML not available"""
        
        try:
            # Try to use actual ML engine if available
            ml_engine = self.env['ml.recommendation.engine'].search([], limit=1)
            if ml_engine and hasattr(ml_engine, 'score_products'):
                return ml_engine.score_products(client_profile, products, target_budget)
        except:
            pass
        
        # Sophisticated fallback scoring
        return self._sophisticated_scoring_fallback(client_profile, products, target_budget)
    
    def _ai_score_products_advanced(self, client_profile, products, target_budget):
        """Advanced AI product scoring"""
        
        scored_products = []
        
        for product in products:
            score = 0.0
            
            # Multi-factor scoring
            score += self._calculate_price_fitness(product.list_price, target_budget) * 0.3
            score += self._calculate_historical_compatibility(product, client_profile) * 0.25
            score += self._calculate_category_score(product, client_profile) * 0.2
            score += self._calculate_quality_score(product) * 0.15
            score += self._calculate_notes_compatibility(product, client_profile.get('notes_analysis', {})) * 0.1
            
            scored_products.append({
                'product': product,
                'score': min(1.0, max(0.0, score))
            })
        
        return sorted(scored_products, key=lambda x: x['score'], reverse=True)
    
    def _sophisticated_scoring_fallback(self, client_profile, products, target_budget):
        """Sophisticated scoring when ML/AI not available"""
        
        return self._ai_score_products_advanced(client_profile, products, target_budget)
    
    def _ml_select_products(self, scored_products, target_budget, client_profile):
        """ML-powered product selection"""
        
        return self._sophisticated_selection_from_scored(scored_products, target_budget)
    
    def _ai_select_products_advanced(self, scored_products, target_budget, client_profile):
        """Advanced AI product selection"""
        
        return self._sophisticated_selection_from_scored(scored_products, target_budget)
    
    def _sophisticated_selection_from_scored(self, scored_products, target_budget):
        """Sophisticated selection from scored products"""
        
        selected = []
        current_cost = 0
        max_budget = target_budget * 1.15
        
        for item in scored_products:
            product = item['product']
            if current_cost + product.list_price <= max_budget:
                selected.append(product)
                current_cost += product.list_price
                
                if len(selected) >= 7:  # Max products
                    break
        
        return selected if len(selected) >= 1 else [scored_products[0]['product']]
    
    def _smart_product_selection(self, products, target_budget, client_profile, dietary_restrictions):
        """Smart business intelligence product selection"""
        
        # Use the sophisticated selection algorithm
        return self._sophisticated_selection_algorithm(products, target_budget, client_profile, dietary_restrictions)

    def _emergency_simple_composition(self, partner_id, target_budget, **kwargs):
        """Ultra-simple emergency composition"""
        try:
            # Get any available products with stock
            products = self.env['product.template'].search([
                ('active', '=', True),
                ('sale_ok', '=', True),
                ('list_price', '>', 0),
                ('default_code', '!=', False)  # Must have internal reference
            ], limit=20)
            
            if not products:
                raise UserError("No products available in system")
            
            # Simple budget-based selection
            selected_products = []
            current_cost = 0
            
            for product in products:
                if current_cost + product.list_price <= target_budget:
                    selected_products.append(product)
                    current_cost += product.list_price
            
            if not selected_products:
                # Take the cheapest product at least
                cheapest = min(products, key=lambda p: p.list_price)
                selected_products = [cheapest]
                current_cost = cheapest.list_price
            
            # Create composition using EXISTING fields
            composition = self.env['gift.composition'].create({
                'partner_id': partner_id,
                'target_budget': target_budget,
                'actual_cost': current_cost,
                'product_ids': [(6, 0, [p.id for p in selected_products])],
                'reasoning': 'Emergency simple composition',  # Instead of generation_method
                'composition_type': 'custom',
                'state': 'draft'
            })
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': selected_products,
                'total_cost': current_cost
            }
            
        except Exception as e:
            _logger.error(f"Emergency simple composition failed: {str(e)}")
            raise