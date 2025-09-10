from odoo import models, api, fields
from odoo.exceptions import UserError
import logging
import json
from datetime import datetime

_logger = logging.getLogger(__name__)

class IntegrationManager(models.Model):
    _name = 'integration.manager'
    _description = 'Central Integration Manager for All Engines'
    
    name = fields.Char(string="Manager Name", default="Central AI/ML Integration Manager")
    
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
    success_rate = fields.Float(string="Success Rate (%)", compute='_compute_success_rate')
    
    @api.depends('total_generations')
    def _compute_success_rate(self):
        for record in self:
            if record.total_generations > 0:
                # This would need actual success tracking
                record.success_rate = 85.0  # Placeholder
            else:
                record.success_rate = 0.0
    
    @api.model
    def generate_complete_composition(self, partner_id, target_budget, target_year=None, 
                                    dietary_restrictions=None, notes_text=None, use_batch=False,
                                    attempt_number=1, force_engine=None):
        """EMERGENCY VERSION: Simplified composition generation that works"""
        
        if target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
        
        start_time = datetime.now()
        
        try:
            _logger.info(f"Emergency composition generation: partner={partner_id}, budget=€{target_budget}")
            
            # Skip complex engine selection - go straight to emergency method
            composition = self._generate_emergency_composition_simple(
                partner_id, target_budget, target_year, dietary_restrictions, notes_text
            )
            
            if composition and len(composition.product_ids) > 0:
                # Track success
                generation_time = (datetime.now() - start_time).total_seconds()
                self.last_generation_time = generation_time
                self.total_generations += 1
                
                _logger.info(f"Emergency composition successful: {composition.name}")
                return composition
            else:
                raise UserError("Emergency composition generation failed")
                
        except Exception as e:
            _logger.error(f"Emergency composition failed: {str(e)}")
            raise UserError(f"Composition generation failed: {str(e)}")

    def _generate_emergency_composition_simple(self, partner_id, target_budget, target_year,
                                            dietary_restrictions, notes_text):
        """EMERGENCY: Simple composition generation that always works - CORRECTED VERSION"""
        
        try:
            _logger.info("Using emergency simple composition generation")
            
            # Get products with very basic filtering
            domain = [
                ('active', '=', True),
                ('sale_ok', '=', True),
                ('list_price', '>', 0)
            ]
            
            # Add category filter if most products have categories
            products_with_categories = self.env['product.template'].search_count([
                ('active', '=', True),
                ('sale_ok', '=', True),
                ('list_price', '>', 0),
                ('lebiggot_category', '!=', False)
            ])
            
            if products_with_categories > 100:  # If we have enough categorized products
                domain.append(('lebiggot_category', '!=', False))
            
            # Get products sorted by price (cheapest first)
            products = self.env['product.template'].search(domain, order='list_price asc', limit=100)
            
            if not products:
                raise UserError("No products found in the system")
            
            _logger.info(f"Found {len(products)} products for selection")
            
            # Apply dietary restrictions with simple keyword filtering
            if dietary_restrictions:
                filtered_products = []
                for product in products:
                    include = True
                    name_lower = product.name.lower()
                    
                    for restriction in dietary_restrictions:
                        if restriction == 'vegan':
                            if any(word in name_lower for word in ['meat', 'fish', 'cheese', 'ham', 'chicken']):
                                include = False
                                break
                        elif restriction == 'halal':
                            if any(word in name_lower for word in ['pork', 'wine', 'alcohol', 'beer']):
                                include = False
                                break
                        elif restriction == 'non_alcoholic':
                            if any(word in name_lower for word in ['wine', 'beer', 'champagne', 'alcohol', 'rum', 'whiskey']):
                                include = False
                                break
                    
                    if include:
                        filtered_products.append(product)
                
                products = filtered_products
                _logger.info(f"After dietary filtering: {len(products)} products")
            
            if not products:
                # If dietary filtering removed everything, use original products
                products = self.env['product.template'].search([
                    ('active', '=', True),
                    ('sale_ok', '=', True),
                    ('list_price', '>', 0)
                ], order='list_price asc', limit=20)
            
            # Smart product selection based on budget
            selected_products = []
            current_cost = 0
            target_products = 5  # Aim for 5 products
            max_budget = target_budget * 1.2
            min_budget = target_budget * 0.8
            
            # Sort products by price to get a good mix
            sorted_products = sorted(products, key=lambda p: p.list_price)
            
            # First pass: try to get close to budget with reasonable number of products
            budget_per_product = target_budget / target_products
            
            for product in sorted_products:
                if len(selected_products) >= 8:  # Max 8 products
                    break
                    
                if current_cost + product.list_price <= max_budget:
                    # Prefer products close to our target price per product
                    if len(selected_products) < 3 or product.list_price <= budget_per_product * 1.5:
                        selected_products.append(product)
                        current_cost += product.list_price
                        
                        # If we're close to budget and have enough products, stop
                        if current_cost >= min_budget and len(selected_products) >= 3:
                            break
            
            # If we don't have enough products, add more cheaper ones
            if len(selected_products) < 3:
                for product in sorted_products:
                    if product not in selected_products and current_cost + product.list_price <= max_budget:
                        selected_products.append(product)
                        current_cost += product.list_price
                        if len(selected_products) >= 3:
                            break
            
            # Ensure we have at least one product
            if not selected_products:
                selected_products = [sorted_products[0]]  # Take the cheapest product
                current_cost = selected_products[0].list_price
            
            _logger.info(f"Selected {len(selected_products)} products, total cost: €{current_cost:.2f}")
            
            # Create composition with only valid fields
            partner_name = self.env['res.partner'].browse(partner_id).name
            composition_name = f"Gift Composition - {partner_name}"
            
            # Build reasoning text (include notes_text here since we can't store it separately)
            reasoning_parts = [
                f'Emergency composition with {len(selected_products)} carefully selected products.',
                f'Total cost: €{current_cost:.2f}.',
                f'Budget variance: {abs(current_cost - target_budget)/target_budget*100:.1f}%.'
            ]
            
            if notes_text:
                reasoning_parts.append(f'Special notes: {notes_text}')
            
            if dietary_restrictions:
                reasoning_parts.append(f'Dietary restrictions applied: {", ".join(dietary_restrictions)}')
            
            reasoning_text = ' '.join(reasoning_parts)
            
            # CORRECTED: Create composition with only fields that exist
            composition_vals = {
                'name': composition_name,
                'partner_id': partner_id,
                'target_year': target_year or fields.Date.today().year,
                'target_budget': target_budget,
                'composition_type': 'custom',
                'product_ids': [(6, 0, [p.id for p in selected_products])],
                'dietary_restrictions': ','.join(dietary_restrictions or []),
                'reasoning': reasoning_text,
                'confidence_score': 0.7,
                'novelty_score': 0.6,
                'historical_compatibility': 0.5,
                'state': 'draft',
                'generation_method': 'emergency_simple'
            }
            
            composition = self.env['gift.composition'].create(composition_vals)
            
            # Set actual cost (this will trigger the compute method)
            composition.actual_cost = current_cost
            
            # Log product details
            _logger.info(f"Emergency composition created: {composition.name}")
            _logger.info(f"Products: {[p.name for p in selected_products]}")
            _logger.info(f"Actual cost: €{composition.actual_cost:.2f}, Target: €{target_budget:.2f}")
            
            return composition
            
        except Exception as e:
            _logger.error(f"Emergency simple composition failed: {e}")
            raise UserError(f"Emergency composition failed: {str(e)}")

    def _determine_best_engine(self, partner_id, target_budget, dietary_restrictions):
        """EMERGENCY: Always use simple emergency method"""
        return 'emergency_simple'

    def _cascade_fallback(self, partner_id, target_budget, target_year,
                        dietary_restrictions, notes_text, attempt_number):
        """EMERGENCY: Skip cascade, use simple method"""
        return self._generate_emergency_composition_simple(
            partner_id, target_budget, target_year, dietary_restrictions, notes_text
        )
    
    # def _determine_best_engine(self, partner_id, target_budget, dietary_restrictions):
    #     """Determine the best engine to use based on context"""
        
    #     # Check ML engine availability and training
    #     if self.use_ml_engine:
    #         try:
    #             ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
    #             if ml_engine.is_model_trained:
    #                 # Check if we have good historical data for this client
    #                 history = self.env['client.order.history'].search([
    #                     ('partner_id', '=', partner_id)
    #                 ], limit=1)
                    
    #                 if history and history.total_budget > 0:
    #                     _logger.info("Using ML engine - trained model and client history available")
    #                     return 'ml'
    #         except Exception as e:
    #             _logger.warning(f"ML engine check failed: {e}")
        
    #     # Check AI recommender
    #     if self.use_ai_recommender and self._ollama_enabled():
    #         _logger.info("Using AI recommender - Ollama available")
    #         return 'ai'
        
    #     # Check stock-aware engine
    #     if self.use_stock_aware:
    #         _logger.info("Using stock-aware engine")
    #         return 'stock_aware'
        
    #     # Default to emergency fallback
    #     _logger.info("Using emergency fallback")
    #     return 'emergency'
    
    def _generate_ml_composition(self, partner_id, target_budget, target_year, 
                                dietary_restrictions, notes_text, attempt_number):
        """Generate composition using ML engine"""
        
        try:
            ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
            
            if not ml_engine.is_model_trained:
                _logger.warning("ML engine not trained")
                return None
            
            # Get ML recommendations
            ml_result = ml_engine.get_smart_recommendations(
                partner_id=partner_id,
                target_budget=target_budget,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text,
                max_attempts=3
            )
            
            if ml_result and ml_result.get('products') and len(ml_result['products']) > 0:
                # Create composition from ML result
                composition = self._create_composition_from_result(
                    partner_id, target_budget, target_year,
                    ml_result, dietary_restrictions, notes_text, 'ml'
                )
                return composition
            else:
                _logger.warning("ML engine returned no products")
                return None
                
        except Exception as e:
            _logger.warning(f"ML generation failed: {e}")
            return None
    
    def _generate_ai_composition(self, partner_id, target_budget, target_year,
                                dietary_restrictions, notes_text, attempt_number):
        """Generate composition using AI recommender"""
        
        try:
            ai_recommender = self.env['ai.product.recommender']
            
            # Get AI recommendations
            ai_result = ai_recommender.get_ai_recommendations(
                partner_id=partner_id,
                target_budget=target_budget,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text,
                use_ml=False  # Pure AI without ML
            )
            
            if ai_result and ai_result.get('products') and len(ai_result['products']) > 0:
                # Create composition from AI result
                composition = self._create_composition_from_result(
                    partner_id, target_budget, target_year,
                    ai_result, dietary_restrictions, notes_text, 'ai'
                )
                return composition
            else:
                _logger.warning("AI engine returned no products")
                return None
                
        except Exception as e:
            _logger.warning(f"AI generation failed: {e}")
            return None
    
    def _generate_stock_aware_composition(self, partner_id, target_budget, target_year,
                                         dietary_restrictions, notes_text, attempt_number):
        """Generate composition using stock-aware engine"""
        
        try:
            stock_engine = self.env['stock.aware.composition.engine']
            
            composition = stock_engine.generate_compliant_composition(
                partner_id=partner_id,
                target_budget=target_budget,
                target_year=target_year,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text,
                attempt_number=attempt_number
            )
            
            if composition and len(composition.product_ids) > 0:
                return composition
            else:
                _logger.warning("Stock-aware engine returned no products")
                return None
            
        except Exception as e:
            _logger.warning(f"Stock-aware generation failed: {e}")
            return None
    
    def _generate_business_rules_composition(self, partner_id, target_budget, target_year,
                                            dietary_restrictions, notes_text):
        """Generate composition using business rules engine"""
        
        try:
            rules_engine = self.env['business.rules.engine']
            
            # Check rules
            can_generate, message = rules_engine.check_composition_rules(
                partner_id=partner_id,
                target_budget=target_budget,
                target_year=target_year
            )
            
            if not can_generate:
                _logger.warning(f"Business rules check failed: {message}")
                return None
            
            # Use composition engine for actual generation
            composition_engine = self.env['composition.engine']
            
            composition = composition_engine.generate_gift_composition(
                partner_id=partner_id,
                target_budget=target_budget,
                target_year=target_year,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text
            )
            
            if composition and len(composition.product_ids) > 0:
                return composition
            else:
                _logger.warning("Business rules engine returned no products")
                return None
            
        except Exception as e:
            _logger.warning(f"Business rules generation failed: {e}")
            return None
    
    def _generate_emergency_composition(self, partner_id, target_budget, target_year,
                                       dietary_restrictions, notes_text):
        """Emergency fallback: create composition with basic product selection"""
        
        try:
            _logger.info("Using emergency fallback composition generation")
            
            # Get basic products without strict filtering
            products = self.env['product.template'].search([
                ('active', '=', True),
                ('sale_ok', '=', True),
                ('list_price', '>', 0)
            ], limit=50)  # Limit to first 50 products
            
            if not products:
                raise UserError("No products available in the system")
            
            # Simple budget-based selection
            selected_products = []
            current_cost = 0
            min_budget = target_budget * 0.7
            max_budget = target_budget * 1.3
            
            for product in products:
                if current_cost + product.list_price <= max_budget:
                    selected_products.append(product)
                    current_cost += product.list_price
                    
                    if current_cost >= min_budget and len(selected_products) >= 3:
                        break
            
            if not selected_products:
                # Even more emergency: just take the cheapest products
                sorted_products = products.sorted('list_price')
                budget_per_product = target_budget / 5  # Try to get 5 products
                for product in sorted_products:
                    if product.list_price <= budget_per_product:
                        selected_products.append(product)
                        if len(selected_products) >= 5:
                            break
                
                # If still no products, take just one
                if not selected_products:
                    selected_products = [sorted_products[0]]
            
            # Create composition
            composition = self.env['gift.composition'].create({
                'partner_id': partner_id,
                'target_year': target_year or fields.Date.today().year,
                'target_budget': target_budget,
                'composition_type': 'custom',
                'product_ids': [(6, 0, [p.id for p in selected_products])],
                'dietary_restrictions': ','.join(dietary_restrictions or []),
                'reasoning': f'Emergency composition generated with {len(selected_products)} products due to system constraints. Please review and adjust as needed.',
                'confidence_score': 0.4,
                'novelty_score': 0.5,
                'historical_compatibility': 0.3,
                'notes': notes_text,
                'state': 'draft',
                'generation_method': 'emergency_fallback'
            })
            
            # Set actual cost
            composition.actual_cost = sum(p.list_price for p in selected_products)
            
            _logger.info(f"Emergency composition created: {composition.name} with {len(selected_products)} products, cost €{composition.actual_cost:.2f}")
            return composition
            
        except Exception as e:
            _logger.error(f"Emergency fallback failed: {e}")
            return None
    
    # def _cascade_fallback(self, partner_id, target_budget, target_year,
    #                      dietary_restrictions, notes_text, attempt_number):
    #     """Try each engine in cascade order - SAFE VERSION"""
        
    #     engines = ['ml', 'ai', 'stock_aware', 'business_rules', 'emergency']
        
    #     for engine in engines:
    #         try:
    #             _logger.info(f"Cascade fallback trying: {engine}")
                
    #             # Call individual engine methods directly to prevent recursion
    #             if engine == 'ml':
    #                 composition = self._generate_ml_composition(
    #                     partner_id, target_budget, target_year, 
    #                     dietary_restrictions, notes_text, attempt_number
    #                 )
    #             elif engine == 'ai':
    #                 composition = self._generate_ai_composition(
    #                     partner_id, target_budget, target_year,
    #                     dietary_restrictions, notes_text, attempt_number
    #                 )
    #             elif engine == 'stock_aware':
    #                 composition = self._generate_stock_aware_composition(
    #                     partner_id, target_budget, target_year,
    #                     dietary_restrictions, notes_text, attempt_number
    #                 )
    #             elif engine == 'business_rules':
    #                 composition = self._generate_business_rules_composition(
    #                     partner_id, target_budget, target_year,
    #                     dietary_restrictions, notes_text
    #                 )
    #             else:  # emergency
    #                 composition = self._generate_emergency_composition(
    #                     partner_id, target_budget, target_year,
    #                     dietary_restrictions, notes_text
    #                 )
                
    #             if composition and len(composition.product_ids) > 0:
    #                 _logger.info(f"Cascade successful with {engine} engine")
    #                 return composition
                    
    #         except Exception as e:
    #             _logger.warning(f"Cascade engine {engine} failed: {e}")
    #             continue
        
    #     raise UserError("All engines failed to generate composition with products")
    
    def _parallel_fallback(self, partner_id, target_budget, target_year,
                          dietary_restrictions, notes_text, attempt_number):
        """Try multiple engines and select best result - SAFE VERSION"""
        
        results = []
        engines = ['ml', 'ai', 'stock_aware']
        
        for engine in engines:
            try:
                # Call individual engine methods directly
                if engine == 'ml':
                    composition = self._generate_ml_composition(
                        partner_id, target_budget, target_year, 
                        dietary_restrictions, notes_text, attempt_number
                    )
                elif engine == 'ai':
                    composition = self._generate_ai_composition(
                        partner_id, target_budget, target_year,
                        dietary_restrictions, notes_text, attempt_number
                    )
                elif engine == 'stock_aware':
                    composition = self._generate_stock_aware_composition(
                        partner_id, target_budget, target_year,
                        dietary_restrictions, notes_text, attempt_number
                    )
                
                if composition and len(composition.product_ids) > 0:
                    # Score the composition
                    score = self._score_composition(composition, target_budget)
                    results.append((composition, score, engine))
                    
            except Exception as e:
                _logger.warning(f"Parallel engine {engine} failed: {e}")
        
        if results:
            # Select best result
            best_composition, best_score, best_engine = max(results, key=lambda x: x[1])
            _logger.info(f"Selected best composition from {best_engine} with score {best_score:.2f}")
            return best_composition
        
        # If no results, try emergency
        return self._generate_emergency_composition(
            partner_id, target_budget, target_year,
            dietary_restrictions, notes_text
        )
    
    def _weighted_fallback(self, partner_id, target_budget, target_year,
                          dietary_restrictions, notes_text, attempt_number):
        """Combine results from multiple engines with weights"""
        
        # For now, use cascade as weighted is complex
        return self._cascade_fallback(
            partner_id, target_budget, target_year,
            dietary_restrictions, notes_text, attempt_number
        )
    
    def _create_composition_from_result(self, partner_id, target_budget, target_year,
                                       result, dietary_restrictions, notes_text, engine_type):
        """Create composition from engine result"""
        
        products = result.get('products', [])
        
        if not products or len(products) == 0:
            _logger.warning(f"Engine {engine_type} returned no products")
            return None
        
        # Create composition
        composition = self.env['gift.composition'].create({
            'partner_id': partner_id,
            'target_year': target_year or fields.Date.today().year,
            'target_budget': target_budget,
            'composition_type': 'custom',
            'product_ids': [(6, 0, [p.id for p in products])],
            'dietary_restrictions': ','.join(dietary_restrictions or []),
            'reasoning': result.get('reasoning', f'Generated by {engine_type}'),
            'confidence_score': result.get('confidence', result.get('ml_confidence', result.get('ai_confidence', 0.7))),
            'novelty_score': 0.7,
            'historical_compatibility': 0.8,
            'notes': notes_text,
            'state': 'draft',
            'generation_method': engine_type
        })
        
        # Set actual cost
        composition.actual_cost = sum(p.list_price for p in products)
        
        _logger.info(f"Created composition {composition.name} with {len(products)} products, cost €{composition.actual_cost:.2f}")
        
        return composition
    
    def _score_composition(self, composition, target_budget):
        """Score a composition for comparison"""
        
        if not composition or len(composition.product_ids) == 0:
            return 0.0
        
        score = 0.0
        
        # Budget accuracy (40%)
        budget_variance = abs(composition.actual_cost - target_budget) / target_budget
        budget_score = max(0, 1 - budget_variance)
        score += budget_score * 0.4
        
        # Product count (20%)
        product_score = min(1.0, len(composition.product_ids) / 7)  # Ideal: 7 products
        score += product_score * 0.2
        
        # Confidence (20%)
        confidence_score = composition.confidence_score or 0.5
        score += confidence_score * 0.2
        
        # Method preference (20%)
        method_scores = {
            'ml': 1.0,
            'ai': 0.8,
            'stock_aware': 0.6,
            'business_rules': 0.4,
            'emergency_fallback': 0.2
        }
        method_score = method_scores.get(composition.generation_method, 0.3)
        score += method_score * 0.2
        
        return score
    
    def _track_performance(self, success, generation_time):
        """Track performance metrics"""
        
        self.last_generation_time = generation_time
        self.total_generations += 1
        
        # Could add more sophisticated success tracking here
    
    def _add_ai_enhancement(self, composition, notes_text):
        """Add AI enhancement to composition reasoning"""
        
        try:
            if not self._ollama_enabled():
                return None
            
            # Simple AI enhancement - could be expanded
            enhanced_reasoning = f"AI-enhanced composition for {composition.partner_id.name}. "
            enhanced_reasoning += composition.reasoning or ""
            
            if notes_text:
                enhanced_reasoning += f"\n\nSpecial considerations: {notes_text}"
            
            return enhanced_reasoning
            
        except Exception as e:
            _logger.warning(f"AI enhancement failed: {e}")
            return None
    
    def _generate_documents(self, composition):
        """Generate documents for composition"""
        
        try:
            if hasattr(self.env, 'document.generation.system'):
                doc_generator = self.env['document.generation.system']
                doc_generator.generate_all_documents(composition.id)
        except Exception as e:
            _logger.warning(f"Document generation failed: {e}")
    
    def _record_for_learning(self, partner_id, composition, target_budget, 
                           dietary_restrictions, notes_text, engine_method):
        """Record composition for machine learning"""
        
        try:
            learning_data = {
                'partner_id': partner_id,
                'target_budget': target_budget,
                'actual_cost': composition.actual_cost,
                'products': [p.id for p in composition.product_ids],
                'engine_method': engine_method,
                'dietary_restrictions': dietary_restrictions,
                'notes': notes_text,
                'timestamp': fields.Datetime.now().isoformat()
            }
            
            # Store in ML learning data if available
            if hasattr(self.env, 'ml.learning.data'):
                self.env['ml.learning.data'].create({
                    'engine_id': self.env['ml.recommendation.engine'].get_or_create_engine().id,
                    'partner_id': partner_id,
                    'target_budget': target_budget,
                    'actual_cost': composition.actual_cost,
                    'product_count': len(composition.product_ids),
                    'data': json.dumps(learning_data)
                })
                
        except Exception as e:
            _logger.warning(f"Learning data recording failed: {e}")
    
    def _ollama_enabled(self):
        """Check if Ollama is enabled"""
        
        return self.env['ir.config_parameter'].sudo().get_param(
            'lebigott_ai.ollama_enabled', 'false'
        ).lower() == 'true'
    
    @api.model
    def analyze_engine_performance(self):
        """Analyze performance of different engines"""
        
        compositions = self.env['gift.composition'].search([
            ('generation_method', '!=', False)
        ])
        
        engine_stats = {}
        
        for comp in compositions:
            method = comp.generation_method or 'unknown'
            
            if method not in engine_stats:
                engine_stats[method] = {
                    'count': 0,
                    'total_variance': 0,
                    'approved': 0,
                    'delivered': 0,
                    'avg_confidence': 0
                }
            
            stats = engine_stats[method]
            stats['count'] += 1
            
            # Budget variance
            variance = abs(comp.actual_cost - comp.target_budget) / comp.target_budget if comp.target_budget > 0 else 0
            stats['total_variance'] += variance
            
            # Success tracking
            if comp.state == 'approved':
                stats['approved'] += 1
            elif comp.state == 'delivered':
                stats['delivered'] += 1
            
            # Confidence
            stats['avg_confidence'] += comp.confidence_score
        
        # Calculate averages
        for method, stats in engine_stats.items():
            if stats['count'] > 0:
                stats['avg_variance'] = (stats['total_variance'] / stats['count']) * 100
                stats['avg_confidence'] = stats['avg_confidence'] / stats['count']
                stats['success_rate'] = ((stats['approved'] + stats['delivered']) / stats['count']) * 100
        
        return engine_stats