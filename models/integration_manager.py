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
    
    @api.model
    def generate_complete_composition(self, partner_id, target_budget, target_year=None, 
                                    dietary_restrictions=None, notes_text=None, use_batch=False,
                                    attempt_number=1, force_engine=None):
        """
        Enhanced composition generation with ML/AI integration and smart fallback
        """
        
        if target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
        
        start_time = datetime.now()
        
        try:
            _logger.info(f"Generating composition (attempt {attempt_number}): partner={partner_id}, budget=€{target_budget}")
            
            # Determine which engine to use
            if force_engine:
                engine_method = force_engine
            else:
                engine_method = self._determine_best_engine(partner_id, target_budget, dietary_restrictions)
            
            # Generate composition based on selected method
            if engine_method == 'ml':
                composition = self._generate_ml_composition(
                    partner_id, target_budget, target_year, 
                    dietary_restrictions, notes_text, attempt_number
                )
            elif engine_method == 'ai':
                composition = self._generate_ai_composition(
                    partner_id, target_budget, target_year,
                    dietary_restrictions, notes_text, attempt_number
                )
            elif engine_method == 'stock_aware':
                composition = self._generate_stock_aware_composition(
                    partner_id, target_budget, target_year,
                    dietary_restrictions, notes_text, attempt_number
                )
            else:  # business_rules or fallback
                composition = self._generate_business_rules_composition(
                    partner_id, target_budget, target_year,
                    dietary_restrictions, notes_text
                )
            
            if composition:
                # Add AI enhancement
                if attempt_number == 1 and self._ollama_enabled():
                    enhanced_reasoning = self._add_ai_enhancement(composition, notes_text)
                    if enhanced_reasoning:
                        composition.reasoning = enhanced_reasoning
                
                # Track performance
                generation_time = (datetime.now() - start_time).total_seconds()
                self._track_performance(True, generation_time)
                
                # Generate documents if not batch
                if not use_batch:
                    self._generate_documents(composition)
                
                # Record for learning
                self._record_for_learning(
                    partner_id, composition, target_budget, 
                    dietary_restrictions, notes_text, engine_method
                )
                
                return composition
            
            else:
                # Try fallback strategy
                if self.fallback_strategy == 'cascade':
                    return self._cascade_fallback(
                        partner_id, target_budget, target_year,
                        dietary_restrictions, notes_text, attempt_number
                    )
                elif self.fallback_strategy == 'parallel':
                    return self._parallel_fallback(
                        partner_id, target_budget, target_year,
                        dietary_restrictions, notes_text, attempt_number
                    )
                else:  # weighted
                    return self._weighted_fallback(
                        partner_id, target_budget, target_year,
                        dietary_restrictions, notes_text, attempt_number
                    )
                
        except Exception as e:
            _logger.error(f"Composition generation failed: {str(e)}")
            self._track_performance(False, 0)
            raise UserError(f"Composition generation failed: {str(e)}")
    
    def _determine_best_engine(self, partner_id, target_budget, dietary_restrictions):
        """Determine the best engine to use based on context"""
        
        # Check ML engine availability and training
        if self.use_ml_engine:
            ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
            if ml_engine.is_model_trained:
                # Check if we have good historical data for this client
                history = self.env['client.order.history'].search([
                    ('partner_id', '=', partner_id)
                ], limit=1)
                
                if history and history.total_budget > 0:
                    _logger.info("Using ML engine - trained model and client history available")
                    return 'ml'
        
        # Check AI recommender
        if self.use_ai_recommender and self._ollama_enabled():
            _logger.info("Using AI recommender - Ollama available")
            return 'ai'
        
        # Check stock-aware engine
        if self.use_stock_aware:
            _logger.info("Using stock-aware engine")
            return 'stock_aware'
        
        # Default to business rules
        _logger.info("Using business rules engine")
        return 'business_rules'
    
    def _generate_ml_composition(self, partner_id, target_budget, target_year, 
                                dietary_restrictions, notes_text, attempt_number):
        """Generate composition using ML engine"""
        
        try:
            ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
            
            # Get ML recommendations
            ml_result = ml_engine.get_smart_recommendations(
                partner_id=partner_id,
                target_budget=target_budget,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text,
                max_attempts=3
            )
            
            if ml_result and ml_result.get('products'):
                # Create composition from ML result
                composition = self._create_composition_from_result(
                    partner_id, target_budget, target_year,
                    ml_result, dietary_restrictions, notes_text, 'ml'
                )
                
                return composition
                
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
            
            if ai_result and ai_result.get('products'):
                # Create composition from AI result
                composition = self._create_composition_from_result(
                    partner_id, target_budget, target_year,
                    ai_result, dietary_restrictions, notes_text, 'ai'
                )
                
                return composition
                
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
            
            return composition
            
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
                raise UserError(message)
            
            # Use composition engine for actual generation
            composition_engine = self.env['composition.engine']
            
            composition = composition_engine.generate_gift_composition(
                partner_id=partner_id,
                target_budget=target_budget,
                target_year=target_year,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text
            )
            
            return composition
            
        except Exception as e:
            _logger.warning(f"Business rules generation failed: {e}")
        
        return None
    
    def _cascade_fallback(self, partner_id, target_budget, target_year,
                         dietary_restrictions, notes_text, attempt_number):
        """Try each engine in cascade order"""
        
        engines = ['ml', 'ai', 'stock_aware', 'business_rules']
        
        for engine in engines:
            try:
                _logger.info(f"Cascade fallback trying: {engine}")
                
                composition = self.generate_complete_composition(
                    partner_id=partner_id,
                    target_budget=target_budget,
                    target_year=target_year,
                    dietary_restrictions=dietary_restrictions,
                    notes_text=notes_text,
                    use_batch=True,
                    attempt_number=attempt_number,
                    force_engine=engine
                )
                
                if composition:
                    return composition
                    
            except Exception as e:
                _logger.warning(f"Cascade engine {engine} failed: {e}")
                continue
        
        raise UserError("All engines failed to generate composition")
    
    def _parallel_fallback(self, partner_id, target_budget, target_year,
                          dietary_restrictions, notes_text, attempt_number):
        """Try multiple engines and select best result"""
        
        results = []
        engines = ['ml', 'ai', 'stock_aware']
        
        for engine in engines:
            try:
                composition = self.generate_complete_composition(
                    partner_id=partner_id,
                    target_budget=target_budget,
                    target_year=target_year,
                    dietary_restrictions=dietary_restrictions,
                    notes_text=notes_text,
                    use_batch=True,
                    attempt_number=attempt_number,
                    force_engine=engine
                )
                
                if composition:
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
        
        raise UserError("No engines produced valid compositions")
    
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
        
        if not products:
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
        
        # Set category breakdown
        categories = {}
        for product in products:
            category = product.lebiggot_category
            categories[category] = categories.get(category, 0) + 1
        
        composition.set_category_breakdown(categories)
        
        return composition
    
    def _score_composition(self, composition, target_budget):
        """Score a composition for quality"""
        
        score = 0.0
        
        # Budget adherence (40%)
        budget_variance = abs(composition.actual_cost - target_budget) / target_budget
        if budget_variance < 0.05:
            score += 40
        elif budget_variance < 0.1:
            score += 30
        elif budget_variance < 0.15:
            score += 20
        else:
            score += 10
        
        # Confidence score (30%)
        score += composition.confidence_score * 30
        
        # Product diversity (20%)
        categories = set(composition.product_ids.mapped('lebiggot_category'))
        diversity_score = min(len(categories) / 5, 1.0)
        score += diversity_score * 20
        
        # Historical compatibility (10%)
        score += composition.historical_compatibility * 10
        
        return score
    
    def _add_ai_enhancement(self, composition, notes_text):
        """Add AI enhancement to composition reasoning"""
        
        try:
            composition_engine = self.env['composition.engine']
            
            if composition_engine._ollama_enabled() and notes_text:
                # Build enhanced prompt
                product_details = []
                for product in composition.product_ids:
                    details = f"{product.name} ({product.lebiggot_category}, €{product.list_price:.2f})"
                    product_details.append(details)
                
                prompt = f"""
                You are a luxury gourmet gift advisor analyzing this composition:
                
                Budget: €{composition.target_budget:.0f}
                Actual Cost: €{composition.actual_cost:.2f}
                Products: {', '.join(product_details)}
                Client Notes: {notes_text}
                
                Provide a sophisticated analysis (100 words max) explaining:
                1. How this selection creates a premium experience
                2. The strategic pairing of flavors and textures
                3. Why this combination represents excellent value
                
                Write in an elegant, professional tone suitable for a premium gift service.
                """
                
                ai_analysis = composition_engine._ollama_complete(prompt)
                
                if ai_analysis:
                    enhanced_reasoning = f"""
                    <div class="ai-enhancement" style="background: #f8f9fa; padding: 15px; border-left: 4px solid #875A7B; margin: 15px 0;">
                        <h5 style="color: #875A7B;">🤖 AI Sommelier Analysis</h5>
                        <p style="margin-bottom: 0;">{ai_analysis}</p>
                    </div>
                    """
                    
                    return enhanced_reasoning + (composition.reasoning or "")
                    
        except Exception as e:
            _logger.warning(f"AI enhancement failed: {str(e)}")
        
        return composition.reasoning
    
    def _generate_documents(self, composition):
        """Generate documents for composition"""
        
        try:
            doc_generator = self.env['document.generation.system']
            documents = doc_generator.generate_all_documents(composition.id)
            _logger.info(f"Generated {len(documents)} document types for composition {composition.id}")
        except Exception as e:
            _logger.warning(f"Document generation failed: {str(e)}")
    
    def _record_for_learning(self, partner_id, composition, target_budget, 
                            dietary_restrictions, notes_text, engine_type):
        """Record composition for ML learning"""
        
        try:
            ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
            
            learning_data = {
                'partner_id': partner_id,
                'composition_id': composition.id,
                'products': composition.product_ids.ids,
                'product_names': composition.product_ids.mapped('name'),
                'categories': list(set(composition.product_ids.mapped('lebiggot_category'))),
                'target_budget': target_budget,
                'actual_cost': composition.actual_cost,
                'dietary_restrictions': dietary_restrictions,
                'notes': notes_text,
                'engine_type': engine_type,
                'confidence_score': composition.confidence_score,
                'timestamp': datetime.now().isoformat()
            }
            
            self.env['ml.learning.data'].create({
                'engine_id': ml_engine.id,
                'partner_id': partner_id,
                'target_budget': target_budget,
                'actual_cost': composition.actual_cost,
                'product_count': len(composition.product_ids),
                'data': json.dumps(learning_data),
                'created_date': fields.Datetime.now()
            })
            
        except Exception as e:
            _logger.warning(f"Failed to record for learning: {e}")
    
    def _track_performance(self, success, generation_time):
        """Track performance metrics"""
        
        self.total_generations += 1
        if generation_time > 0:
            self.last_generation_time = generation_time
    
    def _compute_success_rate(self):
        """Compute success rate from compositions"""
        
        for manager in self:
            total = self.env['gift.composition'].search_count([])
            successful = self.env['gift.composition'].search_count([
                ('state', 'in', ['approved', 'delivered'])
            ])
            
            if total > 0:
                manager.success_rate = (successful / total) * 100
            else:
                manager.success_rate = 0
    
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