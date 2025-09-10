from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import json
import logging
import numpy as np
import requests
import os
from collections import defaultdict
from datetime import datetime, timedelta
import random
import pickle
import base64

_logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.cluster import KMeans
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    _logger.warning("scikit-learn not available. ML features will be limited.")
    SKLEARN_AVAILABLE = False

class MLRecommendationEngine(models.Model):
    _name = 'ml.recommendation.engine'
    _description = 'Advanced ML Recommendation Engine with Training'
    _rec_name = 'name'
    
    # Basic Info
    name = fields.Char(string="Engine Name", default="Señor Bigott ML Engine", required=True)
    model_version = fields.Char(string="Model Version", default="1.0.0")
    
    # Training Status
    is_model_trained = fields.Boolean(string="Model Trained", default=False)
    last_training_date = fields.Datetime(string="Last Training Date")
    training_samples = fields.Integer(string="Training Samples", default=0)
    model_accuracy = fields.Float(string="Model Accuracy (%)", default=0.0)
    training_duration = fields.Float(string="Training Duration (minutes)", default=0.0)
    
    # Model Performance Metrics
    r2_score = fields.Float(string="R² Score", default=0.0)
    mse_score = fields.Float(string="MSE Score", default=0.0)
    cv_score = fields.Float(string="Cross-Validation Score", default=0.0)
    
    # Training Configuration
    min_training_samples = fields.Integer(string="Minimum Training Samples", default=50)
    test_size = fields.Float(string="Test Set Size (%)", default=0.2)
    random_state = fields.Integer(string="Random State", default=42)
    
    # Model Storage
    model_data = fields.Binary(string="Trained Model Data")
    scaler_data = fields.Binary(string="Scaler Data")
    feature_names = fields.Text(string="Feature Names")
    
    # Auto-training Settings
    auto_retrain_enabled = fields.Boolean(string="Auto Retrain Enabled", default=True)
    retrain_frequency_days = fields.Integer(string="Retrain Every (days)", default=7)
    
    # Learning Data Storage
    learning_data_ids = fields.One2many('ml.learning.data', 'engine_id', string="Learning Data")
    
    @api.model
    def get_or_create_engine(self):
        """Get existing engine or create new one"""
        engine = self.search([], limit=1)
        if not engine:
            engine = self.create({'name': 'Señor Bigott ML Engine'})
        return engine
    
    def get_smart_recommendations(self, partner_id, target_budget, dietary_restrictions=None, 
                                notes_text=None, max_attempts=3):
        """
        Advanced ML-powered recommendation with multi-strategy approach
        """
        
        if not SKLEARN_AVAILABLE:
            return self._fallback_recommendations(partner_id, target_budget, dietary_restrictions, notes_text)
        
        _logger.info(f"Starting ML recommendations for partner {partner_id}, budget €{target_budget}")
        
        # Multi-attempt with increasing flexibility
        flexibility_levels = [0.05, 0.10, 0.15]  # 5%, 10%, 15%
        
        for attempt in range(1, max_attempts + 1):
            try:
                flexibility = flexibility_levels[min(attempt - 1, 2)]
                _logger.info(f"ML attempt {attempt}/{max_attempts} with {flexibility*100:.0f}% flexibility")
                
                # Get comprehensive client profile
                client_profile = self._build_comprehensive_client_profile(partner_id, notes_text)
                
                # Get validated product pool
                available_products = self._get_validated_product_pool(dietary_restrictions)
                
                if not available_products:
                    raise UserError("No available products meet the criteria")
                
                # ML-powered product scoring
                if self.is_model_trained:
                    scored_products = self._ml_score_products(client_profile, available_products, target_budget)
                else:
                    scored_products = self._hybrid_score_products(client_profile, available_products, target_budget)
                
                # Advanced selection algorithm
                selected_products = self._advanced_product_selection(
                    scored_products, target_budget, flexibility, attempt, client_profile
                )
                
                if selected_products:
                    final_cost = sum(p.list_price for p in selected_products)
                    variance = abs(final_cost - target_budget) / target_budget * 100
                    
                    if variance <= (flexibility * 100):
                        # Record for continuous learning
                        self._record_recommendation_outcome(partner_id, selected_products, target_budget, client_profile)
                        
                        return {
                            'products': selected_products,
                            'actual_cost': final_cost,
                            'budget_variance': variance,
                            'ml_confidence': self._calculate_ml_confidence(selected_products, client_profile),
                            'reasoning': self._generate_ml_reasoning(selected_products, client_profile, target_budget, attempt),
                            'attempt': attempt,
                            'method': 'ML-Powered' if self.is_model_trained else 'Hybrid-AI',
                            'client_profile': client_profile
                        }
                
            except Exception as e:
                _logger.error(f"ML attempt {attempt} failed: {str(e)}")
                if attempt == max_attempts:
                    raise UserError(f"ML recommendation failed after {max_attempts} attempts: {str(e)}")
        
        raise UserError("Could not generate suitable ML recommendation")
    
    def _build_comprehensive_client_profile(self, partner_id, notes_text=None):
        """Build comprehensive client profile with ML features"""
        
        partner = self.env['res.partner'].browse(partner_id)
        
        # Historical analysis
        base_analysis = self.env['client.order.history'].analyze_client_patterns(partner_id)
        
        # ML feature extraction
        ml_features = self._extract_ml_features(partner_id, base_analysis)
        
        # AI text analysis
        text_analysis = {}
        if notes_text:
            text_analysis = self._ai_analyze_notes(notes_text)
        
        # Behavioral clustering
        cluster_info = self._get_client_cluster(ml_features) if self.is_model_trained else {}
        
        # Market position analysis
        market_position = self._analyze_market_position(partner_id, ml_features)
        
        return {
            'partner_id': partner_id,
            'partner_name': partner.name,
            'base_analysis': base_analysis,
            'ml_features': ml_features,
            'text_analysis': text_analysis,
            'cluster_info': cluster_info,
            'market_position': market_position,
            'profile_confidence': self._calculate_profile_confidence(base_analysis, text_analysis),
            'timestamp': datetime.now()
        }
    
    def _extract_ml_features(self, partner_id, base_analysis):
        """Extract comprehensive ML features"""
        
        if not base_analysis.get('has_history'):
            return self._default_ml_features()
        
        histories = self.env['client.order.history'].search([('partner_id', '=', partner_id)])
        
        # Budget features
        budgets = [h.total_budget for h in histories if h.total_budget > 0]
        budget_features = self._calculate_budget_features(budgets)
        
        # Product features
        product_features = self._calculate_product_features(histories)
        
        # Temporal features
        temporal_features = self._calculate_temporal_features(histories)
        
        # Satisfaction features
        satisfaction_features = self._calculate_satisfaction_features(histories)
        
        # Combine all features
        return {
            **budget_features,
            **product_features,
            **temporal_features,
            **satisfaction_features
        }
    
    def _calculate_budget_features(self, budgets):
        """Calculate budget-related ML features"""
        if not budgets:
            return self._default_budget_features()
        
        budgets = np.array(budgets)
        
        return {
            'avg_budget': float(np.mean(budgets)),
            'budget_std': float(np.std(budgets)),
            'budget_trend': float(np.polyfit(range(len(budgets)), budgets, 1)[0]) if len(budgets) > 1 else 0,
            'budget_volatility': float(np.std(budgets) / np.mean(budgets)) if np.mean(budgets) > 0 else 0,
            'min_budget': float(np.min(budgets)),
            'max_budget': float(np.max(budgets)),
            'budget_growth_rate': float((budgets[-1] - budgets[0]) / budgets[0]) if len(budgets) > 1 and budgets[0] > 0 else 0
        }
    
    def _calculate_product_features(self, histories):
        """Calculate product-related ML features"""
        
        all_products = []
        all_categories = []
        
        for history in histories:
            all_products.extend(history.product_ids)
            categories = history.get_category_structure()
            all_categories.extend(categories.keys())
        
        # Category diversity
        unique_categories = len(set(all_categories))
        category_diversity = min(unique_categories / 8.0, 1.0)
        
        # Premium affinity
        premium_count = sum(1 for p in all_products if hasattr(p, 'product_grade') and p.product_grade in ['premium', 'luxury'])
        premium_affinity = premium_count / len(all_products) if all_products else 0.3
        
        # Brand loyalty
        brands = [p.brand for p in all_products if hasattr(p, 'brand') and p.brand]
        brand_diversity = len(set(brands)) / len(brands) if brands else 0.5
        
        return {
            'category_diversity': category_diversity,
            'premium_affinity': premium_affinity,
            'brand_loyalty': 1 - brand_diversity,
            'avg_products_per_order': len(all_products) / len(histories) if histories else 5,
            'product_consistency': self._calculate_product_consistency(all_products)
        }
    
    def _calculate_temporal_features(self, histories):
        """Calculate time-based ML features"""
        
        if len(histories) < 2:
            return {'order_frequency': 1.0, 'seasonal_variance': 0.0, 'recency_score': 0.5}
        
        # Order frequency
        years = sorted(set(h.order_year for h in histories))
        order_frequency = len(histories) / len(years) if years else 1.0
        
        # Recency score
        latest_year = max(h.order_year for h in histories)
        current_year = datetime.now().year
        recency_score = max(0, 1 - (current_year - latest_year) / 5.0)  # Decay over 5 years
        
        return {
            'order_frequency': order_frequency,
            'seasonal_variance': 0.1,  # Could be enhanced with actual seasonal analysis
            'recency_score': recency_score,
            'client_tenure': len(years)
        }
    
    def _calculate_satisfaction_features(self, histories):
        """Calculate satisfaction-related ML features"""
        
        satisfactions = [float(h.client_satisfaction) for h in histories if h.client_satisfaction]
        
        if not satisfactions:
            return {'avg_satisfaction': 3.5, 'satisfaction_trend': 0.0, 'satisfaction_volatility': 0.0}
        
        satisfactions = np.array(satisfactions)
        
        return {
            'avg_satisfaction': float(np.mean(satisfactions)),
            'satisfaction_trend': float(np.polyfit(range(len(satisfactions)), satisfactions, 1)[0]) if len(satisfactions) > 1 else 0,
            'satisfaction_volatility': float(np.std(satisfactions)),
            'min_satisfaction': float(np.min(satisfactions)),
            'max_satisfaction': float(np.max(satisfactions))
        }
    
    def _ai_analyze_notes(self, notes_text):
        """AI-powered notes analysis using Ollama"""
        
        if not self._ollama_enabled():
            return self._rule_based_notes_analysis(notes_text)
        
        try:
            prompt = f"""
            Analyze these luxury gourmet gift notes and extract client preferences. 
            
            Notes: "{notes_text}"
            
            Return JSON with these exact fields:
            {{
                "style_preference": "traditional|modern|premium|eclectic",
                "price_sensitivity": "low|medium|high",
                "novelty_seeking": true/false,
                "sophistication_level": 1-10,
                "flavor_profile": "bold|mild|varied|premium",
                "occasion_type": "business|personal|celebration|gratitude",
                "dietary_strictness": "strict|moderate|flexible",
                "quality_focus": "artisanal|branded|exclusive|value",
                "presentation_importance": "high|medium|low",
                "cultural_preferences": ["spanish", "french", "italian", "international"]
            }}
            
            Only return valid JSON, nothing else.
            """
            
            response = self._ollama_complete(prompt)
            if response:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        
        except Exception as e:
            _logger.warning(f"AI notes analysis failed: {e}")
        
        return self._rule_based_notes_analysis(notes_text)
    
    def _rule_based_notes_analysis(self, notes_text):
        """Fallback rule-based notes analysis"""
        
        notes_lower = notes_text.lower()
        
        # Style detection
        style = 'traditional'
        if any(word in notes_lower for word in ['modern', 'contemporary', 'innovative', 'trendy']):
            style = 'modern'
        elif any(word in notes_lower for word in ['premium', 'luxury', 'exclusive', 'high-end']):
            style = 'premium'
        elif any(word in notes_lower for word in ['varied', 'diverse', 'eclectic', 'mixed']):
            style = 'eclectic'
        
        # Price sensitivity
        price_sensitivity = 'medium'
        if any(word in notes_lower for word in ['budget', 'affordable', 'cost', 'cheap']):
            price_sensitivity = 'high'
        elif any(word in notes_lower for word in ['premium', 'luxury', 'expensive', 'high-quality']):
            price_sensitivity = 'low'
        
        return {
            'style_preference': style,
            'price_sensitivity': price_sensitivity,
            'novelty_seeking': 'variety' in notes_lower or 'different' in notes_lower or 'new' in notes_lower,
            'sophistication_level': 7,
            'flavor_profile': 'varied',
            'occasion_type': 'business' if 'business' in notes_lower else 'personal',
            'dietary_strictness': 'strict' if any(word in notes_lower for word in ['must', 'cannot', 'strict']) else 'moderate',
            'quality_focus': 'artisanal',
            'presentation_importance': 'high',
            'cultural_preferences': ['spanish']
        }
    
    def _get_validated_product_pool(self, dietary_restrictions=None):
        """Get comprehensive validated product pool"""
        
        domain = [
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('default_code', '!=', False),
            ('list_price', '>', 0),
            ('lebiggot_category', '!=', False),
        ]
        
        products = self.env['product.template'].search(domain)
        
        validated_products = []
        
        for product in products:
            # Stock validation
            if not self._validate_product_stock(product):
                continue
            
            # Quality validation
            if not self._validate_product_quality(product):
                continue
            
            # Dietary validation
            if dietary_restrictions and not self._validate_dietary_compliance(product, dietary_restrictions):
                continue
            
            validated_products.append(product)
        
        _logger.info(f"Validated product pool: {len(validated_products)} products")
        return validated_products
    
    def _validate_product_stock(self, product):
        """Comprehensive stock validation"""
        
        if product.type != 'product':
            return True
        
        stock_quants = self.env['stock.quant'].search([
            ('product_id', 'in', product.product_variant_ids.ids),
            ('location_id.usage', '=', 'internal')
        ])
        
        available_qty = sum(quant.available_quantity for quant in stock_quants)
        return available_qty > 0
    
    def _validate_product_quality(self, product):
        """Validate product meets quality standards"""
        
        # Internal reference required
        if not product.default_code or not product.default_code.strip():
            return False
        
        # Must have category
        if not product.lebiggot_category:
            return False
        
        # Must have positive price
        if product.list_price <= 0:
            return False
        
        return True
    
    def _validate_dietary_compliance(self, product, dietary_restrictions):
        """Validate product meets dietary restrictions"""
        
        for restriction in dietary_restrictions:
            if restriction == 'vegan':
                if hasattr(product, 'is_vegan') and not product.is_vegan:
                    return False
                # Fallback keyword check
                if any(word in product.name.lower() for word in ['meat', 'fish', 'dairy', 'cheese', 'ham']):
                    return False
            
            elif restriction == 'halal':
                if hasattr(product, 'is_halal') and not product.is_halal:
                    return False
                if any(word in product.name.lower() for word in ['pork', 'wine', 'alcohol']):
                    return False
            
            elif restriction == 'non_alcoholic':
                if hasattr(product, 'contains_alcohol') and product.contains_alcohol:
                    return False
                if any(word in product.name.lower() for word in ['wine', 'champagne', 'alcohol', 'liqueur']):
                    return False
        
        return True
    
    def _ml_score_products(self, client_profile, available_products, target_budget):
        """Advanced ML-powered product scoring"""
        
        if not self.is_model_trained:
            return self._hybrid_score_products(client_profile, available_products, target_budget)
        
        try:
            # Load trained model
            model = self._load_trained_model()
            scaler = self._load_scaler()
            
            if not model or not scaler:
                _logger.warning("Could not load ML model, falling back to hybrid scoring")
                return self._hybrid_score_products(client_profile, available_products, target_budget)
            
            scored_products = []
            client_features = self._prepare_client_features_for_ml(client_profile)
            
            for product in available_products:
                product_features = self._prepare_product_features_for_ml(product)
                combined_features = np.concatenate([client_features, product_features])
                
                try:
                    # Scale features
                    scaled_features = scaler.transform([combined_features])[0]
                    
                    # Predict score
                    ml_score = model.predict([scaled_features])[0]
                    
                    # Normalize to 0-1 range
                    normalized_score = max(0, min(1, ml_score))
                    
                    scored_products.append({
                        'product': product,
                        'score': normalized_score,
                        'ml_score': ml_score,
                        'scoring_method': 'ML',
                        'features': {
                            'client_features': client_features,
                            'product_features': product_features
                        }
                    })
                
                except Exception as e:
                    _logger.warning(f"ML scoring failed for product {product.id}: {e}")
                    # Fallback to hybrid scoring for this product
                    fallback_score = self._calculate_hybrid_score(product, client_profile, target_budget)
                    scored_products.append({
                        'product': product,
                        'score': fallback_score,
                        'ml_score': fallback_score,
                        'scoring_method': 'Hybrid-Fallback'
                    })
            
            return sorted(scored_products, key=lambda x: x['score'], reverse=True)
        
        except Exception as e:
            _logger.error(f"ML scoring failed: {e}")
            return self._hybrid_score_products(client_profile, available_products, target_budget)
    
    def _hybrid_score_products(self, client_profile, available_products, target_budget):
        """Hybrid AI/rule-based scoring when ML not available"""
        
        scored_products = []
        base_analysis = client_profile['base_analysis']
        ml_features = client_profile['ml_features']
        text_analysis = client_profile['text_analysis']
        
        for product in available_products:
            score = 0.0
            
            # Historical compatibility (25%)
            score += self._score_historical_compatibility(product, base_analysis) * 0.25
            
            # ML features compatibility (20%)
            score += self._score_ml_features_compatibility(product, ml_features) * 0.20
            
            # Text analysis compatibility (20%)
            score += self._score_text_compatibility(product, text_analysis) * 0.20
            
            # Budget fit (15%)
            score += self._score_budget_fit(product, target_budget) * 0.15
            
            # Market trends (10%)
            score += self._score_market_trends(product) * 0.10
            
            # Quality indicators (10%)
            score += self._score_quality_indicators(product) * 0.10
            
            scored_products.append({
                'product': product,
                'score': min(1.0, max(0.0, score)),
                'scoring_method': 'Hybrid-AI',
                'scoring_breakdown': {
                    'historical': self._score_historical_compatibility(product, base_analysis),
                    'ml_features': self._score_ml_features_compatibility(product, ml_features),
                    'text_analysis': self._score_text_compatibility(product, text_analysis),
                    'budget_fit': self._score_budget_fit(product, target_budget),
                    'market_trends': self._score_market_trends(product),
                    'quality': self._score_quality_indicators(product)
                }
            })
        
        return sorted(scored_products, key=lambda x: x['score'], reverse=True)
    
    def _advanced_product_selection(self, scored_products, target_budget, flexibility, attempt, client_profile):
        """Advanced multi-strategy product selection"""
        
        min_budget = target_budget * (1 - flexibility)
        max_budget = target_budget * (1 + flexibility)
        
        # Multiple selection strategies
        strategies = [
            ('value_optimization', self._value_optimization_selection),
            ('category_balanced', self._category_balanced_selection),
            ('premium_focused', self._premium_focused_selection),
            ('client_optimized', self._client_optimized_selection)
        ]
        
        # Rotate strategy based on attempt
        strategy_name, strategy_func = strategies[(attempt - 1) % len(strategies)]
        
        _logger.info(f"Using selection strategy: {strategy_name}")
        
        try:
            selected_products = strategy_func(scored_products, min_budget, max_budget, client_profile)
            
            if selected_products:
                total_cost = sum(p.list_price for p in selected_products)
                if min_budget <= total_cost <= max_budget:
                    return selected_products
        
        except Exception as e:
            _logger.error(f"Selection strategy {strategy_name} failed: {e}")
        
        # Fallback to simple greedy selection
        return self._greedy_selection_fallback(scored_products, min_budget, max_budget)
    
    def _value_optimization_selection(self, scored_products, min_budget, max_budget, client_profile):
        """Value-optimized selection (score per euro)"""
        
        # Calculate value scores
        value_products = []
        for item in scored_products:
            if item['product'].list_price > 0:
                value_score = item['score'] / item['product'].list_price
                value_products.append({
                    'product': item['product'],
                    'score': item['score'],
                    'value_score': value_score,
                    'original_item': item
                })
        
        value_products.sort(key=lambda x: x['value_score'], reverse=True)
        
        return self._execute_selection_algorithm(value_products, min_budget, max_budget, 'value_score')
    
    def _category_balanced_selection(self, scored_products, min_budget, max_budget, client_profile):
        """Category-balanced selection ensuring diversity"""
        
        # Group by category
        by_category = defaultdict(list)
        for item in scored_products:
            category = item['product'].lebiggot_category
            by_category[category].append(item)
        
        # Sort each category by score
        for category in by_category:
            by_category[category].sort(key=lambda x: x['score'], reverse=True)
        
        selected_products = []
        current_cost = 0
        category_budget = max_budget / len(by_category)
        
        # First pass: one from each category
        for category, items in by_category.items():
            for item in items:
                product = item['product']
                if current_cost + product.list_price <= max_budget:
                    selected_products.append(product)
                    current_cost += product.list_price
                    break
        
        # Second pass: fill remaining budget with best scores
        remaining_items = [item for cat_items in by_category.values() 
                          for item in cat_items[1:]]  # Skip first (already selected)
        remaining_items.sort(key=lambda x: x['score'], reverse=True)
        
        for item in remaining_items:
            product = item['product']
            if current_cost + product.list_price <= max_budget and len(selected_products) < 8:
                selected_products.append(product)
                current_cost += product.list_price
        
        return selected_products if current_cost >= min_budget else []
    
    def _premium_focused_selection(self, scored_products, min_budget, max_budget, client_profile):
        """Premium-focused selection for high-end clients"""
        
        # Prefer premium products
        premium_products = []
        standard_products = []
        
        for item in scored_products:
            product = item['product']
            if hasattr(product, 'product_grade') and product.product_grade in ['premium', 'luxury']:
                premium_products.append(item)
            else:
                standard_products.append(item)
        
        # Sort both groups by score
        premium_products.sort(key=lambda x: x['score'], reverse=True)
        standard_products.sort(key=lambda x: x['score'], reverse=True)
        
        # Combine with premium bias
        prioritized_items = premium_products + standard_products
        
        return self._execute_selection_algorithm(prioritized_items, min_budget, max_budget, 'score')
    
    def _client_optimized_selection(self, scored_products, min_budget, max_budget, client_profile):
        """Client-specific optimized selection"""
        
        ml_features = client_profile['ml_features']
        text_analysis = client_profile['text_analysis']
        
        # Adjust scores based on client profile
        adjusted_items = []
        
        for item in scored_products:
            product = item['product']
            adjusted_score = item['score']
            
            # Premium affinity adjustment
            if ml_features.get('premium_affinity', 0) > 0.7:
                if hasattr(product, 'product_grade') and product.product_grade in ['premium', 'luxury']:
                    adjusted_score += 0.2
            
            # Style preference adjustment
            style_pref = text_analysis.get('style_preference', 'traditional')
            if style_pref == 'premium' and hasattr(product, 'product_grade') and product.product_grade in ['premium', 'luxury']:
                adjusted_score += 0.15
            
            # Novelty seeking adjustment
            if text_analysis.get('novelty_seeking', False):
                # Boost less common categories
                uncommon_categories = ['experience_gastronomica', 'aperitif']
                if product.lebiggot_category in uncommon_categories:
                    adjusted_score += 0.1
            
            adjusted_items.append({
                'product': product,
                'score': min(1.0, adjusted_score),
                'original_score': item['score']
            })
        
        adjusted_items.sort(key=lambda x: x['score'], reverse=True)
        
        return self._execute_selection_algorithm(adjusted_items, min_budget, max_budget, 'score')
    
    def _execute_selection_algorithm(self, prioritized_items, min_budget, max_budget, score_key):
        """Execute selection algorithm with budget constraints"""
        
        selected_products = []
        current_cost = 0
        categories_used = defaultdict(int)
        
        for item in prioritized_items:
            product = item['product']
            category = product.lebiggot_category
            
            # Category diversity constraint (max 2 per category)
            if categories_used[category] >= 2:
                continue
            
            # Budget constraint
            if current_cost + product.list_price <= max_budget:
                selected_products.append(product)
                current_cost += product.list_price
                categories_used[category] += 1
                
                # Stop if we have enough products
                if len(selected_products) >= 7:
                    break
        
        # Check minimum budget
        return selected_products if current_cost >= min_budget else []
    
    def _generate_ml_reasoning(self, selected_products, client_profile, target_budget, attempt):
        """Generate comprehensive ML reasoning"""
        
        actual_cost = sum(p.list_price for p in selected_products)
        variance = abs(actual_cost - target_budget) / target_budget * 100
        
        reasoning_parts = [
            f"<div class='ml-reasoning'>",
            f"<h3>🧠 Advanced ML Recommendation Analysis</h3>",
            
            f"<div class='row mb-3'>",
            f"<div class='col-md-6'>",
            f"<div class='card'>",
            f"<div class='card-body'>",
            f"<h5 class='card-title'>Recommendation Summary</h5>",
            f"<p><strong>Products Selected:</strong> {len(selected_products)}</p>",
            f"<p><strong>Total Cost:</strong> €{actual_cost:.2f}</p>",
            f"<p><strong>Budget Variance:</strong> ±{variance:.1f}%</p>",
            f"<p><strong>ML Confidence:</strong> {self._calculate_ml_confidence(selected_products, client_profile):.1%}</p>",
            f"<p><strong>Attempt:</strong> {attempt}/3</p>",
            f"</div></div></div>",
            
            f"<div class='col-md-6'>",
            f"<div class='card'>",
            f"<div class='card-body'>",
            f"<h5 class='card-title'>Client Profile</h5>",
            f"<p><strong>Profile Confidence:</strong> {client_profile['profile_confidence']:.1%}</p>",
            f"<p><strong>Historical Data:</strong> {'Yes' if client_profile['base_analysis'].get('has_history') else 'No'}</p>",
            f"<p><strong>ML Features:</strong> {len(client_profile['ml_features'])} extracted</p>",
            f"<p><strong>AI Text Analysis:</strong> {'Yes' if client_profile['text_analysis'] else 'No'}</p>",
            f"</div></div></div>",
            f"</div>"
        ]
        
        # ML Analysis Details
        if self.is_model_trained:
            reasoning_parts.extend([
                f"<h4>🎯 ML Model Analysis</h4>",
                f"<div class='alert alert-info'>",
                f"<p><strong>Model Version:</strong> {self.model_version}</p>",
                f"<p><strong>Training Accuracy:</strong> {self.model_accuracy:.1f}%</p>",
                f"<p><strong>Training Samples:</strong> {self.training_samples:,}</p>",
                f"<p><strong>Last Trained:</strong> {self.last_training_date.strftime('%Y-%m-%d') if self.last_training_date else 'Never'}</p>",
                f"</div>"
            ])
        
        # Client Intelligence
        base_analysis = client_profile['base_analysis']
        if base_analysis.get('has_history'):
            ml_features = client_profile['ml_features']
            reasoning_parts.extend([
                f"<h4>📊 Client Intelligence</h4>",
                f"<ul>",
                f"<li><strong>Experience:</strong> {base_analysis.get('years_of_data', 0)} years of data</li>",
                f"<li><strong>Budget Pattern:</strong> {base_analysis.get('budget_trend', 'Unknown').title()}</li>",
                f"<li><strong>Category Diversity:</strong> {ml_features.get('category_diversity', 0):.1%}</li>",
                f"<li><strong>Premium Affinity:</strong> {ml_features.get('premium_affinity', 0):.1%}</li>",
                f"<li><strong>Satisfaction Trend:</strong> {ml_features.get('avg_satisfaction', 0):.1f}/5 stars</li>",
                f"</ul>"
            ])
        
        # AI Text Analysis
        text_analysis = client_profile['text_analysis']
        if text_analysis:
            reasoning_parts.extend([
                f"<h4>🤖 AI Text Analysis</h4>",
                f"<ul>",
                f"<li><strong>Style Preference:</strong> {text_analysis.get('style_preference', 'Unknown').title()}</li>",
                f"<li><strong>Sophistication Level:</strong> {text_analysis.get('sophistication_level', 0)}/10</li>",
                f"<li><strong>Price Sensitivity:</strong> {text_analysis.get('price_sensitivity', 'Unknown').title()}</li>",
                f"<li><strong>Novelty Seeking:</strong> {'Yes' if text_analysis.get('novelty_seeking') else 'No'}</li>",
                f"<li><strong>Quality Focus:</strong> {text_analysis.get('quality_focus', 'Unknown').title()}</li>",
                f"</ul>"
            ])
        
        # Product Breakdown
        categories = defaultdict(list)
        for product in selected_products:
            categories[product.lebiggot_category].append(product.name)
        
        reasoning_parts.extend([
            f"<h4>🎁 Product Selection by Category</h4>",
            f"<div class='row'>"
        ])
        
        for category, products in categories.items():
            category_name = category.replace('_', ' ').title()
            reasoning_parts.extend([
                f"<div class='col-md-6'>",
                f"<p><strong>{category_name}:</strong></p>",
                f"<ul>",
                f"{''.join([f'<li>{product}</li>' for product in products])}",
                f"</ul>",
                f"</div>"
            ])
        
        reasoning_parts.extend([
            f"</div>",
            f"</div>"
        ])
        
        return ''.join(reasoning_parts)
    
    def _calculate_ml_confidence(self, selected_products, client_profile):
        """Calculate ML confidence score"""
        
        base_confidence = 0.6
        
        # Model training bonus
        if self.is_model_trained:
            base_confidence += 0.2
            # Model accuracy bonus
            if self.model_accuracy > 80:
                base_confidence += 0.1
        
        # Client profile bonus
        if client_profile['base_analysis'].get('has_history'):
            base_confidence += 0.15
        
        # Text analysis bonus
        if client_profile['text_analysis']:
            base_confidence += 0.1
        
        # Profile confidence bonus
        profile_conf = client_profile.get('profile_confidence', 0.5)
        base_confidence += (profile_conf - 0.5) * 0.2
        
        return min(1.0, base_confidence)
    
    def _record_recommendation_outcome(self, partner_id, selected_products, target_budget, client_profile):
        """Record recommendation for continuous learning"""
        
        learning_data = {
            'partner_id': partner_id,
            'products': [p.id for p in selected_products],
            'product_names': [p.name for p in selected_products],
            'categories': [p.lebiggot_category for p in selected_products],
            'target_budget': target_budget,
            'actual_cost': sum(p.list_price for p in selected_products),
            'client_profile': client_profile,
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_version,
            'is_model_trained': self.is_model_trained
        }
        
        # Store in database
        self.env['ml.learning.data'].create({
            'engine_id': self.id,
            'partner_id': partner_id,
            'target_budget': target_budget,
            'actual_cost': sum(p.list_price for p in selected_products),
            'product_count': len(selected_products),
            'data': json.dumps(learning_data),
            'created_date': fields.Datetime.now()
        })
    
    def train_models_from_sales_data(self, force_retrain=False):
        """Complete ML model training pipeline"""
        
        if not SKLEARN_AVAILABLE:
            raise UserError("scikit-learn is required for ML training. Please install: pip install scikit-learn")
        
        start_time = datetime.now()
        _logger.info("Starting comprehensive ML model training...")
        
        try:
            # Extract training data
            training_data = self._extract_comprehensive_training_data()
            
            if len(training_data) < self.min_training_samples:
                raise UserError(f"Insufficient training data: {len(training_data)} samples (minimum: {self.min_training_samples})")
            
            # Prepare training dataset
            X, y = self._prepare_training_dataset(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models and select best
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=15, 
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
            }
            
            best_model = None
            best_score = -float('inf')
            best_model_name = None
            
            for name, model in models.items():
                _logger.info(f"Training {name}...")
                model.fit(X_train_scaled, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                avg_cv_score = np.mean(cv_scores)
                
                _logger.info(f"{name} CV score: {avg_cv_score:.3f}")
                
                if avg_cv_score > best_score:
                    best_score = avg_cv_score
                    best_model = model
                    best_model_name = name
            
            # Test best model
            y_pred = best_model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Save model and scaler
            self._save_trained_model(best_model)
            self._save_scaler(scaler)
            
            # Update training status
            training_duration = (datetime.now() - start_time).total_seconds() / 60
            
            self.write({
                'is_model_trained': True,
                'last_training_date': fields.Datetime.now(),
                'training_samples': len(training_data),
                'model_accuracy': r2 * 100,
                'r2_score': r2,
                'mse_score': mse,
                'cv_score': best_score,
                'training_duration': training_duration,
                'model_version': f"{self.model_version[:-1]}{int(self.model_version[-1]) + 1}"
            })
            
            _logger.info(f"ML training completed successfully:")
            _logger.info(f"- Best model: {best_model_name}")
            _logger.info(f"- Training samples: {len(training_data)}")
            _logger.info(f"- R² score: {r2:.3f}")
            _logger.info(f"- MSE: {mse:.3f}")
            _logger.info(f"- Duration: {training_duration:.1f} minutes")
            
            return True
            
        except Exception as e:
            _logger.error(f"ML training failed: {str(e)}")
            raise UserError(f"ML training failed: {str(e)}")
    
    # Utility Methods
    def _ollama_enabled(self):
        return self.env['ir.config_parameter'].sudo().get_param('lebigott_ai.ollama_enabled', 'false').lower() == 'true'
    
    def _ollama_complete(self, prompt):
        try:
            composition_engine = self.env['composition.engine']
            return composition_engine._ollama_complete(prompt)
        except:
            return None
    
    def _load_trained_model(self):
        if not self.model_data:
            return None
        try:
            model_bytes = base64.b64decode(self.model_data)
            return pickle.loads(model_bytes)
        except:
            return None
    
    def _save_trained_model(self, model):
        try:
            model_bytes = pickle.dumps(model)
            self.model_data = base64.b64encode(model_bytes)
        except Exception as e:
            _logger.error(f"Failed to save model: {e}")
    
    def _load_scaler(self):
        if not self.scaler_data:
            return None
        try:
            scaler_bytes = base64.b64decode(self.scaler_data)
            return pickle.loads(scaler_bytes)
        except:
            return None
    
    def _save_scaler(self, scaler):
        try:
            scaler_bytes = pickle.dumps(scaler)
            self.scaler_data = base64.b64encode(scaler_bytes)
        except Exception as e:
            _logger.error(f"Failed to save scaler: {e}")
    
    # Default feature methods for new clients
    def _default_ml_features(self):
        return {
            'avg_budget': 150.0,
            'budget_std': 50.0,
            'budget_trend': 0.0,
            'budget_volatility': 0.3,
            'category_diversity': 0.5,
            'premium_affinity': 0.3,
            'brand_loyalty': 0.5,
            'avg_products_per_order': 5,
            'order_frequency': 1.0,
            'recency_score': 0.5,
            'avg_satisfaction': 3.5,
            'satisfaction_trend': 0.0,
            'client_tenure': 0
        }
    
    def _default_budget_features(self):
        return {
            'avg_budget': 150.0,
            'budget_std': 50.0,
            'budget_trend': 0.0,
            'budget_volatility': 0.3,
            'min_budget': 100.0,
            'max_budget': 200.0,
            'budget_growth_rate': 0.0
        }

    def train_advanced_model(self, training_data, context):
        """
        Advanced model training with context from training wizard
        """
        
        if not SKLEARN_AVAILABLE:
            raise UserError("scikit-learn is required for ML training")
        
        start_time = datetime.now()
        _logger.info(f"Starting advanced ML training with {len(training_data)} samples")
        
        try:
            # Prepare features and targets
            X, y, feature_names = self._prepare_advanced_features(training_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=context['parameters']['test_size'],
                random_state=self.random_state
            )
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Feature selection if enabled
            if context['parameters']['feature_selection']:
                X_train_scaled, X_test_scaled, feature_names = self._select_best_features(
                    X_train_scaled, X_test_scaled, y_train, feature_names
                )
            
            # Model selection based on context
            best_model = self._train_model_by_type(
                context['parameters']['model_type'],
                X_train_scaled, y_train,
                context['parameters']
            )
            
            # Hyperparameter tuning if enabled
            if context['parameters']['hyperparameter_tuning']:
                best_model = self._tune_hyperparameters(
                    best_model, X_train_scaled, y_train,
                    context['parameters']['cv_folds']
                )
            
            # Evaluate model
            y_pred = best_model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                best_model, X_train_scaled, y_train, 
                cv=context['parameters']['cv_folds'], 
                scoring='r2'
            )
            avg_cv_score = np.mean(cv_scores)
            
            # Save model and metadata
            self._save_trained_model(best_model)
            self._save_scaler(scaler)
            self.feature_names = json.dumps(feature_names)
            
            # Update training status
            training_duration = (datetime.now() - start_time).total_seconds() / 60
            
            self.write({
                'is_model_trained': True,
                'last_training_date': fields.Datetime.now(),
                'training_samples': len(training_data),
                'model_accuracy': r2 * 100,
                'r2_score': r2,
                'mse_score': mse,
                'cv_score': avg_cv_score,
                'training_duration': training_duration,
                'model_version': self._increment_version()
            })
            
            # Mark learning data as used for training
            if context['data_sources']['learning']:
                self.learning_data_ids.write({'used_for_training': True})
            
            result = {
                'samples': len(training_data),
                'accuracy': r2 * 100,
                'duration': training_duration,
                'model_type': context['parameters']['model_type'],
                'r2_score': r2,
                'mse_score': mse,
                'cv_score': avg_cv_score
            }
            
            _logger.info(f"Advanced training completed: {result}")
            
            return result
            
        except Exception as e:
            _logger.error(f"Advanced training failed: {str(e)}")
            raise UserError(f"Advanced training failed: {str(e)}")

    def _prepare_advanced_features(self, training_data):
        """
        Prepare advanced features from diverse training data sources
        """
        
        feature_vectors = []
        targets = []
        
        # Track all feature names
        all_feature_names = set()
        
        for sample in training_data:
            # Extract base features
            features = {}
            
            # Common features across all types
            features['partner_id'] = sample.get('partner_id', 0)
            features['budget'] = sample.get('budget', sample.get('total_amount', sample.get('target_budget', 0)))
            features['product_count'] = len(sample.get('products', []))
            features['category_count'] = len(set(sample.get('categories', [])))
            
            # Type-specific features
            if sample['type'] == 'sales_order':
                features.update(self._extract_sales_features(sample))
            elif sample['type'] == 'composition':
                features.update(self._extract_composition_features_advanced(sample))
            elif sample['type'] == 'history':
                features.update(self._extract_history_features_advanced(sample))
            elif sample['type'] == 'learning':
                features.update(sample.get('features', {}))
            
            # Add custom features if present
            if 'features' in sample:
                features.update(sample['features'])
            
            all_feature_names.update(features.keys())
            feature_vectors.append(features)
            targets.append(sample.get('target', 0.5))
        
        # Convert to consistent numpy arrays
        feature_names = sorted(all_feature_names)
        X = np.zeros((len(feature_vectors), len(feature_names)))
        
        for i, features in enumerate(feature_vectors):
            for j, fname in enumerate(feature_names):
                X[i, j] = features.get(fname, 0)
        
        y = np.array(targets)
        
        return X, y, feature_names

    def _extract_sales_features(self, sample):
        """Extract features from sales order sample"""
        
        features = {}
        
        # Date features
        if 'date' in sample and sample['date']:
            date = sample['date']
            if hasattr(date, 'month'):
                features['order_month'] = date.month
                features['order_quarter'] = ((date.month - 1) // 3) + 1
                features['order_year'] = date.year
        
        # Product features
        if 'products' in sample:
            features['unique_products'] = len(set(sample['products']))
        
        # Category distribution
        if 'categories' in sample:
            for i, cat in enumerate(sample['categories'][:5]):  # Top 5 categories
                features[f'has_category_{cat}'] = 1
        
        return features

    def _extract_composition_features_advanced(self, sample):
        """Extract advanced features from composition sample"""
        
        features = {}
        
        # Budget adherence
        if 'budget' in sample and 'actual_cost' in sample:
            budget = sample['budget']
            actual = sample['actual_cost']
            if budget > 0:
                features['budget_variance'] = abs(actual - budget) / budget
                features['over_budget'] = 1 if actual > budget else 0
        
        # Success indicators
        features['is_approved'] = 1 if sample.get('state') == 'approved' else 0
        features['is_delivered'] = 1 if sample.get('state') == 'delivered' else 0
        
        return features

    def _extract_history_features_advanced(self, sample):
        """Extract advanced features from history sample"""
        
        features = {}
        
        # Temporal features
        if 'year' in sample:
            features['history_year'] = sample['year']
            features['years_ago'] = datetime.now().year - sample['year']
        
        # Satisfaction features
        if 'satisfaction' in sample:
            features['satisfaction'] = sample['satisfaction']
            features['high_satisfaction'] = 1 if sample['satisfaction'] >= 4 else 0
        
        return features

    def _select_best_features(self, X_train, X_test, y_train, feature_names, k=50):
        """
        Select best features using statistical methods
        """
        
        from sklearn.feature_selection import SelectKBest, f_regression
        
        # Select top k features
        selector = SelectKBest(f_regression, k=min(k, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        _logger.info(f"Selected {len(selected_features)} features from {len(feature_names)}")
        
        return X_train_selected, X_test_selected, selected_features

    def _train_model_by_type(self, model_type, X_train, y_train, parameters):
        """
        Train model based on specified type
        """
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            )
        
        elif model_type == 'ensemble':
            # Create ensemble of models
            from sklearn.ensemble import VotingRegressor
            
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            gb = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
            
            model = VotingRegressor([('rf', rf), ('gb', gb)])
        
        else:  # auto
            # Train multiple models and select best
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
            }
            
            best_score = -float('inf')
            best_model = None
            
            for name, m in models.items():
                m.fit(X_train, y_train)
                score = m.score(X_train, y_train)
                
                if score > best_score:
                    best_score = score
                    best_model = m
            
            model = best_model
        
        # Train the model
        model.fit(X_train, y_train)
        
        return model

    def _tune_hyperparameters(self, model, X_train, y_train, cv_folds):
        """
        Tune hyperparameters using grid search
        """
        
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grids based on model type
        if isinstance(model, RandomForestRegressor):
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif isinstance(model, GradientBoostingRegressor):
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [5, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.7, 0.8, 0.9]
            }
        
        else:
            # No tuning for other models
            return model
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds,
            scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        _logger.info(f"Best parameters: {grid_search.best_params_}")
        _logger.info(f"Best score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_

    def evaluate_model_performance(self):
        """
        Comprehensive model evaluation
        """
        
        if not self.is_model_trained:
            raise UserError("No trained model to evaluate")
        
        # Load model and scaler
        model = self._load_trained_model()
        scaler = self._load_scaler()
        
        if not model or not scaler:
            raise UserError("Could not load model or scaler")
        
        # Get recent test data
        test_data = self._get_evaluation_test_data()
        
        if len(test_data) < 10:
            raise UserError("Insufficient test data for evaluation")
        
        # Prepare features
        X_test, y_test, _ = self._prepare_advanced_features(test_data)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, explained_variance_score
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        
        # Binary classification metrics (threshold at 0.5)
        y_test_binary = (y_test >= 0.5).astype(int)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        
        # Feature importance (for tree-based models)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = json.loads(self.feature_names) if self.feature_names else []
            if feature_names:
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(feature_names):
                        feature_importance[feature_names[i]] = float(importance)
        
        evaluation_result = {
            'accuracy': r2 * 100,
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'explained_variance': evs,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance,
            'test_samples': len(test_data)
        }
        
        _logger.info(f"Model evaluation complete: {evaluation_result}")
        
        return evaluation_result

    def _get_evaluation_test_data(self):
        """
        Get recent data for model evaluation
        """
        
        test_data = []
        
        # Get recent compositions
        recent_compositions = self.env['gift.composition'].search([
            ('create_date', '>=', fields.Date.today() - timedelta(days=30)),
            ('state', 'in', ['draft', 'approved', 'delivered'])
        ])
        
        for comp in recent_compositions:
            sample = {
                'type': 'composition',
                'partner_id': comp.partner_id.id,
                'budget': comp.target_budget,
                'actual_cost': comp.actual_cost,
                'products': comp.product_ids.ids,
                'categories': list(set(comp.product_ids.mapped('lebiggot_category'))),
                'state': comp.state,
                'target': 1.0 if comp.state in ['approved', 'delivered'] else 0.5
            }
            test_data.append(sample)
        
        # Get recent sales
        recent_orders = self.env['sale.order'].search([
            ('date_order', '>=', fields.Date.today() - timedelta(days=30)),
            ('state', 'in', ['sale', 'done'])
        ], limit=20)
        
        for order in recent_orders:
            sample = {
                'type': 'sales_order',
                'partner_id': order.partner_id.id,
                'date': order.date_order,
                'total_amount': order.amount_total,
                'products': order.order_line.mapped('product_id.product_tmpl_id.id'),
                'categories': list(set(order.order_line.mapped('product_id.product_tmpl_id.lebiggot_category'))),
                'target': 1.0
            }
            test_data.append(sample)
        
        return test_data

    def _increment_version(self):
        """Increment model version number"""
        
        current = self.model_version or "1.0.0"
        parts = current.split('.')
        
        # Increment patch version
        parts[2] = str(int(parts[2]) + 1)
        
        return '.'.join(parts)

    def _extract_comprehensive_training_data(self):
        """
        Extract comprehensive training data from all available sources
        """
        
        training_data = []
        
        # Sales Orders
        orders = self.env['sale.order'].search([
            ('state', 'in', ['sale', 'done']),
            ('amount_total', '>', 0)
        ])
        
        for order in orders:
            sample = {
                'type': 'sales',
                'partner_id': order.partner_id.id,
                'products': order.order_line.mapped('product_id.product_tmpl_id.id'),
                'categories': list(set(order.order_line.mapped('product_id.product_tmpl_id.lebiggot_category'))),
                'budget': order.amount_total,
                'date': order.date_order,
                'target': 1.0  # Successful sale
            }
            training_data.append(sample)
        
        # Gift Compositions
        compositions = self.env['gift.composition'].search([
            ('state', 'in', ['approved', 'delivered'])
        ])
        
        for comp in compositions:
            sample = {
                'type': 'composition',
                'partner_id': comp.partner_id.id,
                'products': comp.product_ids.ids,
                'categories': list(set(comp.product_ids.mapped('lebiggot_category'))),
                'budget': comp.target_budget,
                'actual_cost': comp.actual_cost,
                'confidence': comp.confidence_score,
                'target': 0.9 if comp.state == 'delivered' else 0.7
            }
            training_data.append(sample)
        
        # Client History
        histories = self.env['client.order.history'].search([])
        
        for history in histories:
            if history.product_ids:
                sample = {
                    'type': 'history',
                    'partner_id': history.partner_id.id,
                    'products': history.product_ids.ids,
                    'categories': list(set(history.product_ids.mapped('lebiggot_category'))),
                    'budget': history.total_budget,
                    'satisfaction': float(history.client_satisfaction) if history.client_satisfaction else 3.5,
                    'target': float(history.client_satisfaction) / 5.0 if history.client_satisfaction else 0.7
                }
                training_data.append(sample)
        
        _logger.info(f"Extracted {len(training_data)} training samples")
        
        return training_data

    def _prepare_training_dataset(self, training_data):
        """
        Prepare training dataset with feature engineering
        """
        
        X = []
        y = []
        
        for sample in training_data:
            features = []
            
            # Budget features
            budget = sample.get('budget', 0)
            features.append(budget)
            features.append(np.log1p(budget))  # Log-transformed budget
            
            # Product count
            product_count = len(sample.get('products', []))
            features.append(product_count)
            
            # Category diversity
            categories = sample.get('categories', [])
            features.append(len(set(categories)))
            
            # Budget variance (for compositions)
            if 'actual_cost' in sample and budget > 0:
                variance = abs(sample['actual_cost'] - budget) / budget
                features.append(variance)
            else:
                features.append(0)
            
            # Confidence score (if available)
            features.append(sample.get('confidence', 0.5))
            
            # Satisfaction (if available)
            features.append(sample.get('satisfaction', 3.5))
            
            # Partner features (simplified)
            features.append(sample.get('partner_id', 0) % 100)  # Partner ID hash
            
            # Date features (if available)
            if 'date' in sample and sample['date']:
                date = sample['date']
                if hasattr(date, 'month'):
                    features.append(date.month)
                    features.append(date.year % 100)
                else:
                    features.append(6)  # Default month
                    features.append(24)  # Default year
            else:
                features.append(6)
                features.append(24)
            
            X.append(features)
            y.append(sample.get('target', 0.5))
        
        return np.array(X), np.array(y)

    def _greedy_selection_fallback(self, scored_products, min_budget, max_budget):
        """
        Simple greedy selection as final fallback
        """
        
        selected = []
        current_cost = 0
        
        for item in scored_products[:20]:  # Consider top 20
            product = item['product']
            
            if current_cost + product.list_price <= max_budget:
                selected.append(product)
                current_cost += product.list_price
                
                if current_cost >= min_budget and len(selected) >= 3:
                    break
        
        return selected if current_cost >= min_budget else []

    def _calculate_product_consistency(self, products):
        """
        Calculate product selection consistency
        """
        
        if not products:
            return 0.5
        
        # Count repeat products
        product_ids = [p.id for p in products]
        unique_products = len(set(product_ids))
        total_products = len(product_ids)
        
        if total_products == 0:
            return 0.5
        
        # Higher consistency = more repeat selections
        consistency = 1 - (unique_products / total_products)
        
        return min(1.0, max(0.0, consistency))

    def _prepare_client_features_for_ml(self, client_profile):
        """
        Prepare client features for ML model input
        """
        
        ml_features = client_profile.get('ml_features', {})
        
        features = [
            ml_features.get('avg_budget', 150),
            ml_features.get('budget_std', 50),
            ml_features.get('budget_trend', 0),
            ml_features.get('budget_volatility', 0.3),
            ml_features.get('category_diversity', 0.5),
            ml_features.get('premium_affinity', 0.3),
            ml_features.get('brand_loyalty', 0.5),
            ml_features.get('avg_products_per_order', 5),
            ml_features.get('order_frequency', 1),
            ml_features.get('recency_score', 0.5),
            ml_features.get('avg_satisfaction', 3.5),
            ml_features.get('satisfaction_trend', 0),
            ml_features.get('client_tenure', 0)
        ]
        
        return np.array(features)

    def _prepare_product_features_for_ml(self, product):
        """
        Prepare product features for ML model input
        """
        
        features = [
            float(product.list_price),
            1 if hasattr(product, 'product_grade') and product.product_grade == 'premium' else 0,
            1 if hasattr(product, 'product_grade') and product.product_grade == 'luxury' else 0,
            len(product.name),  # Name length as proxy for complexity
            1 if 'artisan' in product.name.lower() else 0,
            1 if 'exclusive' in product.name.lower() else 0,
            1 if 'limited' in product.name.lower() else 0
        ]
        
        return np.array(features)

    def _get_client_cluster(self, ml_features):
        """
        Get client cluster information using KMeans
        """
        
        try:
            # Simple clustering based on key features
            features = [
                ml_features.get('avg_budget', 150),
                ml_features.get('premium_affinity', 0.3),
                ml_features.get('category_diversity', 0.5)
            ]
            
            # Define cluster centers (simplified)
            if features[0] > 200 and features[1] > 0.6:
                cluster = 'premium'
            elif features[0] < 100:
                cluster = 'value'
            else:
                cluster = 'standard'
            
            return {
                'cluster': cluster,
                'confidence': 0.8
            }
            
        except Exception as e:
            _logger.warning(f"Clustering failed: {e}")
            return {'cluster': 'standard', 'confidence': 0.5}

    def _analyze_market_position(self, partner_id, ml_features):
        """
        Analyze client's position in the market
        """
        
        avg_budget = ml_features.get('avg_budget', 150)
        
        # Get market statistics
        all_histories = self.env['client.order.history'].search([])
        market_budgets = [h.total_budget for h in all_histories if h.total_budget > 0]
        
        if not market_budgets:
            return {'percentile': 50, 'segment': 'standard'}
        
        market_avg = np.mean(market_budgets)
        market_std = np.std(market_budgets)
        
        # Calculate percentile
        percentile = sum(1 for b in market_budgets if b <= avg_budget) / len(market_budgets) * 100
        
        # Determine segment
        if percentile >= 80:
            segment = 'premium'
        elif percentile >= 50:
            segment = 'standard'
        else:
            segment = 'value'
        
        return {
            'percentile': percentile,
            'segment': segment,
            'market_avg': market_avg,
            'market_std': market_std,
            'position': 'above' if avg_budget > market_avg else 'below'
        }

    def _calculate_profile_confidence(self, base_analysis, text_analysis):
        """
        Calculate confidence in client profile
        """
        
        confidence = 0.5
        
        # Historical data bonus
        if base_analysis.get('has_history'):
            confidence += 0.2
            years = base_analysis.get('years_of_data', 0)
            confidence += min(years * 0.05, 0.15)
        
        # Text analysis bonus
        if text_analysis:
            confidence += 0.15
        
        return min(1.0, confidence)

    @api.model
    def cron_auto_train_models(self):
        """
        Automated model training based on schedule
        Called by cron job
        """
        
        _logger.info("Starting automated ML model training...")
        
        try:
            # Check if auto-training is enabled
            if not self.auto_retrain_enabled:
                _logger.info("Auto-training is disabled")
                return
            
            # Get or create engine
            engine = self.get_or_create_engine()
            
            # Check if enough time has passed since last training
            if engine.last_training_date:
                days_since = (fields.Datetime.now() - engine.last_training_date).days
                if days_since < engine.retrain_frequency_days:
                    _logger.info(f"Skipping training - only {days_since} days since last training")
                    return
            
            # Check if we have enough new data
            new_data_count = self.env['ml.learning.data'].search_count([
                ('engine_id', '=', engine.id),
                ('used_for_training', '=', False)
            ])
            
            min_new_samples = int(self.env['ir.config_parameter'].sudo().get_param(
                'lebigott_ai.min_new_samples_for_retrain', '20'
            ))
            
            if new_data_count < min_new_samples:
                _logger.info(f"Insufficient new data for retraining: {new_data_count} < {min_new_samples}")
                return
            
            # Perform training
            result = engine.train_models_from_sales_data(force_retrain=True)
            
            if result:
                # Send notification email
                self._send_training_notification(engine, 'success')
                _logger.info("Automated training completed successfully")
            else:
                self._send_training_notification(engine, 'failed')
                _logger.error("Automated training failed")
                
        except Exception as e:
            _logger.error(f"Cron auto-training error: {str(e)}")
            self._send_training_notification(None, 'error', str(e))

    @api.model
    def cron_clean_old_learning_data(self):
        """
        Clean old learning data to manage database size
        Called by cron job
        """
        
        _logger.info("Cleaning old ML learning data...")
        
        try:
            # Get retention period
            retention_days = int(self.env['ir.config_parameter'].sudo().get_param(
                'lebigott_ai.learning_data_retention_days', '180'
            ))
            
            cutoff_date = fields.Datetime.now() - timedelta(days=retention_days)
            
            # Find old records
            old_records = self.env['ml.learning.data'].search([
                ('created_date', '<', cutoff_date),
                ('used_for_training', '=', True)
            ])
            
            if old_records:
                count = len(old_records)
                old_records.unlink()
                _logger.info(f"Deleted {count} old learning records")
            else:
                _logger.info("No old learning records to delete")
                
            # Also clean old cache entries
            self._clean_old_cache_entries()
            
        except Exception as e:
            _logger.error(f"Cron cleaning error: {str(e)}")

    def _clean_old_cache_entries(self):
        """Clean old cache entries from ir.config_parameter"""
        
        cache_ttl = int(self.env['ir.config_parameter'].sudo().get_param(
            'lebigott_ai.cache_ttl_hours', '24'
        ))
        
        cutoff = datetime.now() - timedelta(hours=cache_ttl)
        cutoff_str = cutoff.strftime('%Y-%m-%d')
        
        # Find and delete old cache entries
        old_params = self.env['ir.config_parameter'].sudo().search([
            ('key', 'like', 'ai_feedback_%'),
            ('key', 'like', f'%{cutoff_str}%')
        ])
        
        if old_params:
            old_params.unlink()
            _logger.info(f"Cleaned {len(old_params)} cache entries")

    def _send_training_notification(self, engine, status, error_msg=None):
        """Send email notification about training status"""
        
        try:
            # Get admin users
            admin_group = self.env.ref('lebigott_ai.group_ml_admin')
            admin_users = admin_group.users
            
            if not admin_users:
                return
            
            # Prepare email content
            if status == 'success':
                subject = "ML Model Training Successful"
                body = f"""
                <p>Automated ML model training completed successfully.</p>
                <ul>
                    <li>Model Version: {engine.model_version}</li>
                    <li>Accuracy: {engine.model_accuracy:.1f}%</li>
                    <li>Training Samples: {engine.training_samples}</li>
                    <li>Duration: {engine.training_duration:.1f} minutes</li>
                </ul>
                """
            elif status == 'failed':
                subject = "ML Model Training Failed"
                body = "<p>Automated ML model training failed. Please check the logs.</p>"
            else:
                subject = "ML Model Training Error"
                body = f"<p>Error during automated training: {error_msg}</p>"
            
            # Send email
            for user in admin_users:
                if user.email:
                    mail_values = {
                        'subject': subject,
                        'body_html': body,
                        'email_to': user.email,
                        'email_from': self.env.company.email or 'noreply@company.com',
                    }
                    self.env['mail.mail'].create(mail_values).send()
                    
        except Exception as e:
            _logger.error(f"Failed to send training notification: {e}")

    # Add to IntegrationManager class

    @api.model
    def cron_analyze_performance(self):
        """
        Analyze and log performance metrics
        Called by cron job
        """
        
        _logger.info("Running performance analytics...")
        
        try:
            # Get performance stats
            stats = self.analyze_engine_performance()
            
            # Log summary
            for engine, metrics in stats.items():
                _logger.info(f"""
                Engine: {engine}
                - Compositions: {metrics['count']}
                - Avg Budget Variance: {metrics.get('avg_variance', 0):.1f}%
                - Success Rate: {metrics.get('success_rate', 0):.1f}%
                - Avg Confidence: {metrics.get('avg_confidence', 0):.2f}
                """)
            
            # Store daily snapshot
            self._store_performance_snapshot(stats)
            
            # Check for anomalies
            self._check_performance_anomalies(stats)
            
        except Exception as e:
            _logger.error(f"Performance analytics error: {str(e)}")

    def _store_performance_snapshot(self, stats):
        """Store daily performance snapshot"""
        
        snapshot_data = {
            'date': fields.Date.today(),
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in config parameter (or could use a dedicated model)
        key = f"ml_performance_snapshot_{fields.Date.today()}"
        self.env['ir.config_parameter'].sudo().set_param(
            key, json.dumps(snapshot_data)
        )

    def _check_performance_anomalies(self, stats):
        """Check for performance anomalies and alert if needed"""
        
        alerts = []
        
        for engine, metrics in stats.items():
            # Check for low success rate
            if metrics.get('success_rate', 100) < 50:
                alerts.append(f"{engine} engine has low success rate: {metrics['success_rate']:.1f}%")
            
            # Check for high budget variance
            if metrics.get('avg_variance', 0) > 25:
                alerts.append(f"{engine} engine has high budget variance: {metrics['avg_variance']:.1f}%")
            
            # Check for low confidence
            if metrics.get('avg_confidence', 1) < 0.5:
                alerts.append(f"{engine} engine has low confidence: {metrics['avg_confidence']:.2f}")
        
        if alerts:
            self._send_anomaly_alert(alerts)

    def _send_anomaly_alert(self, alerts):
        """Send alert about performance anomalies"""
        
        try:
            admin_group = self.env.ref('lebigott_ai.group_ml_admin')
            admin_users = admin_group.users
            
            if not admin_users or not alerts:
                return
            
            body = """
            <p>Performance anomalies detected in ML/AI system:</p>
            <ul>
            """
            
            for alert in alerts:
                body += f"<li>{alert}</li>"
            
            body += "</ul><p>Please review the system performance.</p>"
            
            for user in admin_users:
                if user.email:
                    mail_values = {
                        'subject': "ML/AI Performance Alert",
                        'body_html': body,
                        'email_to': user.email,
                        'email_from': self.env.company.email or 'noreply@company.com',
                    }
                    self.env['mail.mail'].create(mail_values).send()
                    
        except Exception as e:
            _logger.error(f"Failed to send anomaly alert: {e}")

# Learning Data Storage Model
class MLLearningData(models.Model):
    _name = 'ml.learning.data'
    _description = 'ML Learning Data Storage'
    
    engine_id = fields.Many2one('ml.recommendation.engine', string="Engine", required=True, ondelete='cascade')
    partner_id = fields.Many2one('res.partner', string="Client", required=True)
    target_budget = fields.Float(string="Target Budget")
    actual_cost = fields.Float(string="Actual Cost")
    product_count = fields.Integer(string="Product Count")
    data = fields.Text(string="Learning Data JSON")
    created_date = fields.Datetime(string="Created Date", default=fields.Datetime.now)
    used_for_training = fields.Boolean(string="Used for Training", default=False)