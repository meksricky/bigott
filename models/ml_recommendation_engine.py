from odoo import models, fields, api
from odoo.exceptions import UserError
import json
import logging
import numpy as np
import requests
import pickle
import os
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)

class MLRecommendationEngine(models.Model):
    _name = 'ml.recommendation.engine'
    _description = 'Machine Learning Recommendation Engine'
    
    # Model storage fields
    model_version = fields.Char(string="Model Version", default="1.0.0")
    last_training_date = fields.Datetime(string="Last Training Date")
    training_samples = fields.Integer(string="Training Samples")
    model_accuracy = fields.Float(string="Model Accuracy (%)")
    is_model_trained = fields.Boolean(string="Model Trained", default=False)
    
    def __init__(self):
        super().__init__()
        self.models_path = '/opt/odoo/ml_models/'
        self.ensure_models_directory()
        
        # ML Models
        self.recommendation_model = None
        self.price_predictor = None
        self.client_segmentation_model = None
        self.scaler = None
        
        # Load trained models if available
        self.load_trained_models()
    
    def ensure_models_directory(self):
        """Ensure ML models directory exists"""
        os.makedirs(self.models_path, exist_ok=True)
    
    @api.model
    def get_smart_recommendations(self, partner_id, target_budget, dietary_restrictions=None, 
                                notes_text=None, max_attempts=3):
        """
        Main entry point: Get AI/ML-powered product recommendations
        """
        
        for attempt in range(1, max_attempts + 1):
            try:
                _logger.info(f"ML recommendation attempt {attempt} for partner {partner_id}")
                
                # 1. Get client embeddings using Ollama
                client_embedding = self._get_client_embedding(partner_id, notes_text)
                
                # 2. Get available products with embeddings
                available_products = self._get_products_with_embeddings(dietary_restrictions)
                
                # 3. Use trained ML models for recommendation
                if self.is_model_trained:
                    recommendations = self._ml_predict_products(
                        client_embedding, available_products, target_budget, attempt
                    )
                else:
                    # Fallback to rule-based while training models
                    recommendations = self._hybrid_recommendations(
                        client_embedding, available_products, target_budget, attempt
                    )
                
                # 4. Budget optimization using ML price predictor
                optimized_selection = self._ml_budget_optimization(
                    recommendations, target_budget, attempt
                )
                
                # 5. Validate selection
                final_cost = sum(p.list_price for p in optimized_selection)
                variance = abs(final_cost - target_budget) / target_budget * 100
                
                # Allow ±5% normally, extend to ±15% on later attempts
                max_variance = 5 + (attempt - 1) * 5  # 5%, 10%, 15%
                
                if variance <= max_variance:
                    # 6. Record this recommendation for continuous learning
                    self._record_for_online_learning(
                        partner_id, optimized_selection, target_budget, client_embedding
                    )
                    
                    return {
                        'products': optimized_selection,
                        'actual_cost': final_cost,
                        'budget_variance': variance,
                        'ml_confidence': self._calculate_ml_confidence(optimized_selection, client_embedding),
                        'reasoning': self._generate_ml_reasoning(optimized_selection, client_embedding, target_budget),
                        'attempt': attempt,
                        'method': 'ML' if self.is_model_trained else 'Hybrid'
                    }
                
                _logger.warning(f"Attempt {attempt}: Budget variance {variance:.1f}% > {max_variance}%")
                
            except Exception as e:
                _logger.error(f"ML recommendation attempt {attempt} failed: {str(e)}")
                
        raise UserError(f"Could not generate suitable recommendation after {max_attempts} attempts")
    
    def _get_client_embedding(self, partner_id, notes_text=None):
        """Generate client embedding using Ollama + historical data"""
        
        # Get historical data
        client_data = self._extract_client_features(partner_id)
        
        # Get text embedding from Ollama if notes provided
        text_embedding = []
        if notes_text:
            text_embedding = self._get_ollama_embedding(notes_text)
        
        # Combine numerical features with text embedding
        combined_embedding = {
            'partner_id': partner_id,
            'numerical_features': client_data,
            'text_embedding': text_embedding,
            'timestamp': datetime.now().timestamp()
        }
        
        return combined_embedding
    
    def _get_ollama_embedding(self, text):
        """Get text embedding using Ollama"""
        
        if not self._ollama_enabled():
            return []
        
        try:
            embedding_model = self.env['ir.config_parameter'].sudo().get_param(
                'lebigott_ai.ollama_embedding_model', 'nomic-embed-text'
            )
            
            url = self._ollama_base_url().rstrip('/') + '/api/embeddings'
            payload = {
                'model': embedding_model,
                'prompt': text
            }
            
            resp = requests.post(url, json=payload, timeout=30)
            if resp.ok:
                data = resp.json()
                return data.get('embedding', [])
                
        except Exception as e:
            _logger.warning(f'Ollama embedding failed: {e}')
        
        return []
    
    def _extract_client_features(self, partner_id):
        """Extract numerical features for ML model"""
        
        partner = self.env['res.partner'].browse(partner_id)
        histories = self.env['client.order.history'].search([
            ('partner_id', '=', partner_id)
        ], order='order_year desc')
        
        if not histories:
            # New client - use defaults
            return {
                'avg_budget': 100.0,
                'budget_trend': 0.0,
                'years_as_client': 0,
                'avg_satisfaction': 3.5,
                'category_diversity': 0.5,
                'price_sensitivity': 0.5,
                'premium_affinity': 0.3
            }
        
        # Calculate features
        budgets = [h.total_budget for h in histories]
        satisfactions = [float(h.client_satisfaction) for h in histories if h.client_satisfaction]
        
        # Budget trend calculation
        budget_trend = 0.0
        if len(budgets) >= 2:
            recent_avg = np.mean(budgets[:2]) if len(budgets) >= 2 else budgets[0]
            older_avg = np.mean(budgets[2:]) if len(budgets) > 2 else budgets[0]
            budget_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        # Category diversity
        all_categories = []
        for h in histories:
            categories = h.get_category_structure()
            all_categories.extend(categories.keys())
        
        unique_categories = len(set(all_categories))
        category_diversity = min(unique_categories / 8.0, 1.0)  # Normalize to 8 max categories
        
        # Premium affinity (based on average product grade)
        premium_count = 0
        total_products = 0
        for h in histories:
            for product in h.product_ids:
                total_products += 1
                if hasattr(product, 'product_grade') and product.product_grade in ['premium', 'luxury']:
                    premium_count += 1
        
        premium_affinity = premium_count / total_products if total_products > 0 else 0.3
        
        return {
            'avg_budget': np.mean(budgets),
            'budget_trend': budget_trend,
            'years_as_client': len(histories),
            'avg_satisfaction': np.mean(satisfactions) if satisfactions else 3.5,
            'category_diversity': category_diversity,
            'price_sensitivity': 1.0 - (np.std(budgets) / np.mean(budgets) if np.mean(budgets) > 0 else 0.5),
            'premium_affinity': premium_affinity
        }
    
    def _ml_predict_products(self, client_embedding, available_products, target_budget, attempt):
        """Use trained ML model to predict product preferences"""
        
        if not self.recommendation_model:
            raise UserError("ML model not loaded")
        
        product_scores = []
        client_features = self._prepare_client_features_for_ml(client_embedding)
        
        for product in available_products:
            # Combine client and product features
            product_features = self._prepare_product_features_for_ml(product)
            combined_features = np.concatenate([client_features, product_features])
            
            # Predict compatibility score
            try:
                if self.scaler:
                    combined_features = self.scaler.transform([combined_features])[0]
                
                score = self.recommendation_model.predict([combined_features])[0]
                
                # Adjust score based on attempt (add randomness for variety)
                if attempt > 1:
                    score += np.random.normal(0, 0.1 * attempt)
                
                product_scores.append({
                    'product': product,
                    'ml_score': max(0, min(1, score)),
                    'features': combined_features
                })
                
            except Exception as e:
                _logger.warning(f"ML prediction failed for product {product.id}: {e}")
                # Fallback score
                product_scores.append({
                    'product': product,
                    'ml_score': 0.5,
                    'features': combined_features
                })
        
        # Sort by ML score
        return sorted(product_scores, key=lambda x: x['ml_score'], reverse=True)
    
    def _ml_budget_optimization(self, scored_products, target_budget, attempt):
        """Use ML to optimize product selection for budget"""
        
        # Flexible budget based on attempt
        flexibility = 0.05 + (attempt - 1) * 0.05  # 5%, 10%, 15%
        min_budget = target_budget * (1 - flexibility)
        max_budget = target_budget * (1 + flexibility)
        
        # Knapsack-style optimization with ML scores
        selected_products = []
        current_cost = 0
        categories_used = set()
        
        # First pass: one from each category (diversification)
        category_groups = defaultdict(list)
        for item in scored_products:
            category = item['product'].lebiggot_category
            category_groups[category].append(item)
        
        # Select best from each category first
        for category, items in category_groups.items():
            best_item = max(items, key=lambda x: x['ml_score'])
            product = best_item['product']
            
            if current_cost + product.list_price <= max_budget:
                selected_products.append(product)
                current_cost += product.list_price
                categories_used.add(category)
        
        # Second pass: fill remaining budget with highest scoring products
        remaining_items = [item for category_items in category_groups.values() 
                          for item in category_items[1:]]  # Skip already selected
        remaining_items.sort(key=lambda x: x['ml_score'], reverse=True)
        
        for item in remaining_items:
            product = item['product']
            if current_cost + product.list_price <= max_budget and len(selected_products) < 8:
                selected_products.append(product)
                current_cost += product.list_price
        
        # Ensure minimum budget if possible
        if current_cost < min_budget:
            selected_products = self._upgrade_selection_for_budget(
                selected_products, scored_products, min_budget, max_budget
            )
        
        return selected_products
    
    @api.model
    def train_models_from_sales_data(self, force_retrain=False):
        """
        Main training function: Train ML models from historical sales data
        This should be run periodically (daily/weekly) not on every recommendation
        """
        
        if not force_retrain and self.is_model_trained:
            days_since_training = (datetime.now() - self.last_training_date).days if self.last_training_date else 999
            if days_since_training < 7:  # Retrain weekly
                _logger.info("Models recently trained, skipping...")
                return
        
        _logger.info("Starting ML model training from sales data...")
        
        try:
            # 1. Extract training data from sales
            training_data = self._extract_training_data()
            
            if len(training_data) < 50:  # Need minimum data
                _logger.warning("Insufficient training data, using hybrid approach")
                return False
            
            # 2. Train recommendation model
            self._train_recommendation_model(training_data)
            
            # 3. Train price predictor
            self._train_price_predictor(training_data)
            
            # 4. Train client segmentation
            self._train_client_segmentation(training_data)
            
            # 5. Save trained models
            self._save_trained_models()
            
            # 6. Update training status
            self.write({
                'last_training_date': datetime.now(),
                'training_samples': len(training_data),
                'is_model_trained': True,
                'model_version': f"{self.model_version[:-1]}{int(self.model_version[-1]) + 1}"
            })
            
            _logger.info(f"ML model training completed with {len(training_data)} samples")
            return True
            
        except Exception as e:
            _logger.error(f"ML model training failed: {str(e)}")
            return False
    
    def _extract_training_data(self):
        """Extract training data from historical sales orders"""
        
        training_data = []
        
        # Get sales orders from last 3 years
        cutoff_date = datetime.now() - timedelta(days=3*365)
        
        sales_orders = self.env['sale.order'].search([
            ('state', 'in', ['sale', 'done']),
            ('confirmation_date', '>=', cutoff_date),
            ('partner_id.is_company', '=', False)  # Only individual clients
        ])
        
        for order in sales_orders:
            try:
                # Extract client features at time of sale
                client_features = self._extract_historical_client_features(
                    order.partner_id.id, order.confirmation_date
                )
                
                # Extract order features
                order_features = {
                    'total_amount': order.amount_total,
                    'product_count': len(order.order_line),
                    'satisfaction': self._get_order_satisfaction(order),
                    'categories': self._extract_order_categories(order)
                }
                
                # Create training samples for each product in the order
                for line in order.order_line:
                    if line.product_template_id and line.product_uom_qty > 0:
                        product_features = self._prepare_product_features_for_ml(line.product_template_id)
                        
                        # Target: was this product selected? (1 for selected, 0 for not)
                        training_sample = {
                            'client_features': client_features,
                            'product_features': product_features,
                            'order_context': order_features,
                            'target_selected': 1.0,  # This product was actually selected
                            'target_satisfaction': order_features['satisfaction'],
                            'timestamp': order.confirmation_date
                        }
                        
                        training_data.append(training_sample)
                
                # Also create negative samples (products that could have been selected but weren't)
                self._add_negative_samples(training_data[-len(order.order_line):], order)
                
            except Exception as e:
                _logger.warning(f"Failed to extract training data from order {order.id}: {e}")
                continue
        
        return training_data
    
    def _train_recommendation_model(self, training_data):
        """Train the main product recommendation model"""
        
        # Prepare features and targets
        X = []  # Features
        y = []  # Targets (selection probability)
        
        for sample in training_data:
            client_features = list(sample['client_features'].values())
            product_features = list(sample['product_features'])
            order_context = [
                sample['order_context']['total_amount'] / 1000,  # Normalize
                sample['order_context']['product_count'],
                len(sample['order_context']['categories'])
            ]
            
            combined_features = client_features + product_features + order_context
            X.append(combined_features)
            y.append(sample['target_selected'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle any NaN values
        X = np.nan_to_num(X)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train Random Forest (good for this type of problem)
        self.recommendation_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.recommendation_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.recommendation_model.predict(X_test)
        accuracy = 1 - mean_squared_error(y_test, y_pred)  # Convert MSE to accuracy-like metric
        
        self.model_accuracy = max(0, min(100, accuracy * 100))
        
        _logger.info(f"Recommendation model trained with accuracy: {self.model_accuracy:.1f}%")
    
    def _save_trained_models(self):
        """Save trained models to disk"""
        
        try:
            # Save models using joblib
            if self.recommendation_model:
                joblib.dump(self.recommendation_model, f"{self.models_path}/recommendation_model.pkl")
            
            if self.price_predictor:
                joblib.dump(self.price_predictor, f"{self.models_path}/price_predictor.pkl")
                
            if self.scaler:
                joblib.dump(self.scaler, f"{self.models_path}/scaler.pkl")
            
            _logger.info("ML models saved successfully")
            
        except Exception as e:
            _logger.error(f"Failed to save ML models: {str(e)}")
    
    def load_trained_models(self):
        """Load trained models from disk"""
        
        try:
            recommendation_path = f"{self.models_path}/recommendation_model.pkl"
            scaler_path = f"{self.models_path}/scaler.pkl"
            
            if os.path.exists(recommendation_path):
                self.recommendation_model = joblib.load(recommendation_path)
                _logger.info("Recommendation model loaded")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                _logger.info("Scaler loaded")
                
            self.is_model_trained = (self.recommendation_model is not None)
            
        except Exception as e:
            _logger.warning(f"Failed to load ML models: {str(e)}")
            self.is_model_trained = False
    
    def _record_for_online_learning(self, partner_id, selected_products, target_budget, client_embedding):
        """Record recommendation for online learning (continuous improvement)"""
        
        learning_record = {
            'partner_id': partner_id,
            'products': [p.id for p in selected_products],
            'target_budget': target_budget,
            'actual_cost': sum(p.list_price for p in selected_products),
            'client_embedding': client_embedding,
            'timestamp': datetime.now().isoformat(),
            'used_for_learning': False  # Will be used in next training cycle
        }
        
        # Store in parameter for batch processing
        learning_key = f"ml_learning_record_{partner_id}_{int(datetime.now().timestamp())}"
        self.env['ir.config_parameter'].sudo().set_param(
            learning_key, 
            json.dumps(learning_record)
        )
    
    # Helper methods
    def _ollama_enabled(self):
        return self.env['ir.config_parameter'].sudo().get_param('lebigott_ai.ollama_enabled', 'false').lower() == 'true'
    
    def _ollama_base_url(self):
        return self.env['ir.config_parameter'].sudo().get_param('lebigott_ai.ollama_base_url', 'http://localhost:11434')