from odoo import models, fields, api
from odoo.exceptions import UserError
import json
import logging
from datetime import datetime, timedelta
import base64
import io
import csv

_logger = logging.getLogger(__name__)

class MLTrainingWizard(models.TransientModel):
    _name = 'ml.training.wizard'
    _description = 'ML Model Training Wizard'
    
    # Training Configuration
    training_type = fields.Selection([
        ('full', 'Full Training (All Data)'),
        ('incremental', 'Incremental Training (Recent Data)'),
        ('custom', 'Custom Date Range')
    ], string="Training Type", default='full', required=True)
    
    date_from = fields.Date(string="From Date")
    date_to = fields.Date(string="To Date")
    
    # Data Sources
    use_sales_data = fields.Boolean(string="Use Sales Data", default=True)
    use_composition_data = fields.Boolean(string="Use Composition Data", default=True)
    use_client_history = fields.Boolean(string="Use Client History", default=True)
    use_learning_data = fields.Boolean(string="Use Learning Data", default=True)
    
    # Training Parameters
    test_size = fields.Float(string="Test Set Size (%)", default=0.2)
    min_samples = fields.Integer(string="Minimum Training Samples", default=50)
    cross_validation_folds = fields.Integer(string="Cross-Validation Folds", default=5)
    
    # Model Selection
    model_type = fields.Selection([
        ('auto', 'Auto-Select Best Model'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('ensemble', 'Ensemble (Multiple Models)')
    ], string="Model Type", default='auto')
    
    # Advanced Settings
    feature_selection = fields.Boolean(string="Automatic Feature Selection", default=True)
    hyperparameter_tuning = fields.Boolean(string="Hyperparameter Tuning", default=False)
    
    # Information Fields
    current_model_info = fields.Text(string="Current Model Info", compute='_compute_current_model_info')
    estimated_samples = fields.Integer(string="Estimated Samples", compute='_compute_estimated_samples')
    
    @api.depends('training_type', 'date_from', 'date_to')
    def _compute_estimated_samples(self):
        """Estimate number of training samples"""
        for wizard in self:
            sample_count = 0
            
            # Count sales orders
            if wizard.use_sales_data:
                domain = [('state', 'in', ['sale', 'done'])]
                if wizard.training_type == 'custom' and wizard.date_from and wizard.date_to:
                    domain.append(('date_order', '>=', wizard.date_from))
                    domain.append(('date_order', '<=', wizard.date_to))
                elif wizard.training_type == 'incremental':
                    domain.append(('date_order', '>=', fields.Date.today() - timedelta(days=90)))
                
                sample_count += self.env['sale.order'].search_count(domain)
            
            # Count compositions
            if wizard.use_composition_data:
                domain = [('state', 'in', ['approved', 'delivered'])]
                if wizard.training_type == 'custom' and wizard.date_from and wizard.date_to:
                    domain.append(('create_date', '>=', wizard.date_from))
                    domain.append(('create_date', '<=', wizard.date_to))
                elif wizard.training_type == 'incremental':
                    domain.append(('create_date', '>=', fields.Date.today() - timedelta(days=90)))
                
                sample_count += self.env['gift.composition'].search_count(domain)
            
            # Count client histories
            if wizard.use_client_history:
                sample_count += self.env['client.order.history'].search_count([])
            
            # Count learning data
            if wizard.use_learning_data:
                ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
                sample_count += len(ml_engine.learning_data_ids)
            
            wizard.estimated_samples = sample_count
    
    @api.depends('training_type')
    def _compute_current_model_info(self):
        """Display current model information"""
        for wizard in self:
            ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
            
            if ml_engine.is_model_trained:
                info = f"""
Current Model Status:
- Version: {ml_engine.model_version}
- Trained: {ml_engine.last_training_date.strftime('%Y-%m-%d %H:%M') if ml_engine.last_training_date else 'Never'}
- Training Samples: {ml_engine.training_samples:,}
- Accuracy: {ml_engine.model_accuracy:.1f}%
- R² Score: {ml_engine.r2_score:.3f}
- MSE: {ml_engine.mse_score:.3f}
- Cross-Validation Score: {ml_engine.cv_score:.3f}
                """
            else:
                info = "No model currently trained. This will be the first training."
            
            wizard.current_model_info = info
    
    def action_train_model(self):
        """Execute model training"""
        self.ensure_one()
        
        if self.estimated_samples < self.min_samples:
            raise UserError(
                f"Insufficient training data: {self.estimated_samples} samples "
                f"(minimum: {self.min_samples}). Try expanding your data sources or date range."
            )
        
        # Prepare training context
        context = {
            'training_type': self.training_type,
            'date_from': self.date_from,
            'date_to': self.date_to,
            'data_sources': {
                'sales': self.use_sales_data,
                'compositions': self.use_composition_data,
                'history': self.use_client_history,
                'learning': self.use_learning_data
            },
            'parameters': {
                'test_size': self.test_size,
                'cv_folds': self.cross_validation_folds,
                'model_type': self.model_type,
                'feature_selection': self.feature_selection,
                'hyperparameter_tuning': self.hyperparameter_tuning
            }
        }
        
        # Execute training
        ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
        
        try:
            # Extract training data
            training_data = self._extract_training_data(context)
            
            if len(training_data) < self.min_samples:
                raise UserError(f"Extracted only {len(training_data)} samples, minimum is {self.min_samples}")
            
            # Train model
            result = ml_engine.train_advanced_model(
                training_data=training_data,
                context=context
            )
            
            # Show success message
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Training Successful',
                    'message': f"""
                        Model trained successfully!
                        - Samples: {result['samples']}
                        - Accuracy: {result['accuracy']:.1f}%
                        - Duration: {result['duration']:.1f} minutes
                        - Model Type: {result['model_type']}
                    """,
                    'type': 'success',
                    'sticky': False,
                    'next': {'type': 'ir.actions.act_window_close'}
                }
            }
            
        except Exception as e:
            _logger.error(f"Training failed: {str(e)}")
            raise UserError(f"Training failed: {str(e)}")
    
    def _extract_training_data(self, context):
        """Extract training data from various sources"""
        
        training_data = []
        
        # Extract from sales orders
        if context['data_sources']['sales']:
            sales_data = self._extract_sales_data(context)
            training_data.extend(sales_data)
        
        # Extract from compositions
        if context['data_sources']['compositions']:
            composition_data = self._extract_composition_data(context)
            training_data.extend(composition_data)
        
        # Extract from client history
        if context['data_sources']['history']:
            history_data = self._extract_history_data(context)
            training_data.extend(history_data)
        
        # Extract from learning data
        if context['data_sources']['learning']:
            learning_data = self._extract_learning_data(context)
            training_data.extend(learning_data)
        
        _logger.info(f"Extracted {len(training_data)} total training samples")
        
        return training_data
    
    def _extract_sales_data(self, context):
        """Extract training data from sales orders"""
        
        domain = [('state', 'in', ['sale', 'done'])]
        
        if context['training_type'] == 'custom':
            if context['date_from']:
                domain.append(('date_order', '>=', context['date_from']))
            if context['date_to']:
                domain.append(('date_order', '<=', context['date_to']))
        elif context['training_type'] == 'incremental':
            domain.append(('date_order', '>=', fields.Date.today() - timedelta(days=90)))
        
        orders = self.env['sale.order'].search(domain)
        
        training_data = []
        for order in orders:
            # Extract features
            features = self._extract_order_features(order)
            
            # Create training sample
            sample = {
                'type': 'sales_order',
                'partner_id': order.partner_id.id,
                'date': order.date_order,
                'products': order.order_line.mapped('product_id.product_tmpl_id.id'),
                'categories': list(set(order.order_line.mapped('product_id.product_tmpl_id.lebiggot_category'))),
                'total_amount': order.amount_total,
                'features': features,
                'target': 1.0  # Successful sale
            }
            
            training_data.append(sample)
        
        _logger.info(f"Extracted {len(training_data)} samples from sales orders")
        return training_data
    
    def _extract_composition_data(self, context):
        """Extract training data from gift compositions"""
        
        domain = [('state', 'in', ['approved', 'delivered'])]
        
        if context['training_type'] == 'custom':
            if context['date_from']:
                domain.append(('create_date', '>=', context['date_from']))
            if context['date_to']:
                domain.append(('create_date', '<=', context['date_to']))
        elif context['training_type'] == 'incremental':
            domain.append(('create_date', '>=', fields.Date.today() - timedelta(days=90)))
        
        compositions = self.env['gift.composition'].search(domain)
        
        training_data = []
        for comp in compositions:
            # Extract features
            features = self._extract_composition_features(comp)
            
            # Calculate target score based on success indicators
            target_score = self._calculate_composition_success(comp)
            
            sample = {
                'type': 'composition',
                'partner_id': comp.partner_id.id,
                'date': comp.create_date,
                'products': comp.product_ids.ids,
                'categories': list(set(comp.product_ids.mapped('lebiggot_category'))),
                'budget': comp.target_budget,
                'actual_cost': comp.actual_cost,
                'features': features,
                'target': target_score
            }
            
            training_data.append(sample)
        
        _logger.info(f"Extracted {len(training_data)} samples from compositions")
        return training_data
    
    def _extract_history_data(self, context):
        """Extract training data from client history"""
        
        histories = self.env['client.order.history'].search([])
        
        training_data = []
        for history in histories:
            # Extract features
            features = self._extract_history_features(history)
            
            # Create training sample
            sample = {
                'type': 'history',
                'partner_id': history.partner_id.id,
                'year': history.order_year,
                'products': history.product_ids.ids,
                'categories': list(set(history.product_ids.mapped('lebiggot_category'))),
                'budget': history.total_budget,
                'features': features,
                'target': float(history.client_satisfaction) / 5.0 if history.client_satisfaction else 0.7
            }
            
            training_data.append(sample)
        
        _logger.info(f"Extracted {len(training_data)} samples from client history")
        return training_data
    
    def _extract_learning_data(self, context):
        """Extract stored learning data"""
        
        ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
        learning_records = ml_engine.learning_data_ids
        
        if not learning_records:
            return []
        
        training_data = []
        for record in learning_records:
            if record.data:
                try:
                    data = json.loads(record.data)
                    
                    sample = {
                        'type': 'learning',
                        'partner_id': record.partner_id.id,
                        'date': record.created_date,
                        'budget': record.target_budget,
                        'actual_cost': record.actual_cost,
                        'features': data.get('client_profile', {}).get('ml_features', {}),
                        'target': 0.8  # Default success score for learning data
                    }
                    
                    training_data.append(sample)
                    
                except Exception as e:
                    _logger.warning(f"Failed to parse learning data {record.id}: {e}")
        
        _logger.info(f"Extracted {len(training_data)} samples from learning data")
        return training_data
    
    def _extract_order_features(self, order):
        """Extract ML features from sales order"""
        
        features = {
            'order_total': float(order.amount_total),
            'product_count': len(order.order_line),
            'unique_categories': len(set(order.order_line.mapped('product_id.product_tmpl_id.lebiggot_category'))),
            'avg_line_price': float(order.amount_total / len(order.order_line)) if order.order_line else 0,
            'partner_country': 1 if order.partner_id.country_id.code == 'ES' else 0,
            'is_company': 1 if order.partner_id.is_company else 0,
            'order_month': order.date_order.month if order.date_order else 0,
            'order_quarter': ((order.date_order.month - 1) // 3) + 1 if order.date_order else 0
        }
        
        return features
    
    def _extract_composition_features(self, comp):
        """Extract ML features from composition"""
        
        features = {
            'budget': float(comp.target_budget),
            'actual_cost': float(comp.actual_cost),
            'budget_variance': abs(comp.actual_cost - comp.target_budget) / comp.target_budget if comp.target_budget > 0 else 0,
            'product_count': len(comp.product_ids),
            'confidence_score': float(comp.confidence_score),
            'novelty_score': float(comp.novelty_score),
            'historical_compatibility': float(comp.historical_compatibility),
            'unique_categories': len(set(comp.product_ids.mapped('lebiggot_category'))),
            'has_dietary_restrictions': 1 if comp.dietary_restrictions else 0
        }
        
        return features
    
    def _extract_history_features(self, history):
        """Extract ML features from client history"""
        
        features = {
            'total_budget': float(history.total_budget),
            'product_count': len(history.product_ids),
            'satisfaction': float(history.client_satisfaction) if history.client_satisfaction else 3.5,
            'has_notes': 1 if history.notes else 0,
            'year': history.order_year,
            'years_as_client': datetime.now().year - history.order_year
        }
        
        # Add category breakdown
        categories = history.get_category_structure()
        for i, (cat, count) in enumerate(categories.items()):
            if i < 5:  # Limit to top 5 categories
                features[f'category_{i}'] = count
        
        return features
    
    def _calculate_composition_success(self, comp):
        """Calculate success score for a composition"""
        
        score = 0.5  # Base score
        
        # Budget adherence (30%)
        budget_variance = abs(comp.actual_cost - comp.target_budget) / comp.target_budget if comp.target_budget > 0 else 0
        if budget_variance < 0.05:
            score += 0.3
        elif budget_variance < 0.1:
            score += 0.2
        elif budget_variance < 0.15:
            score += 0.1
        
        # Confidence score (20%)
        score += comp.confidence_score * 0.2
        
        # Historical compatibility (20%)
        score += comp.historical_compatibility * 0.2
        
        # Novelty score (10%)
        score += comp.novelty_score * 0.1
        
        # State bonus (20%)
        if comp.state == 'delivered':
            score += 0.2
        elif comp.state == 'approved':
            score += 0.15
        
        return min(1.0, max(0.0, score))
    
    def action_evaluate_model(self):
        """Evaluate current model performance"""
        self.ensure_one()
        
        ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
        
        if not ml_engine.is_model_trained:
            raise UserError("No trained model found. Please train a model first.")
        
        # Run evaluation
        evaluation_result = ml_engine.evaluate_model_performance()
        
        # Create evaluation report
        self._create_evaluation_report(evaluation_result)
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Model Evaluation Complete',
                'message': f"""
                    Model Evaluation Results:
                    - Accuracy: {evaluation_result['accuracy']:.1f}%
                    - Precision: {evaluation_result['precision']:.3f}
                    - Recall: {evaluation_result['recall']:.3f}
                    - F1 Score: {evaluation_result['f1_score']:.3f}
                """,
                'type': 'info',
                'sticky': True,
            }
        }
    
    def action_export_training_data(self):
        """Export training data for analysis"""
        self.ensure_one()
        
        # Extract training data
        context = {
            'training_type': self.training_type,
            'date_from': self.date_from,
            'date_to': self.date_to,
            'data_sources': {
                'sales': self.use_sales_data,
                'compositions': self.use_composition_data,
                'history': self.use_client_history,
                'learning': self.use_learning_data
            }
        }
        
        training_data = self._extract_training_data(context)
        
        if not training_data:
            raise UserError("No training data to export")
        
        # Create CSV
        output = io.StringIO()
        
        # Get all unique feature keys
        all_keys = set()
        for sample in training_data:
            if 'features' in sample:
                all_keys.update(sample['features'].keys())
        
        # Define CSV columns
        fieldnames = ['type', 'partner_id', 'date', 'budget', 'target'] + sorted(all_keys)
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for sample in training_data:
            row = {
                'type': sample.get('type', ''),
                'partner_id': sample.get('partner_id', ''),
                'date': sample.get('date', ''),
                'budget': sample.get('budget', sample.get('total_amount', '')),
                'target': sample.get('target', '')
            }
            
            # Add features
            if 'features' in sample:
                for key, value in sample['features'].items():
                    row[key] = value
            
            writer.writerow(row)
        
        # Create attachment
        csv_data = output.getvalue()
        attachment = self.env['ir.attachment'].create({
            'name': f'ml_training_data_{fields.Date.today()}.csv',
            'type': 'binary',
            'datas': base64.b64encode(csv_data.encode('utf-8')),
            'res_model': self._name,
            'res_id': self.id,
            'mimetype': 'text/csv'
        })
        
        # Return download action
        return {
            'type': 'ir.actions.act_url',
            'url': f'/web/content/{attachment.id}?download=true',
            'target': 'self',
        }
    
    def action_reset_model(self):
        """Reset/delete current model"""
        self.ensure_one()
        
        ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
        
        # Confirm action
        return {
            'type': 'ir.actions.act_window',
            'name': 'Confirm Model Reset',
            'res_model': 'ml.training.reset.confirm',
            'view_mode': 'form',
            'target': 'new',
            'context': {'default_ml_engine_id': ml_engine.id}
        }
    
    def _create_evaluation_report(self, evaluation_result):
        """Create detailed evaluation report"""
        
        report_content = f"""
        ML Model Evaluation Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        Performance Metrics:
        - Accuracy: {evaluation_result['accuracy']:.1f}%
        - Precision: {evaluation_result['precision']:.3f}
        - Recall: {evaluation_result['recall']:.3f}
        - F1 Score: {evaluation_result['f1_score']:.3f}
        
        Feature Importance:
        {self._format_feature_importance(evaluation_result.get('feature_importance', {}))}
        
        Recommendations:
        {self._generate_improvement_recommendations(evaluation_result)}
        """
        
        # Store report (could be saved to a model or sent via email)
        _logger.info(report_content)
    
    def _format_feature_importance(self, importance_dict):
        """Format feature importance for report"""
        if not importance_dict:
            return "Not available"
        
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        lines = []
        for feature, importance in sorted_features:
            lines.append(f"  - {feature}: {importance:.3f}")
        
        return '\n'.join(lines)
    
    def _generate_improvement_recommendations(self, evaluation_result):
        """Generate recommendations for model improvement"""
        
        recommendations = []
        
        accuracy = evaluation_result.get('accuracy', 0)
        
        if accuracy < 70:
            recommendations.append("- Consider collecting more training data")
            recommendations.append("- Review feature engineering approach")
            recommendations.append("- Try hyperparameter tuning")
        elif accuracy < 85:
            recommendations.append("- Model performance is good, consider incremental improvements")
            recommendations.append("- Focus on specific underperforming categories")
        else:
            recommendations.append("- Excellent model performance")
            recommendations.append("- Monitor for overfitting")
            recommendations.append("- Consider ensemble methods for marginal gains")
        
        return '\n'.join(recommendations)


class MLTrainingResetConfirm(models.TransientModel):
    _name = 'ml.training.reset.confirm'
    _description = 'ML Training Reset Confirmation'
    
    ml_engine_id = fields.Many2one('ml.recommendation.engine', string="ML Engine", required=True)
    confirm_text = fields.Char(string="Type 'RESET' to confirm")
    keep_learning_data = fields.Boolean(string="Keep Learning Data", default=True)
    
    def action_confirm_reset(self):
        """Confirm and execute model reset"""
        self.ensure_one()
        
        if self.confirm_text != 'RESET':
            raise UserError("Please type 'RESET' to confirm model deletion")
        
        # Reset model
        self.ml_engine_id.write({
            'is_model_trained': False,
            'model_data': False,
            'scaler_data': False,
            'feature_names': False,
            'last_training_date': False,
            'training_samples': 0,
            'model_accuracy': 0.0,
            'r2_score': 0.0,
            'mse_score': 0.0,
            'cv_score': 0.0,
            'training_duration': 0.0
        })
        
        # Optionally clear learning data
        if not self.keep_learning_data:
            self.ml_engine_id.learning_data_ids.unlink()
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Model Reset Complete',
                'message': 'The ML model has been reset successfully.',
                'type': 'warning',
                'sticky': False,
            }
        }