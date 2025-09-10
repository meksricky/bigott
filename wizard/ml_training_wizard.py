from odoo import models, fields, api
from odoo.exceptions import UserError
import logging

_logger = logging.getLogger(__name__)

class MLTrainingWizard(models.TransientModel):
    _name = 'ml.training.wizard'
    _description = 'ML Model Training Wizard'
    
    training_type = fields.Selection([
        ('full_retrain', 'Full Retrain (Recommended)'),
        ('incremental', 'Incremental Training'),
        ('force_retrain', 'Force Complete Retrain')
    ], string='Training Type', default='full_retrain', required=True)
    
    min_training_samples = fields.Integer('Minimum Training Samples', default=50,
                                        help="Minimum number of sales orders required for training")
    
    # Display current status
    current_model_status = fields.Char('Current Model Status', compute='_compute_model_status')
    last_training_date = fields.Datetime('Last Training', compute='_compute_model_status')
    training_samples_count = fields.Integer('Current Training Samples', compute='_compute_model_status')
    model_accuracy = fields.Float('Model Accuracy (%)', compute='_compute_model_status')
    
    # Training options
    include_recent_data = fields.Boolean('Include Recent Data Only', default=False,
                                       help="Only use data from last 6 months")
    test_after_training = fields.Boolean('Run Test After Training', default=True)
    backup_current_model = fields.Boolean('Backup Current Model', default=True)
    
    @api.depends('training_type')
    def _compute_model_status(self):
        """Get current ML model status"""
        for wizard in self:
            try:
                ml_engine = self.env['ml.recommendation.engine'].search([], limit=1)
                if not ml_engine:
                    ml_engine = self.env['ml.recommendation.engine'].create({})
                
                wizard.current_model_status = "Trained" if ml_engine.is_model_trained else "Not Trained"
                wizard.last_training_date = ml_engine.last_training_date
                wizard.training_samples_count = ml_engine.training_samples or 0
                wizard.model_accuracy = ml_engine.model_accuracy or 0.0
                
            except Exception as e:
                wizard.current_model_status = f"Error: {str(e)}"
                wizard.last_training_date = False
                wizard.training_samples_count = 0
                wizard.model_accuracy = 0.0
    
    def action_check_training_data(self):
        """Check available training data before training"""
        
        # Count available sales orders for training
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=365*3)  # 3 years
        if self.include_recent_data:
            cutoff_date = datetime.now() - timedelta(days=180)  # 6 months
        
        sales_count = self.env['sale.order'].search_count([
            ('state', 'in', ['sale', 'done']),
            ('confirmation_date', '>=', cutoff_date),
            ('partner_id.is_company', '=', False)
        ])
        
        if sales_count < self.min_training_samples:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': '⚠️ Insufficient Training Data',
                    'message': f'Found {sales_count} sales orders, need minimum {self.min_training_samples}. Consider reducing minimum or importing more historical data.',
                    'type': 'warning',
                    'sticky': True,
                }
            }
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': '✅ Training Data Ready',
                'message': f'Found {sales_count} sales orders available for training. Ready to proceed.',
                'type': 'success',
                'sticky': False,
            }
        }
    
    def action_start_training(self):
        """Start ML model training"""
        
        try:
            # Validate training data first
            check_result = self.action_check_training_data()
            if check_result.get('params', {}).get('type') == 'warning':
                return check_result
            
            # Get or create ML engine
            ml_engine = self.env['ml.recommendation.engine'].search([], limit=1)
            if not ml_engine:
                ml_engine = self.env['ml.recommendation.engine'].create({})
            
            # Start training based on type
            force_retrain = self.training_type == 'force_retrain'
            
            _logger.info(f"Starting {self.training_type} ML training...")
            
            # Show progress notification
            self.env['bus.bus']._sendone(
                f'ml_training_{self.env.user.id}',
                {
                    'type': 'ml_training_started',
                    'message': 'ML training started. This may take several minutes...'
                }
            )
            
            # Perform training
            success = ml_engine.train_models_from_sales_data(force_retrain=force_retrain)
            
            if success:
                # Run test if requested
                if self.test_after_training:
                    test_results = self._run_training_test(ml_engine)
                    test_message = f"Test results: {test_results['success_rate']:.1f}% success rate"
                else:
                    test_message = "Training completed successfully"
                
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': '🎉 ML Training Completed',
                        'message': f'{test_message}. Model ready for recommendations.',
                        'type': 'success',
                        'sticky': True,
                    }
                }
            else:
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': '❌ ML Training Failed',
                        'message': 'Training failed. Check logs for details.',
                        'type': 'danger',
                        'sticky': True,
                    }
                }
                
        except Exception as e:
            _logger.error(f"ML training wizard failed: {str(e)}")
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': '❌ Training Error',
                    'message': f'Training failed: {str(e)}',
                    'type': 'danger',
                    'sticky': True,
                }
            }
    
    def _run_training_test(self, ml_engine):
        """Run test recommendations to validate training"""
        
        test_results = {'success_rate': 0, 'tested': 0, 'successful': 0}
        
        try:
            # Get 5 random clients with history
            test_partners = self.env['res.partner'].search([
                ('is_company', '=', False),
                ('order_history_ids', '!=', False)
            ], limit=5)
            
            successful_tests = 0
            
            for partner in test_partners:
                try:
                    # Try to generate recommendation
                    result = ml_engine.get_smart_recommendations(
                        partner_id=partner.id,
                        target_budget=200.0,  # Standard test budget
                        max_attempts=1
                    )
                    
                    if result and result.get('products'):
                        successful_tests += 1
                        
                except Exception:
                    continue
            
            test_results = {
                'success_rate': (successful_tests / len(test_partners) * 100) if test_partners else 0,
                'tested': len(test_partners),
                'successful': successful_tests
            }
            
        except Exception as e:
            _logger.warning(f"Training test failed: {e}")
        
        return test_results

    def action_view_model_status(self):
        """View detailed ML model status"""
        
        ml_engine = self.env['ml.recommendation.engine'].search([], limit=1)
        if not ml_engine:
            ml_engine = self.env['ml.recommendation.engine'].create({})
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'ML Model Status',
            'res_model': 'ml.recommendation.engine',
            'res_id': ml_engine.id,
            'view_mode': 'form',
            'target': 'new',
        }