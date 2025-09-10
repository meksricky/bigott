@api.model
def run_weekly_ml_training(self):
    """Cron job for weekly ML model training"""
    ml_engine = self.env['ml.recommendation.engine']
    
    # Train models from accumulated sales data
    success = ml_engine.train_models_from_sales_data()
    
    if success:
        _logger.info("Weekly ML training completed successfully")
        
        # Send notification to admin
        self.env['mail.mail'].create({
            'subject': '🤖 ML Models Retrained Successfully',
            'body_html': f"""
                <h3>Weekly ML Training Report</h3>
                <p>✅ Models retrained with latest sales data</p>
                <p>📊 Training samples: {ml_engine.training_samples}</p>
                <p>🎯 Model accuracy: {ml_engine.model_accuracy:.1f}%</p>
                <p>📅 Next training: {fields.Date.today() + timedelta(days=7)}</p>
            """,
            'email_to': 'admin@lebiggot.com'
        }).send()