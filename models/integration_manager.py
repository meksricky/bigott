from odoo import models, api, fields
from odoo.exceptions import UserError
import logging

_logger = logging.getLogger(__name__)

class IntegrationManager(models.Model):
    _name = 'integration.manager'
    _description = 'Central Integration Manager for All Engines'
    
    @api.model
    def generate_complete_composition(self, partner_id, target_budget, target_year=None, 
                                    dietary_restrictions=None, notes_text=None, use_batch=False):
        """
        Complete AI/ML-powered composition generation
        """
        
        if target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
        
        try:
            # Try ML approach first
            ml_engine = self.env['ml.recommendation.engine']
            
            # Check if ML models are trained
            ml_model = ml_engine.search([], limit=1)
            if ml_model and ml_model.is_model_trained:
                _logger.info("Using ML-powered recommendation engine")
                
                ml_result = ml_engine.get_smart_recommendations(
                    partner_id=partner_id,
                    target_budget=target_budget,
                    dietary_restrictions=dietary_restrictions,
                    notes_text=notes_text
                )
                
                # Create composition from ML result
                composition = self._create_composition_from_ml_result(
                    partner_id, target_budget, target_year, ml_result, 
                    dietary_restrictions, notes_text
                )
                
            else:
                _logger.info("ML models not trained, using rule-based engine")
                
                # Fallback to rule-based engine
                stock_engine = self.env['stock.aware.composition.engine']
                composition = stock_engine.generate_compliant_composition(
                    partner_id=partner_id,
                    target_budget=target_budget,
                    target_year=target_year,
                    dietary_restrictions=dietary_restrictions,
                    notes_text=notes_text
                )
            
            # Generate documents if not batch
            if composition and not use_batch:
                try:
                    doc_generator = self.env['document.generation.system']
                    documents = doc_generator.generate_all_documents(composition.id)
                    _logger.info(f"Generated {len(documents)} document types")
                except Exception as e:
                    _logger.warning(f"Document generation failed: {str(e)}")
            
            return composition
            
        except Exception as e:
            _logger.error(f"Complete composition generation failed: {str(e)}")
            raise UserError(f"Composition generation failed: {str(e)}")
    
    def _create_composition_from_ml_result(self, partner_id, target_budget, target_year, 
                                         ml_result, dietary_restrictions, notes_text):
        """Create composition from ML recommendation result"""
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner_id,
            'target_year': target_year or fields.Date.today().year,
            'target_budget': target_budget,
            'composition_type': 'custom',
            'product_ids': [(6, 0, [p.id for p in ml_result['products']])],
            'dietary_restrictions': ','.join(dietary_restrictions or []),
            'reasoning': ml_result.get('reasoning', 'ML-powered recommendation'),
            'confidence_score': ml_result.get('ml_confidence', 0.85),
            'novelty_score': 0.8,
            'historical_compatibility': 0.9,
            'notes': notes_text,
            'state': 'draft'
        })
        
        # Set category breakdown
        categories = {}
        for product in ml_result['products']:
            category = product.lebiggot_category
            categories[category] = categories.get(category, 0) + 1
        
        composition.set_category_breakdown(categories)
        
        return composition