from odoo import models, api, fields
from odoo.exceptions import UserError
import logging

_logger = logging.getLogger(__name__)

class IntegrationManager(models.Model):
    _name = 'integration.manager'
    _description = 'Central Integration Manager for All Engines'
    
   @api.model
    def generate_complete_composition(self, partner_id, target_budget, target_year=None, 
                                    dietary_restrictions=None, notes_text=None, use_batch=False,
                                    attempt_number=1):
        """
        Enhanced composition generation with attempt-based variation and smart budget targeting
        """
        
        if target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
        
        try:
            _logger.info(f"Generating composition (attempt {attempt_number}): partner={partner_id}, budget=€{target_budget}")
            
            # Use stock-aware engine with attempt-based variation
            stock_engine = self.env['stock.aware.composition.engine']
            composition = stock_engine.generate_compliant_composition(
                partner_id=partner_id,
                target_budget=target_budget,
                target_year=target_year,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text,
                attempt_number=attempt_number  # Pass attempt for variation
            )
            
            if composition:
                # Add enhanced AI analysis using existing Ollama integration
                if attempt_number == 1:  # Only on first attempt to avoid repetition
                    enhanced_reasoning = self._add_ai_enhancement(composition, notes_text)
                    if enhanced_reasoning:
                        composition.reasoning = enhanced_reasoning
                
                # Generate documents if not batch
                if not use_batch:
                    try:
                        doc_generator = self.env['document.generation.system']
                        documents = doc_generator.generate_all_documents(composition.id)
                        _logger.info(f"Generated {len(documents)} document types")
                    except Exception as e:
                        _logger.warning(f"Document generation failed: {str(e)}")
                
                return composition
                
        except Exception as e:
            _logger.error(f"Composition generation failed: {str(e)}")
            raise UserError(f"Composition generation failed: {str(e)}")

    def _add_ai_enhancement(self, composition, notes_text):
        """Add AI enhancement using existing Ollama integration"""
        
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