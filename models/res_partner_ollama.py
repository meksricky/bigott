# models/res_partner_ollama.py
from odoo import models, fields, api
from odoo.exceptions import UserError
import logging

_logger = logging.getLogger(__name__)

class ResPartner(models.Model):
    _inherit = 'res.partner'
    
    def action_generate_ollama_recommendation(self):
        """Direct action to generate recommendation without wizard"""
        self.ensure_one()
        
        if self.is_company:
            raise UserError("Please select an individual client, not a company.")
        
        # Get or create recommender
        recommender = self.env['ollama.gift.recommender'].search([('active', '=', True)], limit=1)
        if not recommender:
            recommender = self.env['ollama.gift.recommender'].create({
                'name': 'Default Ollama Recommender',
                'ollama_enabled': True
            })
        
        # Default values
        target_budget = 200.0  # Default budget
        dietary_restrictions = []
        
        # Check if partner has dietary restrictions stored
        if hasattr(self, 'dietary_restrictions') and self.dietary_restrictions:
            dietary_restrictions = [self.dietary_restrictions]
        
        _logger.info(f"Generating direct recommendation for {self.name}")
        
        try:
            # Generate recommendation directly
            result = recommender.generate_gift_recommendations(
                partner_id=self.id,
                target_budget=target_budget,
                client_notes='',
                dietary_restrictions=dietary_restrictions
            )
            
            if result.get('success'):
                composition_id = result.get('composition_id')
                
                # Show notification and open composition
                return {
                    'type': 'ir.actions.act_window',
                    'name': f'Gift Composition for {self.name}',
                    'res_model': 'gift.composition',
                    'res_id': composition_id,
                    'view_mode': 'form',
                    'target': 'current',
                }
            else:
                raise UserError(f"Generation failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            _logger.error(f"Direct recommendation failed: {str(e)}")
            raise UserError(f"Failed to generate recommendation: {str(e)}")