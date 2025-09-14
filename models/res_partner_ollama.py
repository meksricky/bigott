# models/res_partner_ollama.py
from odoo import models, fields, api
from odoo.exceptions import UserError
import logging

_logger = logging.getLogger(__name__)

class ResPartner(models.Model):
    _inherit = 'res.partner'
    
    def action_generate_ollama_recommendation(self):
        """Direct action to generate recommendation for ANY client"""
        self.ensure_one()
        
        # NO restriction on company vs individual
        _logger.info(f"Generating recommendation for {self.name} (Company: {self.is_company})")
        
        # Get or create recommender
        recommender = self.env['ollama.gift.recommender'].sudo().search([('active', '=', True)], limit=1)
        if not recommender:
            recommender = self.env['ollama.gift.recommender'].sudo().create({
                'name': 'Default Ollama Recommender',
                'ollama_enabled': False  # Start with fallback mode
            })
        
        # Default values
        target_budget = 200.0
        dietary_restrictions = []
        
        try:
            # Use sudo() to bypass permission issues
            result = recommender.sudo().generate_gift_recommendations(
                partner_id=self.id,
                target_budget=target_budget,
                client_notes=f"Client type: {'Company' if self.is_company else 'Individual'}",
                dietary_restrictions=dietary_restrictions
            )
            
            if result.get('success'):
                composition_id = result.get('composition_id')
                
                # Show success notification
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': 'Success!',
                        'message': f'Recommendation generated for {self.name}',
                        'type': 'success',
                        'sticky': False,
                    }
                }
            else:
                raise UserError(f"Generation failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            _logger.error(f"Recommendation failed: {str(e)}")
            raise UserError(f"Failed to generate recommendation: {str(e)}")