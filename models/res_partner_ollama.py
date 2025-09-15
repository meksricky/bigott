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
        
        _logger.info(f"Generating recommendation for {self.name} (Company: {self.is_company})")
        
        # Get or create recommender
        recommender = self.env['ollama.gift.recommender'].sudo().search([('active', '=', True)], limit=1)
        if not recommender:
            recommender = self.env['ollama.gift.recommender'].sudo().create({
                'name': 'Default Ollama Recommender',
                'ollama_enabled': False
            })
        
        # Default values
        target_budget = 200.0
        dietary_restrictions = []
        
        try:
            # Generate recommendation
            result = recommender.sudo().generate_gift_recommendations(
                partner_id=self.id,
                target_budget=target_budget,
                client_notes=f"Client type: {'Company' if self.is_company else 'Individual'}",
                dietary_restrictions=dietary_restrictions
            )
            
            if result.get('success'):
                composition_id = result.get('composition_id')
                
                # CHANGED: Open the composition directly instead of just showing notification
                return {
                    'type': 'ir.actions.act_window',
                    'name': f'Gift Composition for {self.name}',
                    'res_model': 'gift.composition',
                    'res_id': composition_id,
                    'view_mode': 'form',
                    'target': 'current',  # Opens in main window
                }
            else:
                raise UserError(f"Generation failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            _logger.error(f"Recommendation failed: {str(e)}")
            raise UserError(f"Failed to generate recommendation: {str(e)}")

    def action_open_recommendation_wizard(self):
        """Open wizard with partner pre-selected"""
        self.ensure_one()
        
        return {
            'type': 'ir.actions.act_window',
            'name': f'Generate Recommendation for {self.name}',
            'res_model': 'ollama.recommendation.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': {
                'default_partner_id': self.id,
                'active_model': 'res.partner',
                'active_id': self.id,
            }
        }