from odoo import models, fields, api
from odoo.exceptions import UserError

class SimpleRecommendationWizard(models.TransientModel):
    _name = 'simple.recommendation.wizard'
    _description = 'Simple Gift Recommendation Wizard'
    
    partner_id = fields.Many2one('res.partner', string="Client")
    target_budget = fields.Float(string="Budget", default=100.0)
    
    def action_generate(self):
        if not self.partner_id:
            raise UserError("Please select a client.")
        
        # Just create a simple notification for now
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Success!',
                'message': f'Selected client: {self.partner_id.name}, Budget: {self.target_budget}',
                'type': 'success'
            }
        }