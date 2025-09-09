# models/wizard_models.py
from odoo import models, fields, api
from odoo.exceptions import UserError

class BatchWizard(models.TransientModel):
    _name = 'batch.wizard'
    _description = 'Batch Processing Wizard'
    
    target_year = fields.Integer('Target Year', required=True, 
                                default=lambda self: fields.Date.today().year)
    
    client_selection = fields.Selection([
        ('all_eligible', 'All Eligible Clients'),
        ('by_tier', 'By Client Tier'), 
        ('manual', 'Manual Selection')
    ], string='Client Selection', default='all_eligible', required=True)
    
    def action_start_batch(self):
        """Start batch processing"""
        batch_processor = self.env['batch.composition.processor'].create({
            'target_year': self.target_year,
            'client_selection_mode': self.client_selection,
        })
        batch_processor.action_start_processing()
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Batch Processing',
            'res_model': 'batch.composition.processor',
            'res_id': batch_processor.id,
            'view_mode': 'form',
            'target': 'current',
        }


class ExperienceRetirementWizard(models.TransientModel):
    _name = 'experience.retirement.wizard' 
    _description = 'Experience Retirement Wizard'
    
    experience_id = fields.Many2one('gift.experience', 'Experience', required=True)
    retirement_reason = fields.Text('Retirement Reason', required=True)
    retirement_date = fields.Date('Retirement Date', default=fields.Date.today)
    
    def action_retire(self):
        """Retire the experience"""
        self.experience_id.write({
            'lifecycle_stage': 'retired',
            'retirement_reason': self.retirement_reason,
            'retirement_date': self.retirement_date,
            'active': False
        })
        return {'type': 'ir.actions.act_window_close'}

