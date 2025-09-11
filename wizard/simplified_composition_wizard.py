from odoo import models, fields, api
from odoo.exceptions import UserError

class SimplifiedCompositionWizard(models.TransientModel):
    _name = 'simplified.composition.wizard'
    _description = 'Simplified Gift Composition Wizard'
    
    partner_id = fields.Many2one('res.partner', 'Client', required=True)
    target_year = fields.Integer('Target Year', required=True, default=lambda self: fields.Date.context_today(self).year)
    target_budget = fields.Float('Target Budget (€)', required=True, default=200.0)
    
    dietary_restrictions = fields.Selection([
        ('none', 'None'),
        ('vegan', 'Vegan'),
        ('halal', 'Halal'), 
        ('non_alcoholic', 'Non-Alcoholic'),
    ], string='Dietary Restrictions', default='none')
    
    additional_notes = fields.Text('Additional Notes')
    
    def action_generate_composition(self):
        """Generate composition using simplified engine"""
        
        if self.target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
        
        engine = self.env['simplified.composition.engine']
        
        result = engine.generate_composition(
            partner_id=self.partner_id.id,
            target_budget=self.target_budget,
            target_year=self.target_year,
            dietary_restrictions=self.dietary_restrictions,
            notes_text=self.additional_notes
        )
        
        # Create gift composition record
        composition = self.env['gift.composition'].create({
            'partner_id': self.partner_id.id,
            'target_year': self.target_year,
            'target_budget': self.target_budget,
            'actual_cost': result['total_cost'],
            'product_ids': [(6, 0, [p.id for p in result['products']])],
            'dietary_restrictions': self.additional_notes,
            'generation_method': result['method_used'],
            'state': 'draft'
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Generated Composition',
            'res_model': 'gift.composition',
            'res_id': composition.id,
            'view_mode': 'form',
            'target': 'current'
        }