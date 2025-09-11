# -*- coding: utf-8 -*-
from odoo import api, fields, models, _
from odoo.exceptions import UserError

class SimplifiedCompositionWizard(models.TransientModel):
    _name = "simplified.composition.wizard"
    _description = "Generate Gift Composition (Simplified)"

    partner_id = fields.Many2one('res.partner', string='Customer', required=True)
    target_year = fields.Integer(
        string='Target Year',
        default=lambda self: fields.Date.today().year
    )
    target_budget = fields.Float(string='Target Budget')
    dietary_restrictions = fields.Selection([
        ('none', 'None'),
        ('halal', 'Halal'),
        ('no_alcohol', 'No Alcohol'),
        ('vegetarian', 'Vegetarian'),
    ], string='Dietary Restrictions', default='none')
    additional_notes = fields.Text(string='Additional Notes')

    def action_generate_composition(self):
        self.ensure_one()
        # Try either engine name (adapt to your codebase)
        Engine = self.env.get('composition.engine') or self.env.get('simplified.composition.engine')
        if not Engine:
            raise UserError(_("Composition engine model not found."))

        # Adapt parameter names to your engine’s signature if needed
        result = Engine.generate_composition(
            partner_id=self.partner_id.id,
            target_budget=self.target_budget or 0.0,
            target_year=self.target_year,
            dietary_restrictions=self.dietary_restrictions,
            force_type=None,
            notes_text=(self.additional_notes or '')
        )

        comp_id = (result or {}).get('composition_id')
        if not comp_id:
            # Fallback: last composition for this partner
            comp = self.env['gift.composition'].search(
                [('partner_id', '=', self.partner_id.id)],
                order='id desc', limit=1
            )
            comp_id = comp.id if comp else False

        if not comp_id:
            raise UserError(_("The engine returned no composition. Check rules/stock/budget logs."))

        return {
            'type': 'ir.actions.act_window',
            'name': _('Generated Composition'),
            'res_model': 'gift.composition',
            'view_mode': 'form',
            'target': 'current',
            'res_id': comp_id,
        }
