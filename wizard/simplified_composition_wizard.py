# -*- coding: utf-8 -*-
from odoo import api, fields, models, _
from odoo.exceptions import UserError
import json

class SimplifiedCompositionWizard(models.TransientModel):
    _name = "simplified.composition.wizard"
    _description = "Generate Gift Composition (Simplified)"

    partner_id = fields.Many2one('res.partner', string='Customer', required=True)
    target_year = fields.Integer(
        string='Target Year',
        default=lambda self: fields.Date.today().year,
        required=True
    )
    target_budget = fields.Float(string='Target Budget (€)', required=True)
    dietary_restrictions = fields.Selection([
        ('none', 'None'),
        ('halal', 'Halal'),
        ('no_alcohol', 'No Alcohol'),
        ('vegetarian', 'Vegetarian'),
    ], string='Dietary Restrictions', default='none')
    additional_notes = fields.Text(string='Additional Notes')
    
    # Display fields
    client_info = fields.Html('Client Information', compute='_compute_client_info')

    @api.depends('partner_id')
    def _compute_client_info(self):
        for wizard in self:
            if not wizard.partner_id:
                wizard.client_info = "<p>Select a client to see their information</p>"
                continue
            
            # Get client history
            history = self.env['client.order.history'].search([
                ('partner_id', '=', wizard.partner_id.id)
            ], order='order_year desc', limit=3)
            
            info_html = f"<h4>{wizard.partner_id.name}</h4>"
            
            if history:
                info_html += "<h5>Purchase History:</h5><ul>"
                for h in history:
                    info_html += f"<li><strong>{h.order_year}:</strong> €{h.total_budget:.2f} - {h.box_type} box</li>"
                info_html += "</ul>"
                
                # Show preferred categories
                try:
                    latest_categories = json.loads(history[0].category_breakdown or '{}')
                    if latest_categories:
                        info_html += "<h5>Recent Preferences:</h5><ul>"
                        for category, count in sorted(latest_categories.items(), key=lambda x: x[1], reverse=True)[:3]:
                            info_html += f"<li>{category}: {count} items</li>"
                        info_html += "</ul>"
                except:
                    pass
            else:
                info_html += "<p><em>New client - no purchase history</em></p>"
            
            wizard.client_info = info_html

    def action_generate_composition(self):
        """Generate composition with proper budget compliance (+/- 5%)"""
        self.ensure_one()
        
        if self.target_budget <= 0:
            raise UserError(_("Target budget must be greater than 0"))
        
        # Prepare dietary restrictions
        dietary_list = []
        if self.dietary_restrictions != 'none':
            dietary_list = [self.dietary_restrictions]
        
        try:
            # Try simplified engine first
            result = None
            
            if self.env['simplified.composition.engine'].search([]):
                engine = self.env['simplified.composition.engine']
                try:
                    result = engine.generate_composition(
                        partner_id=self.partner_id.id,
                        target_budget=self.target_budget,
                        target_year=self.target_year,
                        dietary_restrictions=dietary_list,
                        notes_text=self.additional_notes or ''
                    )
                except Exception as e:
                    raise UserError(_("Simplified engine failed: %s") % str(e))
            
            # Fallback to main composition engine
            if not result and self.env['composition.engine'].search([]):
                engine = self.env['composition.engine']
                try:
                    result = engine.generate_composition(
                        partner_id=self.partner_id.id,
                        target_budget=self.target_budget,
                        target_year=self.target_year,
                        dietary_restrictions=dietary_list,
                        force_type=None,
                        notes_text=self.additional_notes or ''
                    )
                except Exception as e:
                    raise UserError(_("Main engine failed: %s") % str(e))
            
            if not result:
                raise UserError(_("No composition engine available"))
            
            comp_id = result.get('composition_id')
            if not comp_id:
                raise UserError(_("Failed to generate composition"))
            
            # Verify budget compliance (+/- 5%)
            composition = self.env['gift.composition'].browse(comp_id)
            self._verify_budget_compliance(composition)
            
            return {
                'type': 'ir.actions.act_window',
                'name': _('Generated Composition'),
                'res_model': 'gift.composition',
                'view_mode': 'form',
                'target': 'current',
                'res_id': comp_id,
            }
            
        except Exception as e:
            raise UserError(_("Composition generation failed: %s") % str(e))
    
    def _verify_budget_compliance(self, composition):
        """Verify the composition is within +/- 5% of target budget"""
        actual_cost = composition.actual_cost
        target = self.target_budget
        
        if target > 0:
            variance_percent = abs(actual_cost - target) / target * 100
            
            if variance_percent > 5.0:
                # Log warning but don't fail - just inform user
                message = _("Budget variance is %.1f%% (€%.2f vs €%.2f target)") % (
                    variance_percent, actual_cost, target
                )
                composition.message_post(body=message, message_type='comment')
    
    def action_preview_products(self):
        """Preview available products for this configuration"""
        
        try:
            if not self.env['simplified.composition.engine'].search([]):
                raise UserError("Simplified composition engine not available")
                
            engine = self.env['simplified.composition.engine']
            dietary_list = [self.dietary_restrictions] if self.dietary_restrictions != 'none' else []
            available_products = engine._get_available_products(dietary_list)
            
            if not available_products:
                raise UserError("No products available with current criteria")
            
            # Show products in a tree view
            return {
                'type': 'ir.actions.act_window',
                'name': f'Available Products ({len(available_products)} found)',
                'res_model': 'product.template',
                'view_mode': 'tree',
                'domain': [('id', 'in', [p.id for p in available_products])],
                'target': 'new',
            }
            
        except Exception as e:
            raise UserError(f"Failed to preview products: {str(e)}")

    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Auto-suggest budget based on client history"""
        if self.partner_id and not self.target_budget:
            history = self.env['client.order.history'].search([
                ('partner_id', '=', self.partner_id.id)
            ], order='order_year desc', limit=1)
            
            if history:
                # Suggest similar budget with slight increase
                suggested_budget = history.total_budget * 1.05
                self.target_budget = suggested_budget