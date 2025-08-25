from odoo import models, fields, api

class ResPartner(models.Model):
    _inherit = 'res.partner'
    
    # Client-specific freeform notes to influence recommendations
    client_notes = fields.Text('Client Notes')

    # Client preferences and dietary restrictions
    dietary_restrictions = fields.Selection([
        ('none', 'None'),
        ('vegan', 'Vegan'),
        ('halal', 'Halal'),
        ('non_alcoholic', 'Non-Alcoholic'),
        ('vegan_halal', 'Vegan + Halal'),
        ('vegan_non_alcoholic', 'Vegan + Non-Alcoholic'),
        ('halal_non_alcoholic', 'Halal + Non-Alcoholic'),
        ('all_restrictions', 'Vegan + Halal + Non-Alcoholic')
    ], string='Dietary Restrictions', default='none')
    
    # Gift history relationships
    order_history_ids = fields.One2many('client.order.history', 'partner_id', 'Gift Order History')
    composition_ids = fields.One2many('gift.composition', 'partner_id', 'Generated Compositions')
    
    # Client insights and analytics
    years_as_client = fields.Integer('Years as Client', compute='_compute_client_insights', store=True)
    total_gift_budget = fields.Float('Total Gift Budget (All Years)', compute='_compute_client_insights', store=True)
    average_annual_budget = fields.Float('Average Annual Budget', compute='_compute_client_insights', store=True)
    budget_growth_rate = fields.Float('Budget Growth Rate %', compute='_compute_client_insights', store=True)
    
    preferred_box_type = fields.Selection([
        ('experience', 'Experience-Based'),
        ('custom', 'Custom Compositions')
    ], string='Preferred Box Type', compute='_compute_client_insights', store=True)
    
    last_order_year = fields.Integer('Last Order Year', compute='_compute_client_insights', store=True)
    client_satisfaction_avg = fields.Float('Average Satisfaction', compute='_compute_client_insights', store=True)
    
    # Business relationship
    client_tier = fields.Selection([
        ('new', 'New Client'),
        ('regular', 'Regular Client'),
        ('premium', 'Premium Client'),
        ('vip', 'VIP Client')
    ], string='Client Tier', compute='_compute_client_tier', store=True)
    
    @api.depends('order_history_ids')
    def _compute_client_insights(self):
        for partner in self:
            histories = partner.order_history_ids
            if histories:
                # Basic metrics
                years = list(set(histories.mapped('order_year')))
                partner.years_as_client = len(years)
                partner.total_gift_budget = sum(histories.mapped('total_budget'))
                partner.average_annual_budget = partner.total_gift_budget / len(years) if years else 0
                partner.last_order_year = max(years) if years else 0
                
                # Budget growth analysis
                if len(years) >= 2:
                    sorted_histories = histories.sorted('order_year')
                    first_budget = sorted_histories[0].total_budget
                    last_budget = sorted_histories[-1].total_budget
                    partner.budget_growth_rate = ((last_budget - first_budget) / first_budget * 100) if first_budget else 0
                else:
                    partner.budget_growth_rate = 0
                
                # Preferred box type
                experience_count = len(histories.filtered(lambda h: h.box_type == 'experience'))
                custom_count = len(histories.filtered(lambda h: h.box_type == 'custom'))
                partner.preferred_box_type = 'experience' if experience_count >= custom_count else 'custom'
                
                # Satisfaction analysis
                satisfactions = [float(h.client_satisfaction) for h in histories if h.client_satisfaction]
                partner.client_satisfaction_avg = sum(satisfactions) / len(satisfactions) if satisfactions else 0
            else:
                partner.years_as_client = 0
                partner.total_gift_budget = 0.0
                partner.average_annual_budget = 0.0
                partner.budget_growth_rate = 0.0
                partner.preferred_box_type = 'custom'
                partner.last_order_year = 0
                partner.client_satisfaction_avg = 0.0
    
    @api.depends('years_as_client', 'total_gift_budget', 'average_annual_budget')
    def _compute_client_tier(self):
        for partner in self:
            if partner.years_as_client == 0:
                partner.client_tier = 'new'
            elif partner.years_as_client >= 3 and partner.average_annual_budget >= 300:
                partner.client_tier = 'vip'
            elif partner.average_annual_budget >= 200:
                partner.client_tier = 'premium'
            elif partner.years_as_client >= 2:
                partner.client_tier = 'regular'
            else:
                partner.client_tier = 'new'
    
    def action_generate_composition(self):
        """Generate new composition for this client"""
        from datetime import datetime
        current_year = datetime.now().year
        return {
            'type': 'ir.actions.act_window',
            'name': f'🧠 Generate Composition for {self.name}',
            'res_model': 'composition.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': {
                'default_partner_id': self.id,
                'default_target_year': current_year,
                'default_target_budget': self.average_annual_budget * 1.1 if self.average_annual_budget else 200,
                'default_dietary_restrictions': self.dietary_restrictions,
                'default_additional_notes': self.client_notes or '',
            }
        }
    
    def action_view_order_history(self):
        """View client's order history"""
        return {
            'type': 'ir.actions.act_window',
            'name': f'Order History: {self.name}',
            'res_model': 'client.order.history',
            'view_mode': 'tree,form',
            'domain': [('partner_id', '=', self.id)],
            'context': {'default_partner_id': self.id}
        }
    
    def action_view_compositions(self):
        """View generated compositions for this client"""
        return {
            'type': 'ir.actions.act_window',
            'name': f'Compositions: {self.name}',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('partner_id', '=', self.id)],
            'context': {'default_partner_id': self.id}
        }

    @api.depends('order_history_ids')
    def _compute_budget_growth_rate(self):
        for partner in self:
            # Simple default calculation
            partner.budget_growth_rate = 5.0  # Default 5% growth