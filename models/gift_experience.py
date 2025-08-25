from odoo import models, fields, api

class GiftExperience(models.Model):
    _name = 'gift.experience'
    _description = 'Curated Gift Experiences (3-5 products with instructions)'
    _rec_name = 'name'

    name = fields.Char('Experience Name', required=True)
    description = fields.Text('Experience Description')
    instructions = fields.Html('Preparation Instructions')

    product_ids = fields.Many2many('product.template', string='Products in Experience')
    product_count = fields.Integer('Product Count', compute='_compute_product_count', store=True)

    experience_theme = fields.Selection([
        ('mediterranean', 'Mediterranean'),
        ('spanish_classics', 'Spanish Classics'),
        ('premium_selection', 'Premium Selection'),
        ('seasonal_special', 'Seasonal Special'),
        ('wine_pairing', 'Wine Pairing'),
        ('cheese_journey', 'Cheese Journey'),
        ('chocolate_discovery', 'Chocolate Discovery'),
        ('regional_specialties', 'Regional Specialties')
    ], string='Experience Theme', required=True)

    is_vegan_friendly = fields.Boolean('Vegan Friendly')
    is_halal_friendly = fields.Boolean('Halal Friendly')
    is_alcohol_free = fields.Boolean('Alcohol Free')

    base_cost = fields.Float('Base Cost (€)', compute='_compute_base_cost', store=True)
    recommended_budget = fields.Float('Recommended Budget (€)')
    min_budget = fields.Float('Minimum Budget (€)', compute='_compute_budget_range', store=True)
    max_budget = fields.Float('Maximum Budget (€)', compute='_compute_budget_range', store=True)

    times_used = fields.Integer('Times Used', compute='_compute_usage', store=True)
    client_usage_ids = fields.One2many('client.order.history', 'experience_id', 'Client Usage')
    last_used_date = fields.Date('Last Used', compute='_compute_usage', store=True)

    average_satisfaction = fields.Float('Average Satisfaction', compute='_compute_satisfaction', store=True)

    active = fields.Boolean('Active', default=True)
    is_premium = fields.Boolean('Premium Experience')

    @api.depends('product_ids')
    def _compute_product_count(self):
        for experience in self:
            experience.product_count = len(experience.product_ids)

    @api.depends('product_ids')
    def _compute_base_cost(self):
        for experience in self:
            experience.base_cost = sum(experience.product_ids.mapped('list_price'))

    @api.depends('base_cost')
    def _compute_budget_range(self):
        for experience in self:
            experience.min_budget = experience.base_cost * 0.9
            experience.max_budget = experience.base_cost * 1.2

    @api.depends('client_usage_ids')
    def _compute_usage(self):
        for experience in self:
            usage = experience.client_usage_ids
            experience.times_used = len(usage)
            experience.last_used_date = max(usage.mapped('order_year')) if usage else False

    @api.depends('client_usage_ids.client_satisfaction')
    def _compute_satisfaction(self):
        for experience in self:
            satisfactions = [
                float(usage.client_satisfaction) for usage in experience.client_usage_ids
                if usage.client_satisfaction
            ]
            experience.average_satisfaction = (
                sum(satisfactions) / len(satisfactions) if satisfactions else 0
            )

    def check_dietary_compatibility(self, dietary_restrictions):
        if 'vegan' in dietary_restrictions and not self.is_vegan_friendly:
            return False
        if 'halal' in dietary_restrictions and not self.is_halal_friendly:
            return False
        if 'non_alcoholic' in dietary_restrictions and not self.is_alcohol_free:
            return False
        return True

    def is_available_for_client(self, partner_id):
        previous_use = self.env['client.order.history'].search([
            ('partner_id', '=', partner_id),
            ('experience_id', '=', self.id)
        ])
        return len(previous_use) == 0

    def action_view_usage_history(self):
        return {
            'type': 'ir.actions.act_window',
            'name': f'Usage History: {self.name}',
            'res_model': 'client.order.history',
            'view_mode': 'tree,form',
            'domain': [('experience_id', '=', self.id)],
            'context': {'default_experience_id': self.id}
        }

    def action_view_compositions(self):
        """Return list of gift.composition records using this experience"""
        return {
            'type': 'ir.actions.act_window',
            'name': f'Compositions Using: {self.name}',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('experience_id', '=', self.id)],
            'context': {'default_experience_id': self.id}
        }
