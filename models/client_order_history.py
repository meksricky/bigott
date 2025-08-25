from odoo import models, fields, api
from datetime import datetime, timedelta

class ClientOrderHistory(models.Model):
    _name = 'client.order.history'
    _description = 'Client Gift Order History for Señor Bigott'
    _order = 'order_year desc'
    _rec_name = 'display_name'
    
    display_name = fields.Char('Display Name', compute='_compute_display_name', store=True)
    partner_id = fields.Many2one('res.partner', 'Client', required=True, ondelete='cascade')
    order_year = fields.Integer('Order Year', required=True)
    total_budget = fields.Float('Total Budget (€)', required=True)
    
    # Box type
    box_type = fields.Selection([
        ('experience', 'Experience-Based Box'),
        ('custom', 'Custom Product Box')
    ], string='Box Type', required=True)
    
    # Experience tracking
    experience_id = fields.Many2one('gift.experience', 'Experience Used', ondelete='set null')
    
    # Products in the box
    product_ids = fields.Many2many('product.template', string='Products in Box')
    total_products = fields.Integer('Total Products', compute='_compute_totals', store=True)
    
    # Category analysis - Text field for Odoo 14 (store JSON string)
    category_breakdown = fields.Text('Category Breakdown')
    
    # Dietary considerations
    dietary_restrictions = fields.Text('Dietary Restrictions Applied')
    
    # Performance tracking
    client_satisfaction = fields.Selection([
        ('1', '⭐'),
        ('2', '⭐⭐'),
        ('3', '⭐⭐⭐'),
        ('4', '⭐⭐⭐⭐'),
        ('5', '⭐⭐⭐⭐⭐')
    ], string='Client Satisfaction')
    
    # Notes
    notes = fields.Text('Order Notes')
    
    # Computed fields
    budget_per_product = fields.Float('Budget per Product', compute='_compute_budget_metrics', store=True)
    
    @api.depends('partner_id', 'order_year', 'box_type')
    def _compute_display_name(self):
        for record in self:
            record.display_name = f"{record.partner_id.name} - {record.order_year} ({record.box_type})"
    
    @api.depends('product_ids')
    def _compute_totals(self):
        for record in self:
            record.total_products = len(record.product_ids)
    
    @api.depends('total_budget', 'total_products')
    def _compute_budget_metrics(self):
        for record in self:
            record.budget_per_product = record.total_budget / record.total_products if record.total_products else 0
    
    def get_category_structure(self):
        """Return category breakdown as dictionary"""
        import json
        if not self.category_breakdown:
            return {}
        try:
            return json.loads(self.category_breakdown)
        except Exception:
            return {}
    
    def set_category_structure(self, categories):
        """Set category structure from dictionary"""
        import json
        try:
            self.category_breakdown = json.dumps(categories or {})
        except Exception:
            self.category_breakdown = '{}'
    
    @api.model
    def analyze_client_patterns(self, partner_id):
        """Analyze client's historical patterns"""
        
        histories = self.search([
            ('partner_id', '=', partner_id)
        ], order='order_year desc', limit=3)
        
        if not histories:
            return {
                'has_history': False,
                'message': 'No historical data available for this client'
            }
        
        # Calculate patterns
        budgets = [h.total_budget for h in histories]
        avg_budget = sum(budgets) / len(budgets)
        avg_products = sum([h.total_products for h in histories]) / len(histories)
        
        # Experience usage
        used_experiences = [h.experience_id.id for h in histories if h.experience_id]
        
        # Most recent category structure
        latest_categories = histories[0].get_category_structure() if histories else {}
        
        # Budget trend analysis
        if len(budgets) >= 2:
            recent_change = (budgets[0] - budgets[1]) / budgets[1] * 100
            if recent_change > 10:
                budget_trend = 'increasing'
            elif recent_change < -10:
                budget_trend = 'decreasing'
            else:
                budget_trend = 'stable'
        else:
            budget_trend = 'unknown'
        
        # Satisfaction analysis
        satisfactions = [float(h.client_satisfaction) for h in histories if h.client_satisfaction]
        avg_satisfaction = sum(satisfactions) / len(satisfactions) if satisfactions else 0
        
        return {
            'has_history': True,
            'years_of_data': len(histories),
            'average_budget': avg_budget,
            'budget_trend': budget_trend,
            'recent_budgets': budgets,
            'average_products': avg_products,
            'average_satisfaction': avg_satisfaction,
            'used_experiences': used_experiences,
            'latest_category_structure': latest_categories,
            'box_type_preference': histories[0].box_type if histories else 'custom'
        }
    
    def action_generate_next_composition(self):
        """Generate composition for next year"""
        return {
            'type': 'ir.actions.act_window',
            'name': '🧠 Generate Next Year Composition',
            'res_model': 'composition.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': {
                'default_partner_id': self.partner_id.id,
                'default_target_year': datetime.now().year + 1,
                'default_target_budget': self.total_budget * 1.1,  # Suggest 10% increase
            }
        }