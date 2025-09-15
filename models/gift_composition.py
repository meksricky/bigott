# models/gift_composition.py

from odoo import models, fields, api
from odoo.exceptions import UserError
from datetime import datetime

class GiftComposition(models.Model):
    _name = 'gift.composition'
    _description = 'Gift Composition'
    _order = 'create_date desc'
    _rec_name = 'display_name'
    _inherit = ['mail.thread', 'mail.activity.mixin']
    
    # Basic Information
    name = fields.Char(string="Name", compute='_compute_name', store=True)
    display_name = fields.Char(string="Display Name", compute='_compute_display_name')
    partner_id = fields.Many2one('res.partner', string="Client", required=True, tracking=True, ondelete='cascade')
    
    # Composition Details
    composition_type = fields.Selection([
        ('custom', 'Custom Composition'),
        ('experience', 'Experience Based'),
        ('ai_generated', 'AI Generated')
    ], string="Composition Type", default='ai_generated', tracking=True)
    
    # Budget Information
    target_budget = fields.Float(string="Target Budget", required=True, tracking=True)
    actual_cost = fields.Float(string="Actual Cost", compute='_compute_actual_cost', store=True)
    budget_variance = fields.Float(string="Budget Variance %", compute='_compute_budget_variance', store=True)
    
    # Products
    product_ids = fields.Many2many('product.template', string="Products", tracking=True)
    product_count = fields.Integer(string="Product Count", compute='_compute_product_count', store=True)
    
    # Composition Metadata
    target_year = fields.Integer(string="Target Year", default=lambda self: datetime.now().year)
    dietary_restrictions = fields.Text(string="Dietary Restrictions")
    client_notes = fields.Text(string="Client Notes")
    
    # AI/Recommendation Data
    reasoning = fields.Html(string="Recommendation Reasoning")
    confidence_score = fields.Float(string="Confidence Score", default=0.0)
    novelty_score = fields.Float(string="Novelty Score", default=0.0)
    
    # Status
    state = fields.Selection([
        ('draft', 'Draft'),
        ('confirmed', 'Confirmed'),
        ('approved', 'Approved'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled')
    ], string="Status", default='draft', tracking=True)
    
    # Timestamps
    confirmation_date = fields.Datetime(string="Confirmation Date")
    delivery_date = fields.Datetime(string="Delivery Date")
    
    @api.depends('partner_id', 'target_year', 'composition_type')
    def _compute_name(self):
        for record in self:
            if record.partner_id:
                type_str = dict(self._fields['composition_type'].selection).get(record.composition_type, '')
                record.name = f"{record.partner_id.name} - {record.target_year} - {type_str}"
            else:
                record.name = f"Gift Composition - {record.target_year}"
    
    @api.depends('name')
    def _compute_display_name(self):
        for record in self:
            record.display_name = record.name or f"Gift Composition #{record.id}"
    
    @api.depends('product_ids')
    def _compute_product_count(self):
        for record in self:
            record.product_count = len(record.product_ids)
    
    @api.depends('product_ids.list_price')
    def _compute_actual_cost(self):
        for record in self:
            record.actual_cost = sum(record.product_ids.mapped('list_price'))
    
    @api.depends('actual_cost', 'target_budget')
    def _compute_budget_variance(self):
        for record in self:
            if record.target_budget > 0:
                variance = ((record.actual_cost - record.target_budget) / record.target_budget) * 100
                record.budget_variance = variance
            else:
                record.budget_variance = 0.0
    
    def action_regenerate(self):
        """Regenerate the recommendation"""
        self.ensure_one()
        
        # Get recommender
        recommender = self.env['ollama.gift.recommender'].get_or_create_recommender()
        
        # Generate new recommendation
        result = recommender.generate_gift_recommendations(
            partner_id=self.partner_id.id,
            target_budget=self.target_budget,
            client_notes=self.client_notes,
            dietary_restrictions=self.dietary_restrictions.split(',') if self.dietary_restrictions else []
        )
        
        if result.get('success'):
            # Update this composition with new products
            self.write({
                'product_ids': [(6, 0, [p.id for p in result.get('products', [])])],
                'confidence_score': result.get('confidence_score', 0.75),
                'reasoning': result.get('reasoning', 'Regenerated recommendation')
            })
            
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Success',
                    'message': 'Recommendation regenerated successfully',
                    'type': 'success',
                }
            }
        
        raise UserError('Failed to regenerate recommendation')
    
    def action_confirm(self):
        """Confirm the composition"""
        self.ensure_one()
        self.write({
            'state': 'confirmed',
            'confirmation_date': fields.Datetime.now()
        })
        self.message_post(body="Gift composition confirmed.")
        return True
    
    def action_approve(self):
        """Approve the composition"""
        self.ensure_one()
        self.write({'state': 'approved'})
        self.message_post(body="Gift composition approved.")
        return True
    
    def action_deliver(self):
        """Mark as delivered"""
        self.ensure_one()
        self.write({
            'state': 'delivered',
            'delivery_date': fields.Datetime.now()
        })
        self.message_post(body="Gift composition delivered.")
        return True
    
    def action_cancel(self):
        """Cancel the composition"""
        self.ensure_one()
        self.write({'state': 'cancelled'})
        self.message_post(body="Gift composition cancelled.")
        return True
    
    def action_reset_to_draft(self):
        """Reset to draft"""
        self.ensure_one()
        self.write({
            'state': 'draft',
            'confirmation_date': False,
            'delivery_date': False
        })
        self.message_post(body="Gift composition reset to draft.")
        return True