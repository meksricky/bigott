# models/recommendation_learning.py
from odoo import models, fields, api
import json
import logging
from datetime import datetime

_logger = logging.getLogger(__name__)

class RecommendationLearning(models.Model):
    _name = 'recommendation.learning'
    _description = 'ML Learning from Sales Patterns'
    
    # Pattern Recognition
    partner_id = fields.Many2one('res.partner', 'Client')
    pattern_type = fields.Selection([
        ('client_specific', 'Client Specific'),
        ('segment', 'Client Segment'),
        ('general', 'General Pattern')
    ])
    
    # Learned Preferences
    preferred_categories = fields.Text('Preferred Categories (JSON)')
    preferred_price_range = fields.Char('Preferred Price Range')
    optimal_product_count = fields.Integer('Optimal Product Count')
    
    # Success Metrics
    success_rate = fields.Float('Success Rate')
    avg_satisfaction = fields.Float('Average Satisfaction')
    
    # Pattern Data
    pattern_data = fields.Text('Pattern Data (JSON)')
    last_updated = fields.Datetime('Last Updated')
    
    # Link to source sales
    source_sale_ids = fields.Many2many('sale.order', string='Source Sales Orders')
    source_composition_ids = fields.Many2many('gift.composition', string='Source Compositions')
    learning_notes = fields.Text('Learning Notes')
    
    @api.model
    def learn_from_sales(self):
        """Analyze sales and update learning patterns"""
        
        # Track which sales we're learning from
        sale_orders = self.env['sale.order'].search([
            ('state', 'in', ['sale', 'done']),
            ('amount_total', '>', 0),
            ('date_order', '>=', fields.Datetime.now() - timedelta(days=365))
        ])
        
        _logger.info(f"Learning from {len(sale_orders)} sales orders")
        
        patterns = {
            'category_frequency': {},
            'price_patterns': {},
            'product_combinations': [],
            'budget_patterns': {},
            'source_sales': []
        }
        
        for order in sale_orders:
            # Record this sale as a source
            patterns['source_sales'].append({
                'sale_id': order.id,
                'sale_name': order.name,
                'partner': order.partner_id.name,
                'date': str(order.date_order),
                'total': order.amount_total
            })
            
            # Analyze product patterns
            for line in order.order_line:
                if line.product_id:
                    product = line.product_id.product_tmpl_id
                    cat = getattr(product, 'lebiggot_category', 'general')
                    patterns['category_frequency'][cat] = patterns['category_frequency'].get(cat, 0) + 1
        
        # Store learned patterns with source tracking
        self._store_patterns_with_sources(patterns, sale_orders)
        
        return True

    @api.model
    def manual_trigger_learning(self):
        """Manually trigger learning - for testing"""
        _logger.info("=== MANUAL LEARNING TRIGGERED ===")
        
        # Check what data we have
        sales = self.env['sale.order'].search_count([('state', 'in', ['sale', 'done'])])
        compositions = self.env['gift.composition'].search_count([])
        
        _logger.info(f"Found {sales} sales orders and {compositions} compositions")
        
        if compositions > 0:
            # Learn from compositions directly
            self._learn_from_compositions()
        
        if sales > 0:
            # Learn from sales
            self.learn_from_sales()
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Learning Complete',
                'message': f'Analyzed {sales} sales and {compositions} compositions',
                'type': 'success',
            }
        }

    def _learn_from_compositions(self):
        """Learn directly from gift compositions"""
        compositions = self.env['gift.composition'].search([])
        
        patterns = {
            'products_used': {},
            'budget_patterns': {},
            'successful_combinations': []
        }
        
        for comp in compositions:
            # Track which products are used
            for product in comp.product_ids:
                patterns['products_used'][product.id] = {
                    'name': product.name,
                    'price': product.list_price,
                    'usage_count': patterns['products_used'].get(product.id, {}).get('usage_count', 0) + 1
                }
            
            # Track budget patterns
            budget_key = f"{int(comp.target_budget/50)*50}-{int(comp.target_budget/50)*50+50}"
            if budget_key not in patterns['budget_patterns']:
                patterns['budget_patterns'][budget_key] = []
            
            patterns['budget_patterns'][budget_key].append({
                'product_count': len(comp.product_ids),
                'actual_cost': comp.actual_cost,
                'variance': comp.budget_variance
            })
        
        # Store patterns
        learning = self.search([('pattern_type', '=', 'general')], limit=1)
        if not learning:
            learning = self.create({'pattern_type': 'general'})
        
        learning.write({
            'pattern_data': json.dumps(patterns, default=str),
            'last_updated': fields.Datetime.now(),
            'learning_notes': f"Learned from {len(compositions)} compositions"
        })
        
        _logger.info(f"Stored patterns from {len(compositions)} compositions")
    
    def _store_patterns_with_sources(self, patterns, source_orders):
        """Store patterns and track sources"""
        learning = self.search([('pattern_type', '=', 'general')], limit=1)
        if not learning:
            learning = self.create({'pattern_type': 'general'})
        
        learning.write({
            'pattern_data': json.dumps(patterns),
            'source_sale_ids': [(6, 0, source_orders.ids)],
            'last_updated': fields.Datetime.now(),
            'learning_notes': f"Learned from {len(source_orders)} sales on {fields.Date.today()}"
        })