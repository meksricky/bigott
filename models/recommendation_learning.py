# models/recommendation_learning.py
from odoo import models, fields, api
import json
import logging
from datetime import datetime, timedelta

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
        """Analyze sales and store actionable patterns"""
        
        # Get successful sales
        sale_orders = self.env['sale.order'].search([
            ('state', 'in', ['sale', 'done']),
            ('amount_total', '>', 0)
        ], limit=1000)  # Limit for performance
        
        # Get successful compositions
        compositions = self.env['gift.composition'].search([
            ('state', 'in', ['confirmed', 'approved', 'delivered']),
            ('actual_cost', '>', 0)
        ])
        
        _logger.info(f"Learning from {len(sale_orders)} sales and {len(compositions)} compositions")
        
        patterns = {
            'products_used': {},
            'budget_patterns': {},
            'category_frequency': {},
            'successful_combinations': [],
            'source_sales': [],
            'source_compositions': [],
            'stats': {
                'total_sales_analyzed': len(sale_orders),
                'total_compositions_analyzed': len(compositions),
                'date_analyzed': str(fields.Datetime.now())
            }
        }
        
        # Analyze compositions (more relevant for gift recommendations)
        for comp in compositions:
            # Track source
            patterns['source_compositions'].append({
                'id': comp.id,
                'name': comp.display_name,
                'partner': comp.partner_id.name,
                'budget': comp.target_budget,
                'actual_cost': comp.actual_cost,
                'product_count': len(comp.product_ids),
                'products': [p.name for p in comp.product_ids]
            })
            
            # Analyze budget patterns
            budget_range = self._get_budget_range(comp.target_budget)
            if budget_range not in patterns['budget_patterns']:
                patterns['budget_patterns'][budget_range] = {
                    'count': 0,
                    'avg_products': 0,
                    'avg_variance': 0,
                    'product_counts': []
                }
            
            patterns['budget_patterns'][budget_range]['count'] += 1
            patterns['budget_patterns'][budget_range]['product_counts'].append(len(comp.product_ids))
            
            # Track product usage
            for product in comp.product_ids:
                if product.id not in patterns['products_used']:
                    patterns['products_used'][product.id] = {
                        'name': product.name,
                        'price': product.list_price,
                        'usage_count': 0,
                        'in_successful_orders': []
                    }
                patterns['products_used'][product.id]['usage_count'] += 1
                patterns['products_used'][product.id]['in_successful_orders'].append(comp.display_name)
        
        # Calculate averages for budget patterns
        for budget_range, data in patterns['budget_patterns'].items():
            if data['product_counts']:
                data['avg_products'] = int(sum(data['product_counts']) / len(data['product_counts']))
                data['optimal_product_count'] = data['avg_products']
        
        # Analyze sales orders
        for order in sale_orders[:500]:  # Limit to first 500
            patterns['source_sales'].append({
                'id': order.id,
                'name': order.name,
                'partner': order.partner_id.name,
                'date': str(order.date_order),
                'total': order.amount_total
            })
            
            # Track products from sales
            for line in order.order_line:
                if line.product_id and line.product_id.product_tmpl_id:
                    product = line.product_id.product_tmpl_id
                    if product.id not in patterns['products_used']:
                        patterns['products_used'][product.id] = {
                            'name': product.name,
                            'price': product.list_price,
                            'usage_count': 0,
                            'in_successful_orders': []
                        }
                    patterns['products_used'][product.id]['usage_count'] += 1
        
        # Find most successful product combinations
        for comp in compositions[:20]:  # Top 20 compositions
            if comp.budget_variance > -10:  # Good budget match
                patterns['successful_combinations'].append({
                    'products': [p.name for p in comp.product_ids],
                    'budget': comp.target_budget,
                    'variance': comp.budget_variance,
                    'confidence': comp.confidence_score
                })
        
        # Store patterns
        learning = self.search([('pattern_type', '=', 'general')], limit=1)
        if not learning:
            learning = self.create({'pattern_type': 'general'})
        
        # Calculate optimal product count from patterns
        all_product_counts = []
        for data in patterns['budget_patterns'].values():
            all_product_counts.extend(data['product_counts'])
        
        optimal_count = int(sum(all_product_counts) / len(all_product_counts)) if all_product_counts else 4
        
        learning.write({
            'pattern_data': json.dumps(patterns, default=str),
            'optimal_product_count': optimal_count,
            'source_sale_ids': [(6, 0, sale_orders[:100].ids)],  # Link first 100 sales
            'source_composition_ids': [(6, 0, compositions[:50].ids)],  # Link first 50 compositions
            'last_updated': fields.Datetime.now(),
            'learning_notes': f"""
            Learned from {len(sale_orders)} sales and {len(compositions)} compositions.
            Found {len(patterns['products_used'])} unique products used.
            Identified {len(patterns['successful_combinations'])} successful combinations.
            Average products per composition: {optimal_count}
            """
        })
        
        _logger.info(f"Learning complete: {optimal_count} optimal products, {len(patterns['products_used'])} products tracked")
        
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

    @api.model
    def get_recommendations_for_budget(self, target_budget, partner_id=None):
        """Get learned recommendations for a specific budget"""
        
        # Try client-specific patterns first
        if partner_id:
            client_learning = self.search([
                ('partner_id', '=', partner_id),
                ('pattern_type', '=', 'client_specific')
            ], limit=1)
            
            if client_learning and client_learning.pattern_data:
                try:
                    data = json.loads(client_learning.pattern_data)
                    return self._extract_budget_recommendations(data, target_budget)
                except:
                    pass
        
        # Fall back to general patterns
        general_learning = self.search([('pattern_type', '=', 'general')], limit=1)
        if general_learning and general_learning.pattern_data:
            try:
                patterns = json.loads(general_learning.pattern_data)
                return self._extract_budget_recommendations(patterns, target_budget)
            except Exception as e:
                _logger.error(f"Failed to parse learning data: {e}")
        
        # Return empty dict if no patterns found
        return {}

    def _extract_budget_recommendations(self, patterns, target_budget):
        """Extract relevant recommendations for a budget range"""
        
        budget_range = self._get_budget_range(target_budget)
        
        result = {
            'budget_range': budget_range,
            'product_count': 4,  # Default
            'categories': []
        }
        
        # Get budget-specific patterns if available
        if 'budget_patterns' in patterns:
            budget_data = patterns['budget_patterns'].get(budget_range, {})
            if budget_data:
                # If it's a list, get average values
                if isinstance(budget_data, list) and budget_data:
                    counts = [b.get('product_count', 4) for b in budget_data if 'product_count' in b]
                    if counts:
                        result['product_count'] = int(sum(counts) / len(counts))
                    
                    # Collect all categories
                    for b in budget_data:
                        if 'categories' in b:
                            result['categories'].extend(b['categories'])
                
                elif isinstance(budget_data, dict):
                    result['product_count'] = budget_data.get('product_count', 4)
                    result['categories'] = budget_data.get('categories', [])
        
        # Get popular products if available
        if 'products_used' in patterns:
            # Sort products by usage count
            products = patterns['products_used']
            if products:
                sorted_products = sorted(
                    [(pid, p.get('usage_count', 0)) for pid, p in products.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                result['popular_products'] = [p[0] for p in sorted_products[:20]]
        
        return result