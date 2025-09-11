# models/product_template.py

from odoo import models, fields, api

class ProductTemplate(models.Model):
    _inherit = 'product.template'
    
    # Gift-specific fields
    lebiggot_category = fields.Selection([
        ('wines', 'Wines'),
        ('spirits', 'Spirits'),
        ('champagne', 'Champagne'),
        ('charcuterie', 'Charcuterie'),
        ('cheese', 'Cheese'),
        ('foie_gras', 'Foie Gras'),
        ('sweets', 'Sweets'),
        ('preserves', 'Preserves'),
        ('olive_oil', 'Olive Oil'),
        ('vinegar', 'Vinegar'),
        ('nuts', 'Nuts'),
        ('coffee', 'Coffee'),
        ('tea', 'Tea'),
        ('chocolates', 'Chocolates'),
        ('crackers', 'Crackers'),
        ('other', 'Other')
    ], string="Gift Category")
    
    product_grade = fields.Selection([
        ('premium', 'Premium'),
        ('luxury', 'Luxury'),
        ('ultra_luxury', 'Ultra Luxury'),
        ('exclusive', 'Exclusive')
    ], string="Product Grade", default='premium')
    
    brand = fields.Char(string="Brand")
    
    # Dietary restriction fields
    is_vegan = fields.Boolean(string="Vegan", default=False)
    is_halal = fields.Boolean(string="Halal", default=False)
    is_gluten_free = fields.Boolean(string="Gluten Free", default=False)
    contains_alcohol = fields.Boolean(string="Contains Alcohol", default=False)
    
    # Gift suitability
    gift_suitable = fields.Boolean(string="Suitable for Gifts", default=True)
    min_gift_budget = fields.Float(string="Minimum Gift Budget", default=0.0)
    max_gift_budget = fields.Float(string="Maximum Gift Budget", default=1000.0)
    
    @api.model
    def search_gift_products(self, domain=None):
        """Search products suitable for gifts"""
        if domain is None:
            domain = []
        
        gift_domain = [
            ('gift_suitable', '=', True),
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('list_price', '>', 0),
            ('default_code', '!=', False),  # Must have internal reference
        ]
        
        return self.search(gift_domain + domain)