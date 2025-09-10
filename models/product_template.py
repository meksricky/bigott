from odoo import models, fields, api

class ProductTemplate(models.Model):
    _inherit = 'product.template'
    
    # Le Biggot specific categorization
    lebiggot_category = fields.Selection([
        ('main_beverage', 'Main Beverage'),
        ('aperitif', 'Aperitif'),
        ('experience_gastronomica', 'Experience Gastronomica'),
        ('foie_gras', 'Foie Gras'),
        ('charcuterie', 'Charcuterie'),
        ('sweets', 'Sweets')
    ], string='Le Biggot Category')

    beverage_family = fields.Selection([
        ('wine', 'Wine'),
        ('cava', 'Cava'),
        ('champagne', 'Champagne'),
        ('vermouth', 'Vermouth'),
        ('tokaj', 'Tokaj'),
    ], string='Beverage Family')

    wine_color = fields.Selection([
        ('red', 'Red'),
        ('white', 'White'),
        ('rose', 'Rosé'),
    ], string='Wine Color')

    foie_variant = fields.Selection([
        ('duck', 'Duck'),
        ('goose', 'Goose'),
    ], string='Foie Variant')

    sweets_subtype = fields.Char('Sweets Subtype')
    is_paletilla = fields.Boolean('Is Paletilla')
    is_charcuterie_item = fields.Boolean('Is Charcuterie Item')
    is_experience_only = fields.Boolean('Experience Only')
    rule_locked = fields.Boolean('Rule Locked')
    substitute_group = fields.Char('Substitute Group')
    supplier_code = fields.Char('Supplier Code')
    production_batch = fields.Char('Production Batch')

    # And add the missing method
    def action_view_business_rules(self):
        return {
            'type': 'ir.actions.act_window',
            'name': 'Business Rules',
            'res_model': 'product.template',
            'res_id': self.id,
            'view_mode': 'form',
        }
    
    # Detailed beverage categorization
    beverage_type = fields.Selection([
        ('cava', 'Cava'),
        ('champagne', 'Champagne'),
        ('red_wine', 'Red Wine'),
        ('white_wine', 'White Wine'),
        ('rose_wine', 'Rosé Wine'),
        ('vermouth', 'Vermouth'),
        ('tokaj', 'Tokaj'),
        ('sherry', 'Sherry'),
        ('port', 'Port Wine')
    ], string='Beverage Type')
    
    # Volume for beverages
    volume_ml = fields.Integer('Volume (ml)')
    
    # Product grade for budget adaptation
    product_grade = fields.Selection([
        ('economical', 'Economical'),
        ('standard', 'Standard'),
        ('premium', 'Premium'),
        ('luxury', 'Luxury')
    ], string='Product Grade', default='standard')
    
    # Dietary properties
    is_vegan = fields.Boolean('Vegan')
    is_halal = fields.Boolean('Halal')
    contains_alcohol = fields.Boolean('Contains Alcohol', default=False)
    
    # Product details
    brand = fields.Char('Brand')
    origin_region = fields.Char('Origin Region')
    vintage_year = fields.Integer('Vintage Year')
    
    # Experience associations
    experience_ids = fields.Many2many('gift.experience', string='Part of Experiences')
    
    # Taste profile for matching
    taste_profile = fields.Text('Taste Profile')
    pairing_suggestions = fields.Text('Pairing Suggestions')
    
    # Usage tracking
    times_recommended = fields.Integer('Times Recommended', compute='_compute_recommendation_stats')
    success_rate = fields.Float('Success Rate %', compute='_compute_recommendation_stats')
    
    @api.depends('name')  # Placeholder - would need actual recommendation tracking
    def _compute_recommendation_stats(self):
        for product in self:
            # This would be computed from actual recommendation usage
            product.times_recommended = 0
            product.success_rate = 0.0
    
    @api.model
    def find_substitutes(self, original_product, budget_direction='same', dietary_restrictions=None):
        """Find substitute products for budget adaptation"""
        
        if not original_product.lebiggot_category:
            return self.env['product.template']
        
        # Base search criteria
        domain = [
            ('lebiggot_category', '=', original_product.lebiggot_category),
            ('id', '!=', original_product.id),
            ('active', '=', True),
            ('sale_ok', '=', True)
        ]
        
        # Volume matching for beverages
        if original_product.volume_ml:
            domain.append(('volume_ml', '=', original_product.volume_ml))
        
        # Dietary restrictions
        if dietary_restrictions:
            if 'vegan' in dietary_restrictions:
                domain.append(('is_vegan', '=', True))
            if 'halal' in dietary_restrictions:
                domain.append(('is_halal', '=', True))
            if 'non_alcoholic' in dietary_restrictions:
                domain.append(('contains_alcohol', '=', False))
        
        # Budget direction
        grade_hierarchy = ['economical', 'standard', 'premium', 'luxury']
        current_grade_index = grade_hierarchy.index(original_product.product_grade)
        
        if budget_direction == 'upgrade':
            if current_grade_index < len(grade_hierarchy) - 1:
                target_grades = grade_hierarchy[current_grade_index + 1:]
                domain.append(('product_grade', 'in', target_grades))
        elif budget_direction == 'downgrade':
            if current_grade_index > 0:
                target_grades = grade_hierarchy[:current_grade_index]
                domain.append(('product_grade', 'in', target_grades))
        else:
            # Same grade, different brand
            domain.append(('product_grade', '=', original_product.product_grade))
            if original_product.brand:
                domain.append(('brand', '!=', original_product.brand))
        
        candidates = self.search(domain)
        
        # Sort by price based on budget direction
        if budget_direction == 'upgrade':
            candidates = candidates.sorted('list_price', reverse=True)
        elif budget_direction == 'downgrade':
            candidates = candidates.sorted('list_price')
        else:
            # For same budget, prefer similar price
            target_price = original_product.list_price
            candidates = candidates.sorted(key=lambda p: abs(p.list_price - target_price))
        
        return candidates[:5]  # Return top 5 candidates

    def action_view_experiences(self):
        """View experiences that include this product"""
        return {
            'type': 'ir.actions.act_window',
            'name': f'Experiences with {self.name}',
            'res_model': 'gift.experience',
            'view_mode': 'tree,form',
            'domain': [('product_ids', 'in', [self.id])],
            'context': {'default_product_ids': [(6, 0, [self.id])]}
        }
