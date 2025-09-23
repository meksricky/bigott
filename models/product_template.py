# models/product_template.py
from odoo import models, fields, api

class ProductTemplate(models.Model):
    _inherit = 'product.template'
    
    # ========== KEEP ALL YOUR EXISTING FIELDS ==========
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
    
    # Dietary restriction fields (EXISTING)
    is_vegan = fields.Boolean(string="Vegan", default=False)
    is_halal = fields.Boolean(string="Halal", default=False)
    is_gluten_free = fields.Boolean(string="Gluten Free", default=False)
    contains_alcohol = fields.Boolean(string="Contains Alcohol", default=False)
    
    # Gift suitability (EXISTING)
    gift_suitable = fields.Boolean(string="Suitable for Gifts", default=True)
    min_gift_budget = fields.Float(string="Minimum Gift Budget", default=0.0)
    max_gift_budget = fields.Float(string="Maximum Gift Budget", default=1000.0)
    
    # Experience-related fields (EXISTING)
    is_experience = fields.Boolean(
        string='Is Experience',
        default=False,
        help='Check if this product is an experience/activity rather than a physical product'
    )
    
    experience_category = fields.Selection([
        ('gastronomy', 'Gastronomy'),
        ('wellness', 'Wellness & Spa'),
        ('adventure', 'Adventure'),
        ('culture', 'Culture & Arts'),
        ('luxury', 'Luxury Experiences'),
        ('other', 'Other')
    ], string='Experience Category')
    
    experience_duration = fields.Float(
        string='Duration (hours)',
        help='Duration of the experience in hours'
    )
    
    experience_location = fields.Char(
        string='Experience Location'
    )
    
    max_participants = fields.Integer(
        string='Maximum Participants',
        default=1
    )
    
    # ========== NEW FIELDS REQUIRED FOR BUSINESS RULES ==========
    
    # CRITICAL FOR R1 RULE (Exact Repeat)
    beverage_family = fields.Selection([
        ('cava', 'Cava'),
        ('champagne', 'Champagne'),
        ('vermouth', 'Vermouth'),
        ('tokaj', 'Tokaji'),  # Note: Tokaj/Tokaji variation
        ('wine', 'Wine (Generic)'),
        ('red_wine', 'Red Wine'),
        ('white_wine', 'White Wine'),
        ('rose_wine', 'Rosé Wine'),
        ('beer', 'Beer'),
        ('spirits_high', 'High Grade Spirits')
    ], string='Beverage Family', 
       help='Used for R1 rule: Cava/Champagne/Vermouth/Tokaji must repeat exactly')
    
    # CRITICAL FOR R2 RULE (Wine Rotation)
    wine_color = fields.Selection([
        ('red', 'Red'),
        ('white', 'White'),
        ('rose', 'Rosé')
    ], string='Wine Color',
       help='Used for R2 rule: Wine brand rotation with same color')
    
    volume_ml = fields.Integer(
        string='Volume (ml)', 
        help='Bottle size in ml - MUST be preserved in transformations'
    )
    
    # CRITICAL FOR R3 RULE (Experience Bundles)
    is_experience_only = fields.Boolean(
        string='Experience Bundle Only',
        help='This is an experience bundle that gets replaced with same-size bundle'
    )
    
    experience_product_count = fields.Integer(
        string='Products in Experience',
        help='Number of products in this experience bundle'
    )
    
    # CRITICAL FOR R4 RULE (Paletilla/Charcuterie)
    is_paletilla = fields.Boolean(
        string='Is Paletilla',
        help='R4 Rule: Paletilla MUST be repeated exactly every year'
    )
    
    is_charcuterie_item = fields.Boolean(
        string='Charcuterie Board Item',
        help='Part of charcuterie selection'
    )
    
    # CRITICAL FOR R5 RULE (Foie Rotation)
    foie_variant = fields.Selection([
        ('duck', 'Duck'),
        ('goose', 'Goose')
    ], string='Foie Gras Variant',
       help='R5 Rule: Alternate Duck ↔ Goose each year')
    
    # CRITICAL FOR R6 RULE (Sweets)
    sweets_subtype = fields.Selection([
        ('lingote', 'Lingote'),
        ('trufa_lebigott', 'Trufa leBigott'),
        ('turron', 'Turrón'),
        ('chocolate', 'Chocolate'),
        ('dulces', 'Traditional Sweets'),
        ('other', 'Other')
    ], string='Sweets Subtype')
    
    is_lingote = fields.Boolean(
        string='Is Lingote',
        help='R6 Rule: Lingote must ALWAYS be repeated exactly'
    )
    
    is_trufa_lebigott = fields.Boolean(
        string='Is Trufa leBigott',
        help='R6 Rule: Trufa leBigott must ALWAYS be repeated exactly'
    )
    
    turron_style = fields.Selection([
        ('jijona', 'Jijona'),
        ('alicante', 'Alicante'),
        ('praline', 'Praliné'),
        ('chocolate', 'Chocolate'),
        ('other', 'Other')
    ], string='Turrón Style',
       help='R6 Rule: Keep same style, brand may change')
    
    # ADDITIONAL DIETARY RESTRICTIONS (for complete halal support)
    contains_pork = fields.Boolean(
        string='Contains Pork',
        default=False,
        help='Critical for Halal restriction'
    )
    
    is_iberian_product = fields.Boolean(
        string='Iberian Product',
        default=False,
        help='Iberian products not allowed for Halal'
    )
    
    contains_dairy = fields.Boolean(
        string='Contains Dairy',
        default=False
    )
    
    contains_nuts = fields.Boolean(
        string='Contains Nuts',
        default=False
    )
    
    # ADDITIONAL CATEGORIZATION
    is_canned_food = fields.Boolean(
        string='Canned Food Item',
        default=False
    )
    
    # Price band calculation
    price_band = fields.Selection([
        ('low', 'Low'),
        ('mid', 'Mid'),
        ('high', 'High'),
        ('prestige', 'Prestige')
    ], string='Price Band', compute='_compute_price_band', store=True)
    
    # Stock helper
    has_stock = fields.Boolean(
        string='Has Stock',
        compute='_compute_has_stock'
    )
    
    # Multi-level client handling
    allowed_for_level = fields.Selection([
        ('all', 'All Levels'),
        ('level1', 'Level 1 Only'),
        ('level2', 'Level 2 Only'),
        ('level3', 'Level 3 Only'),
        ('vip', 'VIP Only')
    ], string='Allowed Client Level', default='all')
    
    # ========== KEEP YOUR EXISTING METHODS ==========
    
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
    
    @api.onchange('is_experience')
    def _onchange_is_experience(self):
        """Clear experience fields if not an experience"""
        if not self.is_experience:
            self.experience_category = False
            self.experience_duration = False
            self.experience_location = False
            self.max_participants = 1
    
    # ========== NEW COMPUTED FIELDS & METHODS ==========
    
    @api.depends('list_price')
    def _compute_price_band(self):
        """Compute price band based on price thresholds"""
        for product in self:
            if product.list_price <= 50:
                product.price_band = 'low'
            elif product.list_price <= 150:
                product.price_band = 'mid'
            elif product.list_price <= 500:
                product.price_band = 'high'
            else:
                product.price_band = 'prestige'
    
    @api.depends('qty_available')
    def _compute_has_stock(self):
        """Check if product has stock available"""
        for product in self:
            product.has_stock = product.qty_available > 0
    
    @api.onchange('lebiggot_category')
    def _onchange_lebiggot_category(self):
        """Auto-set related boolean fields based on category"""
        if self.lebiggot_category == 'foie_gras':
            # Prompt user to set duck/goose variant
            return {
                'warning': {
                    'title': 'Set Foie Variant',
                    'message': 'Please specify if this is Duck or Goose foie gras for rotation rules'
                }
            }
        elif self.lebiggot_category == 'charcuterie':
            self.is_charcuterie_item = True
        elif self.lebiggot_category == 'sweets':
            # Prompt for sweets subtype
            return {
                'warning': {
                    'title': 'Set Sweets Type',
                    'message': 'Please specify the sweets subtype (Lingote, Turrón, etc.)'
                }
            }
    
    @api.onchange('beverage_family')
    def _onchange_beverage_family(self):
        """Auto-set wine color if wine family selected"""
        if self.beverage_family in ['red_wine', 'white_wine', 'rose_wine']:
            if self.beverage_family == 'red_wine':
                self.wine_color = 'red'
            elif self.beverage_family == 'white_wine':
                self.wine_color = 'white'
            elif self.beverage_family == 'rose_wine':
                self.wine_color = 'rose'
    
    def check_business_rules_compliance(self):
        """Check if product has all required fields for business rules"""
        self.ensure_one()
        
        issues = []
        
        # Check R1 compliance
        if self.beverage_family in ['cava', 'champagne', 'vermouth', 'tokaj']:
            if not self.volume_ml:
                issues.append("R1: Missing volume_ml for exact repeat beverage")
        
        # Check R2 compliance
        if self.beverage_family == 'wine':
            if not self.wine_color:
                issues.append("R2: Missing wine_color for wine rotation")
            if not self.volume_ml:
                issues.append("R2: Missing volume_ml for wine")
        
        # Check R5 compliance
        if self.lebiggot_category == 'foie_gras':
            if not self.foie_variant:
                issues.append("R5: Missing foie_variant (duck/goose)")
        
        # Check R6 compliance
        if self.lebiggot_category == 'sweets':
            if not self.sweets_subtype:
                issues.append("R6: Missing sweets_subtype")
        
        if issues:
            return {
                'compliant': False,
                'issues': issues
            }
        
        return {
            'compliant': True,
            'message': 'Product configured correctly for business rules'
        }
    
    def apply_dietary_filter(self, dietary_restrictions):
        """Check if product complies with dietary restrictions"""
        if not dietary_restrictions:
            return True
        
        for restriction in dietary_restrictions:
            if restriction == 'halal':
                if self.contains_pork or self.contains_alcohol or self.is_iberian_product:
                    return False
            elif restriction == 'vegan':
                if not self.is_vegan:
                    return False
            elif restriction == 'gluten_free':
                if not self.is_gluten_free:
                    return False
            elif restriction in ['non_alcoholic', 'no_alcohol']:
                if self.contains_alcohol:
                    return False
            elif restriction == 'no_pork':
                if self.contains_pork:
                    return False
            elif restriction == 'no_iberian':
                if self.is_iberian_product:
                    return False
        
        return True
    
    @api.model
    def get_products_for_rule(self, rule_code):
        """Get products that need specific business rule treatment"""
        if rule_code == 'R1':
            # Exact repeat products
            return self.search([
                ('beverage_family', 'in', ['cava', 'champagne', 'vermouth', 'tokaj'])
            ])
        elif rule_code == 'R2':
            # Wine rotation products
            return self.search([
                ('beverage_family', '=', 'wine')
            ])
        elif rule_code == 'R3':
            # Experience bundles
            return self.search([
                ('is_experience_only', '=', True)
            ])
        elif rule_code == 'R4':
            # Paletilla and charcuterie
            return self.search([
                '|',
                ('is_paletilla', '=', True),
                ('is_charcuterie_item', '=', True)
            ])
        elif rule_code == 'R5':
            # Foie gras rotation
            return self.search([
                ('lebiggot_category', '=', 'foie_gras')
            ])
        elif rule_code == 'R6':
            # Sweets rules
            return self.search([
                ('lebiggot_category', '=', 'sweets')
            ])
        
        return self.browse()