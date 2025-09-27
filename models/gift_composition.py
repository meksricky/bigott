from odoo import models, fields, api
from datetime import datetime
import logging

_logger = logging.getLogger(__name__)

class GiftComposition(models.Model):
    _name = 'gift.composition'
    _description = 'Gift Composition'
    _inherit = ['mail.thread', 'mail.activity.mixin']
    _order = 'create_date desc'
    _rec_name = 'display_name'

    # ============== BASIC FIELDS ==============
    name = fields.Char(string='Name', default='New', copy=False, tracking=True)
    reference = fields.Char(string='Reference', copy=False, readonly=True, 
                          help="Unique reference for this composition")
    display_name = fields.Char(string='Display Name', compute='_compute_display_name', store=True)
    
    partner_id = fields.Many2one('res.partner', string='Client', required=True, ondelete='cascade', tracking=True)
    partner_email = fields.Char(related='partner_id.email', string='Email', readonly=True)
    partner_phone = fields.Char(related='partner_id.phone', string='Phone', readonly=True)
    
    # ============== COMPOSITION TYPE FIELDS ==============
    composition_type = fields.Selection([
        ('experience', 'Experience'),
        ('custom', 'Custom Made'),
        ('hybrid', 'Hybrid')
    ], string='Type', default='custom', required=True, tracking=True,
       help="Experience: Pre-configured package | Custom: AI-generated | Hybrid: Experience + Custom")
    
    experience_code = fields.Char(string='Experience Code', help="Internal code for the experience")
    experience_name = fields.Char(string='Experience Name', help="Display name of the experience")
    experience_category = fields.Selection([
        ('aperitif', 'Aperitif'),
        ('seafood', 'Seafood'),
        ('meat', 'Meat'),
        ('vegetarian', 'Vegetarian'),
        ('vegan', 'Vegan'),
        ('dessert', 'Dessert'),
        ('foie', 'Foie Gras'),
        ('other', 'Other')
    ], string='Experience Category')
    
    # ============== BUDGET FIELDS ==============
    currency_id = fields.Many2one('res.currency', default=lambda self: self.env.company.currency_id)
    target_budget = fields.Monetary(string='Target Budget', currency_field='currency_id', required=True, tracking=True)
    actual_cost = fields.Monetary(string='Actual Cost', currency_field='currency_id', compute='_compute_actual_cost', store=True)
    budget_variance = fields.Float(string='Budget Variance %', compute='_compute_budget_variance', store=True)
    
    # ADD THESE VARIANCE FIELDS (referenced in view)
    variance_amount = fields.Float(string="Variance Amount", compute="_compute_variance", store=True)
    variance_percentage = fields.Float(string="Variance %", compute="_compute_variance", store=True)
    
    target_year = fields.Integer(string='Target Year', default=lambda self: datetime.now().year, required=True)
    
    # ============== PRODUCT FIELDS ==============
    product_ids = fields.Many2many('product.template', string='All Products')
    
    # Categorized Products
    beverage_product_ids = fields.Many2many(
        'product.template', 
        'composition_beverage_rel',
        'composition_id', 
        'product_id',
        string='Main Beverages',
        help='Cava, Champagne, Red Wine, Rosé Wine, White Wine'
    )
    
    aperitif_product_ids = fields.Many2many(
        'product.template',
        'composition_aperitif_rel', 
        'composition_id',
        'product_id',
        string='Aperitifs',
        help='Vermouth, Tokaji, Beer, High-Grade Alcohol'
    )
    
    foie_product_ids = fields.Many2many(
        'product.template',
        'composition_foie_rel',
        'composition_id', 
        'product_id',
        string='Foie Gras',
        help='Duck or Goose (only if not in Experience)'
    )
    
    canned_product_ids = fields.Many2many(
        'product.template',
        'composition_canned_rel',
        'composition_id',
        'product_id',
        string='Canned Foods',
        help='Fish, Vegetables, Dressings'
    )
    
    charcuterie_product_ids = fields.Many2many(
        'product.template',
        'composition_charcuterie_rel',
        'composition_id',
        'product_id',
        string='Charcuterie Board',
        help='Paletilla, Ibericos, Cheese'
    )
    
    sweet_product_ids = fields.Many2many(
        'product.template',
        'composition_sweet_rel',
        'composition_id',
        'product_id',
        string='Sweets',
        help='Turrón, Lingote, Trufas, Dulces'
    )
    
    # ============== DIETARY & NOTES FIELDS ==============
    dietary_restrictions = fields.Text(string='Dietary Restrictions')
    dietary_restriction_type = fields.Selection([
        ('none', 'None'),
        ('halal', 'Halal'),
        ('vegan', 'Vegan'),
        ('vegetarian', 'Vegetarian'),
        ('gluten_free', 'Gluten Free'),
        ('non_alcoholic', 'Non-Alcoholic'),
        ('multiple', 'Multiple')
    ], string='Restriction Type', default='none')
    
    client_notes = fields.Text(string='Client Notes')
    internal_notes = fields.Text(string='Internal Notes')
    
    # ============== AI GENERATION FIELDS ==============
    ai_reasoning = fields.Text(string='AI Reasoning')
    confidence_score = fields.Float(string='Confidence Score', digits=(3, 2))
    generation_method = fields.Selection([
        ('ollama', 'Ollama AI'),
        ('fallback', 'Rule-Based'),
        ('manual', 'Manual'),
        ('experience', 'Experience-Based'),
        ('hybrid', 'Hybrid')
    ], string='Generation Method', default='manual')
    
    compliance_status = fields.Char(string="Compliance Status", readonly=True)
    generation_strategy = fields.Char(string="Generation Strategy", readonly=True)
    
    # ============== STATE FIELD ==============
    state = fields.Selection([
        ('draft', 'Draft'),
        ('confirmed', 'Confirmed'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled'),
        ('replaced', 'Replaced'),  # ADD THIS LINE
    ], string='State', default='draft', track_visibility='onchange')
    
    # ============== STATISTICS FIELDS ==============
    product_count = fields.Integer(string='Product Count', compute='_compute_product_count', store=True)
    category_distribution = fields.Text(string='Category Distribution', compute='_compute_category_distribution')
    
    # ADD THESE CATEGORY TRACKING FIELDS (referenced in view)
    total_categories = fields.Integer(string="Total Categories", compute="_compute_categories", store=True)
    wine_count = fields.Integer(string="Wine Products", compute="_compute_categories", store=True)
    cheese_count = fields.Integer(string="Cheese Products", compute="_compute_categories", store=True)
    gourmet_count = fields.Integer(string="Gourmet Products", compute="_compute_categories", store=True)
    categories_distribution = fields.Html(string="Category Distribution HTML", compute="_compute_categories")
    
    # Price statistics
    avg_product_price = fields.Float(string="Average Price", compute="_compute_price_statistics", store=True)
    min_product_price = fields.Float(string="Min Price", compute="_compute_price_statistics", store=True)
    max_product_price = fields.Float(string="Max Price", compute="_compute_price_statistics", store=True)
    price_std_deviation = fields.Float(string="Price Std Dev", compute="_compute_price_statistics", store=True)


    active = fields.Boolean(
        string='Active', 
        default=True,
        help="If unchecked, it will allow you to hide the composition without removing it."
    )
    
    # ============== COMPUTED METHODS ==============
    
    @api.depends('name', 'partner_id', 'composition_type', 'experience_name')
    def _compute_display_name(self):
        for record in self:
            parts = []
            
            # Add type indicator
            if record.composition_type == 'experience':
                parts.append('🎁')
            elif record.composition_type == 'hybrid':
                parts.append('🎨')
            else:
                parts.append('🔧')
            
            # Add reference or name
            if record.reference:
                parts.append(record.reference)
            elif record.name and record.name != 'New':
                parts.append(record.name)
            
            # Add client name
            if record.partner_id:
                parts.append(f"- {record.partner_id.name}")
            
            # Add experience name if applicable
            if record.experience_name and record.composition_type in ['experience', 'hybrid']:
                parts.append(f"({record.experience_name})")
            
            record.display_name = ' '.join(parts) if parts else 'New Composition'
    
    @api.depends('product_ids', 'beverage_product_ids', 'aperitif_product_ids', 
                 'foie_product_ids', 'canned_product_ids', 'charcuterie_product_ids', 
                 'sweet_product_ids')
    def _compute_actual_cost(self):
        for record in self:
            total = 0.0
            
            # Get all unique products
            all_products = set()
            
            # Add categorized products
            for field_name in ['beverage_product_ids', 'aperitif_product_ids', 'foie_product_ids',
                              'canned_product_ids', 'charcuterie_product_ids', 'sweet_product_ids']:
                products = getattr(record, field_name)
                all_products.update(products.ids)
            
            # Also include main product_ids
            all_products.update(record.product_ids.ids)
            
            # Calculate total
            for product_id in all_products:
                product = self.env['product.template'].browse(product_id)
                total += product.list_price
            
            record.actual_cost = total
    
    @api.depends('target_budget', 'actual_cost')
    def _compute_budget_variance(self):
        for record in self:
            if record.target_budget > 0:
                variance = ((record.actual_cost - record.target_budget) / record.target_budget) * 100
                record.budget_variance = variance
            else:
                record.budget_variance = 0.0
    
    @api.depends('actual_cost', 'target_budget')
    def _compute_variance(self):
        """Compute variance amount and percentage"""
        for record in self:
            record.variance_amount = record.actual_cost - record.target_budget
            if record.target_budget > 0:
                record.variance_percentage = ((record.actual_cost - record.target_budget) / record.target_budget) * 100
            else:
                record.variance_percentage = 0
    
    @api.depends('product_ids', 'beverage_product_ids', 'aperitif_product_ids',
                 'foie_product_ids', 'canned_product_ids', 'charcuterie_product_ids',
                 'sweet_product_ids')
    def _compute_product_count(self):
        for record in self:
            # Count unique products
            all_products = set()
            
            # Add all categorized products
            for field_name in ['product_ids', 'beverage_product_ids', 'aperitif_product_ids',
                              'foie_product_ids', 'canned_product_ids', 'charcuterie_product_ids',
                              'sweet_product_ids']:
                products = getattr(record, field_name)
                all_products.update(products.ids)
            
            record.product_count = len(all_products)
    
    @api.depends('beverage_product_ids', 'aperitif_product_ids', 'foie_product_ids',
                 'canned_product_ids', 'charcuterie_product_ids', 'sweet_product_ids')
    
    def _compute_category_distribution(self):
        """Compute the distribution of products by category"""
        for record in self:
            distribution = []
            
            categories = {
                'Beverages': len(record.beverage_product_ids),
                'Aperitifs': len(record.aperitif_product_ids),
                'Foie Gras': len(record.foie_product_ids),
                'Canned Foods': len(record.canned_product_ids),
                'Charcuterie': len(record.charcuterie_product_ids),
                'Sweets': len(record.sweet_product_ids),
            }
            
            for cat, count in categories.items():
                if count > 0:
                    distribution.append(f"{cat}: {count}")
            
            record.category_distribution = " | ".join(distribution) if distribution else "No categorized products"
    
    @api.depends('product_ids')
    def _compute_categories(self):
        """Compute category distribution with HTML visualization"""
        for record in self:
            categories = {}
            wine_count = 0
            cheese_count = 0
            gourmet_count = 0
            
            for product in record.product_ids:
                if hasattr(product, 'categ_id') and product.categ_id:
                    cat_name = product.categ_id.name
                    categories[cat_name] = categories.get(cat_name, 0) + 1
                    
                    # Count specific categories
                    cat_lower = cat_name.lower()
                    if any(word in cat_lower for word in ['wine', 'vino', 'champagne', 'cava']):
                        wine_count += 1
                    elif any(word in cat_lower for word in ['cheese', 'queso']):
                        cheese_count += 1
                    elif 'gourmet' in cat_lower:
                        gourmet_count += 1
            
            # Build HTML distribution
            html = "<div style='font-family: sans-serif;'>"
            if record.product_ids:
                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(record.product_ids) * 100)
                    html += f"""
                        <div style='margin: 8px 0;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                                <strong>{cat}:</strong>
                                <span>{count} products ({percentage:.1f}%)</span>
                            </div>
                            <div style='background: #f0f0f0; height: 20px; width: 100%; border-radius: 3px; overflow: hidden;'>
                                <div style='background: #7c7bad; height: 100%; width: {percentage}%; border-radius: 3px; transition: width 0.3s;'></div>
                            </div>
                        </div>
                    """
            else:
                html += "<p>No products selected yet</p>"
            html += "</div>"
            
            record.categories_distribution = html
            record.total_categories = len(categories)
            record.wine_count = wine_count
            record.cheese_count = cheese_count
            record.gourmet_count = gourmet_count
    
    @api.depends('product_ids')
    def _compute_price_statistics(self):
        """Compute price statistics"""
        for record in self:
            if record.product_ids:
                prices = record.product_ids.mapped('list_price')
                record.avg_product_price = sum(prices) / len(prices) if prices else 0
                record.min_product_price = min(prices) if prices else 0
                record.max_product_price = max(prices) if prices else 0
                
                # Calculate standard deviation
                if len(prices) > 1:
                    avg = record.avg_product_price
                    variance = sum((p - avg) ** 2 for p in prices) / len(prices)
                    record.price_std_deviation = variance ** 0.5
                else:
                    record.price_std_deviation = 0
            else:
                record.avg_product_price = 0
                record.min_product_price = 0
                record.max_product_price = 0
                record.price_std_deviation = 0
    
    # ============== CREATE/WRITE METHODS ==============
    
    @api.model_create_multi
    def create(self, vals_list):
        for vals in vals_list:
            if vals.get('name', 'New') == 'New':
                # Generate sequence based on composition type
                comp_type = vals.get('composition_type', 'custom')
                prefix = {
                    'experience': 'EXP',
                    'hybrid': 'HYB',
                    'custom': 'CST'
                }.get(comp_type, 'GFT')
                
                # Generate sequence
                sequence = self.env['ir.sequence'].next_by_code('gift.composition')
                if sequence:
                    vals['name'] = sequence
                    vals['reference'] = sequence
                else:
                    # Fallback if no sequence exists
                    import random
                    vals['name'] = f'{prefix}{random.randint(1000, 9999)}'
                    vals['reference'] = vals['name']
        
        return super().create(vals_list)
    
    # ============== ACTION METHODS ==============
    
    def action_confirm(self):
        """Confirm the composition"""
        for record in self:
            record.state = 'confirmed'
            
            # Log the confirmation using mail thread
            body = f"""
            <p><strong>Composition Confirmed</strong></p>
            <ul>
                <li>Type: {record.composition_type}</li>
                <li>Budget: €{record.target_budget:.2f}</li>
                <li>Actual Cost: €{record.actual_cost:.2f}</li>
                <li>Products: {record.product_count}</li>
            </ul>
            """
            
            if record.experience_name:
                body += f"<p>Experience: {record.experience_name}</p>"
            
            record.message_post(body=body, subject="Composition Confirmed")
        
        return True
    
    def action_cancel(self):
        """Cancel the composition"""
        for record in self:
            record.state = 'cancelled'
        return True
    
    def action_set_draft(self):
        """Reset to draft"""
        for record in self:
            record.state = 'draft'
        return True
    
    def action_deliver(self):
        """Mark as delivered"""
        for record in self:
            record.state = 'delivered'
            # Create a learning record if applicable
            if record.generation_method in ['ollama', 'experience', 'hybrid']:
                record._create_learning_record()
        return True
    
    def _create_learning_record(self):
        """Create a learning record for AI improvement"""
        self.ensure_one()
        
        # Check if the model exists
        if 'recommendation.learning' in self.env:
            learning_vals = {
                'composition_id': self.id,
                'partner_id': self.partner_id.id,
                'composition_type': self.composition_type,
                'experience_code': self.experience_code,
                'budget': self.target_budget,
                'actual_cost': self.actual_cost,
                'product_count': self.product_count,
                'confidence_score': self.confidence_score,
                'state': 'delivered',
                'feedback_score': 0.0,
            }
            
            self.env['recommendation.learning'].create(learning_vals)
    
    def action_duplicate(self):
        """Duplicate composition with updated year"""
        self.ensure_one()
        
        new_comp = self.copy({
            'target_year': datetime.now().year,
            'state': 'draft',
            'name': 'New'
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Duplicated Composition',
            'res_model': 'gift.composition',
            'res_id': new_comp.id,
            'view_mode': 'form',
            'target': 'current',
        }
    
    def action_regenerate(self):
        """Directly regenerate composition without wizard"""
        self.ensure_one()
        
        # Archive current composition
        self.write({
            'state': 'replaced',
            'active': False
        })
        
        # Get recommender
        recommender = self.env['ollama.gift.recommender'].get_or_create_recommender()
        
        # Parse existing dietary restrictions back to list
        dietary = []
        if self.dietary_restrictions:
            dietary_text = self.dietary_restrictions.lower()
            if 'halal' in dietary_text:
                dietary.append('halal')
            if 'vegan' in dietary_text:
                dietary.append('vegan')
            if 'vegetarian' in dietary_text:
                dietary.append('vegetarian')
            if 'no alcohol' in dietary_text or 'non_alcoholic' in dietary_text:
                dietary.append('non_alcoholic')
            if 'gluten' in dietary_text:
                dietary.append('gluten_free')
        
        _logger.info(f"""
        ╔═══════════════════════════════════════════════════════╗
        ║         🔄 REGENERATING COMPOSITION                   ║
        ╠═══════════════════════════════════════════════════════╣
        ║ Original: {self.reference:<43} ║
        ║ Client: {self.partner_id.name[:40]:<40} ║
        ║ Budget: €{self.target_budget:>15,.2f}                     ║
        ╚═══════════════════════════════════════════════════════╝
        """)
        
        # Generate new composition with same parameters
        result = recommender.generate_gift_recommendations(
            partner_id=self.partner_id.id,
            target_budget=self.target_budget,
            client_notes=self.client_notes,
            dietary_restrictions=dietary,
            composition_type=self.composition_type or 'custom'
        )
        
        if result.get('success'):
            new_composition = self.env['gift.composition'].browse(result['composition_id'])
            
            # Link to old composition if field exists
            if hasattr(new_composition, 'regenerated_from_id'):
                new_composition.regenerated_from_id = self.id
            
            # Add note about regeneration
            new_composition.internal_notes = f"Regenerated from {self.reference}\n{new_composition.internal_notes or ''}"
            
            # Show success message
            return {
                'type': 'ir.actions.act_window',
                'name': f'Regenerated Composition',
                'res_model': 'gift.composition',
                'res_id': new_composition.id,
                'view_mode': 'form',
                'target': 'current',
                'context': {
                    'show_notification': True,
                    'notification_message': f'Successfully regenerated from {self.reference}'
                }
            }
        else:
            raise UserError(f"Regeneration failed: {result.get('error', 'Unknown error')}")
    
    def get_categorized_products(self):
        """Get products organized by category - useful for reports"""
        self.ensure_one()
        
        categories = {}
        
        if self.composition_type == 'experience' and self.experience_name:
            # For experiences, show as single category
            categories[self.experience_name] = self.product_ids
        else:
            # Show by product categories
            if self.beverage_product_ids:
                categories['Beverages'] = self.beverage_product_ids
            if self.aperitif_product_ids:
                categories['Aperitifs'] = self.aperitif_product_ids
            if self.foie_product_ids:
                categories['Foie Gras'] = self.foie_product_ids
            if self.canned_product_ids:
                categories['Canned Foods'] = self.canned_product_ids
            if self.charcuterie_product_ids:
                categories['Charcuterie'] = self.charcuterie_product_ids
            if self.sweet_product_ids:
                categories['Sweets'] = self.sweet_product_ids
            
            # Add any uncategorized products
            categorized_ids = set()
            for products in categories.values():
                categorized_ids.update(products.ids)
            
            uncategorized = self.product_ids.filtered(lambda p: p.id not in categorized_ids)
            if uncategorized:
                categories['Other'] = uncategorized
        
        return categories
    
    def auto_categorize_products(self):
        """Enhanced auto-categorization based on product attributes and category hierarchy"""
        self.ensure_one()
        
        # Define comprehensive keyword mappings
        category_keywords = {
            'beverage': {
                'keywords': ['cava', 'champagne', 'wine', 'vino', 'rosé', 'tinto', 'blanco', 
                            'rioja', 'ribera', 'verdejo', 'albariño', 'tempranillo'],
                'categories': ['BEBIDA/CAVA', 'BEBIDA/CHAMPAGNE', 'BEBIDA/VINO']
            },
            'aperitif': {
                'keywords': ['vermouth', 'vermut', 'tokaji', 'beer', 'cerveza', 'whisky', 
                            'gin', 'vodka', 'brandy', 'cognac', 'licor'],
                'categories': ['BEBIDA/ALCOHOL ALTA GRADUACIÓN', 'BEBIDA/VERMUT']
            },
            'foie': {
                'keywords': ['foie', 'pato', 'oca', 'duck', 'goose', 'micuit', 'mi-cuit'],
                'categories': ['COMIDA/FOIE']
            },
            'canned': {
                'keywords': ['conserva', 'lata', 'anchoa', 'atún', 'bonito', 'sardina',
                            'mejillón', 'berberecho', 'ventresca', 'melva'],
                'categories': ['COMIDA/CONSERVAS', 'COMIDA/PESCADO']
            },
            'charcuterie': {
                'keywords': ['jamón', 'chorizo', 'salchichón', 'lomo', 'cecina', 'queso',
                            'cheese', 'paletilla', 'ibérico', 'bellota', 'cebo'],
                'categories': ['COMIDA/IBERICOS', 'COMIDA/QUESOS']
            },
            'sweet': {
                'keywords': ['turrón', 'chocolate', 'dulce', 'sweet', 'trufa', 'lingote',
                            'bombón', 'mazapán', 'polvorón', 'mantecado'],
                'categories': ['COMIDA/DULCES', 'COMIDA/TURRON']
            }
        }
        
        # Clear existing categorizations
        categorized = {
            'beverage': [],
            'aperitif': [],
            'foie': [],
            'canned': [],
            'charcuterie': [],
            'sweet': []
        }
        
        for product in self.product_ids:
            name_lower = product.name.lower() if product.name else ''
            categ_name = product.categ_id.complete_name.lower() if product.categ_id else ''
            
            categorized_flag = False
            
            # Check each category
            for cat_type, cat_data in category_keywords.items():
                # Check category path first (most reliable)
                for cat_path in cat_data['categories']:
                    if cat_path.lower() in categ_name:
                        categorized[cat_type].append(product.id)
                        categorized_flag = True
                        break
                
                # If not categorized yet, check keywords
                if not categorized_flag:
                    for keyword in cat_data['keywords']:
                        if keyword in name_lower:
                            categorized[cat_type].append(product.id)
                            categorized_flag = True
                            break
                
                if categorized_flag:
                    break
        
        # Update the categorized fields
        self.write({
            'beverage_product_ids': [(6, 0, categorized['beverage'])],
            'aperitif_product_ids': [(6, 0, categorized['aperitif'])],
            'foie_product_ids': [(6, 0, categorized['foie'])],
            'canned_product_ids': [(6, 0, categorized['canned'])],
            'charcuterie_product_ids': [(6, 0, categorized['charcuterie'])],
            'sweet_product_ids': [(6, 0, categorized['sweet'])],
        })
        
        return True
    
    @api.model
    def get_experience_compositions(self):
        """Get all experience-based compositions"""
        return self.search([('composition_type', '=', 'experience')])
    
    @api.model
    def get_custom_compositions(self):
        """Get all custom compositions"""
        return self.search([('composition_type', '=', 'custom')])
    
    @api.model
    def get_compositions_by_dietary(self, dietary_type):
        """Get compositions by dietary restriction type"""
        if dietary_type == 'halal':
            return self.search([
                '|',
                ('dietary_restriction_type', '=', 'halal'),
                ('dietary_restrictions', 'ilike', 'halal')
            ])
        elif dietary_type == 'vegan':
            return self.search([
                '|',
                ('dietary_restriction_type', '=', 'vegan'),
                ('dietary_restrictions', 'ilike', 'vegan')
            ])
        # Add more as needed
        return self.browse()