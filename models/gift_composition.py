# models/gift_composition.py
from odoo import models, fields, api
from datetime import datetime
import logging

_logger = logging.getLogger(__name__)

class GiftComposition(models.Model):
    _name = 'gift.composition'
    _description = 'Gift Composition'
    _inherit = ['mail.thread', 'mail.activity.mixin']  # ADD THIS LINE for chatter support
    _order = 'create_date desc'
    _rec_name = 'display_name'

    # Keep all existing fields
    name = fields.Char(string='Reference', default='New', copy=False, tracking=True)
    display_name = fields.Char(string='Display Name', compute='_compute_display_name', store=True)
    
    partner_id = fields.Many2one('res.partner', string='Client', required=True, ondelete='cascade', tracking=True)
    partner_email = fields.Char(related='partner_id.email', string='Email', readonly=True)
    partner_phone = fields.Char(related='partner_id.phone', string='Phone', readonly=True)
    
    # NEW FIELDS for Experience Support
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
    
    # Budget Information (keep existing)
    currency_id = fields.Many2one('res.currency', default=lambda self: self.env.company.currency_id)
    target_budget = fields.Monetary(string='Target Budget', currency_field='currency_id', required=True, tracking=True)
    actual_cost = fields.Monetary(string='Actual Cost', currency_field='currency_id', compute='_compute_actual_cost', store=True)
    budget_variance = fields.Float(string='Budget Variance %', compute='_compute_budget_variance', store=True)
    
    target_year = fields.Integer(string='Target Year', default=lambda self: datetime.now().year, required=True)
    
    # Products - Main field (keep existing)
    product_ids = fields.Many2many('product.template', string='All Products')
    
    # NEW: Categorized Products for better organization
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
    
    # Dietary and Notes (keep existing + enhance)
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
    
    # AI Information (keep existing)
    ai_reasoning = fields.Text(string='AI Reasoning')
    confidence_score = fields.Float(string='Confidence Score', digits=(3, 2))
    generation_method = fields.Selection([
        ('ollama', 'Ollama AI'),
        ('fallback', 'Rule-Based'),
        ('manual', 'Manual'),
        ('experience', 'Experience-Based'),
        ('hybrid', 'Hybrid')
    ], string='Generation Method', default='manual')
    
    # State (keep existing)
    state = fields.Selection([
        ('draft', 'Draft'),
        ('confirmed', 'Confirmed'),
        ('delivered', 'Delivered'),
        ('cancelled', 'Cancelled')
    ], string='State', default='draft', tracking=True)
    
    # Statistics
    product_count = fields.Integer(string='Product Count', compute='_compute_product_count', store=True)
    
    # NEW: Category distribution
    category_distribution = fields.Text(string='Category Distribution', compute='_compute_category_distribution')
    
    # Rest of the methods remain the same...
    
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
            
            # Add reference
            if record.name and record.name != 'New':
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
                
                # Generate sequence - simpler approach
                vals['name'] = self.env['ir.sequence'].next_by_code('gift.composition') or f'{prefix}/001'
        
        return super().create(vals_list)
    
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
        """Open wizard to regenerate the composition"""
        self.ensure_one()
        
        # Prepare dietary restrictions for wizard
        dietary = 'none'
        if self.dietary_restriction_type:
            dietary = self.dietary_restriction_type
        elif self.dietary_restrictions:
            # Try to parse from text
            if 'halal' in self.dietary_restrictions.lower():
                dietary = 'halal'
            elif 'vegan' in self.dietary_restrictions.lower():
                dietary = 'vegan'
            elif 'vegetarian' in self.dietary_restrictions.lower():
                dietary = 'vegetarian'
        
        wizard_vals = {
            'partner_id': self.partner_id.id,
            'target_budget': self.target_budget,
            'target_year': self.target_year,
            'dietary_restrictions': dietary,
            'dietary_restrictions_text': self.dietary_restrictions,
            'client_notes': self.client_notes,
            'engine_type': self.composition_type if self.composition_type != 'custom' else 'custom',
        }
        
        # Only add experience code if it exists
        if self.experience_code:
            wizard_vals['selected_experience'] = self.experience_code
        
        wizard = self.env['ollama.recommendation.wizard'].create(wizard_vals)
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Regenerate Composition',
            'res_model': 'ollama.recommendation.wizard',
            'res_id': wizard.id,
            'view_mode': 'form',
            'target': 'new',
        }
    
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
        """Automatically categorize products based on their categories or names"""
        self.ensure_one()
        
        beverages = []
        aperitifs = []
        foie = []
        canned = []
        charcuterie = []
        sweets = []
        
        for product in self.product_ids:
            name_lower = product.name.lower()
            categ_name = product.categ_id.name.lower() if product.categ_id else ''
            
            # Check categories based on keywords
            if any(word in name_lower or word in categ_name for word in 
                   ['cava', 'champagne', 'wine', 'vino', 'rosé', 'tinto', 'blanco']):
                beverages.append(product.id)
            elif any(word in name_lower or word in categ_name for word in 
                     ['vermouth', 'vermut', 'tokaji', 'beer', 'cerveza', 'alcohol', 'whisky', 'gin']):
                aperitifs.append(product.id)
            elif any(word in name_lower or word in categ_name for word in 
                     ['foie', 'pato', 'oca', 'duck', 'goose']):
                foie.append(product.id)
            elif any(word in name_lower or word in categ_name for word in 
                     ['conserva', 'canned', 'lata', 'anchoa', 'atún', 'bonito']):
                canned.append(product.id)
            elif any(word in name_lower or word in categ_name for word in 
                     ['jamón', 'chorizo', 'salchichón', 'queso', 'cheese', 'paletilla', 'ibérico']):
                charcuterie.append(product.id)
            elif any(word in name_lower or word in categ_name for word in 
                     ['turrón', 'chocolate', 'dulce', 'sweet', 'trufa', 'lingote']):
                sweets.append(product.id)
        
        # Update categorized fields
        if beverages:
            self.beverage_product_ids = [(6, 0, beverages)]
        if aperitifs:
            self.aperitif_product_ids = [(6, 0, aperitifs)]
        if foie:
            self.foie_product_ids = [(6, 0, foie)]
        if canned:
            self.canned_product_ids = [(6, 0, canned)]
        if charcuterie:
            self.charcuterie_product_ids = [(6, 0, charcuterie)]
        if sweets:
            self.sweet_product_ids = [(6, 0, sweets)]
        
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