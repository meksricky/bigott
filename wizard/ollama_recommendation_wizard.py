# -*- coding: utf-8 -*-
# wizard/ollama_recommendation_wizard.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import logging
from datetime import datetime
import json

from wizard.experience_option import ExperienceOption

_logger = logging.getLogger(__name__)

# Define the experiences from your data
EXPERIENCES_2024 = {
    'EXP_PEPERONCINI': {
        'name': 'Experience Peperoncini',
        'products': ['LB-PEPERON-QUE212', 'ANC-CONS100', 'LB-ACEITU-140'],
        'category': 'aperitif'
    },
    'EXP_APERITIVO_ESFERIFICADO': {
        'name': 'Aperitivo Esferificado Experience',
        'products': ['LB-MOU-BONPIPARRA120', 'ESF-CAV-NE60', 'LB-PIM-CARA110'],
        'category': 'aperitif'
    },
    'EXP_GILDA_ESFERIFICADA': {
        'name': 'Gilda Esferificada Experience',
        'products': ['ANC-TOJA-ORO75', 'ESF-CAV-VER60', 'MOU-ZUB-PIPA100'],
        'category': 'aperitif'
    },
    'EXP_GILDA': {
        'name': 'Classic Gilda Experience',
        'products': ['ANC-CONS100', 'LB-ACEITU-140', 'GUIN-AGIN212'],
        'category': 'aperitif'
    },
    'EXP_GILDA_RELLENA': {
        'name': 'Gilda Rellena Experience',
        'products': ['PEP-ATUN-EMP314', 'LB-ACEITU-140', 'LB-CHU-PIMAMA95'],
        'category': 'aperitif'
    },
    'EXP_APERITIVO': {
        'name': 'Aperitivo Experience',
        'products': ['LB-MOU-BONPIPARRA120', 'ANC-CONS100', 'REGAÑA-CARBON'],
        'category': 'aperitif'
    },
    'EXP_HUERTA': {
        'name': 'Experience de la Huerta',
        'products': ['BON-ARRAIN212', 'LB-ENSA-NORM250', 'SAL-AÑA-ECO250'],
        'category': 'seafood',
        'variants': [
            ['BON-ARRAIN212', 'LB-ENSA-NORM250', 'SAL-AÑA-ECO250', 'ACEI-ARAN250', 'VIN-VIND250']
        ]
    },
    'EXP_PIMIENTO_RELLENO': {
        'name': 'Pimiento Relleno Experience',
        'products': ['PIQ-ABA250-DO-TAPAANCHA', 'BON-ARRO185', 'CREMA-OLAS-ATUN110'],
        'variants': [
            ['PIQ-ABA250-DO-TAPAANCHA', 'BON-ARRO185', 'CREMA-OLAS-ATUN110', 'LB-CHU-PIMAMA95'],
            ['PIQ-ABA250-DO-TAPAANCHA', 'BON-ARRO185', 'CREMA-OLAS-ATUN110', 'ANC-CONS100'],
            ['PIQ-ABA250-DO-TAPAANCHA', 'BON-ARRO185', 'CREMA-OLAS-ATUN110', 'ANC-CONS100', 'LB-CHU-PIMAMA95']
        ],
        'category': 'seafood'
    },
    'EXP_PINTXO_NORTE_BONITO': {
        'name': 'Pintxo del Norte de Bonito',
        'products': ['BON-ARRO185', 'ALCA-ECO-ALI-EMP-212', 'PAN-DANVI-HERB130'],
        'category': 'seafood'
    },
    'EXP_BONITO': {
        'name': 'Bonito Experience',
        'products': ['BON-ARRO185', 'CEBOLLITA-BORETANA', 'CREMA-PIQUILLO'],
        'category': 'seafood'
    },
    'EXP_BONITO_CONFITADO': {
        'name': 'Bonito Confitado Experience',
        'products': ['BON-ARRO185', 'LB-CHU-PIMAMA95', 'ANACAR-ROSAL-LIMA40', 'PAN-CHIP-PAPRI75'],
        'category': 'seafood'
    },
    'EXP_BONITO_TRUFADO': {
        'name': 'Bonito Trufado Experience',
        'products': ['BONITO-TRUFA-EMP', 'ALM-ROSAL-TRUF40', 'LB-ENSA-NORM250'],
        'category': 'seafood'
    },
    'EXP_MAR_MONTAÑA': {
        'name': 'Mar y Montaña Experience',
        'products': ['BON-ARRO185', 'LB-MOU-BONPIPARRA120', 'LB-HONGO-BOL212'],
        'category': 'seafood'
    },
    'EXP_MI_CUIT': {
        'name': 'Experience Mi Cuit',
        'products': ['BON-ORTIZ-MICU190', 'CEBO-CUNA85', 'ACEI-ARB500-NOS'],
        'category': 'seafood'
    },
    'EXP_LUBINA': {
        'name': 'Lubina Experience',
        'products': ['LUB-CV200', 'ENSA-MIX-EMP135', 'LB-CHU-PIMAMA95'],
        'variants': [
            ['MOU-ZUBI-MARA', 'LUB-CV200'],
            ['LUB-CV200', 'BEREN-SECA-EMP135', 'MOU-ZUBI-VER']
        ],
        'category': 'seafood'
    },
    'EXP_DORADA': {
        'name': 'Dorada Experience',
        'products': ['DOR-CV200', 'BEREN-SECA-EMP135', 'MOU-ZUBI-VER'],
        'category': 'seafood'
    },
    'EXP_SALMON': {
        'name': 'Salmon Experience',
        'products': ['SALM-CV200', 'PATE-CV-SALM100', 'PAN-CHIP-ENE75'],
        'variants': [
            ['SALM-CV200', 'SALSA-MOSMIEL212', 'REGAÑA-CARBON'],
            ['SALM-CV200', 'PATE-CV-SALM100', 'SALSA-MOSMIEL212', 'REGAÑA-CARBON']
        ],
        'category': 'seafood'
    },
    'EXP_PINTXO_SALMON': {
        'name': 'Pintxo de Salmón',
        'products': ['SALM-CV200', 'SALSA-MOSMIEL212', 'REGAÑA-CARBON'],
        'variants': [
            ['SALM-CV200', 'PATE-CV-SALM100', 'SALSA-MOSMIEL212', 'REGAÑA-CARBON']
        ],
        'category': 'seafood'
    },
    'EXP_DEL_SELLA': {
        'name': 'Experience del Sella',
        'products': ['SALM-CV200', 'PASTEL-SELLA145'],
        'category': 'seafood'
    },
    'EXP_LOMO_SALMON': {
        'name': 'Lomo Salmón Experience',
        'products': ['SALM-UBA150', 'SALSA-MOSMIEL212'],
        'category': 'seafood',
        'needs_cold': True
    },
    'EXP_TXULETON': {
        'name': 'Txuletón Experience',
        'products': ['PAT-ETX-TXU130', 'QUESO-TORTA-CASAR-70', 'PIQ-ABA250-DO-TAPAANCHA'],
        'category': 'meat'
    },
    'EXP_BACALAO_VIZCAINA': {
        'name': 'Bacalao a la Vizcaína Experience',
        'products': ['BAC-CV200', 'LB-PIM-CARA110', 'LB-SALSA-TOMATE'],
        'category': 'seafood'
    },
    'EXP_PIMIENTOS_RELLENOS_BACALAO': {
        'name': 'Pimientos Rellenos de Bacalao',
        'products': ['BAC-CV200', 'PATE-CV-BAC100', 'PIQ-ABA250-DO-TAPAANCHA'],
        'variants': [
            ['BAC-CV200', 'PATE-CV-BAC100', 'LB-CHU-PIMAMA95', 'PIQ-ABA250-DO-TAPAANCHA']
        ],
        'category': 'seafood'
    },
    'EXP_PINTXO_BACALAO': {
        'name': 'Pintxo de Bacalao Experience',
        'products': ['BAC-CV200', 'MOU-ZUBI-BER', 'REGAÑA-CARBON'],
        'category': 'seafood'
    },
    'EXP_PINTXO_NORTE_BACALAO': {
        'name': 'Pintxo del Norte de Bacalao',
        'products': ['BAC-CV200', 'MOU-ZUB-PIPA100', 'PIQ-ABA250-DO-TAPAANCHA'],
        'category': 'seafood'
    },
    'EXP_BACALAO_BOLETUS': {
        'name': 'Bacalao con Boletus',
        'products': ['BAC-CV200', 'LB-HONGO-BOL212', 'PATE-CV-BAC100', 'PAN-CHIP-CHAMP75'],
        'category': 'seafood'
    },
    'EXP_ESTURION': {
        'name': 'Esturión Experience',
        'products': ['ESTURION-RIO90', 'LB-PIM-ASADO212', 'ACEI-ARB250-NOS'],
        'category': 'seafood'
    },
    'EXP_VEGETARIANA': {
        'name': 'Vegetarian Experience',
        'products': ['LB-SALSA-TOMATE', 'LB-HONGO-BOL212', 'PIQ-ABA250-DO-TAPAANCHA'],
        'variants': [
            ['LB-SALSA-TOMATE', 'LB-HONGO-BOL212', 'PAN-DAVI-CHIVE130']
        ],
        'category': 'vegetarian',
        'dietary': ['vegetarian']
    },
    'EXP_PIQUILLOS_VEGANOS': {
        'name': 'Piquillos Veganos Experience',
        'products': ['PIQ-ABA250-DO-TAPAANCHA', 'HUM-ZUBI100', 'LB-HONGO-BOL212', 'LB-CHU-PIMAMA95'],
        'category': 'vegan',
        'dietary': ['vegan']
    },
    'EXP_VEGANA': {
        'name': 'Vegan Experience',
        'products': ['LB-SALSA-TOMATE', 'ALCA-ECO-ALI-EMP-212', 'LB-HONGO-BOL212'],
        'variants': [
            ['LB-ENSA-NORM250', 'ALCA-ECO-ALI-EMP-212', 'PAN-DAVI-HERB130']
        ],
        'category': 'vegan',
        'dietary': ['vegan']
    },
    'EXP_CHEESECAKE': {
        'name': 'Cheesecake Experience',
        'products': ['TAR-QUE-ETX130', 'CHU-CENE42-FAV', 'SWEET014'],
        'variants': [
            ['TAR-QUE-ETX130', 'CHU-CENE42-FAV'],
            ['TAR-QUE-ETX130', 'CHU-CENE42-FAV', 'DEST010']
        ],
        'category': 'dessert'
    },
    'EXP_MAGRET_PATO': {
        'name': 'Magret de Pato Experience',
        'products': ['JAM-ZUBI-PATO', 'LB-HONGO-BOL212', 'SALSA-MOSMIEL212'],
        'category': 'meat'
    },
    'EXP_FOIE_CARAMELO': {
        'name': 'Foie Caramelo Experience',
        'products': ['BLOC-ETX-PATO130VER', 'LB-CHU-GRANA70', 'SWEET014'],
        'category': 'foie'
    },
    'EXP_FOIE_ALMENDRA': {
        'name': 'Foie Almendra Experience',
        'products': ['BLOC-ETX-PATO130VER', 'LB-CHU-GRANA70', 'BRUYERE001'],
        'category': 'foie'
    }
}


class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'Ollama Gift Recommendation Wizard'

    # --- Currency for Monetary fields ---
    currency_id = fields.Many2one(
        'res.currency',
        default=lambda self: self.env.company.currency_id,
        required=True
    )

    partner_id = fields.Many2one(
        'res.partner',
        string="Client",
        required=True,
        default=lambda self: self._context.get('default_partner_id')
    )

    target_budget = fields.Monetary(
        string="Target Budget",
        required=True,
        default=100.0,
        currency_field='currency_id'
    )
    
    target_year = fields.Integer(
        string="Target Year",
        default=lambda self: datetime.now().year,
        required=True
    )

    client_notes = fields.Text(string="Client Notes")
    
    # Enhanced dietary restrictions with dropdown selection
    dietary_restrictions = fields.Selection([
        ('none', 'No Restrictions'),
        ('vegan', 'Vegan'),
        ('vegetarian', 'Vegetarian'),
        ('halal', 'Halal'),
        ('gluten_free', 'Gluten Free'),
        ('non_alcoholic', 'Non-Alcoholic'),
        ('multiple', 'Multiple Restrictions')
    ], string="Dietary Restrictions", default='none')
    
    dietary_restrictions_text = fields.Text(
        string="Additional Dietary Details",
        help="Specify multiple restrictions or additional dietary requirements"
    )
    
    # NEW FIELDS for enhanced functionality
    engine_type = fields.Selection([
        ('experience', '🎁 Experience-Based'),
        ('custom', '🔧 Custom Composition'),
        ('hybrid', '🎨 Hybrid (Experience + Custom)')
    ], string="Composition Type", default='custom', required=True,
       help="Choose between pre-configured experiences or custom product selection")

    # Experience selection using Many2one to transient model
    experience_option_ids = fields.One2many(
        'experience.option',
        'wizard_id',
        string="Available Experiences"
    )
    
    selected_experience_id = fields.Many2one(
        'experience.option',
        string="Select Experience",
        domain="[('id', 'in', experience_option_ids)]",
        help="Choose a pre-configured experience package"
    )
    
    experience_category_filter = fields.Selection([
        ('all', 'All Categories'),
        ('aperitif', 'Aperitifs'),
        ('seafood', 'Seafood'),
        ('meat', 'Meat'),
        ('vegetarian', 'Vegetarian'),
        ('vegan', 'Vegan'),
        ('dessert', 'Desserts'),
        ('foie', 'Foie Gras')
    ], string="Experience Category", default='all')

    # Auto-populated from client history
    client_dietary_history = fields.Text(
        string="Previous Dietary Restrictions",
        compute='_compute_client_dietary_history',
        readonly=True
    )

    # Keep existing fields
    recommender_id = fields.Many2one(
        'ollama.gift.recommender',
        string="Recommender",
        default=lambda self: self._default_recommender()
    )
    
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating...'),
        ('done', 'Done'),
        ('error', 'Error')
    ], default='draft', string="State")

    composition_id = fields.Many2one('gift.composition', string="Generated Composition")
    recommended_products = fields.Many2many('product.template', string="Recommended Products")
    total_cost = fields.Monetary(string="Total Cost", currency_field='currency_id')
    confidence_score = fields.Float(string="Confidence Score", digits=(3, 2))
    
    result_message = fields.Html(string="Result", readonly=True)
    error_message = fields.Text(string="Error Message", readonly=True)
    client_info = fields.Html(string="Client Information", compute='_compute_client_info')
    
    # NEW: Experience preview field
    experience_preview = fields.Html(string="Experience Preview", compute='_compute_experience_preview')
    composition_display_type = fields.Char(
        string="Composition Type",
        compute='_compute_composition_display_type'
    )

    # Keep existing checkbox fields for compatibility
    is_vegan = fields.Boolean(string="Vegan")
    is_halal = fields.Boolean(string="Halal")
    is_gluten_free = fields.Boolean(string="Gluten Free")
    is_non_alcoholic = fields.Boolean(string="Non-Alcoholic")

    # ========== MODEL INITIALIZATION ==========
    
    @api.model
    def default_get(self, fields_list):
        """Override default_get to populate experience options"""
        defaults = super().default_get(fields_list)
        
        # Create experience options
        if 'experience_option_ids' in fields_list:
            defaults['experience_option_ids'] = self._prepare_experience_options()
        
        return defaults
    
    def _prepare_experience_options(self):
        """Prepare experience option records"""
        options = []
        for code, exp_data in EXPERIENCES_2024.items():
            # Calculate estimated cost
            products = exp_data.get('products', [])
            estimated_cost = 0.0
            for product_code in products:
                product = self.env['product.template'].search([
                    ('default_code', '=', product_code)
                ], limit=1)
                if product:
                    estimated_cost += product.list_price
            
            # Prepare dietary info
            dietary_info = ', '.join(exp_data.get('dietary', [])) if exp_data.get('dietary') else ''
            
            option_vals = {
                'code': code,
                'name': exp_data['name'],
                'category': exp_data.get('category', 'other'),
                'products': ', '.join(products),
                'products_count': len(products),
                'estimated_cost': estimated_cost,
                'dietary_info': dietary_info,
            }
            options.append((0, 0, option_vals))
        
        return options

    # ========== COMPUTED METHODS ==========
    
    @api.depends('composition_id', 'engine_type')
    def _compute_composition_display_type(self):
        """Display whether this is an Experience or Custom Made composition"""
        for record in self:
            if record.composition_id:
                if record.engine_type == 'experience':
                    record.composition_display_type = "✨ EXPERIENCE"
                elif record.engine_type == 'hybrid':
                    record.composition_display_type = "🎨 HYBRID"
                else:
                    record.composition_display_type = "🔧 CUSTOM MADE"
            else:
                record.composition_display_type = ""

    @api.depends('selected_experience_id')
    def _compute_experience_preview(self):
        """Preview the selected experience details"""
        for record in self:
            if record.selected_experience_id:
                exp_option = record.selected_experience_id
                html = f"""
                <div style="padding: 10px; background: #f8f9fa; border-radius: 8px; margin: 10px 0;">
                    <h4 style="color: #2c3e50; margin-bottom: 10px;">
                        <i class="fa fa-gift"></i> {exp_option.name}
                    </h4>
                    <p><strong>Category:</strong> {exp_option.category.title() if exp_option.category else 'Other'}</p>
                    <p><strong>Estimated Cost:</strong> €{exp_option.estimated_cost:.2f}</p>
                    <p><strong>Number of Products:</strong> {exp_option.products_count}</p>
                """
                
                if exp_option.dietary_info:
                    html += f"<p><strong>Dietary:</strong> {exp_option.dietary_info}</p>"
                
                # Show products
                html += "<p><strong>Products Included:</strong></p><ul>"
                
                for product_ref in exp_option.get_product_list():
                    product = record.env['product.template'].search([
                        ('default_code', '=', product_ref)
                    ], limit=1)
                    if product:
                        html += f"<li>{product.name} (€{product.list_price:.2f})</li>"
                    else:
                        html += f"<li>{product_ref} (Product to be configured)</li>"
                
                html += "</ul></div>"
                record.experience_preview = html
            else:
                record.experience_preview = ""

    @api.depends('partner_id')
    def _compute_client_dietary_history(self):
        """Auto-retrieve dietary restrictions from previous orders"""
        for record in self:
            if record.partner_id:
                previous_comps = self.env['gift.composition'].search([
                    ('partner_id', '=', record.partner_id.id)
                ], order='create_date desc', limit=5)
                
                dietary_history = []
                for comp in previous_comps:
                    if hasattr(comp, 'dietary_restrictions') and comp.dietary_restrictions:
                        dietary_history.append(comp.dietary_restrictions)
                
                if dietary_history:
                    unique_restrictions = list(set(dietary_history))
                    record.client_dietary_history = f"Previously used: {', '.join(unique_restrictions)}"
                else:
                    record.client_dietary_history = "No previous dietary restrictions found"
            else:
                record.client_dietary_history = ""

    @api.depends('partner_id')
    def _compute_client_info(self):
        """Compute comprehensive client information display"""
        for record in self:
            if record.partner_id:
                partner = record.partner_id
                html_parts = []
                
                html_parts.append(f"<h4>{partner.name}</h4>")
                
                if partner.email:
                    html_parts.append(f"<p><i class='fa fa-envelope'></i> {partner.email}</p>")
                if partner.phone:
                    html_parts.append(f"<p><i class='fa fa-phone'></i> {partner.phone}</p>")
                
                if partner.parent_id:
                    html_parts.append(f"<p><strong>Company:</strong> {partner.parent_id.name}</p>")
                
                past_compositions = self.env['gift.composition'].search([
                    ('partner_id', '=', partner.id)
                ], order='create_date desc', limit=3)
                
                if past_compositions:
                    html_parts.append("<h5>Recent Gift History:</h5>")
                    html_parts.append("<ul>")
                    for comp in past_compositions:
                        date_str = comp.create_date.strftime('%Y-%m-%d') if comp.create_date else 'N/A'
                        comp_type = getattr(comp, 'composition_type', 'custom').title()
                        budget = getattr(comp, 'target_budget', 0)
                        product_count = len(comp.product_ids) if hasattr(comp, 'product_ids') else 0
                        html_parts.append(
                            f"<li>{date_str} - €{budget:.2f} - {product_count} products - {comp_type}</li>"
                        )
                    html_parts.append("</ul>")
                
                record.client_info = ''.join(html_parts) if html_parts else '<p>No client information available.</p>'
            else:
                record.client_info = '<p>Please select a client to view their information.</p>'

    # ========== ONCHANGE METHODS ==========
    
    @api.onchange('engine_type')
    def _onchange_engine_type(self):
        """Clear experience selection when switching to custom"""
        if self.engine_type == 'custom':
            self.selected_experience_id = False
            self.experience_category_filter = 'all'

    @api.onchange('experience_category_filter')
    def _onchange_experience_category(self):
        """Filter experiences when category changes"""
        # Update domain for the selected_experience_id field
        domain = []
        if self.experience_category_filter and self.experience_category_filter != 'all':
            domain = [('category', '=', self.experience_category_filter)]
            
            # Clear selection if it doesn't match the new category
            if self.selected_experience_id and self.selected_experience_id.category != self.experience_category_filter:
                self.selected_experience_id = False
        
        return {
            'domain': {
                'selected_experience_id': domain
            }
        }

    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Auto-populate dietary restrictions from history"""
        if self.partner_id and self.client_dietary_history:
            if 'halal' in self.client_dietary_history.lower():
                self.dietary_restrictions = 'halal'
            elif 'vegan' in self.client_dietary_history.lower():
                self.dietary_restrictions = 'vegan'
            elif 'vegetarian' in self.client_dietary_history.lower():
                self.dietary_restrictions = 'vegetarian'

    @api.onchange('dietary_restrictions')
    def _onchange_dietary_restrictions(self):
        """Update checkboxes based on dropdown selection"""
        if self.dietary_restrictions:
            self.is_vegan = (self.dietary_restrictions == 'vegan')
            self.is_halal = (self.dietary_restrictions == 'halal')
            self.is_gluten_free = (self.dietary_restrictions == 'gluten_free')
            self.is_non_alcoholic = (self.dietary_restrictions == 'non_alcoholic')

    @api.onchange('is_vegan', 'is_halal', 'is_gluten_free', 'is_non_alcoholic')
    def _onchange_dietary_checkboxes(self):
        """Keep existing implementation for backward compatibility"""
        toggles = []
        if self.is_vegan:
            toggles.append('vegan')
        if self.is_halal:
            toggles.append('halal')
        if self.is_gluten_free:
            toggles.append('gluten_free')
        if self.is_non_alcoholic:
            toggles.append('non_alcoholic')

        if toggles:
            if len(toggles) > 1:
                self.dietary_restrictions = 'multiple'
                self.dietary_restrictions_text = ', '.join(toggles)
            else:
                self.dietary_restrictions = toggles[0]

    @api.constrains('target_budget')
    def _check_target_budget(self):
        for rec in self:
            if rec.target_budget <= 0:
                raise ValidationError("Target budget must be greater than 0.")
            if rec.target_budget > 1_000_000:
                raise ValidationError("Target budget seems unusually high. Please confirm.")

    # ========== MAIN ACTION METHOD ==========
    
    def action_generate_recommendation(self):
        """Generate gift recommendation based on selected type"""
        self.ensure_one()
        
        # Validate inputs
        if not self.partner_id:
            raise UserError("Please select a client.")
        
        if self.engine_type in ['experience', 'hybrid'] and not self.selected_experience_id:
            raise UserError("Please select an experience for this composition type.")
        
        if not self.recommender_id:
            raise UserError("No recommender available. Please configure an Ollama recommender first.")
        
        _logger.info(f"Generating {self.engine_type} recommendation for {self.partner_id.name}")
        
        try:
            self.write({'state': 'generating'})
            
            # Prepare dietary restrictions
            dietary = self._prepare_dietary_restrictions()
            
            # Generate based on type
            if self.engine_type == 'experience':
                result = self._generate_experience_based()
            elif self.engine_type == 'hybrid':
                result = self._generate_hybrid()
            else:
                # Use existing method for custom
                result = self.recommender_id.generate_gift_recommendations(
                    partner_id=self.partner_id.id,
                    target_budget=self.target_budget,
                    client_notes=self.client_notes or '',
                    dietary_restrictions=dietary
                )
            
            # Process result
            if result.get('success'):
                # Update composition with type information
                if result.get('composition_id'):
                    composition = self.env['gift.composition'].browse(result['composition_id'])
                    exp_code = self.selected_experience_id.code if self.selected_experience_id else False
                    exp_name = self.selected_experience_id.name if self.selected_experience_id else False
                    
                    composition.write({
                        'composition_type': self.engine_type,
                        'experience_code': exp_code,
                        'experience_name': exp_name,
                        'experience_category': self.selected_experience_id.category if self.selected_experience_id else False,
                    })
                
                self.write({
                    'state': 'done',
                    'result_message': self._format_enhanced_success_message(result),
                    'composition_id': result.get('composition_id'),
                    'recommended_products': [(6, 0, [p.id for p in result.get('products', [])])],
                    'total_cost': result.get('total_cost', 0.0),
                    'confidence_score': result.get('confidence_score', 0.0)
                })
                
                return {
                    'type': 'ir.actions.act_window',
                    'name': 'Recommendation Results',
                    'res_model': 'ollama.recommendation.wizard',
                    'res_id': self.id,
                    'view_mode': 'form',
                    'target': 'new',
                    'context': dict(self._context, show_results=True),
                }
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                self.write({
                    'state': 'error',
                    'error_message': error_msg,
                    'result_message': f"<div class='alert alert-danger'>{error_msg}</div>"
                })
                raise UserError(f"Recommendation failed: {error_msg}")
                
        except Exception as e:
            self.write({
                'state': 'error',
                'error_message': str(e)
            })
            _logger.exception(f"Recommendation wizard error: {e}")
            raise

    # ========== GENERATION METHODS ==========
    
    def _prepare_dietary_restrictions(self):
        """Prepare dietary restrictions list"""
        dietary = []
        
        if self.dietary_restrictions and self.dietary_restrictions != 'none':
            if self.dietary_restrictions == 'multiple':
                if self.dietary_restrictions_text:
                    dietary = [r.strip() for r in self.dietary_restrictions_text.split(',')]
            else:
                dietary = [self.dietary_restrictions]
        
        # Also check checkboxes for backward compatibility
        if self.is_vegan and 'vegan' not in dietary:
            dietary.append('vegan')
        if self.is_halal and 'halal' not in dietary:
            dietary.append('halal')
        if self.is_gluten_free and 'gluten_free' not in dietary:
            dietary.append('gluten_free')
        if self.is_non_alcoholic and 'non_alcoholic' not in dietary:
            dietary.append('non_alcoholic')
        
        return dietary

    def _generate_experience_based(self):
        """Generate recommendation using selected experience"""
        if not self.selected_experience_id:
            return {'success': False, 'error': 'No experience selected'}
        
        exp_option = self.selected_experience_id
        exp_code = exp_option.code
        
        # Get full experience data
        exp = EXPERIENCES_2024.get(exp_code, {})
        if not exp:
            return {'success': False, 'error': 'Invalid experience selected'}
        
        products = []
        total_cost = 0.0
        missing_products = []
        
        # Get products for the experience
        product_refs = exp.get('products', [])
        
        # Check for variants
        if 'variants' in exp and exp['variants']:
            # Select variant based on budget
            best_variant = None
            best_cost_diff = float('inf')
            
            for variant in exp['variants']:
                variant_cost = 0
                variant_products = []
                
                for ref in variant:
                    product = self.env['product.template'].search([
                        ('default_code', '=', ref)
                    ], limit=1)
                    if product:
                        variant_products.append(product)
                        variant_cost += product.list_price
                
                cost_diff = abs(variant_cost - self.target_budget)
                if cost_diff < best_cost_diff:
                    best_cost_diff = cost_diff
                    best_variant = variant_products
                    total_cost = variant_cost
            
            if best_variant:
                products = best_variant
        else:
            # Use standard products
            for product_ref in product_refs:
                product = self.env['product.template'].search([
                    ('default_code', '=', product_ref)
                ], limit=1)
                
                if product:
                    products.append(product)
                    total_cost += product.list_price
                else:
                    missing_products.append(product_ref)
        
        if missing_products:
            _logger.warning(f"Missing products for experience {exp_code}: {missing_products}")
        
        # Create composition
        try:
            composition = self.env['gift.composition'].create({
                'partner_id': self.partner_id.id,
                'target_budget': self.target_budget,
                'target_year': self.target_year,
                'product_ids': [(6, 0, [p.id for p in products])],
                'dietary_restrictions': ', '.join(self._prepare_dietary_restrictions()),
                'client_notes': self.client_notes,
                'generation_method': 'experience',
                'composition_type': 'experience',
                'experience_code': exp_code,
                'experience_name': exp['name'],
                'experience_category': exp.get('category', 'other'),
                'confidence_score': 0.95,
                'ai_reasoning': f"Experience-based composition: {exp['name']}"  # FIXED: Using ai_reasoning
            })
            
            return {
                'success': True,
                'composition_id': composition.id,
                'products': products,
                'total_cost': total_cost,
                'product_count': len(products),
                'confidence_score': 0.95,
                'message': f'Successfully generated {exp["name"]} experience'
            }
        except Exception as e:
            _logger.error(f"Failed to create experience composition: {e}")
            return {
                'success': False,
                'error': f'Failed to create composition: {str(e)}'
            }

    def _generate_hybrid(self):
        """Generate hybrid recommendation (experience + custom products)"""
        # First get the experience products
        exp_result = self._generate_experience_based()
        
        if not exp_result.get('success'):
            return exp_result
        
        # Calculate remaining budget
        remaining_budget = self.target_budget - exp_result.get('total_cost', 0)
        
        if remaining_budget > 50:  # Only add custom if significant budget remains
            # Get additional custom products
            custom_result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=remaining_budget,
                client_notes=f"Complement to {self.selected_experience_id.name}",
                dietary_restrictions=self._prepare_dietary_restrictions()
            )
            
            if custom_result.get('success'):
                # Merge products
                all_products = exp_result.get('products', []) + custom_result.get('products', [])
                total_cost = exp_result.get('total_cost', 0) + custom_result.get('total_cost', 0)
                
                # Update composition
                composition = self.env['gift.composition'].browse(exp_result['composition_id'])
                composition.write({
                    'product_ids': [(6, 0, [p.id for p in all_products])],
                    'composition_type': 'hybrid',
                    'ai_reasoning': f"Hybrid: {exp_result.get('message')} + custom products"
                })
                
                return {
                    'success': True,
                    'composition_id': composition.id,
                    'products': all_products,
                    'total_cost': total_cost,
                    'product_count': len(all_products),
                    'confidence_score': 0.85,
                    'message': 'Successfully generated hybrid composition'
                }
        
        return exp_result

    def _format_enhanced_success_message(self, result):
        """Format an enhanced success message with categories"""
        message_parts = [
            "<div class='alert alert-success'>",
            f"<h4>✅ {self.composition_display_type} Generated Successfully!</h4>"
        ]
        
        if self.selected_experience_id:
            message_parts.append(f"<p><strong>Experience:</strong> {self.selected_experience_id.name}</p>")
        
        message_parts.extend([
            f"<p><strong>Products Selected:</strong> {result.get('product_count', 0)}</p>",
            f"<p><strong>Total Cost:</strong> €{result.get('total_cost', 0):.2f}</p>",
            f"<p><strong>Target Budget:</strong> €{self.target_budget:.2f}</p>",
            f"<p><strong>Budget Variance:</strong> {((result.get('total_cost', 0) - self.target_budget) / self.target_budget * 100):.1f}%</p>"
        ])
        
        if result.get('confidence_score'):
            message_parts.append(f"<p><strong>Confidence Score:</strong> {result.get('confidence_score', 0) * 100:.0f}%</p>")
        
        message_parts.append("</div>")
        return '\n'.join(message_parts)

    # ========== SUPPORTING METHODS ==========
    
    @api.model
    def _default_recommender(self):
        rec = self.env['ollama.gift.recommender'].search([('active', '=', True)], limit=1)
        if not rec:
            rec = self.env['ollama.gift.recommender'].create({'name': 'Default Ollama Recommender'})
        return rec

    def action_view_composition(self):
        """View the generated composition"""
        self.ensure_one()
        if not self.composition_id:
            raise UserError("No composition generated yet.")
        
        return {
            'type': 'ir.actions.act_window',
            'name': f'Gift Composition for {self.partner_id.name}',
            'res_model': 'gift.composition',
            'res_id': self.composition_id.id,
            'view_mode': 'form',
            'target': 'current',
        }

    def action_generate_another(self):
        """Generate another recommendation with same settings"""
        self.ensure_one()
        
        new_wizard = self.create({
            'currency_id': self.currency_id.id,
            'partner_id': self.partner_id.id,
            'target_budget': self.target_budget,
            'target_year': self.target_year,
            'client_notes': self.client_notes,
            'dietary_restrictions': self.dietary_restrictions,
            'dietary_restrictions_text': self.dietary_restrictions_text,
            'is_vegan': self.is_vegan,
            'is_halal': self.is_halal,
            'is_gluten_free': self.is_gluten_free,
            'is_non_alcoholic': self.is_non_alcoholic,
            'recommender_id': self.recommender_id.id,
            'engine_type': self.engine_type,
            'experience_category_filter': self.experience_category_filter,
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Generate Another Recommendation',
            'res_model': 'ollama.recommendation.wizard',
            'res_id': new_wizard.id,
            'view_mode': 'form',
            'target': 'new',
        }

    def action_test_connection(self):
        """Test Ollama connection"""
        self.ensure_one()
        
        if not self.recommender_id:
            raise UserError("Please select a recommender first.")
        
        try:
            # Test the connection
            test_result = self.recommender_id.test_ollama_connection()
            
            if test_result:
                return {
                    'type': 'ir.actions.client',
                    'tag': 'display_notification',
                    'params': {
                        'title': 'Success',
                        'message': 'Ollama connection successful!',
                        'type': 'success',
                        'sticky': False,
                    }
                }
            else:
                raise UserError("Could not connect to Ollama. Please check your configuration.")
        except Exception as e:
            raise UserError(f"Connection test failed: {str(e)}")

    def _format_success_message(self, result):
        """Keep for backward compatibility"""
        return self._format_enhanced_success_message(result)


# Add the transient model reference field
ExperienceOption._fields['wizard_id'] = fields.Many2one('ollama.recommendation.wizard', string='Wizard')