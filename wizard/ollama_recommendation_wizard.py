# -*- coding: utf-8 -*-
# wizard/ollama_recommendation_wizard.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import logging
from datetime import datetime
import json

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
        'category': 'seafood'
    },
    'EXP_PIMIENTO_RELLENO': {
        'name': 'Pimiento Relleno Experience',
        'products': ['PIQ-ABA250-DO-TAPAANCHA', 'BON-ARRO185', 'CREMA-OLAS-ATUN110'],
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
        'category': 'seafood'
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
        'category': 'seafood'
    },
    'EXP_PINTXO_BACALAO': {
        'name': 'Pintxo de Bacalao Experience',
        'products': ['BAC-CV200', 'MOU-ZUBI-BER', 'REGAÑA-CARBON'],
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
        'category': 'vegetarian',
        'dietary': ['vegetarian']
    },
    'EXP_VEGANA': {
        'name': 'Vegan Experience',
        'products': ['LB-SALSA-TOMATE', 'ALCA-ECO-ALI-EMP-212', 'LB-HONGO-BOL212'],
        'category': 'vegan',
        'dietary': ['vegan']
    },
    'EXP_CHEESECAKE': {
        'name': 'Cheesecake Experience',
        'products': ['TAR-QUE-ETX130', 'CHU-CENE42-FAV', 'SWEET014'],
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
    
    # Enhanced dietary restrictions
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
    
    # Engine type selection
    engine_type = fields.Selection([
        ('experience', '🎁 Experience-Based'),
        ('custom', '🔧 Custom Composition'),
        ('hybrid', '🎨 Hybrid (Experience + Custom)')
    ], string="Composition Type", default='custom', required=True,
       help="Choose between pre-configured experiences or custom product selection")

    # Experience selection - simple selection field
    selected_experience = fields.Selection(
        selection='_get_experience_selection',
        string="Select Experience",
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

    # Recommender configuration
    recommender_id = fields.Many2one(
        'ollama.gift.recommender',
        string="Recommender",
        default=lambda self: self._default_recommender()
    )
    
    # State management
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating...'),
        ('done', 'Done'),
        ('error', 'Error')
    ], default='draft', string="State")

    # Results
    composition_id = fields.Many2one('gift.composition', string="Generated Composition")
    recommended_products = fields.Many2many('product.template', string="Recommended Products")
    total_cost = fields.Monetary(string="Total Cost", currency_field='currency_id')
    confidence_score = fields.Float(string="Confidence Score", digits=(3, 2))
    
    result_message = fields.Html(string="Result", readonly=True)
    error_message = fields.Text(string="Error Message", readonly=True)
    client_info = fields.Html(string="Client Information", compute='_compute_client_info')
    experience_preview = fields.Html(string="Experience Preview", compute='_compute_experience_preview')
    composition_display_type = fields.Char(
        string="Composition Type",
        compute='_compute_composition_display_type'
    )

    # Legacy checkbox fields for compatibility
    is_vegan = fields.Boolean(string="Vegan")
    is_halal = fields.Boolean(string="Halal")
    is_gluten_free = fields.Boolean(string="Gluten Free")
    is_non_alcoholic = fields.Boolean(string="Non-Alcoholic")

    # ========== SELECTION METHOD ==========
    
    @api.model
    def _get_experience_selection(self):
        """Get available experiences for selection"""
        selections = []
        for key, exp in EXPERIENCES_2024.items():
            name = exp['name']
            category = exp.get('category', 'other')
            selections.append((key, f"{name} ({category.title()})"))
        return selections

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

    @api.depends('selected_experience')
    def _compute_experience_preview(self):
        """Preview the selected experience details"""
        for record in self:
            if record.selected_experience and record.selected_experience in EXPERIENCES_2024:
                exp = EXPERIENCES_2024[record.selected_experience]
                html = f"""
                <div style="padding: 10px; background: #f8f9fa; border-radius: 8px; margin: 10px 0;">
                    <h4 style="color: #2c3e50; margin-bottom: 10px;">
                        <i class="fa fa-gift"></i> {exp['name']}
                    </h4>
                    <p><strong>Category:</strong> {exp.get('category', 'other').title()}</p>
                    <p><strong>Products Included:</strong></p>
                    <ul>
                """
                
                total_cost = 0.0
                for product_ref in exp.get('products', []):
                    product = record.env['product.template'].search([
                        ('default_code', '=', product_ref)
                    ], limit=1)
                    if product:
                        html += f"<li>{product.name} (€{product.list_price:.2f})</li>"
                        total_cost += product.list_price
                    else:
                        html += f"<li>{product_ref} (Product to be configured)</li>"
                
                html += f"""
                    </ul>
                    <p><strong>Estimated Cost:</strong> €{total_cost:.2f}</p>
                    <p><strong>Number of Products:</strong> {len(exp.get('products', []))}</p>
                """
                
                if exp.get('dietary'):
                    html += f"<p><strong>Dietary:</strong> {', '.join(exp['dietary'])}</p>"
                
                html += "</div>"
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

    # ========== ACTION METHODS (rest of your existing methods) ==========
    
    # ... (include all the other methods from your original file:
    #      action_generate_recommendation, _generate_experience_based, 
    #      _generate_hybrid, _prepare_dietary_restrictions, etc.)
    
    # I'm not repeating them all here to save space, but they should all be included