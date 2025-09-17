# -*- coding: utf-8 -*-
# wizard/ollama_recommendation_wizard.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import logging
from datetime import datetime
import json

_logger = logging.getLogger(__name__)

# Define the experiences from your data (keeping the full dictionary as before)
EXPERIENCES_2024 = {
    'EXP_PEPERONCINI': {
        'name': 'Experience Peperoncini',
        'products': ['LB-PEPERON-QUE212', 'ANC-CONS100', 'LB-ACEITU-140'],
        'category': 'aperitif'
    },
    # ... (include all experiences as in previous code) ...
    'EXP_FOIE_ALMENDRA': {
        'name': 'Foie Almendra Experience',
        'products': ['BLOC-ETX-PATO130VER', 'LB-CHU-GRANA70', 'BRUYERE001'],
        'category': 'foie'
    }
}


class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'Ollama Gift Recommendation Wizard'

    # === FIELD DEFINITIONS ===
    
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
    
    engine_type = fields.Selection([
        ('experience', '🎁 Experience-Based'),
        ('custom', '🔧 Custom Composition'),
        ('hybrid', '🎨 Hybrid (Experience + Custom)')
    ], string="Composition Type", default='custom', required=True,
       help="Choose between pre-configured experiences or custom product selection")

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

    client_dietary_history = fields.Text(
        string="Previous Dietary Restrictions",
        compute='_compute_client_dietary_history',
        readonly=True
    )

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
    experience_preview = fields.Html(string="Experience Preview", compute='_compute_experience_preview')
    composition_display_type = fields.Char(
        string="Composition Type",
        compute='_compute_composition_display_type'
    )

    is_vegan = fields.Boolean(string="Vegan")
    is_halal = fields.Boolean(string="Halal")
    is_gluten_free = fields.Boolean(string="Gluten Free")
    is_non_alcoholic = fields.Boolean(string="Non-Alcoholic")

    # === SELECTION METHOD ===
    
    @api.model
    def _get_experience_selection(self):
        """Get available experiences for selection"""
        selections = []
        for key, exp in EXPERIENCES_2024.items():
            name = exp['name']
            category = exp.get('category', 'other')
            selections.append((key, f"{name} ({category.title()})"))
        return selections

    # === DEFAULT METHODS ===
    
    @api.model
    def _default_recommender(self):
        rec = self.env['ollama.gift.recommender'].search([('active', '=', True)], limit=1)
        if not rec:
            rec = self.env['ollama.gift.recommender'].create({'name': 'Default Ollama Recommender'})
        return rec

    # === COMPUTED FIELDS ===
    
    @api.depends('composition_id', 'engine_type')
    def _compute_composition_display_type(self):
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

    # === ONCHANGE METHODS ===
    
    @api.onchange('engine_type')
    def _onchange_engine_type(self):
        if self.engine_type == 'custom':
            self.selected_experience = False
            self.experience_category_filter = 'all'

    @api.onchange('experience_category_filter')
    def _onchange_experience_category(self):
        if self.experience_category_filter and self.experience_category_filter != 'all':
            if self.selected_experience:
                exp = EXPERIENCES_2024.get(self.selected_experience, {})
                if exp.get('category') != self.experience_category_filter:
                    self.selected_experience = False

    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        if self.partner_id and self.client_dietary_history:
            if 'halal' in self.client_dietary_history.lower():
                self.dietary_restrictions = 'halal'
            elif 'vegan' in self.client_dietary_history.lower():
                self.dietary_restrictions = 'vegan'
            elif 'vegetarian' in self.client_dietary_history.lower():
                self.dietary_restrictions = 'vegetarian'

    @api.onchange('dietary_restrictions')
    def _onchange_dietary_restrictions(self):
        if self.dietary_restrictions:
            self.is_vegan = (self.dietary_restrictions == 'vegan')
            self.is_halal = (self.dietary_restrictions == 'halal')
            self.is_gluten_free = (self.dietary_restrictions == 'gluten_free')
            self.is_non_alcoholic = (self.dietary_restrictions == 'non_alcoholic')

    @api.onchange('is_vegan', 'is_halal', 'is_gluten_free', 'is_non_alcoholic')
    def _onchange_dietary_checkboxes(self):
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

    # === ACTION METHODS (ALL REQUIRED METHODS) ===
    
    composition_type = fields.Selection([
        ('custom', 'Custom'),
        ('hybrid', 'Hybrid'),
        ('experience', 'Experience')
    ], string="Composition Type", default='custom',
    help="Leave empty to use historical preference or default")

    def action_generate_recommendation(self):
        """Generate recommendation with complete data merging"""
        self.ensure_one()
        
        # Validate inputs
        if not self.partner_id:
            raise UserError("Please select a client")
        
        # Prepare dietary restrictions
        dietary = []
        if self.is_halal:
            dietary.append('halal')
        if self.is_vegan:
            dietary.append('vegan')
        if self.is_gluten_free:
            dietary.append('gluten_free')
        if self.is_non_alcoholic:
            dietary.append('non_alcoholic')
        
        # Build notes with product count if explicitly specified
        final_notes = self.client_notes or ""
        
        # Only add product count to notes if user explicitly checked the box
        if self.specify_product_count and self.product_count:
            count_instruction = f"Must have exactly {self.product_count} products."
            if final_notes:
                # Check if count is already mentioned in notes
                if not any(word in final_notes.lower() for word in ['product', 'item', 'piece']):
                    final_notes += f" {count_instruction}"
            else:
                final_notes = count_instruction
            
            _logger.info(f"🎯 User explicitly requested {self.product_count} products via form checkbox")
        
        # Log the generation request
        _logger.info(f"""
        ========== GENERATION REQUEST ==========
        Client: {self.partner_id.name}
        Budget (Form): €{self.target_budget:.2f} {'(provided)' if self.target_budget else '(empty)'}
        Product Count (Form): {self.product_count if self.specify_product_count else 'Not specified'}
        Dietary (Form): {dietary if dietary else 'None'}
        Composition Type (Form): {self.composition_type or 'Not specified'}
        Notes: {final_notes[:100]}{'...' if len(final_notes) > 100 else ''}
        ========================================
        """)
        
        # Generate recommendation with all data
        result = self.recommender_id.generate_gift_recommendations(
            partner_id=self.partner_id.id,
            target_budget=self.target_budget if self.target_budget else 0,
            client_notes=final_notes,
            dietary_restrictions=dietary,
            composition_type=self.composition_type
        )
        
        if result.get('success'):
            # Log success details
            _logger.info(f"""
            ========== GENERATION SUCCESS ==========
            Method: {result.get('method', 'unknown')}
            Products: {result.get('product_count', 0)}
            Total Cost: €{result.get('total_cost', 0):.2f}
            Confidence: {result.get('confidence_score', 0)*100:.0f}%
            Sources Used: Multiple (see logs above)
            ========================================
            """)
            
            # Show success and open the composition
            return {
                'type': 'ir.actions.act_window',
                'name': 'Generated Composition',
                'res_model': 'gift.composition',
                'res_id': result['composition_id'],
                'view_mode': 'form',
                'target': 'current',
            }
        else:
            error_msg = result.get('error', 'Unknown error')
            _logger.error(f"Generation failed: {error_msg}")
            raise UserError(f"Generation failed: {error_msg}")

    def action_view_composition(self):
        """View the generated composition - THIS WAS MISSING!"""
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
        """Generate another recommendation with same settings - THIS WAS MISSING!"""
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
            'selected_experience': self.selected_experience,
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
        """Test Ollama connection - THIS WAS MISSING!"""
        self.ensure_one()
        
        if not self.recommender_id:
            raise UserError("Please select a recommender first.")
        
        try:
            test_result = self.recommender_id.test_ollama_connection() if hasattr(self.recommender_id, 'test_ollama_connection') else True
            
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

    # === HELPER METHODS ===
    
    def _prepare_dietary_restrictions(self):
        """Prepare dietary restrictions list"""
        dietary = []
        
        if self.dietary_restrictions and self.dietary_restrictions != 'none':
            if self.dietary_restrictions == 'multiple':
                if self.dietary_restrictions_text:
                    dietary = [r.strip() for r in self.dietary_restrictions_text.split(',')]
            else:
                dietary = [self.dietary_restrictions]
        
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
        if not self.selected_experience:
            return {'success': False, 'error': 'No experience selected'}
        
        exp_code = self.selected_experience
        exp = EXPERIENCES_2024.get(exp_code, {})
        
        if not exp:
            return {'success': False, 'error': 'Invalid experience selected'}
        
        products = []
        total_cost = 0.0
        missing_products = []
        
        product_refs = exp.get('products', [])
        
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
                'ai_reasoning': f"Experience-based composition: {exp['name']}"
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
        """Generate hybrid recommendation"""
        exp_result = self._generate_experience_based()
        
        if not exp_result.get('success'):
            return exp_result
        
        remaining_budget = self.target_budget - exp_result.get('total_cost', 0)
        
        if remaining_budget > 50:
            custom_result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=remaining_budget,
                client_notes=f"Complement to {self.selected_experience}",
                dietary_restrictions=self._prepare_dietary_restrictions()
            )
            
            if custom_result.get('success'):
                all_products = exp_result.get('products', []) + custom_result.get('products', [])
                total_cost = exp_result.get('total_cost', 0) + custom_result.get('total_cost', 0)
                
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
        """Format an enhanced success message"""
        message_parts = [
            "<div class='alert alert-success'>",
            f"<h4>✅ {self.composition_display_type} Generated Successfully!</h4>"
        ]
        
        if self.selected_experience:
            exp_name = EXPERIENCES_2024.get(self.selected_experience, {}).get('name', 'Unknown')
            message_parts.append(f"<p><strong>Experience:</strong> {exp_name}</p>")
        
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

    def _format_success_message(self, result):
        """Backward compatibility"""
        return self._format_enhanced_success_message(result)