from datetime import datetime
from odoo import models, fields, api, _
from odoo.exceptions import UserError
import logging

_logger = logging.getLogger(__name__)

class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'Ollama Gift Recommendation Wizard'
    
    # State Management
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating'),
        ('done', 'Done'),
        ('error', 'Error')
    ], string='State', default='draft')
    
    # Basic Fields
    partner_id = fields.Many2one(
        'res.partner', 
        string='Client', 
        required=True,
        help="Select the client for whom to generate recommendations"
    )
    
    currency_id = fields.Many2one(
        'res.currency',
        default=lambda self: self.env.company.currency_id
    )
    
    target_budget = fields.Float(
        string='Target Budget',
        default=1000.0,
        help="Leave empty to use historical average"
    )
    
    target_year = fields.Integer(
        string='Target Year',
        default=lambda self: fields.Date.today().year
    )
    
    # Composition Settings
    engine_type = fields.Selection([
        ('custom', 'Custom'),
        ('hybrid', 'Hybrid'),
        ('experience', 'Experience')
    ], string='Engine Type', default='custom')
    
    composition_type = fields.Selection([
        ('custom', 'Custom'),
        ('hybrid', 'Hybrid'),
        ('experience', 'Experience')
    ], string="Composition Type", default='custom',
       help="Custom: Mixed products | Hybrid: Wine-focused | Experience: Activity-focused")
    
    composition_display_type = fields.Char(
        string='Composition Type',
        compute='_compute_composition_display_type'
    )
    
    # Product Count Settings
    specify_product_count = fields.Boolean(
        string="Specify Product Count",
        default=False,
        help="Check to enforce exact product count"
    )
    
    product_count = fields.Integer(
        string="Number of Products",
        default=12,
        help="Exact number of products to include"
    )
    
    # Experience Fields
    experience_category_filter = fields.Selection([
        ('all', 'All Categories'),
        ('gastronomy', 'Gastronomy'),
        ('wellness', 'Wellness & Spa'),
        ('adventure', 'Adventure'),
        ('culture', 'Culture & Arts'),
        ('luxury', 'Luxury Experiences')
    ], string='Experience Category', default='all')
    
    selected_experience = fields.Many2one(
        'product.template',
        string='Selected Experience',
        domain="[('is_experience', '=', True)]"
    )
    
    experience_preview = fields.Html(
        string='Experience Preview',
        compute='_compute_experience_preview'
    )
    
    # Dietary Restrictions
    dietary_restrictions = fields.Selection([
        ('none', 'None'),
        ('halal', 'Halal'),
        ('vegan', 'Vegan'),
        ('gluten_free', 'Gluten Free'),
        ('multiple', 'Multiple Restrictions')
    ], string='Dietary Restrictions', default='none')
    
    dietary_restrictions_text = fields.Char(
        string='Custom Dietary Restrictions',
        help="Enter multiple restrictions separated by commas"
    )
    
    # Legacy dietary fields (for compatibility)
    is_halal = fields.Boolean(string='Halal')
    is_vegan = fields.Boolean(string='Vegan')
    is_gluten_free = fields.Boolean(string='Gluten Free')
    is_non_alcoholic = fields.Boolean(string='Non-Alcoholic')
    
    # Client Information
    client_notes = fields.Text(
        string='Client Notes',
        help="Any special requests or preferences. You can specify composition type, budget, product count here."
    )
    
    client_info = fields.Html(
        string='Client Information',
        compute='_compute_client_info'
    )
    
    client_dietary_history = fields.Char(
        string='Previous Dietary Restrictions',
        compute='_compute_client_dietary_history'
    )
    
    client_history_summary = fields.Html(
        string="Client History",
        compute='_compute_client_history',
        readonly=True
    )
    
    # Results
    composition_id = fields.Many2one(
        'gift.composition',
        string='Generated Composition'
    )
    
    recommended_products = fields.Many2many(
        'product.template',
        string='Recommended Products'
    )
    
    total_cost = fields.Float(
        string='Total Cost',
        compute='_compute_totals'
    )
    
    confidence_score = fields.Float(
        string='Confidence Score',
        default=0.0
    )
    
    result_message = fields.Html(
        string='Result Message'
    )
    
    error_message = fields.Text(
        string='Error Message'
    )
    
    # Recommender Settings
    recommender_id = fields.Many2one(
        'ollama.gift.recommender',
        string='Recommender Engine',
        default=lambda self: self.env['ollama.gift.recommender'].get_or_create_recommender()
    )

    has_last_year_data = fields.Boolean(
        string='Has Last Year Data',
        compute='_compute_history_analysis'
    )
    
    force_business_rules = fields.Boolean(
        string='Apply Business Rules',
        default=False,
        help='Force application of business rules R1-R6 from Master Guide'
    )

    has_previous_orders = fields.Boolean(
        string='Has Previous Orders',
        compute='_compute_history_analysis'
    )
    
    # ================== COMPUTED FIELDS ==================
    
    @api.depends('composition_type', 'engine_type')
    def _compute_composition_display_type(self):
        for wizard in self:
            if wizard.engine_type:
                wizard.composition_display_type = dict(
                    self._fields['engine_type'].selection
                ).get(wizard.engine_type, wizard.engine_type)
            else:
                wizard.composition_display_type = 'Custom'
    
    @api.depends('selected_experience')
    def _compute_experience_preview(self):
        for wizard in self:
            if wizard.selected_experience:
                exp = wizard.selected_experience
                html = f"""
                <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: #f9f9f9;">
                    <h4 style="margin: 0 0 10px 0; color: #333;">{exp.name}</h4>
                    <table style="width: 100%;">
                        <tr>
                            <td style="width: 30%;"><b>Price:</b></td>
                            <td>€{exp.list_price:.2f}</td>
                        </tr>
                        <tr>
                            <td><b>Category:</b></td>
                            <td>{exp.categ_id.name if exp.categ_id else 'N/A'}</td>
                        </tr>
                """
                
                # Add experience-specific fields if they exist
                if hasattr(exp, 'experience_category') and exp.experience_category:
                    html += f"""
                        <tr>
                            <td><b>Experience Type:</b></td>
                            <td>{dict(exp._fields['experience_category'].selection).get(exp.experience_category, exp.experience_category)}</td>
                        </tr>
                    """
                
                if hasattr(exp, 'experience_duration') and exp.experience_duration:
                    html += f"""
                        <tr>
                            <td><b>Duration:</b></td>
                            <td>{exp.experience_duration} hours</td>
                        </tr>
                    """
                
                if hasattr(exp, 'experience_location') and exp.experience_location:
                    html += f"""
                        <tr>
                            <td><b>Location:</b></td>
                            <td>{exp.experience_location}</td>
                        </tr>
                    """
                
                if hasattr(exp, 'max_participants') and exp.max_participants:
                    html += f"""
                        <tr>
                            <td><b>Max Participants:</b></td>
                            <td>{exp.max_participants}</td>
                        </tr>
                    """
                
                html += f"""
                    </table>
                    <div style="margin-top: 10px;">
                        <b>Description:</b><br/>
                        {exp.description_sale or 'No description available'}
                    </div>
                </div>
                """
                
                wizard.experience_preview = html
            else:
                wizard.experience_preview = False
    
    @api.depends('partner_id')
    def _compute_client_info(self):
        for wizard in self:
            if not wizard.partner_id:
                wizard.client_info = '<p>Select a client to see their information</p>'
                continue
            
            partner = wizard.partner_id
            
            # Get order statistics
            orders = self.env['sale.order'].search([
                ('partner_id', '=', partner.id),
                ('state', 'in', ['sale', 'done'])
            ])
            
            total_orders = len(orders)
            total_value = sum(orders.mapped('amount_untaxed'))
            avg_value = total_value / total_orders if total_orders > 0 else 0
            
            # Get last order date
            last_order = orders[0] if orders else None
            last_order_date = last_order.date_order.strftime('%d/%m/%Y') if last_order else 'N/A'
            
            html = f"""
            <div style="padding: 10px; background: #f9f9f9; border-radius: 5px;">
                <h4 style="margin-top: 0;">{partner.name}</h4>
                <table style="width: 100%;">
                    <tr>
                        <td><b>Email:</b></td>
                        <td>{partner.email or 'N/A'}</td>
                    </tr>
                    <tr>
                        <td><b>Phone:</b></td>
                        <td>{partner.phone or partner.mobile or 'N/A'}</td>
                    </tr>
                    <tr>
                        <td><b>Total Orders:</b></td>
                        <td>{total_orders}</td>
                    </tr>
                    <tr>
                        <td><b>Average Order:</b></td>
                        <td>€{avg_value:.2f}</td>
                    </tr>
                    <tr>
                        <td><b>Last Order:</b></td>
                        <td>{last_order_date}</td>
                    </tr>
                </table>
            </div>
            """
            
            wizard.client_info = html
    
    @api.depends('partner_id')
    def _compute_client_dietary_history(self):
        for wizard in self:
            if not wizard.partner_id:
                wizard.client_dietary_history = ''
                continue
            
            # Get previous compositions
            compositions = self.env['gift.composition'].search([
                ('partner_id', '=', wizard.partner_id.id)
            ], limit=5, order='create_date desc')
            
            dietary_set = set()
            for comp in compositions:
                if comp.dietary_restrictions:
                    restrictions = comp.dietary_restrictions.split(',')
                    dietary_set.update([r.strip() for r in restrictions])
            
            if dietary_set:
                wizard.client_dietary_history = f"Previously used: {', '.join(dietary_set)}"
            else:
                wizard.client_dietary_history = "No previous dietary restrictions"
    
    @api.depends('partner_id')
    def _compute_client_history(self):
        """Compute client history summary for display"""
        for wizard in self:
            if not wizard.partner_id:
                wizard.client_history_summary = '<p style="color: #666;">Select a client to see history</p>'
                continue
            
            try:
                recommender = self.env['ollama.gift.recommender'].get_or_create_recommender()
                patterns = recommender._analyze_client_purchase_patterns(wizard.partner_id.id)
                
                if not patterns or patterns.get('total_orders', 0) == 0:
                    wizard.client_history_summary = '''
                    <div style="background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;">
                        <h4 style="margin-top: 0; color: #856404;">📋 No Purchase History</h4>
                        <p style="margin-bottom: 0; color: #856404;">This is a new client with no previous orders.</p>
                    </div>
                    '''
                else:
                    # Build rich history summary
                    favorites_count = len(patterns.get('favorite_products', []))
                    top_categories = list(patterns.get('preferred_categories', {}).keys())[:3]
                    
                    # Determine recommendation
                    suggested_budget = patterns.get('avg_order_value', 1000)
                    if patterns.get('budget_trend') == 'increasing':
                        suggested_budget *= 1.1
                        trend_icon = '📈'
                        trend_text = 'increasing'
                    elif patterns.get('budget_trend') == 'decreasing':
                        suggested_budget *= 0.95
                        trend_icon = '📉'
                        trend_text = 'decreasing'
                    else:
                        trend_icon = '➡️'
                        trend_text = 'stable'
                    
                    html = f'''
                    <div style="background: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745;">
                        <h4 style="margin-top: 0; color: #155724;">📊 Client History Analysis</h4>
                        <table style="width: 100%; color: #155724;">
                            <tr>
                                <td><b>Total Orders:</b></td>
                                <td>{patterns.get('total_orders', 0)}</td>
                            </tr>
                            <tr>
                                <td><b>Avg Order Value:</b></td>
                                <td>€{patterns.get('avg_order_value', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td><b>Avg Products/Order:</b></td>
                                <td>{patterns.get('avg_product_count', 0):.0f} items</td>
                            </tr>
                            <tr>
                                <td><b>Budget Trend:</b></td>
                                <td>{trend_icon} {trend_text.upper()}</td>
                            </tr>
                            <tr>
                                <td><b>Favorite Products:</b></td>
                                <td>{favorites_count} recurring items</td>
                            </tr>
                            <tr>
                                <td><b>Top Categories:</b></td>
                                <td>{', '.join(top_categories) if top_categories else 'Various'}</td>
                            </tr>
                        </table>
                        <hr style="border-color: #c3e6cb;">
                        <p style="margin-bottom: 0; color: #155724;">
                            <b>💡 Suggested Configuration:</b><br>
                            Budget: <b>€{suggested_budget:.0f}</b> | 
                            Products: <b>{patterns.get('avg_product_count', 12):.0f}</b> items
                        </p>
                    </div>
                    '''
                    wizard.client_history_summary = html
            except Exception as e:
                _logger.error(f"Error computing client history: {e}")
                wizard.client_history_summary = f'<p style="color: red;">Error loading history: {str(e)}</p>'
    
    @api.depends('recommended_products')
    def _compute_totals(self):
        for wizard in self:
            wizard.total_cost = sum(wizard.recommended_products.mapped('list_price'))

    @api.depends('partner_id')
    def _compute_history_analysis(self):
        """Compute if partner has last year data"""
        for wizard in self:
            wizard.has_last_year_data = False
            wizard.has_previous_orders = False
            
            if wizard.partner_id:
                # Check for last year's orders
                last_year = datetime.now().year - 1
                last_year_orders = self.env['sale.order'].search([
                    ('partner_id', '=', wizard.partner_id.id),
                    ('date_order', '>=', f'{last_year}-01-01'),
                    ('date_order', '<=', f'{last_year}-12-31'),
                    ('state', 'in', ['sale', 'done'])
                ], limit=1)
                
                wizard.has_last_year_data = bool(last_year_orders)
                
                # Check for any previous orders
                all_orders = self.env['sale.order'].search([
                    ('partner_id', '=', wizard.partner_id.id),
                    ('state', 'in', ['sale', 'done'])
                ], limit=1)
                
                wizard.has_previous_orders = bool(all_orders)
    
    # ================== ONCHANGE METHODS ==================
    
    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Auto-fill suggestions based on client history"""
        if self.partner_id:
            try:
                recommender = self.env['ollama.gift.recommender'].get_or_create_recommender()
                patterns = recommender._analyze_client_purchase_patterns(self.partner_id.id)
                
                if patterns and patterns.get('total_orders', 0) > 0:
                    # Suggest budget based on history
                    suggested_budget = patterns.get('avg_order_value', 0)
                    if patterns.get('budget_trend') == 'increasing':
                        suggested_budget *= 1.1
                    elif patterns.get('budget_trend') == 'decreasing':
                        suggested_budget *= 0.95
                    
                    # Only update if user hasn't manually set a budget
                    if not self.target_budget or self.target_budget == 1000.0:
                        self.target_budget = suggested_budget
                    
                    # Suggest product count
                    if patterns.get('avg_product_count') and not self.specify_product_count:
                        self.product_count = int(round(patterns['avg_product_count']))
            except:
                pass  # Silently fail, don't disrupt the UI
    
    @api.onchange('dietary_restrictions')
    def _onchange_dietary_restrictions(self):
        """Update legacy fields based on selection"""
        if self.dietary_restrictions == 'halal':
            self.is_halal = True
            self.is_vegan = False
            self.is_gluten_free = False
        elif self.dietary_restrictions == 'vegan':
            self.is_vegan = True
            self.is_halal = False
            self.is_gluten_free = False
        elif self.dietary_restrictions == 'gluten_free':
            self.is_gluten_free = True
            self.is_halal = False
            self.is_vegan = False
        elif self.dietary_restrictions == 'none':
            self.is_halal = False
            self.is_vegan = False
            self.is_gluten_free = False
    
    @api.onchange('engine_type')
    def _onchange_engine_type(self):
        """Update composition type to match engine type"""
        if self.engine_type:
            self.composition_type = self.engine_type
    
    @api.onchange('experience_category_filter')
    def _onchange_experience_category_filter(self):
        """Filter experiences based on category"""
        if self.experience_category_filter and self.experience_category_filter != 'all':
            # Update domain for selected_experience field
            return {
                'domain': {
                    'selected_experience': [
                        ('is_experience', '=', True),
                        ('experience_category', '=', self.experience_category_filter)
                    ]
                }
            }
        else:
            return {
                'domain': {
                    'selected_experience': [('is_experience', '=', True)]
                }
            }
    
    # ================== ACTION METHODS ==================
    
    # def action_generate_recommendation(self):
    #     """Generate recommendation with complete data merging"""
    #     self.ensure_one()
        
    #     # Validate inputs
    #     if not self.partner_id:
    #         raise UserError("Please select a client")
        
    #     # Update state
    #     self.state = 'generating'
        
    #     try:
    #         # Prepare dietary restrictions
    #         dietary = self._prepare_dietary_restrictions()
            
    #         # Build notes with all specifications
    #         final_notes = self._prepare_final_notes()
            
    #         # Log the generation request
    #         self._log_generation_request(dietary, final_notes)
            
    #         # Include selected experience if applicable
    #         if self.composition_type == 'experience' and self.selected_experience:
    #             final_notes += f" Include experience: {self.selected_experience.name}"
            
    #         # Generate recommendation with all data
    #         result = self.recommender_id.generate_gift_recommendations(
    #             partner_id=self.partner_id.id,
    #             target_budget=self.target_budget if self.target_budget else 0,
    #             client_notes=final_notes,
    #             dietary_restrictions=dietary,
    #             composition_type=self.composition_type
    #         )
            
    #         if result.get('success'):
    #             self._process_success_result(result)
                
    #             # Show success and open the composition
    #             return {
    #                 'type': 'ir.actions.act_window',
    #                 'name': 'Generated Composition',
    #                 'res_model': 'gift.composition',
    #                 'res_id': result['composition_id'],
    #                 'view_mode': 'form',
    #                 'target': 'current',
    #             }
    #         else:
    #             self._process_error_result(result)
                
    #     except Exception as e:
    #         _logger.error(f"Generation failed with exception: {e}")
    #         self.state = 'error'
    #         self.error_message = str(e)
    #         raise
    
    def action_generate_recommendation(self):
        """Enhanced generation with composition engine option"""
        self.ensure_one()
        
        # Determine which engine to use
        use_composition_engine = (
            self.engine_type == 'experience' or 
            self.force_business_rules or
            self.has_last_year_data
        )
        
        if use_composition_engine:
            # Use the new composition engine
            engine = self.env['gift.composition.engine'].search([('active', '=', True)], limit=1)
            if not engine:
                engine = self.env['gift.composition.engine'].create({
                    'name': 'Master Engine',
                    'ollama_recommender_id': self.recommender_id.id
                })
            
            result = engine.generate_complete_composition(
                partner_id=self.partner_id.id,
                target_budget=self.target_budget,
                client_notes=self.additional_notes,
                dietary_restrictions=self.dietary_restrictions,
                composition_type=self.composition_type,
                wizard_data=self._prepare_wizard_data()
            )
        else:
            # Use existing recommender
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=self.target_budget,
                client_notes=self.additional_notes,
                dietary_restrictions=self.dietary_restrictions
            )
        
        if result.get('success'):
            # Return the composition view
            return {
                'type': 'ir.actions.act_window',
                'res_model': 'gift.composition',
                'res_id': result.get('composition_id'),
                'view_mode': 'form',
                'target': 'current'
            }
    def _prepare_dietary_restrictions(self):
        """Prepare dietary restrictions list"""
        dietary = []
        
        # From selection field
        if self.dietary_restrictions == 'halal':
            dietary.append('halal')
        elif self.dietary_restrictions == 'vegan':
            dietary.append('vegan')
        elif self.dietary_restrictions == 'gluten_free':
            dietary.append('gluten_free')
        elif self.dietary_restrictions == 'multiple' and self.dietary_restrictions_text:
            # Parse multiple restrictions
            restrictions = self.dietary_restrictions_text.split(',')
            dietary.extend([r.strip().lower() for r in restrictions])
        
        # Also check legacy fields
        if self.is_halal and 'halal' not in dietary:
            dietary.append('halal')
        if self.is_vegan and 'vegan' not in dietary:
            dietary.append('vegan')
        if self.is_gluten_free and 'gluten_free' not in dietary:
            dietary.append('gluten_free')
        if self.is_non_alcoholic and 'non_alcoholic' not in dietary:
            dietary.append('non_alcoholic')
        
        return dietary
    
    def _prepare_final_notes(self):
        """Prepare final notes including product count if specified"""
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
        
        return final_notes
    
    def _log_generation_request(self, dietary, final_notes):
        """Log the generation request details"""
        _logger.info(f"""
        ========== GENERATION REQUEST ==========
        Client: {self.partner_id.name}
        Budget (Form): €{self.target_budget:.2f} {'(provided)' if self.target_budget else '(empty)'}
        Product Count (Form): {self.product_count if self.specify_product_count else 'Not specified'}
        Dietary (Form): {dietary if dietary else 'None'}
        Composition Type: {self.composition_type}
        Engine Type: {self.engine_type}
        Experience: {self.selected_experience.name if self.selected_experience else 'None'}
        Notes: {final_notes[:100]}{'...' if len(final_notes) > 100 else ''}
        ========================================
        """)
    
    def _process_success_result(self, result):
        """Process successful generation result"""
        self.state = 'done'
        self.composition_id = result.get('composition_id')
        
        # Get composition products
        if self.composition_id:
            self.recommended_products = [(6, 0, self.composition_id.product_ids.ids)]
        
        self.total_cost = result.get('total_cost', 0)
        self.confidence_score = result.get('confidence_score', 0)
        
        # Build result message
        self.result_message = f"""
        <div style="background: #d4edda; padding: 15px; border-radius: 5px;">
            <h4 style="color: #155724;">✅ Recommendation Generated Successfully!</h4>
            <ul style="color: #155724;">
                <li><b>Method:</b> {result.get('method', 'unknown')}</li>
                <li><b>Products:</b> {result.get('product_count', 0)}</li>
                <li><b>Total Cost:</b> €{result.get('total_cost', 0):.2f}</li>
                <li><b>Confidence:</b> {result.get('confidence_score', 0)*100:.0f}%</li>
                <li><b>Message:</b> {result.get('message', '')}</li>
            </ul>
        </div>
        """
        
        # Log success
        _logger.info(f"""
        ========== GENERATION SUCCESS ==========
        Method: {result.get('method', 'unknown')}
        Products: {result.get('product_count', 0)}
        Total Cost: €{result.get('total_cost', 0):.2f}
        Confidence: {result.get('confidence_score', 0)*100:.0f}%
        ========================================
        """)
    
    def _process_error_result(self, result):
        """Process error result"""
        self.state = 'error'
        error_msg = result.get('error', 'Unknown error')
        self.error_message = error_msg
        _logger.error(f"Generation failed: {error_msg}")
        raise UserError(f"Generation failed: {error_msg}")
    
    def action_generate_another(self):
        """Reset wizard for another generation"""
        self.ensure_one()
        
        # Keep client and basic settings
        partner_id = self.partner_id.id
        
        # Create new wizard with same client
        new_wizard = self.create({
            'partner_id': partner_id,
            'target_budget': self.target_budget,
            'target_year': self.target_year,
        })
        
        # Return action to open new wizard
        return {
            'type': 'ir.actions.act_window',
            'name': 'Generate Another Recommendation',
            'res_model': 'ollama.recommendation.wizard',
            'res_id': new_wizard.id,
            'view_mode': 'form',
            'target': 'new',
        }
    
    def action_view_composition(self):
        """Open the generated composition"""
        self.ensure_one()
        if not self.composition_id:
            raise UserError("No composition to view")
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Gift Composition',
            'res_model': 'gift.composition',
            'res_id': self.composition_id.id,
            'view_mode': 'form',
            'target': 'current',
        }
    
    def action_test_connection(self):
        """Test Ollama connection"""
        self.ensure_one()
        
        result = self.recommender_id.test_ollama_connection()
        
        if result['success']:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Connection Test',
                    'message': result['message'],
                    'type': 'success',
                    'sticky': False,
                }
            }
        else:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Connection Test Failed',
                    'message': result['message'],
                    'type': 'warning',
                    'sticky': True,
                }
            }