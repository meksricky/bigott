from odoo import models, fields, api, _
from odoo.exceptions import UserError
import logging

_logger = logging.getLogger(__name__)

class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'Ollama Gift Recommendation Wizard'
    
    # Complete EXPERIENCIAS_DATA from EXPERIENCIAS 2025
    EXPERIENCES_DATA = {
        'X-EXP-HALAL': {
            'name': 'HALAL',
            'products': ['TOKAJ-MUSCAT', 'PATO-130', 'BACALAO-200', 'ASADA-250', 'PIMIENTO-95', 'BLANCOS-4FRUTOS', 'ACEITE-VIRGEN'],
            'category': 'halal',
            'dietary': ['halal'],
            'price': 72.60,
            'description': 'Complete halal-compliant premium selection from IDOM'
        },
        'X-EXP-BACALAO': {
            'name': 'Experiencia Gastronómica de Bacalao Personalizada Idom',
            'products': ['BACALAO-200', 'ACEITE-OLIVA-200', 'ASADA-CARBON-250', 'GELEE-PIMIENTO-95'],
            'category': 'seafood',
            'price': 64.48,
            'description': 'Premium cod experience with artisanal accompaniments'
        },
        'X-EXP-VEGETARIANA': {
            'name': 'Experiencia Gastronómica Vegetariana Personalizada Idom',
            'products': ['ALCACHOFA-180', 'BERENJENA-90', 'HIERBAS-PROVENZA-180', 'TORTA-CASAR-100', 'PERLAS-CHOCOLATE', 'COOKIES-CHOCOLATE', 'PASTAS-VEGANAS'],
            'category': 'vegetarian',
            'dietary': ['vegetarian'],
            'price': 64.48,
            'description': 'Gourmet vegetarian selection'
        },
        'X-EXP-CHEESECAKE': {
            'name': 'Experiencia Gastronómica Cheesecake',
            'products': ['CHEESECAKE-130', 'GRANADA-70', 'WAFFLE-200', 'CAJA-CARTON'],
            'category': 'dessert',
            'price': 45.00,
            'description': 'Artisanal cheesecake experience'
        },
        'X-EXP-GILDA': {
            'name': 'EXP- GILDA',
            'products': ['ANCHOA-CONSORCIO', 'LB-ACEITU-140', 'GUIN-AGIN212'],
            'category': 'aperitif',
            'price': 0.46,
            'description': 'Traditional Basque gilda experience'
        },
        'X-EXP-LUBINA': {
            'name': 'EXP LUBINA',
            'products': ['LUB-CV200', 'ENSA-MIX-EMP135', 'LB-CHU-PIMAMA95'],
            'category': 'seafood',
            'price': 0.46,
            'description': 'Premium sea bass experience'
        },
        'X-EXP-BON-TOMATE': {
            'name': 'EXP BONITO',
            'products': ['BONITO', 'LB-SALSA-TOMATE', 'LB-HONGO-BOL212'],
            'category': 'seafood',
            'price': 0.46,
            'description': 'Bonito tuna with tomato and boletus'
        }
    }

    # ================== STATE MANAGEMENT (FROM ORIGINAL) ==================
    
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating'),
        ('done', 'Done'),
        ('error', 'Error')
    ], string='State', default='draft')
    
    # ================== CLIENT & BUDGET FIELDS (FROM ORIGINAL) ==================
    
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
        help="Leave at 0 to auto-detect from history. The system will use notes > form > history."
    )
    
    target_year = fields.Integer(
        string='Target Year',
        default=lambda self: fields.Date.today().year
    )
    
    # ================== COMPOSITION SETTINGS (FROM ORIGINAL) ==================
    
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
    
    # ================== PRODUCT COUNT SETTINGS (FROM ORIGINAL) ==================
    
    specify_product_count = fields.Boolean(
        string="Specify Exact Product Count",
        default=False,
        help="Check to enforce exact product count. This will be strictly enforced."
    )
    
    product_count = fields.Integer(
        string="Number of Products",
        default=12,
        help="Exact number of products to include. Will be STRICTLY enforced if checkbox is checked."
    )
    
    # ================== EXPERIENCE FIELDS (FROM ORIGINAL) ==================
    
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
    
    # ================== DIETARY RESTRICTIONS (FROM ORIGINAL) ==================
    
    dietary_restrictions = fields.Selection([
        ('none', 'None'),
        ('halal', 'Halal'),
        ('vegan', 'Vegan'),
        ('vegetarian', 'Vegetarian'),
        ('gluten_free', 'Gluten Free'),
        ('non_alcoholic', 'Non-Alcoholic'),
        ('multiple', 'Multiple Restrictions')
    ], string='Dietary Restrictions', default='none')
    
    dietary_restrictions_text = fields.Text(
        string='Additional Dietary Details',
        help="Enter additional restrictions separated by commas (e.g., 'no nuts, lactose-free')"
    )
    
    # Individual dietary checkboxes (for fine control)
    is_halal = fields.Boolean(string='Halal')
    is_vegan = fields.Boolean(string='Vegan')
    is_vegetarian = fields.Boolean(string='Vegetarian')
    is_gluten_free = fields.Boolean(string='Gluten Free')
    is_non_alcoholic = fields.Boolean(string='Non-Alcoholic')
    
    # ================== CLIENT NOTES & HISTORY (FROM ORIGINAL) ==================
    
    client_notes = fields.Text(
        string='Client Notes & Preferences',
        placeholder="You can specify:\n"
                   "• Budget (overrides form value)\n"
                   "• Product count (e.g., '23 products')\n"
                   "• Categories (e.g., 'include 3 wines, 2 cheeses')\n"
                   "• Special requests\n"
                   "• Exclusions (e.g., 'no chocolate')",
        help="These notes are intelligently parsed and take PRECEDENCE over form values"
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
        string="Client History & Recommendations",
        compute='_compute_client_history',
        readonly=True
    )
    
    # ================== BUSINESS RULES AWARENESS (FROM ORIGINAL) ==================
    
    has_previous_orders = fields.Boolean(
        compute='_compute_has_previous_orders'
    )
    
    has_last_year_data = fields.Boolean(
        compute='_compute_has_last_year_data'
    )
    
    business_rules_applicable = fields.Boolean(
        string='Business Rules Applicable',
        compute='_compute_business_rules_applicable'
    )
    
    expected_strategy = fields.Char(
        string='Expected Strategy',
        compute='_compute_expected_strategy'
    )
    
    # ================== RESULTS (FROM ORIGINAL) ==================
    
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
    
    product_count_actual = fields.Integer(
        string='Actual Product Count',
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
    
    # ================== RECOMMENDER SETTINGS (FROM ORIGINAL) ==================
    
    recommender_id = fields.Many2one(
        'ollama.gift.recommender',
        string='Recommender Engine',
        default=lambda self: self.env['ollama.gift.recommender'].get_or_create_recommender()
    )
    
    ollama_status = fields.Char(
        string='Ollama Status',
        compute='_compute_ollama_status'
    )
    
    # ================== NEW FIELDS REQUIRED BY VIEW ==================
    
    # Additional notes field (renamed from client_notes for the view)
    additional_notes = fields.Text(
        string='Additional Notes',
        related='client_notes',  # Link to existing client_notes field
        help="Any special notes for this recommendation"
    )
    
    # Force composition type (for view compatibility)
    force_composition_type = fields.Selection([
        ('auto', '🤖 AI Auto-Select'),
        ('custom', '🎨 Custom Mix'),
        ('hybrid', '🍷 Wine Focus'),
        ('experience', '🎭 Experience-Based')
    ], string='Composition Type', default='auto',
       help="Let AI decide or force a specific type")
    
    # Partner dietary restrictions (computed from partner)
    partner_dietary_restrictions = fields.Char(
        string="Client's Dietary Profile",
        compute='_compute_partner_dietary'
    )
    
    # AI recommendation fields for the view
    budget_recommendation = fields.Char(
        string='Budget Strategy',
        compute='_compute_ai_recommendations'
    )
    
    approach_recommendation = fields.Char(
        string='Recommended Approach',
        compute='_compute_ai_recommendations'
    )
    
    risk_level = fields.Selection([
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'),
        ('high', 'High Risk')
    ], string='Risk Level', compute='_compute_ai_recommendations')
    
    # ================== ALL COMPUTE METHODS (FROM ORIGINAL + NEW) ==================
    
    @api.depends('recommender_id')
    def _compute_ollama_status(self):
        for wizard in self:
            if wizard.recommender_id:
                if wizard.recommender_id.ollama_enabled:
                    wizard.ollama_status = '🟢 Ollama Enabled (Advanced parsing active)'
                else:
                    wizard.ollama_status = '🟡 Ollama Disabled (Using basic parsing)'
            else:
                wizard.ollama_status = '🔴 No recommender configured'
    
    @api.depends('partner_id')
    def _compute_has_previous_orders(self):
        for wizard in self:
            if not wizard.partner_id:
                wizard.has_previous_orders = False
            else:
                orders = self.env['sale.order'].search([
                    ('partner_id', '=', wizard.partner_id.id),
                    ('state', 'in', ['sale', 'done'])
                ], limit=1)
                wizard.has_previous_orders = bool(orders)
    
    @api.depends('partner_id', 'recommender_id')
    def _compute_has_last_year_data(self):
        for wizard in self:
            if not wizard.partner_id or not wizard.recommender_id:
                wizard.has_last_year_data = False
            else:
                last_products = wizard.recommender_id._get_last_year_products(wizard.partner_id.id)
                wizard.has_last_year_data = bool(last_products)
    
    @api.depends('has_last_year_data', 'has_previous_orders')
    def _compute_business_rules_applicable(self):
        for wizard in self:
            wizard.business_rules_applicable = wizard.has_last_year_data
    
    @api.depends('business_rules_applicable', 'has_previous_orders', 'client_notes')
    def _compute_expected_strategy(self):
        for wizard in self:
            notes_lower = wizard.client_notes.lower() if wizard.client_notes else ""
            
            if 'all new' in notes_lower or 'completely different' in notes_lower:
                wizard.expected_strategy = '🆕 Fresh Generation (requested in notes)'
            elif wizard.business_rules_applicable:
                wizard.expected_strategy = '📋 Business Rules + 80/20 Rule'
            elif wizard.has_previous_orders:
                wizard.expected_strategy = '📊 Pattern-Based Generation'
            else:
                wizard.expected_strategy = '👥 Similar Clients Analysis'
    
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
            
            orders = self.env['sale.order'].search([
                ('partner_id', '=', partner.id),
                ('state', 'in', ['sale', 'done'])
            ])
            
            total_orders = len(orders)
            total_value = sum(orders.mapped('amount_untaxed'))
            avg_value = total_value / total_orders if total_orders > 0 else 0
            
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
    
    @api.depends('partner_id', 'recommender_id')
    def _compute_client_history(self):
        """Compute comprehensive client history with patterns and recommendations"""
        for wizard in self:
            if not wizard.partner_id or not wizard.recommender_id:
                wizard.client_history_summary = '<p style="color: #666;">Select a client to see history</p>'
                continue
            
            try:
                patterns = wizard.recommender_id._analyze_client_purchase_patterns(wizard.partner_id.id)
                last_products = wizard.recommender_id._get_last_year_products(wizard.partner_id.id)
                
                if not patterns or patterns.get('total_orders', 0) == 0:
                    wizard.client_history_summary = '''
                    <div style="background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;">
                        <h4 style="margin-top: 0; color: #856404;">📋 New Client - No Purchase History</h4>
                        <p style="margin-bottom: 0; color: #856404;">
                            Will use similar clients analysis or fresh generation.
                        </p>
                    </div>
                    '''
                else:
                    favorites_count = len(patterns.get('favorite_products', []))
                    top_categories = list(patterns.get('preferred_categories', {}).keys())[:3]
                    
                    suggested_budget = patterns.get('avg_order_value', 1000)
                    trend = patterns.get('budget_trend', 'stable')
                    if trend == 'increasing':
                        suggested_budget *= 1.1
                        trend_icon = '📈'
                    elif trend == 'decreasing':
                        suggested_budget *= 0.95
                        trend_icon = '📉'
                    else:
                        trend_icon = '➡️'
                    
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
                                <td>{trend_icon} {trend.upper()}</td>
                            </tr>
                            <tr>
                                <td><b>Favorite Products:</b></td>
                                <td>{favorites_count} recurring items</td>
                            </tr>
                            <tr>
                                <td><b>Top Categories:</b></td>
                                <td>{', '.join(top_categories) if top_categories else 'Various'}</td>
                            </tr>
                    '''
                    
                    if patterns.get('preferred_price_range'):
                        price_range = patterns['preferred_price_range']
                        html += f'''
                            <tr>
                                <td><b>Price Range:</b></td>
                                <td>€{price_range.get('min', 0):.0f} - €{price_range.get('max', 0):.0f}</td>
                            </tr>
                        '''
                    
                    html += f'''
                        </table>
                        <hr style="border-color: #c3e6cb;">
                        <p style="margin-bottom: 10px; color: #155724;">
                            <b>💡 AI Recommendations:</b><br>
                            • Suggested Budget: <b>€{suggested_budget:.0f}</b><br>
                            • Suggested Products: <b>{patterns.get('avg_product_count', 12):.0f}</b> items
                        </p>
                    '''
                    
                    if last_products:
                        html += f'''
                        <div style="background: #c3e6cb; padding: 10px; border-radius: 3px; margin-top: 10px;">
                            <b>🔧 Business Rules Ready:</b><br>
                            Found {len(last_products)} products from last year.<br>
                            Rules R1-R6 will be applied with 80/20 transformation.
                        </div>
                        '''
                    
                    html += '</div>'
                    wizard.client_history_summary = html
                    
            except Exception as e:
                _logger.error(f"Error computing client history: {e}")
                wizard.client_history_summary = f'<p style="color: red;">Error loading history: {str(e)}</p>'
    
    @api.depends('recommended_products')
    def _compute_totals(self):
        for wizard in self:
            wizard.total_cost = sum(wizard.recommended_products.mapped('list_price'))
            wizard.product_count_actual = len(wizard.recommended_products)
    
    # ================== NEW COMPUTED FIELDS FOR VIEW ==================
    
    @api.depends('partner_id')
    def _compute_partner_dietary(self):
        """Get partner's dietary restrictions from history"""
        for wizard in self:
            if not wizard.partner_id:
                wizard.partner_dietary_restrictions = 'No client selected'
            else:
                # Check last compositions for dietary patterns
                compositions = self.env['gift.composition'].search([
                    ('partner_id', '=', wizard.partner_id.id)
                ], limit=3, order='create_date desc')
                
                if compositions:
                    dietary_list = []
                    for comp in compositions:
                        if comp.dietary_restrictions:
                            dietary_list.append(comp.dietary_restrictions)
                    
                    if dietary_list:
                        wizard.partner_dietary_restrictions = f"History: {', '.join(set(dietary_list))}"
                    else:
                        wizard.partner_dietary_restrictions = "No dietary restrictions in history"
                else:
                    wizard.partner_dietary_restrictions = "New client - no history available"
    
    @api.depends('partner_id', 'has_previous_orders', 'has_last_year_data', 'target_budget')
    def _compute_ai_recommendations(self):
        """Compute AI recommendations based on client analysis"""
        for wizard in self:
            if not wizard.partner_id:
                wizard.budget_recommendation = "Select a client for AI analysis"
                wizard.approach_recommendation = "Awaiting client selection"
                wizard.risk_level = 'high'
            else:
                # Analyze patterns if recommender available
                if wizard.recommender_id:
                    patterns = wizard.recommender_id._analyze_client_purchase_patterns(wizard.partner_id.id)
                    
                    # Budget recommendation
                    if patterns and patterns.get('avg_order_value'):
                        avg = patterns['avg_order_value']
                        trend = patterns.get('budget_trend', 'stable')
                        
                        if trend == 'increasing':
                            suggested = avg * 1.1
                            wizard.budget_recommendation = f"Suggest €{suggested:.0f} (10% increase from €{avg:.0f} avg)"
                        elif trend == 'decreasing':
                            suggested = avg * 0.95
                            wizard.budget_recommendation = f"Suggest €{suggested:.0f} (5% decrease from €{avg:.0f} avg)"
                        else:
                            wizard.budget_recommendation = f"Maintain at €{avg:.0f} (stable history)"
                    else:
                        # For new clients, use similar clients analysis
                        similar_avg = 1000.0  # Default
                        wizard.budget_recommendation = f"Using default €{wizard.target_budget or similar_avg:.0f} (no history)"
                    
                    # Approach recommendation - MORE DETAILED
                    if wizard.has_last_year_data:
                        wizard.approach_recommendation = "Apply Business Rules R1-R6 with 80/20 transformation"
                    elif wizard.has_previous_orders:
                        wizard.approach_recommendation = "Use pattern-based generation from purchase history"
                    else:
                        # Check for similar clients
                        wizard.approach_recommendation = "Fresh generation using similar clients analysis"
                    
                    # Risk level
                    if wizard.has_last_year_data:
                        wizard.risk_level = 'low'
                    elif wizard.has_previous_orders:
                        wizard.risk_level = 'medium'
                    else:
                        wizard.risk_level = 'high'
                else:
                    wizard.budget_recommendation = f"Target: €{wizard.target_budget:.0f}"
                    wizard.approach_recommendation = "Standard generation"
                    wizard.risk_level = 'medium'
    
    # ================== ONCHANGE METHODS (FROM ORIGINAL) ==================
    
    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Auto-fill suggestions based on client history"""
        if self.partner_id and self.recommender_id:
            try:
                patterns = self.recommender_id._analyze_client_purchase_patterns(self.partner_id.id)
                
                if patterns and patterns.get('total_orders', 0) > 0:
                    suggested_budget = patterns.get('avg_order_value', 0)
                    if patterns.get('budget_trend') == 'increasing':
                        suggested_budget *= 1.1
                    elif patterns.get('budget_trend') == 'decreasing':
                        suggested_budget *= 0.95
                    
                    if not self.target_budget or self.target_budget == 1000.0:
                        self.target_budget = suggested_budget
                    
                    if patterns.get('avg_product_count'):
                        self.product_count = int(round(patterns['avg_product_count']))
            except:
                pass
    
    @api.onchange('dietary_restrictions')
    def _onchange_dietary_restrictions(self):
        """Update individual checkboxes based on selection"""
        if self.dietary_restrictions == 'halal':
            self.is_halal = True
            self.is_non_alcoholic = True
            self.is_vegan = False
            self.is_vegetarian = False
            self.is_gluten_free = False
        elif self.dietary_restrictions == 'vegan':
            self.is_vegan = True
            self.is_halal = False
            self.is_vegetarian = False
            self.is_gluten_free = False
            self.is_non_alcoholic = False
        elif self.dietary_restrictions == 'vegetarian':
            self.is_vegetarian = True
            self.is_halal = False
            self.is_vegan = False
            self.is_gluten_free = False
            self.is_non_alcoholic = False
        elif self.dietary_restrictions == 'gluten_free':
            self.is_gluten_free = True
            self.is_halal = False
            self.is_vegan = False
            self.is_vegetarian = False
            self.is_non_alcoholic = False
        elif self.dietary_restrictions == 'non_alcoholic':
            self.is_non_alcoholic = True
            self.is_halal = False
            self.is_vegan = False
            self.is_vegetarian = False
            self.is_gluten_free = False
        elif self.dietary_restrictions == 'none':
            self.is_halal = False
            self.is_vegan = False
            self.is_vegetarian = False
            self.is_gluten_free = False
            self.is_non_alcoholic = False
    
    @api.onchange('engine_type')
    def _onchange_engine_type(self):
        """Update composition type to match engine type"""
        if self.engine_type:
            self.composition_type = self.engine_type
    
    @api.onchange('experience_category_filter')
    def _onchange_experience_category_filter(self):
        """Filter experiences based on category"""
        if self.experience_category_filter and self.experience_category_filter != 'all':
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
    
    @api.onchange('force_composition_type')
    def _onchange_force_composition_type(self):
        """Update composition type based on forced selection"""
        if self.force_composition_type and self.force_composition_type != 'auto':
            self.composition_type = self.force_composition_type
            self.engine_type = self.force_composition_type
    
    # ================== ACTION METHODS (FROM ORIGINAL - KEEPING EXACT SAME) ==================
    
    def action_generate_recommendation(self):
        """Generate recommendation with smart validation"""
        self.ensure_one()
        
        if not self.partner_id:
            raise UserError("Please select a client")
        
        self.state = 'generating'
        
        try:
            dietary = self._prepare_dietary_restrictions()
            final_notes = self._prepare_final_notes()
            
            # CRITICAL: Ensure budget is passed correctly
            actual_budget = self.target_budget if self.target_budget > 0 else 1000.0
            
            _logger.info(f"""
            ========================================
            🎁 GIFT RECOMMENDATION GENERATION
            ========================================
            Client: {self.partner_id.name}
            Budget: €{actual_budget:.2f}
            Range (±5%): €{actual_budget*0.95:.2f} - €{actual_budget*1.05:.2f}
            Products: {self.product_count if self.specify_product_count else 'Auto (12)'}
            Dietary: {dietary if dietary else 'None'}
            Type: {self.composition_type}
            Force Type: {self.force_composition_type}
            ========================================
            """)
            
            # Add budget enforcement to notes if not already there
            if actual_budget > 0 and str(actual_budget) not in final_notes:
                final_notes = f"Target budget is €{actual_budget:.0f}. {final_notes}"
            
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=actual_budget,  # Ensure budget is always passed
                client_notes=final_notes,
                dietary_restrictions=dietary,
                composition_type=self.composition_type if self.force_composition_type == 'auto' else self.force_composition_type
            )
            
            if result.get('success'):
                composition_id = result.get('composition_id')
                composition = self.env['gift.composition'].browse(composition_id)
                
                actual_cost = composition.actual_cost or sum(p.list_price for p in composition.product_ids)
                
                if self.target_budget > 0:
                    variance = (actual_cost - self.target_budget) / self.target_budget * 100
                    
                    _logger.info(f"""
                    ========================================
                    ✅ GENERATION SUCCESSFUL
                    ========================================
                    Composition ID: {composition.id}
                    Products: {len(composition.product_ids)}
                    Total Cost: €{actual_cost:.2f}
                    Target: €{self.target_budget:.2f}
                    Variance: {variance:+.1f}%
                    Status: {'✅ IN RANGE' if abs(variance) <= 5 else '⚠️ OUTSIDE ±5%'}
                    ========================================
                    """)
                    
                    for i, product in enumerate(composition.product_ids[:12], 1):
                        _logger.info(f"  {i}. {product.name[:40]}: €{product.list_price:.2f}")
                    
                    if self.target_budget >= 1000:
                        min_appropriate_price = 20.0
                    elif self.target_budget >= 500:
                        min_appropriate_price = 15.0
                    elif self.target_budget >= 200:
                        min_appropriate_price = 10.0
                    else:
                        min_appropriate_price = 5.0
                    
                    low_price_products = [p for p in composition.product_ids if p.list_price < min_appropriate_price]
                    
                    if low_price_products:
                        _logger.warning(f"""
                        ⚠️ Found {len(low_price_products)} products below €{min_appropriate_price:.2f}:
                        {', '.join([f'{p.name[:20]} (€{p.list_price:.2f})' for p in low_price_products[:3]])}
                        These should be excluded in future generations for €{self.target_budget:.2f} budgets.
                        """)
                    
                    if abs(variance) > 10:
                        _logger.warning(f"⚠️ High variance: {variance:+.1f}%. Consider adjusting product selection logic.")
                
                self._process_success_result(result)
                
                return {
                    'type': 'ir.actions.act_window',
                    'name': 'Generated Gift Composition',
                    'res_model': 'gift.composition',
                    'res_id': composition_id,
                    'view_mode': 'form',
                    'target': 'current',
                }
            else:
                self._process_error_result(result)
                
        except Exception as e:
            _logger.error(f"❌ Generation failed: {str(e)}")
            self.state = 'error'
            self.error_message = str(e)
            raise
    
    # Alternative name for view compatibility
    def action_generate_composition(self):
        """Alias for action_generate_recommendation for view compatibility"""
        return self.action_generate_recommendation()
    
    def _prepare_dietary_restrictions(self):
        """Prepare comprehensive dietary restrictions list"""
        dietary = []
        
        if self.dietary_restrictions == 'halal':
            dietary.extend(['halal', 'no_pork', 'no_alcohol', 'no_iberian'])
        elif self.dietary_restrictions == 'vegan':
            dietary.append('vegan')
        elif self.dietary_restrictions == 'vegetarian':
            dietary.append('vegetarian')
        elif self.dietary_restrictions == 'gluten_free':
            dietary.append('gluten_free')
        elif self.dietary_restrictions == 'non_alcoholic':
            dietary.append('non_alcoholic')
        elif self.dietary_restrictions == 'multiple' and self.dietary_restrictions_text:
            restrictions = self.dietary_restrictions_text.split(',')
            dietary.extend([r.strip().lower() for r in restrictions])
        
        if self.is_halal and 'halal' not in dietary:
            dietary.extend(['halal', 'no_pork', 'no_alcohol', 'no_iberian'])
        if self.is_vegan and 'vegan' not in dietary:
            dietary.append('vegan')
        if self.is_vegetarian and 'vegetarian' not in dietary:
            dietary.append('vegetarian')
        if self.is_gluten_free and 'gluten_free' not in dietary:
            dietary.append('gluten_free')
        if self.is_non_alcoholic and 'non_alcoholic' not in dietary:
            dietary.append('non_alcoholic')
        
        seen = set()
        unique_dietary = []
        for item in dietary:
            if item not in seen:
                seen.add(item)
                unique_dietary.append(item)
        
        return unique_dietary
    
    def _prepare_final_notes(self):
        """Prepare final notes that will be parsed by Ollama"""
        notes_parts = []
        
        # CRITICAL: Always include budget in notes for enforcement
        if self.target_budget > 0:
            notes_parts.append(f"IMPORTANT: The total budget MUST be approximately €{self.target_budget:.0f} (±10%)")
            notes_parts.append(f"Select products to reach a total between €{self.target_budget*0.9:.0f} and €{self.target_budget*1.1:.0f}")
        
        if self.client_notes:
            notes_parts.append(self.client_notes)
        
        if self.specify_product_count and self.product_count:
            count_instruction = f"Include EXACTLY {self.product_count} products."
            notes_parts.append(count_instruction)
            _logger.info(f"🎯 Adding product count to notes: {self.product_count}")
        else:
            # Default product count for budget
            if self.target_budget >= 2000:
                notes_parts.append("Include 15-20 products for this premium budget")
            elif self.target_budget >= 1000:
                notes_parts.append("Include 12-15 products for this budget")
            elif self.target_budget >= 500:
                notes_parts.append("Include 8-12 products for this budget")
            else:
                notes_parts.append("Include 5-8 products for this budget")
        
        if self.composition_type == 'hybrid':
            notes_parts.append("Focus on premium wines (40-50% of budget) with complementary gourmet products")
        elif self.composition_type == 'experience':
            notes_parts.append("Include experience-based products or vouchers")
            
            if self.selected_experience:
                notes_parts.append(f"Include this experience: {self.selected_experience.name}")
        
        # Add product price guidance based on budget
        if self.target_budget >= 1000:
            notes_parts.append("Select premium products, minimum €30 per item, with some flagship items over €100")
        elif self.target_budget >= 500:
            notes_parts.append("Select quality products between €20-80 per item")
        elif self.target_budget >= 200:
            notes_parts.append("Select products between €15-50 per item")
        
        final_notes = ". ".join(notes_parts)
        
        _logger.info(f"FINAL NOTES FOR AI: {final_notes[:300]}...")
        
        return final_notes
    
    def _process_success_result(self, result):
        """Process successful generation result"""
        self.state = 'done'
        self.composition_id = result.get('composition_id')
        
        if self.composition_id:
            self.recommended_products = [(6, 0, self.composition_id.product_ids.ids)]
        
        self.confidence_score = result.get('confidence_score', 0)
        
        method = result.get('method', 'unknown')
        method_display = {
            'business_rules_with_enforcement': '📋 Business Rules + Requirements',
            'business_rules_transformation': '📋 Business Rules Applied',
            '8020_rule': '📊 80/20 Rule Applied',
            'pattern_based_enhanced': '🔍 Pattern-Based Generation',
            'similar_clients': '👥 Similar Clients Analysis',
            'universal_enforcement': '🎯 Universal Generation',
            'fresh_generation': '🆕 Fresh Composition'
        }.get(method, method)
        
        actual_count = result.get('product_count', 0)
        actual_cost = result.get('total_cost', 0)
        target_count = self.product_count if self.specify_product_count else None
        
        count_compliance = '✅' if not target_count or actual_count == target_count else '⚠️'
        budget_variance = ((actual_cost - self.target_budget) / self.target_budget * 100) if self.target_budget else 0
        budget_compliance = '✅' if abs(budget_variance) <= 15 else '⚠️'
        
        self.result_message = f"""
        <div style="background: #d4edda; padding: 15px; border-radius: 5px;">
            <h4 style="color: #155724;">✅ Recommendation Generated Successfully!</h4>
            
            <div style="margin-top: 10px;">
                <b>Generation Method:</b> {method_display}<br>
                <b>Confidence Score:</b> {result.get('confidence_score', 0)*100:.0f}%
            </div>
            
            <table style="width: 100%; margin-top: 15px; color: #155724;">
                <tr style="background: #c3e6cb;">
                    <th style="padding: 5px; text-align: left;">Requirement</th>
                    <th style="padding: 5px; text-align: center;">Target</th>
                    <th style="padding: 5px; text-align: center;">Actual</th>
                    <th style="padding: 5px; text-align: center;">Status</th>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Products</b></td>
                    <td style="padding: 5px; text-align: center;">{target_count if target_count else 'Auto'}</td>
                    <td style="padding: 5px; text-align: center;">{actual_count}</td>
                    <td style="padding: 5px; text-align: center;">{count_compliance}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Budget</b></td>
                    <td style="padding: 5px; text-align: center;">€{self.target_budget:.2f}</td>
                    <td style="padding: 5px; text-align: center;">€{actual_cost:.2f}</td>
                    <td style="padding: 5px; text-align: center;">{budget_compliance} ({budget_variance:+.1f}%)</td>
                </tr>
            </table>
        """
        
        if result.get('rules_applied'):
            rules_count = len(result['rules_applied'])
            self.result_message += f"""
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #c3e6cb;">
                <b>Business Rules Applied:</b> {rules_count} transformations
            </div>
            """
        
        if result.get('message'):
            self.result_message += f"""
            <div style="margin-top: 10px; padding: 10px; background: #c3e6cb; border-radius: 3px;">
                <b>Details:</b> {result['message']}
            </div>
            """
        
        self.result_message += "</div>"
        
        _logger.info(f"""
        ========== GENERATION SUCCESS ==========
        Method: {method_display}
        Products: {actual_count} {'✅' if count_compliance == '✅' else '⚠️ (target: ' + str(target_count) + ')'}
        Total Cost: €{actual_cost:.2f}
        Variance: {budget_variance:+.1f}%
        Confidence: {result.get('confidence_score', 0)*100:.0f}%
        Rules Applied: {len(result.get('rules_applied', []))}
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
        
        partner_id = self.partner_id.id
        
        new_wizard = self.create({
            'partner_id': partner_id,
            'target_budget': self.target_budget,
            'target_year': self.target_year,
        })
        
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