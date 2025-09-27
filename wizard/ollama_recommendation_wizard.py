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

    regenerate_composition_id = fields.Many2one(
        'gift.composition',
        string='Composition to Regenerate',
        help='Original composition being regenerated'
    )
    
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
        """Generate recommendation using full AI intelligence with ALL features"""
        self.ensure_one()
        
        if not self.partner_id:
            raise UserError("Please select a client / Por favor seleccione un cliente")
        
        self.state = 'generating'
        
        try:
            # 1. Prepare dietary restrictions comprehensively
            dietary = self._prepare_dietary_restrictions()
            
            # 2. Get the simplified notes (not the verbose context)
            client_notes = self.client_notes if self.client_notes else ""
            
            # 3. Parse notes for budget override FIRST
            budget_to_use = self.target_budget
            if client_notes:
                # Extract budget from notes if specified
                import re
                budget_patterns = [
                    r'budget[:\s]*(?:of\s+)?(?:€|eur)?\s*(\d+)',
                    r'(?:€|eur)\s*(\d+)',
                    r'(\d{4,5})\b(?:\s*(?:€|eur|euros?))?',
                ]
                
                notes_lower = client_notes.lower()
                for pattern in budget_patterns:
                    match = re.search(pattern, notes_lower)
                    if match:
                        try:
                            extracted_budget = float(match.group(1))
                            if 100 <= extracted_budget <= 10000:
                                budget_to_use = extracted_budget
                                _logger.info(f"📝 Budget override from notes: €{extracted_budget}")
                                break
                        except:
                            pass
            
            # 4. If still no budget, use historical average
            if budget_to_use <= 0:
                if self.partner_id and self.recommender_id:
                    patterns = self.recommender_id._analyze_client_purchase_patterns(self.partner_id.id)
                    if patterns and patterns.get('avg_order_value'):
                        budget_to_use = patterns['avg_order_value']
                        _logger.info(f"📊 Using historical average: €{budget_to_use:,.2f}")
                    else:
                        budget_to_use = 1000.0  # Default fallback
                        _logger.info("📊 No history found, using default €1000")
            
            # 5. Determine composition type (could be in notes or form)
            composition_type = self.force_composition_type if self.force_composition_type != 'auto' else self.composition_type
            
            # Check notes for composition type override
            if client_notes:
                notes_lower = client_notes.lower()
                if 'experience' in notes_lower or 'experiencia' in notes_lower:
                    composition_type = 'experience'
                    _logger.info("📝 Found 'experience' type in notes")
                elif 'hybrid' in notes_lower or 'wine' in notes_lower or 'vino' in notes_lower:
                    composition_type = 'hybrid'
                    _logger.info("📝 Found 'hybrid' type in notes")
            
            # 6. If regenerating, archive old composition
            if hasattr(self, 'regenerate_composition_id') and self.regenerate_composition_id:
                self.regenerate_composition_id.write({
                    'state': 'replaced',
                    'active': False
                })
                _logger.info(f"📝 Archiving old composition #{self.regenerate_composition_id.reference}")
            
            _logger.info(f"""
            ╔═══════════════════════════════════════════════════════╗
            ║        🎁 AI GIFT RECOMMENDATION REQUEST             ║
            ╠═══════════════════════════════════════════════════════╣
            ║ Client: {self.partner_id.name[:40]:<40} ║
            ║ Budget: €{budget_to_use:>15,.2f}                     ║
            ║ Dietary: {str(dietary)[:39] if dietary else 'None':<39} ║
            ║ Type: {composition_type:<43} ║
            ║ Notes: {('Yes - ' + str(len(client_notes)) + ' chars')[:39] if client_notes else 'None':<39} ║
            ║ Ollama Status: {('Connected' if self.recommender_id.ollama_enabled else 'Basic Mode'):<31} ║
            ╚═══════════════════════════════════════════════════════╝
            """)
            
            # 7. Call the recommendation engine with SIMPLE notes, not verbose context
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=budget_to_use,
                client_notes=client_notes,  # Pass actual client notes, not verbose instructions
                dietary_restrictions=dietary,
                composition_type=composition_type
            )
            
            # 8. Process result
            if result.get('success'):
                composition_id = result.get('composition_id')
                
                # Link to regenerated composition if applicable
                if hasattr(self, 'regenerate_composition_id') and self.regenerate_composition_id:
                    new_composition = self.env['gift.composition'].browse(composition_id)
                    if hasattr(new_composition, 'regenerated_from_id'):
                        new_composition.regenerated_from_id = self.regenerate_composition_id.id
                
                # Validate the result
                self._validate_and_process_result(result, budget_to_use)
                
                # Show success notification
                message = result.get('message', 'Composition generated successfully')
                self.env['bus.bus'].sendone(
                    (self._cr.dbname, 'res.partner', self.env.user.partner_id.id),
                    {'type': 'simple_notification', 
                    'title': '✅ Success',
                    'message': message,
                    'sticky': False,
                    'warning': False}
                )
                
                # Open the composition
                return {
                    'type': 'ir.actions.act_window',
                    'name': f'Gift Composition',
                    'res_model': 'gift.composition',
                    'res_id': composition_id,
                    'view_mode': 'form',
                    'target': 'current',
                }
            else:
                # Handle error
                error_msg = result.get('error', 'Unknown error occurred')
                _logger.error(f"❌ Generation failed: {error_msg}")
                
                self.state = 'error'
                self.error_message = error_msg
                
                # Show error notification
                self.env['bus.bus'].sendone(
                    (self._cr.dbname, 'res.partner', self.env.user.partner_id.id),
                    {'type': 'simple_notification',
                    'title': '❌ Generation Failed',
                    'message': error_msg,
                    'sticky': True,
                    'warning': True}
                )
                
                raise UserError(f"Generation failed: {error_msg}")
                
        except UserError:
            raise
        except Exception as e:
            _logger.error(f"❌ Unexpected error: {str(e)}", exc_info=True)
            self.state = 'error'
            self.error_message = str(e)
            
            # Show error notification
            self.env['bus.bus'].sendone(
                (self._cr.dbname, 'res.partner', self.env.user.partner_id.id),
                {'type': 'simple_notification',
                'title': '❌ Unexpected Error',
                'message': str(e),
                'sticky': True,
                'warning': True}
            )
            
            raise UserError(f"Unexpected error: {str(e)}")
    
    def _validate_and_process_result(self, result, expected_budget):
        """Intelligently validate and process the generation result"""
        composition_id = result.get('composition_id')
        composition = self.env['gift.composition'].browse(composition_id)
        
        # Calculate metrics
        actual_cost = composition.actual_cost or sum(p.list_price for p in composition.product_ids)
        product_count = len(composition.product_ids)
        
        # Calculate variance
        if expected_budget > 0:
            variance_pct = ((actual_cost - expected_budget) / expected_budget) * 100
            variance_acceptable = abs(variance_pct) <= 15  # Allow 15% variance
        else:
            variance_pct = 0
            variance_acceptable = True
        
        # Check for quality issues
        quality_issues = []
        
        # Check for zero-price products
        zero_price_products = [p for p in composition.product_ids if p.list_price <= 0]
        if zero_price_products:
            quality_issues.append(f"Found {len(zero_price_products)} products with €0 price")
        
        # Check for unrealistic product count
        if expected_budget > 0:
            avg_price = actual_cost / product_count if product_count > 0 else 0
            
            # Dynamic validation based on budget
            if actual_cost < expected_budget * 0.5:
                quality_issues.append(f"Total too low: €{actual_cost:.2f} vs €{expected_budget:.2f} expected")
            elif actual_cost > expected_budget * 2:
                quality_issues.append(f"Total too high: €{actual_cost:.2f} vs €{expected_budget:.2f} expected")
            
            # Check if product count makes sense
            if avg_price > 0:
                expected_avg = expected_budget / product_count
                if avg_price < expected_budget * 0.01:  # Average less than 1% of budget
                    quality_issues.append(f"Products too cheap for budget level")
        
        # Log comprehensive result
        _logger.info(f"""
        ╔═══════════════════════════════════════════════════════╗
        ║           ✅ GENERATION COMPLETE                      ║
        ╠═══════════════════════════════════════════════════════╣
        ║ Composition ID: #{composition.id:<37} ║
        ║ Products: {product_count:<44} ║
        ║ Total Cost: €{actual_cost:>15,.2f}                   ║
        ║ Target: €{expected_budget:>15,.2f}                   ║
        ║ Variance: {f'{variance_pct:+.1f}%':<43} ║
        ║ Status: {('✅ OK' if variance_acceptable else '⚠️ Check'):<45} ║
        ╚═══════════════════════════════════════════════════════╝
        """)
        
        # List top products
        for i, product in enumerate(composition.product_ids[:10], 1):
            _logger.info(f"  {i:2}. {product.default_code or 'N/A':<15} {product.name[:40]:<40} €{product.list_price:>8.2f}")
        
        if product_count > 10:
            _logger.info(f"  ... and {product_count - 10} more products")
        
        # Report quality issues if any
        if quality_issues:
            _logger.warning(f"""
        ⚠️ QUALITY ISSUES DETECTED:
        {chr(10).join(f'  - {issue}' for issue in quality_issues)}
        Consider reviewing the composition.
            """)
        
        # Process success
        self.state = 'done'
        self.composition_id = composition_id
        self.recommended_products = [(6, 0, composition.product_ids.ids)]
        self.confidence_score = result.get('confidence_score', 0.85)
        
        # Build result message
        self._build_result_message(result, actual_cost, expected_budget, product_count, quality_issues)
    
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
        """Parse client notes for key info, then prepare MINIMAL context for AI"""
        
        # First, extract key information from client notes if provided
        extracted_data = {}
        if self.client_notes:
            extracted_data = self._extract_key_info_from_notes(self.client_notes)
        
        # Build minimal context with extracted info
        context_parts = []
        
        # 1. Budget - check if client specified a different budget in notes
        budget_to_use = self.target_budget
        if extracted_data.get('budget'):
            budget_to_use = extracted_data['budget']
            _logger.info(f"📝 Budget override from notes: €{budget_to_use}")
            context_parts.append(f"Budget: €{budget_to_use}")
        elif self.target_budget > 0:
            context_parts.append(f"Budget: €{self.target_budget}")
        
        # 2. Product count - from notes or explicit field
        if extracted_data.get('product_count'):
            context_parts.append(f"Count: {extracted_data['product_count']} products")
        elif self.specify_product_count and self.product_count:
            context_parts.append(f"Count: {self.product_count} products")
        
        # 3. Dietary - combine all sources
        dietary = set()
        
        # From checkboxes
        if self.is_halal:
            dietary.add('halal')
        if self.is_vegan:
            dietary.add('vegan')
        if self.is_vegetarian:
            dietary.add('vegetarian')
        if self.is_non_alcoholic:
            dietary.add('no-alcohol')
        
        # From notes
        if extracted_data.get('dietary'):
            dietary.update(extracted_data['dietary'])
        
        if dietary:
            context_parts.append(f"Dietary: {', '.join(dietary)}")
        
        # 4. Special requirements from notes
        if extracted_data.get('special_requirements'):
            for req in extracted_data['special_requirements'][:2]:  # Max 2
                context_parts.append(req)
        
        # 5. Type
        if extracted_data.get('composition_type'):
            context_parts.append(f"Type: {extracted_data['composition_type']}")
        elif self.composition_type:
            context_parts.append(f"Type: {self.composition_type}")
        
        # 6. Add remaining client notes (stuff we couldn't parse)
        if extracted_data.get('remaining_notes'):
            context_parts.append(extracted_data['remaining_notes'][:100])
        
        # Join and limit
        final_context = ' | '.join(context_parts)[:500]
        
        _logger.info(f"AI Context ({len(final_context)} chars): {final_context}")
        
        # CRITICAL: Update the actual budget being used
        if extracted_data.get('budget'):
            self.target_budget = extracted_data['budget']
        
        return final_context

    def _extract_key_info_from_notes(self, notes):
        """Extract structured information from free-text notes"""
        
        if not notes:
            return {}
        
        extracted = {
            'budget': None,
            'product_count': None,
            'dietary': [],
            'composition_type': None,
            'special_requirements': [],
            'remaining_notes': notes
        }
        
        notes_lower = notes.lower()
        
        # 1. Extract budget (CRITICAL - was being ignored!)
        import re
        
        # Look for budget patterns
        budget_patterns = [
            r'budget[:\s]+(?:of\s+)?(?:€|eur)?\s*(\d+)',
            r'presupuesto[:\s]+(?:de\s+)?(?:€|eur)?\s*(\d+)',
            r'(?:€|eur)\s*(\d+)\s*budget',
            r'(\d{3,5})\s*(?:€|eur|euros?)',
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, notes_lower)
            if match:
                try:
                    budget_value = float(match.group(1))
                    if 100 <= budget_value <= 10000:
                        extracted['budget'] = budget_value
                        _logger.info(f"📝 Extracted budget from notes: €{budget_value}")
                        # Remove this from remaining notes
                        notes = notes[:match.start()] + notes[match.end():]
                        break
                except:
                    pass
        
        # 2. Extract product count
        count_patterns = [
            r'(\d+)\s+products?',
            r'(\d+)\s+items?',
            r'products?[:\s]+(\d+)',
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, notes_lower)
            if match:
                try:
                    count = int(match.group(1))
                    if 1 <= count <= 50:
                        extracted['product_count'] = count
                        notes = notes[:match.start()] + notes[match.end():]
                        break
                except:
                    pass
        
        # 3. Extract dietary restrictions
        dietary_keywords = {
            'halal': ['halal'],
            'vegan': ['vegan', 'vegano'],
            'vegetarian': ['vegetarian', 'vegetariano'],
            'no-alcohol': ['no alcohol', 'sin alcohol', 'non-alcoholic'],
            'gluten-free': ['gluten free', 'sin gluten', 'celiac'],
        }
        
        for diet_type, keywords in dietary_keywords.items():
            if any(kw in notes_lower for kw in keywords):
                extracted['dietary'].append(diet_type)
        
        # 4. Extract composition type
        if 'experience' in notes_lower or 'experiencia' in notes_lower:
            extracted['composition_type'] = 'experience'
        elif 'sweet' in notes_lower or 'dulce' in notes_lower:
            extracted['special_requirements'].append('include sweets')
        elif 'wine' in notes_lower or 'vino' in notes_lower:
            extracted['composition_type'] = 'hybrid'
        
        # 5. Clean up remaining notes
        # Remove extracted parts to avoid duplication
        for term in ['budget', 'presupuesto', 'halal', 'vegan', 'experience']:
            notes = re.sub(rf'\b{term}\b', '', notes, flags=re.IGNORECASE)
        
        extracted['remaining_notes'] = ' '.join(notes.split())  # Clean whitespace
        
        return extracted
    
    def _build_result_message(self, result, actual_cost, expected_budget, product_count, quality_issues):
        """Build comprehensive result message in HTML"""
        method = result.get('method', 'unknown')
        method_display = {
            'business_rules_with_enforcement': '📋 Business Rules + AI Enhancement',
            'business_rules_transformation': '📋 Business Rules Applied',
            '8020_rule': '📊 80/20 Rule Applied',
            'pattern_based_enhanced': '🔍 Pattern-Based AI Generation',
            'similar_clients': '👥 Similar Clients Analysis',
            'universal_enforcement': '🎯 Universal AI Generation',
            'fresh_generation': '🆕 Fresh AI Composition',
            'ollama_enhanced': '🤖 Ollama AI Enhanced'
        }.get(method, f'🎁 {method}')
        
        # Calculate compliance
        if expected_budget > 0:
            variance = ((actual_cost - expected_budget) / expected_budget) * 100
            if abs(variance) <= 5:
                budget_status = '✅ Excellent'
                status_color = '#28a745'
            elif abs(variance) <= 10:
                budget_status = '✅ Good'
                status_color = '#28a745'
            elif abs(variance) <= 15:
                budget_status = '⚠️ Acceptable'
                status_color = '#ffc107'
            else:
                budget_status = '⚠️ Review Needed'
                status_color = '#dc3545'
        else:
            variance = 0
            budget_status = '✅ OK'
            status_color = '#28a745'
        
        # Build HTML message
        self.result_message = f"""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {status_color};">
                <h3 style="margin-top: 0; color: #2c3e50;">
                    ✅ AI Composition Generated Successfully
                </h3>
                
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">
                    <div>
                        <h4 style="color: #6c757d; margin-bottom: 10px;">Generation Details</h4>
                        <table style="width: 100%;">
                            <tr>
                                <td style="padding: 5px; color: #495057;"><b>Method:</b></td>
                                <td style="padding: 5px;">{method_display}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px; color: #495057;"><b>Confidence:</b></td>
                                <td style="padding: 5px;">{result.get('confidence_score', 0.85)*100:.0f}%</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px; color: #495057;"><b>Products:</b></td>
                                <td style="padding: 5px;">{product_count} items</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div>
                        <h4 style="color: #6c757d; margin-bottom: 10px;">Budget Analysis</h4>
                        <table style="width: 100%;">
                            <tr>
                                <td style="padding: 5px; color: #495057;"><b>Target:</b></td>
                                <td style="padding: 5px;">€{expected_budget:,.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px; color: #495057;"><b>Actual:</b></td>
                                <td style="padding: 5px;">€{actual_cost:,.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px; color: #495057;"><b>Variance:</b></td>
                                <td style="padding: 5px;">{variance:+.1f}%</td>
                            </tr>
                            <tr>
                                <td style="padding: 5px; color: #495057;"><b>Status:</b></td>
                                <td style="padding: 5px; color: {status_color}; font-weight: bold;">{budget_status}</td>
                            </tr>
                        </table>
                    </div>
                </div>
        """
        
        # Add quality issues if any
        if quality_issues:
            self.result_message += f"""
                <div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 15px; border-left: 4px solid #ffc107;">
                    <h4 style="margin-top: 0; color: #856404;">
                        ⚠️ Quality Review Recommended
                    </h4>
                    <ul style="margin-bottom: 0; color: #856404;">
            """
            for issue in quality_issues:
                self.result_message += f"<li>{issue}</li>"
            self.result_message += """
                    </ul>
                </div>
            """
        
        # Add AI insights if available
        if result.get('ai_insights'):
            self.result_message += f"""
                <div style="background: #d1ecf1; padding: 10px; border-radius: 5px; margin-top: 15px; border-left: 4px solid #17a2b8;">
                    <h4 style="margin-top: 0; color: #0c5460;">
                        💡 AI Insights
                    </h4>
                    <p style="margin-bottom: 0; color: #0c5460;">
                        {result['ai_insights']}
                    </p>
                </div>
            """
        
        # Add rules applied if any
        if result.get('rules_applied'):
            rules_count = len(result['rules_applied'])
            self.result_message += f"""
                <div style="background: #e8f5e9; padding: 10px; border-radius: 5px; margin-top: 15px; border-left: 4px solid #4caf50;">
                    <p style="margin: 0; color: #2e7d32;">
                        <b>📋 Business Rules Applied:</b> {rules_count} transformations
                    </p>
                </div>
            """
        
        self.result_message += """
            </div>
        </div>
        """
    
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