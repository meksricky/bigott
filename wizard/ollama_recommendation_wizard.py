from odoo import models, fields, api, _
from odoo.exceptions import UserError
import logging
import json

_logger = logging.getLogger(__name__)

class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'Ollama Gift Recommendation Wizard'
    
    # Complete EXPERIENCIAS_DATA from EXPERIENCIAS 2025
    EXPERIENCES_DATA = {
        'X-EXP-HALAL': {
            'name': 'HALAL',
            'products': ['TOKAJ-MUSCAT', 'PATO-130', 'BACALAO-200', 'ASADA-250', 'PIMIENTO-95', 'BLANCOS-4FRUTOS', 'ACEITE-VIRGEN'],
            'category': 'gastronomy',
            'subcategory': 'halal',
            'dietary': ['halal', 'no_pork', 'no_alcohol'],
            'price': 72.60,
            'description': 'Complete halal-compliant premium selection from IDOM',
            'icon': '🥘',
            'tags': ['halal', 'premium', 'cultural']
        },
        'X-EXP-BACALAO': {
            'name': 'Experiencia Gastronómica de Bacalao Personalizada Idom',
            'products': ['BACALAO-200', 'ACEITE-OLIVA-200', 'ASADA-CARBON-250', 'GELEE-PIMIENTO-95'],
            'category': 'gastronomy',
            'subcategory': 'seafood',
            'price': 64.48,
            'description': 'Premium cod experience with artisanal accompaniments',
            'icon': '🐟',
            'tags': ['seafood', 'gourmet', 'artisanal']
        },
        'X-EXP-VEGETARIANA': {
            'name': 'Experiencia Gastronómica Vegetariana Personalizada Idom',
            'products': ['ALCACHOFA-180', 'BERENJENA-90', 'HIERBAS-PROVENZA-180', 'TORTA-CASAR-100', 'PERLAS-CHOCOLATE', 'COOKIES-CHOCOLATE', 'PASTAS-VEGANAS'],
            'category': 'gastronomy',
            'subcategory': 'vegetarian',
            'dietary': ['vegetarian'],
            'price': 64.48,
            'description': 'Gourmet vegetarian selection with artisanal treats',
            'icon': '🥗',
            'tags': ['vegetarian', 'healthy', 'sustainable']
        },
        'X-EXP-CHEESECAKE': {
            'name': 'Experiencia Gastronómica Cheesecake',
            'products': ['CHEESECAKE-130', 'GRANADA-70', 'WAFFLE-200', 'CAJA-CARTON'],
            'category': 'gastronomy',
            'subcategory': 'dessert',
            'price': 45.00,
            'description': 'Artisanal cheesecake experience with premium accompaniments',
            'icon': '🍰',
            'tags': ['dessert', 'sweet', 'artisanal']
        },
        'X-EXP-GILDA': {
            'name': 'EXP- GILDA',
            'products': ['ANCHOA-CONSORCIO', 'LB-ACEITU-140', 'GUIN-AGIN212'],
            'category': 'gastronomy',
            'subcategory': 'aperitif',
            'price': 34.50,
            'description': 'Traditional Basque gilda experience with premium ingredients',
            'icon': '🍢',
            'tags': ['basque', 'traditional', 'aperitif']
        },
        'X-EXP-LUBINA': {
            'name': 'EXP LUBINA',
            'products': ['LUB-CV200', 'ENSA-MIX-EMP135', 'LB-CHU-PIMAMA95'],
            'category': 'gastronomy',
            'subcategory': 'seafood',
            'price': 48.75,
            'description': 'Premium sea bass experience with gourmet sides',
            'icon': '🐟',
            'tags': ['seafood', 'mediterranean', 'fresh']
        },
        'X-EXP-BON-TOMATE': {
            'name': 'EXP BONITO',
            'products': ['BONITO', 'LB-SALSA-TOMATE', 'LB-HONGO-BOL212'],
            'category': 'gastronomy',
            'subcategory': 'seafood',
            'price': 52.30,
            'description': 'Bonito tuna with tomato and boletus mushrooms',
            'icon': '🍅',
            'tags': ['seafood', 'mediterranean', 'seasonal']
        },
        # Wellness Experiences
        'X-EXP-SPA-RELAX': {
            'name': 'Spa & Relaxation Experience',
            'products': [],
            'category': 'wellness',
            'subcategory': 'spa',
            'price': 150.00,
            'description': '2-hour spa session with massage and thermal circuit',
            'icon': '💆',
            'tags': ['spa', 'relaxation', 'wellness']
        },
        # Adventure Experiences
        'X-EXP-WINE-TOUR': {
            'name': 'Wine Tasting Tour',
            'products': [],
            'category': 'adventure',
            'subcategory': 'wine',
            'price': 85.00,
            'description': 'Guided tour through premium wineries with tasting session',
            'icon': '🍷',
            'tags': ['wine', 'tour', 'tasting']
        },
        # Cultural Experiences
        'X-EXP-MUSEUM': {
            'name': 'Museum & Art Gallery Pass',
            'products': [],
            'category': 'culture',
            'subcategory': 'art',
            'price': 60.00,
            'description': 'Annual pass to major museums and galleries',
            'icon': '🎨',
            'tags': ['culture', 'art', 'education']
        }
    }

    # ================== WIZARD STEP MANAGEMENT ==================
    
    wizard_step = fields.Selection([
        ('client', 'Client Selection'),
        ('budget', 'Budget & Requirements'),
        ('composition', 'Composition Type'),
        ('dietary', 'Dietary & Preferences'),
        ('preview', 'Preview & Generate')
    ], string='Current Step', default='client')
    
    state = fields.Selection([
        ('draft', 'Draft'),
        ('preview', 'Preview'),
        ('generating', 'Generating'),
        ('done', 'Done'),
        ('error', 'Error')
    ], string='State', default='draft')
    
    # ================== CLIENT FIELDS ==================
    
    partner_id = fields.Many2one(
        'res.partner', 
        string='Client', 
        required=True,
        help="Select the client for whom to generate recommendations"
    )
    
    client_segment = fields.Selection([
        ('vip', '⭐ VIP Client'),
        ('corporate', '🏢 Corporate'),
        ('regular', '👤 Regular'),
        ('new', '🆕 New Client')
    ], string='Client Segment', compute='_compute_client_segment')
    
    # ================== BUDGET FIELDS ==================
    
    currency_id = fields.Many2one(
        'res.currency',
        default=lambda self: self.env.company.currency_id
    )
    
    budget_source = fields.Selection([
        ('manual', 'Manual Entry'),
        ('history', 'From History'),
        ('notes', 'From Notes')
    ], string='Budget Source', default='manual')
    
    target_budget = fields.Float(
        string='Target Budget',
        default=1000.0,
        help="Target budget for the gift composition"
    )
    
    budget_flexibility = fields.Selection([
        ('strict', '±5% - Strict'),
        ('normal', '±10% - Normal'),
        ('flexible', '±15% - Flexible')
    ], string='Budget Flexibility', default='normal')
    
    target_year = fields.Integer(
        string='Target Year',
        default=lambda self: fields.Date.today().year
    )
    
    # ================== COMPOSITION TYPE ==================
    
    composition_strategy = fields.Selection([
        ('auto', '🤖 AI Auto-Select'),
        ('custom', '🎨 Custom Mix'),
        ('hybrid', '🍷 Wine-Focused'),
        ('experience', '🎭 Experience-Based'),
        ('seasonal', '🎄 Seasonal'),
        ('themed', '🎪 Themed')
    ], string='Composition Strategy', default='auto')
    
    # Experience Selection
    experience_category = fields.Selection([
        ('gastronomy', '🍽️ Gastronomy'),
        ('wellness', '💆 Wellness & Spa'),
        ('adventure', '🎯 Adventure'),
        ('culture', '🎨 Culture & Arts'),
        ('luxury', '💎 Luxury')
    ], string='Experience Category')
    
    selected_experience_key = fields.Selection(
        selection='_get_experience_selection',
        string='Pre-defined Experience'
    )
    
    custom_experience_id = fields.Many2one(
        'product.template',
        string='Custom Experience',
        domain="[('is_experience', '=', True)]"
    )
    
    # Theme Selection
    gift_theme = fields.Selection([
        ('christmas', '🎄 Christmas'),
        ('corporate', '🏢 Corporate'),
        ('birthday', '🎂 Birthday'),
        ('anniversary', '💍 Anniversary'),
        ('thank_you', '🙏 Thank You'),
        ('celebration', '🎉 Celebration')
    ], string='Gift Theme')
    
    # ================== PRODUCT REQUIREMENTS ==================
    
    product_count_mode = fields.Selection([
        ('auto', 'Auto (12-15)'),
        ('exact', 'Exact Count'),
        ('range', 'Range')
    ], string='Product Count Mode', default='auto')
    
    product_count_exact = fields.Integer(
        string="Exact Count",
        default=12
    )
    
    product_count_min = fields.Integer(
        string="Min Products",
        default=10
    )
    
    product_count_max = fields.Integer(
        string="Max Products",
        default=15
    )
    
    # ================== DIETARY & PREFERENCES ==================
    
    dietary_profile = fields.Selection([
        ('none', '✅ No Restrictions'),
        ('halal', '☪️ Halal'),
        ('vegan', '🌱 Vegan'),
        ('vegetarian', '🥬 Vegetarian'),
        ('gluten_free', '🌾 Gluten Free'),
        ('non_alcoholic', '🚫 Non-Alcoholic'),
        ('custom', '⚙️ Custom Mix')
    ], string='Dietary Profile', default='none')
    
    dietary_details = fields.Text(
        string='Dietary Details',
        placeholder="Specify any allergies, restrictions, or preferences..."
    )
    
    # Category Preferences
    include_wine = fields.Boolean(string='Include Wine', default=True)
    include_spirits = fields.Boolean(string='Include Spirits')
    include_gourmet = fields.Boolean(string='Include Gourmet Food', default=True)
    include_sweets = fields.Boolean(string='Include Sweets', default=True)
    include_experiences = fields.Boolean(string='Include Experiences')
    
    # ================== CLIENT CONTEXT ==================
    
    occasion_notes = fields.Text(
        string='Occasion & Context',
        placeholder="What's the occasion? Any special requirements?"
    )
    
    client_preferences = fields.Text(
        string='Known Preferences',
        compute='_compute_client_preferences'
    )
    
    # ================== PREVIEW FIELDS ==================
    
    preview_generated = fields.Boolean(
        string='Preview Generated',
        default=False
    )
    
    preview_html = fields.Html(
        string='Composition Preview',
        readonly=True
    )
    
    preview_products = fields.Many2many(
        'product.template',
        'wizard_preview_products_rel',
        string='Preview Products'
    )
    
    preview_total = fields.Float(
        string='Preview Total',
        compute='_compute_preview_total'
    )
    
    preview_confidence = fields.Float(
        string='Preview Confidence',
        default=0.0
    )
    
    # ================== RESULTS ==================
    
    composition_id = fields.Many2one(
        'gift.composition',
        string='Generated Composition'
    )
    
    generation_method = fields.Char(
        string='Generation Method'
    )
    
    result_message = fields.Html(
        string='Result'
    )
    
    # ================== SYSTEM STATUS ==================
    
    recommender_id = fields.Many2one(
        'ollama.gift.recommender',
        string='Recommender Engine',
        default=lambda self: self.env['ollama.gift.recommender'].get_or_create_recommender()
    )
    
    ai_status = fields.Html(
        string='AI Status',
        compute='_compute_ai_status'
    )
    
    # ================== COMPUTED FIELDS ==================
    
    @api.depends('partner_id')
    def _compute_client_segment(self):
        for wizard in self:
            if not wizard.partner_id:
                wizard.client_segment = 'new'
            else:
                orders = self.env['sale.order'].search([
                    ('partner_id', '=', wizard.partner_id.id),
                    ('state', 'in', ['sale', 'done'])
                ])
                
                total_value = sum(orders.mapped('amount_untaxed'))
                
                if total_value > 10000:
                    wizard.client_segment = 'vip'
                elif wizard.partner_id.is_company:
                    wizard.client_segment = 'corporate'
                elif len(orders) > 0:
                    wizard.client_segment = 'regular'
                else:
                    wizard.client_segment = 'new'
    
    @api.depends('partner_id')
    def _compute_client_preferences(self):
        for wizard in self:
            if not wizard.partner_id:
                wizard.client_preferences = ''
                continue
            
            # Analyze historical preferences
            prefs = []
            
            # Get last compositions
            compositions = self.env['gift.composition'].search([
                ('partner_id', '=', wizard.partner_id.id)
            ], limit=3, order='create_date desc')
            
            if compositions:
                # Analyze product patterns
                all_products = compositions.mapped('product_ids')
                categories = {}
                for product in all_products:
                    cat = product.categ_id.name if product.categ_id else 'Other'
                    categories[cat] = categories.get(cat, 0) + 1
                
                top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                if top_categories:
                    prefs.append(f"Prefers: {', '.join([c[0] for c in top_categories])}")
                
                # Check dietary history
                dietary_history = set()
                for comp in compositions:
                    if comp.dietary_restrictions:
                        dietary_history.add(comp.dietary_restrictions)
                
                if dietary_history:
                    prefs.append(f"Previous dietary: {', '.join(dietary_history)}")
            
            wizard.client_preferences = ' | '.join(prefs) if prefs else 'No history available'
    
    @api.depends('preview_products')
    def _compute_preview_total(self):
        for wizard in self:
            wizard.preview_total = sum(wizard.preview_products.mapped('list_price'))
    
    @api.depends('recommender_id')
    def _compute_ai_status(self):
        for wizard in self:
            if wizard.recommender_id and wizard.recommender_id.ollama_enabled:
                status = """
                <div style="background: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
                    <span style="color: #155724;">
                        <b>🟢 AI Enhanced Mode</b><br>
                        Ollama is connected and ready for intelligent parsing
                    </span>
                </div>
                """
            else:
                status = """
                <div style="background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107;">
                    <span style="color: #856404;">
                        <b>🟡 Basic Mode</b><br>
                        Using rule-based engine (Ollama not connected)
                    </span>
                </div>
                """
            wizard.ai_status = status
    
    # ================== SELECTION METHODS ==================
    
    def _get_experience_selection(self):
        """Get experience selection based on category"""
        selections = []
        
        for key, exp in self.EXPERIENCES_DATA.items():
            category_label = {
                'gastronomy': '🍽️',
                'wellness': '💆',
                'adventure': '🎯',
                'culture': '🎨',
                'luxury': '💎'
            }.get(exp['category'], '📦')
            
            label = f"{category_label} {exp['name']} (€{exp['price']:.2f})"
            selections.append((key, label))
        
        return selections
    
    # ================== ONCHANGE METHODS ==================
    
    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Auto-fill based on client history"""
        if self.partner_id and self.recommender_id:
            # Get client patterns
            patterns = self.recommender_id._analyze_client_purchase_patterns(self.partner_id.id)
            
            if patterns and patterns.get('total_orders', 0) > 0:
                # Suggest budget
                avg_budget = patterns.get('avg_order_value', 0)
                if patterns.get('budget_trend') == 'increasing':
                    self.target_budget = avg_budget * 1.1
                    self.budget_source = 'history'
                elif patterns.get('budget_trend') == 'decreasing':
                    self.target_budget = avg_budget * 0.95
                    self.budget_source = 'history'
                else:
                    self.target_budget = avg_budget
                    self.budget_source = 'history'
    
    @api.onchange('experience_category')
    def _onchange_experience_category(self):
        """Filter experiences by category"""
        if self.experience_category:
            # Reset selection when category changes
            self.selected_experience_key = False
            
            # Update domain for custom experience
            return {
                'domain': {
                    'custom_experience_id': [
                        ('is_experience', '=', True),
                        ('experience_category', '=', self.experience_category)
                    ]
                }
            }
    
    @api.onchange('composition_strategy')
    def _onchange_composition_strategy(self):
        """Update fields based on strategy"""
        if self.composition_strategy == 'experience':
            self.include_experiences = True
        elif self.composition_strategy == 'hybrid':
            self.include_wine = True
    
    @api.onchange('dietary_profile')
    def _onchange_dietary_profile(self):
        """Update category preferences based on dietary profile"""
        if self.dietary_profile == 'halal':
            self.include_wine = False
            self.include_spirits = False
            self.dietary_details = "Halal certified products only. No alcohol, no pork."
        elif self.dietary_profile == 'vegan':
            self.dietary_details = "Vegan products only. No animal products."
        elif self.dietary_profile == 'vegetarian':
            self.dietary_details = "Vegetarian products. No meat or fish."
        elif self.dietary_profile == 'non_alcoholic':
            self.include_wine = False
            self.include_spirits = False
            self.dietary_details = "No alcoholic beverages."
    
    # ================== WIZARD NAVIGATION ==================
    
    def action_next_step(self):
        """Move to next wizard step"""
        self.ensure_one()
        
        steps = ['client', 'budget', 'composition', 'dietary', 'preview']
        current_index = steps.index(self.wizard_step)
        
        if current_index < len(steps) - 1:
            self.wizard_step = steps[current_index + 1]
        
        return {'type': 'ir.actions.do_nothing'}
    
    def action_previous_step(self):
        """Move to previous wizard step"""
        self.ensure_one()
        
        steps = ['client', 'budget', 'composition', 'dietary', 'preview']
        current_index = steps.index(self.wizard_step)
        
        if current_index > 0:
            self.wizard_step = steps[current_index - 1]
        
        return {'type': 'ir.actions.do_nothing'}
    
    # ================== PREVIEW METHODS ==================
    
    def action_generate_preview(self):
        """Generate a preview before final generation"""
        self.ensure_one()
        
        if not self.partner_id:
            raise UserError("Please select a client first")
        
        self.state = 'preview'
        
        try:
            # Prepare parameters
            params = self._prepare_generation_params()
            
            # Call recommender for preview (dry run)
            preview_result = self.recommender_id.generate_preview(
                partner_id=self.partner_id.id,
                params=params
            )
            
            if preview_result.get('success'):
                self._display_preview(preview_result)
                self.preview_generated = True
                self.preview_confidence = preview_result.get('confidence', 0.85)
            else:
                raise UserError(f"Preview failed: {preview_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            _logger.error(f"Preview generation failed: {str(e)}")
            raise UserError(f"Could not generate preview: {str(e)}")
        
        return {'type': 'ir.actions.do_nothing'}
    
    def _display_preview(self, preview_result):
        """Display the preview in HTML format"""
        products = preview_result.get('products', [])
        total_cost = preview_result.get('total_cost', 0)
        method = preview_result.get('method', 'auto')
        
        # Build preview HTML
        html = f"""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                <h3 style="margin-top: 0; color: #2c3e50;">
                    <span style="font-size: 24px;">🎁</span> Gift Composition Preview
                </h3>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
                    <div style="text-align: center; padding: 10px; background: #e8f4fd; border-radius: 5px;">
                        <div style="font-size: 24px; color: #1976d2;">📦</div>
                        <div style="font-weight: bold; margin-top: 5px;">{len(products)}</div>
                        <div style="font-size: 12px; color: #666;">Products</div>
                    </div>
                    <div style="text-align: center; padding: 10px; background: #e8f5e9; border-radius: 5px;">
                        <div style="font-size: 24px; color: #4caf50;">💰</div>
                        <div style="font-weight: bold; margin-top: 5px;">€{total_cost:.2f}</div>
                        <div style="font-size: 12px; color: #666;">Total Value</div>
                    </div>
                    <div style="text-align: center; padding: 10px; background: #fff3e0; border-radius: 5px;">
                        <div style="font-size: 24px; color: #ff9800;">📊</div>
                        <div style="font-weight: bold; margin-top: 5px;">{self.preview_confidence*100:.0f}%</div>
                        <div style="font-size: 12px; color: #666;">Match Score</div>
                    </div>
                </div>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 5px;">
                <h4 style="margin-top: 0; color: #495057;">Product Selection</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
        """
        
        # Add products to preview
        for i, product in enumerate(products[:20], 1):
            category_icon = self._get_category_icon(product.get('category', ''))
            html += f"""
                <div style="padding: 8px; background: #f8f9fa; border-radius: 4px; display: flex; align-items: center;">
                    <span style="font-size: 20px; margin-right: 10px;">{category_icon}</span>
                    <div style="flex: 1;">
                        <div style="font-weight: 500; font-size: 14px;">{product.get('name', 'Product')[:40]}</div>
                        <div style="color: #28a745; font-size: 12px;">€{product.get('price', 0):.2f}</div>
                    </div>
                </div>
            """
        
        html += """
                </div>
            </div>
            
            <div style="margin-top: 15px; padding: 15px; background: #d1ecf1; border-radius: 5px; border-left: 4px solid #17a2b8;">
                <div style="color: #0c5460;">
                    <b>💡 Generation Method:</b> {method_display}<br>
                    <b>✅ Ready to Generate:</b> Click "Generate Composition" to create the final gift
                </div>
            </div>
        </div>
        """.format(method_display=self._get_method_display(method))
        
        self.preview_html = html
        
        # Store preview products for reference
        if products:
            product_ids = []
            for p in products:
                if p.get('id'):
                    product_ids.append(p['id'])
            
            if product_ids:
                self.preview_products = [(6, 0, product_ids)]
    
    def _get_category_icon(self, category):
        """Get icon for product category"""
        icons = {
            'wine': '🍷',
            'spirits': '🥃',
            'gourmet': '🧀',
            'sweets': '🍫',
            'experience': '🎭',
            'seafood': '🐟',
            'meat': '🥩',
            'vegetarian': '🥗',
            'oil': '🫒',
            'sauce': '🥫',
            'cheese': '🧀',
            'chocolate': '🍫'
        }
        
        category_lower = category.lower()
        for key, icon in icons.items():
            if key in category_lower:
                return icon
        return '📦'
    
    def _get_method_display(self, method):
        """Get display name for generation method"""
        methods = {
            'business_rules': '📋 Business Rules Applied',
            'pattern_based': '📊 Pattern-Based Selection',
            'similar_clients': '👥 Similar Clients Analysis',
            'fresh': '🆕 Fresh Generation',
            'auto': '🤖 AI Auto-Selection'
        }
        return methods.get(method, method)
    
    # ================== GENERATION METHODS ==================
    
    def action_generate_composition(self):
        """Generate final composition"""
        self.ensure_one()
        
        if not self.partner_id:
            raise UserError("Please select a client")
        
        self.state = 'generating'
        
        try:
            # Prepare parameters
            params = self._prepare_generation_params()
            
            # Log generation request
            _logger.info(f"""
            ═══════════════════════════════════════
            🎁 GIFT COMPOSITION GENERATION
            ═══════════════════════════════════════
            Client: {self.partner_id.name}
            Budget: €{self.target_budget:.2f}
            Strategy: {self.composition_strategy}
            Products: {self._get_product_count_display()}
            Dietary: {self.dietary_profile}
            ═══════════════════════════════════════
            """)
            
            # Generate composition
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=self.target_budget,
                client_notes=params.get('notes', ''),
                dietary_restrictions=params.get('dietary', []),
                composition_type=params.get('composition_type', 'custom')
            )
            
            if result.get('success'):
                self._process_success(result)
                
                # Open the composition
                return {
                    'type': 'ir.actions.act_window',
                    'name': 'Generated Gift Composition',
                    'res_model': 'gift.composition',
                    'res_id': result.get('composition_id'),
                    'view_mode': 'form',
                    'target': 'current',
                }
            else:
                raise UserError(f"Generation failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            _logger.error(f"Generation failed: {str(e)}")
            self.state = 'error'
            raise
    
    def _prepare_generation_params(self):
        """Prepare parameters for generation"""
        params = {
            'budget': self.target_budget,
            'flexibility': self.budget_flexibility,
            'composition_type': self.composition_strategy,
            'dietary': [],
            'notes': '',
            'categories': {}
        }
        
        # Process dietary restrictions
        dietary = []
        if self.dietary_profile == 'halal':
            dietary = ['halal', 'no_pork', 'no_alcohol', 'no_iberian']
        elif self.dietary_profile == 'vegan':
            dietary = ['vegan']
        elif self.dietary_profile == 'vegetarian':
            dietary = ['vegetarian']
        elif self.dietary_profile == 'gluten_free':
            dietary = ['gluten_free']
        elif self.dietary_profile == 'non_alcoholic':
            dietary = ['non_alcoholic']
        elif self.dietary_profile == 'custom' and self.dietary_details:
            dietary = [d.strip() for d in self.dietary_details.split(',')]
        
        params['dietary'] = dietary
        
        # Build notes
        notes_parts = []
        
        # Add product count requirement
        if self.product_count_mode == 'exact':
            notes_parts.append(f"Must have exactly {self.product_count_exact} products")
        elif self.product_count_mode == 'range':
            notes_parts.append(f"Include {self.product_count_min} to {self.product_count_max} products")
        
        # Add category preferences
        categories = []
        if self.include_wine:
            categories.append('wine')
        if self.include_spirits:
            categories.append('spirits')
        if self.include_gourmet:
            categories.append('gourmet food')
        if self.include_sweets:
            categories.append('sweets and chocolates')
        if self.include_experiences:
            categories.append('experiences')
        
        if categories:
            notes_parts.append(f"Include: {', '.join(categories)}")
        
        # Add experience if selected
        if self.composition_strategy == 'experience':
            if self.selected_experience_key:
                exp_data = self.EXPERIENCES_DATA.get(self.selected_experience_key, {})
                notes_parts.append(f"Include {exp_data.get('name', 'selected experience')}")
            elif self.custom_experience_id:
                notes_parts.append(f"Include experience: {self.custom_experience_id.name}")
        
        # Add theme if selected
        if self.gift_theme:
            theme_name = dict(self._fields['gift_theme'].selection).get(self.gift_theme, '')
            notes_parts.append(f"Theme: {theme_name}")
        
        # Add occasion notes
        if self.occasion_notes:
            notes_parts.append(self.occasion_notes)
        
        params['notes'] = ". ".join(notes_parts)
        
        return params
    
    def _get_product_count_display(self):
        """Get display string for product count"""
        if self.product_count_mode == 'exact':
            return f"Exactly {self.product_count_exact}"
        elif self.product_count_mode == 'range':
            return f"{self.product_count_min}-{self.product_count_max}"
        else:
            return "Auto (12-15)"
    
    def _process_success(self, result):
        """Process successful generation"""
        self.state = 'done'
        self.composition_id = result.get('composition_id')
        self.generation_method = result.get('method', 'auto')
        
        # Build success message
        composition = self.env['gift.composition'].browse(result.get('composition_id'))
        
        html = f"""
        <div style="background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
            <h4 style="margin-top: 0; color: #155724;">
                ✅ Composition Generated Successfully!
            </h4>
            
            <table style="width: 100%; margin-top: 15px;">
                <tr>
                    <td style="padding: 5px;"><b>Composition ID:</b></td>
                    <td>#{composition.id}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Products:</b></td>
                    <td>{len(composition.product_ids)} items</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Total Value:</b></td>
                    <td>€{composition.actual_cost:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Method:</b></td>
                    <td>{self._get_method_display(self.generation_method)}</td>
                </tr>
            </table>
            
            <div style="margin-top: 15px;">
                <button class="btn btn-primary" onclick="window.location.reload()">
                    View Composition
                </button>
            </div>
        </div>
        """
        
        self.result_message = html
    
    # ================== UTILITY METHODS ==================
    
    def action_reset_wizard(self):
        """Reset wizard to initial state"""
        self.ensure_one()
        
        self.write({
            'wizard_step': 'client',
            'state': 'draft',
            'preview_generated': False,
            'preview_html': False,
            'preview_products': [(5, 0, 0)],
            'result_message': False
        })
        
        return {'type': 'ir.actions.do_nothing'}
    
    def action_test_ai_connection(self):
        """Test AI connection"""
        self.ensure_one()
        
        if self.recommender_id:
            result = self.recommender_id.test_ollama_connection()
            
            notification_type = 'success' if result['success'] else 'warning'
            
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'AI Connection Test',
                    'message': result['message'],
                    'type': notification_type,
                    'sticky': False,
                }
            }