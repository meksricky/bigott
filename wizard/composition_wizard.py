# Advanced Composition Wizard - Schema Aware
# wizard/composition_wizard.py

from odoo import models, fields, api
from odoo.exceptions import UserError
import json
import logging

_logger = logging.getLogger(__name__)

class CompositionWizard(models.TransientModel):
    _name = 'composition.wizard'
    _description = 'Advanced AI/ML Gift Composition Wizard'
    
    # Basic Configuration
    partner_id = fields.Many2one('res.partner', 'Client', required=True, 
                                 domain="[('is_company', '=', False)]")
    target_year = fields.Integer('Target Year', required=True, 
                                 default=lambda self: fields.Date.context_today(self).year)
    target_budget = fields.Float('Target Budget (€)', required=True, default=200.0)
    
    # Advanced Options
    dietary_restrictions = fields.Selection([
        ('none', 'None'),
        ('vegan', 'Vegan'),
        ('halal', 'Halal'), 
        ('non_alcoholic', 'Non-Alcoholic'),
        ('vegan_halal', 'Vegan + Halal'),
        ('vegan_non_alcoholic', 'Vegan + Non-Alcoholic'),
        ('halal_non_alcoholic', 'Halal + Non-Alcoholic'),
        ('all_restrictions', 'Vegan + Halal + Non-Alcoholic')
    ], string='Dietary Restrictions', default='none')
    
    force_engine = fields.Selection([
        ('auto', 'Auto-Select Best Engine'),
        ('ml', 'ML Engine (if trained)'),
        ('ai', 'AI Engine'),
        ('smart', 'Smart Business Engine'),
        ('sophisticated', 'Sophisticated Fallback')
    ], string='Engine Selection', default='auto')
    
    # Enhanced Notes
    additional_notes = fields.Text('Additional Notes & Preferences',
                                  help="Provide any special requirements, preferences, or context for this composition")
    
    # Client Information Display
    partner_dietary_restrictions = fields.Selection(
        related='partner_id.dietary_restrictions', 
        string='Client Dietary Profile',
        readonly=True
    )
    
    # Advanced Analytics & Insights
    client_info = fields.Html('Client Intelligence', compute='_compute_client_intelligence')
    engine_status = fields.Html('Engine Status', compute='_compute_engine_status')
    recommendations = fields.Html('AI Recommendations', compute='_compute_ai_recommendations')
    
    # Risk Assessment
    risk_level = fields.Selection([
        ('low', 'Low Risk - Established Client'),
        ('medium', 'Medium Risk - Some History'), 
        ('high', 'High Risk - New Client')
    ], compute='_compute_risk_assessment', string='Risk Assessment')
    
    # Budget Intelligence
    budget_recommendation = fields.Char('Budget Recommendation', compute='_compute_budget_intelligence')
    expected_products = fields.Char('Expected Product Count', compute='_compute_budget_intelligence')
    
    @api.depends('partner_id')
    def _compute_client_intelligence(self):
        """Build comprehensive client intelligence dashboard"""
        
        for record in self:
            if not record.partner_id:
                record.client_info = ""
                continue
                
            try:
                # Build client profile using integration manager logic
                integration_manager = self.env['integration.manager']
                client_profile = integration_manager._build_client_profile(record.partner_id.id, "")
                
                # Build rich HTML dashboard
                info_html = self._build_client_dashboard(client_profile)
                record.client_info = info_html
                
            except Exception as e:
                _logger.warning(f"Client intelligence computation failed: {e}")
                record.client_info = f"<div class='alert alert-warning'>Client analysis temporarily unavailable</div>"
    
    @api.depends('partner_id', 'target_budget')
    def _compute_engine_status(self):
        """Display current engine availability and status"""
        
        for record in self:
            try:
                status_html = "<div class='row'>"
                
                # ML Engine Status
                ml_status = self._check_ml_engine_status()
                status_html += f"""
                <div class='col-md-6'>
                    <div class='card {ml_status['css_class']}'>
                        <div class='card-body text-center'>
                            <h6 class='card-title'>🧠 ML Engine</h6>
                            <p class='card-text'>{ml_status['status']}</p>
                            <small class='text-muted'>{ml_status['details']}</small>
                        </div>
                    </div>
                </div>
                """
                
                # AI Engine Status
                ai_status = self._check_ai_engine_status()
                status_html += f"""
                <div class='col-md-6'>
                    <div class='card {ai_status['css_class']}'>
                        <div class='card-body text-center'>
                            <h6 class='card-title'>🤖 AI Engine</h6>
                            <p class='card-text'>{ai_status['status']}</p>
                            <small class='text-muted'>{ai_status['details']}</small>
                        </div>
                    </div>
                </div>
                """
                
                status_html += "</div>"
                record.engine_status = status_html
                
            except Exception as e:
                _logger.warning(f"Engine status computation failed: {e}")
                record.engine_status = "<div class='alert alert-info'>Engine status check unavailable</div>"
    
    @api.depends('partner_id', 'target_budget', 'dietary_restrictions', 'additional_notes')
    def _compute_ai_recommendations(self):
        """Generate AI-powered recommendations for the composition"""
        
        for record in self:
            if not record.partner_id:
                record.recommendations = ""
                continue
            
            try:
                recommendations = self._generate_ai_recommendations(record)
                record.recommendations = recommendations
                
            except Exception as e:
                _logger.warning(f"AI recommendations computation failed: {e}")
                record.recommendations = "<div class='alert alert-info'>AI recommendations temporarily unavailable</div>"
    
    @api.depends('partner_id')
    def _compute_risk_assessment(self):
        """Assess client risk level for composition generation"""
        
        for record in self:
            if not record.partner_id:
                record.risk_level = 'medium'
                continue
            
            try:
                # Check client history
                history_count = self.env['client.order.history'].search_count([
                    ('partner_id', '=', record.partner_id.id)
                ])
                
                if history_count >= 3:
                    record.risk_level = 'low'
                elif history_count >= 1:
                    record.risk_level = 'medium'
                else:
                    record.risk_level = 'high'
                    
            except Exception as e:
                _logger.warning(f"Risk assessment failed: {e}")
                record.risk_level = 'medium'
    
    @api.depends('partner_id', 'target_budget')
    def _compute_budget_intelligence(self):
        """Provide intelligent budget analysis and recommendations"""
        
        for record in self:
            if not record.partner_id or not record.target_budget:
                record.budget_recommendation = ""
                record.expected_products = ""
                continue
            
            try:
                # Analyze historical budget patterns
                history_records = self.env['client.order.history'].search([
                    ('partner_id', '=', record.partner_id.id)
                ])
                
                if history_records:
                    avg_budget = sum(h.total_budget for h in history_records if h.total_budget > 0) / len(history_records)
                    
                    if record.target_budget < avg_budget * 0.8:
                        record.budget_recommendation = f"Consider increasing to €{avg_budget:.0f} (historical average)"
                    elif record.target_budget > avg_budget * 1.5:
                        record.budget_recommendation = f"Above average budget - premium selection possible"
                    else:
                        record.budget_recommendation = "Budget aligns with historical patterns"
                else:
                    record.budget_recommendation = "Appropriate budget for new client"
                
                # Calculate expected products
                expected_count = max(3, min(8, int(record.target_budget / 40)))
                record.expected_products = f"{expected_count-1}-{expected_count+1} products expected"
                
            except Exception as e:
                _logger.warning(f"Budget intelligence computation failed: {e}")
                record.budget_recommendation = "Budget analysis unavailable"
                record.expected_products = "3-5 products expected"

    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Auto-populate fields based on partner selection"""
        
        if self.partner_id:
            # Set dietary restrictions from partner
            try:
                if hasattr(self.partner_id, 'dietary_restrictions') and self.partner_id.dietary_restrictions != 'none':
                    self.dietary_restrictions = self.partner_id.dietary_restrictions
            except:
                pass
            
            # Suggest budget based on historical data
            try:
                history_records = self.env['client.order.history'].search([
                    ('partner_id', '=', self.partner_id.id)
                ])
                
                if history_records:
                    budgets = [h.total_budget for h in history_records if h.total_budget > 0]
                    if budgets:
                        avg_budget = sum(budgets) / len(budgets)
                        # Suggest 10% increase from average
                        self.target_budget = avg_budget * 1.1
            except:
                pass
    
    def action_generate_composition(self):
        """Generate composition using Advanced AI/ML Integration Manager"""
        
        if self.target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
        
        try:
            _logger.info(f"Advanced wizard generating composition for {self.partner_id.name}")
            
            # Parse dietary restrictions
            dietary_list = self._parse_dietary_restrictions()
            
            # Get or create integration manager
            integration_manager = self._get_integration_manager()
            
            # Determine force engine
            force_engine = None if self.force_engine == 'auto' else self.force_engine
            
            # Generate composition using advanced system
            composition = integration_manager.generate_complete_composition(
                partner_id=self.partner_id.id,
                target_budget=self.target_budget,
                target_year=self.target_year,
                dietary_restrictions=dietary_list,
                notes_text=self.additional_notes,
                use_batch=False,
                attempt_number=1,
                force_engine=force_engine
            )
            
            if not composition:
                raise UserError("Failed to generate composition with the advanced AI/ML system")
            
            # Log success
            _logger.info(f"Advanced composition generated successfully: {composition.name}")
            _logger.info(f"Products: {len(composition.product_ids)}, Cost: €{composition.actual_cost:.2f}")
            
            # Show success notification
            self._show_success_notification(composition)
            
            # Return action to view the generated composition
            return {
                'type': 'ir.actions.act_window',
                'name': f'AI Composition - {composition.name}',
                'res_model': 'gift.composition',
                'res_id': composition.id,
                'view_mode': 'form',
                'target': 'current',
                'context': {
                    'default_composition_id': composition.id,
                    'from_wizard': True
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            _logger.error(f"Advanced composition generation failed: {error_msg}")
            
            # Enhanced error handling with specific messages
            if "No products available" in error_msg:
                raise UserError("No suitable products found. Please:\n"
                              "• Check that products are active and available for sale\n"
                              "• Consider relaxing dietary restrictions\n"
                              "• Verify product categories are properly configured")
            elif "budget" in error_msg.lower():
                raise UserError("Budget configuration issue. Please:\n"
                              "• Ensure target budget is reasonable (€50-€1000)\n"
                              "• Check that products exist within budget range\n"
                              "• Consider adjusting budget based on recommendations")
            elif "engine" in error_msg.lower():
                raise UserError("AI/ML engine configuration issue. Please:\n"
                              "• Try selecting 'Sophisticated Fallback' engine\n"
                              "• Contact system administrator if problem persists\n"
                              "• Check that all required modules are installed")
            else:
                raise UserError(f"Composition generation failed: {error_msg}\n\n"
                              "Please try:\n"
                              "• Using 'Sophisticated Fallback' engine\n"
                              "• Adjusting budget or restrictions\n"
                              "• Contacting administrator if issue persists")
    
    def action_analyze_client(self):
        """Perform deep client analysis for better recommendations"""
        
        if not self.partner_id:
            raise UserError("Please select a client first")
        
        try:
            # Get integration manager
            integration_manager = self._get_integration_manager()
            
            # Build comprehensive client profile
            client_profile = integration_manager._build_client_profile(
                self.partner_id.id, 
                self.additional_notes or ""
            )
            
            # Create detailed analysis report
            analysis_report = self._build_analysis_report(client_profile)
            
            # Return action to show analysis
            return {
                'type': 'ir.actions.act_window',
                'name': f'Client Analysis - {self.partner_id.name}',
                'res_model': 'ir.actions.act_window',
                'view_mode': 'form',
                'target': 'new',
                'context': {
                    'default_name': f'Analysis for {self.partner_id.name}',
                    'analysis_content': analysis_report
                }
            }
            
        except Exception as e:
            raise UserError(f"Client analysis failed: {str(e)}")
    
    # Helper Methods
    
    def _parse_dietary_restrictions(self):
        """Parse dietary restrictions selection into list"""
        
        dietary_list = []
        if self.dietary_restrictions and self.dietary_restrictions != 'none':
            if 'vegan' in self.dietary_restrictions:
                dietary_list.append('vegan')
            if 'halal' in self.dietary_restrictions:
                dietary_list.append('halal')
            if 'non_alcoholic' in self.dietary_restrictions:
                dietary_list.append('non_alcoholic')
        
        return dietary_list
    
    def _get_integration_manager(self):
        """Get or create integration manager"""
        
        integration_manager = self.env['integration.manager'].search([], limit=1)
        
        if not integration_manager:
            integration_manager = self.env['integration.manager'].create({
                'name': 'Advanced AI/ML Integration Manager',
                'use_ml_engine': True,
                'use_ai_recommender': True,
                'fallback_strategy': 'cascade'
            })
            _logger.info("Created new integration manager")
        
        return integration_manager
    
    def _build_client_dashboard(self, client_profile):
        """Build rich HTML client dashboard"""
        
        dashboard_html = "<div class='container-fluid'>"
        
        # Client Overview
        dashboard_html += f"""
        <div class='row mb-3'>
            <div class='col-md-12'>
                <div class='card border-primary'>
                    <div class='card-header bg-primary text-white'>
                        <h6 class='mb-0'>📊 Client Intelligence Dashboard</h6>
                    </div>
                    <div class='card-body'>
        """
        
        # Key Metrics Row
        dashboard_html += "<div class='row text-center'>"
        
        # History Status
        history_icon = "✅" if client_profile.get('has_history') else "🆕"
        history_text = f"{client_profile.get('order_count', 0)} orders" if client_profile.get('has_history') else "New Client"
        
        dashboard_html += f"""
        <div class='col-md-3'>
            <div class='border rounded p-2'>
                <div style='font-size: 24px;'>{history_icon}</div>
                <div><strong>History</strong></div>
                <div class='text-muted'>{history_text}</div>
            </div>
        </div>
        """
        
        # Budget Pattern
        avg_budget = client_profile.get('avg_budget', 200)
        dashboard_html += f"""
        <div class='col-md-3'>
            <div class='border rounded p-2'>
                <div style='font-size: 24px;'>💰</div>
                <div><strong>Avg Budget</strong></div>
                <div class='text-muted'>€{avg_budget:.0f}</div>
            </div>
        </div>
        """
        
        # Risk Level
        risk_level = client_profile.get('risk_level', 'medium')
        risk_icons = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
        dashboard_html += f"""
        <div class='col-md-3'>
            <div class='border rounded p-2'>
                <div style='font-size: 24px;'>{risk_icons.get(risk_level, '🟡')}</div>
                <div><strong>Risk Level</strong></div>
                <div class='text-muted'>{risk_level.title()}</div>
            </div>
        </div>
        """
        
        # Preferences
        prefs_count = len(client_profile.get('preferred_categories', []))
        dashboard_html += f"""
        <div class='col-md-3'>
            <div class='border rounded p-2'>
                <div style='font-size: 24px;'>🎯</div>
                <div><strong>Preferences</strong></div>
                <div class='text-muted'>{prefs_count} categories</div>
            </div>
        </div>
        """
        
        dashboard_html += "</div>"  # End metrics row
        
        # Detailed Information
        if client_profile.get('preferred_categories'):
            dashboard_html += f"""
            <div class='mt-3'>
                <strong>Preferred Categories:</strong>
                <div class='mt-1'>
                    {', '.join([cat.title().replace('_', ' ') for cat in client_profile['preferred_categories']])}
                </div>
            </div>
            """
        
        # Notes Analysis
        if client_profile.get('notes_analysis'):
            notes_analysis = client_profile['notes_analysis']
            if notes_analysis.get('positive_keywords'):
                dashboard_html += f"""
                <div class='mt-3'>
                    <strong>Notes Insights:</strong>
                    <div class='mt-1'>
                        <span class='badge badge-success'>Positive: {', '.join(notes_analysis['positive_keywords'])}</span>
                    </div>
                </div>
                """
        
        dashboard_html += """
                    </div>
                </div>
            </div>
        </div>
        </div>
        """
        
        return dashboard_html
    
    def _check_ml_engine_status(self):
        """Check ML engine availability and training status"""
        
        try:
            ml_engines = self.env['ml.recommendation.engine'].search([])
            
            if not ml_engines:
                return {
                    'status': 'Not Available',
                    'details': 'No ML engine configured',
                    'css_class': 'border-secondary'
                }
            
            ml_engine = ml_engines[0]
            
            if getattr(ml_engine, 'is_model_trained', False):
                accuracy = getattr(ml_engine, 'model_accuracy', 0)
                return {
                    'status': 'Ready',
                    'details': f'Trained model ({accuracy:.1f}% accuracy)',
                    'css_class': 'border-success'
                }
            else:
                return {
                    'status': 'Not Trained',
                    'details': 'Model training required',
                    'css_class': 'border-warning'
                }
                
        except Exception as e:
            return {
                'status': 'Error',
                'details': 'Status check failed',
                'css_class': 'border-danger'
            }
    
    def _check_ai_engine_status(self):
        """Check AI engine availability"""
        
        try:
            # Check if AI recommender exists
            ai_engines = self.env['ai.product.recommender'].search([])
            
            if ai_engines:
                return {
                    'status': 'Ready',
                    'details': 'AI algorithms available',
                    'css_class': 'border-success'
                }
            else:
                return {
                    'status': 'Fallback',
                    'details': 'Using built-in intelligence',
                    'css_class': 'border-info'
                }
                
        except Exception as e:
            return {
                'status': 'Ready',
                'details': 'Built-in AI available',
                'css_class': 'border-info'
            }
    
    def _generate_ai_recommendations(self, record):
        """Generate AI-powered recommendations display"""
        
        recommendations_html = "<div class='card border-info'>"
        recommendations_html += """
        <div class='card-header bg-info text-white'>
            <h6 class='mb-0'>🤖 AI Recommendations</h6>
        </div>
        <div class='card-body'>
        """
        
        # Budget recommendations
        if record.target_budget:
            if record.target_budget < 100:
                recommendations_html += """
                <div class='alert alert-warning mb-2'>
                    <strong>Budget Advisory:</strong> Consider increasing budget for better product variety
                </div>
                """
            elif record.target_budget > 500:
                recommendations_html += """
                <div class='alert alert-info mb-2'>
                    <strong>Premium Budget:</strong> Luxury selection and premium products recommended
                </div>
                """
        
        # Dietary recommendations
        if record.dietary_restrictions != 'none':
            recommendations_html += """
            <div class='alert alert-success mb-2'>
                <strong>Dietary Compliance:</strong> Products will be filtered for specified restrictions
            </div>
            """
        
        # Engine recommendations
        if record.risk_level == 'high':
            recommendations_html += """
            <div class='alert alert-primary mb-2'>
                <strong>New Client Strategy:</strong> Experience-based compositions recommended for discovery
            </div>
            """
        elif record.risk_level == 'low':
            recommendations_html += """
            <div class='alert alert-primary mb-2'>
                <strong>Established Client:</strong> ML engine can leverage historical preferences
            </div>
            """
        
        recommendations_html += """
        </div>
        </div>
        """
        
        return recommendations_html
    
    def _show_success_notification(self, composition):
        """Show success notification with composition details"""
        
        try:
            message = f"""
            Composition generated successfully!
            
            📦 Products: {len(composition.product_ids)}
            💰 Cost: €{composition.actual_cost:.2f}
            🎯 Budget: €{composition.target_budget:.2f}
            """
            
            # You can extend this with more sophisticated notifications
            _logger.info(f"Success notification: {message}")
            
        except Exception as e:
            _logger.warning(f"Success notification failed: {e}")
    
    def _build_analysis_report(self, client_profile):
        """Build detailed analysis report"""
        
        report = f"""
        CLIENT ANALYSIS REPORT
        =====================
        
        Client: {client_profile.get('name', 'Unknown')}
        Analysis Date: {fields.Datetime.now()}
        
        OVERVIEW:
        - Order History: {client_profile.get('order_count', 0)} orders
        - Average Budget: €{client_profile.get('avg_budget', 0):.2f}
        - Risk Level: {client_profile.get('risk_level', 'Unknown').title()}
        
        PREFERENCES:
        - Categories: {', '.join(client_profile.get('preferred_categories', ['None']))}
        
        RECOMMENDATIONS:
        - Suitable for {'ML engine' if client_profile.get('has_history') else 'AI discovery engine'}
        - {'Historical patterns available' if client_profile.get('has_history') else 'Focus on category diversity'}
        """
        
        return report