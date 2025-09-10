from odoo import models, fields, api
from odoo.exceptions import UserError

class CompositionWizard(models.TransientModel):
    _name = 'composition.wizard'
    _description = 'Generate Gift Composition Wizard'
    
    partner_id = fields.Many2one('res.partner', 'Client', required=True, 
                                 domain="[('is_company', '=', False)]")
    target_year = fields.Integer('Target Year', required=True, 
                                 default=lambda self: fields.Date.context_today(self).year)
    target_budget = fields.Float('Target Budget (€)', required=True, default=200.0)
    
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
    
    force_composition_type = fields.Selection([
        ('auto', 'Let Señor Bigott Decide'),
        ('experience', 'Force Experience-Based'),
        ('custom', 'Force Custom Composition')
    ], string='Composition Type', default='auto')

    # Free text notes from operator to influence recommendations
    additional_notes = fields.Text('Additional Notes')

    partner_dietary_restrictions = fields.Selection(
        related='partner_id.dietary_restrictions', 
        string='Client Dietary Restrictions',
        readonly=True
    )
    
    # Client insights display
    client_info = fields.Html('Client Information', compute='_compute_client_info')

    budget_recommendation = fields.Char('Budget Recommendation', compute='_compute_recommendations')
    approach_recommendation = fields.Char('Approach Recommendation', compute='_compute_recommendations')
    risk_level = fields.Selection([
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'), 
        ('high', 'High Risk')
    ], compute='_compute_recommendations')

    @api.depends('partner_id')
    def _compute_client_info(self):
        for record in self:
            if record.partner_id:
                # Get client history analysis
                try:
                    client_history = self.env['client.order.history'].analyze_client_patterns(record.partner_id.id)
                    
                    info_html = "<div style='background: #f8f9fa; padding: 10px; border-radius: 5px;'>"
                    
                    if client_history.get('has_history'):
                        info_html += f"<p><strong>📊 Client History:</strong> {client_history.get('years_of_data', 0)} years</p>"
                        info_html += f"<p><strong>💰 Average Budget:</strong> €{client_history.get('average_budget', 0):.2f}</p>"
                        info_html += f"<p><strong>🎁 Preferred Type:</strong> {client_history.get('box_type_preference', 'Mixed').title()}</p>"
                        info_html += f"<p><strong>⭐ Satisfaction:</strong> {client_history.get('average_satisfaction', 0):.1f}/5</p>"
                    else:
                        info_html += "<p><strong>🆕 New Client:</strong> No purchase history available</p>"
                        info_html += "<p>The AI will use market trends and preferences analysis.</p>"
                    
                    info_html += "</div>"
                    record.client_info = info_html
                except:
                    record.client_info = "<p>Client analysis not available.</p>"
            else:
                record.client_info = ""

    @api.depends('partner_id', 'target_budget')
    def _compute_recommendations(self):
        for record in self:
            if record.partner_id:
                try:
                    client_history = self.env['client.order.history'].analyze_client_patterns(record.partner_id.id)
                    
                    # Budget recommendation
                    if client_history.get('has_history'):
                        avg_budget = client_history.get('average_budget', 200)
                        if record.target_budget < avg_budget * 0.8:
                            record.budget_recommendation = f"Consider increasing to €{avg_budget:.0f} (client's average)"
                        elif record.target_budget > avg_budget * 1.5:
                            record.budget_recommendation = f"High budget vs. average (€{avg_budget:.0f})"
                        else:
                            record.budget_recommendation = "Budget looks appropriate"
                    else:
                        record.budget_recommendation = "Standard budget for new client"
                    
                    # Approach recommendation
                    if client_history.get('has_history'):
                        preferred_type = client_history.get('box_type_preference', 'custom')
                        record.approach_recommendation = f"Recommend {preferred_type} composition"
                        record.risk_level = 'low'
                    else:
                        record.approach_recommendation = "Experience-based recommended for new clients"
                        record.risk_level = 'high'
                        
                except:
                    record.budget_recommendation = "Analysis not available"
                    record.approach_recommendation = "Standard approach"
                    record.risk_level = 'medium'
            else:
                record.budget_recommendation = ""
                record.approach_recommendation = ""
                record.risk_level = 'medium'

    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Auto-populate fields when partner is selected"""
        if self.partner_id:
            # Set dietary restrictions from partner
            if self.partner_id.dietary_restrictions and self.partner_id.dietary_restrictions != 'none':
                self.dietary_restrictions = self.partner_id.dietary_restrictions
            
            # Suggest budget based on historical data
            if hasattr(self.partner_id, 'average_annual_budget') and self.partner_id.average_annual_budget and self.partner_id.average_annual_budget > 0:
                # Suggest 10% increase from average
                self.target_budget = self.partner_id.average_annual_budget * 1.1
    
    def action_generate_composition(self):
        """Generate composition using Integration Manager (FIXED VERSION)"""
        
        if self.target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
        
        # Parse dietary restrictions
        dietary_list = []
        if self.dietary_restrictions != 'none':
            if 'vegan' in self.dietary_restrictions:
                dietary_list.append('vegan')
            if 'halal' in self.dietary_restrictions:
                dietary_list.append('halal')
            if 'non_alcoholic' in self.dietary_restrictions:
                dietary_list.append('non_alcoholic')
        
        # Force composition type if specified
        force_type = None
        if self.force_composition_type != 'auto':
            force_type = self.force_composition_type
        
        try:
            # ✅ FIXED: Use Integration Manager instead of composition.engine
            integration_manager = self.env['integration.manager']
            
            # Get or create the default integration manager
            if not integration_manager.search([]):
                integration_manager = integration_manager.create({
                    'name': 'Default Integration Manager',
                    'use_ml_engine': True,
                    'use_ai_recommender': True,
                    'use_stock_aware': True,
                    'use_business_rules': True,
                    'fallback_strategy': 'cascade'
                })
            else:
                integration_manager = integration_manager.search([], limit=1)
            
            # Generate composition using the new integrated system
            composition = integration_manager.generate_complete_composition(
                partner_id=self.partner_id.id,
                target_budget=self.target_budget,
                target_year=self.target_year,
                dietary_restrictions=dietary_list,
                notes_text=self.additional_notes,
                use_batch=False,
                attempt_number=1,
                force_engine=None  # Let the system decide the best engine
            )
            
            if not composition:
                raise UserError("Failed to generate composition. Please check system configuration.")
            
            # Return action to view the generated composition
            return {
                'type': 'ir.actions.act_window',
                'name': f'Generated Composition - {composition.name}',
                'res_model': 'gift.composition',
                'res_id': composition.id,
                'view_mode': 'form',
                'target': 'current',
                'context': {
                    'default_composition_id': composition.id
                }
            }
            
        except Exception as e:
            # Enhanced error handling with specific messages
            error_msg = str(e)
            
            if "No products available" in error_msg:
                raise UserError("No suitable products found. Please check:\n"
                              "• Product categories are properly set\n"
                              "• Products have prices and are active\n"
                              "• Dietary restrictions aren't too restrictive")
            elif "ML recommendation failed" in error_msg:
                raise UserError("ML system error. Falling back to AI recommender.\n"
                              "Please try again or contact administrator.")
            elif "Integration Manager" in error_msg:
                raise UserError("System configuration error. Please check:\n"
                              "• Integration Manager is properly configured\n"
                              "• All engines are enabled\n"
                              "• ML model is trained")
            else:
                raise UserError(f"Composition generation failed: {error_msg}\n\n"
                              "Please check system logs for details.")