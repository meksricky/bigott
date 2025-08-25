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
        for wizard in self:
            if wizard.partner_id:
                partner = wizard.partner_id
                
                # Safe field access with defaults
                years_as_client = getattr(partner, 'years_as_client', 0) or 0
                client_tier = getattr(partner, 'client_tier', 'new') or 'new'
                avg_budget = getattr(partner, 'average_annual_budget', 0.0) or 0.0
                preferred_type = getattr(partner, 'preferred_box_type', 'custom') or 'custom'
                satisfaction = getattr(partner, 'client_satisfaction_avg', 0.0) or 0.0
                dietary = getattr(partner, 'dietary_restrictions', 'none') or 'none'

                # FIXED: Use simple div structure that Odoo can render properly
                wizard.client_info = f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 10px;">
                        <div>
                            <strong style="color: #495057;">Client Tier</strong><br/>
                            <span style="background: #007bff; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">
                                {client_tier.upper()}
                            </span>
                        </div>
                        <div>
                            <strong style="color: #495057;">Experience</strong><br/>
                            <span style="color: #007bff; font-weight: bold;">{years_as_client} years</span>
                        </div>
                        <div>
                            <strong style="color: #495057;">Avg Budget</strong><br/>
                            <span style="color: #28a745; font-weight: bold;">€{avg_budget:.0f}</span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                        <div>
                            <strong style="color: #495057;">Prefers</strong><br/>
                            <span style="background: #ffc107; color: #212529; padding: 3px 8px; border-radius: 12px; font-size: 12px;">
                                {preferred_type.upper()}
                            </span>
                        </div>
                        <div>
                            <strong style="color: #495057;">Satisfaction</strong><br/>
                            <span style="color: #dc3545; font-weight: bold;">{satisfaction:.1f}/5 ⭐</span>
                        </div>
                        <div>
                            <strong style="color: #495057;">Diet</strong><br/>
                            <span style="background: #6c757d; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">
                                {dietary.replace('_', ' ').upper()}
                            </span>
                        </div>
                    </div>
                </div>
                """
            else:
                wizard.client_info = """
                <div style="text-align: center; padding: 40px; color: #6c757d;">
                    <i class="fa fa-user-plus" style="font-size: 48px; margin-bottom: 15px;"></i><br/>
                    <strong>Select a client to see their profile insights</strong><br/>
                    <small>Señor Bigott will analyze their preferences and history</small>
                </div>
                """

    @api.depends('partner_id', 'target_budget')
    def _compute_recommendations(self):
        for wizard in self:
            if wizard.partner_id:
                partner = wizard.partner_id
                
                # Budget recommendation logic
                if hasattr(partner, 'average_annual_budget') and partner.average_annual_budget > 0:
                    if wizard.target_budget > partner.average_annual_budget * 1.2:
                        wizard.budget_recommendation = "Above typical range - high upsell opportunity"
                    elif wizard.target_budget < partner.average_annual_budget * 0.8:
                        wizard.budget_recommendation = "Below typical range - consider increasing"
                    else:
                        wizard.budget_recommendation = "Within optimal range"
                else:
                    wizard.budget_recommendation = "New client - tier-based recommendation"
                
                # Approach recommendation
                if hasattr(partner, 'client_tier') and partner.client_tier == 'new':
                    wizard.approach_recommendation = "Experience-based recommended for onboarding"
                elif hasattr(partner, 'preferred_box_type') and getattr(partner, 'preferred_box_type', '') == 'experience':
                    wizard.approach_recommendation = "Experience-based preferred by client"
                else:
                    wizard.approach_recommendation = "Custom composition based on preferences"
                    
                # Risk assessment
                confidence = 1.0
                years_as_client = getattr(partner, 'years_as_client', 0) or 0
                avg_budget = getattr(partner, 'average_annual_budget', 0.0) or 0.0
                
                if years_as_client < 1:
                    confidence -= 0.3
                if not getattr(partner, 'dietary_restrictions', None):
                    confidence -= 0.1
                if avg_budget == 0:
                    confidence -= 0.2
                    
                if confidence >= 0.7:
                    wizard.risk_level = 'low'
                elif confidence >= 0.4:
                    wizard.risk_level = 'medium'
                else:
                    wizard.risk_level = 'high'
            else:
                wizard.budget_recommendation = False
                wizard.approach_recommendation = False
                wizard.risk_level = False
    
    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Auto-fill based on client data"""
        if self.partner_id:
            # Set dietary restrictions from client profile
            if hasattr(self.partner_id, 'dietary_restrictions') and self.partner_id.dietary_restrictions and self.partner_id.dietary_restrictions != 'none':
                self.dietary_restrictions = self.partner_id.dietary_restrictions
            
            # Suggest budget based on historical data
            if hasattr(self.partner_id, 'average_annual_budget') and self.partner_id.average_annual_budget and self.partner_id.average_annual_budget > 0:
                # Suggest 10% increase from average
                self.target_budget = self.partner_id.average_annual_budget * 1.1
    
    def action_generate_composition(self):
        """Generate composition using Señor Bigott engine"""
        
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
        
        # Generate composition
        engine = self.env['composition.engine']
        composition = engine.generate_composition(
            partner_id=self.partner_id.id,
            target_budget=self.target_budget,
            target_year=self.target_year,
            dietary_restrictions=dietary_list,
            force_type=force_type,
            notes_text=self.additional_notes
        )
        
        # Return action to view the generated composition
        return {
            'type': 'ir.actions.act_window',
            'name': f'Generated Composition - {composition.name}',
            'res_model': 'gift.composition',
            'res_id': composition.id,
            'view_mode': 'form',
            'target': 'current',
        }