# -*- coding: utf-8 -*-
# wizard/ollama_recommendation_wizard.py
# HYBRID VERSION - Combining working composition wizard logic with Ollama features

from odoo import models, fields, api, _
from odoo.exceptions import UserError, ValidationError
import json
import logging

_logger = logging.getLogger(__name__)


class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'AI-Powered Gift Recommendation Wizard'

    # === BASIC FIELDS (from composition wizard that works) ===
    partner_id = fields.Many2one(
        'res.partner', 
        string='Client', 
        required=True,
        domain="[('is_company', '=', False)]"  # Only individuals
    )
    
    target_year = fields.Integer(
        string='Target Year', 
        required=True,
        default=lambda self: fields.Date.context_today(self).year
    )
    
    target_budget = fields.Float(
        string='Target Budget (€)', 
        required=True, 
        default=200.0
    )
    
    # === DIETARY RESTRICTIONS (enhanced) ===
    dietary_restrictions = fields.Selection([
        ('none', 'None'),
        ('vegan', 'Vegan'),
        ('halal', 'Halal'),
        ('non_alcoholic', 'Non-Alcoholic'),
        ('gluten_free', 'Gluten Free'),
        ('vegan_halal', 'Vegan + Halal'),
        ('vegan_non_alcoholic', 'Vegan + Non-Alcoholic'),
        ('halal_non_alcoholic', 'Halal + Non-Alcoholic'),
        ('all_restrictions', 'All Restrictions')
    ], string='Dietary Restrictions', default='none')
    
    # === OLLAMA SPECIFIC FIELDS ===
    use_ollama = fields.Boolean(
        string='Use AI Assistant', 
        default=True,
        help='Use Ollama AI for intelligent recommendations'
    )
    
    recommender_id = fields.Many2one(
        'ollama.gift.recommender',
        string='AI Recommender',
        default=lambda self: self._get_or_create_recommender()
    )
    
    client_notes = fields.Text(
        string='Special Requirements',
        help='Any special preferences, occasion details, or requirements...'
    )
    
    # === ENGINE SELECTION ===
    engine_type = fields.Selection([
        ('auto', 'Auto-Select (Recommended)'),
        ('ollama', 'Ollama AI Engine'),
        ('composition', 'Standard Composition Engine'),
        ('hybrid', 'Hybrid (AI + Business Rules)')
    ], string='Engine Type', default='auto')
    
    # === STATE MANAGEMENT ===
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating...'),
        ('done', 'Complete'),
        ('error', 'Error')
    ], default='draft', readonly=True)
    
    # === RESULTS FIELDS ===
    composition_id = fields.Many2one('gift.composition', readonly=True)
    recommended_products = fields.Many2many(
        'product.template', 
        string='Recommended Products', 
        readonly=True
    )
    total_cost = fields.Float(string='Total Cost', readonly=True)
    confidence_score = fields.Float(string='AI Confidence', readonly=True)
    result_message = fields.Html(string='Results', readonly=True)
    error_message = fields.Text(readonly=True)
    
    # === CLIENT INTELLIGENCE (from composition wizard) ===
    client_info = fields.Html(
        string='Client Intelligence', 
        compute='_compute_client_info'
    )
    
    # === HELPER METHODS ===
    
    @api.model
    def _get_or_create_recommender(self):
        """Get existing or create default recommender"""
        recommender = self.env['ollama.gift.recommender'].search([
            ('active', '=', True)
        ], limit=1)
        
        if not recommender:
            recommender = self.env['ollama.gift.recommender'].create({
                'name': 'Default AI Recommender',
                'active': True
            })
        return recommender.id
    
    @api.depends('partner_id', 'target_budget')
    def _compute_client_info(self):
        """Compute client intelligence display (working logic from composition wizard)"""
        for wizard in self:
            if not wizard.partner_id:
                wizard.client_info = "<p>Select a client to see their information</p>"
                continue
            
            # Build client profile HTML
            info_html = f"<h4>{wizard.partner_id.name}</h4>"
            
            # Get purchase history
            history = self.env['client.order.history'].search([
                ('partner_id', '=', wizard.partner_id.id)
            ], order='order_year desc', limit=5)
            
            if history:
                info_html += "<h5>Purchase History:</h5><ul>"
                for h in history:
                    info_html += f"""
                    <li>
                        <strong>{h.order_year}:</strong> 
                        €{h.total_budget:.2f} - {h.box_type or 'Standard'} box
                    </li>
                    """
                info_html += "</ul>"
                
                # Average budget
                avg_budget = sum(h.total_budget for h in history) / len(history)
                if wizard.target_budget and avg_budget:
                    variance = ((wizard.target_budget - avg_budget) / avg_budget) * 100
                    if variance > 20:
                        info_html += f"""
                        <div class='alert alert-info'>
                            Budget is {variance:.0f}% higher than average (€{avg_budget:.2f})
                        </div>
                        """
                    elif variance < -20:
                        info_html += f"""
                        <div class='alert alert-warning'>
                            Budget is {abs(variance):.0f}% lower than average (€{avg_budget:.2f})
                        </div>
                        """
            else:
                info_html += "<p><em>New client - no purchase history</em></p>"
            
            wizard.client_info = info_html
    
    def _parse_dietary_restrictions(self):
        """Parse dietary restrictions into list format"""
        if self.dietary_restrictions == 'none':
            return []
        elif self.dietary_restrictions == 'all_restrictions':
            return ['vegan', 'halal', 'non_alcoholic', 'gluten_free']
        elif '_' in self.dietary_restrictions:
            # Handle combined restrictions
            return self.dietary_restrictions.split('_')
        else:
            return [self.dietary_restrictions]
    
    # === MAIN ACTION METHOD ===
    
    def action_generate_recommendation(self):
        """Generate recommendation without composition.engine dependency"""
        self.ensure_one()
        
        # Validation
        if not self.partner_id:
            raise UserError(_("Please select a client"))
        
        if self.target_budget <= 0:
            raise ValidationError(_("Target budget must be greater than 0"))
        
        try:
            # Update state
            self.write({'state': 'generating'})
            self.env.cr.commit()
            
            # Parse dietary restrictions
            dietary_list = self._parse_dietary_restrictions()
            
            # Get recommender
            if not self.recommender_id:
                self.recommender_id = self._get_or_create_recommender()
            
            # Generate using recommender
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=self.target_budget,
                client_notes=self.client_notes or '',
                dietary_restrictions=dietary_list
            )
            
            if result and result.get('success'):
                # Process successful result
                composition_id = result.get('composition_id')
                if composition_id:
                    composition = self.env['gift.composition'].browse(composition_id)
                    
                    # Calculate variance
                    budget_variance = ((composition.actual_cost - self.target_budget) / self.target_budget * 100) if self.target_budget else 0
                    
                    result_html = f"""
                    <div class="alert alert-success">
                        <h4>✓ Recommendation Generated Successfully!</h4>
                    </div>
                    <div class="mt-3">
                        <strong>Composition:</strong> {composition.name}<br/>
                        <strong>Products:</strong> {len(composition.product_ids)} items<br/>
                        <strong>Total Cost:</strong> €{composition.actual_cost:.2f}<br/>
                        <strong>Budget Variance:</strong> {budget_variance:.1f}%
                    </div>
                    """
                    
                    # Update wizard
                    self.write({
                        'state': 'done',
                        'composition_id': composition_id,
                        'recommended_products': [(6, 0, composition.product_ids.ids)],
                        'total_cost': composition.actual_cost,
                        'confidence_score': result.get('confidence_score', 0.85),
                        'result_message': result_html
                    })
                    
                    return {
                        'type': 'ir.actions.act_window',
                        'name': 'Gift Recommendation Results',
                        'res_model': 'ollama.recommendation.wizard',
                        'res_id': self.id,
                        'view_mode': 'form',
                        'target': 'new',
                        'context': {'show_results': True}
                    }
            
            # If we get here, generation failed
            raise UserError(_("Failed to generate recommendations"))
            
        except Exception as e:
            error_msg = str(e)
            error_html = f"""
            <div class="alert alert-danger">
                <h4>✗ Generation Failed</h4>
                <p>{error_msg}</p>
            </div>
            """
            
            self.write({
                'state': 'error',
                'error_message': error_msg,
                'result_message': error_html
            })
            _logger.exception("Recommendation generation error")
            raise
    
    def action_view_composition(self):
        """Open the generated composition"""
        self.ensure_one()
        if not self.composition_id:
            raise UserError(_("No composition generated yet"))
        
        return {
            'type': 'ir.actions.act_window',
            'name': f'Gift Composition - {self.partner_id.name}',
            'res_model': 'gift.composition',
            'res_id': self.composition_id.id,
            'view_mode': 'form',
            'target': 'current'
        }
    
    def action_generate_another(self):
        """Reset and generate another"""
        self.ensure_one()
        
        # Create new wizard with same parameters
        new_wizard = self.create({
            'partner_id': self.partner_id.id,
            'target_year': self.target_year,
            'target_budget': self.target_budget,
            'dietary_restrictions': self.dietary_restrictions,
            'client_notes': self.client_notes,
            'use_ollama': self.use_ollama,
            'recommender_id': self.recommender_id.id,
            'engine_type': self.engine_type
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Generate Another Recommendation',
            'res_model': 'ollama.recommendation.wizard',
            'res_id': new_wizard.id,
            'view_mode': 'form',
            'target': 'new'
        }
    
    def action_test_connection(self):
        """Test Ollama connection"""
        self.ensure_one()
        
        if not self.recommender_id:
            self.recommender_id = self._get_or_create_recommender()
        
        result = self.recommender_id.test_ollama_connection()
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Connection Test',
                'message': result.get('message', 'Test completed'),
                'type': 'success' if result.get('success') else 'danger',
                'sticky': False
            }
        }