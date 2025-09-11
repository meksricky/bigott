# wizard/ollama_recommendation_wizard.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import logging

_logger = logging.getLogger(__name__)

class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'Ollama Gift Recommendation Wizard'
    
    # Client Selection
    partner_id = fields.Many2one('res.partner', string="Client", required=False,
                                domain=[('is_company', '=', False)])
    
    # Budget
    target_budget = fields.Float(string="Target Budget (€)", required=True, default=100.0)
    
    # Client Notes
    client_notes = fields.Text(string="Client Notes", 
                              help="Special requests, preferences, occasion details, etc.")
    
    # Dietary Restrictions
    dietary_restrictions = fields.Text(string="Dietary Restrictions",
                                      help="Enter restrictions separated by commas (e.g., vegan, halal, gluten_free, non_alcoholic)")
    
    # Quick dietary restriction checkboxes
    is_vegan = fields.Boolean(string="Vegan")
    is_halal = fields.Boolean(string="Halal")
    is_gluten_free = fields.Boolean(string="Gluten Free") 
    is_non_alcoholic = fields.Boolean(string="Non-Alcoholic")
    
    # Recommender Selection
    recommender_id = fields.Many2one('ollama.gift.recommender', string="Recommender",
                                    default=lambda self: self._default_recommender())
    
    # Results (populated after generation)
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating...'),
        ('done', 'Complete'),
        ('error', 'Error')
    ], default='draft', string="State")
    
    result_message = fields.Text(string="Result Message", readonly=True)
    composition_id = fields.Many2one('gift.composition', string="Generated Composition", readonly=True)
    error_message = fields.Text(string="Error Details", readonly=True)
    
    # Display fields for results
    recommended_products = fields.Many2many('product.template', string="Recommended Products", readonly=True)
    total_cost = fields.Float(string="Total Cost (€)", readonly=True)
    confidence_score = fields.Float(string="Confidence Score", readonly=True)
    
    @api.model
    def _default_recommender(self):
        """Get default recommender"""
        recommender = self.env['ollama.gift.recommender'].search([('active', '=', True)], limit=1)
        if not recommender:
            # Create default recommender if none exists
            recommender = self.env['ollama.gift.recommender'].create({
                'name': 'Default Ollama Recommender'
            })
        return recommender
    
    @api.onchange('is_vegan', 'is_halal', 'is_gluten_free', 'is_non_alcoholic')
    def _onchange_dietary_checkboxes(self):
        """Update dietary restrictions text based on checkboxes"""
        restrictions = []
        
        if self.is_vegan:
            restrictions.append('vegan')
        if self.is_halal:
            restrictions.append('halal')
        if self.is_gluten_free:
            restrictions.append('gluten_free')
        if self.is_non_alcoholic:
            restrictions.append('non_alcoholic')
        
        if restrictions:
            # Merge with existing restrictions
            existing = []
            if self.dietary_restrictions:
                existing = [r.strip() for r in self.dietary_restrictions.split(',') if r.strip()]
            
            # Combine and deduplicate
            all_restrictions = list(set(restrictions + existing))
            self.dietary_restrictions = ', '.join(all_restrictions)
    
    @api.constrains('target_budget')
    def _check_target_budget(self):
        """Validate target budget"""
        for record in self:
            if record.target_budget <= 0:
                raise ValidationError("Target budget must be greater than 0.")
            if record.target_budget > 10000:
                raise ValidationError("Target budget seems unusually high. Please confirm.")
    
    def action_generate_recommendation(self):
        """Generate recommendation using Ollama"""
        self.ensure_one()
        
        if not self.partner_id:
            raise UserError("Please select a client.")
        
        if not self.recommender_id:
            raise UserError("No recommender available. Please configure an Ollama recommender first.")
        
        try:
            # Update state
            self.state = 'generating'
            
            # Parse dietary restrictions
            dietary_restrictions = []
            if self.dietary_restrictions:
                dietary_restrictions = [r.strip() for r in self.dietary_restrictions.split(',') if r.strip()]
            
            # Generate recommendation
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=self.target_budget,
                client_notes=self.client_notes,
                dietary_restrictions=dietary_restrictions
            )
            
            if result['success']:
                # Update wizard with results
                self.state = 'done'
                self.result_message = result['message']
                self.composition_id = result['composition_id']
                self.recommended_products = [(6, 0, [p.id for p in result['products']])]
                self.total_cost = result['total_cost']
                self.confidence_score = result['confidence_score']
                
                # Return action to view results
                return {
                    'type': 'ir.actions.act_window',
                    'name': 'Recommendation Results',
                    'res_model': 'ollama.recommendation.wizard',
                    'res_id': self.id,
                    'view_mode': 'form',
                    'target': 'new',
                    'context': {'show_results': True}
                }
            else:
                # Handle error
                self.state = 'error'
                self.error_message = result.get('error', 'Unknown error occurred')
                self.result_message = result['message']
                
                raise UserError(f"Recommendation failed: {result['message']}")
                
        except Exception as e:
            self.state = 'error'
            self.error_message = str(e)
            _logger.error(f"Recommendation wizard error: {str(e)}")
            raise
    
    def action_view_composition(self):
        """View the generated composition"""
        self.ensure_one()
        
        if not self.composition_id:
            raise UserError("No composition generated yet.")
        
        return {
            'type': 'ir.actions.act_window',
            'name': f'Gift Composition for {self.partner_id.name}',
            'res_model': 'gift.composition',
            'res_id': self.composition_id.id,
            'view_mode': 'form',
            'target': 'current'
        }
    
    def action_generate_another(self):
        """Generate another recommendation for the same client"""
        self.ensure_one()
        
        # Create new wizard with same client and budget
        new_wizard = self.create({
            'partner_id': self.partner_id.id,
            'target_budget': self.target_budget,
            'client_notes': self.client_notes,
            'dietary_restrictions': self.dietary_restrictions,
            'recommender_id': self.recommender_id.id
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Generate Another Recommendation',
            'res_model': 'ollama.recommendation.wizard',
            'res_id': new_wizard.id,
            'view_mode': 'form',
            'target': 'new'
        }
    
    def action_test_recommender_connection(self):
        """Test the Ollama connection"""
        self.ensure_one()
        
        if not self.recommender_id:
            raise UserError("Please select a recommender first.")
        
        result = self.recommender_id.test_ollama_connection()
        
        if result['success']:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Connection Test Successful',
                    'message': result['message'],
                    'type': 'success',
                    'sticky': False
                }
            }
        else:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Connection Test Failed',
                    'message': result['message'],
                    'type': 'danger',
                    'sticky': True
                }
            }