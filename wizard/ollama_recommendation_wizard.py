# -*- coding: utf-8 -*-
# wizard/ollama_recommendation_wizard.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import logging

_logger = logging.getLogger(__name__)


class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'Ollama Gift Recommendation Wizard'

    # Currency for Monetary fields
    currency_id = fields.Many2one(
        'res.currency',
        default=lambda self: self.env.company.currency_id,
        required=True
    )

    # Client - with onchange to immediately save
    partner_id = fields.Many2one(
        'res.partner',
        string="Client",
        required=True,
        domain="[('is_company', '=', False)]"
    )
    
    # Hidden field to store partner ID
    stored_partner_id = fields.Integer(string="Stored Partner ID")

    # Budget
    target_budget = fields.Monetary(
        string="Target Budget",
        required=True,
        default=100.0,
        currency_field='currency_id'
    )

    # Notes & Dietary
    client_notes = fields.Text(string="Client Notes")
    dietary_restrictions = fields.Text(
        string="Dietary Restrictions",
        help="Comma-separated: vegan, halal, gluten_free, non_alcoholic"
    )
    is_vegan = fields.Boolean(string="Vegan")
    is_halal = fields.Boolean(string="Halal")
    is_gluten_free = fields.Boolean(string="Gluten Free")
    is_non_alcoholic = fields.Boolean(string="Non-Alcoholic")

    # Engine
    recommender_id = fields.Many2one(
        'ollama.gift.recommender',
        string="Recommender",
        default=lambda self: self.env['ollama.gift.recommender'].search([('active', '=', True)], limit=1)
    )

    # State & Results
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating...'),
        ('done', 'Complete'),
        ('error', 'Error'),
    ], default='draft')

    result_message = fields.Text(readonly=True)
    composition_id = fields.Many2one('gift.composition', readonly=True)
    error_message = fields.Text(readonly=True)

    recommended_products = fields.Many2many('product.template', readonly=True)
    total_cost = fields.Monetary(readonly=True, currency_field='currency_id')
    confidence_score = fields.Float(readonly=True)

    @api.onchange('partner_id')
    def _onchange_partner_id(self):
        """Store partner ID immediately when selected"""
        if self.partner_id:
            self.stored_partner_id = self.partner_id.id
            _logger.info(f"Partner selected and stored: {self.partner_id.name} (ID: {self.stored_partner_id})")

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
            existing = []
            if self.dietary_restrictions:
                existing = [r.strip() for r in self.dietary_restrictions.split(',') if r.strip()]
            self.dietary_restrictions = ', '.join(sorted(set(existing + toggles)))

    def action_generate_recommendation(self):
        """Generate recommendation using stored partner ID"""
        self.ensure_one()
        
        # Use the stored partner ID
        partner_id = self.stored_partner_id or (self.partner_id.id if self.partner_id else False)
        
        if not partner_id:
            # Try to get from context as last resort
            partner_id = self._context.get('default_partner_id') or self._context.get('active_id')
            
        if not partner_id:
            raise UserError("Please select a client. If you've selected one, please re-select to ensure it's registered.")
            
        # Get the partner record
        partner = self.env['res.partner'].browse(partner_id).exists()
        if not partner:
            raise UserError("Invalid client selected. Please select a valid client.")
            
        if self.target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
        
        try:
            _logger.info(f"Generating recommendation for {partner.name} (ID: {partner.id})")
            
            # Get or create recommender
            if not self.recommender_id:
                recommender = self.env['ollama.gift.recommender'].search([('active', '=', True)], limit=1)
                if not recommender:
                    recommender = self.env['ollama.gift.recommender'].create({
                        'name': 'Default Ollama Recommender',
                        'ollama_enabled': True
                    })
                self.recommender_id = recommender
            
            # Parse dietary restrictions
            dietary_list = []
            if self.dietary_restrictions:
                dietary_list = [r.strip() for r in self.dietary_restrictions.split(',') if r.strip()]
            
            # For now, just show success to test if partner is working
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Debug Success',
                    'message': f'Partner: {partner.name}, Budget: €{self.target_budget}',
                    'type': 'success',
                    'sticky': True
                }
            }
            
            # Uncomment below when partner issue is fixed
            """
            # Update state
            self.write({'state': 'generating'})
            
            # Generate recommendation
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=partner.id,
                target_budget=self.target_budget,
                client_notes=self.client_notes or '',
                dietary_restrictions=dietary_list
            )
            
            if result.get('success'):
                # Update wizard with results
                self.write({
                    'state': 'done',
                    'result_message': result.get('message', 'Recommendation generated successfully'),
                    'composition_id': result.get('composition_id'),
                    'recommended_products': [(6, 0, [p.id for p in result.get('products', [])])],
                    'total_cost': result.get('total_cost', 0.0),
                    'confidence_score': result.get('confidence_score', 0.0)
                })
                
                # Return the same wizard to show results
                return {
                    'type': 'ir.actions.act_window',
                    'name': 'Recommendation Results',
                    'res_model': 'ollama.recommendation.wizard',
                    'res_id': self.id,
                    'view_mode': 'form',
                    'target': 'new'
                }
            else:
                # Handle failure
                self.write({
                    'state': 'error',
                    'error_message': result.get('error', 'Unknown error occurred')
                })
                raise UserError(f"Recommendation failed: {result.get('message', 'Unknown error')}")
            """
                
        except Exception as e:
            self.write({
                'state': 'error',
                'error_message': str(e)
            })
            _logger.error(f"Recommendation generation failed: {str(e)}")
            raise

    def action_test_recommender_connection(self):
        """Test connection - simplified"""
        self.ensure_one()
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Test',
                'message': 'Button works!',
                'type': 'info',
            }
        }

    def action_view_composition(self):
        self.ensure_one()
        if not self.composition_id:
            raise UserError("No composition generated yet.")
        return {
            'type': 'ir.actions.act_window',
            'name': f'Gift Composition',
            'res_model': 'gift.composition',
            'res_id': self.composition_id.id,
            'view_mode': 'form',
            'target': 'current',
        }

    def action_generate_another(self):
        """Reset wizard for another generation"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'Generate New Recommendation',
            'res_model': 'ollama.recommendation.wizard',
            'view_mode': 'form',
            'target': 'new',
        }