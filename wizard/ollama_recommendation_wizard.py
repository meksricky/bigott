# -*- coding: utf-8 -*-
# wizard/ollama_recommendation_wizard.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import logging

_logger = logging.getLogger(__name__)


class OllamaRecommendationWizard(models.TransientModel):
    _name = 'ollama.recommendation.wizard'
    _description = 'Ollama Gift Recommendation Wizard'

    # --- Currency for Monetary fields ---
    currency_id = fields.Many2one(
        'res.currency',
        default=lambda self: self.env.company.currency_id,
        required=True
    )

    # --- Client ---
    partner_id = fields.Many2one(
        'res.partner',
        string="Client",
        required=True
    )

    # --- Budget ---
    target_budget = fields.Monetary(
        string="Target Budget",
        required=True,
        default=100.0,
        currency_field='currency_id'
    )

    # --- Notes & Dietary ---
    client_notes = fields.Text(string="Client Notes")
    dietary_restrictions = fields.Text(
        string="Dietary Restrictions",
        help="Comma-separated: vegan, halal, gluten_free, non_alcoholic"
    )
    is_vegan = fields.Boolean(string="Vegan")
    is_halal = fields.Boolean(string="Halal")
    is_gluten_free = fields.Boolean(string="Gluten Free")
    is_non_alcoholic = fields.Boolean(string="Non-Alcoholic")

    # --- Engine ---
    recommender_id = fields.Many2one(
        'ollama.gift.recommender',
        string="Recommender",
        default=lambda self: self._default_recommender()
    )

    # --- State & Results ---
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating...'),
        ('done', 'Complete'),
        ('error', 'Error'),
    ], default='draft')

    result_message = fields.Text(readonly=True)
    composition_id = fields.Many2one('gift.composition', readonly=True)
    error_message = fields.Text(readonly=True)

    recommended_products = fields.Many2many('product.template', string="Recommended Products", readonly=True)
    total_cost = fields.Monetary(string="Total Cost", readonly=True, currency_field='currency_id')
    confidence_score = fields.Float(string="Confidence Score", readonly=True)

    @api.model
    def default_get(self, fields_list):
        """Override default_get to handle context values properly"""
        res = super().default_get(fields_list)
        
        # Get partner from context
        if 'partner_id' not in res:
            # Try different context keys
            partner_id = self._context.get('default_partner_id') or \
                        self._context.get('active_id') if self._context.get('active_model') == 'res.partner' else False
            
            if partner_id:
                res['partner_id'] = partner_id
                
        return res

    @api.model
    def _default_recommender(self):
        """Get or create a default recommender"""
        rec = self.env['ollama.gift.recommender'].search([('active', '=', True)], limit=1)
        if not rec:
            rec = self.env['ollama.gift.recommender'].create({
                'name': 'Default Ollama Recommender',
                'active': True
            })
        return rec

    @api.onchange('is_vegan', 'is_halal', 'is_gluten_free', 'is_non_alcoholic')
    def _onchange_dietary_checkboxes(self):
        """Update dietary restrictions based on checkboxes"""
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
            all_restrictions = list(set(existing + toggles))
            self.dietary_restrictions = ', '.join(sorted(all_restrictions))

    @api.constrains('target_budget')
    def _check_target_budget(self):
        """Validate budget constraints"""
        for rec in self:
            if rec.target_budget <= 0:
                raise ValidationError("Target budget must be greater than 0.")
            if rec.target_budget > 1_000_000:
                raise ValidationError("Target budget seems unusually high. Please confirm.")

    def action_generate_recommendation(self):
        """SIMPLIFIED: Generate gift recommendations using Ollama"""
        self.ensure_one()

        # Simple partner check - no complex resolution
        if not self.partner_id:
            raise UserError("Please select a client.")

        if not self.recommender_id:
            # Try to create one
            self.recommender_id = self._default_recommender()
            if not self.recommender_id:
                raise UserError("No recommender available. Please configure an Ollama recommender first.")

        _logger.info("Generating recommendation for partner %s with budget %s", 
                    self.partner_id.name, self.target_budget)

        try:
            # Set state to generating
            self.write({'state': 'generating'})
            self.env.cr.commit()  # Commit to show the state change
            
            # Parse dietary restrictions
            dietary = []
            if self.dietary_restrictions:
                dietary = [r.strip() for r in self.dietary_restrictions.split(',') if r.strip()]

            # ALTERNATIVE: Use composition engine directly if Ollama fails
            try:
                # Try Ollama first
                result = self.recommender_id.generate_gift_recommendations(
                    partner_id=self.partner_id.id,
                    target_budget=self.target_budget,
                    client_notes=self.client_notes or '',
                    dietary_restrictions=dietary
                )
            except Exception as ollama_error:
                _logger.warning("Ollama failed, falling back to composition engine: %s", ollama_error)
                
                # Fallback to composition engine
                composition_engine = self.env['composition.engine']
                if composition_engine:
                    comp_result = composition_engine.generate_composition(
                        partner_id=self.partner_id.id,
                        target_budget=self.target_budget,
                        target_year=fields.Date.today().year,
                        dietary_restrictions=dietary,
                        notes_text=self.client_notes or ''
                    )
                    
                    if comp_result and comp_result.get('composition_id'):
                        composition = self.env['gift.composition'].browse(comp_result['composition_id'])
                        result = {
                            'success': True,
                            'composition_id': composition.id,
                            'products': composition.product_ids,
                            'total_cost': composition.actual_cost,
                            'confidence_score': 0.75,
                            'message': 'Generated using fallback engine'
                        }
                    else:
                        raise UserError("Both Ollama and fallback engine failed")
                else:
                    raise ollama_error

            if result.get('success'):
                # Update wizard with results
                update_vals = {
                    'state': 'done',
                    'result_message': result.get('message', 'Recommendations generated successfully'),
                    'composition_id': result.get('composition_id'),
                    'total_cost': result.get('total_cost', 0.0),
                    'confidence_score': result.get('confidence_score', 0.0)
                }
                
                # Add recommended products
                products = result.get('products', [])
                if products:
                    update_vals['recommended_products'] = [(6, 0, [p.id for p in products])]
                
                self.write(update_vals)

                # Return action to show results
                return {
                    'type': 'ir.actions.act_window',
                    'name': 'Recommendation Results',
                    'res_model': 'ollama.recommendation.wizard',
                    'res_id': self.id,
                    'view_mode': 'form',
                    'target': 'new',
                    'context': {'show_results': True},
                }

            else:
                # Handle failure
                error_msg = result.get('error', 'Unknown error occurred')
                self.write({
                    'state': 'error',
                    'error_message': error_msg,
                    'result_message': result.get('message', 'Recommendation generation failed')
                })
                raise UserError(f"Recommendation failed: {result.get('message', error_msg)}")

        except Exception as e:
            # Handle unexpected errors
            self.write({
                'state': 'error',
                'error_message': str(e)
            })
            _logger.exception("Recommendation wizard error: %s", e)
            raise

    def action_view_composition(self):
        """Open the generated composition"""
        self.ensure_one()
        if not self.composition_id:
            raise UserError("No composition generated yet.")
        
        return {
            'type': 'ir.actions.act_window',
            'name': f'Gift Composition for {self.partner_id.name or ""}',
            'res_model': 'gift.composition',
            'res_id': self.composition_id.id,
            'view_mode': 'form',
            'target': 'current',
        }

    def action_generate_another(self):
        """Create a new wizard with same parameters"""
        self.ensure_one()
        
        new_wizard = self.create({
            'currency_id': self.currency_id.id,
            'partner_id': self.partner_id.id,
            'target_budget': self.target_budget,
            'client_notes': self.client_notes,
            'dietary_restrictions': self.dietary_restrictions,
            'is_vegan': self.is_vegan,
            'is_halal': self.is_halal,
            'is_gluten_free': self.is_gluten_free,
            'is_non_alcoholic': self.is_non_alcoholic,
            'recommender_id': self.recommender_id.id,
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Generate Another Recommendation',
            'res_model': 'ollama.recommendation.wizard',
            'res_id': new_wizard.id,
            'view_mode': 'form',
            'target': 'new',
        }

    def action_test_recommender_connection(self):
        """Test the Ollama connection"""
        self.ensure_one()
        
        if not self.recommender_id:
            raise UserError("Please select a recommender first.")
        
        result = self.recommender_id.test_ollama_connection()
        
        title = 'Connection Test Successful' if result.get('success') else 'Connection Test Failed'
        level = 'success' if result.get('success') else 'danger'
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': title,
                'message': result.get('message', 'Test completed'),
                'type': level,
                'sticky': not result.get('success'),
            }
        }