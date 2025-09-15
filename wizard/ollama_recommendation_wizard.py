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
        required=True,
        default=lambda self: self._context.get('default_partner_id')
    )

    # --- Budget (Monetary, not Float) ---
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

    # ----------------- Defaults / Onchanges -----------------

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
            # Combine existing with new toggles, removing duplicates
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

    # ----------------- Internals -----------------

    def _resolve_partner(self):
        """Resolve partner from (1) form field, (2) read() cache, (3) context/active_id."""
        self.ensure_one()

        # 1) Trust the in-memory value first
        if self.partner_id and self.partner_id.exists():
            return self.partner_id

        # 2) read() may return int or (id, display_name)
        vals = self.read(['partner_id'])[0]
        raw = vals.get('partner_id')
        pid = None
        if isinstance(raw, (list, tuple)) and raw:
            pid = raw[0]
        elif isinstance(raw, int):
            pid = raw

        # 3) Context fallbacks
        ctx = dict(self._context or {})
        if not pid:
            pid = ctx.get('default_partner_id')
        if not pid and ctx.get('active_model') == 'res.partner':
            pid = ctx.get('active_id')

        if not pid:
            return self.env['res.partner']  # empty recordset

        partner = self.env['res.partner'].browse(pid)
        return partner if partner.exists() else self.env['res.partner']

    # ----------------- Main Actions -----------------

    def action_generate_recommendation(self):
        """Generate gift recommendations using Ollama"""
        self.ensure_one()

        # Robust partner resolve
        partner = self._resolve_partner()
        if not partner:
            raise UserError("Please select a client.")
        
        # Keep partner set for UI continuity
        if not self.partner_id:
            self.partner_id = partner.id

        if not self.recommender_id:
            raise UserError("No recommender available. Please configure an Ollama recommender first.")

        _logger.info("Wizard %s generating for partner %s budget=%s", 
                    self.id, partner.id, self.target_budget)

        try:
            # Set state to generating
            self.write({'state': 'generating'})
            
            # Parse dietary restrictions
            dietary = []
            if self.dietary_restrictions:
                dietary = [r.strip() for r in self.dietary_restrictions.split(',') if r.strip()]

            # Call the recommender
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=partner.id,
                target_budget=self.target_budget,
                client_notes=self.client_notes or '',
                dietary_restrictions=dietary
            )

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
        
        # Create new wizard with same values
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
        
        # Call test method on recommender
        result = self.recommender_id.test_ollama_connection()
        
        # Show notification
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

    def action_reset(self):
        """Reset the wizard to draft state"""
        self.ensure_one()
        self.write({
            'state': 'draft',
            'result_message': False,
            'error_message': False,
            'composition_id': False,
            'recommended_products': [(5, 0, 0)],  # Clear all products
            'total_cost': 0.0,
            'confidence_score': 0.0
        })
        
        return {
            'type': 'ir.actions.act_window',
            'res_model': 'ollama.recommendation.wizard',
            'res_id': self.id,
            'view_mode': 'form',
            'target': 'new',
        }