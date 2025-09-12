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

    recommended_products = fields.Many2many('product.template', readonly=True)
    total_cost = fields.Monetary(readonly=True, currency_field='currency_id')
    confidence_score = fields.Float(readonly=True)

    # ----------------- Defaults / Onchanges -----------------

    @api.model
    def _default_recommender(self):
        rec = self.env['ollama.gift.recommender'].search([('active', '=', True)], limit=1)
        if not rec:
            rec = self.env['ollama.gift.recommender'].create({'name': 'Default Ollama Recommender'})
        return rec

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

    @api.constrains('target_budget')
    def _check_target_budget(self):
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

    # ----------------- Actions -----------------

    def action_generate_recommendation(self):
        """Simplified test method"""
        self.ensure_one()
        
        # Just test if we can access the data
        if not self.partner_id:
            # Instead of raising an error, return a notification
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Debug Info',
                    'message': f'Partner ID is empty. Wizard ID: {self.id}',
                    'type': 'warning',
                    'sticky': True
                }
            }
        
        # If we get here, the partner was selected successfully
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Success!',
                'message': f'Client: {self.partner_id.name}, Budget: €{self.target_budget}',
                'type': 'success',
                'sticky': True
            }
        }

    def action_view_composition(self):
        self.ensure_one()
        if not self.composition_id:
            raise UserError("No composition generated yet.")
        return {
            'type': 'ir.actions.act_window',
            'name': 'Gift Composition for %s' % (self.partner_id.name or ''),
            'res_model': 'gift.composition',
            'res_id': self.composition_id.id,
            'view_mode': 'form',
            'target': 'current',
        }

    def action_generate_another(self):
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
                'message': result.get('message'),
                'type': level,
                'sticky': not result.get('success'),
            }
        }
