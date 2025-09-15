# -*- coding: utf-8 -*-
# wizard/ollama_recommendation_wizard.py

from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import logging
from datetime import datetime

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

    # --- Budget and Year ---
    target_budget = fields.Monetary(
        string="Target Budget",
        required=True,
        default=100.0,
        currency_field='currency_id'
    )
    
    target_year = fields.Integer(
        string="Target Year",
        default=lambda self: datetime.now().year,
        required=True
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
    
    # Add engine_type for compatibility with the view
    engine_type = fields.Selection([
        ('ollama', 'Ollama AI'),
        ('manual', 'Manual Selection')
    ], string="Engine Type", default='ollama')
    
    # Add use_ollama for compatibility with the view
    use_ollama = fields.Boolean(
        string="Use Ollama",
        default=True,
        compute='_compute_use_ollama'
    )
    
    # Client Intelligence field
    client_info = fields.Html(
        string="Client Information",
        compute='_compute_client_info'
    )

    # --- State & Results ---
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generating', 'Generating...'),
        ('done', 'Complete'),
        ('error', 'Error'),
    ], default='draft')

    result_message = fields.Html(readonly=True)
    composition_id = fields.Many2one('gift.composition', readonly=True)
    error_message = fields.Text(readonly=True)

    recommended_products = fields.Many2many('product.template', readonly=True)
    total_cost = fields.Monetary(readonly=True, currency_field='currency_id')
    confidence_score = fields.Float(readonly=True)

    # ----------------- Compute Methods -----------------
    
    @api.depends('engine_type')
    def _compute_use_ollama(self):
        for record in self:
            record.use_ollama = record.engine_type == 'ollama'
    
    @api.depends('partner_id')
    def _compute_client_info(self):
        for record in self:
            if record.partner_id:
                # Generate HTML summary of client information
                html_parts = []
                
                # Basic info
                html_parts.append(f"<p><strong>Client:</strong> {record.partner_id.name}</p>")
                
                if record.partner_id.email:
                    html_parts.append(f"<p><strong>Email:</strong> {record.partner_id.email}</p>")
                
                if record.partner_id.phone:
                    html_parts.append(f"<p><strong>Phone:</strong> {record.partner_id.phone}</p>")
                
                # Get order history - check if sale.order exists
                if 'sale.order' in self.env:
                    orders = self.env['sale.order'].search([
                        ('partner_id', '=', record.partner_id.id),
                        ('state', 'in', ['sale', 'done'])
                    ], limit=5, order='date_order desc')
                    
                    if orders:
                        html_parts.append("<p><strong>Recent Orders:</strong></p><ul>")
                        for order in orders:
                            date_str = order.date_order.strftime('%Y-%m-%d') if order.date_order else 'N/A'
                            html_parts.append(f"<li>{order.name} - {date_str} - {order.amount_total:.2f} {order.currency_id.symbol}</li>")
                        html_parts.append("</ul>")
                
                # Get previous gift compositions if they exist
                if 'gift.composition' in self.env:
                    compositions = self.env['gift.composition'].search([
                        ('partner_id', '=', record.partner_id.id)
                    ], limit=3, order='create_date desc')
                    
                    if compositions:
                        html_parts.append("<p><strong>Previous Gift Compositions:</strong></p><ul>")
                        for comp in compositions:
                            date_str = comp.create_date.strftime('%Y-%m-%d') if comp.create_date else 'N/A'
                            html_parts.append(f"<li>{date_str} - €{comp.target_budget:.2f} - {len(comp.product_ids)} products</li>")
                        html_parts.append("</ul>")
                
                # Combine all parts
                record.client_info = ''.join(html_parts) if html_parts else '<p>No client information available.</p>'
            else:
                record.client_info = '<p>Please select a client to view their information.</p>'

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
            # Add new toggles without duplicates
            all_restrictions = list(set(existing + toggles))
            self.dietary_restrictions = ', '.join(sorted(all_restrictions))

    @api.constrains('target_budget')
    def _check_target_budget(self):
        for rec in self:
            if rec.target_budget <= 0:
                raise ValidationError("Target budget must be greater than 0.")
            if rec.target_budget > 1_000_000:
                raise ValidationError("Target budget seems unusually high. Please confirm.")

    # ----------------- Actions -----------------

    def action_generate_recommendation(self):
        """Generate gift recommendation"""
        self.ensure_one()
        
        # Validate inputs
        if not self.partner_id:
            raise UserError("Please select a client.")
        
        if not self.recommender_id:
            raise UserError("No recommender available. Please configure an Ollama recommender first.")
        
        _logger.info(f"Wizard {self.id} generating for partner {self.partner_id.id} budget={self.target_budget}")
        
        try:
            # Update state
            self.write({'state': 'generating'})
            
            # Parse dietary restrictions
            dietary = []
            if self.dietary_restrictions:
                dietary = [r.strip() for r in self.dietary_restrictions.split(',') if r.strip()]
            
            # Call the recommender
            result = self.recommender_id.generate_gift_recommendations(
                partner_id=self.partner_id.id,
                target_budget=self.target_budget,
                client_notes=self.client_notes or '',
                dietary_restrictions=dietary
            )
            
            # Process result
            if result.get('success'):
                # Update wizard with results
                self.write({
                    'state': 'done',
                    'result_message': self._format_success_message(result),
                    'composition_id': result.get('composition_id'),
                    'recommended_products': [(6, 0, [p.id for p in result.get('products', [])])],
                    'total_cost': result.get('total_cost', 0.0),
                    'confidence_score': result.get('confidence_score', 0.0)
                })
                
                # Return the same wizard in done state
                return {
                    'type': 'ir.actions.act_window',
                    'name': 'Recommendation Results',
                    'res_model': 'ollama.recommendation.wizard',
                    'res_id': self.id,
                    'view_mode': 'form',
                    'target': 'new',
                    'context': dict(self._context, show_results=True),
                }
            else:
                # Handle failure
                error_msg = result.get('error', 'Unknown error occurred')
                self.write({
                    'state': 'error',
                    'error_message': error_msg,
                    'result_message': f"<div class='alert alert-danger'>{error_msg}</div>"
                })
                
                raise UserError(f"Recommendation failed: {error_msg}")
                
        except Exception as e:
            self.write({
                'state': 'error',
                'error_message': str(e)
            })
            _logger.exception(f"Recommendation wizard error: {e}")
            raise

    def _format_success_message(self, result):
        """Format a nice success message"""
        message_parts = [
            "<div class='alert alert-success'>",
            "<h4>✅ Recommendation Generated Successfully!</h4>",
            f"<p><strong>Products Selected:</strong> {result.get('product_count', len(result.get('products', [])))}</p>",
            f"<p><strong>Total Cost:</strong> €{result.get('total_cost', 0):.2f}</p>",
            f"<p><strong>Target Budget:</strong> €{self.target_budget:.2f}</p>"
        ]
        
        if result.get('confidence_score'):
            message_parts.append(f"<p><strong>Confidence Score:</strong> {result.get('confidence_score', 0) * 100:.0f}%</p>")
        
        if result.get('message'):
            message_parts.append(f"<p>{result.get('message')}</p>")
        
        message_parts.append("</div>")
        return '\n'.join(message_parts)

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
            'target': 'current',
        }

    def action_generate_another(self):
        """Generate another recommendation with same settings"""
        self.ensure_one()
        
        # Create new wizard with same settings
        new_wizard = self.create({
            'currency_id': self.currency_id.id,
            'partner_id': self.partner_id.id,
            'target_budget': self.target_budget,
            'target_year': self.target_year,
            'client_notes': self.client_notes,
            'dietary_restrictions': self.dietary_restrictions,
            'is_vegan': self.is_vegan,
            'is_halal': self.is_halal,
            'is_gluten_free': self.is_gluten_free,
            'is_non_alcoholic': self.is_non_alcoholic,
            'recommender_id': self.recommender_id.id,
            'engine_type': self.engine_type,
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': 'Generate Another Recommendation',
            'res_model': 'ollama.recommendation.wizard',
            'res_id': new_wizard.id,
            'view_mode': 'form',
            'target': 'new',
        }

    def action_test_connection(self):
        """Test Ollama connection"""
        self.ensure_one()
        
        if not self.recommender_id:
            raise UserError("Please select a recommender first.")
        
        # Test the connection
        result = self.recommender_id.test_ollama_connection()
        
        # Prepare notification
        title = 'Connection Test Successful' if result.get('success') else 'Connection Test Failed'
        message = result.get('message', 'Connection test completed')
        notification_type = 'success' if result.get('success') else 'danger'
        
        # Return notification
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': title,
                'message': message,
                'type': notification_type,
                'sticky': not result.get('success'),  # Keep error messages visible
                'next': {'type': 'ir.actions.act_window_close'} if result.get('success') else None
            }
        }
    
    # Backward compatibility method names (in case they're referenced elsewhere)
    def action_test_recommender_connection(self):
        """Alias for action_test_connection for backward compatibility"""
        return self.action_test_connection()