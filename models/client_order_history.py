# -*- coding: utf-8 -*-
from odoo import models, fields, api, _
from collections import defaultdict
from datetime import datetime
import json

class ClientOrderHistory(models.Model):
    _name = 'client.order.history'
    _description = 'Client Gift Order History for Señor Bigott'
    _order = 'order_year desc'
    _rec_name = 'display_name'

    display_name = fields.Char('Display Name', compute='_compute_display_name', store=True)
    partner_id = fields.Many2one('res.partner', 'Client', required=True, ondelete='cascade')
    order_year = fields.Integer('Order Year', required=True)
    total_budget = fields.Float('Total Budget (€)', required=True)

    # Box type
    box_type = fields.Selection([
        ('experience', 'Experience-Based Box'),
        ('custom', 'Custom Product Box')
    ], string='Box Type', required=True)

    # Experience tracking
    experience_id = fields.Many2one('gift.experience', 'Experience Used', ondelete='set null')

    # Products in the box (unique templates)
    product_ids = fields.Many2many('product.template', string='Products in Box')

    # Category breakdown (JSON text: {"category": qty, ...})
    category_breakdown = fields.Text('Category Breakdown')

    # Totals
    total_products = fields.Integer('Total Products', compute='_compute_totals', store=True)

    # Dietary considerations
    dietary_restrictions = fields.Text('Dietary Restrictions Applied')

    # Performance tracking
    client_satisfaction = fields.Selection([
        ('1', '⭐'),
        ('2', '⭐⭐'),
        ('3', '⭐⭐⭐'),
        ('4', '⭐⭐⭐⭐'),
        ('5', '⭐⭐⭐⭐⭐')
    ], string='Client Satisfaction')

    # Notes
    notes = fields.Text('Order Notes')

    # Computed fields
    budget_per_product = fields.Float('Budget per Product', compute='_compute_budget_metrics', store=True)

    # ---------- Computes ----------
    @api.depends('partner_id', 'order_year', 'box_type')
    def _compute_display_name(self):
        for record in self:
            record.display_name = f"{record.partner_id.name} - {record.order_year} ({record.box_type})"

    @api.depends('category_breakdown')
    def _compute_totals(self):
        for rec in self:
            try:
                data = json.loads(rec.category_breakdown or '{}')
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}
            rec.total_products = sum(int(v or 0) for v in data.values())

    @api.depends('total_budget', 'total_products')
    def _compute_budget_metrics(self):
        for record in self:
            record.budget_per_product = record.total_budget / record.total_products if record.total_products else 0.0

    # ---------- JSON helpers ----------
    def get_category_structure(self):
        if not self.category_breakdown:
            return {}
        try:
            val = json.loads(self.category_breakdown)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}

    def set_category_structure(self, categories):
        try:
            self.category_breakdown = json.dumps(categories or {})
        except Exception:
            self.category_breakdown = '{}'

    # ---------- Analyze (used by engine) ----------
    @api.model
    def analyze_client_patterns(self, partner_id):
        histories = self.search([('partner_id', '=', partner_id)], order='order_year desc', limit=3)
        if not histories:
            return self._analyze_from_sales_fallback(partner_id)

        budgets = [h.total_budget for h in histories]
        avg_budget = sum(budgets) / len(budgets)
        avg_products = sum([h.total_products for h in histories]) / len(histories)
        used_experiences = [h.experience_id.id for h in histories if h.experience_id]
        latest_categories = histories[0].get_category_structure() if histories else {}

        if len(budgets) >= 2 and budgets[1]:
            recent_change = (budgets[0] - budgets[1]) / budgets[1] * 100
            budget_trend = 'increasing' if recent_change > 10 else ('decreasing' if recent_change < -10 else 'stable')
        else:
            budget_trend = 'unknown'

        satisfactions = [float(h.client_satisfaction) for h in histories if h.client_satisfaction]
        avg_satisfaction = sum(satisfactions) / len(satisfactions) if satisfactions else 0

        return {
            'has_history': True,
            'years_of_data': len(histories),
            'average_budget': avg_budget,
            'budget_trend': budget_trend,
            'recent_budgets': budgets,
            'average_products': avg_products,
            'average_satisfaction': avg_satisfaction,
            'used_experiences': used_experiences,
            'latest_category_structure': latest_categories,
            'box_type_preference': histories[0].box_type if histories else 'custom'
        }

    # ---------- Build history from sales ----------
    _EXPERIENCE_THRESHOLD = 0.7   # ≥70% of an order’s amount looks like an experience
    _LOOKBACK_YEARS = 3

    def _is_experience_line(self, line):
        pt = line.product_id.product_tmpl_id
        if hasattr(pt, 'is_experience_bundle') and pt.is_experience_bundle:
            return True, getattr(pt, 'experience_id', False)
        name = (pt.name or '').lower()
        if 'experience' in name or 'box' in name:
            return True, getattr(pt, 'experience_id', False)
        return False, False

    def _map_category(self, pt):
        return getattr(pt, 'lebiggot_category', False) or 'other'

    @api.model
    def rebuild_from_sales(self, date_from=False, date_to=False):
        Sale = self.env['sale.order']

        if not date_to:
            date_to = fields.Datetime.now()
        if not date_from:
            date_from = date_to.replace(year=date_to.year - self._LOOKBACK_YEARS)

        domain = [
            ('state', 'in', ['sale', 'done']),
            ('confirmation_date', '>=', date_from),
            ('confirmation_date', '<=', date_to),
        ]
        orders = Sale.search(domain, order='confirmation_date asc')

        buckets = defaultdict(lambda: {
            'amount': 0.0,
            'qty': 0,
            'categories': defaultdict(int),
            'tmpl_ids': set(),
            'exp_amount': 0.0,
            'experience_id': False,
            'box_type': 'custom',
        })

        for so in orders:
            year = (so.confirmation_date or so.date_order or fields.Datetime.now()).year
            key = (so.partner_id.id, year)

            # Use amount_total as-is (your model stores Float in € already)
            buckets[key]['amount'] += so.amount_total

            exp_amount = 0.0
            for line in so.order_line:
                qty = int(line.product_uom_qty or 0)
                if not qty:
                    continue
                pt = line.product_id.product_tmpl_id
                buckets[key]['qty'] += qty
                buckets[key]['tmpl_ids'].add(pt.id)
                buckets[key]['categories'][self._map_category(pt)] += qty

                is_exp, maybe_exp = self._is_experience_line(line)
                if is_exp:
                    exp_amount += line.price_total
                    if not buckets[key]['experience_id'] and maybe_exp:
                        buckets[key]['experience_id'] = maybe_exp.id

            if buckets[key]['amount'] and (exp_amount / max(buckets[key]['amount'], 1e-6)) >= self._EXPERIENCE_THRESHOLD:
                buckets[key]['box_type'] = 'experience'

        for (partner_id, year), data in buckets.items():
            vals = {
                'partner_id': partner_id,
                'order_year': year,
                'box_type': data['box_type'],
                'experience_id': data['experience_id'] or False,
                'total_budget': data['amount'],
                'category_breakdown': json.dumps(dict(data['categories'])),
                'product_ids': [(6, 0, list(data['tmpl_ids']))],
            }
            rec = self.search([('partner_id', '=', partner_id), ('order_year', '=', year)], limit=1)
            if rec:
                rec.write(vals)
            else:
                self.create(vals)
        return True

    @api.model
    def _analyze_from_sales_fallback(self, partner_id):
        Sale = self.env['sale.order']
        date_to = fields.Datetime.now()
        date_from = date_to.replace(year=date_to.year - self._LOOKBACK_YEARS)

        orders = Sale.search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done']),
            ('confirmation_date', '>=', date_from),
            ('confirmation_date', '<=', date_to),
        ], order='confirmation_date desc')

        if not orders:
            return {'has_history': False, 'message': 'No historical data available for this client'}

        yearly = defaultdict(lambda: {'amount': 0.0, 'qty': 0, 'categories': defaultdict(int), 'box_type': 'custom'})
        used_exps = set()

        for so in orders:
            y = (so.confirmation_date or so.date_order or fields.Datetime.now()).year
            yearly[y]['amount'] += so.amount_total

            exp_amount = 0.0
            for line in so.order_line:
                qty = int(line.product_uom_qty or 0)
                if not qty:
                    continue
                pt = line.product_id.product_tmpl_id
                yearly[y]['qty'] += qty
                yearly[y]['categories'][self._map_category(pt)] += qty

                is_exp, maybe_exp = self._is_experience_line(line)
                if is_exp:
                    exp_amount += line.price_total
                    if maybe_exp:
                        used_exps.add(maybe_exp.id)

            if yearly[y]['amount'] and (exp_amount / max(yearly[y]['amount'], 1e-6)) >= self._EXPERIENCE_THRESHOLD:
                yearly[y]['box_type'] = 'experience'

        years = sorted(yearly.keys(), reverse=True)[:3]
        budgets = [yearly[y]['amount'] for y in years]
        avg_budget = sum(budgets) / len(budgets)
        avg_products = sum(yearly[y]['qty'] for y in years) / len(years)
        latest_categories = dict(yearly[years[0]]['categories']) if years else {}

        if len(budgets) >= 2 and budgets[1]:
            change = (budgets[0] - budgets[1]) / budgets[1] * 100
            trend = 'increasing' if change > 10 else ('decreasing' if change < -10 else 'stable')
        else:
            trend = 'unknown'

        return {
            'has_history': True,
            'years_of_data': len(years),
            'average_budget': avg_budget,
            'budget_trend': trend,
            'recent_budgets': budgets,
            'average_products': avg_products,
            'average_satisfaction': 0,
            'used_experiences': list(used_exps),
            'latest_category_structure': latest_categories,
            'box_type_preference': yearly[years[0]]['box_type'] if years else 'custom',
        }

    def action_generate_next_composition(self):
        return {
            'type': 'ir.actions.act_window',
            'name': '🧠 Generate Next Year Composition',
            'res_model': 'composition.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': {
                'default_partner_id': self.partner_id.id,
                'default_target_year': datetime.now().year + 1,
                'default_target_budget': self.total_budget * 1.1 if self.total_budget else 0.0,
            }
        }
