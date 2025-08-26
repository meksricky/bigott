from odoo import models, fields, api
from datetime import datetime, timedelta

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
    
    # Products in the box
    product_ids = fields.Many2many('product.template', string='Products in Box')
    total_products = fields.Integer('Total Products', compute='_compute_totals', store=True)
    
    # Category analysis - Text field for Odoo 14 (store JSON string)
    category_breakdown = fields.Text('Category Breakdown')
    
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
    
    @api.depends('partner_id', 'order_year', 'box_type')
    def _compute_display_name(self):
        for record in self:
            record.display_name = f"{record.partner_id.name} - {record.order_year} ({record.box_type})"
    
    @api.depends('product_ids')
    def _compute_totals(self):
        for record in self:
            record.total_products = len(record.product_ids)
    
    @api.depends('total_budget', 'total_products')
    def _compute_budget_metrics(self):
        for record in self:
            record.budget_per_product = record.total_budget / record.total_products if record.total_products else 0
    
    def get_category_structure(self):
        """Return category breakdown as dictionary"""
        import json
        if not self.category_breakdown:
            return {}
        try:
            return json.loads(self.category_breakdown)
        except Exception:
            return {}
    
    def set_category_structure(self, categories):
        """Set category structure from dictionary"""
        import json
        try:
            self.category_breakdown = json.dumps(categories or {})
        except Exception:
            self.category_breakdown = '{}'
    
    @api.model
    def analyze_client_patterns(self, partner_id):
        """Analyze client's historical patterns"""
        
        histories = self.search([
            ('partner_id', '=', partner_id)
        ], order='order_year desc', limit=3)
        
        if not histories:
            return {
                'has_history': False,
                'message': 'No historical data available for this client'
            }
        
        # Calculate patterns
        budgets = [h.total_budget for h in histories]
        avg_budget = sum(budgets) / len(budgets)
        avg_products = sum([h.total_products for h in histories]) / len(histories)
        
        # Experience usage
        used_experiences = [h.experience_id.id for h in histories if h.experience_id]
        
        # Most recent category structure
        latest_categories = histories[0].get_category_structure() if histories else {}
        
        # Budget trend analysis
        if len(budgets) >= 2:
            recent_change = (budgets[0] - budgets[1]) / budgets[1] * 100
            if recent_change > 10:
                budget_trend = 'increasing'
            elif recent_change < -10:
                budget_trend = 'decreasing'
            else:
                budget_trend = 'stable'
        else:
            budget_trend = 'unknown'
        
        # Satisfaction analysis
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
    
    def action_generate_next_composition(self):
        """Generate composition for next year"""
        return {
            'type': 'ir.actions.act_window',
            'name': '🧠 Generate Next Year Composition',
            'res_model': 'composition.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': {
                'default_partner_id': self.partner_id.id,
                'default_target_year': datetime.now().year + 1,
                'default_target_budget': self.total_budget * 1.1,  # Suggest 10% increase
            }
        }

    _EXPERIENCE_THRESHOLD = 0.7   # ≥70% of an order’s amount from “experience” lines -> treat as experience
    _LOOKBACK_YEARS = 3

    def _is_experience_line(self, line):
        """Use your real flag if you have one on product.template (e.g., is_experience_bundle)."""
        pt = line.product_id.product_tmpl_id
        if hasattr(pt, 'is_experience_bundle') and pt.is_experience_bundle:
            return True, getattr(pt, 'experience_id', False)
        name = (pt.name or '').lower()
        if 'experience' in name or 'box' in name:
            return True, getattr(pt, 'experience_id', False)
        return False, False

    def _map_category(self, pt):
        """Map a product to your internal category key used in category_breakdown JSON."""
        return getattr(pt, 'lebiggot_category', False) or 'other'

    @api.model
    def rebuild_from_sales(self, date_from=False, date_to=False):
        """Aggregate sale orders into client.order.history per partner & year."""
        Sale = self.env['sale.order']
        company = self.env.company
        company_currency = company.currency_id

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

        # (partner_id, year) -> bucket
        buckets = defaultdict(lambda: {
            'amount_company': 0.0,
            'qty_total': 0,
            'categories': defaultdict(int),
            'product_tmpl_ids': set(),
            'experience_amount_company': 0.0,
            'experience_id': False,
            'box_type': 'custom',
        })

        for so in orders:
            year = (so.confirmation_date or so.date_order or fields.Datetime.now()).year
            key = (so.partner_id.id, year)

            # convert order total to company currency
            order_total_company = so.currency_id._convert(
                so.amount_total, company_currency, so.company_id,
                so.date_order or so.confirmation_date or fields.Datetime.now()
            )
            buckets[key]['amount_company'] += order_total_company

            exp_amount_company = 0.0
            for line in so.order_line:
                qty = int(line.product_uom_qty or 0)
                if not qty:
                    continue

                pt = line.product_id.product_tmpl_id
                buckets[key]['qty_total'] += qty
                buckets[key]['product_tmpl_ids'].add(pt.id)

                # category counts
                cat = self._map_category(pt)
                buckets[key]['categories'][cat] += qty

                # convert line total to company currency for experience share
                line_total_company = so.currency_id._convert(
                    line.price_total, company_currency, so.company_id,
                    so.date_order or so.confirmation_date or fields.Datetime.now()
                )
                is_exp, maybe_exp = self._is_experience_line(line)
                if is_exp:
                    exp_amount_company += line_total_company
                    if not buckets[key]['experience_id'] and maybe_exp:
                        buckets[key]['experience_id'] = maybe_exp.id

            # decide box_type for THIS order; apply to the yearly bucket if majority exp
            if order_total_company and (exp_amount_company / max(order_total_company, 1e-6)) >= self._EXPERIENCE_THRESHOLD:
                buckets[key]['box_type'] = 'experience'

        # upsert per (partner, year)
        for (partner_id, year), data in buckets.items():
            vals = {
                'partner_id': partner_id,
                'order_year': year,
                'box_type': data['box_type'],
                'experience_id': data['experience_id'] or False,
                'total_budget': data['amount_company'],
                # total_products is computed from category_breakdown now
                'category_breakdown': json.dumps(data['categories']),
                # Optional: keep unique products
                'product_ids': [(6, 0, list(data['product_tmpl_ids']))],
                # Optional: fill notes/dietary_restrictions from partner or tags here if you track them
            }
            rec = self.search([('partner_id', '=', partner_id), ('order_year', '=', year)], limit=1)
            if rec:
                rec.write(vals)
            else:
                self.create(vals)

        return True

    # Optional: if no history exists yet, analyze directly from sales on-the-fly
    @api.model
    def _analyze_from_sales_fallback(self, partner_id):
        Sale = self.env['sale.order']
        company_currency = self.env.company.currency_id
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

        # group by year
        yearly = defaultdict(lambda: {'amount': 0.0, 'qty': 0, 'categories': defaultdict(int), 'box_type': 'custom'})
        recent_year = None
        recent_categories = {}
        used_exps = set()

        for so in orders:
            y = (so.confirmation_date or so.date_order or fields.Datetime.now()).year
            if recent_year is None:
                recent_year = y
            amount_company = so.currency_id._convert(so.amount_total, company_currency, so.company_id,
                                                     so.date_order or so.confirmation_date or fields.Datetime.now())
            yearly[y]['amount'] += amount_company

            exp_amount_company = 0.0
            for line in so.order_line:
                qty = int(line.product_uom_qty or 0)
                if not qty:
                    continue
                pt = line.product_id.product_tmpl_id
                yearly[y]['qty'] += qty
                yearly[y]['categories'][self._map_category(pt)] += qty

                line_total_company = so.currency_id._convert(line.price_total, company_currency, so.company_id,
                                                             so.date_order or so.confirmation_date or fields.Datetime.now())
                is_exp, maybe_exp = self._is_experience_line(line)
                if is_exp:
                    exp_amount_company += line_total_company
                    if maybe_exp:
                        used_exps.add(maybe_exp.id)

            if yearly[y]['amount'] and (exp_amount_company / max(yearly[y]['amount'], 1e-6)) >= self._EXPERIENCE_THRESHOLD:
                yearly[y]['box_type'] = 'experience'

        # build metrics
        years = sorted(yearly.keys(), reverse=True)[:3]
        budgets = [yearly[y]['amount'] for y in years]
        avg_budget = sum(budgets) / len(budgets)
        avg_products = sum(yearly[y]['qty'] for y in years) / len(years)
        latest_categories = yearly[years[0]]['categories'] if years else {}

        if len(budgets) >= 2 and budgets[1] != 0:
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
            'average_satisfaction': 0,    # unknown from sales
            'used_experiences': list(used_exps),
            'latest_category_structure': latest_categories,
            'box_type_preference': yearly[years[0]]['box_type'] if years else 'custom',
        }

    # PATCH your analyze method to use the fallback if empty
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

        if len(budgets) >= 2 and budgets[1] != 0:
            recent_change = (budgets[0] - budgets[1]) / budgets[1] * 100
            if recent_change > 10:
                budget_trend = 'increasing'
            elif recent_change < -10:
                budget_trend = 'decreasing'
            else:
                budget_trend = 'stable'
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
            'box_type_preference': histories[0].box_type if histories else 'custom',
        }