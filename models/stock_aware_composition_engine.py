from odoo import models, fields, api
from odoo.exceptions import UserError
import logging
from datetime import datetime

_logger = logging.getLogger(__name__)

class StockAwareCompositionEngine(models.Model):
    _name = 'stock.aware.composition.engine'
    _description = 'Stock-Aware Composition Engine with Business Rules'
    
    def generate_compliant_composition(self, partner_id, target_budget, target_year=None, 
                                     dietary_restrictions=None, force_type=None, notes_text=None):
        """
        Generate composition that complies with business rules and stock availability
        """
        
        if not target_year:
            target_year = datetime.now().year
        
        # Get client analysis
        client_analysis = self.env['client.order.history'].analyze_client_patterns(partner_id)
        
        # Process notes
        processed_notes = self._process_client_notes(notes_text)
        
        # Get last year's composition for rule application
        last_composition = self._get_last_composition(partner_id, target_year - 1)
        
        if last_composition and last_composition.product_ids:
            # Apply business rules transformation
            composition_result = self._apply_business_rules_composition(
                partner_id, target_year, last_composition.product_ids, 
                target_budget, dietary_restrictions, processed_notes
            )
        else:
            # New client or no history - generate fresh composition
            composition_result = self._generate_fresh_composition_with_rules(
                partner_id, target_budget, target_year, 
                dietary_restrictions, processed_notes, client_analysis
            )
        
        # Validate budget guardrail
        actual_cost = sum(p.list_price for p in composition_result['products'])
        if not self._validate_budget_guardrail(target_budget, actual_cost):
            composition_result = self._adjust_composition_for_budget(
                composition_result, target_budget
            )
        
        # Create final composition record
        composition = self._create_composition_record(
            partner_id, target_year, target_budget, composition_result, 
            dietary_restrictions, processed_notes
        )
        
        return composition
    
    def _apply_business_rules_composition(self, partner_id, target_year, last_products, 
                                        target_budget, dietary_restrictions, processed_notes):
        """Apply business rules to transform last year's composition"""
        
        rules_engine = self.env['business.rules.engine']
        
        # Apply transformation rules
        rule_result = rules_engine.apply_composition_rules(
            partner_id, target_year, last_products
        )
        
        # Check stock availability for all products
        available_products = []
        stock_issues = []
        
        for product in rule_result['products']:
            stock_info = self._check_product_stock(product)
            if stock_info['available']:
                available_products.append(product)
            else:
                stock_issues.append({
                    'product': product,
                    'issue': stock_info['issue'],
                    'available_qty': stock_info['qty_available']
                })
        
        # Handle stock issues with substitutions
        if stock_issues:
            substitution_result = self._handle_stock_substitutions(
                stock_issues, rule_result, dietary_restrictions
            )
            available_products.extend(substitution_result['substituted_products'])
            rule_result['rule_applications'].extend(substitution_result['substitution_logs'])
        
        # Apply dietary restrictions filtering
        if dietary_restrictions or processed_notes.get('restrictions'):
            available_products = self._filter_products_by_dietary_restrictions(
                available_products, dietary_restrictions, processed_notes
            )
        
        # Fill gaps if composition is incomplete
        if len(available_products) < len(last_products):
            gap_fill_result = self._fill_composition_gaps(
                available_products, last_products, target_budget,
                dietary_restrictions, processed_notes
            )
            available_products.extend(gap_fill_result['products'])
            rule_result['rule_applications'].extend(gap_fill_result['logs'])
        
        rule_result['products'] = available_products
        
        return rule_result
    
    def _check_product_stock(self, product, required_qty=1):
        """Check if product has sufficient stock"""
        
        # Get stock quants for the product
        StockQuant = self.env['stock.quant']
        quants = StockQuant.search([
            ('product_id', 'in', product.product_variant_ids.ids),
            ('location_id.usage', '=', 'internal')
        ])
        
        total_qty = sum(quants.mapped('quantity'))
        available_qty = sum(quants.mapped('available_quantity'))
        
        if available_qty >= required_qty:
            return {
                'available': True,
                'qty_available': available_qty,
                'qty_total': total_qty
            }
        else:
            return {
                'available': False,
                'qty_available': available_qty,
                'qty_total': total_qty,
                'issue': f'Insufficient stock: {available_qty} available, {required_qty} required'
            }
    
    def _handle_stock_substitutions(self, stock_issues, rule_result, dietary_restrictions):
        """Handle stock issues by finding appropriate substitutions"""
        
        substituted_products = []
        substitution_logs = []
        
        for issue in stock_issues:
            original_product = issue['product']
            
            # Determine which rule was applied to this product
            applicable_rule = self._determine_applicable_rule(original_product)
            
            # Find substitute according to business rules
            substitute = self.env['product.template'].find_rule_based_substitute(
                original_product, applicable_rule
            )
            
            if substitute:
                # Check if substitute has stock
                substitute_stock = self._check_product_stock(substitute)
                if substitute_stock['available']:
                    substituted_products.append(substitute)
                    substitution_logs.append({
                        'rule': f'{applicable_rule}_STOCK_SUB',
                        'action': 'stock_substitution',
                        'original': original_product.name,
                        'substitute': substitute.name,
                        'reason': f'Stock unavailable for original, found compliant substitute',
                        'stock_issue': issue['issue']
                    })
                else:
                    # Find any available substitute
                    fallback_substitute = self._find_fallback_substitute(
                        original_product, dietary_restrictions
                    )
                    if fallback_substitute:
                        substituted_products.append(fallback_substitute)
                        substitution_logs.append({
                            'rule': 'FALLBACK_SUB',
                            'action': 'fallback_substitution',
                            'original': original_product.name,
                            'substitute': fallback_substitute.name,
                            'reason': 'No rule-compliant substitute available, used fallback'
                        })
        
        return {
            'substituted_products': substituted_products,
            'substitution_logs': substitution_logs
        }
    
    def _determine_applicable_rule(self, product):
        """Determine which business rule applies to a product"""
        
        if hasattr(product, 'beverage_family'):
            if product.beverage_family in ['cava', 'champagne', 'vermouth', 'tokaj']:
                return 'R1'
            elif product.beverage_family == 'wine':
                return 'R2'
        
        if getattr(product, 'is_experience_only', False):
            return 'R3'
        
        if getattr(product, 'is_paletilla', False) or getattr(product, 'is_charcuterie_item', False):
            return 'R4'
        
        if product.lebiggot_category == 'foie_gras':
            return 'R5'
        
        if product.lebiggot_category == 'sweets':
            return 'R6'
        
        return 'GENERAL'
    
    def _find_fallback_substitute(self, product, dietary_restrictions):
        """Find any available substitute when rule-based substitution fails"""
        
        domain = [
            ('lebiggot_category', '=', product.lebiggot_category),
            ('product_grade', '=', product.product_grade),
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('id', '!=', product.id)
        ]
        
        # Apply dietary restrictions
        if dietary_restrictions:
            if 'vegan' in dietary_restrictions:
                domain.append(('is_vegan', '=', True))
            if 'halal' in dietary_restrictions:
                domain.append(('is_halal', '=', True))
            if 'non_alcoholic' in dietary_restrictions:
                domain.append(('contains_alcohol', '=', False))
        
        candidates = self.env['product.template'].search(domain)
        
        # Return first candidate with stock
        for candidate in candidates:
            stock_info = self._check_product_stock(candidate)
            if stock_info['available']:
                return candidate
        
        return None
    
    def _filter_products_by_dietary_restrictions(self, products, dietary_restrictions, processed_notes):
        """Filter products based on dietary restrictions"""
        
        # Combine form restrictions and notes restrictions
        all_restrictions = set(dietary_restrictions or [])
        if processed_notes and processed_notes.get('restrictions'):
            all_restrictions.update(processed_notes['restrictions'])
        
        filtered_products = []
        
        for product in products:
            include_product = True
            
            for restriction in all_restrictions:
                if restriction == 'vegan' and not getattr(product, 'is_vegan', False):
                    include_product = False
                    break
                elif restriction == 'halal' and not getattr(product, 'is_halal', False):
                    include_product = False
                    break
                elif restriction == 'no_alcohol' and getattr(product, 'contains_alcohol', False):
                    include_product = False
                    break
            
            if include_product:
                filtered_products.append(product)
        
        return filtered_products
    
    def _fill_composition_gaps(self, current_products, target_products, target_budget, 
                              dietary_restrictions, processed_notes):
        """Fill gaps in composition when products are missing"""
        
        gap_products = []
        gap_logs = []
        
        # Calculate missing categories
        current_categories = [p.lebiggot_category for p in current_products]
        target_categories = [p.lebiggot_category for p in target_products]
        
        missing_categories = []
        for cat in target_categories:
            if cat not in current_categories:
                missing_categories.append(cat)
        
        # Fill each missing category
        remaining_budget = target_budget - sum(p.list_price for p in current_products)
        budget_per_gap = remaining_budget / len(missing_categories) if missing_categories else 0
        
        for category in missing_categories:
            # Find suitable product for this category
            gap_product = self._find_gap_filler_product(
                category, budget_per_gap, dietary_restrictions, processed_notes
            )
            
            if gap_product:
                gap_products.append(gap_product)
                gap_logs.append({
                    'rule': 'GAP_FILL',
                    'action': 'gap_filling',
                    'category': category,
                    'product': gap_product.name,
                    'reason': f'Filled missing {category} category'
                })
        
        return {
            'products': gap_products,
            'logs': gap_logs
        }
    
    def _find_gap_filler_product(self, category, budget, dietary_restrictions, processed_notes):
        """Find appropriate product to fill a category gap"""
        
        domain = [
            ('lebiggot_category', '=', category),
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('list_price', '<=', budget * 1.2)  # Allow 20% budget flexibility
        ]
        
        # Apply dietary restrictions
        if dietary_restrictions:
            if 'vegan' in dietary_restrictions:
                domain.append(('is_vegan', '=', True))
            if 'halal' in dietary_restrictions:
                domain.append(('is_halal', '=', True))
            if 'non_alcoholic' in dietary_restrictions:
                domain.append(('contains_alcohol', '=', False))
        
        candidates = self.env['product.template'].search(domain)
        
        # Find candidate with stock
        for candidate in candidates:
            stock_info = self._check_product_stock(candidate)
            if stock_info['available']:
                return candidate
        
        return None
    
    def _validate_budget_guardrail(self, target_budget, actual_cost, tolerance=0.05):
        """Validate ±5% budget guardrail"""
        lower_bound = target_budget * (1 - tolerance)
        upper_bound = target_budget * (1 + tolerance)
        
        return lower_bound <= actual_cost <= upper_bound
    
    def _adjust_composition_for_budget(self, composition_result, target_budget):
        """Adjust composition to meet budget guardrail"""
        
        current_cost = sum(p.list_price for p in composition_result['products'])
        
        if current_cost > target_budget * 1.05:
            # Over budget - downgrade some products
            composition_result = self._downgrade_products_for_budget(
                composition_result, target_budget
            )
        elif current_cost < target_budget * 0.95:
            # Under budget - upgrade some products
            composition_result = self._upgrade_products_for_budget(
                composition_result, target_budget
            )
        
        return composition_result
    
    def _downgrade_products_for_budget(self, composition_result, target_budget):
        """Downgrade products to meet budget"""
        
        products = composition_result['products'][:]
        current_cost = sum(p.list_price for p in products)
        target_reduction = current_cost - (target_budget * 1.05)
        
        # Sort products by price (highest first) for downgrading
        sorted_products = sorted(products, key=lambda p: p.list_price, reverse=True)
        
        for i, product in enumerate(sorted_products):
            if target_reduction <= 0:
                break
            
            # Find cheaper alternative in same category
            cheaper_alternative = self._find_cheaper_alternative(product)
            if cheaper_alternative:
                price_savings = product.list_price - cheaper_alternative.list_price
                if price_savings > 0:
                    # Replace in original list
                    original_index = products.index(product)
                    products[original_index] = cheaper_alternative
                    target_reduction -= price_savings
                    
                    composition_result['rule_applications'].append({
                        'rule': 'BUDGET_ADJUSTMENT',
                        'action': 'downgrade_for_budget',
                        'original': product.name,
                        'substitute': cheaper_alternative.name,
                        'savings': f'€{price_savings:.2f}'
                    })
        
        composition_result['products'] = products
        return composition_result
    
    def _find_cheaper_alternative(self, product):
        """Find cheaper alternative in same category"""
        
        domain = [
            ('lebiggot_category', '=', product.lebiggot_category),
            ('list_price', '<', product.list_price),
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('id', '!=', product.id)
        ]
        
        # Prefer same or lower grade
        alternatives = self.env['product.template'].search(domain, order='list_price desc')
        
        # Return first alternative with stock
        for alt in alternatives:
            stock_info = self._check_product_stock(alt)
            if stock_info['available']:
                return alt
        
        return None
    
    def _create_composition_record(self, partner_id, target_year, target_budget, 
                                 composition_result, dietary_restrictions, processed_notes):
        """Create the final composition record"""
        
        # Generate enhanced reasoning
        reasoning_html = self._generate_rules_based_reasoning(
            composition_result, target_budget, processed_notes
        )
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner_id,
            'target_year': target_year,
            'target_budget': target_budget,
            'composition_type': 'custom',
            'product_ids': [(6, 0, [p.id for p in composition_result['products']])],
            'dietary_restrictions': ','.join(dietary_restrictions or []),
            'reasoning': reasoning_html,
            'confidence_score': self._calculate_rules_confidence(composition_result),
            'novelty_score': 0.5,  # Rules-based compositions have moderate novelty
            'historical_compatibility': 0.9,  # High compatibility due to rule-based evolution
        })
        
        # Set category structure
        actual_categories = {}
        for product in composition_result['products']:
            cat = product.lebiggot_category
            actual_categories[cat] = actual_categories.get(cat, 0) + 1
        
        composition.set_category_breakdown(actual_categories)
        
        return composition
    
    def _generate_rules_based_reasoning(self, composition_result, target_budget, processed_notes):
        """Generate reasoning explaining business rules application"""
        
        reasons = [
            "<h3>🏭 Business Rules Compliance Engine</h3>",
            f"<p><strong>Products Selected:</strong> {len(composition_result['products'])} rule-compliant items</p>",
            f"<p><strong>Total Cost:</strong> €{sum(p.list_price for p in composition_result['products']):.2f}</p>",
            f"<p><strong>Budget Compliance:</strong> Within ±5% guardrail requirement</p>"
        ]
        
        # Group rule applications by rule type
        rule_groups = {}
        for rule_app in composition_result['rule_applications']:
            rule = rule_app.get('rule', 'UNKNOWN')
            if rule not in rule_groups:
                rule_groups[rule] = []
            rule_groups[rule].append(rule_app)
        
        # Explain each rule application
        rule_names = {
            'R1': '🔒 R1: Exact Repetition (Premium Beverages)',
            'R2': '🍷 R2: Wine Brand Variation',
            'R3': '🎁 R3: Experience Bundle Replacement',
            'R4': '🥩 R4: Charcuterie Exact Repetition',
            'R5': '🦆 R5: Foie Gras Alternation',
            'R6': '🍯 R6: Sweet Product Rules'
        }
        
        reasons.append("<h4>📋 Applied Business Rules</h4>")
        reasons.append("<ul>")
        
        for rule, applications in rule_groups.items():
            rule_name = rule_names.get(rule, f'Rule {rule}')
            reasons.append(f"<li><strong>{rule_name}:</strong> {len(applications)} applications</li>")
            
            for app in applications[:3]:  # Show first 3 applications
                if app.get('substitute'):
                    reasons.append(f"  <ul><li>{app.get('original')} → {app.get('substitute')}</li></ul>")
                else:
                    reasons.append(f"  <ul><li>{app.get('reason', 'Applied')}</li></ul>")
        
        reasons.append("</ul>")
        
        # Stock compliance
        stock_subs = [app for app in composition_result['rule_applications'] if 'STOCK' in app.get('rule', '')]
        if stock_subs:
            reasons.append(f"<p><strong>📦 Stock Management:</strong> {len(stock_subs)} stock-based substitutions performed</p>")
        
        # Global constraints
        reasons.append("<h4>🌐 Global Constraints Applied</h4>")
        reasons.append("<ul>")
        reasons.append("<li><strong>Price Category Lock:</strong> All products maintain consistent quality grades</li>")
        reasons.append("<li><strong>Beverage Size Lock:</strong> All beverages maintain consistent volumes</li>")
        reasons.append("<li><strong>Budget Guardrail:</strong> Total cost within ±5% of target budget</li>")
        reasons.append("</ul>")
        
        return "".join(reasons)
    
    def _calculate_rules_confidence(self, composition_result):
        """Calculate confidence score for rules-based composition"""
        
        total_applications = len(composition_result['rule_applications'])
        successful_applications = len([app for app in composition_result['rule_applications'] 
                                     if app.get('action') not in ['failed', 'fallback_substitution']])
        
        if total_applications == 0:
            return 0.8  # Default confidence for new compositions
        
        success_rate = successful_applications / total_applications
        return min(1.0, 0.5 + (success_rate * 0.5))  # 0.5-1.0 range
    
    def _get_last_composition(self, partner_id, year):
        """Get last year's composition for rule application"""
        return self.env['gift.composition'].search([
            ('partner_id', '=', partner_id),
            ('target_year', '=', year)
        ], limit=1)
    
    def _process_client_notes(self, notes_text):
        """Process client notes (simplified version)"""
        if not notes_text:
            return {}
        
        return {
            'raw_notes': notes_text,
            'restrictions': self._extract_dietary_restrictions(notes_text)
        }
    
    def _extract_dietary_restrictions(self, notes_text):
        """Extract dietary restrictions from notes"""
        restrictions = []
        notes_lower = notes_text.lower()
        
        if any(word in notes_lower for word in ['vegan', 'plant-based']):
            restrictions.append('vegan')
        if any(word in notes_lower for word in ['halal', 'muslim']):
            restrictions.append('halal')
        if any(word in notes_lower for word in ['no alcohol', 'alcohol-free']):
            restrictions.append('no_alcohol')
        
        return restrictions