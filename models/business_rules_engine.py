from odoo import models, fields, api
from odoo.exceptions import UserError
import logging
from collections import defaultdict

_logger = logging.getLogger(__name__)

class BusinessRulesEngine(models.Model):
    _name = 'business.rules.engine'
    _description = 'Le Biggot Business Rules Engine for Gift Compositions'
    
    def apply_composition_rules(self, partner_id, target_year, last_composition_products=None):
        """
        Apply business rules R1-R6 for gift composition
        Returns: dict with categorized product selections and rule applications
        """
        
        if not last_composition_products:
            # New client or no history - generate fresh composition
            return self._generate_fresh_composition(partner_id, target_year)
        
        # Existing client - apply transformation rules
        return self._apply_transformation_rules(partner_id, target_year, last_composition_products)
    
    def _apply_transformation_rules(self, partner_id, target_year, last_products):
        """Apply R1-R6 transformation rules to last year's composition"""
        
        result = {
            'products': [],
            'rule_applications': [],
            'substitutions': [],
            'locked_attributes': {
                'price_categories': set(),
                'beverage_sizes': set()
            }
        }
        
        # Categorize last year's products
        categorized_products = self._categorize_products(last_products)
        
        # Apply rules by category
        for category, products in categorized_products.items():
            if category in ['cava', 'champagne', 'vermouth', 'tokaj']:
                # R1: Repeat exactly
                new_products, rule_log = self._apply_rule_r1(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
                
            elif category == 'wine':
                # R2: Same color, different brand, same size & grade
                new_products, rule_log = self._apply_rule_r2(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
                
            elif category == 'experience':
                # R3: Replace with new same-size bundles from pool
                new_products, rule_log = self._apply_rule_r3(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
                
            elif category in ['paletilla', 'charcuterie']:
                # R4: Repeat exactly
                new_products, rule_log = self._apply_rule_r4(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
                
            elif category == 'foie_gras':
                # R5: Alternate Duck ↔ Goose
                new_products, rule_log = self._apply_rule_r5(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
                
            elif category == 'sweets':
                # R6: Lingote exact, Turrón keeps subtype & grade
                new_products, rule_log = self._apply_rule_r6(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
        
        # Apply global constraints
        result = self._apply_global_constraints(result)
        
        return result
    
    def _categorize_products(self, products):
        """Categorize products for rule application"""
        categories = defaultdict(list)
        
        for product in products:
            # Determine category based on product attributes
            if hasattr(product, 'beverage_family'):
                if product.beverage_family in ['cava', 'champagne', 'vermouth', 'tokaj']:
                    categories[product.beverage_family].append(product)
                elif product.beverage_family == 'wine':
                    categories['wine'].append(product)
            elif hasattr(product, 'is_experience_only') and product.is_experience_only:
                categories['experience'].append(product)
            elif hasattr(product, 'is_paletilla') and product.is_paletilla:
                categories['paletilla'].append(product)
            elif hasattr(product, 'is_charcuterie_item') and product.is_charcuterie_item:
                categories['charcuterie'].append(product)
            elif product.lebiggot_category == 'foie_gras':
                categories['foie_gras'].append(product)
            elif product.lebiggot_category == 'sweets':
                categories['sweets'].append(product)
            else:
                # Default category
                categories[product.lebiggot_category or 'other'].append(product)
        
        return categories
    
    def _apply_rule_r1(self, products):
        """R1: Repeat exact products for Cava/Champagne/Vermouth/Tokaj"""
        selected_products = []
        rule_logs = []
        
        for product in products:
            # Check stock availability
            if self._check_stock_availability(product):
                selected_products.append(product)
                rule_logs.append({
                    'rule': 'R1',
                    'action': 'repeat_exact',
                    'product': product.name,
                    'reason': f'Repeated exact {product.beverage_family} as per R1'
                })
            else:
                # Find exact substitute (same product, different supplier/vintage)
                substitute = self._find_exact_substitute(product)
                if substitute:
                    selected_products.append(substitute)
                    rule_logs.append({
                        'rule': 'R1',
                        'action': 'substitute_exact',
                        'original': product.name,
                        'substitute': substitute.name,
                        'reason': 'Stock unavailable, found exact substitute'
                    })
                else:
                    rule_logs.append({
                        'rule': 'R1',
                        'action': 'failed',
                        'product': product.name,
                        'reason': 'No exact substitute available'
                    })
        
        return selected_products, rule_logs
    
    def _apply_rule_r2(self, wine_products):
        """R2: Wines - same color, different brand, same size & grade"""
        selected_products = []
        rule_logs = []
        
        for wine in wine_products:
            # Search for different brand with same attributes
            substitute_domain = [
                ('beverage_family', '=', 'wine'),
                ('wine_color', '=', wine.wine_color),
                ('volume_ml', '=', wine.volume_ml),
                ('product_grade', '=', wine.product_grade),
                ('brand', '!=', wine.brand),
                ('active', '=', True),
                ('sale_ok', '=', True)
            ]
            
            substitutes = self.env['product.template'].search(substitute_domain)
            
            if substitutes:
                # Pick first available substitute
                for substitute in substitutes:
                    if self._check_stock_availability(substitute):
                        selected_products.append(substitute)
                        rule_logs.append({
                            'rule': 'R2',
                            'action': 'brand_change',
                            'original': f"{wine.name} ({wine.brand})",
                            'substitute': f"{substitute.name} ({substitute.brand})",
                            'attributes_maintained': f"{wine.wine_color} wine, {wine.volume_ml}ml, {wine.product_grade}"
                        })
                        break
                else:
                    # No substitute available, keep original if in stock
                    if self._check_stock_availability(wine):
                        selected_products.append(wine)
                        rule_logs.append({
                            'rule': 'R2',
                            'action': 'keep_original',
                            'product': wine.name,
                            'reason': 'No alternative brand available'
                        })
            else:
                # Keep original if no alternatives exist
                if self._check_stock_availability(wine):
                    selected_products.append(wine)
                    rule_logs.append({
                        'rule': 'R2',
                        'action': 'keep_original',
                        'product': wine.name,
                        'reason': 'No alternative brands found'
                    })
        
        return selected_products, rule_logs
    
    def _apply_rule_r3(self, experience_products):
        """R3: Replace Experiences with new same-size bundles from pool"""
        selected_products = []
        rule_logs = []
        
        for exp_product in experience_products:
            # Find new experience bundle with same product count
            original_count = self._get_experience_product_count(exp_product)
            
            new_experience = self._find_new_experience_bundle(original_count, exclude_ids=[exp_product.id])
            
            if new_experience:
                selected_products.extend(new_experience.product_ids)
                rule_logs.append({
                    'rule': 'R3',
                    'action': 'experience_replacement',
                    'original': exp_product.name,
                    'new_experience': new_experience.name,
                    'product_count': len(new_experience.product_ids)
                })
            else:
                # Keep original if no new experience available
                selected_products.append(exp_product)
                rule_logs.append({
                    'rule': 'R3',
                    'action': 'keep_original',
                    'product': exp_product.name,
                    'reason': 'No new experience bundle available'
                })
        
        return selected_products, rule_logs
    
    def _apply_rule_r4(self, charcuterie_products):
        """R4: Paletilla & Charcuterie repeated exactly"""
        selected_products = []
        rule_logs = []
        
        for product in charcuterie_products:
            if self._check_stock_availability(product):
                selected_products.append(product)
                rule_logs.append({
                    'rule': 'R4',
                    'action': 'repeat_exact',
                    'product': product.name,
                    'reason': 'Charcuterie repeated exactly as per R4'
                })
            else:
                # Find exact substitute
                substitute = self._find_exact_substitute(product)
                if substitute:
                    selected_products.append(substitute)
                    rule_logs.append({
                        'rule': 'R4',
                        'action': 'substitute_exact',
                        'original': product.name,
                        'substitute': substitute.name,
                        'reason': 'Stock unavailable, found exact substitute'
                    })
        
        return selected_products, rule_logs
    
    def _apply_rule_r5(self, foie_products):
        """R5: Foie alternates Duck ↔ Goose"""
        selected_products = []
        rule_logs = []
        
        for foie in foie_products:
            # Determine current variant and target variant
            current_variant = getattr(foie, 'foie_variant', 'duck')
            target_variant = 'goose' if current_variant == 'duck' else 'duck'
            
            # Search for opposite variant
            substitute_domain = [
                ('lebiggot_category', '=', 'foie_gras'),
                ('foie_variant', '=', target_variant),
                ('product_grade', '=', foie.product_grade),
                ('active', '=', True),
                ('sale_ok', '=', True)
            ]
            
            substitutes = self.env['product.template'].search(substitute_domain)
            
            if substitutes and self._check_stock_availability(substitutes[0]):
                selected_products.append(substitutes[0])
                rule_logs.append({
                    'rule': 'R5',
                    'action': 'foie_alternation',
                    'original': f"{foie.name} ({current_variant})",
                    'substitute': f"{substitutes[0].name} ({target_variant})",
                    'reason': f'Alternated from {current_variant} to {target_variant}'
                })
            else:
                # Keep original if no alternative
                if self._check_stock_availability(foie):
                    selected_products.append(foie)
                    rule_logs.append({
                        'rule': 'R5',
                        'action': 'keep_original',
                        'product': foie.name,
                        'reason': f'No {target_variant} variant available'
                    })
        
        return selected_products, rule_logs
    
    def _apply_rule_r6(self, sweet_products):
        """R6: Lingote repeats exact, Turrón keeps subtype & grade but brand may change"""
        selected_products = []
        rule_logs = []
        
        for sweet in sweet_products:
            sweet_subtype = getattr(sweet, 'sweets_subtype', 'other')
            
            if sweet_subtype == 'lingote':
                # Repeat exactly
                if self._check_stock_availability(sweet):
                    selected_products.append(sweet)
                    rule_logs.append({
                        'rule': 'R6',
                        'action': 'repeat_exact',
                        'product': sweet.name,
                        'reason': 'Lingote repeated exactly as per R6'
                    })
                
            elif sweet_subtype == 'turron':
                # Keep subtype & grade, brand may change
                substitute_domain = [
                    ('lebiggot_category', '=', 'sweets'),
                    ('sweets_subtype', '=', 'turron'),
                    ('product_grade', '=', sweet.product_grade),
                    ('brand', '!=', sweet.brand),
                    ('active', '=', True),
                    ('sale_ok', '=', True)
                ]
                
                substitutes = self.env['product.template'].search(substitute_domain)
                
                if substitutes and self._check_stock_availability(substitutes[0]):
                    selected_products.append(substitutes[0])
                    rule_logs.append({
                        'rule': 'R6',
                        'action': 'turron_brand_change',
                        'original': f"{sweet.name} ({sweet.brand})",
                        'substitute': f"{substitutes[0].name} ({substitutes[0].brand})",
                        'attributes_maintained': f"{sweet_subtype}, {sweet.product_grade}"
                    })
                else:
                    # Keep original
                    if self._check_stock_availability(sweet):
                        selected_products.append(sweet)
                        rule_logs.append({
                            'rule': 'R6',
                            'action': 'keep_original',
                            'product': sweet.name,
                            'reason': 'No alternative turrón brand available'
                        })
        
        return selected_products, rule_logs
    
    def _apply_global_constraints(self, result):
        """Apply global constraints: price category lock, beverage size lock, budget ±5%"""
        
        # Lock price categories
        price_categories = [p.product_grade for p in result['products'] if p.product_grade]
        result['locked_attributes']['price_categories'] = set(price_categories)
        
        # Lock beverage sizes  
        beverage_sizes = [p.volume_ml for p in result['products'] if hasattr(p, 'volume_ml') and p.volume_ml]
        result['locked_attributes']['beverage_sizes'] = set(beverage_sizes)
        
        # Budget validation will be handled at composition level
        
        return result
    
    def _check_stock_availability(self, product, min_qty=1):
        """Check if product has sufficient stock"""
        # Simplified stock check - extend with actual inventory integration
        return True  # For now, assume all products are available
    
    def _find_exact_substitute(self, product):
        """Find exact substitute (same product, different supplier/lot/vintage)"""
        domain = [
            ('name', '=', product.name),
            ('lebiggot_category', '=', product.lebiggot_category),
            ('product_grade', '=', product.product_grade),
            ('id', '!=', product.id),
            ('active', '=', True),
            ('sale_ok', '=', True)
        ]
        
        substitutes = self.env['product.template'].search(domain, limit=1)
        return substitutes[0] if substitutes else None
    
    def _get_experience_product_count(self, experience_product):
        """Get number of products in an experience bundle"""
        # This would need to be implemented based on your experience structure
        return 3  # Default
    
    def _find_new_experience_bundle(self, product_count, exclude_ids=None):
        """Find new experience bundle with same product count"""
        domain = [
            ('product_count', '=', product_count),
            ('active', '=', True)
        ]
        
        if exclude_ids:
            domain.append(('id', 'not in', exclude_ids))
        
        experiences = self.env['gift.experience'].search(domain, limit=1)
        return experiences[0] if experiences else None
    
    def validate_budget_guardrail(self, target_budget, actual_cost, tolerance=0.05):
        """Validate that actual cost is within ±5% of target budget"""
        lower_bound = target_budget * (1 - tolerance)
        upper_bound = target_budget * (1 + tolerance)
        
        return lower_bound <= actual_cost <= upper_bound
    
    def generate_rule_report(self, rule_applications):
        """Generate human-readable report of rule applications"""
        report = []
        
        rule_names = {
            'R1': 'Exact Repetition (Cava/Champagne/Vermouth/Tokaj)',
            'R2': 'Wine Brand Variation (same color/size/grade)',
            'R3': 'Experience Bundle Replacement',
            'R4': 'Exact Repetition (Paletilla/Charcuterie)',
            'R5': 'Foie Gras Alternation (Duck ↔ Goose)',
            'R6': 'Sweet Product Rules (Lingote/Turrón)'
        }
        
        for rule_app in rule_applications:
            rule_name = rule_names.get(rule_app['rule'], rule_app['rule'])
            report.append(f"**{rule_name}**: {rule_app.get('reason', 'Applied')}")
        
        return report