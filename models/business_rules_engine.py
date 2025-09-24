# models/business_rules_engine.py
from odoo import models, fields, api
from odoo.exceptions import UserError
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

_logger = logging.getLogger(__name__)

class BusinessRulesEngine(models.Model):
    _name = 'business.rules.engine'
    _description = 'Le Biggot Business Rules Engine for Gift Compositions'
    
    def _validate_product_price(self, product):
        """CRITICAL: Validate product has a meaningful price"""
        try:
            price = float(product.list_price)
            return price >= 10.0  # Minimum €10 per product
        except (ValueError, TypeError, AttributeError):
            _logger.warning(f"Invalid price for product {product.name if product else 'unknown'}")
            return False
    
    def apply_composition_rules(self, partner_id, target_year, last_composition_products=None):
        """
        Apply business rules R1-R6 for gift composition
        Returns: dict with categorized product selections and rule applications
        """
        
        if not last_composition_products:
            # New client or no history - generate fresh composition
            result = self._generate_fresh_composition(partner_id, target_year)
        else:
            # Existing client - apply transformation rules
            result = self._apply_transformation_rules(partner_id, target_year, last_composition_products)
        
        # CRITICAL: Final validation - remove any zero-price products that slipped through
        valid_products = []
        removed_count = 0
        for product in result.get('products', []):
            if self._validate_product_price(product):
                valid_products.append(product)
            else:
                _logger.error(f"FINAL CHECK: Removing {product.name} with invalid price €{product.list_price}")
                removed_count += 1
        
        result['products'] = valid_products
        
        # Log summary
        total = sum(p.list_price for p in valid_products)
        _logger.info(f"Business Rules Result: {len(valid_products)} products, Total: €{total:.2f}")
        if removed_count > 0:
            _logger.warning(f"Removed {removed_count} products with invalid prices")
        
        return result
    
    def _apply_transformation_rules(self, partner_id, target_year, last_products):
        """Apply R1-R6 transformation rules to last year's composition"""
        
        result = {
            'products': [],
            'rule_applications': [],
            'substitutions': [],
            'locked_attributes': {
                'price_categories': set(),
                'beverage_sizes': set(),
                # Items that belong to an Experience and must not be reused elsewhere
                'experience_item_ids': set(),
                # Flag to indicate if selected Experience already contains Foie
                'experience_has_foie': False,
            }
        }
        
        # Categorize last year's products
        categorized_products = self._categorize_products(last_products)
        
        # 1) Handle Experiences FIRST to derive dependencies (e.g., Foie suppression)
        experience_products = categorized_products.get('experience', [])
        if experience_products:
            new_products, rule_log, exp_meta = self._apply_rule_r3_with_meta(experience_products)
            result['products'].extend(new_products)
            result['rule_applications'].extend(rule_log)
            # Track experience items and foie presence
            result['locked_attributes']['experience_item_ids'].update(p.id for p in new_products)
            result['locked_attributes']['experience_has_foie'] = bool(exp_meta.get('has_foie'))
            # Remove to avoid double-processing
            categorized_products.pop('experience', None)
        
        # 2) Apply rules for remaining categories
        for category, products in categorized_products.items():
            if category in ['cava', 'champagne', 'vermouth', 'tokaj', 'tokaji']:
                # R1: Repeat exactly
                new_products, rule_log = self._apply_rule_r1(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
                
            elif category == 'wine':
                # R2: Same color, different brand, same size & grade
                new_products, rule_log = self._apply_rule_r2(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
                
            elif category in ['paletilla', 'charcuterie']:
                # R4: Repeat exactly
                new_products, rule_log = self._apply_rule_r4(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
                
            elif category == 'foie_gras':
                # R5: Alternate Duck ↔ Goose
                # If Experience already includes Foie, DO NOT add separate Foie
                if not result['locked_attributes'].get('experience_has_foie'):
                    new_products, rule_log = self._apply_rule_r5(products)
                    result['products'].extend(new_products)
                    result['rule_applications'].extend(rule_log)
                
            elif category == 'sweets':
                # R6: Lingote exact, Turrón keeps subtype & grade
                new_products, rule_log = self._apply_rule_r6(products)
                result['products'].extend(new_products)
                result['rule_applications'].extend(rule_log)
        
        # Apply ordering for Experience-controlled placement
        result['products'] = self._apply_experience_placement_order(
            result['products'], result['locked_attributes'].get('experience_item_ids', set())
        )
        
        # Apply global constraints
        result = self._apply_global_constraints(result)
        
        return result
    
    def _categorize_products(self, products):
        """Categorize products for rule application"""
        categories = defaultdict(list)
        
        for product in products:
            # Skip products with invalid prices
            if not self._validate_product_price(product):
                _logger.warning(f"Skipping categorization for {product.name} - invalid price €{product.list_price}")
                continue
                
            # R1: Exact repeat beverages
            if hasattr(product, 'beverage_family') and product.beverage_family:
                if product.beverage_family in ['cava', 'champagne', 'vermouth', 'tokaj', 'tokaji']:
                    categories[product.beverage_family].append(product)
                elif product.beverage_family in ['wine', 'red_wine', 'white_wine', 'rose_wine']:
                    categories['wine'].append(product)
            
            # R3: Experience bundles
            elif hasattr(product, 'is_experience_only') and product.is_experience_only:
                categories['experience'].append(product)
            
            # R4: Paletilla (MUST repeat)
            elif hasattr(product, 'is_paletilla') and product.is_paletilla:
                categories['paletilla'].append(product)
            
            # R4: Other charcuterie
            elif hasattr(product, 'is_charcuterie_item') and product.is_charcuterie_item:
                categories['charcuterie'].append(product)
            
            # R5: Foie Gras
            elif product.lebiggot_category == 'foie_gras':
                categories['foie_gras'].append(product)
            
            # R6: Sweets
            elif product.lebiggot_category == 'sweets':
                categories['sweets'].append(product)
            
            else:
                # Other products by category
                categories[product.lebiggot_category or 'other'].append(product)
        
        return categories
    
    def _apply_rule_r1(self, products):
        """R1: Repeat exact products for Cava/Champagne/Vermouth/Tokaj - WITH PRICE VALIDATION"""
        selected_products = []
        rule_logs = []
        
        for product in products:
            # CRITICAL: Check price first
            if not self._validate_product_price(product):
                _logger.error(f"R1: Skipping {product.name} - invalid price €{product.list_price}")
                # Find substitute with valid price
                substitute = self._find_exact_substitute_with_price(product)
                if substitute:
                    selected_products.append(substitute)
                    rule_logs.append({
                        'rule': 'R1',
                        'action': 'substitute_price',
                        'original': product.name,
                        'substitute': substitute.name,
                        'price': substitute.list_price,
                        'reason': f'Original had invalid price, substituted'
                    })
            elif self._check_stock_availability(product):
                selected_products.append(product)
                rule_logs.append({
                    'rule': 'R1',
                    'action': 'repeat_exact',
                    'product': product.name,
                    'price': product.list_price,
                    'reason': f'Repeated exact {product.beverage_family} as per R1'
                })
            else:
                # Find exact substitute (same product, different supplier/vintage)
                substitute = self._find_exact_substitute_with_price(product)
                if substitute:
                    selected_products.append(substitute)
                    rule_logs.append({
                        'rule': 'R1',
                        'action': 'substitute_exact',
                        'original': product.name,
                        'substitute': substitute.name,
                        'price': substitute.list_price,
                        'reason': 'Stock unavailable, found exact substitute'
                    })
                else:
                    rule_logs.append({
                        'rule': 'R1',
                        'action': 'failed',
                        'product': product.name,
                        'reason': 'No exact substitute available with valid price'
                    })
        
        return selected_products, rule_logs
    
    def _apply_rule_r2(self, wine_products):
        """R2: Wines - same color, different brand, same size & grade - WITH PRICE VALIDATION"""
        selected_products = []
        rule_logs = []
        
        for wine in wine_products:
            # CRITICAL: Skip wines with invalid prices
            if not self._validate_product_price(wine):
                _logger.error(f"R2: Wine {wine.name} has invalid price €{wine.list_price}")
                continue
                
            # Determine wine color
            wine_color = wine.wine_color if hasattr(wine, 'wine_color') else None
            if not wine_color:
                # Try to detect from beverage_family
                if hasattr(wine, 'beverage_family'):
                    if wine.beverage_family == 'red_wine':
                        wine_color = 'red'
                    elif wine.beverage_family == 'white_wine':
                        wine_color = 'white'
                    elif wine.beverage_family == 'rose_wine':
                        wine_color = 'rose'
            
            # Search for different brand with same attributes AND valid price
            substitute_domain = [
                ('lebiggot_category', '=', 'wines'),
                ('active', '=', True),
                ('sale_ok', '=', True),
                ('list_price', '>=', max(10.0, wine.list_price * 0.7)),  # Similar price range
                ('list_price', '<=', wine.list_price * 1.3)
            ]
            
            if wine_color:
                substitute_domain.append(('wine_color', '=', wine_color))
            
            if hasattr(wine, 'volume_ml') and wine.volume_ml:
                substitute_domain.append(('volume_ml', '=', wine.volume_ml))
            
            if hasattr(wine, 'product_grade') and wine.product_grade:
                substitute_domain.append(('product_grade', '=', wine.product_grade))
            
            if hasattr(wine, 'brand') and wine.brand:
                substitute_domain.append(('brand', '!=', wine.brand))
            
            substitutes = self.env['product.template'].search(substitute_domain)
            
            if substitutes:
                # Pick first available substitute with valid price
                for substitute in substitutes:
                    if self._validate_product_price(substitute) and self._check_stock_availability(substitute):
                        selected_products.append(substitute)
                        rule_logs.append({
                            'rule': 'R2',
                            'action': 'brand_change',
                            'original': f"{wine.name} ({wine.brand if hasattr(wine, 'brand') else 'N/A'})",
                            'substitute': f"{substitute.name} ({substitute.brand})",
                            'price': substitute.list_price,
                            'attributes_maintained': f"{wine_color or 'unknown'} wine, {wine.volume_ml if hasattr(wine, 'volume_ml') else 'N/A'}ml, {wine.product_grade}"
                        })
                        break
                else:
                    # No substitute available, keep original if in stock and has valid price
                    if self._validate_product_price(wine) and self._check_stock_availability(wine):
                        selected_products.append(wine)
                        rule_logs.append({
                            'rule': 'R2',
                            'action': 'keep_original',
                            'product': wine.name,
                            'price': wine.list_price,
                            'reason': 'No alternative brand available with valid price'
                        })
            else:
                # Keep original if no alternatives exist
                if self._validate_product_price(wine) and self._check_stock_availability(wine):
                    selected_products.append(wine)
                    rule_logs.append({
                        'rule': 'R2',
                        'action': 'keep_original',
                        'product': wine.name,
                        'price': wine.list_price,
                        'reason': 'No alternative brands found'
                    })
        
        return selected_products, rule_logs
    
    def _apply_rule_r3(self, experience_products):
        """R3: Replace Experiences with new same-size bundles from pool"""
        selected_products = []
        rule_logs = []
        
        for exp_product in experience_products:
            # Determine target pack size
            original_count = self._get_experience_product_count(exp_product)
            
            # Prefer wizard-defined experiences dictionary if available
            experience_items = self._choose_experience_from_dictionary(original_count)
            if experience_items:
                # Validate all items have valid prices
                all_valid = all(self._validate_product_price(item) for item in experience_items)
                if all_valid:
                    selected_products.extend(experience_items)
                    rule_logs.append({
                        'rule': 'R3',
                        'action': 'experience_replacement',
                        'original': exp_product.name,
                        'new_experience': 'Wizard: EXPERIENCES_DATA',
                        'product_count': original_count,
                        'total_price': sum(item.list_price for item in experience_items)
                    })
                    continue
            
            # Fallback to product-based bundles in database
            new_experience = self._find_new_experience_bundle(original_count, exclude_ids=[exp_product.id])
            if new_experience and self._validate_product_price(new_experience):
                if hasattr(new_experience, 'experience_product_ids'):
                    # Validate all products in bundle
                    if all(self._validate_product_price(p) for p in new_experience.experience_product_ids):
                        selected_products.extend(new_experience.experience_product_ids)
                else:
                    selected_products.append(new_experience)
                rule_logs.append({
                    'rule': 'R3',
                    'action': 'experience_replacement',
                    'original': exp_product.name,
                    'new_experience': new_experience.name,
                    'product_count': original_count,
                    'price': new_experience.list_price
                })
            else:
                # Keep original if valid price
                if self._validate_product_price(exp_product):
                    selected_products.append(exp_product)
                    rule_logs.append({
                        'rule': 'R3',
                        'action': 'keep_original',
                        'product': exp_product.name,
                        'price': exp_product.list_price,
                        'reason': 'No new experience bundle available with valid price'
                    })
        
        return selected_products, rule_logs

    def _apply_rule_r3_with_meta(self, experience_products):
        """Like _apply_rule_r3 but returns metadata: {'has_foie': bool}"""
        selected_products, rule_logs = self._apply_rule_r3(experience_products)
        has_foie = False
        for p in selected_products:
            try:
                if getattr(p, 'lebiggot_category', None) == 'foie_gras':
                    has_foie = True
                    break
            except Exception:
                pass
        return selected_products, rule_logs, {'has_foie': has_foie}
    
    def _apply_rule_r4(self, charcuterie_products):
        """R4: Paletilla & Charcuterie repeated exactly - WITH PRICE VALIDATION"""
        selected_products = []
        rule_logs = []
        
        for product in charcuterie_products:
            # Paletilla MUST be repeated
            is_paletilla = hasattr(product, 'is_paletilla') and product.is_paletilla
            
            # Check price first
            if not self._validate_product_price(product):
                substitute = self._find_exact_substitute_with_price(product)
                if substitute:
                    selected_products.append(substitute)
                    rule_logs.append({
                        'rule': 'R4',
                        'action': 'substitute_price',
                        'original': product.name,
                        'substitute': substitute.name,
                        'price': substitute.list_price,
                        'reason': 'Invalid price, found substitute'
                    })
                elif is_paletilla:
                    rule_logs.append({
                        'rule': 'R4',
                        'action': 'CRITICAL_FAILURE',
                        'product': product.name,
                        'reason': 'PALETILLA WITH INVALID PRICE - MUST BE RESOLVED'
                    })
            elif self._check_stock_availability(product):
                selected_products.append(product)
                rule_logs.append({
                    'rule': 'R4',
                    'action': 'repeat_exact',
                    'product': product.name,
                    'price': product.list_price,
                    'reason': f"{'Paletilla' if is_paletilla else 'Charcuterie'} repeated exactly as per R4"
                })
            else:
                # Find exact substitute
                substitute = self._find_exact_substitute_with_price(product)
                if substitute:
                    selected_products.append(substitute)
                    rule_logs.append({
                        'rule': 'R4',
                        'action': 'substitute_exact',
                        'original': product.name,
                        'substitute': substitute.name,
                        'price': substitute.list_price,
                        'reason': 'Stock unavailable, found exact substitute'
                    })
                elif is_paletilla:
                    # Paletilla is critical - log error
                    rule_logs.append({
                        'rule': 'R4',
                        'action': 'CRITICAL_FAILURE',
                        'product': product.name,
                        'reason': 'PALETILLA NOT AVAILABLE - MUST BE RESOLVED'
                    })
        
        return selected_products, rule_logs
    
    def _apply_rule_r5(self, foie_products):
        """R5: Foie alternates Duck ↔ Goose - WITH PRICE VALIDATION"""
        selected_products = []
        rule_logs = []
        
        for foie in foie_products:
            # Skip products with invalid prices
            if not self._validate_product_price(foie):
                _logger.error(f"R5: Foie {foie.name} has invalid price €{foie.list_price}")
                continue
                
            # Determine current variant and target variant
            current_variant = getattr(foie, 'foie_variant', None)
            
            if not current_variant:
                # Try to detect from name
                if 'duck' in foie.name.lower() or 'pato' in foie.name.lower():
                    current_variant = 'duck'
                elif 'goose' in foie.name.lower() or 'oca' in foie.name.lower():
                    current_variant = 'goose'
                else:
                    current_variant = 'duck'  # Default assumption
            
            target_variant = 'goose' if current_variant == 'duck' else 'duck'
            
            # Search for opposite variant with valid price
            substitute_domain = [
                ('lebiggot_category', '=', 'foie_gras'),
                ('foie_variant', '=', target_variant),
                ('product_grade', '=', foie.product_grade),
                ('active', '=', True),
                ('sale_ok', '=', True),
                ('list_price', '>=', 10.0)  # Minimum price
            ]
            
            substitutes = self.env['product.template'].search(substitute_domain)
            
            # Find substitute with valid price
            selected = None
            for sub in substitutes:
                if self._validate_product_price(sub) and self._check_stock_availability(sub):
                    selected = sub
                    break
            
            if selected:
                selected_products.append(selected)
                rule_logs.append({
                    'rule': 'R5',
                    'action': 'foie_alternation',
                    'original': f"{foie.name} ({current_variant})",
                    'substitute': f"{selected.name} ({target_variant})",
                    'price': selected.list_price,
                    'reason': f'Alternated from {current_variant} to {target_variant}'
                })
            else:
                # Keep original if no alternative and has valid price
                if self._validate_product_price(foie) and self._check_stock_availability(foie):
                    selected_products.append(foie)
                    rule_logs.append({
                        'rule': 'R5',
                        'action': 'keep_original',
                        'product': foie.name,
                        'price': foie.list_price,
                        'reason': f'No {target_variant} variant available with valid price'
                    })
        
        return selected_products, rule_logs
    
    def _apply_rule_r6(self, sweet_products):
        """R6: Lingote/Trufas repeat exact, Turrón keeps subtype & grade but brand may change - WITH PRICE VALIDATION"""
        selected_products = []
        rule_logs = []
        
        for sweet in sweet_products:
            # Skip products with invalid prices
            if not self._validate_product_price(sweet):
                _logger.error(f"R6: Sweet {sweet.name} has invalid price €{sweet.list_price}")
                continue
                
            # Check if it's Lingote or Trufa leBigott (must repeat exactly)
            is_lingote = hasattr(sweet, 'is_lingote') and sweet.is_lingote
            is_trufa = hasattr(sweet, 'is_trufa_lebigott') and sweet.is_trufa_lebigott
            
            if is_lingote or is_trufa:
                # Must repeat exactly
                if self._check_stock_availability(sweet):
                    selected_products.append(sweet)
                    rule_logs.append({
                        'rule': 'R6',
                        'action': 'repeat_exact',
                        'product': sweet.name,
                        'price': sweet.list_price,
                        'reason': f"{'Lingote' if is_lingote else 'Trufa leBigott'} repeated exactly as per R6"
                    })
                else:
                    # Critical - these must be available
                    substitute = self._find_exact_substitute_with_price(sweet)
                    if substitute:
                        selected_products.append(substitute)
                        rule_logs.append({
                            'rule': 'R6',
                            'action': 'substitute_exact',
                            'original': sweet.name,
                            'substitute': substitute.name,
                            'price': substitute.list_price,
                            'reason': 'Found exact substitute'
                        })
                
            else:
                # Check if it's Turrón
                sweet_subtype = getattr(sweet, 'sweets_subtype', None)
                turron_style = getattr(sweet, 'turron_style', None)
                
                if sweet_subtype == 'turron' or turron_style:
                    # Keep style & grade, brand may change
                    substitute_domain = [
                        ('lebiggot_category', '=', 'sweets'),
                        ('sweets_subtype', '=', 'turron'),
                        ('product_grade', '=', sweet.product_grade),
                        ('active', '=', True),
                        ('sale_ok', '=', True),
                        ('list_price', '>=', 10.0)  # Minimum price
                    ]
                    
                    if turron_style:
                        substitute_domain.append(('turron_style', '=', turron_style))
                    
                    if hasattr(sweet, 'brand') and sweet.brand:
                        # Try different brand first
                        alt_domain = substitute_domain + [('brand', '!=', sweet.brand)]
                        substitutes = self.env['product.template'].search(alt_domain)
                    else:
                        substitutes = self.env['product.template'].search(substitute_domain)
                    
                    # Find substitute with valid price
                    selected = None
                    for sub in substitutes:
                        if self._validate_product_price(sub) and self._check_stock_availability(sub):
                            selected = sub
                            break
                    
                    if selected:
                        selected_products.append(selected)
                        rule_logs.append({
                            'rule': 'R6',
                            'action': 'turron_variation',
                            'original': f"{sweet.name}",
                            'substitute': f"{selected.name}",
                            'price': selected.list_price,
                            'attributes_maintained': f"Turrón {turron_style or 'style'}, {sweet.product_grade}"
                        })
                    else:
                        # Keep original
                        if self._check_stock_availability(sweet):
                            selected_products.append(sweet)
                            rule_logs.append({
                                'rule': 'R6',
                                'action': 'keep_original',
                                'product': sweet.name,
                                'price': sweet.list_price,
                                'reason': 'No alternative turrón available'
                            })
                else:
                    # Other sweets - can be changed
                    substitute_domain = [
                        ('lebiggot_category', '=', 'sweets'),
                        ('product_grade', '=', sweet.product_grade),
                        ('id', '!=', sweet.id),
                        ('active', '=', True),
                        ('sale_ok', '=', True),
                        ('list_price', '>=', 10.0)  # Minimum price
                    ]
                    
                    substitutes = self.env['product.template'].search(substitute_domain, limit=5)
                    
                    # Find substitute with valid price
                    selected = None
                    for sub in substitutes:
                        if self._validate_product_price(sub) and self._check_stock_availability(sub):
                            selected = sub
                            break
                    
                    if selected:
                        selected_products.append(selected)
                        rule_logs.append({
                            'rule': 'R6',
                            'action': 'sweet_variation',
                            'original': sweet.name,
                            'substitute': selected.name,
                            'price': selected.list_price,
                            'reason': 'Sweet product varied for freshness'
                        })
                    else:
                        selected_products.append(sweet)
                        rule_logs.append({
                            'rule': 'R6',
                            'action': 'keep_original',
                            'product': sweet.name,
                            'price': sweet.list_price,
                            'reason': 'Keeping original sweet'
                        })
        
        return selected_products, rule_logs
    
    def _apply_global_constraints(self, result):
        """Apply global constraints: price category lock, beverage size lock, budget ±5%"""
        
        # Lock price categories
        price_categories = []
        for p in result['products']:
            if hasattr(p, 'product_grade') and p.product_grade:
                price_categories.append(p.product_grade)
        result['locked_attributes']['price_categories'] = set(price_categories)
        
        # Lock beverage sizes  
        beverage_sizes = []
        for p in result['products']:
            if hasattr(p, 'volume_ml') and p.volume_ml:
                beverage_sizes.append(p.volume_ml)
        result['locked_attributes']['beverage_sizes'] = set(beverage_sizes)
        
        # Budget validation will be handled at composition level
        
        return result

    # ===================== EXPERIENCE HELPERS (Wizard Dictionary) =====================
    def _load_experiences_data(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load EXPERIENCES_DATA dict from the wizard module if available."""
        try:
            from odoo.addons.lebigott_ai_recommendations_14.wizard.ollama_recommendation_wizard import (
                OllamaRecommendationWizard,
            )
            return getattr(OllamaRecommendationWizard, 'EXPERIENCES_DATA', None)
        except Exception:
            try:
                # Fallback based on relative import when module name differs
                from ..wizard.ollama_recommendation_wizard import OllamaRecommendationWizard  # type: ignore
                return getattr(OllamaRecommendationWizard, 'EXPERIENCES_DATA', None)
            except Exception:
                _logger.warning("EXPERIENCES_DATA not found in wizard; falling back to DB bundles if any.")
                return None

    def _resolve_experience_products(self, product_codes: List[str]) -> List[Any]:
        """Resolve list of default_code strings to product.template records."""
        resolved: List[Any] = []
        Product = self.env['product.template'].sudo()
        for code in product_codes:
            product = Product.search([
                ('default_code', '=', code),
                ('sale_ok', '=', True),
                ('active', '=', True),
                ('list_price', '>=', 10.0)  # CRITICAL: Minimum price
            ], limit=1)
            if not product:
                # Fallback: try name ilike
                product = Product.search([
                    ('name', 'ilike', code),
                    ('sale_ok', '=', True),
                    ('active', '=', True),
                    ('list_price', '>=', 10.0)  # CRITICAL: Minimum price
                ], limit=1)
            if product and self._validate_product_price(product) and self._check_stock_availability(product):
                resolved.append(product)
        return resolved

    def _choose_experience_from_dictionary(self, target_size: int) -> List[Any]:
        """Pick a different experience with the same number of items from EXPERIENCES_DATA."""
        data = self._load_experiences_data()
        if not data:
            return []
        # Build candidates by size
        candidates: List[List[Any]] = []
        for key, exp in data.items():
            products_list = exp.get('products') or []
            if not isinstance(products_list, list):
                continue
            if len(products_list) == target_size:
                resolved = self._resolve_experience_products(products_list)
                if len(resolved) == target_size:
                    # Validate all products have valid prices
                    if all(self._validate_product_price(p) for p in resolved):
                        candidates.append(resolved)
        # Pick first viable candidate for determinism
        return candidates[0] if candidates else []

    def _apply_experience_placement_order(self, products: List[Any], experience_item_ids: set) -> List[Any]:
        """Ensure experience items appear before others in their category and apply specific ordering.

        Specific sequence within experience items by keywords:
        1) fish loins (ventresca/loin/loin-like)
        2) goose foie block/mousse (oca/goose + foie)
        3) cheesecake
        4) anchovies
        """
        def specific_rank(p: Any) -> int:
            name = (getattr(p, 'name', '') or '').lower()
            if any(k in name for k in ['ventresca', 'loin', 'lomo de atún', 'fish loin']):
                return 0
            if ('foie' in name) and any(k in name for k in ['oca', 'goose', 'bloc', 'mousse']):
                return 1
            if 'cheesecake' in name:
                return 2
            if any(k in name for k in ['anchoa', 'anchovy', 'anchovies']):
                return 3
            return 9

        def category_key(p: Any) -> str:
            # Coarse grouping to place experience items before others in same broad category
            cat = getattr(p, 'lebiggot_category', '') or ''
            return cat

        def sort_key(p: Any):
            in_exp = 0 if p.id in experience_item_ids else 1
            return (category_key(p), in_exp, specific_rank(p))

        try:
            return sorted(products, key=sort_key)
        except Exception:
            return products
    
    def _check_stock_availability(self, product, min_qty=1):
        """Check if product has sufficient stock - WITH PRICE VALIDATION"""
        # FIRST check if product has valid price
        if not self._validate_product_price(product):
            _logger.warning(f"❌ Product {product.name} has invalid price: €{product.list_price}")
            return False
            
        # Check has_stock computed field if available
        if hasattr(product, 'has_stock'):
            return product.has_stock
        
        # Check qty_available
        if hasattr(product, 'qty_available'):
            return product.qty_available >= min_qty
        
        # Check through variants
        for variant in product.product_variant_ids:
            stock_quants = self.env['stock.quant'].search([
                ('product_id', '=', variant.id),
                ('location_id.usage', '=', 'internal')
            ])
            if sum(stock_quants.mapped('available_quantity')) >= min_qty:
                return True
        
        return False
    
    def _find_exact_substitute(self, product):
        """Find exact substitute (same product, different supplier/lot/vintage) - DEPRECATED"""
        # Use the new method with price validation
        return self._find_exact_substitute_with_price(product)
    
    def _find_exact_substitute_with_price(self, product):
        """Find substitute with VALID PRICE (min €10)"""
        domain = [
            ('name', '=', product.name),
            ('lebiggot_category', '=', product.lebiggot_category),
            ('product_grade', '=', product.product_grade),
            ('id', '!=', product.id),
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('list_price', '>=', 10.0),  # CRITICAL: Minimum price
            ('qty_available', '>', 0)
        ]
        
        # Preserve critical attributes
        if hasattr(product, 'volume_ml') and product.volume_ml:
            domain.append(('volume_ml', '=', product.volume_ml))
        
        if hasattr(product, 'beverage_family') and product.beverage_family:
            domain.append(('beverage_family', '=', product.beverage_family))
        
        if hasattr(product, 'is_paletilla') and product.is_paletilla:
            domain.append(('is_paletilla', '=', True))
        
        if hasattr(product, 'is_lingote') and product.is_lingote:
            domain.append(('is_lingote', '=', True))
        
        substitutes = self.env['product.template'].search(domain, limit=5)
        
        for sub in substitutes:
            if self._validate_product_price(sub) and self._check_stock_availability(sub):
                return sub
        
        return None
    
    def _get_experience_product_count(self, experience_product):
        """Get number of products in an experience bundle"""
        if hasattr(experience_product, 'experience_product_count'):
            return experience_product.experience_product_count or 3
        
        # Try to count from related products
        if hasattr(experience_product, 'experience_product_ids'):
            return len(experience_product.experience_product_ids)
        
        return 3  # Default
    
    def _find_new_experience_bundle(self, product_count, exclude_ids=None):
        """Find new experience bundle with same product count - WITH PRICE VALIDATION"""
        domain = [
            ('is_experience_only', '=', True),
            ('experience_product_count', '=', product_count),
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('list_price', '>=', 10.0)  # Minimum price
        ]
        
        if exclude_ids:
            domain.append(('id', 'not in', exclude_ids))
        
        experiences = self.env['product.template'].search(domain)
        
        # Find one with stock and valid price
        for exp in experiences:
            if self._validate_product_price(exp) and self._check_stock_availability(exp):
                return exp
        
        return None
    
    def _generate_fresh_composition(self, partner_id, target_year):
        """Generate fresh composition for new clients - WITH STRICT PRICE VALIDATION"""
        
        budget = 1000  # Default budget
        min_price = 10.0  # Never select products under €10
        max_price = budget * 0.4  # No single product over 40% of budget
        
        domain = [
            ('sale_ok', '=', True),
            ('active', '=', True),
            ('list_price', '>=', min_price),
            ('list_price', '<=', max_price),
            ('qty_available', '>', 0)
        ]
        
        products = self.env['product.template'].sudo().search(domain, limit=500)
        
        # CRITICAL: Double-validate prices
        valid_products = []
        for p in products:
            if self._validate_product_price(p) and self._check_stock_availability(p):
                valid_products.append(p)
        
        if not valid_products:
            _logger.error("No valid products found for fresh composition!")
            return {
                'products': [],
                'rule_applications': [{
                    'rule': 'NEW_CLIENT',
                    'action': 'failed',
                    'reason': 'No products with valid prices found'
                }],
                'substitutions': [],
                'locked_attributes': {}
            }
        
        # Select a balanced mix
        selected = []
        categories_needed = {
            'wine': 3,
            'foie_gras': 1,
            'charcuterie': 2,
            'experience': 1,
            'sweets': 2,
            'other': 3
        }
        
        for category, count in categories_needed.items():
            cat_products = [p for p in valid_products if self._get_product_category(p) == category]
            for product in cat_products[:count]:
                if self._validate_product_price(product):
                    selected.append(product)
        
        # If not enough products, add more from any category
        while len(selected) < 12:
            for p in valid_products:
                if p not in selected and self._validate_product_price(p):
                    selected.append(p)
                    if len(selected) >= 12:
                        break
        
        total = sum(p.list_price for p in selected)
        
        return {
            'products': selected,
            'rule_applications': [{
                'rule': 'NEW_CLIENT',
                'action': 'fresh_generation',
                'reason': 'No previous history - generating fresh composition',
                'product_count': len(selected),
                'total_price': total
            }],
            'substitutions': [],
            'locked_attributes': {}
        }
    
    def _get_product_category(self, product):
        """Categorize product for rules application"""
        if hasattr(product, 'beverage_family'):
            if product.beverage_family in ['wine', 'red_wine', 'white_wine', 'rose_wine']:
                return 'wine'
            elif product.beverage_family in ['cava', 'champagne']:
                return 'champagne'
        
        if hasattr(product, 'lebiggot_category'):
            if product.lebiggot_category == 'foie_gras':
                return 'foie_gras'
            elif product.lebiggot_category == 'charcuterie':
                return 'charcuterie'
            elif product.lebiggot_category == 'sweets':
                return 'sweets'
            elif product.lebiggot_category == 'experience':
                return 'experience'
        
        return 'other'
    
    def validate_budget_guardrail(self, target_budget, actual_cost, tolerance=0.05):
        """Validate that actual cost is within ±5% of target budget"""
        lower_bound = target_budget * (1 - tolerance)
        upper_bound = target_budget * (1 + tolerance)
        
        return lower_bound <= actual_cost <= upper_bound
    
    def generate_rule_report(self, rule_applications):
        """Generate human-readable report of rule applications"""
        report = []
        
        rule_names = {
            'R1': 'Exact Repetition (Cava/Champagne/Vermouth/Tokaji)',
            'R2': 'Wine Brand Variation (same color/size/grade)',
            'R3': 'Experience Bundle Replacement',
            'R4': 'Exact Repetition (Paletilla/Charcuterie)',
            'R5': 'Foie Gras Alternation (Duck ↔ Goose)',
            'R6': 'Sweet Product Rules (Lingote/Turrón)',
            'NEW_CLIENT': 'New Client - Fresh Composition'
        }
        
        for rule_app in rule_applications:
            rule_name = rule_names.get(rule_app['rule'], rule_app['rule'])
            report.append(f"**{rule_name}**: {rule_app.get('reason', 'Applied')}")
        
        return report