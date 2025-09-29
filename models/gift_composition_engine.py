# models/gift_composition_engine.py
"""
Gift Composition Engine - Integrates all recommendation strategies
Combines existing Ollama system with new business rules (R1-R6)
"""

from odoo import models, fields, api
import logging
from datetime import datetime
import json

_logger = logging.getLogger(__name__)

class GiftCompositionEngine(models.Model):
    """
    Master engine that coordinates between:
    1. Existing ollama_gift_recommender (advanced AI features)
    2. Business rules engine (R1-R6 from Master Guide)  
    3. Experience handler (2025 experiences)
    """
    
    _name = 'gift.composition.engine'
    _description = 'Gift Composition Master Engine'
    _inherit = ['mail.thread', 'mail.activity.mixin']
    
    # === EXPERIENCE SPECIFICATIONS (From EXPERIENCIAS 2025) ===
    EXPERIENCES_2025 = {
        'X-EXP-PEPERONCINI': {
            'name': 'EXP PEPERONCINI',
            'products': ['LB-PEPERON-QUE212', 'ANCHOA', 'LB-ACEITU-140'],
            'size': 3,
            'category': 'aperitivo'
        },
        'X-EXP-APER-ESF': {
            'name': 'EXP- APERITIVO ESFERIFICADO',
            'products': ['LB-MOU-BONPIPARRA120', 'ESF-CAV-NE60', 'LB-PIM-CARA110'],
            'size': 3,
            'category': 'aperitivo'
        },
        'X-EXP-GILDA-ESF': {
            'name': 'EXP- GILDA ESFERIFICADA',
            'products': ['ANCHOA SERIE ORO', 'ESF-CAV-VER60', 'MOU-ZUB-PIPA100'],
            'size': 3,
            'category': 'gilda'
        },
        'X-EXP-GILDA': {
            'name': 'EXP- GILDA',
            'products': ['ANCHOA CONSORCIO Y OLASAGASTI', 'LB-ACEITU-140', 'GUIN-AGIN212'],
            'size': 3,
            'category': 'gilda'
        },
        'X-EXP-GILDARELLENA': {
            'name': 'EXP- GILDA RELLENA',
            'products': ['PEP-ATUN-EMP314', 'LB-ACEITU-140', 'PIQUILLO'],
            'size': 3,
            'category': 'gilda'
        },
        'X-EXP-HUERTA': {
            'name': 'EXPERIENCIA DE LA HUERTA',
            'products': ['BON-ARRAIN212', 'LB-ENSA-NORM250', 'SAL-AÑA-ECO250', 'ACEITE 250 ARANA', 'VINAGRE 1980'],
            'size': 5,
            'category': 'vegetariana'
        },
        'X-EXP-BON-BORTE': {
            'name': 'EXPERIENCIA DEL NORTE',
            'products': ['PIQUILLO', 'BONITO ARROYABE', 'CREMA-OLAS-ATUN110', 'ANCHOA'],
            'size': 4,
            'category': 'seafood'
        },
        'X-EXP-MAR-MONTAÑA': {
            'name': 'EXP MAR Y MONTAÑA',
            'products': ['BONITO ARROYABE', 'LB-MOU-BONPIPARRA120', 'LB-HONGO-BOL212'],
            'size': 3,
            'category': 'premium'
        },
        'X-EXP-LUBINA': {
            'name': 'EXP LUBINA',
            'products': ['LUB-CV200', 'ENSA-MIX-EMP135', 'LB-CHU-PIMAMA95'],
            'size': 3,
            'category': 'seafood'
        }
    }
    
    # === BUSINESS RULES (R1-R6 from Master Guide) ===
    BUSINESS_RULES = {
        'R1': {
            'name': 'Experience Rotation',
            'description': 'Replace experience with different one of same size',
            'mandatory': True
        },
        'R2': {
            'name': 'Beverage Rules',
            'description': 'Cava/Champagne/Vermouth/Tokaji repeat exactly, wines rotate brand',
            'mandatory': True
        },
        'R3': {
            'name': 'Foie Gras Alternation',
            'description': 'Alternate Duck ↔ Goose each year',
            'mandatory': True
        },
        'R4': {
            'name': 'Charcuterie Preservation',
            'description': 'Paletilla must always repeat',
            'mandatory': True
        },
        'R5': {
            'name': 'Sweets Rotation',
            'description': 'Lingote & Trufas repeat, Turrón rotates',
            'mandatory': True
        },
        'R6': {
            'name': 'Multi-Level Differentiation',
            'description': 'No repeats across company levels',
            'mandatory': True
        }
    }
    
    # === FIELDS ===
    name = fields.Char('Engine Name', default='Master Composition Engine')
    ollama_recommender_id = fields.Many2one('ollama.gift.recommender', string='Ollama Engine')
    active = fields.Boolean(default=True)
    
    # === MAIN ORCHESTRATION METHOD ===
    @api.model
    def generate_complete_composition(self, partner_id, target_budget, 
                                     client_notes='', dietary_restrictions=None,
                                     composition_type=None, wizard_data=None):
        """
        Main orchestration method that combines all strategies
        """
        
        _logger.info(f"=== MASTER ENGINE START for Partner {partner_id} ===")
        
        # Step 1: Get all data sources
        context = self._gather_complete_context(
            partner_id, target_budget, client_notes, 
            dietary_restrictions, composition_type, wizard_data
        )
        
        # Step 2: Determine primary strategy
        strategy = self._determine_strategy(context)
        _logger.info(f"Selected Strategy: {strategy}")
        
        # Step 3: Execute strategy with business rules
        if strategy == 'last_year_transform':
            result = self._execute_80_20_transformation(context)
        elif strategy == 'experience_based':
            result = self._execute_experience_composition(context)
        elif strategy == 'pattern_based':
            result = self._execute_pattern_generation(context)
        elif strategy == 'new_client':
            result = self._execute_new_client_generation(context)
        else:
            result = self._execute_hybrid_generation(context)
        
        # Step 4: Apply business rules validation
        validated_result = self._apply_business_rules(result, context)
        
        # Step 5: Final optimization
        optimized_result = self._final_optimization(validated_result, context)
        
        # Step 6: Create composition record
        composition = self._create_composition_record(optimized_result, context)
        
        return {
            'success': True,
            'composition_id': composition.id,
            'products': optimized_result['products'],
            'total_cost': optimized_result['total_cost'],
            'strategy_used': strategy,
            'confidence': optimized_result.get('confidence', 0.85),
            'explanation': self._generate_explanation(optimized_result, context)
        }
    
    # === CONTEXT GATHERING ===
    def _gather_complete_context(self, partner_id, target_budget, notes, dietary, comp_type, wizard):
        """Gather all context from multiple sources"""
        
        partner = self.env['res.partner'].browse(partner_id)
        
        context = {
            'partner': partner,
            'partner_id': partner_id,
            'target_budget': target_budget,
            'year': datetime.now().year,
            'notes': notes,
            'dietary': dietary or [],
            'composition_type': comp_type,
            'wizard_data': wizard or {},
            
            # Historical data
            'last_year_order': self._get_last_year_order(partner_id),
            'order_history': self._get_order_history(partner_id),
            'purchase_patterns': self._analyze_purchase_patterns(partner_id),
            
            # Business context
            'client_level': self._determine_client_level(partner),
            'is_new_client': not bool(self._get_order_history(partner_id)),
            'has_last_year': bool(self._get_last_year_order(partner_id)),
            
            # Parsed requirements
            'requirements': {},  # Will be filled by parsing
            
            # Available products
            'available_products': self._get_available_products(dietary)
        }
        
        # Parse notes with Ollama if available
        if self.ollama_recommender_id and self.ollama_recommender_id.ollama_enabled:
            context['requirements'] = self._parse_notes_with_ollama(notes, context)
        else:
            context['requirements'] = self._basic_notes_parsing(notes)
        
        return context
    
    # === STRATEGY DETERMINATION ===
    def _determine_strategy(self, context):
        """Determine which generation strategy to use"""
        
        if context['composition_type'] == 'experience':
            return 'experience_based'
        
        if context['has_last_year']:
            # Check if requirements demand major changes
            if context['requirements'].get('major_change_requested'):
                return 'pattern_based'
            else:
                return 'last_year_transform'  # 80/20 rule
        
        if context['is_new_client']:
            return 'new_client'
        
        if len(context['order_history']) >= 3:
            return 'pattern_based'
        
        return 'hybrid_generation'
    
    # === 80/20 TRANSFORMATION (R1-R6) ===
    def _execute_80_20_transformation(self, context):
        """Apply 80/20 rule with business rules R1-R6"""
        
        last_year = context['last_year_order']
        if not last_year:
            return self._execute_pattern_generation(context)
        
        products_to_keep = []
        products_to_replace = []
        total_cost = 0
        
        # Parse last year's products
        last_products = self._parse_order_products(last_year)
        
        for product_data in last_products:
            product = product_data['product']
            category = self._categorize_product(product)
            
            # Apply business rules
            if category == 'experience':
                # R1: Replace with different experience of same size
                new_exp = self._rotate_experience(product_data)
                products_to_replace.append(new_exp)
                
            elif category in ['cava', 'champagne', 'vermouth', 'tokaji']:
                # R2: Repeat exactly
                products_to_keep.append(product)
                
            elif category in ['red_wine', 'white_wine', 'rose_wine']:
                # R2: Rotate brand, keep type
                new_wine = self._rotate_wine(product)
                products_to_replace.append(new_wine)
                
            elif category == 'foie_gras':
                # R3: Alternate Duck ↔ Goose
                new_foie = self._alternate_foie(product)
                products_to_replace.append(new_foie)
                
            elif category == 'paletilla':
                # R4: Must repeat
                products_to_keep.append(product)
                
            elif category in ['lingote', 'trufas']:
                # R5: Always repeat
                products_to_keep.append(product)
                
            elif category == 'turron':
                # R5: Rotate brand
                new_turron = self._rotate_turron(product)
                products_to_replace.append(new_turron)
                
            else:
                # 80/20 rule for others
                if self._should_keep_product(product_data, context):
                    products_to_keep.append(product)
                else:
                    replacement = self._find_replacement(product, category, context)
                    products_to_replace.append(replacement)
        
        # Combine results
        final_products = products_to_keep + products_to_replace
        total_cost = sum(p.list_price for p in final_products if p)
        
        return {
            'products': final_products,
            'total_cost': total_cost,
            'kept_products': len(products_to_keep),
            'replaced_products': len(products_to_replace),
            'method': '80_20_transform'
        }
    
    # === EXPERIENCE ROTATION (R1) ===
    def _rotate_experience(self, old_experience_data):
        """Find replacement experience of same size"""
        
        old_code = old_experience_data.get('code')
        old_exp = self.EXPERIENCES_2025.get(old_code)
        
        if not old_exp:
            return None
        
        old_size = old_exp['size']
        old_category = old_exp['category']
        
        # Find candidates of same size
        candidates = []
        for code, exp in self.EXPERIENCES_2025.items():
            if code != old_code and exp['size'] == old_size:
                # Prefer same category but not required
                if exp['category'] == old_category:
                    candidates.insert(0, code)  # Priority
                else:
                    candidates.append(code)
        
        if not candidates:
            _logger.warning(f"No replacement found for experience {old_code}")
            return None
        
        # Get the actual product
        new_code = candidates[0]
        product = self.env['product.template'].search([
            ('default_code', '=', new_code),
            ('sale_ok', '=', True)
        ], limit=1)
        
        return product
    
    # === WINE ROTATION (R2) ===
    def _rotate_wine(self, old_wine):
        """Rotate wine brand keeping same type"""
        
        wine_type = self._get_wine_type(old_wine)
        
        new_wines = self.env['product.template'].search([
            ('categ_id.name', 'ilike', wine_type),
            ('id', '!=', old_wine.id),
            ('sale_ok', '=', True),
            ('list_price', '>=', old_wine.list_price * 0.8),
            ('list_price', '<=', old_wine.list_price * 1.2)
        ], limit=5)
        
        if new_wines:
            # Prefer different brand
            for wine in new_wines:
                if self._get_product_brand(wine) != self._get_product_brand(old_wine):
                    return wine
            return new_wines[0]
        
        return old_wine  # Fallback to same if no alternative
    
    # === HELPER METHODS ===
    def _get_last_year_order(self, partner_id):
        """Get last year's order for this client"""
        
        last_year = datetime.now().year - 1
        
        orders = self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('date_order', '>=', f'{last_year}-01-01'),
            ('date_order', '<=', f'{last_year}-12-31'),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc', limit=1)
        
        return orders[0] if orders else None
    
    def _get_order_history(self, partner_id):
        """Get all historical orders"""
        
        return self.env['sale.order'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['sale', 'done'])
        ], order='date_order desc')
    
    def _apply_business_rules(self, result, context):
        """Validate and apply all business rules"""
        
        # Check R6: Multi-level differentiation
        if context.get('client_level'):
            result = self._ensure_level_differentiation(result, context)
        
        # Validate budget
        if not (context['target_budget'] * 0.95 <= result['total_cost'] <= context['target_budget'] * 1.05):
            result = self._adjust_to_budget(result, context)
        
        return result
    
    def _create_composition_record(self, result, context):
        """Create the final gift.composition record"""
        
        return self.env['gift.composition'].create({
            'partner_id': context['partner_id'],
            'target_budget': context['target_budget'],
            'target_year': context['year'],
            'product_ids': [(6, 0, [p.id for p in result['products']])],
            'actual_cost': result['total_cost'],
            'composition_type': context.get('composition_type', 'custom'),
            'generation_method': 'composition_engine',
            'confidence_score': result.get('confidence', 0.85),
            'client_notes': context.get('notes', ''),
            'dietary_restrictions': ', '.join(context.get('dietary', [])),
            'ai_reasoning': json.dumps(result, default=str),
            'state': 'draft'
        })
    
    def _generate_explanation(self, result, context):
        """Generate human-readable explanation"""
        
        explanation = f"""
        Composition Strategy: {result.get('method', 'unknown')}
        Total Products: {len(result['products'])}
        Total Cost: €{result['total_cost']:.2f}
        Target Budget: €{context['target_budget']:.2f}
        Variance: {((result['total_cost'] - context['target_budget']) / context['target_budget'] * 100):.1f}%
        """
        
        if result.get('method') == '80_20_transform':
            explanation += f"""
        Applied Business Rules:
        - Kept {result.get('kept_products', 0)} products (80%)
        - Replaced {result.get('replaced_products', 0)} products (20%)
        - Experience rotation applied: {result.get('experience_rotated', False)}
        - Wine brands rotated: {result.get('wines_rotated', 0)}
        """
        
        return explanation

    def _parse_order_products(self, order):
        """Parse products from a sale order"""
        products = []
        for line in order.order_line:
            if line.product_id and line.price_unit > 0:
                products.append({
                    'product': line.product_id.product_tmpl_id,
                    'quantity': line.product_uom_qty,
                    'price': line.price_unit,
                    'code': line.product_id.default_code
                })
        return products

    def _categorize_product(self, product):
        """Categorize product for business rules"""
        if not product:
            return 'unknown'
        
        # Check by code patterns
        code = product.default_code or ''
        name = product.name.lower()
        
        if code.startswith('X-EXP-'):
            return 'experience'
        if 'cava' in name:
            return 'cava'
        if 'champagne' in name:
            return 'champagne'
        if 'vermouth' in name:
            return 'vermouth'
        if 'tokaji' in name:
            return 'tokaji'
        if 'red wine' in name or 'vino tinto' in name:
            return 'red_wine'
        if 'white wine' in name or 'vino blanco' in name:
            return 'white_wine'
        if 'foie' in name:
            return 'foie_gras'
        if 'paletilla' in name:
            return 'paletilla'
        if 'lingote' in name:
            return 'lingote'
        if 'trufas' in name:
            return 'trufas'
        if 'turron' in name or 'turrón' in name:
            return 'turron'
        
        # Use category if available
        if hasattr(product, 'categ_id'):
            return product.categ_id.name.lower()
        
        return 'general'

    def _alternate_foie(self, old_foie):
        """Alternate between duck and goose foie"""
        name = old_foie.name.lower()
        
        if 'duck' in name or 'pato' in name:
            # Switch to goose
            search_term = 'goose'
        else:
            # Switch to duck
            search_term = 'duck'
        
        new_foie = self.env['product.template'].search([
            ('name', 'ilike', search_term),
            ('name', 'ilike', 'foie'),
            ('sale_ok', '=', True),
            ('list_price', '>=', old_foie.list_price * 0.8),
            ('list_price', '<=', old_foie.list_price * 1.2)
        ], limit=1)
        
        return new_foie if new_foie else old_foie