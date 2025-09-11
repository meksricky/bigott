# -*- coding: utf-8 -*-
from odoo import models, fields, api
from odoo.exceptions import UserError
import logging
import json
from datetime import datetime

_logger = logging.getLogger(__name__)

class SimplifiedCompositionEngine(models.Model):
    _name = 'simplified.composition.engine'
    _description = 'Simplified Budget-Focused Composition Engine'
    
    @api.model
    def generate_composition(self, partner_id, target_budget, target_year=None, dietary_restrictions=None, notes_text=None):
        """Main entry point - simplified approach focused on budget and existing patterns"""
        
        if not target_year:
            target_year = datetime.now().year
            
        if target_budget <= 0:
            raise UserError("Target budget must be greater than 0")
            
        # Step 1: Get available products (only internal reference + stock filters)
        available_products = self._get_available_products(dietary_restrictions)
        
        if not available_products:
            raise UserError("No products available that meet the criteria")
        
        # Step 2: Check if customer has history
        client_history = self._get_client_history(partner_id)
        
        if client_history:
            # Existing customer - modify their last composition
            composition = self._generate_from_history(
                partner_id, target_budget, available_products, client_history
            )
        else:
            # New customer - find similar customers and adapt
            composition = self._generate_from_similar_patterns(
                partner_id, target_budget, available_products
            )
        
        # Step 3: Apply business rules to final composition
        final_composition = self._apply_business_rules(composition, dietary_restrictions)
        
        # Step 4: Ensure budget compliance (+/- 5%)
        final_composition = self._ensure_budget_compliance_strict(final_composition, target_budget)
        
        # Step 5: Create the composition record
        composition_record = self._create_composition_record(
            partner_id, target_budget, target_year, final_composition, dietary_restrictions, notes_text
        )
        
        return {
            'composition_id': composition_record.id,
            'products': final_composition,
            'total_cost': sum(p.list_price for p in final_composition),
            'product_count': len(final_composition),
            'method_used': 'history_based' if client_history else 'pattern_based',
            'reasoning': f"Generated using {('history_based' if client_history else 'pattern_based')} method with {len(final_composition)} products"
        }
    
    def _get_available_products(self, dietary_restrictions=None):
        """Get products with only essential filters: internal reference + stock + basic dietary"""
        
        domain = [
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('list_price', '>', 0),
            ('default_code', '!=', False),  # Must have internal reference
        ]
        
        products = self.env['product.template'].search(domain)
        
        # Filter by stock availability
        available_products = []
        for product in products:
            if self._has_stock(product):
                available_products.append(product)
        
        # Apply simple dietary restrictions
        if dietary_restrictions:
            available_products = self._apply_dietary_filters(available_products, dietary_restrictions)
        
        _logger.info(f"Found {len(available_products)} available products")
        return available_products
    
    def _has_stock(self, product):
        """Simple stock check"""
        if product.type != 'product':
            return True
            
        stock_quants = self.env['stock.quant'].search([
            ('product_id', 'in', product.product_variant_ids.ids),
            ('location_id.usage', '=', 'internal')
        ])
        
        available_qty = sum(quant.available_quantity for quant in stock_quants)
        return available_qty > 0
    
    def _apply_dietary_filters(self, products, dietary_restrictions):
        """Simple keyword-based dietary filtering"""
        if not dietary_restrictions:
            return products
        
        filtered_products = []
        
        for product in products:
            include_product = True
            name_lower = product.name.lower()
            
            # Handle different restriction formats
            restrictions = dietary_restrictions if isinstance(dietary_restrictions, list) else [dietary_restrictions]
            
            for restriction in restrictions:
                if restriction == 'vegetarian':
                    if any(word in name_lower for word in ['meat', 'fish', 'ham', 'chorizo']):
                        include_product = False
                        break
                elif restriction == 'halal':
                    if any(word in name_lower for word in ['pork', 'wine', 'alcohol']):
                        include_product = False
                        break
                elif restriction == 'no_alcohol':
                    if any(word in name_lower for word in ['wine', 'champagne', 'alcohol', 'beer', 'cava']):
                        include_product = False
                        break
            
            if include_product:
                filtered_products.append(product)
        
        return filtered_products
    
    def _get_client_history(self, partner_id):
        """Get client's purchase history"""
        history = self.env['client.order.history'].search([
            ('partner_id', '=', partner_id)
        ], order='order_year desc', limit=1)
        
        if history:
            return {
                'last_products': history.product_ids,
                'last_budget': history.total_budget,
                'preferred_categories': json.loads(history.category_breakdown or '{}'),
                'box_type': history.box_type
            }
        return None
    
    def _generate_from_history(self, partner_id, target_budget, available_products, client_history):
        """For existing customers: modify their last composition"""
        
        last_products = client_history['last_products']
        composition = []
        current_cost = 0
        
        # Try to include similar products from last year (70% retention)
        for product in last_products:
            if current_cost >= target_budget * 0.95:
                break
                
            # Check if exact product is available
            if product in available_products and (current_cost + product.list_price) <= target_budget:
                composition.append(product)
                current_cost += product.list_price
                continue
            
            # Find similar product in same category
            similar = self._find_similar_product(product, available_products, target_budget - current_cost)
            if similar:
                composition.append(similar)
                current_cost += similar.list_price
        
        # Fill remaining budget with new products
        for product in available_products:
            if current_cost >= target_budget * 0.95:
                break
            if product not in composition and (current_cost + product.list_price) <= target_budget:
                composition.append(product)
                current_cost += product.list_price
        
        return composition
    
    def _find_similar_product(self, original_product, available_products, max_price):
        """Find a product similar to the original within price range"""
        
        # First try: same category, similar price
        original_category = getattr(original_product, 'lebiggot_category', 'other')
        original_price = original_product.list_price
        
        candidates = []
        for product in available_products:
            if getattr(product, 'lebiggot_category', 'other') == original_category:
                if product.list_price <= max_price:
                    price_diff = abs(product.list_price - original_price)
                    candidates.append((product, price_diff))
        
        if candidates:
            candidates.sort(key=lambda x: x[1])  # Sort by price difference
            return candidates[0][0]
        
        # Fallback: any product in price range
        for product in available_products:
            if product.list_price <= max_price:
                return product
        
        return None
    
    def _generate_from_similar_patterns(self, partner_id, target_budget, available_products):
        """For new customers: find patterns from similar budget customers"""
        
        # Find customers with similar budget ranges
        similar_histories = self.env['client.order.history'].search([
            ('total_budget', '>=', target_budget * 0.8),
            ('total_budget', '<=', target_budget * 1.2)
        ], order='order_year desc', limit=10)
        
        if not similar_histories:
            similar_histories = self.env['client.order.history'].search([], order='order_year desc', limit=5)
        
        # Analyze patterns
        category_frequency = {}
        for history in similar_histories:
            categories = json.loads(history.category_breakdown or '{}')
            for category, count in categories.items():
                category_frequency[category] = category_frequency.get(category, 0) + count
        
        # Build composition based on patterns
        composition = []
        current_cost = 0
        
        # Sort categories by popularity
        sorted_categories = sorted(category_frequency.items(), key=lambda x: x[1], reverse=True)
        
        for category, _ in sorted_categories:
            if current_cost >= target_budget * 0.95:
                break
            
            category_products = [p for p in available_products 
                               if getattr(p, 'lebiggot_category', 'other') == category 
                               and p not in composition]
            
            if category_products:
                category_products.sort(key=lambda p: p.list_price)
                for product in category_products:
                    if (current_cost + product.list_price) <= target_budget:
                        composition.append(product)
                        current_cost += product.list_price
                        break
        
        # Fill remaining budget
        remaining_products = [p for p in available_products if p not in composition]
        remaining_products.sort(key=lambda p: p.list_price)
        
        for product in remaining_products:
            if current_cost >= target_budget * 0.95:
                break
            if (current_cost + product.list_price) <= target_budget:
                composition.append(product)
                current_cost += product.list_price
        
        return composition
    
    def _apply_business_rules(self, composition, dietary_restrictions):
        """Apply business rules to any composition"""
        
        enhanced_composition = []
        for product in composition:
            if self._has_stock(product):
                enhanced_composition.append(product)
            else:
                substitute = self._find_substitute(product, composition)
                if substitute:
                    enhanced_composition.append(substitute)
        
        return enhanced_composition
    
    def _find_substitute(self, product, current_composition):
        """Find substitute using simple business logic"""
        
        # Look for similar product in same category
        original_category = getattr(product, 'lebiggot_category', 'other')
        
        substitutes = self.env['product.template'].search([
            ('lebiggot_category', '=', original_category),
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('default_code', '!=', False),
            ('list_price', '>=', product.list_price * 0.8),
            ('list_price', '<=', product.list_price * 1.2)
        ])
        
        for substitute in substitutes:
            if substitute not in current_composition and self._has_stock(substitute):
                return substitute
        
        return None
    
    def _ensure_budget_compliance_strict(self, composition, target_budget):
        """Ensure composition is within +/- 5% of target budget"""
        
        if not composition:
            return composition
        
        current_cost = sum(p.list_price for p in composition)
        min_budget = target_budget * 0.95  # -5%
        max_budget = target_budget * 1.05  # +5%
        
        # If over budget, remove most expensive items
        while current_cost > max_budget and composition:
            composition.sort(key=lambda p: p.list_price, reverse=True)
            removed = composition.pop(0)
            current_cost -= removed.list_price
        
        # If under budget, try to add more items to reach minimum
        if current_cost < min_budget:
            available_products = self._get_available_products()
            remaining_products = [p for p in available_products if p not in composition]
            remaining_products.sort(key=lambda p: p.list_price)
            
            for product in remaining_products:
                if current_cost + product.list_price <= max_budget:
                    composition.append(product)
                    current_cost += product.list_price
                    
                    # Stop when we reach acceptable range
                    if current_cost >= min_budget:
                        break
        
        return composition
    
    def _create_composition_record(self, partner_id, target_budget, target_year, products, dietary_restrictions, notes_text):
        """Create the gift composition record"""
        
        total_cost = sum(p.list_price for p in products)
        variance_percent = abs(total_cost - target_budget) / target_budget * 100
        
        reasoning = f"""
        <h3>Simplified Composition Generated</h3>
        <p><strong>Products:</strong> {len(products)} items selected</p>
        <p><strong>Total Cost:</strong> €{total_cost:.2f}</p>
        <p><strong>Target Budget:</strong> €{target_budget:.2f}</p>
        <p><strong>Budget Variance:</strong> {variance_percent:.1f}%</p>
        """
        
        if dietary_restrictions:
            restrictions = dietary_restrictions if isinstance(dietary_restrictions, list) else [dietary_restrictions]
            reasoning += f"<p><strong>Dietary Restrictions:</strong> {', '.join(restrictions)}</p>"
        
        if notes_text:
            reasoning += f"<p><strong>Notes:</strong> {notes_text}</p>"
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner_id,
            'target_year': target_year,
            'target_budget': target_budget,
            'composition_type': 'custom',
            'product_ids': [(6, 0, [p.id for p in products])],
            'dietary_restrictions': ', '.join(dietary_restrictions) if dietary_restrictions else '',
            'reasoning': reasoning,
            'confidence_score': 0.85 if variance_percent <= 5 else 0.70,
            'state': 'draft'
        })
        
        return composition