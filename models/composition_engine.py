from odoo import models, fields, api
from datetime import datetime
import random
from odoo.exceptions import UserError
import json
import logging
import requests

_logger = logging.getLogger(__name__)

class CompositionEngine(models.Model):
    _name = 'composition.engine'
    _description = 'Señor Bigott AI Composition Engine'
    
    def _ollama_enabled(self):
        IrConfig = self.env['ir.config_parameter'].sudo()
        return IrConfig.get_param('lebigott_ai.ollama_enabled', 'false').lower() == 'true'

    def _ollama_base_url(self):
        IrConfig = self.env['ir.config_parameter'].sudo()
        return IrConfig.get_param('lebigott_ai.ollama_base_url', 'http://localhost:11434')

    def _ollama_model(self):
        IrConfig = self.env['ir.config_parameter'].sudo()
        return IrConfig.get_param('lebigott_ai.ollama_model', 'llama3')

    def _ollama_complete(self, prompt):
        if not self._ollama_enabled():
            return None
        try:
            url = self._ollama_base_url().rstrip('/') + '/api/generate'
            payload = {
                'model': self._ollama_model(),
                'prompt': prompt,
                'stream': False,
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.ok:
                data = resp.json()
                return data.get('response')
        except Exception as e:
            _logger.warning('Ollama request failed: %s', e)
        return None

    def generate_composition(self, partner_id, target_budget, target_year=None, dietary_restrictions=None, force_type=None, notes_text=None):
        """Main entry point for generating gift compositions"""
        
        if not target_year:
            target_year = datetime.now().year
        
        # Analyze client history
        client_analysis = self.env['client.order.history'].analyze_client_patterns(partner_id)
        
        # Determine composition approach
        if force_type:
            composition_type = force_type
        elif client_analysis['has_history']:
            composition_type = client_analysis['box_type_preference']
        else:
            # New client - prefer experience-based approach
            composition_type = 'experience'
        
        if composition_type == 'experience':
            return self._generate_experience_composition(
                partner_id, target_budget, target_year, 
                dietary_restrictions, client_analysis, notes_text
            )
        else:
            return self._generate_custom_composition(
                partner_id, target_budget, target_year,
                dietary_restrictions, client_analysis, notes_text
            )
    
    def _generate_experience_composition(self, partner_id, target_budget, target_year, 
                                       dietary_restrictions, client_analysis, notes_text=None):
        """Generate experience-based composition"""
        
        # Get available experiences for this client
        available_experiences = self._get_available_experiences(
            partner_id, target_budget, dietary_restrictions
        )
        
        if not available_experiences:
            # Fall back to custom composition
            return self._generate_custom_composition(
                partner_id, target_budget, target_year,
                dietary_restrictions, client_analysis, notes_text
            )
        
        # Select best experience based on budget and history
        selected_experience = self._select_best_experience(
            available_experiences, target_budget, client_analysis
        )
        
        # Create composition record
        composition = self.env['gift.composition'].create({
            'partner_id': partner_id,
            'target_year': target_year,
            'target_budget': target_budget,
            'composition_type': 'experience',
            'experience_id': selected_experience.id,
            'dietary_restrictions': ','.join(dietary_restrictions or []),
            'reasoning': self._generate_experience_reasoning(
                selected_experience, client_analysis, target_budget, notes_text
            ),
            'confidence_score': self._calculate_experience_confidence(
                selected_experience, target_budget, client_analysis
            ),
            'novelty_score': self._calculate_novelty_score(
                selected_experience, client_analysis
            )
        })
        
        return composition
    
    def _generate_custom_composition(self, partner_id, target_budget, target_year,
                                   dietary_restrictions, client_analysis, notes_text=None):
        """Generate custom product composition"""
        
        if not client_analysis or not client_analysis['has_history']:
            # New client - use default structure
            category_structure = self._get_default_category_structure()
            base_products = self.env['product.template']
        else:
            # Use historical structure
            category_structure = client_analysis['latest_category_structure']
            base_products = self._get_client_base_products(partner_id)
        
        # Determine budget direction
        if client_analysis and client_analysis['has_history']:
            budget_direction = self._determine_budget_direction(
                target_budget, client_analysis['recent_budgets']
            )
        else:
            budget_direction = 'same'
        
        # Build product selection
        selected_products = self._build_product_selection(
            category_structure, target_budget, budget_direction,
            dietary_restrictions, base_products
        )
        
        # Create composition record
        reasoning_html = self._generate_custom_reasoning(
            selected_products, budget_direction, client_analysis, notes_text
        )
        
        # Optionally get Ollama enhancement
        ollama_note = None
        if self._ollama_enabled():
            prompt = (
                "You are an assistant generating a short client-facing explanation (max 80 words) for a gourmet gift box. "
                f"Budget: €{target_budget:.0f}. Category mix: {json.dumps(self._analyze_product_categories(selected_products))}. "
                f"Client notes: {notes_text or 'none'}. Keep it friendly and premium."
            )
            ollama_note = self._ollama_complete(prompt)
            if ollama_note:
                reasoning_html += f"<hr/><p><strong>🤖 Ollama Suggestion:</strong> {ollama_note}</p>"
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner_id,
            'target_year': target_year,
            'target_budget': target_budget,
            'composition_type': 'custom',
            'product_ids': [(6, 0, [p.id for p in selected_products])],
            'dietary_restrictions': ','.join(dietary_restrictions or []),
            'reasoning': reasoning_html,
            'confidence_score': self._calculate_custom_confidence(
                selected_products, target_budget, client_analysis
            ),
            'novelty_score': self._calculate_custom_novelty(
                selected_products, client_analysis
            )
        })
        
        # Set category structure
        actual_categories = self._analyze_product_categories(selected_products)
        composition.set_category_breakdown(actual_categories)
        
        return composition
    
    def _get_available_experiences(self, partner_id, target_budget, dietary_restrictions):
        """Get experiences available for this client"""
        
        experiences = self.env['gift.experience'].search([('active', '=', True)])
        available = []
        
        for experience in experiences:
            # Check if client has used this experience before
            if not experience.is_available_for_client(partner_id):
                continue
            
            # Check dietary compatibility
            if dietary_restrictions and not experience.check_dietary_compatibility(dietary_restrictions):
                continue
            
            # Check budget compatibility (within 25% of target)
            budget_variance = abs(experience.base_cost - target_budget) / target_budget
            if budget_variance > 0.25:
                continue
            
            available.append(experience)
        
        return available
    
    def _select_best_experience(self, experiences, target_budget, client_analysis):
        """Select the best experience from available options"""
        
        if not experiences:
            return None
        
        # Score each experience
        scored_experiences = []
        for exp in experiences:
            score = 0
            
            # Budget fit (closer to target is better)
            budget_diff = abs(exp.base_cost - target_budget)
            budget_score = max(0, 1 - (budget_diff / target_budget))
            score += budget_score * 0.4
            
            # Experience track record (higher satisfaction = better)
            if exp.average_satisfaction > 0:
                satisfaction_score = exp.average_satisfaction / 5.0  # Convert to 0-1 scale
                score += satisfaction_score * 0.3
            else:
                score += 0.15  # Default for untested experiences
            
            # Novelty (less used experiences score higher)
            novelty_score = max(0, 1 - (exp.times_used / 20))  # Diminish after 20 uses
            score += novelty_score * 0.3
            
            scored_experiences.append((exp, score))
        
        # Return highest scoring experience
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        return scored_experiences[0][0]
    
    def _determine_budget_direction(self, target_budget, recent_budgets):
        """Determine if budget increased, decreased, or stayed same"""
        
        if not recent_budgets or len(recent_budgets) < 1:
            return 'same'
        
        last_budget = recent_budgets[0]
        change_percent = (target_budget - last_budget) / last_budget
        
        if change_percent > 0.15:  # 15% increase
            return 'upgrade'
        elif change_percent < -0.15:  # 15% decrease
            return 'downgrade'
        else:
            return 'same'
    
    def _get_default_category_structure(self):
        """Default category structure for new clients"""
        return {
            'main_beverage': 1,
            'aperitif': 1, 
            'experience_gastronomica': 1,
            'foie_gras': 1,
            'charcuterie': 2,
            'sweets': 1
        }
    
    def _build_product_selection(self, category_structure, target_budget, 
                               budget_direction, dietary_restrictions, base_products=None):
        """Build product selection based on category structure and budget"""
        
        selected_products = []
        remaining_budget = target_budget
        categories_to_fill = len(category_structure)
        
        # Process each category
        for category, quantity in category_structure.items():
            category_budget = remaining_budget / categories_to_fill
            category_products = self._select_category_products(
                category, quantity, category_budget,
                budget_direction, dietary_restrictions, base_products
            )
            
            selected_products.extend(category_products)
            remaining_budget -= sum(p.list_price for p in category_products)
            categories_to_fill -= 1
        
        return selected_products
    
    def _select_category_products(self, category, quantity, budget_per_category,
                                budget_direction, dietary_restrictions, base_products):
        """Select products for a specific category"""
        
        # Base search
        domain = [
            ('lebiggot_category', '=', category),
            ('active', '=', True),
            ('sale_ok', '=', True)
        ]
        
        # Apply dietary restrictions
        if dietary_restrictions:
            if 'vegan' in dietary_restrictions:
                domain.append(('is_vegan', '=', True))
            if 'halal' in dietary_restrictions:
                domain.append(('is_halal', '=', True))
            if 'non_alcoholic' in dietary_restrictions:
                domain.append(('contains_alcohol', '=', False))
        
        available_products = self.env['product.template'].search(domain)
        
        # Filter by budget and grade
        if budget_direction == 'upgrade':
            available_products = available_products.filtered(
                lambda p: p.product_grade in ['premium', 'luxury']
            )
        elif budget_direction == 'downgrade':
            available_products = available_products.filtered(
                lambda p: p.product_grade in ['economical', 'standard']
            )
        
        # Sort by price to fit budget
        target_price_per_item = budget_per_category / quantity if quantity > 0 else budget_per_category
        available_products = available_products.sorted(
            key=lambda p: abs(p.list_price - target_price_per_item)
        )
        
        # Select required quantity, avoiding duplicates
        selected = []
        for i in range(min(quantity, len(available_products))):
            if available_products[i] not in selected:
                selected.append(available_products[i])
        
        return selected
    
    def _generate_experience_reasoning(self, experience, client_analysis, target_budget, notes_text=None):
        """Generate reasoning for experience selection"""
        
        reasons = [
            f"<h3>🎁 Señor Bigott Selected: {experience.name}</h3>",
            f"<p><strong>Experience Theme:</strong> {experience.experience_theme.replace('_', ' ').title()}</p>",
            f"<p><strong>Products Included:</strong> {len(experience.product_ids)} carefully curated items</p>",
            f"<p><strong>Cost Analysis:</strong> €{experience.base_cost:.2f} (Target: €{target_budget:.2f})</p>"
        ]
        
        # Budget variance analysis
        variance = abs(experience.base_cost - target_budget)
        variance_pct = (variance / target_budget) * 100
        if variance_pct < 5:
            reasons.append(f"<p><strong>✅ Perfect Budget Match:</strong> Within {variance_pct:.1f}% of target</p>")
        elif experience.base_cost < target_budget:
            reasons.append(f"<p><strong>💰 Under Budget:</strong> Saves €{target_budget - experience.base_cost:.2f}</p>")
        else:
            reasons.append(f"<p><strong>⬆️ Premium Option:</strong> €{experience.base_cost - target_budget:.2f} above target for enhanced experience</p>")
        
        if client_analysis and client_analysis['has_history']:
            reasons.append(f"<p><strong>📊 Client History:</strong> Based on {client_analysis['years_of_data']} years of preferences</p>")
            reasons.append(f"<p><strong>🆕 Experience Novelty:</strong> This experience has never been sent to this client</p>")
            
            if client_analysis['average_satisfaction'] > 0:
                reasons.append(f"<p><strong>⭐ Historical Satisfaction:</strong> Average rating {client_analysis['average_satisfaction']:.1f}/5</p>")
        else:
            reasons.append("<p><strong>🎯 New Client Strategy:</strong> Selected premium experience to establish preferences</p>")
        
        # Experience performance
        if experience.times_used > 0:
            reasons.append(f"<p><strong>📈 Experience Track Record:</strong> Successfully delivered {experience.times_used} times</p>")
            if experience.average_satisfaction > 0:
                reasons.append(f"<p><strong>⭐ Average Satisfaction:</strong> {experience.average_satisfaction:.1f}/5 stars</p>")
        
        if notes_text:
            reasons.append(f"<p><strong>📝 Client Notes Considered:</strong> {notes_text}</p>")
        return "".join(reasons)
    
    def _generate_custom_reasoning(self, products, budget_direction, client_analysis, notes_text=None):
        """Generate reasoning for custom composition"""
        
        reasons = [
            "<h3>🎨 Señor Bigott Custom Composition</h3>",
            f"<p><strong>Products Selected:</strong> {len(products)} carefully chosen items</p>",
            f"<p><strong>Total Cost:</strong> €{sum(p.list_price for p in products):.2f}</p>"
        ]
        
        # Budget strategy explanation
        if budget_direction == 'upgrade':
            reasons.append("<p><strong>📈 Budget Strategy:</strong> Premium upgrades applied - enhanced wine selection, luxury items, and artisanal specialties</p>")
        elif budget_direction == 'downgrade':
            reasons.append("<p><strong>💡 Budget Strategy:</strong> Maintained structural integrity with economical choices while preserving gourmet quality</p>")
        else:
            reasons.append("<p><strong>🔄 Budget Strategy:</strong> Brand variations and curated improvements from previous selections</p>")
        
        if client_analysis and client_analysis['has_history']:
            reasons.append(f"<p><strong>📊 Historical Analysis:</strong> Category structure optimized from {client_analysis['years_of_data']} years of successful gifts</p>")
            reasons.append(f"<p><strong>🧠 Preference Learning:</strong> Product selection fine-tuned for this client's established taste profile</p>")
            
            if client_analysis['budget_trend'] == 'increasing':
                reasons.append("<p><strong>📈 Budget Trend:</strong> Client shows increasing budget confidence - premium selections applied</p>")
            elif client_analysis['budget_trend'] == 'decreasing':
                reasons.append("<p><strong>📉 Budget Optimization:</strong> Adjusted for budget consciousness while maintaining experience quality</p>")
        
        # Category breakdown
        categories = {}
        for product in products:
            cat = product.lebiggot_category
            categories[cat] = categories.get(cat, 0) + 1
        
        cat_text = ", ".join([f"{cat.replace('_', ' ').title()}: {count}" for cat, count in categories.items()])
        reasons.append(f"<p><strong>🎯 Category Balance:</strong> {cat_text}</p>")
        
        # Dietary compliance
        dietary_products = []
        for product in products:
            dietary_attrs = []
            if product.is_vegan:
                dietary_attrs.append("Vegan")
            if product.is_halal:
                dietary_attrs.append("Halal")
            if not product.contains_alcohol:
                dietary_attrs.append("Alcohol-free")
            if dietary_attrs:
                dietary_products.append(f"{product.name} ({', '.join(dietary_attrs)})")
        
        if dietary_products:
            reasons.append(f"<p><strong>🌱 Dietary Considerations:</strong> {len(dietary_products)} products with special dietary properties</p>")
        
        if notes_text:
            reasons.append(f"<p><strong>📝 Client Notes Considered:</strong> {notes_text}</p>")
        return "".join(reasons)
    
    def _calculate_experience_confidence(self, experience, target_budget, client_analysis):
        """Calculate confidence score for experience selection"""
        
        confidence = 0.5  # Base confidence
        
        # Budget fit (closer = higher confidence)
        budget_variance = abs(experience.base_cost - target_budget) / target_budget
        budget_confidence = max(0, 1 - budget_variance)
        confidence += budget_confidence * 0.3
        
        # Historical data quality
        if client_analysis and client_analysis['has_history']:
            history_confidence = min(0.2, client_analysis['years_of_data'] * 0.07)
            confidence += history_confidence
        
        # Experience track record
        if experience.times_used > 0:
            # Performance bonus based on usage and satisfaction
            usage_confidence = min(0.2, experience.times_used * 0.01)
            if experience.average_satisfaction > 0:
                satisfaction_bonus = (experience.average_satisfaction / 5.0) * 0.1
                confidence += usage_confidence + satisfaction_bonus
            else:
                confidence += usage_confidence
        
        return min(1.0, confidence)
    
    def _calculate_custom_confidence(self, products, target_budget, client_analysis):
        """Calculate confidence score for custom composition"""
        
        confidence = 0.5  # Base confidence
        
        # Budget fit
        actual_cost = sum(p.list_price for p in products)
        budget_variance = abs(actual_cost - target_budget) / target_budget
        budget_confidence = max(0, 1 - budget_variance)
        confidence += budget_confidence * 0.3
        
        # Historical data quality
        if client_analysis and client_analysis['has_history']:
            history_confidence = min(0.2, client_analysis['years_of_data'] * 0.07)
            confidence += history_confidence
        
        # Product selection completeness
        if len(products) >= 6:  # Full composition
            confidence += 0.2
        elif len(products) >= 4:  # Adequate composition
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_novelty_score(self, experience, client_analysis):
        """Calculate novelty score for experience"""
        
        if not client_analysis or not client_analysis['has_history']:
            return 1.0  # Everything is novel for new clients
        
        # Base novelty from not being used before
        base_novelty = 0.8
        
        # Theme variety bonus
        if experience.experience_theme not in ['mediterranean', 'spanish_classics']:  # Assuming these are common
            base_novelty += 0.2
        
        return min(1.0, base_novelty)
    
    def _calculate_custom_novelty(self, products, client_analysis):
        """Calculate novelty score for custom composition"""
        
        if not client_analysis or not client_analysis['has_history']:
            return 0.7  # Moderate novelty for new clients
        
        # Calculate how many products are new vs. repeated
        # This would require historical product tracking
        return 0.6  # Default moderate novelty
    
    def _analyze_product_categories(self, products):
        """Analyze category breakdown of selected products"""
        categories = {}
        for product in products:
            cat = product.lebiggot_category
            categories[cat] = categories.get(cat, 0) + 1
        return categories
    
    def _get_client_base_products(self, partner_id):
        """Get products from client's previous orders for reference"""
        histories = self.env['client.order.history'].search([
            ('partner_id', '=', partner_id)
        ], limit=1, order='order_year desc')
        
        if histories:
            return histories[0].product_ids
        return self.env['product.template']