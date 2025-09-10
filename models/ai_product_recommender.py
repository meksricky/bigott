from odoo import models, fields, api
from odoo.exceptions import UserError
import json
import logging
import random
from collections import defaultdict
import statistics

_logger = logging.getLogger(__name__)

class AIProductRecommender(models.Model):
    _name = 'ai.product.recommender'
    _description = 'AI-Powered Product Recommendation Engine'
    
    def recommend_products_for_budget(self, partner_id, target_budget, dietary_restrictions=None, 
                                    notes_text=None, attempt_number=1):
        """
        Smart AI recommendation system that learns from sales data
        Returns products within ±5% budget (extendable to ±15%)
        """
        
        # Get validated product pool
        available_products = self._get_validated_product_pool(dietary_restrictions)
        if not available_products:
            raise UserError("No available products meet your criteria")
        
        # Get AI-powered client insights
        client_insights = self._get_ai_client_insights(partner_id, notes_text)
        
        # Get budget flexibility parameters
        budget_params = self._calculate_budget_flexibility(target_budget, client_insights, attempt_number)
        
        # Use AI to score and select products
        scored_products = self._ai_score_products(
            available_products, client_insights, target_budget, notes_text
        )
        
        # Smart budget-aware selection
        selected_products = self._smart_budget_selection(
            scored_products, budget_params, client_insights
        )
        
        # Learn from this recommendation for future use
        self._record_recommendation_for_learning(partner_id, selected_products, client_insights)
        
        return {
            'products': selected_products,
            'actual_cost': sum(p.list_price for p in selected_products),
            'ai_reasoning': self._generate_ai_reasoning(selected_products, client_insights, target_budget),
            'confidence': self._calculate_ai_confidence(selected_products, client_insights),
            'learning_applied': True
        }
    
    def _get_validated_product_pool(self, dietary_restrictions=None):
        """Get products that pass all validation checks"""
        
        # Base domain for valid products
        domain = [
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('default_code', '!=', False),  # Must have internal reference
            ('list_price', '>', 0),
            ('lebiggot_category', '!=', False),  # Must have category
        ]
        
        products = self.env['product.template'].search(domain)
        
        # Stock validation
        validated_products = []
        for product in products:
            if self._validate_product_availability(product):
                validated_products.append(product)
        
        # Dietary restrictions filtering
        if dietary_restrictions:
            validated_products = self._apply_dietary_filters(validated_products, dietary_restrictions)
        
        return validated_products
    
    def _validate_product_availability(self, product):
        """Validate product stock and business rules"""
        
        # Check stock for stockable products
        if product.type == 'product':
            stock_quants = self.env['stock.quant'].search([
                ('product_id', 'in', product.product_variant_ids.ids),
                ('location_id.usage', '=', 'internal')
            ])
            available_qty = sum(quant.available_quantity for quant in stock_quants)
            if available_qty <= 0:
                return False
        
        # Internal reference validation
        if not product.default_code or not product.default_code.strip():
            return False
        
        # Category validation
        if not product.lebiggot_category:
            return False
        
        return True
    
    def _get_ai_client_insights(self, partner_id, notes_text=None):
        """Get comprehensive AI-powered client analysis"""
        
        # Base analysis from existing system
        base_analysis = self.env['client.order.history'].analyze_client_patterns(partner_id)
        
        # Enhanced AI analysis using Ollama if available
        ai_insights = {}
        if self.env['composition.engine']._ollama_enabled() and notes_text:
            ai_insights = self._get_ollama_client_insights(partner_id, notes_text, base_analysis)
        
        # Merge insights
        return {
            'historical': base_analysis,
            'ai_enhanced': ai_insights,
            'partner_id': partner_id,
            'notes': notes_text
        }
    
    def _get_ollama_client_insights(self, partner_id, notes_text, base_analysis):
        """Use Ollama AI to analyze client preferences"""
        
        partner = self.env['res.partner'].browse(partner_id)
        
        # Build comprehensive prompt
        prompt = f"""
        Analyze this client for luxury gourmet gift recommendations:
        
        CLIENT: {partner.name}
        NOTES: {notes_text}
        HISTORY: {base_analysis.get('years_of_data', 0)} years, avg budget €{base_analysis.get('average_budget', 0):.0f}
        TREND: {base_analysis.get('budget_trend', 'unknown')}
        SATISFACTION: {base_analysis.get('average_satisfaction', 0)}/5
        
        Extract and return JSON with:
        {{
            "style_preference": "traditional|modern|premium|experimental",
            "price_sensitivity": "low|medium|high",
            "novelty_seeking": true/false,
            "category_priorities": ["category1", "category2", ...],
            "flavor_profile": "bold|mild|varied|premium",
            "gift_occasion": "business|personal|celebration|thank_you",
            "risk_tolerance": "conservative|moderate|adventurous"
        }}
        
        Only return valid JSON.
        """
        
        try:
            ollama_response = self.env['composition.engine']._ollama_complete(prompt)
            if ollama_response:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', ollama_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            _logger.warning(f"Ollama analysis failed: {e}")
        
        # Fallback to rule-based analysis
        return self._fallback_client_analysis(notes_text, base_analysis)
    
    def _fallback_client_analysis(self, notes_text, base_analysis):
        """Rule-based client analysis when AI is unavailable"""
        
        notes_lower = (notes_text or '').lower()
        
        # Style preference
        if any(word in notes_lower for word in ['traditional', 'classic', 'authentic']):
            style = 'traditional'
        elif any(word in notes_lower for word in ['modern', 'contemporary', 'innovative']):
            style = 'modern'
        elif any(word in notes_lower for word in ['premium', 'luxury', 'exclusive']):
            style = 'premium'
        else:
            style = 'traditional'  # Default
        
        return {
            'style_preference': style,
            'price_sensitivity': 'medium',
            'novelty_seeking': 'variety' in notes_lower or 'different' in notes_lower,
            'category_priorities': list(base_analysis.get('latest_category_structure', {}).keys()),
            'flavor_profile': 'varied',
            'gift_occasion': 'business',
            'risk_tolerance': 'moderate'
        }
    
    def _ai_score_products(self, products, client_insights, target_budget, notes_text):
        """AI-powered product scoring based on client insights"""
        
        scored_products = []
        historical = client_insights.get('historical', {})
        ai_enhanced = client_insights.get('ai_enhanced', {})
        
        for product in products:
            score = 0.0
            
            # Base compatibility score
            score += self._score_historical_compatibility(product, historical)
            
            # AI-enhanced scoring
            if ai_enhanced:
                score += self._score_ai_compatibility(product, ai_enhanced)
            
            # Budget fit score
            score += self._score_budget_fit(product, target_budget)
            
            # Notes relevance score
            if notes_text:
                score += self._score_notes_relevance(product, notes_text)
            
            # Learning from sales data score
            score += self._score_market_learning(product, client_insights)
            
            scored_products.append({
                'product': product,
                'score': min(1.0, score),
                'reasons': self._get_scoring_reasons(product, score)
            })
        
        return sorted(scored_products, key=lambda x: x['score'], reverse=True)
    
    def _score_historical_compatibility(self, product, historical):
        """Score based on client's historical preferences"""
        
        if not historical.get('has_history'):
            return 0.3  # Neutral score for new clients
        
        score = 0.0
        
        # Category preference
        category_prefs = historical.get('latest_category_structure', {})
        if product.lebiggot_category in category_prefs:
            score += 0.3
        
        # Budget range compatibility
        avg_budget = historical.get('average_budget', 0)
        if avg_budget > 0:
            budget_per_product = avg_budget / historical.get('average_products', 5)
            price_ratio = product.list_price / budget_per_product
            if 0.8 <= price_ratio <= 1.2:  # Within 20% of expected price
                score += 0.2
        
        # Satisfaction correlation
        if historical.get('average_satisfaction', 0) >= 4:
            score += 0.1  # Boost for historically satisfied clients
        
        return score
    
    def _score_ai_compatibility(self, product, ai_insights):
        """Score based on AI analysis of client preferences"""
        
        score = 0.0
        
        # Style preference matching
        style_pref = ai_insights.get('style_preference', 'traditional')
        if style_pref == 'premium' and hasattr(product, 'product_grade') and product.product_grade in ['premium', 'luxury']:
            score += 0.3
        elif style_pref == 'traditional' and any(word in product.name.lower() for word in ['tradicional', 'artisan', 'classic']):
            score += 0.2
        
        # Category priority matching
        category_priorities = ai_insights.get('category_priorities', [])
        if product.lebiggot_category in category_priorities[:3]:  # Top 3 priorities
            position = category_priorities.index(product.lebiggot_category)
            score += 0.3 - (position * 0.1)  # Higher score for higher priority
        
        # Risk tolerance matching
        risk_tolerance = ai_insights.get('risk_tolerance', 'moderate')
        if risk_tolerance == 'adventurous':
            score += 0.1  # Boost for unique products
        
        return score
    
    def _score_budget_fit(self, product, target_budget):
        """Score how well product fits target budget per item"""
        
        # Estimate products per composition (5-7 typical)
        estimated_products = 6
        budget_per_product = target_budget / estimated_products
        
        price_ratio = product.list_price / budget_per_product
        
        if 0.5 <= price_ratio <= 1.5:  # Good fit
            return 0.3 - abs(1.0 - price_ratio) * 0.2
        elif price_ratio < 0.5:  # Too cheap
            return 0.1
        else:  # Too expensive
            return 0.05
    
    def _calculate_budget_flexibility(self, target_budget, client_insights, attempt_number):
        """Calculate budget flexibility based on client and attempt"""
        
        # Base flexibility: ±5%
        base_flexibility = 0.05
        
        # Extend to ±15% if needed
        max_flexibility = 0.15
        
        # Increase flexibility with attempts
        flexibility = min(base_flexibility + (attempt_number - 1) * 0.03, max_flexibility)
        
        # Adjust based on client profile
        historical = client_insights.get('historical', {})
        if historical.get('budget_trend') == 'increasing':
            flexibility += 0.02  # More flexible for growing budgets
        
        ai_enhanced = client_insights.get('ai_enhanced', {})
        if ai_enhanced.get('price_sensitivity') == 'low':
            flexibility += 0.03  # More flexible for price-insensitive clients
        
        return {
            'target': target_budget,
            'min_budget': target_budget * (1 - flexibility),
            'max_budget': target_budget * (1 + flexibility),
            'flexibility_percent': flexibility * 100
        }
    
    def _smart_budget_selection(self, scored_products, budget_params, client_insights):
        """Smart selection algorithm that optimizes for budget and quality"""
        
        target_budget = budget_params['target']
        min_budget = budget_params['min_budget']
        max_budget = budget_params['max_budget']
        
        # Multiple selection strategies
        strategies = [
            self._greedy_selection,
            self._balanced_selection,
            self._premium_selection,
        ]
        
        best_selection = None
        best_score = 0
        
        for strategy in strategies:
            try:
                selection = strategy(scored_products, budget_params, client_insights)
                if selection:
                    total_cost = sum(p.list_price for p in selection)
                    
                    # Check budget compliance
                    if min_budget <= total_cost <= max_budget:
                        # Calculate selection quality score
                        quality_score = self._evaluate_selection_quality(selection, client_insights)
                        if quality_score > best_score:
                            best_selection = selection
                            best_score = quality_score
            except Exception as e:
                _logger.warning(f"Selection strategy failed: {e}")
                continue
        
        if not best_selection:
            # Fallback: simple budget-constrained selection
            best_selection = self._fallback_selection(scored_products, budget_params)
        
        return best_selection
    
    def _greedy_selection(self, scored_products, budget_params, client_insights):
        """Greedy selection: best score per euro"""
        
        # Calculate value score (score per euro)
        value_products = []
        for item in scored_products:
            value_score = item['score'] / max(item['product'].list_price, 1)
            value_products.append({
                'product': item['product'],
                'score': item['score'],
                'value_score': value_score
            })
        
        value_products.sort(key=lambda x: x['value_score'], reverse=True)
        
        selected = []
        current_cost = 0
        categories_used = set()
        
        for item in value_products:
            product = item['product']
            
            # Avoid category repetition (max 2 per category)
            category = product.lebiggot_category
            category_count = sum(1 for p in selected if p.lebiggot_category == category)
            if category_count >= 2:
                continue
            
            if current_cost + product.list_price <= budget_params['max_budget']:
                selected.append(product)
                current_cost += product.list_price
                categories_used.add(category)
                
                # Target 5-7 products
                if len(selected) >= 7:
                    break
        
        return selected if len(selected) >= 3 else None
    
    def _balanced_selection(self, scored_products, budget_params, client_insights):
        """Balanced selection across categories"""
        
        # Group by category
        by_category = defaultdict(list)
        for item in scored_products:
            by_category[item['product'].lebiggot_category].append(item)
        
        # Sort each category by score
        for category in by_category:
            by_category[category].sort(key=lambda x: x['score'], reverse=True)
        
        selected = []
        current_cost = 0
        category_budget = budget_params['target'] / len(by_category)
        
        # Select best from each category
        for category, items in by_category.items():
            category_spent = 0
            for item in items:
                product = item['product']
                if (current_cost + product.list_price <= budget_params['max_budget'] and
                    category_spent + product.list_price <= category_budget * 1.5):  # Allow 50% over category budget
                    selected.append(product)
                    current_cost += product.list_price
                    category_spent += product.list_price
                    break  # One per category first
        
        # Fill remaining budget with highest scoring products
        remaining_products = [item for cat_items in by_category.values() 
                            for item in cat_items[1:]]  # Skip first (already selected)
        remaining_products.sort(key=lambda x: x['score'], reverse=True)
        
        for item in remaining_products:
            product = item['product']
            if current_cost + product.list_price <= budget_params['max_budget']:
                selected.append(product)
                current_cost += product.list_price
                if len(selected) >= 7:
                    break
        
        return selected if len(selected) >= 3 else None
    
    def _record_recommendation_for_learning(self, partner_id, selected_products, client_insights):
        """Record this recommendation for future learning"""
        
        learning_data = {
            'partner_id': partner_id,
            'timestamp': fields.Datetime.now(),
            'products': [p.id for p in selected_products],
            'total_cost': sum(p.list_price for p in selected_products),
            'client_insights': client_insights,
            'categories': [p.lebiggot_category for p in selected_products]
        }
        
        # Store in a simple learning cache (could be database or file)
        # This builds the learning dataset for future improvements
        cache_key = f"recommendation_learning_{partner_id}_{fields.Date.today()}"
        self.env['ir.config_parameter'].sudo().set_param(cache_key, json.dumps(learning_data))
    
    def _generate_ai_reasoning(self, selected_products, client_insights, target_budget):
        """Generate AI-powered reasoning for the selection"""
        
        historical = client_insights.get('historical', {})
        ai_enhanced = client_insights.get('ai_enhanced', {})
        
        reasoning_parts = [
            f"<h4>🤖 AI-Powered Recommendation</h4>",
            f"<p><strong>Target Budget:</strong> €{target_budget:.2f}</p>",
            f"<p><strong>Selected Products:</strong> {len(selected_products)}</p>",
            f"<p><strong>Total Cost:</strong> €{sum(p.list_price for p in selected_products):.2f}</p>",
        ]
        
        if historical.get('has_history'):
            reasoning_parts.extend([
                f"<h5>📊 Learning from History</h5>",
                f"<ul>",
                f"<li>Client data: {historical.get('years_of_data', 0)} years of preferences</li>",
                f"<li>Budget trend: {historical.get('budget_trend', 'stable').title()}</li>",
                f"<li>Satisfaction: {historical.get('average_satisfaction', 0):.1f}/5 stars</li>",
                f"</ul>"
            ])
        
        if ai_enhanced:
            reasoning_parts.extend([
                f"<h5>🧠 AI Analysis Applied</h5>",
                f"<ul>",
                f"<li>Style preference: {ai_enhanced.get('style_preference', 'N/A').title()}</li>",
                f"<li>Risk tolerance: {ai_enhanced.get('risk_tolerance', 'N/A').title()}</li>",
                f"<li>Price sensitivity: {ai_enhanced.get('price_sensitivity', 'N/A').title()}</li>",
                f"</ul>"
            ])
        
        # Product breakdown
        categories = defaultdict(list)
        for product in selected_products:
            categories[product.lebiggot_category].append(product)
        
        reasoning_parts.append("<h5>🎁 Selected Products by Category</h5>")
        for category, products in categories.items():
            reasoning_parts.append(f"<p><strong>{category.replace('_', ' ').title()}:</strong> {', '.join([p.name for p in products])}</p>")
        
        return ''.join(reasoning_parts)