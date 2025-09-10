from odoo import models, fields, api
from odoo.exceptions import UserError
import json
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import random

_logger = logging.getLogger(__name__)

class AIProductRecommender(models.Model):
    _name = 'ai.product.recommender'
    _description = 'AI-Powered Product Recommendation System'
    
    name = fields.Char(string="Recommender Name", default="AI Product Recommender")
    
    # Configuration
    min_confidence_threshold = fields.Float(string="Min Confidence Threshold", default=0.6)
    max_recommendation_attempts = fields.Integer(string="Max Attempts", default=3)
    
    # Learning Settings
    learning_enabled = fields.Boolean(string="Learning Enabled", default=True)
    feedback_weight = fields.Float(string="Feedback Weight", default=0.2)
    
    @api.model
    def get_ai_recommendations(self, partner_id, target_budget, dietary_restrictions=None, 
                              notes_text=None, use_ml=True):
        """
        Main entry point for AI recommendations
        Integrates with ML engine when available
        """
        
        _logger.info(f"Starting AI recommendations for partner {partner_id}, budget €{target_budget}")
        
        # Check if ML engine is available and trained
        ml_engine = self.env['ml.recommendation.engine'].get_or_create_engine()
        
        if use_ml and ml_engine.is_model_trained:
            # Use full ML recommendations
            return ml_engine.get_smart_recommendations(
                partner_id=partner_id,
                target_budget=target_budget,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text,
                max_attempts=self.max_recommendation_attempts
            )
        else:
            # Use AI-enhanced recommendations without ML
            return self._ai_only_recommendations(
                partner_id=partner_id,
                target_budget=target_budget,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text
            )
    
    def _ai_only_recommendations(self, partner_id, target_budget, dietary_restrictions=None, notes_text=None):
        """AI recommendations without ML model"""
        
        # Get client insights
        client_insights = self._analyze_client_comprehensive(partner_id, notes_text)
        
        # Get available products
        available_products = self._get_filtered_products(dietary_restrictions)
        
        if not available_products:
            raise UserError("No products available matching the criteria")
        
        # AI score products
        scored_products = self._ai_score_products(
            available_products, 
            client_insights, 
            target_budget, 
            notes_text
        )
        
        # Select optimal products
        selected_products = self._ai_select_products(
            scored_products, 
            target_budget, 
            client_insights
        )
        
        if not selected_products:
            raise UserError("Could not find suitable product combination")
        
        # Generate result
        actual_cost = sum(p.list_price for p in selected_products)
        variance = abs(actual_cost - target_budget) / target_budget * 100
        
        return {
            'products': selected_products,
            'actual_cost': actual_cost,
            'budget_variance': variance,
            'ai_confidence': self._calculate_confidence(selected_products, client_insights),
            'reasoning': self._generate_ai_reasoning(selected_products, client_insights, target_budget),
            'method': 'AI-Enhanced',
            'client_insights': client_insights
        }
    
    def _analyze_client_comprehensive(self, partner_id, notes_text=None):
        """Comprehensive client analysis with AI enhancement"""
        
        # Get historical data
        history_analysis = self.env['client.order.history'].analyze_client_patterns(partner_id)
        
        # AI text analysis
        ai_enhanced = {}
        if notes_text and self._ollama_enabled():
            ai_enhanced = self._ai_analyze_notes(notes_text, history_analysis)
        
        # Market position
        market_position = self._analyze_market_position(partner_id)
        
        # Learning from past recommendations
        past_recommendations = self._get_past_recommendations(partner_id)
        
        return {
            'partner_id': partner_id,
            'historical': history_analysis,
            'ai_enhanced': ai_enhanced,
            'market_position': market_position,
            'past_recommendations': past_recommendations,
            'confidence': self._calculate_analysis_confidence(history_analysis, ai_enhanced)
        }
    
    def _ai_analyze_notes(self, notes_text, base_analysis):
        """AI-powered notes analysis using Ollama"""
        
        if not self._ollama_enabled():
            return self._fallback_notes_analysis(notes_text)
        
        try:
            prompt = f"""
            Analyze these luxury gift notes for a client with this history:
            - Years of data: {base_analysis.get('years_of_data', 0)}
            - Budget trend: {base_analysis.get('budget_trend', 'stable')}
            - Average satisfaction: {base_analysis.get('average_satisfaction', 0)}/5
            
            Notes: "{notes_text}"
            
            Extract JSON with these fields:
            {{
                "style_preference": "traditional|modern|premium|eclectic",
                "price_sensitivity": "low|medium|high",
                "novelty_seeking": true/false,
                "sophistication_level": 1-10,
                "flavor_profile": "bold|mild|varied|premium",
                "occasion_type": "business|personal|celebration",
                "quality_focus": "artisanal|branded|exclusive|value",
                "gift_purpose": "appreciation|celebration|relationship|incentive",
                "personalization_need": "high|medium|low",
                "cultural_preferences": ["spanish", "french", "italian", "international"]
            }}
            
            Return only valid JSON.
            """
            
            response = self._ollama_complete(prompt)
            if response:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        
        except Exception as e:
            _logger.warning(f"AI notes analysis failed: {e}")
        
        return self._fallback_notes_analysis(notes_text)
    
    def _fallback_notes_analysis(self, notes_text):
        """Rule-based fallback when AI unavailable"""
        
        notes_lower = (notes_text or '').lower()
        
        return {
            'style_preference': self._detect_style(notes_lower),
            'price_sensitivity': self._detect_price_sensitivity(notes_lower),
            'novelty_seeking': any(word in notes_lower for word in ['variety', 'different', 'new', 'unique']),
            'sophistication_level': 7,
            'flavor_profile': 'varied',
            'occasion_type': self._detect_occasion(notes_lower),
            'quality_focus': 'artisanal',
            'gift_purpose': 'appreciation',
            'personalization_need': 'medium',
            'cultural_preferences': ['spanish']
        }
    
    def _detect_style(self, notes_lower):
        """Detect style preference from notes"""
        if any(word in notes_lower for word in ['traditional', 'classic', 'authentic']):
            return 'traditional'
        elif any(word in notes_lower for word in ['modern', 'contemporary', 'innovative']):
            return 'modern'
        elif any(word in notes_lower for word in ['premium', 'luxury', 'exclusive']):
            return 'premium'
        elif any(word in notes_lower for word in ['varied', 'diverse', 'eclectic']):
            return 'eclectic'
        return 'traditional'
    
    def _detect_price_sensitivity(self, notes_lower):
        """Detect price sensitivity from notes"""
        if any(word in notes_lower for word in ['budget', 'affordable', 'cost', 'value']):
            return 'high'
        elif any(word in notes_lower for word in ['premium', 'luxury', 'best', 'finest']):
            return 'low'
        return 'medium'
    
    def _detect_occasion(self, notes_lower):
        """Detect occasion type from notes"""
        if any(word in notes_lower for word in ['business', 'corporate', 'client', 'partner']):
            return 'business'
        elif any(word in notes_lower for word in ['birthday', 'anniversary', 'celebration']):
            return 'celebration'
        return 'personal'
    
    def _get_filtered_products(self, dietary_restrictions=None):
        """Get products filtered by availability and dietary restrictions"""
        
        domain = [
            ('active', '=', True),
            ('sale_ok', '=', True),
            ('list_price', '>', 0),
            ('lebiggot_category', '!=', False),
        ]
        
        products = self.env['product.template'].search(domain)
        
        # Filter by dietary restrictions
        if dietary_restrictions:
            products = products.filtered(
                lambda p: self._check_dietary_compliance(p, dietary_restrictions)
            )
        
        # Filter by stock availability
        products = products.filtered(lambda p: self._check_stock_availability(p))
        
        return products
    
    def _check_dietary_compliance(self, product, restrictions):
        """Check if product meets dietary restrictions"""
        
        for restriction in restrictions:
            if restriction == 'vegan':
                if hasattr(product, 'is_vegan') and not product.is_vegan:
                    return False
                if any(word in product.name.lower() for word in ['meat', 'fish', 'dairy', 'cheese']):
                    return False
            
            elif restriction == 'halal':
                if hasattr(product, 'is_halal') and not product.is_halal:
                    return False
                if any(word in product.name.lower() for word in ['pork', 'wine', 'alcohol']):
                    return False
            
            elif restriction == 'non_alcoholic':
                if any(word in product.name.lower() for word in ['wine', 'champagne', 'alcohol']):
                    return False
        
        return True
    
    def _check_stock_availability(self, product):
        """Check if product has available stock"""
        
        if product.type != 'product':
            return True
        
        stock_quants = self.env['stock.quant'].search([
            ('product_id', 'in', product.product_variant_ids.ids),
            ('location_id.usage', '=', 'internal')
        ])
        
        available_qty = sum(quant.available_quantity for quant in stock_quants)
        return available_qty > 0
    
    def _ai_score_products(self, products, client_insights, target_budget, notes_text):
        """AI-powered product scoring"""
        
        scored_products = []
        
        for product in products:
            score = 0.0
            
            # Historical compatibility (30%)
            if client_insights['historical'].get('has_history'):
                score += self._score_historical_match(product, client_insights['historical']) * 0.30
            else:
                score += 0.15  # Default for new clients
            
            # AI analysis compatibility (25%)
            if client_insights['ai_enhanced']:
                score += self._score_ai_match(product, client_insights['ai_enhanced']) * 0.25
            
            # Budget fitness (20%)
            score += self._score_budget_fitness(product, target_budget) * 0.20
            
            # Quality indicators (15%)
            score += self._score_quality(product) * 0.15
            
            # Market trends (10%)
            score += self._score_market_trends(product) * 0.10
            
            scored_products.append({
                'product': product,
                'score': min(1.0, max(0.0, score)),
                'scoring_breakdown': {
                    'historical': self._score_historical_match(product, client_insights['historical']),
                    'ai_match': self._score_ai_match(product, client_insights['ai_enhanced']) if client_insights['ai_enhanced'] else 0,
                    'budget': self._score_budget_fitness(product, target_budget),
                    'quality': self._score_quality(product),
                    'trends': self._score_market_trends(product)
                }
            })
        
        return sorted(scored_products, key=lambda x: x['score'], reverse=True)
    
    def _score_historical_match(self, product, historical):
        """Score based on historical preferences"""
        
        score = 0.5  # Base score
        
        # Category match
        if historical.get('category_preferences'):
            if product.lebiggot_category in historical['category_preferences']:
                score += 0.3
        
        # Price range match
        avg_price = historical.get('average_product_price', 0)
        if avg_price > 0:
            price_diff = abs(product.list_price - avg_price) / avg_price
            if price_diff < 0.2:
                score += 0.2
        
        return min(1.0, score)
    
    def _score_ai_match(self, product, ai_enhanced):
        """Score based on AI analysis"""
        
        score = 0.5
        
        # Style preference match
        style = ai_enhanced.get('style_preference', 'traditional')
        if style == 'premium' and hasattr(product, 'product_grade'):
            if product.product_grade in ['premium', 'luxury']:
                score += 0.3
        
        # Quality focus match
        quality_focus = ai_enhanced.get('quality_focus', 'artisanal')
        if quality_focus == 'artisanal' and 'artisan' in product.name.lower():
            score += 0.2
        elif quality_focus == 'exclusive' and 'exclusive' in product.name.lower():
            score += 0.2
        
        return min(1.0, score)
    
    def _score_budget_fitness(self, product, target_budget):
        """Score based on budget fit"""
        
        # Ideal product price is 10-20% of total budget
        ideal_price = target_budget * 0.15
        price_diff = abs(product.list_price - ideal_price) / ideal_price
        
        if price_diff < 0.2:
            return 1.0
        elif price_diff < 0.5:
            return 0.7
        elif price_diff < 1.0:
            return 0.4
        else:
            return 0.2
    
    def _score_quality(self, product):
        """Score based on quality indicators"""
        
        score = 0.5
        
        # Premium grade
        if hasattr(product, 'product_grade'):
            if product.product_grade == 'luxury':
                score += 0.5
            elif product.product_grade == 'premium':
                score += 0.3
        
        # Price as quality proxy
        if product.list_price > 50:
            score += 0.2
        
        return min(1.0, score)
    
    def _score_market_trends(self, product):
        """Score based on market trends and popularity"""
        
        # Check recent sales
        recent_sales = self.env['sale.order.line'].search_count([
            ('product_id.product_tmpl_id', '=', product.id),
            ('create_date', '>=', fields.Date.today() - timedelta(days=90))
        ])
        
        if recent_sales > 10:
            return 1.0
        elif recent_sales > 5:
            return 0.7
        elif recent_sales > 0:
            return 0.5
        else:
            return 0.3
    
    def _ai_select_products(self, scored_products, target_budget, client_insights):
        """AI-powered product selection with budget optimization"""
        
        selected_products = []
        current_cost = 0
        min_budget = target_budget * 0.85
        max_budget = target_budget * 1.15
        
        # Category diversity tracking
        categories_used = defaultdict(int)
        max_per_category = 2
        
        # Selection strategy based on client insights
        if client_insights['ai_enhanced'].get('style_preference') == 'premium':
            # Premium strategy: fewer, higher-value items
            target_count = 4
        else:
            # Standard strategy: balanced selection
            target_count = 6
        
        for item in scored_products:
            product = item['product']
            category = product.lebiggot_category
            
            # Skip if category limit reached
            if categories_used[category] >= max_per_category:
                continue
            
            # Check if adding product keeps within budget
            if current_cost + product.list_price <= max_budget:
                selected_products.append(product)
                current_cost += product.list_price
                categories_used[category] += 1
                
                # Stop if target reached
                if len(selected_products) >= target_count:
                    if current_cost >= min_budget:
                        break
        
        # Validate selection
        if current_cost < min_budget or not selected_products:
            # Try alternative selection strategy
            return self._fallback_selection(scored_products, target_budget)
        
        return selected_products
    
    def _fallback_selection(self, scored_products, target_budget):
        """Fallback selection when primary strategy fails"""
        
        selected = []
        current_cost = 0
        max_budget = target_budget * 1.2
        
        # Simple greedy selection
        for item in scored_products[:10]:  # Top 10 products
            product = item['product']
            if current_cost + product.list_price <= max_budget:
                selected.append(product)
                current_cost += product.list_price
                
                if current_cost >= target_budget * 0.8:
                    break
        
        return selected if selected else []
    
    def _calculate_confidence(self, selected_products, client_insights):
        """Calculate confidence score for recommendation"""
        
        base_confidence = 0.5
        
        # Historical data bonus
        if client_insights['historical'].get('has_history'):
            base_confidence += 0.2
        
        # AI analysis bonus
        if client_insights['ai_enhanced']:
            base_confidence += 0.15
        
        # Product diversity bonus
        categories = set(p.lebiggot_category for p in selected_products)
        if len(categories) >= 3:
            base_confidence += 0.1
        
        # Past recommendation success bonus
        if client_insights['past_recommendations']:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _generate_ai_reasoning(self, selected_products, client_insights, target_budget):
        """Generate comprehensive AI reasoning"""
        
        actual_cost = sum(p.list_price for p in selected_products)
        variance = abs(actual_cost - target_budget) / target_budget * 100
        
        reasoning_parts = [
            f"<div class='ai-reasoning'>",
            f"<h4>🤖 AI-Powered Recommendation Analysis</h4>",
            
            f"<div class='recommendation-summary'>",
            f"<p><strong>Products Selected:</strong> {len(selected_products)}</p>",
            f"<p><strong>Total Cost:</strong> €{actual_cost:.2f}</p>",
            f"<p><strong>Budget Variance:</strong> {variance:.1f}%</p>",
            f"<p><strong>Confidence Score:</strong> {self._calculate_confidence(selected_products, client_insights):.1%}</p>",
            f"</div>"
        ]
        
        # Client insights
        if client_insights['historical'].get('has_history'):
            historical = client_insights['historical']
            reasoning_parts.extend([
                f"<h5>📊 Historical Analysis</h5>",
                f"<ul>",
                f"<li>Experience: {historical.get('years_of_data', 0)} years of data</li>",
                f"<li>Budget trend: {historical.get('budget_trend', 'stable').title()}</li>",
                f"<li>Satisfaction: {historical.get('average_satisfaction', 0):.1f}/5</li>",
                f"</ul>"
            ])
        
        # AI insights
        if client_insights['ai_enhanced']:
            ai = client_insights['ai_enhanced']
            reasoning_parts.extend([
                f"<h5>🧠 AI Analysis</h5>",
                f"<ul>",
                f"<li>Style: {ai.get('style_preference', 'N/A').title()}</li>",
                f"<li>Quality Focus: {ai.get('quality_focus', 'N/A').title()}</li>",
                f"<li>Occasion: {ai.get('occasion_type', 'N/A').title()}</li>",
                f"</ul>"
            ])
        
        # Product categories
        categories = defaultdict(list)
        for product in selected_products:
            categories[product.lebiggot_category].append(product.name)
        
        reasoning_parts.append(f"<h5>🎁 Selected Products</h5>")
        for category, products in categories.items():
            category_name = category.replace('_', ' ').title()
            reasoning_parts.append(f"<p><strong>{category_name}:</strong> {', '.join(products)}</p>")
        
        reasoning_parts.append(f"</div>")
        
        return ''.join(reasoning_parts)
    
    def _analyze_market_position(self, partner_id):
        """Analyze client's market position"""
        
        # Calculate percentile based on order history
        all_clients = self.env['client.order.history'].search([])
        client_budgets = [c.total_budget for c in all_clients if c.total_budget > 0]
        
        if not client_budgets:
            return {'percentile': 50, 'segment': 'standard'}
        
        client_history = self.env['client.order.history'].search([
            ('partner_id', '=', partner_id)
        ], limit=1)
        
        if not client_history or not client_history.total_budget:
            return {'percentile': 50, 'segment': 'standard'}
        
        client_budget = client_history.total_budget
        percentile = (sum(1 for b in client_budgets if b <= client_budget) / len(client_budgets)) * 100
        
        if percentile >= 80:
            segment = 'premium'
        elif percentile >= 50:
            segment = 'standard'
        else:
            segment = 'value'
        
        return {
            'percentile': percentile,
            'segment': segment,
            'average_market_budget': np.mean(client_budgets)
        }
    
    def _get_past_recommendations(self, partner_id):
        """Get past recommendation data for learning"""
        
        # Check for past compositions
        past_compositions = self.env['gift.composition'].search([
            ('partner_id', '=', partner_id),
            ('state', 'in', ['approved', 'delivered'])
        ], limit=5)
        
        if not past_compositions:
            return []
        
        recommendations = []
        for comp in past_compositions:
            recommendations.append({
                'date': comp.create_date,
                'products': comp.product_ids.mapped('name'),
                'budget': comp.target_budget,
                'actual_cost': comp.actual_cost,
                'categories': list(set(comp.product_ids.mapped('lebiggot_category')))
            })
        
        return recommendations
    
    def _ollama_enabled(self):
        """Check if Ollama is enabled"""
        return self.env['ir.config_parameter'].sudo().get_param(
            'lebigott_ai.ollama_enabled', 'false'
        ).lower() == 'true'
    
    def _ollama_complete(self, prompt):
        """Get Ollama completion"""
        try:
            composition_engine = self.env['composition.engine']
            return composition_engine._ollama_complete(prompt)
        except:
            return None
    
    def record_feedback(self, partner_id, products, satisfaction_score):
        """Record feedback for continuous learning"""
        
        if not self.learning_enabled:
            return
        
        learning_data = {
            'partner_id': partner_id,
            'products': [p.id for p in products],
            'satisfaction': satisfaction_score,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store for future training
        cache_key = f"ai_feedback_{partner_id}_{fields.Date.today()}"
        self.env['ir.config_parameter'].sudo().set_param(cache_key, json.dumps(learning_data))
        
        _logger.info(f"Recorded AI feedback: Partner {partner_id}, Satisfaction {satisfaction_score}")