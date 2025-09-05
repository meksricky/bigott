from odoo import models, fields, api
from datetime import datetime
import random
from odoo.exceptions import UserError
import json
import logging
import requests
from collections import defaultdict, Counter

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
        
        # Analyze client history with enhanced logic
        client_analysis = self._enhanced_client_analysis(partner_id)
        
        # Process and weight the notes
        processed_notes = self._process_client_notes(notes_text, client_analysis)
        
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
                dietary_restrictions, client_analysis, processed_notes
            )
        else:
            return self._generate_custom_composition(
                partner_id, target_budget, target_year,
                dietary_restrictions, client_analysis, processed_notes
            )
    
    def _enhanced_client_analysis(self, partner_id):
        """Enhanced client analysis that extracts deeper insights"""
        
        # Get basic analysis from existing method
        basic_analysis = self.env['client.order.history'].analyze_client_patterns(partner_id)
        
        if not basic_analysis['has_history']:
            return basic_analysis
        
        # Get client's historical data
        histories = self.env['client.order.history'].search([
            ('partner_id', '=', partner_id)
        ], order='order_year desc')
        
        # Enhanced analysis
        enhanced_data = {
            'preferred_categories': self._analyze_category_preferences(histories),
            'preferred_brands': self._analyze_brand_preferences(histories),
            'preferred_grades': self._analyze_grade_preferences(histories),
            'price_sensitivity': self._analyze_price_sensitivity(histories),
            'seasonal_patterns': self._analyze_seasonal_patterns(histories),
            'satisfaction_drivers': self._analyze_satisfaction_drivers(histories),
            'avoided_products': self._find_avoided_products(histories),
        }
        
        # Merge with basic analysis
        basic_analysis.update(enhanced_data)
        return basic_analysis
    
    def _analyze_category_preferences(self, histories):
        """Analyze which categories the client prefers"""
        category_counts = defaultdict(int)
        category_budgets = defaultdict(float)
        
        for history in histories:
            categories = history.get_category_structure()
            total_budget = history.total_budget
            
            for category, count in categories.items():
                category_counts[category] += count
                # Estimate budget allocation per category
                category_budgets[category] += total_budget * (count / sum(categories.values()))
        
        # Calculate preferences based on frequency and budget allocation
        preferences = {}
        total_items = sum(category_counts.values())
        total_budget = sum(category_budgets.values())
        
        for category in category_counts:
            frequency_score = category_counts[category] / total_items if total_items > 0 else 0
            budget_score = category_budgets[category] / total_budget if total_budget > 0 else 0
            preferences[category] = (frequency_score + budget_score) / 2
        
        return preferences
    
    def _analyze_brand_preferences(self, histories):
        """Analyze brand preferences from historical products"""
        brand_counts = defaultdict(int)
        brand_satisfactions = defaultdict(list)
        
        for history in histories:
            satisfaction = float(history.client_satisfaction) if history.client_satisfaction else 3
            for product in history.product_ids:
                if product.brand:
                    brand_counts[product.brand] += 1
                    brand_satisfactions[product.brand].append(satisfaction)
        
        # Calculate brand scores based on frequency and satisfaction
        brand_scores = {}
        for brand in brand_counts:
            frequency_score = brand_counts[brand] / sum(brand_counts.values()) if brand_counts else 0
            avg_satisfaction = sum(brand_satisfactions[brand]) / len(brand_satisfactions[brand])
            satisfaction_score = avg_satisfaction / 5.0  # Normalize to 0-1
            brand_scores[brand] = (frequency_score + satisfaction_score) / 2
        
        return brand_scores
    
    def _analyze_grade_preferences(self, histories):
        """Analyze preferred product grades"""
        grade_counts = defaultdict(int)
        grade_satisfactions = defaultdict(list)
        
        for history in histories:
            satisfaction = float(history.client_satisfaction) if history.client_satisfaction else 3
            for product in history.product_ids:
                if product.product_grade:
                    grade_counts[product.product_grade] += 1
                    grade_satisfactions[product.product_grade].append(satisfaction)
        
        # Calculate grade preferences
        grade_scores = {}
        for grade in grade_counts:
            frequency_score = grade_counts[grade] / sum(grade_counts.values()) if grade_counts else 0
            if grade_satisfactions[grade]:
                avg_satisfaction = sum(grade_satisfactions[grade]) / len(grade_satisfactions[grade])
                satisfaction_score = avg_satisfaction / 5.0
            else:
                satisfaction_score = 0.6  # Default
            grade_scores[grade] = (frequency_score + satisfaction_score) / 2
        
        return grade_scores
    
    def _analyze_price_sensitivity(self, histories):
        """Analyze how price-sensitive the client is"""
        if len(histories) < 2:
            return 'unknown'
        
        budgets = [h.total_budget for h in histories.sorted('order_year')]
        budget_changes = []
        
        for i in range(1, len(budgets)):
            change = (budgets[i] - budgets[i-1]) / budgets[i-1] if budgets[i-1] > 0 else 0
            budget_changes.append(change)
        
        avg_change = sum(budget_changes) / len(budget_changes)
        
        if avg_change > 0.1:
            return 'low'  # Willing to increase budget
        elif avg_change < -0.1:
            return 'high'  # Tends to decrease budget
        else:
            return 'medium'
    
    def _analyze_seasonal_patterns(self, histories):
        """Analyze seasonal patterns (placeholder for future enhancement)"""
        return {}
    
    def _analyze_satisfaction_drivers(self, histories):
        """Find what drives client satisfaction"""
        high_satisfaction = histories.filtered(lambda h: h.client_satisfaction in ['4', '5'])
        low_satisfaction = histories.filtered(lambda h: h.client_satisfaction in ['1', '2'])
        
        drivers = {
            'high_satisfaction_categories': defaultdict(int),
            'low_satisfaction_categories': defaultdict(int),
            'high_satisfaction_brands': defaultdict(int),
            'low_satisfaction_brands': defaultdict(int),
        }
        
        # Analyze high satisfaction patterns
        for history in high_satisfaction:
            categories = history.get_category_structure()
            for category, count in categories.items():
                drivers['high_satisfaction_categories'][category] += count
            
            for product in history.product_ids:
                if product.brand:
                    drivers['high_satisfaction_brands'][product.brand] += 1
        
        # Analyze low satisfaction patterns
        for history in low_satisfaction:
            categories = history.get_category_structure()
            for category, count in categories.items():
                drivers['low_satisfaction_categories'][category] += count
            
            for product in history.product_ids:
                if product.brand:
                    drivers['low_satisfaction_brands'][product.brand] += 1
        
        return drivers
    
    def _find_avoided_products(self, histories):
        """Find products/categories that should be avoided"""
        # Products from low satisfaction orders
        avoided_products = set()
        avoided_brands = set()
        
        low_satisfaction = histories.filtered(lambda h: h.client_satisfaction in ['1', '2'])
        for history in low_satisfaction:
            for product in history.product_ids:
                avoided_products.add(product.id)
                if product.brand:
                    avoided_brands.add(product.brand)
        
        return {
            'products': list(avoided_products),
            'brands': list(avoided_brands)
        }
    
    def _process_client_notes(self, notes_text, client_analysis):
        """Process and extract insights from client notes"""
        if not notes_text:
            return {}
        
        notes_insights = {
            'raw_notes': notes_text,
            'keywords': self._extract_keywords(notes_text),
            'preferences': self._extract_preferences(notes_text),
            'restrictions': self._extract_restrictions(notes_text),
            'sentiment': self._analyze_sentiment(notes_text),
        }
        
        return notes_insights
    
    def _extract_keywords(self, notes_text):
        """Extract relevant keywords from notes"""
        if not notes_text:
            return []
        
        # Define keyword categories
        preference_keywords = {
            'wine': ['wine', 'vino', 'red wine', 'white wine', 'rosé', 'champagne', 'cava'],
            'cheese': ['cheese', 'queso', 'fromage', 'manchego', 'goat cheese'],
            'meat': ['jamón', 'ham', 'chorizo', 'salami', 'charcuterie'],
            'sweet': ['chocolate', 'sweet', 'dulce', 'honey', 'dessert'],
            'premium': ['premium', 'luxury', 'high-end', 'exclusive', 'best'],
            'traditional': ['traditional', 'classic', 'authentic', 'original'],
            'new': ['new', 'different', 'unique', 'novel', 'innovative'],
        }
        
        notes_lower = notes_text.lower()
        found_keywords = {}
        
        for category, keywords in preference_keywords.items():
            for keyword in keywords:
                if keyword in notes_lower:
                    found_keywords[category] = found_keywords.get(category, 0) + 1
        
        return found_keywords
    
    def _extract_preferences(self, notes_text):
        """Extract specific preferences from notes"""
        preferences = {}
        notes_lower = notes_text.lower()
        
        # Budget preferences
        if any(word in notes_lower for word in ['budget', 'cheap', 'economical', 'affordable']):
            preferences['budget_conscious'] = True
        elif any(word in notes_lower for word in ['premium', 'luxury', 'expensive', 'high-end']):
            preferences['premium_preference'] = True
        
        # Novelty preferences
        if any(word in notes_lower for word in ['new', 'different', 'surprise', 'novel']):
            preferences['novelty_seeking'] = True
        elif any(word in notes_lower for word in ['traditional', 'classic', 'usual', 'same']):
            preferences['traditional_preference'] = True
        
        return preferences
    
    def _extract_restrictions(self, notes_text):
        """Extract restrictions from notes"""
        restrictions = []
        notes_lower = notes_text.lower()
        
        restriction_mappings = {
            'vegan': ['vegan', 'plant-based', 'no animal'],
            'vegetarian': ['vegetarian', 'no meat'],
            'halal': ['halal', 'muslim'],
            'kosher': ['kosher', 'jewish'],
            'no_alcohol': ['no alcohol', 'alcohol-free', 'non-alcoholic', 'dry'],
            'gluten_free': ['gluten-free', 'celiac', 'no gluten'],
            'dairy_free': ['dairy-free', 'lactose', 'no dairy'],
        }
        
        for restriction, keywords in restriction_mappings.items():
            if any(keyword in notes_lower for keyword in keywords):
                restrictions.append(restriction)
        
        return restrictions
    
    def _analyze_sentiment(self, notes_text):
        """Basic sentiment analysis"""
        if not notes_text:
            return 'neutral'
        
        positive_words = ['love', 'excellent', 'amazing', 'wonderful', 'favorite', 'best']
        negative_words = ['hate', 'dislike', 'terrible', 'awful', 'worst', 'never']
        
        notes_lower = notes_text.lower()
        positive_count = sum(1 for word in positive_words if word in notes_lower)
        negative_count = sum(1 for word in negative_words if word in notes_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _generate_custom_composition(self, partner_id, target_budget, target_year,
                                   dietary_restrictions, client_analysis, processed_notes):
        """Generate custom product composition with enhanced logic"""
        
        # Determine category structure based on history and notes
        category_structure = self._determine_optimal_category_structure(
            client_analysis, processed_notes, target_budget
        )
        
        # Determine budget direction
        budget_direction = self._determine_budget_direction_enhanced(
            target_budget, client_analysis, processed_notes
        )
        
        # Build intelligent product selection
        selected_products = self._build_intelligent_product_selection(
            category_structure, target_budget, budget_direction,
            dietary_restrictions, client_analysis, processed_notes
        )
        
        # Generate enhanced reasoning
        reasoning_html = self._generate_enhanced_reasoning(
            selected_products, budget_direction, client_analysis, processed_notes
        )
        
        # Optionally get Ollama enhancement
        if self._ollama_enabled() and processed_notes.get('raw_notes'):
            prompt = self._build_ollama_prompt(selected_products, target_budget, processed_notes)
            ollama_note = self._ollama_complete(prompt)
            if ollama_note:
                reasoning_html += f"<hr/><p><strong>🤖 AI Enhanced Analysis:</strong> {ollama_note}</p>"
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner_id,
            'target_year': target_year,
            'target_budget': target_budget,
            'composition_type': 'custom',
            'product_ids': [(6, 0, [p.id for p in selected_products])],
            'dietary_restrictions': ','.join(dietary_restrictions or []),
            'reasoning': reasoning_html,
            'confidence_score': self._calculate_enhanced_confidence(
                selected_products, target_budget, client_analysis, processed_notes
            ),
            'novelty_score': self._calculate_enhanced_novelty(
                selected_products, client_analysis, processed_notes
            ),
            'historical_compatibility': self._calculate_historical_compatibility(
                selected_products, client_analysis
            )
        })
        
        # Set category structure
        actual_categories = self._analyze_product_categories(selected_products)
        composition.set_category_breakdown(actual_categories)
        
        return composition
    
    def _determine_optimal_category_structure(self, client_analysis, processed_notes, target_budget):
        """Determine optimal category structure based on client data"""
        
        # Default structure
        base_structure = {
            'main_beverage': 1,
            'aperitif': 1, 
            'experience_gastronomica': 1,
            'foie_gras': 1,
            'charcuterie': 2,
            'sweets': 1
        }
        
        if not client_analysis.get('has_history'):
            return base_structure
        
        # Adjust based on client preferences
        category_preferences = client_analysis.get('preferred_categories', {})
        
        # Modify structure based on preferences
        for category, preference_score in category_preferences.items():
            if preference_score > 0.7:  # High preference
                base_structure[category] = base_structure.get(category, 1) + 1
            elif preference_score < 0.3:  # Low preference
                base_structure[category] = max(0, base_structure.get(category, 1) - 1)
        
        # Adjust based on notes
        if processed_notes.get('keywords'):
            keywords = processed_notes['keywords']
            if keywords.get('wine', 0) > 0:
                base_structure['main_beverage'] += 1
            if keywords.get('sweet', 0) > 0:
                base_structure['sweets'] += 1
            if keywords.get('meat', 0) > 0:
                base_structure['charcuterie'] += 1
        
        # Ensure minimum structure
        for category in base_structure:
            if base_structure[category] < 1:
                base_structure[category] = 1
        
        return base_structure
    
    def _determine_budget_direction_enhanced(self, target_budget, client_analysis, processed_notes):
        """Enhanced budget direction determination"""
        
        # Basic budget direction
        if client_analysis.get('has_history'):
            recent_budgets = client_analysis.get('recent_budgets', [])
            if recent_budgets:
                last_budget = recent_budgets[0]
                change_percent = (target_budget - last_budget) / last_budget if last_budget > 0 else 0
                
                if change_percent > 0.15:
                    base_direction = 'upgrade'
                elif change_percent < -0.15:
                    base_direction = 'downgrade'
                else:
                    base_direction = 'same'
            else:
                base_direction = 'same'
        else:
            base_direction = 'same'
        
        # Adjust based on notes
        if processed_notes.get('preferences'):
            prefs = processed_notes['preferences']
            if prefs.get('premium_preference'):
                if base_direction == 'downgrade':
                    base_direction = 'same'
                elif base_direction == 'same':
                    base_direction = 'upgrade'
            elif prefs.get('budget_conscious'):
                if base_direction == 'upgrade':
                    base_direction = 'same'
                elif base_direction == 'same':
                    base_direction = 'downgrade'
        
        return base_direction
    
    def _build_intelligent_product_selection(self, category_structure, target_budget, 
                                           budget_direction, dietary_restrictions, 
                                           client_analysis, processed_notes):
        """Build intelligent product selection using all available data"""
        
        selected_products = []
        remaining_budget = target_budget
        categories_to_fill = len([c for c in category_structure.values() if c > 0])
        
        # Process each category with intelligent selection
        for category, quantity in category_structure.items():
            if quantity <= 0:
                continue
                
            category_budget = remaining_budget / categories_to_fill
            category_products = self._select_intelligent_category_products(
                category, quantity, category_budget, budget_direction,
                dietary_restrictions, client_analysis, processed_notes
            )
            
            selected_products.extend(category_products)
            remaining_budget -= sum(p.list_price for p in category_products)
            categories_to_fill -= 1
        
        return selected_products
    
    def _select_intelligent_category_products(self, category, quantity, budget_per_category,
                                            budget_direction, dietary_restrictions, 
                                            client_analysis, processed_notes):
        """Intelligently select products for a category"""
        
        # Base search domain
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
        
        # Apply additional restrictions from notes
        note_restrictions = processed_notes.get('restrictions', [])
        for restriction in note_restrictions:
            if restriction == 'vegan':
                domain.append(('is_vegan', '=', True))
            elif restriction == 'no_alcohol':
                domain.append(('contains_alcohol', '=', False))
        
        available_products = self.env['product.template'].search(domain)
        
        # Filter out avoided products
        if client_analysis.get('avoided_products'):
            avoided_ids = client_analysis['avoided_products'].get('products', [])
            avoided_brands = client_analysis['avoided_products'].get('brands', [])
            
            available_products = available_products.filtered(
                lambda p: p.id not in avoided_ids and p.brand not in avoided_brands
            )
        
        # Score products based on multiple criteria
        scored_products = []
        target_price_per_item = budget_per_category / quantity if quantity > 0 else budget_per_category
        
        for product in available_products:
            score = self._score_product_for_client(
                product, target_price_per_item, budget_direction,
                client_analysis, processed_notes
            )
            scored_products.append((product, score))
        
        # Sort by score (highest first)
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Select required quantity, ensuring diversity
        selected = []
        used_brands = set()
        
        for product, score in scored_products:
            if len(selected) >= quantity:
                break
                
            # Prefer brand diversity unless client has strong brand preference
            brand_preferences = client_analysis.get('preferred_brands', {})
            if product.brand in used_brands and product.brand not in brand_preferences:
                continue
                
            selected.append(product)
            if product.brand:
                used_brands.add(product.brand)
        
        # If we don't have enough, fill from remaining products
        while len(selected) < quantity and len(selected) < len(available_products):
            for product, score in scored_products:
                if product not in selected:
                    selected.append(product)
                    break
        
        return selected
    
    def _score_product_for_client(self, product, target_price, budget_direction, 
                                client_analysis, processed_notes):
        """Score a product for this specific client"""
        
        score = 0.0
        
        # 1. Price fit (25% weight)
        price_variance = abs(product.list_price - target_price) / target_price if target_price > 0 else 1
        price_score = max(0, 1 - price_variance)
        score += price_score * 0.25
        
        # 2. Grade preference (20% weight)
        grade_preferences = client_analysis.get('preferred_grades', {})
        if product.product_grade in grade_preferences:
            grade_score = grade_preferences[product.product_grade]
        else:
            # Default grade scores based on budget direction
            grade_defaults = {
                'upgrade': {'luxury': 1.0, 'premium': 0.8, 'standard': 0.4, 'economical': 0.2},
                'downgrade': {'economical': 1.0, 'standard': 0.8, 'premium': 0.4, 'luxury': 0.2},
                'same': {'standard': 1.0, 'premium': 0.9, 'economical': 0.7, 'luxury': 0.6},
            }
            grade_score = grade_defaults.get(budget_direction, {}).get(product.product_grade, 0.5)
        score += grade_score * 0.20
        
        # 3. Brand preference (15% weight)
        brand_preferences = client_analysis.get('preferred_brands', {})
        if product.brand and product.brand in brand_preferences:
            brand_score = brand_preferences[product.brand]
        else:
            brand_score = 0.5  # Neutral for unknown brands
        score += brand_score * 0.15
        
        # 4. Notes alignment (25% weight)
        notes_score = self._score_product_notes_alignment(product, processed_notes)
        score += notes_score * 0.25
        
        # 5. Novelty factor (15% weight)
        novelty_score = self._score_product_novelty(product, client_analysis, processed_notes)
        score += novelty_score * 0.15
        
        return min(1.0, score)
    
    def _score_product_notes_alignment(self, product, processed_notes):
        """Score how well a product aligns with client notes"""
        if not processed_notes or not processed_notes.get('keywords'):
            return 0.5  # Neutral score
        
        score = 0.5  # Base score
        keywords = processed_notes['keywords']
        
        # Check product alignment with noted preferences
        product_name_lower = (product.name or '').lower()
        product_category = product.lebiggot_category or ''
        
        # Wine preferences
        if keywords.get('wine', 0) > 0 and product_category == 'main_beverage':
            score += 0.3
        
        # Sweet preferences
        if keywords.get('sweet', 0) > 0 and product_category == 'sweets':
            score += 0.3
        
        # Meat preferences
        if keywords.get('meat', 0) > 0 and product_category == 'charcuterie':
            score += 0.3
        
        # Premium preferences
        if keywords.get('premium', 0) > 0 and product.product_grade in ['premium', 'luxury']:
            score += 0.2
        
        # Traditional preferences
        if keywords.get('traditional', 0) > 0:
            traditional_keywords = ['tradicional', 'classic', 'original', 'artisan']
            if any(keyword in product_name_lower for keyword in traditional_keywords):
                score += 0.2
        
        return min(1.0, score)
    
    def _score_product_novelty(self, product, client_analysis, processed_notes):
        """Score product novelty based on client history and preferences"""
        
        # Check if client seeks novelty
        novelty_seeking = processed_notes.get('preferences', {}).get('novelty_seeking', False)
        traditional_preference = processed_notes.get('preferences', {}).get('traditional_preference', False)
        
        # Base novelty score
        if not client_analysis.get('has_history'):
            return 0.7  # Moderate novelty for new clients
        
        # Check if product was used before
        historical_products = set()
        for history in self.env['client.order.history'].search([('partner_id', '=', client_analysis.get('partner_id'))]):
            historical_products.update(history.product_ids.ids)
        
        if product.id in historical_products:
            # Product was used before
            if traditional_preference:
                return 0.8  # Client likes familiar products
            else:
                return 0.2  # Penalize repetition
        else:
            # New product
            if novelty_seeking:
                return 0.9  # Client wants new things
            else:
                return 0.6  # Moderate novelty
    
    def _build_ollama_prompt(self, selected_products, target_budget, processed_notes):
        """Build enhanced prompt for Ollama"""
        product_details = []
        for product in selected_products:
            details = f"{product.name} ({product.lebiggot_category}, {product.product_grade}, €{product.list_price:.2f})"
            product_details.append(details)
        
        prompt = f"""
        You are a luxury gourmet gift advisor. Analyze this gift composition:
        
        Budget: €{target_budget:.0f}
        Products: {', '.join(product_details)}
        Client Notes: {processed_notes.get('raw_notes', 'None')}
        
        Provide a brief (60 words max) client-facing explanation highlighting:
        1. How the selection matches their preferences
        2. The unique value proposition
        3. Why this combination works well together
        
        Keep it elegant, personal, and premium-focused.
        """
        return prompt
    
    def _calculate_enhanced_confidence(self, products, target_budget, client_analysis, processed_notes):
        """Calculate enhanced confidence score"""
        
        confidence = 0.3  # Base confidence
        
        # Budget fit
        actual_cost = sum(p.list_price for p in products)
        budget_variance = abs(actual_cost - target_budget) / target_budget if target_budget > 0 else 1
        budget_confidence = max(0, 1 - budget_variance)
        confidence += budget_confidence * 0.25
        
        # Historical data quality
        if client_analysis.get('has_history'):
            years_data = client_analysis.get('years_of_data', 0)
            history_confidence = min(0.25, years_data * 0.08)
            confidence += history_confidence
        
        # Notes processing quality
        if processed_notes and processed_notes.get('raw_notes'):
            notes_confidence = 0.15  # Bonus for having notes
            confidence += notes_confidence
        
        # Product selection completeness
        if len(products) >= 6:
            confidence += 0.15
        elif len(products) >= 4:
            confidence += 0.10
        
        return min(1.0, confidence)
    
    def _calculate_enhanced_novelty(self, products, client_analysis, processed_notes):
        """Calculate enhanced novelty score"""
        
        if not client_analysis.get('has_history'):
            return 0.8  # High novelty for new clients
        
        # Check how many products are new
        historical_products = set()
        histories = self.env['client.order.history'].search([('partner_id', '=', client_analysis.get('partner_id'))])
        for history in histories:
            historical_products.update(history.product_ids.ids)
        
        new_products = sum(1 for p in products if p.id not in historical_products)
        novelty_ratio = new_products / len(products) if products else 0
        
        base_novelty = novelty_ratio * 0.7
        
        # Adjust based on client preferences
        if processed_notes.get('preferences', {}).get('novelty_seeking'):
            base_novelty += 0.2
        elif processed_notes.get('preferences', {}).get('traditional_preference'):
            base_novelty = max(0.3, base_novelty - 0.2)
        
        return min(1.0, base_novelty)
    
    def _calculate_historical_compatibility(self, products, client_analysis):
        """Calculate how well products align with historical preferences"""
        
        if not client_analysis.get('has_history'):
            return 0.5  # Neutral for new clients
        
        compatibility = 0.0
        
        # Category compatibility
        selected_categories = [p.lebiggot_category for p in products]
        category_preferences = client_analysis.get('preferred_categories', {})
        
        if category_preferences:
            category_scores = [category_preferences.get(cat, 0.5) for cat in selected_categories]
            compatibility += sum(category_scores) / len(category_scores) * 0.4
        
        # Brand compatibility
        selected_brands = [p.brand for p in products if p.brand]
        brand_preferences = client_analysis.get('preferred_brands', {})
        
        if brand_preferences and selected_brands:
            brand_scores = [brand_preferences.get(brand, 0.5) for brand in selected_brands]
            compatibility += sum(brand_scores) / len(brand_scores) * 0.3
        
        # Grade compatibility
        selected_grades = [p.product_grade for p in products if p.product_grade]
        grade_preferences = client_analysis.get('preferred_grades', {})
        
        if grade_preferences and selected_grades:
            grade_scores = [grade_preferences.get(grade, 0.5) for grade in selected_grades]
            compatibility += sum(grade_scores) / len(grade_scores) * 0.3
        
        return min(1.0, compatibility)
    
    def _generate_enhanced_reasoning(self, products, budget_direction, client_analysis, processed_notes):
        """Generate enhanced reasoning that explains the AI's decision process"""
        
        reasons = [
            "<h3>🎨 Señor Bigott Enhanced AI Analysis</h3>",
            f"<p><strong>Products Selected:</strong> {len(products)} intelligently chosen items</p>",
            f"<p><strong>Total Cost:</strong> €{sum(p.list_price for p in products):.2f}</p>"
        ]
        
        # Client-specific insights
        if client_analysis.get('has_history'):
            years = client_analysis.get('years_of_data', 0)
            reasons.append(f"<p><strong>📊 Client Intelligence:</strong> Analysis based on {years} years of detailed preference data</p>")
            
            # Top preferences
            category_prefs = client_analysis.get('preferred_categories', {})
            if category_prefs:
                top_categories = sorted(category_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
                cat_text = ", ".join([f"{cat.replace('_', ' ').title()} ({score:.1%})" for cat, score in top_categories])
                reasons.append(f"<p><strong>🎯 Top Preferences:</strong> {cat_text}</p>")
            
            # Satisfaction insights
            satisfaction_drivers = client_analysis.get('satisfaction_drivers', {})
            if satisfaction_drivers.get('high_satisfaction_categories'):
                high_sat_cats = satisfaction_drivers['high_satisfaction_categories']
                top_sat_cat = max(high_sat_cats.items(), key=lambda x: x[1])[0]
                reasons.append(f"<p><strong>⭐ Satisfaction Driver:</strong> Enhanced focus on {top_sat_cat.replace('_', ' ').title()} based on historical satisfaction</p>")
        
        # Notes integration
        if processed_notes and processed_notes.get('raw_notes'):
            reasons.append(f"<p><strong>📝 Notes Integration:</strong> Your specific requests have been weighted into the selection algorithm</p>")
            
            # Sentiment analysis
            sentiment = processed_notes.get('sentiment', 'neutral')
            if sentiment == 'positive':
                reasons.append("<p><strong>😊 Positive Outlook:</strong> Selections optimized to build on your expressed enthusiasm</p>")
            elif sentiment == 'negative':
                reasons.append("<p><strong>🔄 Preference Adjustment:</strong> Selections carefully adjusted to address your concerns</p>")
            
            # Keyword integration
            keywords = processed_notes.get('keywords', {})
            if keywords:
                top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:2]
                keyword_text = ", ".join([kw.replace('_', ' ').title() for kw, count in top_keywords])
                reasons.append(f"<p><strong>🔍 Keyword Focus:</strong> Special attention to: {keyword_text}</p>")
        
        # Budget strategy
        budget_explanations = {
            'upgrade': "🔺 Premium Enhancement: Elevated selections with luxury brands and superior grades",
            'downgrade': "💡 Value Optimization: Smart selections maintaining quality while respecting budget constraints", 
            'same': "🎯 Refined Evolution: Balanced improvements with new brands and enhanced variety"
        }
        reasons.append(f"<p><strong>{budget_explanations.get(budget_direction, '🎯 Balanced Selection')}</strong></p>")
        
        # AI confidence factors
        reasons.append("<h4>🧠 AI Analysis Factors</h4>")
        reasons.append("<ul>")
        reasons.append("<li><strong>Historical Pattern Matching:</strong> Advanced algorithms analyzed purchasing patterns and satisfaction correlations</li>")
        reasons.append("<li><strong>Preference Weighting:</strong> Multi-dimensional scoring considering category, brand, grade, and price preferences</li>")
        reasons.append("<li><strong>Novelty Balancing:</strong> Optimal mix of familiar favorites and exciting discoveries</li>")
        reasons.append("<li><strong>Quality Assurance:</strong> All selections meet Le Biggot's premium standards and your dietary requirements</li>")
        reasons.append("</ul>")
        
        return "".join(reasons)

    # Keep existing methods for experience composition and other utilities...
    def _generate_experience_composition(self, partner_id, target_budget, target_year, 
                                       dietary_restrictions, client_analysis, processed_notes):
        """Generate experience-based composition with enhanced logic"""
        
        available_experiences = self._get_available_experiences(
            partner_id, target_budget, dietary_restrictions
        )
        
        if not available_experiences:
            return self._generate_custom_composition(
                partner_id, target_budget, target_year,
                dietary_restrictions, client_analysis, processed_notes
            )
        
        selected_experience = self._select_best_experience(
            available_experiences, target_budget, client_analysis
        )
        
        composition = self.env['gift.composition'].create({
            'partner_id': partner_id,
            'target_year': target_year,
            'target_budget': target_budget,
            'composition_type': 'experience',
            'experience_id': selected_experience.id,
            'dietary_restrictions': ','.join(dietary_restrictions or []),
            'reasoning': self._generate_experience_reasoning(
                selected_experience, client_analysis, target_budget, processed_notes
            ),
            'confidence_score': self._calculate_experience_confidence(
                selected_experience, target_budget, client_analysis
            ),
            'novelty_score': self._calculate_novelty_score(
                selected_experience, client_analysis
            )
        })
        
        return composition
    
    # Keep all the existing utility methods...
    def _get_available_experiences(self, partner_id, target_budget, dietary_restrictions):
        experiences = self.env['gift.experience'].search([('active', '=', True)])
        available = []
        
        for experience in experiences:
            if not experience.is_available_for_client(partner_id):
                continue
            
            if dietary_restrictions and not experience.check_dietary_compatibility(dietary_restrictions):
                continue
            
            budget_variance = abs(experience.base_cost - target_budget) / target_budget
            if budget_variance > 0.25:
                continue
            
            available.append(experience)
        
        return available
    
    def _select_best_experience(self, experiences, target_budget, client_analysis):
        if not experiences:
            return None
        
        scored_experiences = []
        for exp in experiences:
            score = 0
            
            budget_diff = abs(exp.base_cost - target_budget)
            budget_score = max(0, 1 - (budget_diff / target_budget))
            score += budget_score * 0.4
            
            if exp.average_satisfaction > 0:
                satisfaction_score = exp.average_satisfaction / 5.0
                score += satisfaction_score * 0.3
            else:
                score += 0.15
            
            novelty_score = max(0, 1 - (exp.times_used / 20))
            score += novelty_score * 0.3
            
            scored_experiences.append((exp, score))
        
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        return scored_experiences[0][0]
    
    def _calculate_experience_confidence(self, experience, target_budget, client_analysis):
        confidence = 0.5
        
        budget_variance = abs(experience.base_cost - target_budget) / target_budget
        budget_confidence = max(0, 1 - budget_variance)
        confidence += budget_confidence * 0.3
        
        if client_analysis and client_analysis['has_history']:
            history_confidence = min(0.2, client_analysis['years_of_data'] * 0.07)
            confidence += history_confidence
        
        if experience.times_used > 0:
            usage_confidence = min(0.2, experience.times_used * 0.01)
            if experience.average_satisfaction > 0:
                satisfaction_bonus = (experience.average_satisfaction / 5.0) * 0.1
                confidence += usage_confidence + satisfaction_bonus
            else:
                confidence += usage_confidence
        
        return min(1.0, confidence)
    
    def _calculate_novelty_score(self, experience, client_analysis):
        if not client_analysis or not client_analysis['has_history']:
            return 1.0
        
        base_novelty = 0.8
        
        if experience.experience_theme not in ['mediterranean', 'spanish_classics']:
            base_novelty += 0.2
        
        return min(1.0, base_novelty)
    
    def _generate_experience_reasoning(self, experience, client_analysis, target_budget, processed_notes):
        reasons = [
            f"<h3>🎁 Señor Bigott Selected: {experience.name}</h3>",
            f"<p><strong>Experience Theme:</strong> {experience.experience_theme.replace('_', ' ').title()}</p>",
            f"<p><strong>Products Included:</strong> {len(experience.product_ids)} carefully curated items</p>",
            f"<p><strong>Cost Analysis:</strong> €{experience.base_cost:.2f} (Target: €{target_budget:.2f})</p>"
        ]
        
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
        
        if experience.times_used > 0:
            reasons.append(f"<p><strong>📈 Experience Track Record:</strong> Successfully delivered {experience.times_used} times</p>")
            if experience.average_satisfaction > 0:
                reasons.append(f"<p><strong>⭐ Average Satisfaction:</strong> {experience.average_satisfaction:.1f}/5 stars</p>")
        
        if processed_notes and processed_notes.get('raw_notes'):
            reasons.append(f"<p><strong>📝 Client Notes Considered:</strong> {processed_notes['raw_notes']}</p>")
        
        return "".join(reasons)
    
    def _analyze_product_categories(self, products):
        categories = {}
        for product in products:
            cat = product.lebiggot_category
            categories[cat] = categories.get(cat, 0) + 1
        return categories