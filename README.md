# 🎁 Señor Bigott Gift Composition Engine

An AI-powered gift recommendation system for Le Biggot's gourmet gift boxes.

## Features

### 🧠 AI-Powered Recommendations
- Historical client analysis (3+ years of data)
- Budget adaptation (upgrade/maintain/downgrade logic)
- Dietary restriction handling (vegan, halal, non-alcoholic)
- Confidence scoring and novelty metrics

### 📦 Two Composition Types
1. **Experience-Based**: Curated 3-5 product experiences with instructions
2. **Custom Compositions**: Bespoke selections based on client preferences

### 📊 Client Intelligence
- Automatic client tier classification
- Budget trend analysis
- Satisfaction tracking
- Preference learning

### 🎯 Business Logic
- Category structure preservation
- Volume matching for beverages
- Brand variation strategies
- Experience non-repetition rules

## Installation

1. Copy the module to your Odoo addons directory
2. Update the apps list
3. Install "Señor Bigott Gift Composition Engine"
4. Load demo data to see the system in action

## Usage

1. **Setup Clients**: Add companies with dietary preferences
2. **Create Experiences**: Design curated 3-5 product experiences
3. **Import Products**: Categorize products using Le Biggot classification
4. **Add History**: Record past gift orders for learning
5. **Generate Compositions**: Use the AI wizard to create new recommendations

## Module Structure

```
senor_bigott_composition/
├── __manifest__.py
├── models/
│   ├── client_order_history.py
│   ├── gift_experience.py
│   ├── gift_composition.py
│   ├── composition_engine.py
│   ├── product_template.py
│   └── res_partner.py
├── views/
│   ├── menu_views.xml
│   ├── gift_composition_views.xml
│   ├── client_history_views.xml
│   ├── experience_views.xml
│   ├── product_template_views.xml
│   └── res_partner_views.xml
├── wizard/
│   └── composition_wizard_views.xml
├── data/
│   ├── product_categories_data.xml
│   └── demo_data.xml
└── security/
    └── ir.model.access.csv
```

## Categories

- **Main Beverage**: Cava, Champagne, Red/White Wine
- **Aperitif**: Vermouth, Tokaj, Sherry
- **Experience Gastronomica**: Curated tastings
- **Foie Gras**: Luxury liver products
- **Charcuterie**: Ibérico hams, cheeses
- **Sweets**: Turrón, chocolate, traditional confections

## AI Engine Logic

The composition engine follows sophisticated business rules:

1. **Client Analysis**: Examines 3-year purchase history
2. **Budget Direction**: Determines upgrade/maintain/downgrade strategy
3. **Category Preservation**: Maintains structural consistency
4. **Dietary Compliance**: Filters products by restrictions
5. **Novelty Injection**: Introduces new products while respecting preferences
6. **Confidence Scoring**: Rates recommendation quality

## Demo Data

The module includes comprehensive demo data:
- 3 sample clients with different dietary needs
- 15+ products across all categories
- 4 curated experiences
- Historical order data spanning 2-3 years
- Sample AI-generated compositions

## Technical Details

- **Odoo Version**: 17.0+
- **Dependencies**: base, sale, product
- **License**: LGPL-3
- **Database**: Uses JSON fields for category structures (Odoo 17+ compatibility)

---

*Created with ❤️ for Le Biggot's gourmet gift experiences*
=======
# bigott
