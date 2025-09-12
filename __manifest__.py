# -*- coding: utf-8 -*-
{
    'name': 'Ollama Gift Recommender',
    'version': '1.0.0',
    'category': 'Sales/Sales',
    'summary': 'AI-powered gift recommendations using Ollama',
    'description': """
Ollama Gift Recommender
=======================

AI-powered gift recommendation system using Ollama for intelligent product suggestions.

Features:
- Ollama AI integration for intelligent recommendations
- Client history analysis
- Budget-aware product selection
- Dietary restrictions support
- Stock availability checking
- Internal reference validation
- Fallback recommendation system
- Performance analytics and monitoring
    """,
    'author': 'Your Company',
    'website': 'https://www.yourcompany.com',
    'depends': [
        'base',
        'sale',
        'sale_management',
        'stock',
        'product',
        'mail',
        'contacts',
        'web',
    ],
    'external_dependencies': {
        'python': [
            'requests',
        ],
    },
    'data': [
        # Security - MUST be loaded in this order
        'security/security_groups.xml',
        'security/ir.model.access.csv',
        # 'security/ir.rule.xml',
        
        # Data
        'data/sequences.xml',
        'data/ollama_config_data.xml',
        
        # Views
        'views/ollama_recommendation_wizard_views.xml',
        'views/product_template_views.xml',
        'views/gift_composition_views.xml',
        'views/ollama_gift_recommender_views.xml', 
        'views/menu_views_bkp.xml',
    ],
    'demo': [],
    'installable': True,
    'application': False,
    'auto_install': False,
    'license': 'LGPL-3',
}