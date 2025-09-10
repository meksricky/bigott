# -*- coding: utf-8 -*-
{
    'name': 'Señor Bigott AI/ML Gift Recommendation System',
    'version': '3.0.0',
    'category': 'Sales/Sales',
    'summary': 'Advanced AI/ML-powered luxury gift recommendation and composition system',
    'description': """
Señor Bigott AI/ML Gift Recommendation System
==============================================

Complete AI/ML-powered system for luxury gourmet gift recommendations.
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
        'delivery',
    ],
    'external_dependencies': {
        'python': [
            'numpy',
            'scikit-learn',
            'joblib',
            'requests',
        ],
    },
    'data': [
        # Security
        'security/security_groups.xml',
        'security/ir.model.access.csv',
        
        # Data files
        'data/ir_config_parameter_data.xml',
        'data/ir_cron_data.xml',
        # 'data/cron_jobs.xml',
        'data/default_data.xml',
        'data/sequences.xml',
        # 'data/system_config.xml',
        'data/product_categories_data.xml',
        
        # Views - Core
        'views/product_template_views.xml',
        'views/res_partner_views.xml',
        'views/gift_composition_views.xml',
        'views/client_history_views.xml',
        'views/experience_views.xml',
        # 'views/batch_processing_views.xml',
        
        # Views - Engines
        # 'views/ml_engine_views.xml',
        'views/ai_product_recommender_views.xml',
        'views/integration_manager_views.xml',
        # 'views/stock_aware_composition_views.xml',
        # 'views/business_rules_engine_views.xml',
        'views/composition_engine_views.xml',
        'views/document_generation_views.xml',
        
        # Views - Rebuilding history
        'views/rebuild_history_wizard_views.xml',
        
        # Menu
        'views/menu_views.xml',
        
        # Wizards
        'wizard/ml_training_wizard_views.xml',
        'wizard/batch_composition_wizard_views.xml',
        'wizard/rebuild_history_wizard_views.xml',
        'wizard/composition_wizard.py',
        
    ],
    'demo': [
        # 'demo/additional_demo_data.xml',
    ],
    'assets': {
        'web.assets_backend': [
            'lebigott_ai/static/src/css/gift_composition.css',
            'lebigott_ai/static/src/js/gift_composition_widget.js',
            'lebigott_ai/static/src/js/ml_dashboard.js',
        ],
    },
    'installable': True,
    'application': True,
    'auto_install': False,
    'license': 'LGPL-3',
}