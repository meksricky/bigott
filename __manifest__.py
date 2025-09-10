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