{
    'name': 'Señor Bigott Gift Composition Engine',
    'version': '14.0.1.0.0',
    'category': 'Sales',
    'summary': 'AI-powered gift composition engine for Le Biggot gourmet experiences',
    'description': """
        Señor Bigott Gift Composition Engine
        ===================================
        * Analyzes 3-year client history for personalized compositions
        * Manages Experience-based and Custom box logic
        * Prevents experience repetition per client
        * Adapts to budget changes with upgrade/downgrade logic
        * Maintains category balance and quantity rules
        * Handles dietary restrictions (vegan, halal, non-alcoholic)
        
        Features:
        - Experience-based gift boxes (3-5 curated products)
        - Custom compositions based on client preferences
        - AI budget adaptation (upgrade/maintain/downgrade)
        - Historical pattern learning and analysis
        - Category structure preservation across years
        - Comprehensive dietary restriction management
    """,
    'author': 'Le Biggot',
    'website': 'https://www.lebiggot.com',
    'depends': ['base', 'sale_management', 'product', 'mail', 'utm', 'web'],
    'data': [
        'security/ir.model.access.csv',
        'data/sequences.xml',
        'data/product_categories_data.xml',
        'data/demo_data.xml',
        'wizard/composition_wizard_views.xml',        
        'views/client_history_views.xml',
        'views/experience_views.xml',
        'views/gift_composition_views.xml',
        'views/product_template_views.xml',
        'views/res_partner_views.xml',
        'views/menu_views.xml',
    ],
    # 'demo': [
    #     'demo/additional_demo_data.xml',
    # ],
    'installable': True,
    'application': True,
    'auto_install': False,
    'license': 'LGPL-3',
}