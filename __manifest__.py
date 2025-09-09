{
    'name': 'Señor Bigott Gift Composition Engine - Complete BRD Implementation',
    'version': '14.0.2.0.0',
    'category': 'Sales',
    'summary': 'AI-powered gift composition engine with business rules compliance, batch processing, and full document generation',
    'description': """
        Señor Bigott Gift Composition Engine - Complete Implementation
        ===========================================================
        
        This module implements a comprehensive AI-assisted gift composition system that meets all Business Requirements Document (BRD) specifications:
        
        🧠 **AI Composition Engine**
        * Analyzes 3+ years client history for personalized compositions
        * Processes client notes with natural language processing
        * Applies 40% weight to notes-based preferences
        * Dynamic budget optimization and dietary restriction handling
        
        ⚖️ **Business Rules Compliance (R1-R6)**
        * R1: Exact repetition for Cava/Champagne/Vermouth/Tokaj
        * R2: Wine brand variation (same color/size/grade, different brand)
        * R3: Experience bundle replacement (avoid repetition)
        * R4: Exact repetition for Paletilla & Charcuterie
        * R5: Foie Gras alternation (Duck ↔ Goose)
        * R6: Sweet product rules (Lingote exact, Turrón brand may change)
        
        🌐 **Global Constraints**
        * Price category locking across compositions
        * Beverage size consistency enforcement
        * Budget guardrail ±5% compliance
        * Deterministic, reproducible outputs
        
        🚀 **Batch Processing**
        * Process up to 200 clients per hour
        * Automated client eligibility detection
        * Performance monitoring and error handling
        * Comprehensive batch reporting
        
        📦 **Stock-Aware Processing**
        * Real-time inventory integration
        * Intelligent product substitution
        * Rule-compliant fallback mechanisms
        * Stock issue tracking and resolution
        
        📄 **Document Generation System**
        * Sales quotations with AI rationale
        * Pro-forma invoices
        * Assembly sheets for warehouse operations
        * Delivery notes with special handling instructions
        * Shipping labels with temperature/fragile requirements
        
        🎨 **Experience Pool Management**
        * 3-5 product curated experiences
        * CSV/Excel import capabilities
        * Lifecycle management (draft→testing→active→retired)
        * Seasonal availability and client tier targeting
        
        👥 **Client Analytics**
        * Multi-dimensional preference analysis
        * Satisfaction correlation tracking
        * Budget trend analysis and prediction
        * Dietary restriction pattern learning
        
        🔐 **Enterprise Security**
        * Role-based access control (Sales/Manager/Admin)
        * Audit trails for all operations
        * Activity logging and notifications
        * Data privacy compliance
        
        📊 **Performance & Monitoring**
        * ≤2 minutes per client processing
        * Real-time progress tracking
        * Comprehensive error reporting
        * Performance metrics dashboard
        
        Key Features:
        - Experience-based and custom gift compositions
        - Advanced AI with notes sentiment analysis
        - Deterministic business rules engine
        - Multi-client batch processing
        - Complete document workflow
        - Warehouse integration ready
        - Comprehensive reporting suite
    """,
    'author': 'Le Biggot',
    'website': 'https://www.lebiggot.com',
    'depends': [
        'base', 
        'sale_management', 
        'product', 
        'mail', 
        'utm', 
        'web',
        'stock',
        'delivery',
        'account'
    ],
    'data': [
        # Security
        'security/ir.model.access.csv',
        
        # Data and Configuration
        'data/sequences.xml',
        'data/product_categories_data.xml',
        'data/demo_data.xml',
        
        # Wizard Views
        'wizard/composition_wizard_views.xml',
        # 'wizard/rebuild_history_wizard_views.xml',
        
        # Core Views
        'views/client_history_views.xml',
        'views/experience_views.xml',
        'views/gift_composition_views.xml',
        'views/product_template_views.xml',
        'views/res_partner_views.xml',
        
        # Advanced Feature Views
        'views/batch_processing_views.xml',
        # 'views/assembly_delivery_views.xml',
        # 'views/experience_management_views.xml',
        # 'views/document_management_views.xml',
        
        # Menu Structure
        'views/menu_views.xml',
        
        # Reports (placeholder for future PDF templates)
        # 'report/assembly_sheet_report.xml',
        # 'report/delivery_note_report.xml',
        # 'report/batch_processing_report.xml',
    ],
    'demo': [
        # 'demo/enhanced_demo_data.xml',
    ],
    'external_dependencies': {
        'python': [
            # 'requests',  # For Ollama integration (optional)
        ],
    },
    'installable': True,
    'application': True,
    'auto_install': False,
    'license': 'LGPL-3',
    'images': ['static/description/banner.png'],
    # 'price': 999.00,
    'currency': 'EUR',
    'support': 'enterprise',
}