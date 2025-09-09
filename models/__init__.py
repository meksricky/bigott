# models/__init__.py - CORRECTED VERSION
# Load models in proper dependency order

# 1. Base models (no dependencies)
from . import res_partner
from . import product_template

# 2. Core business models  
from . import gift_experience
from . import client_order_history

# 3. Engine models (depend on base models)
from . import business_rules_engine
from . import composition_engine

# 4. Advanced engines (depend on core engines)
from . import stock_aware_composition_engine

# 5. Document and processing models (depend on compositions)
from . import gift_composition
from . import document_generation_system
from . import assembly_delivery_models  # This includes multiple models
from . import batch_composition_processor
from . import gift_experience_pool

# 6. Integration layer (depends on all)
from . import integration_manager

# 7. MISSING IMPORTS - Add these if files exist:
# from . import gift_experience_pool  # Contains experience.pool.importer
# from . import wizard_models  # If you have separate wizard files