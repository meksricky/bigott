from odoo import models, api, fields
import logging

_logger = logging.getLogger(__name__)

class IntegrationManager(models.Model):
    _name = 'integration.manager'
    _description = 'Central Integration Manager for All Engines'
    
    @api.model
    def generate_complete_composition(self, partner_id, target_budget, target_year=None, 
                                    dietary_restrictions=None, notes_text=None, use_batch=False):
        """
        Central method that orchestrates all engines for complete composition generation
        """
        
        try:
            # 1. Use stock-aware engine for compliant composition
            stock_engine = self.env['stock.aware.composition.engine']
            composition = stock_engine.generate_compliant_composition(
                partner_id=partner_id,
                target_budget=target_budget,
                target_year=target_year,
                dietary_restrictions=dietary_restrictions,
                notes_text=notes_text
            )
            
            # 2. Auto-generate all required documents
            if composition and not use_batch:
                doc_generator = self.env['document.generation.system']
                try:
                    documents = doc_generator.generate_all_documents(composition.id)
                    _logger.info(f"Generated {len(documents)} document types for composition {composition.name}")
                except Exception as e:
                    _logger.warning(f"Document generation failed for {composition.name}: {str(e)}")
            
            return composition
            
        except Exception as e:
            _logger.error(f"Complete composition generation failed: {str(e)}")
            raise
    
    @api.model
    def validate_system_integrity(self):
        """Validate that all system components are properly configured"""
        
        validation_results = {
            'engines_available': True,
            'sequences_configured': True,
            'categories_configured': True,
            'experiences_available': True,
            'errors': []
        }
        
        # Check engines
        try:
            self.env['composition.engine']
            self.env['business.rules.engine'] 
            self.env['stock.aware.composition.engine']
            self.env['document.generation.system']
        except Exception as e:
            validation_results['engines_available'] = False
            validation_results['errors'].append(f"Engine availability: {str(e)}")
        
        # Check sequences
        sequences = [
            'gift.composition.sequence',
            'gift.experience.code',
            'batch.composition.processor'
        ]
        
        for seq_code in sequences:
            if not self.env['ir.sequence'].search([('code', '=', seq_code)]):
                validation_results['sequences_configured'] = False
                validation_results['errors'].append(f"Missing sequence: {seq_code}")
        
        # Check product categories
        categories = [
            'main_beverage', 'aperitif', 'experience_gastronomica',
            'foie_gras', 'charcuterie', 'sweets'
        ]
        
        for cat in categories:
            if not self.env['product.category'].search([('name', '=', cat.replace('_', ' ').title())]):
                validation_results['categories_configured'] = False
                validation_results['errors'].append(f"Missing category: {cat}")
        
        # Check experiences
        if not self.env['gift.experience'].search([('active', '=', True)]):
            validation_results['experiences_available'] = False
            validation_results['errors'].append("No active experiences available")
        
        return validation_results