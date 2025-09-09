from odoo import models, fields, api
from odoo.exceptions import UserError
import logging

_logger = logging.getLogger(__name__)

class DocumentGenerationSystem(models.Model):
    _name = 'document.generation.system'
    _description = 'Document Generation for Gift Compositions'
    
    def generate_all_documents(self, composition_id):
        """Generate all required documents for a composition"""
        
        composition = self.env['gift.composition'].browse(composition_id)
        if not composition:
            raise UserError("Composition not found")
        
        documents = {}
        
        try:
            # 1. Generate Quotation
            documents['quotation'] = self._generate_quotation(composition)
            
            # 2. Generate Pro-forma Invoice
            documents['proforma'] = self._generate_proforma_invoice(composition)
            
            # 3. Generate Assembly/Picking Sheet
            documents['assembly_sheet'] = self._generate_assembly_sheet(composition)
            
            # 4. Generate Delivery Note
            documents['delivery_note'] = self._generate_delivery_note(composition)
            
            # 5. Generate Labels
            documents['labels'] = self._generate_labels(composition)
            
            # Update composition with document references
            composition.write({
                'document_ids': [(0, 0, {
                    'name': 'Generated Documents Package',
                    'document_type': 'package',
                    'composition_id': composition.id,
                    'generated_date': fields.Datetime.now()
                })]
            })
            
            return documents
            
        except Exception as e:
            _logger.error(f"Error generating documents for composition {composition.name}: {str(e)}")
            raise UserError(f"Document generation failed: {str(e)}")
    
    def _generate_quotation(self, composition):
        """Generate sales quotation from composition"""
        
        # Prepare quotation data
        quotation_lines = []
        for product in composition.product_ids:
            quotation_lines.append((0, 0, {
                'product_id': product.product_variant_id.id if product.product_variant_id else False,
                'product_template_id': product.id,
                'name': product.name,
                'product_uom_qty': 1,
                'price_unit': product.list_price,
                'product_uom': product.uom_id.id if product.uom_id else self.env.ref('uom.product_uom_unit').id
            }))
        
        # Create quotation
        quotation = self.env['sale.order'].create({
            'partner_id': composition.partner_id.id,
            'date_order': fields.Datetime.now(),
            'validity_date': fields.Date.add(fields.Date.today(), days=30),
            'client_order_ref': f"Gift Composition {composition.name}",
            'note': self._generate_quotation_notes(composition),
            'order_line': quotation_lines,
            'gift_composition_id': composition.id,
            'state': 'draft'
        })
        
        # Generate quotation PDF
        quotation_pdf = self._generate_document_pdf(quotation, 'sale.action_report_saleorder')
        
        return {
            'quotation_id': quotation.id,
            'quotation_name': quotation.name,
            'pdf_data': quotation_pdf,
            'total_amount': quotation.amount_total
        }
    
    def _generate_proforma_invoice(self, composition):
        """Generate pro-forma invoice"""
        
        # Create invoice from composition
        invoice_lines = []
        for product in composition.product_ids:
            invoice_lines.append((0, 0, {
                'product_id': product.product_variant_id.id if product.product_variant_id else False,
                'name': product.name,
                'quantity': 1,
                'price_unit': product.list_price,
                'product_uom_id': product.uom_id.id if product.uom_id else self.env.ref('uom.product_uom_unit').id
            }))
        
        proforma = self.env['account.move'].create({
            'move_type': 'out_invoice',
            'partner_id': composition.partner_id.id,
            'invoice_date': fields.Date.today(),
            'ref': f"Pro-forma: {composition.name}",
            'narration': f"Pro-forma invoice for gift composition {composition.name}",
            'invoice_line_ids': invoice_lines,
            'gift_composition_id': composition.id,
            'state': 'draft'
        })
        
        # Mark as pro-forma
        proforma.write({
            'name': f"PRO-{proforma.name}",
            'invoice_payment_state': 'not_paid'
        })
        
        # Generate PDF
        proforma_pdf = self._generate_document_pdf(proforma, 'account.account_invoices')
        
        return {
            'proforma_id': proforma.id,
            'proforma_name': proforma.name,
            'pdf_data': proforma_pdf,
            'total_amount': proforma.amount_total
        }
    
    def _generate_assembly_sheet(self, composition):
        """Generate assembly/picking sheet for operations"""
        
        assembly_data = {
            'composition_id': composition.id,
            'composition_name': composition.name,
            'client_name': composition.partner_id.name,
            'target_year': composition.target_year,
            'assembly_date': fields.Date.today(),
            'products': [],
            'special_instructions': self._extract_assembly_instructions(composition),
            'dietary_restrictions': composition.dietary_restrictions,
            'total_items': len(composition.product_ids)
        }
        
        # Prepare product list with assembly details
        for product in composition.product_ids:
            assembly_data['products'].append({
                'name': product.name,
                'category': product.lebiggot_category,
                'brand': product.brand or 'N/A',
                'grade': product.product_grade or 'standard',
                'location': self._get_product_location(product),
                'special_handling': self._get_special_handling_notes(product),
                'checked': False
            })
        
        # Generate assembly sheet document
        assembly_sheet = self.env['gift.assembly.sheet'].create({
            'name': f"Assembly-{composition.name}",
            'composition_id': composition.id,
            'partner_id': composition.partner_id.id,
            'assembly_date': fields.Date.today(),
            'product_count': len(composition.product_ids),
            'special_instructions': assembly_data['special_instructions'],
            'state': 'draft'
        })
        
        # Create assembly line items
        for product_data in assembly_data['products']:
            self.env['gift.assembly.line'].create({
                'assembly_sheet_id': assembly_sheet.id,
                'product_id': next(p.id for p in composition.product_ids if p.name == product_data['name']),
                'location': product_data['location'],
                'special_handling': product_data['special_handling'],
                'checked': False
            })
        
        assembly_pdf = self._generate_assembly_sheet_pdf(assembly_sheet)
        
        return {
            'assembly_sheet_id': assembly_sheet.id,
            'assembly_name': assembly_sheet.name,
            'pdf_data': assembly_pdf,
            'product_count': assembly_data['total_items']
        }
    
    def _generate_delivery_note(self, composition):
        """Generate delivery note"""
        
        delivery_note = self.env['gift.delivery.note'].create({
            'name': f"Delivery-{composition.name}",
            'composition_id': composition.id,
            'partner_id': composition.partner_id.id,
            'delivery_date': fields.Date.today(),
            'delivery_address': self._format_delivery_address(composition.partner_id),
            'special_delivery_instructions': self._get_delivery_instructions(composition),
            'product_count': len(composition.product_ids),
            'state': 'draft'
        })
        
        # Create delivery line items
        for product in composition.product_ids:
            self.env['gift.delivery.line'].create({
                'delivery_note_id': delivery_note.id,
                'product_id': product.id,
                'quantity': 1,
                'temperature_requirements': self._get_temperature_requirements(product),
                'fragile': self._is_fragile_product(product)
            })
        
        delivery_pdf = self._generate_delivery_note_pdf(delivery_note)
        
        return {
            'delivery_note_id': delivery_note.id,
            'delivery_name': delivery_note.name,
            'pdf_data': delivery_pdf,
            'delivery_address': delivery_note.delivery_address
        }
    
    def _generate_labels(self, composition):
        """Generate shipping and product labels"""
        
        labels = []
        
        # 1. Main shipping label
        shipping_label = {
            'type': 'shipping',
            'composition_name': composition.name,
            'client_name': composition.partner_id.name,
            'delivery_address': self._format_delivery_address(composition.partner_id),
            'product_count': len(composition.product_ids),
            'weight_estimate': self._calculate_total_weight(composition.product_ids),
            'special_handling': self._get_package_handling_requirements(composition)
        }
        labels.append(shipping_label)
        
        # 2. Individual product labels
        for product in composition.product_ids:
            product_label = {
                'type': 'product',
                'product_name': product.name,
                'brand': product.brand or '',
                'category': product.lebiggot_category,
                'barcode': product.barcode or '',
                'temperature_req': self._get_temperature_requirements(product),
                'fragile': self._is_fragile_product(product),
                'composition_ref': composition.name
            }
            labels.append(product_label)
        
        # 3. Gift composition summary label
        summary_label = {
            'type': 'summary',
            'composition_name': composition.name,
            'client_tier': getattr(composition.partner_id, 'client_tier', 'standard'),
            'year': composition.target_year,
            'total_value': composition.actual_cost,
            'dietary_restrictions': composition.dietary_restrictions or 'None',
            'assembly_date': fields.Date.today()
        }
        labels.append(summary_label)
        
        # Generate label PDFs
        label_pdfs = []
        for label in labels:
            pdf_data = self._generate_label_pdf(label)
            label_pdfs.append({
                'type': label['type'],
                'pdf_data': pdf_data,
                'label_data': label
            })
        
        return {
            'labels': labels,
            'label_pdfs': label_pdfs,
            'total_labels': len(labels)
        }
    
    # Helper methods for document generation
    
    def _generate_quotation_notes(self, composition):
        """Generate notes for quotation"""
        notes = [
            f"Gift composition curated by Señor Bigott AI for year {composition.target_year}",
            f"Composition confidence: {composition.confidence_score:.1%}",
        ]
        
        if composition.dietary_restrictions:
            notes.append(f"Dietary requirements: {composition.dietary_restrictions}")
        
        if composition.reasoning:
            notes.append("AI selection rationale available upon request")
        
        return "\n".join(notes)
    
    def _extract_assembly_instructions(self, composition):
        """Extract special assembly instructions"""
        instructions = []
        
        # Check for fragile items
        fragile_items = [p for p in composition.product_ids if self._is_fragile_product(p)]
        if fragile_items:
            instructions.append(f"FRAGILE: Handle {len(fragile_items)} items with extra care")
        
        # Check for temperature requirements
        cold_items = [p for p in composition.product_ids if self._requires_cold_storage(p)]
        if cold_items:
            instructions.append(f"COLD STORAGE: {len(cold_items)} items require refrigeration")
        
        # Check dietary restrictions
        if composition.dietary_restrictions:
            instructions.append(f"DIETARY: Composition is {composition.dietary_restrictions} compliant")
        
        return " | ".join(instructions)
    
    def _get_product_location(self, product):
        """Get warehouse location for product"""
        # This would integrate with actual warehouse management
        category_locations = {
            'main_beverage': 'Wine Cellar A1-A4',
            'aperitif': 'Wine Cellar B1-B2', 
            'foie_gras': 'Cold Storage C1',
            'charcuterie': 'Cold Storage C2-C4',
            'sweets': 'Dry Storage D1-D3',
            'experience_gastronomica': 'Special Items E1'
        }
        
        return category_locations.get(product.lebiggot_category, 'General Storage')
    
    def _get_special_handling_notes(self, product):
        """Get special handling requirements for product"""
        notes = []
        
        if self._is_fragile_product(product):
            notes.append("FRAGILE")
        
        if self._requires_cold_storage(product):
            notes.append("KEEP COLD")
        
        if product.lebiggot_category == 'main_beverage':
            notes.append("UPRIGHT")
        
        return " | ".join(notes)
    
    def _is_fragile_product(self, product):
        """Check if product is fragile"""
        fragile_categories = ['main_beverage', 'aperitif']
        return product.lebiggot_category in fragile_categories
    
    def _requires_cold_storage(self, product):
        """Check if product requires cold storage"""
        cold_categories = ['foie_gras', 'charcuterie']
        return product.lebiggot_category in cold_categories
    
    def _get_temperature_requirements(self, product):
        """Get temperature requirements for product"""
        if product.lebiggot_category in ['foie_gras', 'charcuterie']:
            return '2-4°C'
        elif product.lebiggot_category in ['main_beverage', 'aperitif']:
            return '12-16°C'
        else:
            return 'Room Temperature'
    
    def _format_delivery_address(self, partner):
        """Format delivery address"""
        address_parts = []
        
        if partner.street:
            address_parts.append(partner.street)
        if partner.street2:
            address_parts.append(partner.street2)
        if partner.city:
            address_parts.append(partner.city)
        if partner.zip:
            address_parts.append(partner.zip)
        if partner.country_id:
            address_parts.append(partner.country_id.name)
        
        return "\n".join(address_parts)
    
    def _get_delivery_instructions(self, composition):
        """Get special delivery instructions"""
        instructions = []
        
        # Check client tier for special handling
        client_tier = getattr(composition.partner_id, 'client_tier', 'standard')
        if client_tier == 'vip':
            instructions.append("VIP CLIENT - Priority handling")
        
        # Check for fragile items
        fragile_count = len([p for p in composition.product_ids if self._is_fragile_product(p)])
        if fragile_count > 0:
            instructions.append(f"{fragile_count} fragile items - Handle with care")
        
        # Temperature requirements
        cold_count = len([p for p in composition.product_ids if self._requires_cold_storage(p)])
        if cold_count > 0:
            instructions.append(f"{cold_count} items require cold chain")
        
        return " | ".join(instructions)
    
    def _calculate_total_weight(self, products):
        """Calculate estimated total weight"""
        # This would use actual product weights
        category_weights = {
            'main_beverage': 1.2,  # kg per bottle
            'aperitif': 0.8,
            'foie_gras': 0.3,
            'charcuterie': 0.5,
            'sweets': 0.3,
            'experience_gastronomica': 0.4
        }
        
        total_weight = 0
        for product in products:
            weight = category_weights.get(product.lebiggot_category, 0.5)
            total_weight += weight
        
        return f"{total_weight:.1f} kg"
    
    def _get_package_handling_requirements(self, composition):
        """Get package-level handling requirements"""
        requirements = []
        
        if any(self._is_fragile_product(p) for p in composition.product_ids):
            requirements.append("FRAGILE")
        
        if any(self._requires_cold_storage(p) for p in composition.product_ids):
            requirements.append("REFRIGERATED")
        
        requirements.append("THIS SIDE UP")
        
        return " | ".join(requirements)
    
    # PDF generation methods (simplified - would use proper report templates)
    
    def _generate_document_pdf(self, record, report_action):
        """Generate PDF for a document using Odoo's reporting system"""
        try:
            # This would use Odoo's actual report generation
            return f"PDF data for {record._name} {record.id}"
        except Exception as e:
            _logger.error(f"PDF generation failed: {str(e)}")
            return None
    
    def _generate_assembly_sheet_pdf(self, assembly_sheet):
        """Generate assembly sheet PDF"""
        return f"Assembly sheet PDF for {assembly_sheet.name}"
    
    def _generate_delivery_note_pdf(self, delivery_note):
        """Generate delivery note PDF"""
        return f"Delivery note PDF for {delivery_note.name}"
    
    def _generate_label_pdf(self, label_data):
        """Generate label PDF"""
        return f"Label PDF for {label_data['type']} label"