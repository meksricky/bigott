from odoo import models, fields, api

class GiftAssemblySheet(models.Model):
    _name = 'gift.assembly.sheet'
    _description = 'Gift Assembly Sheet for Operations'
    _inherit = ['mail.thread', 'mail.activity.mixin']
    _order = 'assembly_date desc, name desc'
    
    name = fields.Char('Assembly Sheet Number', required=True, copy=False, readonly=True, default='New')
    composition_id = fields.Many2one('gift.composition', 'Gift Composition', required=True)
    partner_id = fields.Many2one('res.partner', 'Client', required=True)
    assembly_date = fields.Date('Assembly Date', default=fields.Date.today, required=True)
    
    product_count = fields.Integer('Product Count', compute='_compute_product_count', store=True)
    assembly_line_ids = fields.One2many('gift.assembly.line', 'assembly_sheet_id', 'Assembly Lines')
    
    special_instructions = fields.Text('Special Instructions')
    
    state = fields.Selection([
        ('draft', 'Draft'),
        ('ready', 'Ready for Assembly'),
        ('in_progress', 'Assembly in Progress'),
        ('completed', 'Assembly Completed'),
        ('cancelled', 'Cancelled')
    ], string='Status', default='draft', tracking=True)
    
    assigned_user_id = fields.Many2one('res.users', 'Assigned Assembler')
    start_time = fields.Datetime('Assembly Start Time')
    completion_time = fields.Datetime('Assembly Completion Time')
    
    assembly_duration = fields.Float('Assembly Duration (hours)', compute='_compute_assembly_duration', store=True)
    
    @api.depends('assembly_line_ids')
    def _compute_product_count(self):
        for sheet in self:
            sheet.product_count = len(sheet.assembly_line_ids)
    
    @api.depends('start_time', 'completion_time')
    def _compute_assembly_duration(self):
        for sheet in self:
            if sheet.start_time and sheet.completion_time:
                delta = sheet.completion_time - sheet.start_time
                sheet.assembly_duration = delta.total_seconds() / 3600.0
            else:
                sheet.assembly_duration = 0.0
    
    @api.model
    def create(self, vals):
        if vals.get('name', 'New') == 'New':
            vals['name'] = self.env['ir.sequence'].next_by_code('gift.assembly.sheet') or 'New'
        return super().create(vals)
    
    def action_start_assembly(self):
        self.write({
            'state': 'in_progress',
            'start_time': fields.Datetime.now(),
            'assigned_user_id': self.env.user.id
        })
    
    def action_complete_assembly(self):
        # Check if all items are checked
        unchecked_lines = self.assembly_line_ids.filtered(lambda l: not l.checked)
        if unchecked_lines:
            raise UserError(f"Cannot complete assembly. {len(unchecked_lines)} items still unchecked.")
        
        self.write({
            'state': 'completed',
            'completion_time': fields.Datetime.now()
        })
    
    def action_print_assembly_sheet(self):
        """Print assembly sheet"""
        return self.env.ref('bigott.action_report_assembly_sheet').report_action(self)


class GiftAssemblyLine(models.Model):
    _name = 'gift.assembly.line'
    _description = 'Gift Assembly Line Item'
    _order = 'sequence, product_id'
    
    assembly_sheet_id = fields.Many2one('gift.assembly.sheet', 'Assembly Sheet', required=True, ondelete='cascade')
    sequence = fields.Integer('Sequence', default=10)
    
    product_id = fields.Many2one('product.template', 'Product', required=True)
    location = fields.Char('Storage Location')
    special_handling = fields.Char('Special Handling Notes')
    
    checked = fields.Boolean('Checked/Assembled', default=False)
    checked_by = fields.Many2one('res.users', 'Checked By')
    checked_time = fields.Datetime('Checked Time')
    
    notes = fields.Text('Assembly Notes')
    
    @api.onchange('checked')
    def _onchange_checked(self):
        if self.checked:
            self.checked_by = self.env.user
            self.checked_time = fields.Datetime.now()
        else:
            self.checked_by = False
            self.checked_time = False


class GiftDeliveryNote(models.Model):
    _name = 'gift.delivery.note'
    _description = 'Gift Delivery Note'
    _inherit = ['mail.thread', 'mail.activity.mixin']
    _order = 'delivery_date desc, name desc'
    
    name = fields.Char('Delivery Note Number', required=True, copy=False, readonly=True, default='New')
    composition_id = fields.Many2one('gift.composition', 'Gift Composition', required=True)
    partner_id = fields.Many2one('res.partner', 'Client', required=True)
    
    delivery_date = fields.Date('Planned Delivery Date', default=fields.Date.today, required=True)
    actual_delivery_date = fields.Date('Actual Delivery Date')
    
    delivery_address = fields.Text('Delivery Address', required=True)
    special_delivery_instructions = fields.Text('Special Delivery Instructions')
    
    product_count = fields.Integer('Product Count', compute='_compute_product_count', store=True)
    delivery_line_ids = fields.One2many('gift.delivery.line', 'delivery_note_id', 'Delivery Lines')
    
    state = fields.Selection([
        ('draft', 'Draft'),
        ('ready', 'Ready for Delivery'),
        ('in_transit', 'In Transit'),
        ('delivered', 'Delivered'),
        ('failed', 'Delivery Failed'),
        ('cancelled', 'Cancelled')
    ], string='Status', default='draft', tracking=True)
    
    carrier_id = fields.Many2one('delivery.carrier', 'Delivery Carrier')
    tracking_number = fields.Char('Tracking Number')
    
    delivery_person = fields.Char('Delivery Person')
    recipient_name = fields.Char('Received By')
    delivery_signature = fields.Binary('Delivery Signature')
    delivery_photo = fields.Binary('Delivery Photo')
    
    @api.depends('delivery_line_ids')
    def _compute_product_count(self):
        for note in self:
            note.product_count = sum(note.delivery_line_ids.mapped('quantity'))
    
    @api.model
    def create(self, vals):
        if vals.get('name', 'New') == 'New':
            vals['name'] = self.env['ir.sequence'].next_by_code('gift.delivery.note') or 'New'
        return super().create(vals)
    
    def action_confirm_delivery(self):
        self.write({
            'state': 'delivered',
            'actual_delivery_date': fields.Date.today()
        })
    
    def action_print_delivery_note(self):
        """Print delivery note"""
        return self.env.ref('bigott.action_report_delivery_note').report_action(self)


class GiftDeliveryLine(models.Model):
    _name = 'gift.delivery.line'
    _description = 'Gift Delivery Line Item'
    _order = 'product_id'
    
    delivery_note_id = fields.Many2one('gift.delivery.note', 'Delivery Note', required=True, ondelete='cascade')
    product_id = fields.Many2one('product.template', 'Product', required=True)
    quantity = fields.Float('Quantity', default=1.0, required=True)
    
    temperature_requirements = fields.Char('Temperature Requirements')
    fragile = fields.Boolean('Fragile Item')
    
    delivered = fields.Boolean('Delivered', default=False)
    delivery_notes = fields.Text('Delivery Notes')


class GiftCompositionDocument(models.Model):
    _name = 'gift.composition.document'
    _description = 'Generated Documents for Gift Compositions'
    _order = 'generated_date desc'
    
    name = fields.Char('Document Name', required=True)
    composition_id = fields.Many2one('gift.composition', 'Gift Composition', required=True)
    
    document_type = fields.Selection([
        ('quotation', 'Sales Quotation'),
        ('proforma', 'Pro-forma Invoice'),
        ('assembly_sheet', 'Assembly Sheet'),
        ('delivery_note', 'Delivery Note'),
        ('labels', 'Shipping Labels'),
        ('package', 'Document Package')
    ], string='Document Type', required=True)
    
    generated_date = fields.Datetime('Generated Date', default=fields.Datetime.now)
    generated_by = fields.Many2one('res.users', 'Generated By', default=lambda self: self.env.user)
    
    # References to actual documents
    sale_order_id = fields.Many2one('sale.order', 'Related Quotation')
    invoice_id = fields.Many2one('account.move', 'Related Invoice')
    assembly_sheet_id = fields.Many2one('gift.assembly.sheet', 'Related Assembly Sheet')
    delivery_note_id = fields.Many2one('gift.delivery.note', 'Related Delivery Note')
    
    # Document content
    pdf_data = fields.Binary('PDF Document')
    pdf_filename = fields.Char('PDF Filename')
    
    state = fields.Selection([
        ('draft', 'Draft'),
        ('generated', 'Generated'),
        ('sent', 'Sent to Client'),
        ('archived', 'Archived')
    ], string='Status', default='draft')


# Extend Gift Composition model to include document tracking
class GiftComposition(models.Model):
    _inherit = 'gift.composition'
    
    # Document relationships
    document_ids = fields.One2many('gift.composition.document', 'composition_id', 'Generated Documents')
    assembly_sheet_ids = fields.One2many('gift.assembly.sheet', 'composition_id', 'Assembly Sheets')
    delivery_note_ids = fields.One2many('gift.delivery.note', 'composition_id', 'Delivery Notes')
    
    # Document status tracking
    documents_generated = fields.Boolean('Documents Generated', compute='_compute_document_status', store=True)
    assembly_completed = fields.Boolean('Assembly Completed', compute='_compute_assembly_status', store=True)
    delivery_completed = fields.Boolean('Delivery Completed', compute='_compute_delivery_status', store=True)
    
    @api.depends('document_ids')
    def _compute_document_status(self):
        for comp in self:
            required_docs = ['quotation', 'assembly_sheet', 'delivery_note']
            generated_types = comp.document_ids.mapped('document_type')
            comp.documents_generated = all(doc_type in generated_types for doc_type in required_docs)
    
    @api.depends('assembly_sheet_ids.state')
    def _compute_assembly_status(self):
        for comp in self:
            comp.assembly_completed = any(sheet.state == 'completed' for sheet in comp.assembly_sheet_ids)
    
    @api.depends('delivery_note_ids.state')
    def _compute_delivery_status(self):
        for comp in self:
            comp.delivery_completed = any(note.state == 'delivered' for note in comp.delivery_note_ids)
    
    def action_generate_documents(self):
        """Generate all required documents for this composition"""
        doc_generator = self.env['document.generation.system']
        
        try:
            documents = doc_generator.generate_all_documents(self.id)
            
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Documents Generated Successfully',
                    'message': f'Generated {len(documents)} document types for composition {self.name}',
                    'type': 'success',
                    'sticky': False,
                }
            }
        except Exception as e:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': 'Document Generation Failed',
                    'message': str(e),
                    'type': 'danger',
                    'sticky': True,
                }
            }
    
    def action_view_documents(self):
        """View all generated documents"""
        return {
            'type': 'ir.actions.act_window',
            'name': f'Documents for {self.name}',
            'res_model': 'gift.composition.document',
            'view_mode': 'tree,form',
            'domain': [('composition_id', '=', self.id)],
            'context': {'default_composition_id': self.id}
        }
    
    def action_view_assembly_sheets(self):
        """View assembly sheets"""
        return {
            'type': 'ir.actions.act_window',
            'name': f'Assembly Sheets for {self.name}',
            'res_model': 'gift.assembly.sheet',
            'view_mode': 'tree,form',
            'domain': [('composition_id', '=', self.id)],
            'context': {'default_composition_id': self.id}
        }
    
    def action_view_delivery_notes(self):
        """View delivery notes"""
        return {
            'type': 'ir.actions.act_window',
            'name': f'Delivery Notes for {self.name}',
            'res_model': 'gift.delivery.note',
            'view_mode': 'tree,form',
            'domain': [('composition_id', '=', self.id)],
            'context': {'default_composition_id': self.id}
        }