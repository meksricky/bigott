from odoo import models, fields, api
from odoo.exceptions import UserError
import logging
from datetime import datetime
import threading
import queue
import time

_logger = logging.getLogger(__name__)

class BatchCompositionProcessor(models.Model):
    _name = 'batch.composition.processor'
    _description = 'Batch Processing for Multiple Client Compositions'
    _inherit = ['mail.thread', 'mail.activity.mixin']
    
    name = fields.Char('Batch Name', required=True, default='New Batch', tracking=True)
    batch_date = fields.Datetime('Batch Date', default=fields.Datetime.now, required=True)
    target_year = fields.Integer('Target Year', required=True, default=lambda self: fields.Date.today().year)
    
    # Batch configuration
    client_ids = fields.Many2many('res.partner', string='Clients to Process', 
                                  domain="[('is_company', '=', False)]")
    client_count = fields.Integer('Client Count', compute='_compute_client_count', store=True)
    
    default_budget = fields.Float('Default Budget (€)', default=200.0,
                                  help="Default budget for clients without historical data")
    budget_adjustment_factor = fields.Float('Budget Adjustment Factor', default=1.1,
                                           help="Multiply last year's budget by this factor")
    
    # Processing status
    state = fields.Selection([
        ('draft', 'Draft'),
        ('ready', 'Ready to Process'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled')
    ], string='Status', default='draft', tracking=True)
    
    # Results tracking
    total_clients = fields.Integer('Total Clients', compute='_compute_processing_stats', store=True)
    processed_clients = fields.Integer('Processed Clients', compute='_compute_processing_stats', store=True)
    successful_compositions = fields.Integer('Successful Compositions', compute='_compute_processing_stats', store=True)
    failed_compositions = fields.Integer('Failed Compositions', compute='_compute_processing_stats', store=True)
    
    processing_start_time = fields.Datetime('Processing Start Time')
    processing_end_time = fields.Datetime('Processing End Time')
    processing_duration = fields.Float('Processing Duration (minutes)', compute='_compute_processing_duration', store=True)
    
    # Performance metrics
    average_processing_time = fields.Float('Average Time per Client (seconds)', compute='_compute_performance_metrics', store=True)
    compositions_per_hour = fields.Float('Compositions per Hour', compute='_compute_performance_metrics', store=True)
    
    # Results
    composition_ids = fields.One2many('gift.composition', 'batch_processor_id', 'Generated Compositions')
    processing_log_ids = fields.One2many('batch.processing.log', 'batch_processor_id', 'Processing Logs')
    
    # Error handling
    error_summary = fields.Text('Error Summary', compute='_compute_error_summary')
    retry_failed = fields.Boolean('Retry Failed Clients', default=False)
    
    @api.depends('client_ids')
    def _compute_client_count(self):
        for batch in self:
            batch.client_count = len(batch.client_ids)
    
    @api.depends('composition_ids', 'processing_log_ids')
    def _compute_processing_stats(self):
        for batch in self:
            batch.total_clients = len(batch.client_ids)
            batch.processed_clients = len(batch.processing_log_ids)
            batch.successful_compositions = len(batch.composition_ids.filtered(lambda c: c.state != 'cancelled'))
            batch.failed_compositions = len(batch.processing_log_ids.filtered(lambda l: l.status == 'failed'))
    
    @api.depends('processing_start_time', 'processing_end_time')
    def _compute_processing_duration(self):
        for batch in self:
            if batch.processing_start_time and batch.processing_end_time:
                delta = batch.processing_end_time - batch.processing_start_time
                batch.processing_duration = delta.total_seconds() / 60.0
            else:
                batch.processing_duration = 0.0
    
    @api.depends('processing_duration', 'processed_clients')
    def _compute_performance_metrics(self):
        for batch in self:
            if batch.processing_duration > 0 and batch.processed_clients > 0:
                batch.average_processing_time = (batch.processing_duration * 60) / batch.processed_clients
                batch.compositions_per_hour = batch.processed_clients / (batch.processing_duration / 60.0)
            else:
                batch.average_processing_time = 0.0
                batch.compositions_per_hour = 0.0
    
    @api.depends('processing_log_ids.error_message')
    def _compute_error_summary(self):
        for batch in self:
            errors = batch.processing_log_ids.filtered(lambda l: l.status == 'failed')
            if errors:
                error_messages = [f"• {log.client_id.name}: {log.error_message}" for log in errors]
                batch.error_summary = "\n".join(error_messages)
            else:
                batch.error_summary = "No errors recorded"
    
    @api.model
    def create(self, vals):
        if vals.get('name', 'New Batch') == 'New Batch':
            vals['name'] = self.env['ir.sequence'].next_by_code('batch.composition.processor') or 'New Batch'
        return super().create(vals)
    
    def action_add_all_eligible_clients(self):
        """Add all eligible clients to the batch"""
        
        # Find clients with gift history
        clients_with_history = self.env['res.partner'].search([
            ('is_company', '=', False),
            ('order_history_ids', '!=', False)
        ])
        
        # Find clients who haven't received composition for target year
        existing_compositions = self.env['gift.composition'].search([
            ('target_year', '=', self.target_year),
            ('partner_id', 'in', clients_with_history.ids)
        ])
        
        clients_needing_composition = clients_with_history.filtered(
            lambda c: c.id not in existing_compositions.mapped('partner_id').ids
        )
        
        self.client_ids = [(6, 0, clients_needing_composition.ids)]
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Clients Added',
                'message': f'Added {len(clients_needing_composition)} eligible clients to batch',
                'type': 'success',
                'sticky': False,
            }
        }
    
    def action_start_batch_processing(self):
        """Start batch processing of all clients"""
        
        if not self.client_ids:
            raise UserError("No clients selected for processing")
        
        if self.state != 'ready':
            self.state = 'ready'
        
        # Validate batch before processing
        validation_result = self._validate_batch()
        if not validation_result['valid']:
            raise UserError(f"Batch validation failed: {validation_result['error']}")
        
        # Start processing
        self.write({
            'state': 'processing',
            'processing_start_time': fields.Datetime.now()
        })
        
        # Process in background using queue jobs if available, otherwise synchronous
        try:
            if hasattr(self.env, 'queue_job'):
                # Use queue_job if available
                self.with_delay()._process_batch_async()
            else:
                # Process synchronously
                self._process_batch_sync()
        except Exception as e:
            _logger.error(f"Batch processing failed: {str(e)}")
            self.write({
                'state': 'failed',
                'processing_end_time': fields.Datetime.now()
            })
            raise
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Batch Processing Started',
                'message': f'Processing {len(self.client_ids)} clients for year {self.target_year}',
                'type': 'info',
                'sticky': False,
            }
        }
    
    def _validate_batch(self):
        """Validate batch configuration before processing"""
        
        # Check if target year is reasonable
        current_year = fields.Date.today().year
        if self.target_year < current_year or self.target_year > current_year + 2:
            return {'valid': False, 'error': f'Target year {self.target_year} is not in reasonable range'}
        
        # Check if clients exist
        if not self.client_ids:
            return {'valid': False, 'error': 'No clients selected'}
        
        # Check for existing compositions
        existing_comps = self.env['gift.composition'].search([
            ('partner_id', 'in', self.client_ids.ids),
            ('target_year', '=', self.target_year)
        ])
        
        if existing_comps and not self.retry_failed:
            return {'valid': False, 'error': f'{len(existing_comps)} clients already have compositions for {self.target_year}'}
        
        # Check budget parameters
        if self.default_budget <= 0:
            return {'valid': False, 'error': 'Default budget must be positive'}
        
        if self.budget_adjustment_factor <= 0:
            return {'valid': False, 'error': 'Budget adjustment factor must be positive'}
        
        return {'valid': True}
    
    def _process_batch_sync(self):
        """Process batch synchronously"""
        
        successful_count = 0
        failed_count = 0
        
        for i, client in enumerate(self.client_ids):
            try:
                _logger.info(f"Processing client {i+1}/{len(self.client_ids)}: {client.name}")
                
                # Calculate target budget for this client
                target_budget = self._calculate_client_budget(client)
                
                # Generate composition
                composition = self._generate_single_composition(client, target_budget)
                
                if composition:
                    # Log successful processing
                    self._create_processing_log(client, 'success', composition)
                    successful_count += 1
                else:
                    # Log failed processing
                    self._create_processing_log(client, 'failed', None, "Composition generation returned None")
                    failed_count += 1
                
                # Commit after each client to avoid losing progress
                self.env.cr.commit()
                
            except Exception as e:
                _logger.error(f"Failed to process client {client.name}: {str(e)}")
                self._create_processing_log(client, 'failed', None, str(e))
                failed_count += 1
                
                # Continue with next client
                continue
        
        # Mark batch as completed
        self.write({
            'state': 'completed' if failed_count == 0 else 'completed',  # Always mark as completed
            'processing_end_time': fields.Datetime.now()
        })
        
        _logger.info(f"Batch processing completed. Success: {successful_count}, Failed: {failed_count}")
    
    def _process_batch_async(self):
        """Process batch asynchronously using queue jobs"""
        
        # This would be implemented with queue_job module
        # For now, fall back to synchronous processing
        self._process_batch_sync()
    
    def _calculate_client_budget(self, client):
        """Calculate target budget for a client"""
        
        # Try to get historical budget
        last_history = self.env['client.order.history'].search([
            ('partner_id', '=', client.id)
        ], order='order_year desc', limit=1)
        
        if last_history:
            # Use adjusted historical budget
            return last_history.total_budget * self.budget_adjustment_factor
        else:
            # Use default budget
            return self.default_budget
    
    def _generate_single_composition(self, client, target_budget):
        """Generate composition for a single client"""
        
        # Use the stock-aware composition engine
        engine = self.env['stock.aware.composition.engine']
        
        composition = engine.generate_compliant_composition(
            partner_id=client.id,
            target_budget=target_budget,
            target_year=self.target_year,
            dietary_restrictions=None,  # Could be enhanced to read from client profile
            force_type=None,
            notes_text=getattr(client, 'client_notes', None)
        )
        
        # Link composition to this batch
        if composition:
            composition.batch_processor_id = self.id
        
        return composition
    
    def _create_processing_log(self, client, status, composition=None, error_message=None):
        """Create processing log entry"""
        
        log_vals = {
            'batch_processor_id': self.id,
            'client_id': client.id,
            'status': status,
            'processing_time': fields.Datetime.now(),
            'composition_id': composition.id if composition else None,
            'error_message': error_message
        }
        
        return self.env['batch.processing.log'].create(log_vals)
    
    def action_retry_failed_clients(self):
        """Retry processing for failed clients"""
        
        failed_logs = self.processing_log_ids.filtered(lambda l: l.status == 'failed')
        if not failed_logs:
            raise UserError("No failed clients to retry")
        
        failed_clients = failed_logs.mapped('client_id')
        
        # Create new batch for retry
        retry_batch = self.copy({
            'name': f"{self.name} - Retry",
            'client_ids': [(6, 0, failed_clients.ids)],
            'retry_failed': True,
            'state': 'draft'
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': f'Retry Batch: {retry_batch.name}',
            'res_model': 'batch.composition.processor',
            'res_id': retry_batch.id,
            'view_mode': 'form',
            'target': 'current'
        }
    
    def action_view_results(self):
        """View batch processing results"""
        return {
            'type': 'ir.actions.act_window',
            'name': f'Batch Results: {self.name}',
            'res_model': 'gift.composition',
            'view_mode': 'tree,form',
            'domain': [('batch_processor_id', '=', self.id)],
            'context': {'default_batch_processor_id': self.id}
        }
    
    def action_generate_batch_report(self):
        """Generate comprehensive batch processing report"""
        
        report_data = {
            'batch_name': self.name,
            'processing_date': self.batch_date,
            'target_year': self.target_year,
            'total_clients': self.total_clients,
            'successful_compositions': self.successful_compositions,
            'failed_compositions': self.failed_compositions,
            'processing_duration': self.processing_duration,
            'average_processing_time': self.average_processing_time,
            'compositions_per_hour': self.compositions_per_hour,
            'total_budget': sum(self.composition_ids.mapped('target_budget')),
            'total_actual_cost': sum(self.composition_ids.mapped('actual_cost')),
            'error_summary': self.error_summary
        }
        
        # This would generate a PDF report
        return self.env.ref('bigott.action_report_batch_processing').report_action(self)


class BatchProcessingLog(models.Model):
    _name = 'batch.processing.log'
    _description = 'Batch Processing Log Entry'
    _order = 'processing_time desc'
    
    batch_processor_id = fields.Many2one('batch.composition.processor', 'Batch Processor', required=True, ondelete='cascade')
    client_id = fields.Many2one('res.partner', 'Client', required=True)
    
    status = fields.Selection([
        ('success', 'Success'),
        ('failed', 'Failed'),
        ('skipped', 'Skipped')
    ], string='Status', required=True)
    
    processing_time = fields.Datetime('Processing Time', required=True)
    composition_id = fields.Many2one('gift.composition', 'Generated Composition')
    
    error_message = fields.Text('Error Message')
    processing_duration = fields.Float('Processing Duration (seconds)')
    
    notes = fields.Text('Additional Notes')


# Extend Gift Composition to track batch processing
class GiftComposition(models.Model):
    _inherit = 'gift.composition'
    
    batch_processor_id = fields.Many2one('batch.composition.processor', 'Batch Processor', readonly=True)
    is_batch_generated = fields.Boolean('Batch Generated', compute='_compute_is_batch_generated', store=True)
    
    @api.depends('batch_processor_id')
    def _compute_is_batch_generated(self):
        for comp in self:
            comp.is_batch_generated = bool(comp.batch_processor_id)


class BatchWizard(models.TransientModel):
    _name = 'batch.wizard'
    _description = 'Batch Processing Wizard'
    
    target_year = fields.Integer('Target Year', required=True, default=lambda self: fields.Date.today().year)
    
    client_selection = fields.Selection([
        ('all_eligible', 'All Eligible Clients'),
        ('by_tier', 'By Client Tier'),
        ('manual', 'Manual Selection')
    ], string='Client Selection', default='all_eligible', required=True)
    
    client_tier = fields.Selection([
        ('new', 'New Clients'),
        ('regular', 'Regular Clients'),
        ('premium', 'Premium Clients'),
        ('vip', 'VIP Clients')
    ], string='Client Tier')
    
    selected_client_ids = fields.Many2many('res.partner', string='Selected Clients')
    
    default_budget = fields.Float('Default Budget (€)', default=200.0)
    budget_adjustment_factor = fields.Float('Budget Adjustment Factor', default=1.1)
    
    def action_create_batch(self):
        """Create batch processor with selected clients"""
        
        # Determine client list
        if self.client_selection == 'all_eligible':
            clients = self._get_all_eligible_clients()
        elif self.client_selection == 'by_tier':
            clients = self._get_clients_by_tier()
        else:
            clients = self.selected_client_ids
        
        if not clients:
            raise UserError("No eligible clients found")
        
        # Create batch processor
        batch = self.env['batch.composition.processor'].create({
            'name': f"Batch {self.target_year} - {len(clients)} clients",
            'target_year': self.target_year,
            'client_ids': [(6, 0, clients.ids)],
            'default_budget': self.default_budget,
            'budget_adjustment_factor': self.budget_adjustment_factor,
            'state': 'ready'
        })
        
        return {
            'type': 'ir.actions.act_window',
            'name': f'Batch Processor: {batch.name}',
            'res_model': 'batch.composition.processor',
            'res_id': batch.id,
            'view_mode': 'form',
            'target': 'current'
        }
    
    def _get_all_eligible_clients(self):
        """Get all clients eligible for compositions"""
        return self.env['res.partner'].search([
            ('is_company', '=', False),
            ('order_history_ids', '!=', False)
        ])
    
    def _get_clients_by_tier(self):
        """Get clients by tier"""
        return self.env['res.partner'].search([
            ('is_company', '=', False),
            ('client_tier', '=', self.client_tier)
        ])