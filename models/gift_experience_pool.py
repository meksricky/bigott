from odoo import models, fields, api
from odoo.exceptions import UserError, ValidationError
import logging
import csv
import base64
import io

_logger = logging.getLogger(__name__)

class GiftExperience(models.Model):
    _inherit = 'gift.experience'
    
    # Enhanced fields for business rules compliance
    product_count = fields.Integer('Product Count', compute='_compute_product_count', store=True)
    experience_code = fields.Char('Experience Code', required=True, copy=False)
    
    # Seasonal availability
    seasonal_availability = fields.Selection([
        ('all_year', 'All Year'),
        ('spring', 'Spring'),
        ('summer', 'Summer'),
        ('autumn', 'Autumn'),
        ('winter', 'Winter'),
        ('christmas', 'Christmas Special'),
        ('new_year', 'New Year Special')
    ], string='Seasonal Availability', default='all_year')
    
    # Client targeting
    target_client_tier = fields.Selection([
        ('all', 'All Clients'),
        ('new', 'New Clients Only'),
        ('regular', 'Regular Clients'),
        ('premium', 'Premium Clients'),
        ('vip', 'VIP Clients Only')
    ], string='Target Client Tier', default='all')
    
    # Budget constraints for R3 rule compliance
    min_budget = fields.Float('Minimum Budget', compute='_compute_budget_range', store=True)
    max_budget = fields.Float('Maximum Budget', compute='_compute_budget_range', store=True)
    budget_flexibility = fields.Float('Budget Flexibility %', default=15.0,
                                     help="Acceptable budget variance percentage")
    
    # Usage tracking for R3 rule (avoid repetition)
    usage_count = fields.Integer('Total Usage Count', compute='_compute_usage_stats', store=True)
    last_used_date = fields.Date('Last Used', compute='_compute_usage_stats', store=True)
    clients_used = fields.Integer('Unique Clients Served', compute='_compute_usage_stats', store=True)
    
    # Experience lifecycle
    lifecycle_stage = fields.Selection([
        ('draft', 'Draft'),
        ('testing', 'Testing'),
        ('active', 'Active'),
        ('seasonal_pause', 'Seasonal Pause'),
        ('retired', 'Retired')
    ], string='Lifecycle Stage', default='draft', tracking=True)
    
    retirement_reason = fields.Text('Retirement Reason')
    retirement_date = fields.Date('Retirement Date')
    
    # Quality metrics
    average_client_rating = fields.Float('Average Client Rating', compute='_compute_quality_metrics', store=True)
    repeat_request_rate = fields.Float('Repeat Request Rate %', compute='_compute_quality_metrics', store=True)
    complaint_count = fields.Integer('Complaint Count', compute='_compute_quality_metrics', store=True)
    
    # Stock integration
    all_products_available = fields.Boolean('All Products Available', compute='_compute_stock_status')
    stock_issues = fields.Text('Stock Issues', compute='_compute_stock_status')
    
    @api.depends('product_ids')
    def _compute_product_count(self):
        for experience in self:
            experience.product_count = len(experience.product_ids)
    
    @api.depends('base_cost', 'budget_flexibility')
    def _compute_budget_range(self):
        for experience in self:
            if experience.base_cost:
                flexibility = experience.budget_flexibility / 100.0
                experience.min_budget = experience.base_cost * (1 - flexibility)
                experience.max_budget = experience.base_cost * (1 + flexibility)
            else:
                experience.min_budget = 0.0
                experience.max_budget = 0.0
    
    @api.depends('client_usage_ids')
    def _compute_usage_stats(self):
        for experience in self:
            usage_records = experience.client_usage_ids
            experience.usage_count = len(usage_records)
            experience.clients_used = len(usage_records.mapped('partner_id'))
            experience.last_used_date = max(usage_records.mapped('order_year')) if usage_records else False
    
    @api.depends('client_usage_ids.client_satisfaction')
    def _compute_quality_metrics(self):
        for experience in self:
            usage_records = experience.client_usage_ids
            
            # Average rating
            ratings = [float(r.client_satisfaction) for r in usage_records if r.client_satisfaction]
            experience.average_client_rating = sum(ratings) / len(ratings) if ratings else 0.0
            
            # Repeat requests (placeholder - would need actual repeat tracking)
            experience.repeat_request_rate = 0.0  # TODO: Implement repeat tracking
            
            # Complaints (placeholder - would need complaint tracking)
            experience.complaint_count = 0  # TODO: Implement complaint tracking
    
    @api.depends('product_ids')
    def _compute_stock_status(self):
        for experience in self:
            if not experience.product_ids:
                experience.all_products_available = True
                experience.stock_issues = ""
                continue
            
            stock_issues = []
            all_available = True
            
            for product in experience.product_ids:
                # Check stock (simplified - would integrate with actual stock management)
                stock_info = self._check_product_stock(product)
                if not stock_info['available']:
                    all_available = False
                    stock_issues.append(f"{product.name}: {stock_info['issue']}")
            
            experience.all_products_available = all_available
            experience.stock_issues = "\n".join(stock_issues)
    
    def _check_product_stock(self, product):
        """Check product stock availability"""
        # Simplified stock check - would integrate with actual inventory
        return {'available': True, 'issue': ''}
    
    @api.model
    def create(self, vals):
        if not vals.get('experience_code'):
            # Generate experience code
            sequence = self.env['ir.sequence'].next_by_code('gift.experience.code')
            vals['experience_code'] = sequence or f"EXP{len(self.search([])) + 1:04d}"
        return super().create(vals)
    
    @api.constrains('product_count')
    def _check_product_count(self):
        """Ensure experience has 3-5 products as per BRD"""
        for experience in self:
            if experience.product_count < 3 or experience.product_count > 5:
                raise ValidationError(f"Experience must have 3-5 products. Current: {experience.product_count}")
    
    @api.constrains('budget_flexibility')
    def _check_budget_flexibility(self):
        """Ensure budget flexibility is reasonable"""
        for experience in self:
            if not (0 <= experience.budget_flexibility <= 50):
                raise ValidationError("Budget flexibility must be between 0% and 50%")
    
    def is_suitable_for_client(self, partner_id, target_budget, target_year):
        """Check if experience is suitable for client and budget"""
        
        # Check lifecycle stage
        if self.lifecycle_stage not in ['active', 'testing']:
            return False, "Experience not active"
        
        # Check seasonal availability
        if not self._check_seasonal_availability(target_year):
            return False, "Not available in target season"
        
        # Check client tier compatibility
        if not self._check_client_tier_compatibility(partner_id):
            return False, "Client tier not compatible"
        
        # Check budget compatibility
        if not self._check_budget_compatibility(target_budget):
            return False, f"Budget outside range (${self.min_budget:.0f}-${self.max_budget:.0f})"
        
        # Check if client has used this experience before (R3 rule)
        if self._client_has_used_experience(partner_id):
            return False, "Client has used this experience before"
        
        # Check stock availability
        if not self.all_products_available:
            return False, "Stock issues with experience products"
        
        return True, "Suitable"
    
    def _check_seasonal_availability(self, target_year):
        """Check if experience is available in target season"""
        if self.seasonal_availability == 'all_year':
            return True
        
        # Simplified seasonal check - would need proper date logic
        return True  # For now, assume all experiences are available
    
    def _check_client_tier_compatibility(self, partner_id):
        """Check if experience is suitable for client tier"""
        if self.target_client_tier == 'all':
            return True
        
        partner = self.env['res.partner'].browse(partner_id)
        client_tier = getattr(partner, 'client_tier', 'new')
        
        return self.target_client_tier == client_tier
    
    def _check_budget_compatibility(self, target_budget):
        """Check if target budget is compatible with experience"""
        return self.min_budget <= target_budget <= self.max_budget
    
    def _client_has_used_experience(self, partner_id):
        """Check if client has used this experience before"""
        usage = self.env['client.order.history'].search([
            ('partner_id', '=', partner_id),
            ('experience_id', '=', self.id)
        ], limit=1)
        
        return bool(usage)
    
    def action_test_experience(self):
        """Mark experience as testing phase"""
        self.lifecycle_stage = 'testing'
    
    def action_activate_experience(self):
        """Activate experience for use"""
        # Validate experience before activation
        validation = self._validate_for_activation()
        if not validation['valid']:
            raise UserError(f"Cannot activate experience: {validation['error']}")
        
        self.lifecycle_stage = 'active'
    
    def action_retire_experience(self):
        """Retire experience from use"""
        return {
            'type': 'ir.actions.act_window',
            'name': 'Retire Experience',
            'res_model': 'experience.retirement.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': {'default_experience_id': self.id}
        }
    
    def _validate_for_activation(self):
        """Validate experience for activation"""
        
        # Check product count
        if not (3 <= self.product_count <= 5):
            return {'valid': False, 'error': f'Must have 3-5 products, currently has {self.product_count}'}
        
        # Check base cost
        if not self.base_cost:
            return {'valid': False, 'error': 'Base cost must be set'}
        
        # Check all products are active
        inactive_products = self.product_ids.filtered(lambda p: not p.active)
        if inactive_products:
            return {'valid': False, 'error': f'{len(inactive_products)} products are inactive'}
        
        # Check dietary compatibility settings
        if not any([self.is_vegan_friendly, self.is_halal_friendly, not self.is_alcohol_free]):
            # At least one dietary option should be clear
            pass
        
        return {'valid': True}


class ExperiencePoolImporter(models.Model):
    _name = 'experience.pool.importer'
    _description = 'Import Experience Pools from CSV/Excel'
    
    name = fields.Char('Import Name', required=True, default='New Import')
    import_file = fields.Binary('Import File', required=True)
    import_filename = fields.Char('Import Filename')
    
    file_type = fields.Selection([
        ('csv', 'CSV File'),
        ('excel', 'Excel File')
    ], string='File Type', required=True, default='csv')
    
    import_date = fields.Datetime('Import Date', default=fields.Datetime.now)
    imported_by = fields.Many2one('res.users', 'Imported By', default=lambda self: self.env.user)
    
    state = fields.Selection([
        ('draft', 'Draft'),
        ('validating', 'Validating'),
        ('importing', 'Importing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], string='Status', default='draft')
    
    # Import results
    total_rows = fields.Integer('Total Rows')
    successful_imports = fields.Integer('Successful Imports')
    failed_imports = fields.Integer('Failed Imports')
    
    import_log = fields.Text('Import Log')
    error_details = fields.Text('Error Details')
    
    imported_experience_ids = fields.One2many('gift.experience', 'importer_id', 'Imported Experiences')
    
    def action_validate_import(self):
        """Validate import file structure"""
        
        if not self.import_file:
            raise UserError("Please upload an import file")
        
        self.state = 'validating'
        
        try:
            # Decode file
            file_content = base64.b64decode(self.import_file)
            
            if self.file_type == 'csv':
                validation_result = self._validate_csv(file_content)
            else:
                validation_result = self._validate_excel(file_content)
            
            if validation_result['valid']:
                self.total_rows = validation_result['row_count']
                self.import_log = f"Validation successful. Found {validation_result['row_count']} rows to import."
                self.state = 'draft'  # Ready for import
            else:
                self.error_details = validation_result['errors']
                self.state = 'failed'
                
        except Exception as e:
            self.error_details = f"Validation failed: {str(e)}"
            self.state = 'failed'
    
    def action_start_import(self):
        """Start importing experiences"""
        
        if self.state != 'draft':
            raise UserError("Please validate the file first")
        
        self.state = 'importing'
        
        try:
            file_content = base64.b64decode(self.import_file)
            
            if self.file_type == 'csv':
                import_result = self._import_csv(file_content)
            else:
                import_result = self._import_excel(file_content)
            
            self.successful_imports = import_result['successful']
            self.failed_imports = import_result['failed']
            self.import_log = import_result['log']
            
            if import_result['failed'] == 0:
                self.state = 'completed'
            else:
                self.state = 'completed'  # Completed with errors
                
        except Exception as e:
            self.error_details = f"Import failed: {str(e)}"
            self.state = 'failed'
    
    def _validate_csv(self, file_content):
        """Validate CSV file structure"""
        
        try:
            # Parse CSV
            csv_file = io.StringIO(file_content.decode('utf-8'))
            reader = csv.DictReader(csv_file)
            
            # Required columns
            required_columns = [
                'name', 'experience_theme', 'product_codes', 
                'is_vegan_friendly', 'is_halal_friendly', 'is_alcohol_free'
            ]
            
            # Check headers
            headers = reader.fieldnames
            missing_columns = [col for col in required_columns if col not in headers]
            
            if missing_columns:
                return {
                    'valid': False,
                    'errors': f"Missing required columns: {', '.join(missing_columns)}"
                }
            
            # Count rows
            row_count = sum(1 for row in reader)
            
            return {'valid': True, 'row_count': row_count}
            
        except Exception as e:
            return {'valid': False, 'errors': f"CSV parsing error: {str(e)}"}
    
    def _validate_excel(self, file_content):
        """Validate Excel file structure"""
        # Would implement Excel validation using openpyxl or xlrd
        return {'valid': False, 'errors': 'Excel import not implemented yet'}
    
    def _import_csv(self, file_content):
        """Import experiences from CSV"""
        
        successful = 0
        failed = 0
        log_entries = []
        
        try:
            csv_file = io.StringIO(file_content.decode('utf-8'))
            reader = csv.DictReader(csv_file)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # Create experience
                    experience = self._create_experience_from_row(row)
                    if experience:
                        successful += 1
                        log_entries.append(f"Row {row_num}: Created experience '{experience.name}'")
                    else:
                        failed += 1
                        log_entries.append(f"Row {row_num}: Failed to create experience")
                        
                except Exception as e:
                    failed += 1
                    log_entries.append(f"Row {row_num}: Error - {str(e)}")
            
            return {
                'successful': successful,
                'failed': failed,
                'log': '\n'.join(log_entries)
            }
            
        except Exception as e:
            return {
                'successful': 0,
                'failed': self.total_rows,
                'log': f"Import failed: {str(e)}"
            }
    
    def _import_excel(self, file_content):
        """Import experiences from Excel"""
        return {
            'successful': 0,
            'failed': 0,
            'log': 'Excel import not implemented yet'
        }
    
    def _create_experience_from_row(self, row):
        """Create experience from CSV row"""
        
        # Parse product codes
        product_codes = row.get('product_codes', '').split(',')
        product_codes = [code.strip() for code in product_codes if code.strip()]
        
        # Find products
        products = self.env['product.template'].search([
            ('default_code', 'in', product_codes)
        ])
        
        if len(products) != len(product_codes):
            found_codes = products.mapped('default_code')
            missing_codes = [code for code in product_codes if code not in found_codes]
            raise UserError(f"Products not found: {', '.join(missing_codes)}")
        
        # Create experience
        experience_vals = {
            'name': row['name'],
            'experience_theme': row.get('experience_theme', 'premium_selection'),
            'description': row.get('description', ''),
            'instructions': row.get('instructions', ''),
            'product_ids': [(6, 0, products.ids)],
            'is_vegan_friendly': row.get('is_vegan_friendly', '').lower() in ['true', '1', 'yes'],
            'is_halal_friendly': row.get('is_halal_friendly', '').lower() in ['true', '1', 'yes'],
            'is_alcohol_free': row.get('is_alcohol_free', '').lower() in ['true', '1', 'yes'],
            'recommended_budget': float(row.get('recommended_budget', 0)) or None,
            'importer_id': self.id,
            'lifecycle_stage': 'draft'
        }
        
        return self.env['gift.experience'].create(experience_vals)


class ExperienceRetirementWizard(models.TransientModel):
    _name = 'experience.retirement.wizard'
    _description = 'Experience Retirement Wizard'
    
    experience_id = fields.Many2one('gift.experience', 'Experience', required=True)
    retirement_reason = fields.Text('Retirement Reason', required=True)
    retirement_date = fields.Date('Retirement Date', default=fields.Date.today, required=True)
    
    replacement_experience_id = fields.Many2one('gift.experience', 'Replacement Experience',
                                               help="Optional replacement experience")
    
    notify_users = fields.Boolean('Notify Users', default=True,
                                 help="Notify sales team about retirement")
    
    def action_retire_experience(self):
        """Retire the experience"""
        
        self.experience_id.write({
            'lifecycle_stage': 'retired',
            'retirement_reason': self.retirement_reason,
            'retirement_date': self.retirement_date,
            'active': False
        })
        
        # Log retirement
        self.experience_id.message_post(
            body=f"Experience retired: {self.retirement_reason}",
            subject="Experience Retired"
        )
        
        # Notify users if requested
        if self.notify_users:
            self._notify_experience_retirement()
        
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Experience Retired',
                'message': f'Experience "{self.experience_id.name}" has been retired',
                'type': 'info',
                'sticky': False,
            }
        }
    
    def _notify_experience_retirement(self):
        """Notify sales team about experience retirement"""
        
        # Find sales team users
        sales_group = self.env.ref('sales_team.group_sale_salesman', raise_if_not_found=False)
        if sales_group:
            users = sales_group.users
            
            # Create activity for each user
            for user in users:
                self.env['mail.activity'].create({
                    'activity_type_id': self.env.ref('mail.mail_activity_data_todo').id,
                    'summary': f'Experience Retired: {self.experience_id.name}',
                    'note': f'The experience "{self.experience_id.name}" has been retired. Reason: {self.retirement_reason}',
                    'res_id': self.experience_id.id,
                    'res_model': 'gift.experience',
                    'user_id': user.id,
                    'date_deadline': fields.Date.today()
                })


# Extend Gift Experience model with importer relationship
class GiftExperienceExtended(models.Model):
    _inherit = 'gift.experience'
    
    importer_id = fields.Many2one('experience.pool.importer', 'Imported From', readonly=True)
    is_imported = fields.Boolean('Imported', compute='_compute_is_imported', store=True)
    
    @api.depends('importer_id')
    def _compute_is_imported(self):
        for experience in self:
            experience.is_imported = bool(experience.importer_id)