# -*- coding: utf-8 -*-
# wizard/experience_option.py

from odoo import models, fields, api

class ExperienceOption(models.TransientModel):
    """Transient model to hold experience options for the wizard"""
    _name = 'experience.option'
    _description = 'Experience Option'
    _order = 'category, name'
    
    code = fields.Char(string='Code', required=True)
    name = fields.Char(string='Name', required=True)
    category = fields.Selection([
        ('aperitif', 'Aperitif'),
        ('seafood', 'Seafood'),
        ('meat', 'Meat'),
        ('vegetarian', 'Vegetarian'),
        ('vegan', 'Vegan'),
        ('dessert', 'Dessert'),
        ('foie', 'Foie Gras'),
        ('other', 'Other')
    ], string='Category', default='other')
    products = fields.Text(string='Product Codes', help='Comma-separated product codes')
    products_count = fields.Integer(string='Products Count')
    estimated_cost = fields.Float(string='Estimated Cost')
    dietary_info = fields.Char(string='Dietary Info')
    
    def get_product_list(self):
        """Get the list of product codes"""
        if self.products:
            return [p.strip() for p in self.products.split(',')]
        return []