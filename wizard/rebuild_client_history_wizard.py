# -*- coding: utf-8 -*-
from odoo import models, fields

class RebuildClientHistoryWizard(models.TransientModel):
    _name = 'rebuild.client.history.wizard'
    _description = 'Rebuild Client History from Sales'

    date_from = fields.Date('From')
    date_to = fields.Date('To')

    def action_run(self):
        # Convert to datetime at midnight to match domain types
        df = self.date_from and fields.Datetime.to_datetime(self.date_from) or False
        dt = self.date_to and fields.Datetime.to_datetime(self.date_to) or False
        self.env['client.order.history'].rebuild_from_sales(date_from=df, date_to=dt)
        return {'type': 'ir.actions.client', 'tag': 'reload'}
