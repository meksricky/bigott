odoo.define('lebigott_ai.GiftCompositionWidget', function (require) {
    "use strict";

    var FieldMany2many = require('web.relational_fields').FieldMany2many;
    var fieldRegistry = require('web.field_registry');
    
    var GiftCompositionWidget = FieldMany2many.extend({
        className: 'o_field_gift_composition',
        
        _renderReadonly: function () {
            var self = this;
            this._super.apply(this, arguments);
            
            // Add custom styling for product cards
            this.$el.find('.o_data_row').each(function () {
                var $row = $(this);
                $row.addClass('gift-product-card');
                
                // Add category badge
                var category = $row.find('.o_data_cell[name="lebiggot_category"]').text();
                if (category) {
                    var $badge = $('').text(category);
                    $row.find('.o_data_cell:first').append($badge);
                }
            });
            
            // Add total cost summary
            this._addCostSummary();
        },
        
        _addCostSummary: function () {
            var total = 0;
            this.$el.find('.o_data_cell[name="list_price"]').each(function () {
                var price = parseFloat($(this).text().replace(/[^0-9.-]+/g, ''));
                if (!isNaN(price)) {
                    total += price;
                }
            });
            
            var $summary = $('');
            $summary.append($('').text('Total Cost: '));
            $summary.append($('').text('€' + total.toFixed(2)));
            
            this.$el.append($summary);
        },
    });
    
    fieldRegistry.add('gift_composition_products', GiftCompositionWidget);
    
    return GiftCompositionWidget;
});