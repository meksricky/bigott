odoo.define('lebigott_ai.MLDashboard', function (require) {
    "use strict";

    var core = require('web.core');
    var Widget = require('web.Widget');
    var ajax = require('web.ajax');
    var _t = core._t;

    var MLDashboard = Widget.extend({
        template: 'MLDashboard',
        events: {
            'click .ml-train-button': '_onTrainModel',
            'click .ml-evaluate-button': '_onEvaluateModel',
            'click .ml-refresh-button': '_onRefreshStats',
        },

        init: function (parent, options) {
            this._super(parent);
            this.modelData = {};
            this.performanceData = {};
        },

        start: function () {
            var self = this;
            return this._super().then(function () {
                self._loadDashboardData();
                // Auto-refresh every 30 seconds
                setInterval(function () {
                    self._loadDashboardData();
                }, 30000);
            });
        },

        _loadDashboardData: function () {
            var self = this;
            
            return this._rpc({
                model: 'ml.recommendation.engine',
                method: 'get_dashboard_data',
                args: [],
            }).then(function (data) {
                self.modelData = data;
                self._updateDashboard();
            });
        },

        _updateDashboard: function () {
            var self = this;
            
            // Update accuracy gauge
            this._updateGauge('accuracy-gauge', this.modelData.accuracy);
            
            // Update metrics
            this.$('.ml-samples-count').text(this.modelData.training_samples || 0);
            this.$('.ml-r2-score').text((this.modelData.r2_score || 0).toFixed(3));
            this.$('.ml-last-training').text(this.modelData.last_training || 'Never');
            
            // Update confidence bars
            this.$('.ml-confidence-fill').each(function () {
                var confidence = $(this).data('confidence') || 0;
                $(this).css('width', (confidence * 100) + '%');
            });
            
            // Update feature importance chart
            if (this.modelData.feature_importance) {
                this._renderFeatureImportance(this.modelData.feature_importance);
            }
        },

        _updateGauge: function (elementId, value) {
            var canvas = document.getElementById(elementId);
            if (!canvas) return;
            
            var ctx = canvas.getContext('2d');
            var centerX = canvas.width / 2;
            var centerY = canvas.height / 2;
            var radius = 60;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw background arc
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, Math.PI * 0.7, Math.PI * 2.3);
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 15;
            ctx.stroke();
            
            // Draw value arc
            var angle = (value / 100) * Math.PI * 1.6 + Math.PI * 0.7;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, Math.PI * 0.7, angle);
            ctx.strokeStyle = this._getColorForValue(value);
            ctx.lineWidth = 15;
            ctx.lineCap = 'round';
            ctx.stroke();
            
            // Draw text
            ctx.font = 'bold 24px Arial';
            ctx.fillStyle = '#495057';
            ctx.textAlign = 'center';
            ctx.fillText(value.toFixed(1) + '%', centerX, centerY + 5);
        },

        _getColorForValue: function (value) {
            if (value >= 80) return '#28a745';
            if (value >= 60) return '#ffc107';
            return '#dc3545';
        },

        _renderFeatureImportance: function (importance) {
            var $container = this.$('.ml-feature-importance');
            $container.empty();
            
            // Sort features by importance
            var features = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 10);
            
            features.forEach(function (feature) {
                var $bar = $('');
                $bar.append($('').text(feature[0]));
                
                var $value = $('');
                var $fill = $('').css('width', (feature[1] * 100) + '%');
                $value.append($fill);
                $bar.append($value);
                
                $container.append($bar);
            });
        },

        _onTrainModel: function (ev) {
            ev.preventDefault();
            var self = this;
            
            this.do_action({
                type: 'ir.actions.act_window',
                res_model: 'ml.training.wizard',
                views: [[false, 'form']],
                target: 'new',
                context: {},
            }, {
                on_close: function () {
                    self._loadDashboardData();
                }
            });
        },

        _onEvaluateModel: function (ev) {
            ev.preventDefault();
            var self = this;
            
            this._rpc({
                model: 'ml.recommendation.engine',
                method: 'evaluate_model_performance',
                args: [],
            }).then(function (result) {
                self.do_notify(
                    _t('Model Evaluation Complete'),
                    _t('Accuracy: ') + result.accuracy.toFixed(1) + '%',
                    false
                );
                self._loadDashboardData();
            });
        },

        _onRefreshStats: function (ev) {
            ev.preventDefault();
            this._loadDashboardData();
            this.do_notify(_t('Dashboard Refreshed'), '', false);
        },
    });

    core.action_registry.add('ml_dashboard', MLDashboard);
    
    return MLDashboard;
});