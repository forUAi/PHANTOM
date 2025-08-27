"""Interactive Plotly-based reports."""

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from .html import Report
from ..utils.logging import get_logger

logger = get_logger(__name__)


class InteractiveReport(Report):
    """Interactive report with Plotly visualizations."""
    
    def __init__(self):
        super().__init__()
        self.figures = {}
    
    def save_html(self, filename: str):
        """Generate interactive HTML report with Plotly."""
        import os
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Generate all visualizations
        self._create_error_timeline()
        self._create_activation_heatmap()
        self._create_precision_analysis()
        self._create_operation_sunburst()
        
        # Build HTML
        html_content = self._build_interactive_html()
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Interactive report saved to: {filename}")
    
    def _create_error_timeline(self):
        """Create timeline of errors through the network."""
        if not self.metrics_per_op:
            return
        
        fig = go.Figure()
        
        # Add traces for different error metrics
        ops = list(range(len(self.metrics_per_op)))
        abs_errors = [m.get('max_abs_error', 0) for m in self.metrics_per_op]
        rel_errors = [m.get('max_rel_error', 0) for m in self.metrics_per_op]
        
        fig.add_trace(go.Scatter(
            x=ops,
            y=abs_errors,
            mode='lines+markers',
            name='Absolute Error',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=ops,
            y=rel_errors,
            mode='lines+markers',
            name='Relative Error',
            line=dict(color='#f39c12', width=2),
            marker=dict(size=8)
        ))
        
        # Add tolerance lines
        fig.add_hline(
            y=self.tolerances.get('abs', 1e-3),
            line_dash="dash",
            line_color="red",
            annotation_text="Abs Tolerance"
        )
        
        fig.add_hline(
            y=self.tolerances.get('rel', 1e-3),
            line_dash="dash",
            line_color="orange",
            annotation_text="Rel Tolerance"
        )
        
        # Highlight first bad op
        if self.first_bad_op is not None:
            fig.add_vline(
                x=self.first_bad_op,
                line_dash="dash",
                line_color="red",
                annotation_text=f"First Bad Op #{self.first_bad_op}"
            )
        
        fig.update_layout(
            title="Error Propagation Through Network",
            xaxis_title="Operation Index",
            yaxis_title="Error Value",
            yaxis_type="log",
            hovermode='x unified',
            template='plotly_dark'
        )
        
        self.figures['error_timeline'] = fig.to_html(include_plotlyjs='cdn')
    
    def _create_activation_heatmap(self):
        """Create interactive heatmap of activation differences."""
        if not self.ref_outputs or not self.target_outputs:
            return
        
        # Calculate differences
        diff = (self.ref_outputs[-1] - self.target_outputs[-1]).abs()
        
        # Reshape for visualization
        if diff.dim() > 2:
            # Flatten to 2D, preserving structure
            orig_shape = diff.shape
            if diff.dim() == 4:  # B, C, H, W
                diff = diff.reshape(orig_shape[0] * orig_shape[1], -1)
            else:
                diff = diff.flatten(0, -2)
        
        # Downsample if too large
        max_size = 256
        if diff.shape[0] > max_size or diff.shape[1] > max_size:
            stride_h = max(1, diff.shape[0] // max_size)
            stride_w = max(1, diff.shape[1] // max_size)
            diff = diff[::stride_h, ::stride_w]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=diff.cpu().numpy(),
            colorscale='Viridis',
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Difference: %{z:.6f}<extra></extra>',
            colorbar=dict(title='Absolute Difference')
        ))
        
        fig.update_layout(
            title="Activation Difference Heatmap (Interactive)",
            xaxis_title="Feature Dimension",
            yaxis_title="Batch/Channel Dimension",
            template='plotly_dark',
            height=600
        )
        
        self.figures['activation_heatmap'] = fig.to_html(include_plotlyjs='cdn')
    
    def _create_precision_analysis(self):
        """Create precision and quantization analysis charts."""
        if not hasattr(self, 'quant_stats') or not self.quant_stats:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Value Distribution', 
                'Precision Utilization',
                'Saturation Analysis',
                'Bit Pattern Distribution'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'pie'}]
            ]
        )
        
        # Value distribution
        if 'values' in self.quant_stats:
            fig.add_trace(
                go.Histogram(
                    x=self.quant_stats['values'],
                    nbinsx=50,
                    name='Value Distribution',
                    marker_color='#3498db'
                ),
                row=1, col=1
            )
        
        # Precision utilization
        precisions = ['fp64', 'fp32', 'fp16', 'bf16', 'int8']
        utilization = self.quant_stats.get('precision_util', [0.1, 0.3, 0.4, 0.15, 0.05])
        fig.add_trace(
            go.Bar(
                x=precisions,
                y=utilization,
                name='Precision Usage',
                marker_color='#2ecc71'
            ),
            row=1, col=2
        )
        
        # Saturation scatter
        if 'saturation_points' in self.quant_stats:
            fig.add_trace(
                go.Scatter(
                    x=self.quant_stats['saturation_points']['x'],
                    y=self.quant_stats['saturation_points']['y'],
                    mode='markers',
                    name='Saturation',
                    marker=dict(
                        size=8,
                        color=self.quant_stats['saturation_points']['intensity'],
                        colorscale='Reds',
                        showscale=True
                    )
                ),
                row=2, col=1
            )
        
        # Bit pattern distribution (for special values)
        special_values = {
            'Normal': self.quant_stats.get('percent_normal', 85),
            'Zero': self.quant_stats.get('percent_zero', 10),
            'Inf': self.quant_stats.get('percent_inf', 0.1),
            'NaN': self.quant_stats.get('percent_nan', 0.1),
            'Denormal': self.quant_stats.get('percent_denormal', 4.8)
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(special_values.keys()),
                values=list(special_values.values()),
                hole=0.3,
                marker_colors=['#2ecc71', '#95a5a6', '#e74c3c', '#c0392b', '#f39c12']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Precision and Quantization Analysis",
            template='plotly_dark',
            showlegend=False,
            height=800
        )
        
        self.figures['precision_analysis'] = fig.to_html(include_plotlyjs='cdn')
    
    def _create_operation_sunburst(self):
        """Create sunburst chart of operations hierarchy."""
        if not self.graph:
            return
        
        # Build hierarchy data
        labels = []
        parents = []
        values = []
        colors = []
        
        # Root
        labels.append("Model")
        parents.append("")
        values.append(100)
        colors.append("#3498db")
        
        # Group operations by type
        op_groups = {}
        for op in self.graph:
            op_type = op.module_type if hasattr(op, 'module_type') else 'Unknown'
            if op_type not in op_groups:
                op_groups[op_type] = []
            op_groups[op_type].append(op)
        
        # Add groups and operations
        for group_name, ops in op_groups.items():
            # Add group
            labels.append(group_name)
            parents.append("Model")
            values.append(len(ops))
            colors.append("#2ecc71" if group_name in ['Conv2d', 'Linear'] else "#f39c12")
            
            # Add individual ops
            for i, op in enumerate(ops[:10]):  # Limit to 10 per group
                op_label = f"{op.name if hasattr(op, 'name') else f'op_{i}'}"
                labels.append(op_label)
                parents.append(group_name)
                values.append(1)
                
                # Color based on error level
                if hasattr(op, 'error_level'):
                    if op.error_level > self.tolerances.get('abs', 1e-3):
                        colors.append("#e74c3c")  # Red for high error
                    else:
                        colors.append("#27ae60")  # Green for low error
                else:
                    colors.append("#95a5a6")  # Gray for unknown
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Operation Hierarchy",
            template='plotly_dark',
            height=600
        )
        
        self.figures['operation_sunburst'] = fig.to_html(include_plotlyjs='cdn')
    
    def _build_interactive_html(self) -> str:
        """Build complete interactive HTML report."""
        from ..utils.env import get_env_info
        
        env_info = get_env_info()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PHANTOM Interactive Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 3em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .status-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 30px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .status-pass {{
            background: linear-gradient(135deg, #2ecc71, #27ae60);
            color: white;
        }}
        .status-fail {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .chart-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab-button {{
            padding: 12px 24px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }}
        .tab-button:hover {{
            background: #667eea;
            color: white;
        }}
        .tab-button.active {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-color: transparent;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .code-block {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            margin: 20px 0;
        }}
        .recommendation {{
            background: linear-gradient(135deg, #f39c12, #e67e22);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin: 10px 0;
        }}
        .footer {{
            text-align: center;
            color: white;
            margin-top: 50px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 PHANTOM Interactive Debug Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Version:</strong> 0.2.0</p>
            
            <div class="status-badge {'status-fail' if self.first_bad_op is not None else 'status-pass'}">
                {'❌ DRIFT DETECTED' if self.first_bad_op is not None else '✅ NO DRIFT'}
            </div>
            
            {f'<p><strong>First Bad Operation:</strong> #{self.first_bad_op} - {self.graph[self.first_bad_op].name if self.graph and self.first_bad_op is not None and self.first_bad_op < len(self.graph) else "Unknown"}</p>' if self.first_bad_op is not None else ''}
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{self.metrics.get('max_abs_error', 0):.2e}</div>
                <div class="metric-label">Max Absolute Error</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.metrics.get('max_rel_error', 0):.2e}</div>
                <div class="metric-label">Max Relative Error</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(self.graph) if self.graph else 0}</div>
                <div class="metric-label">Total Operations</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{env_info.get('gpu_name', 'CPU')}</div>
                <div class="metric-label">Target Device</div>
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab-button active" onclick="showTab('timeline')">📈 Error Timeline</button>
            <button class="tab-button" onclick="showTab('heatmap')">🔥 Heatmap</button>
            <button class="tab-button" onclick="showTab('precision')">🎯 Precision</button>
            <button class="tab-button" onclick="showTab('operations')">🌳 Operations</button>
            <button class="tab-button" onclick="showTab('fusion')">⚡ Kernel Fusion</button>
        </div>
        
        <div id="timeline" class="tab-content active">
            <div class="chart-container">
                {self.figures.get('error_timeline', '<p>No timeline data available</p>')}
            </div>
        </div>
        
        <div id="heatmap" class="tab-content">
            <div class="chart-container">
                {self.figures.get('activation_heatmap', '<p>No heatmap data available</p>')}
            </div>
        </div>
        
        <div id="precision" class="tab-content">
            <div class="chart-container">
                {self.figures.get('precision_analysis', '<p>No precision data available</p>')}
            </div>
        </div>
        
        <div id="operations" class="tab-content">
            <div class="chart-container">
                {self.figures.get('operation_sunburst', '<p>No operation data available</p>')}
            </div>
        </div>
        
        <div id="fusion" class="tab-content">
            <div class="chart-container">
                <h2>⚡ Kernel Fusion Analysis</h2>
                {self._format_fusion_info()}
            </div>
        </div>
        
        <div class="chart-container">
            <h2>🎯 Recommendations</h2>
            {self._format_recommendations()}
        </div>
        
        <div class="code-block">
            <strong>🔄 Replay Command:</strong><br>
            phantom replay runs/{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.json
        </div>
        
        <div class="footer">
            <p>PHANTOM v0.2.0 - Cross-Accelerator Deterministic Debugger</p>
            <p>Made with ❤️ by ML Systems Engineers</p>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Remove active from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            
            // Activate button
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""
        return html_content
    
    def _format_fusion_info(self) -> str:
        """Format kernel fusion information."""
        if not hasattr(self, 'fusion_info') or not self.fusion_info:
            return "<p>No kernel fusion analysis available</p>"
        
        html = f"""
        <p><strong>Backend:</strong> {self.fusion_info.get('backend', 'Unknown')}</p>
        <p><strong>Fusion Opportunities:</strong> {self.fusion_info.get('total_opportunities', 0)}</p>
        
        <h3>Detected Patterns:</h3>
        <ul>
        """
        
        for pattern in self.fusion_info.get('patterns', []):
            impact_color = {
                'high': '#e74c3c',
                'medium': '#f39c12',
                'low': '#2ecc71'
            }.get(pattern.impact, '#95a5a6')
            
            html += f"""
            <li style="margin: 10px 0;">
                <strong>{pattern.name}</strong> 
                <span style="color: {impact_color};">({pattern.impact} impact)</span>
                - Confidence: {pattern.confidence:.1%}
                - Ops: {' → '.join(pattern.ops)}
            </li>
            """
        
        html += "</ul>"
        return html
    
    def _format_recommendations(self) -> str:
        """Format recommendations."""
        recommendations = []
        
        # Add fusion recommendations
        if hasattr(self, 'fusion_info') and self.fusion_info:
            recommendations.extend(self.fusion_info.get('recommendations', []))
        
        # Add precision recommendations
        if self.first_bad_op is not None:
            recommendations.append(
                f"Consider increasing precision at operation #{self.first_bad_op} "
                "or adjusting epsilon values for normalization layers."
            )
        
        if not recommendations:
            recommendations = ["No specific recommendations at this time."]
        
        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">💡 {rec}</div>'
        
        return html