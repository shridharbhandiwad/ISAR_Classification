"""HTML report generator."""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import base64
from jinja2 import Template


class HTMLReportGenerator:
    """Generate HTML reports for ISAR analysis."""
    
    def __init__(self, project_name: str = "ISAR Image Analysis"):
        self.project_name = project_name
    
    def generate_training_report(
        self,
        report_data: Dict[str, Any],
        output_path: str
    ):
        """Generate HTML training report."""
        html_content = self._get_training_template().render(
            project_name=self.project_name,
            **report_data
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def generate_evaluation_report(
        self,
        report_data: Dict[str, Any],
        output_path: str
    ):
        """Generate HTML evaluation report."""
        html_content = self._get_evaluation_template().render(
            project_name=self.project_name,
            **report_data
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def generate_comparison_report(
        self,
        report_data: Dict[str, Any],
        output_path: str
    ):
        """Generate HTML comparison report."""
        html_content = self._get_comparison_template().render(
            project_name=self.project_name,
            **report_data
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _get_base_style(self) -> str:
        """Get base CSS styles."""
        return """
        <style>
            :root {
                --primary: #2563eb;
                --primary-dark: #1d4ed8;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --gray-50: #f9fafb;
                --gray-100: #f3f4f6;
                --gray-200: #e5e7eb;
                --gray-600: #4b5563;
                --gray-800: #1f2937;
                --gray-900: #111827;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: var(--gray-50);
                color: var(--gray-800);
                line-height: 1.6;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            header {
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
                padding: 2rem;
                margin-bottom: 2rem;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            header h1 {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            header .subtitle {
                opacity: 0.9;
                font-size: 1rem;
            }
            
            .card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }
            
            .card h2 {
                color: var(--gray-900);
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid var(--gray-100);
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
            }
            
            .metric-card {
                background: var(--gray-50);
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: var(--primary);
            }
            
            .metric-label {
                font-size: 0.875rem;
                color: var(--gray-600);
                margin-top: 0.25rem;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
            }
            
            th, td {
                padding: 0.75rem;
                text-align: left;
                border-bottom: 1px solid var(--gray-200);
            }
            
            th {
                background: var(--gray-50);
                font-weight: 600;
                color: var(--gray-600);
                font-size: 0.875rem;
                text-transform: uppercase;
            }
            
            tr:hover {
                background: var(--gray-50);
            }
            
            .visualization {
                margin-top: 1rem;
            }
            
            .visualization img {
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .success { color: var(--success); }
            .warning { color: var(--warning); }
            .danger { color: var(--danger); }
            
            footer {
                text-align: center;
                padding: 2rem;
                color: var(--gray-600);
                font-size: 0.875rem;
            }
            
            .two-column {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
            }
            
            @media (max-width: 768px) {
                .two-column {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """
    
    def _get_training_template(self) -> Template:
        """Get training report template."""
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }} - {{ project_name }}</title>
            {{ style }}
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>{{ title }}</h1>
                    <p class="subtitle">Generated: {{ timestamp }}</p>
                </header>
                
                <div class="card">
                    <h2>Model Configuration</h2>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Model</td>
                            <td>{{ model_name }}</td>
                        </tr>
                        {% for key, value in config.get('model', {}).items() %}
                        <tr>
                            <td>{{ key }}</td>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="card">
                    <h2>Training Results</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.4f"|format(training_results.best_val_acc or 0) }}</div>
                            <div class="metric-label">Best Validation Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.4f"|format(training_results.best_val_loss or 0) }}</div>
                            <div class="metric-label">Best Validation Loss</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ training_results.total_epochs or 0 }}</div>
                            <div class="metric-label">Epochs Trained</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.1f"|format((training_results.training_time or 0) / 60) }} min</div>
                            <div class="metric-label">Training Time</div>
                        </div>
                    </div>
                </div>
                
                {% if visualizations %}
                <div class="card">
                    <h2>Training Visualizations</h2>
                    {% for name, path in visualizations.items() %}
                    <div class="visualization">
                        <h3>{{ name }}</h3>
                        <img src="{{ path }}" alt="{{ name }}">
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <footer>
                    <p>{{ project_name }} - Automated Report</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return Template(template_str.replace('{{ style }}', self._get_base_style()))
    
    def _get_evaluation_template(self) -> Template:
        """Get evaluation report template."""
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }} - {{ project_name }}</title>
            {{ style }}
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>{{ title }}</h1>
                    <p class="subtitle">Generated: {{ timestamp }}</p>
                </header>
                
                <div class="card">
                    <h2>Overall Metrics</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.2f"|format((metrics.accuracy or 0) * 100) }}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.2f"|format((metrics.precision or 0) * 100) }}%</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.2f"|format((metrics.recall or 0) * 100) }}%</div>
                            <div class="metric-label">Recall</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.2f"|format((metrics.f1_score or 0) * 100) }}%</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Per-Class Performance</h2>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                        </tr>
                        {% for class_name, class_metrics in per_class_metrics.items() %}
                        <tr>
                            <td>{{ class_name }}</td>
                            <td>{{ "%.2f"|format((class_metrics.precision or 0) * 100) }}%</td>
                            <td>{{ "%.2f"|format((class_metrics.recall or 0) * 100) }}%</td>
                            <td>{{ "%.2f"|format((class_metrics.f1 or 0) * 100) }}%</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                {% if inference_time %}
                <div class="card">
                    <h2>Inference Performance</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.2f"|format(inference_time.mean_time_ms or 0) }} ms</div>
                            <div class="metric-label">Mean Inference Time</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ num_samples }}</div>
                            <div class="metric-label">Test Samples</div>
                        </div>
                    </div>
                </div>
                {% endif %}
                
                {% if visualizations %}
                <div class="card">
                    <h2>Visualizations</h2>
                    {% for name, path in visualizations.items() %}
                    <div class="visualization">
                        <h3>{{ name }}</h3>
                        <img src="{{ path }}" alt="{{ name }}">
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <footer>
                    <p>{{ project_name }} - Automated Report</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return Template(template_str.replace('{{ style }}', self._get_base_style()))
    
    def _get_comparison_template(self) -> Template:
        """Get comparison report template."""
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }} - {{ project_name }}</title>
            {{ style }}
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>{{ title }}</h1>
                    <p class="subtitle">Generated: {{ timestamp }}</p>
                </header>
                
                <div class="card">
                    <h2>Model Comparison</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                            <th>Inference Time</th>
                        </tr>
                        {% for model_name, metrics in models.items() %}
                        <tr>
                            <td>{{ model_name }}</td>
                            <td>{{ "%.2f"|format((metrics.accuracy or 0) * 100) }}%</td>
                            <td>{{ "%.2f"|format((metrics.precision or 0) * 100) }}%</td>
                            <td>{{ "%.2f"|format((metrics.recall or 0) * 100) }}%</td>
                            <td>{{ "%.2f"|format((metrics.f1 or 0) * 100) }}%</td>
                            <td>{{ "%.2f"|format(metrics.inference_time or 0) }} ms</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <footer>
                    <p>{{ project_name }} - Automated Report</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return Template(template_str.replace('{{ style }}', self._get_base_style()))
