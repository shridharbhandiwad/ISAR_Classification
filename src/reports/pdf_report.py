"""PDF report generator using fpdf2."""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None


class PDFReportGenerator:
    """Generate PDF reports for ISAR analysis."""
    
    def __init__(self, project_name: str = "ISAR Image Analysis"):
        self.project_name = project_name
    
    def _create_pdf(self) -> 'FPDF':
        """Create a new PDF document."""
        if FPDF is None:
            raise ImportError("fpdf2 is required for PDF generation. Install with: pip install fpdf2")
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        return pdf
    
    def _add_header(self, pdf: 'FPDF', title: str, subtitle: str = ""):
        """Add header to PDF."""
        pdf.add_page()
        
        # Title background
        pdf.set_fill_color(37, 99, 235)  # Primary blue
        pdf.rect(0, 0, 210, 40, 'F')
        
        # Title
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_xy(10, 12)
        pdf.cell(190, 10, title, align='C')
        
        # Subtitle
        if subtitle:
            pdf.set_font('Helvetica', '', 10)
            pdf.set_xy(10, 25)
            pdf.cell(190, 10, subtitle, align='C')
        
        # Reset text color
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(50)
    
    def _add_section(self, pdf: 'FPDF', title: str):
        """Add section header."""
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_fill_color(243, 244, 246)  # Light gray
        pdf.cell(0, 10, title, ln=True, fill=True)
        pdf.ln(2)
    
    def _add_metric_row(self, pdf: 'FPDF', label: str, value: str):
        """Add a metric row."""
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(80, 8, label, border=0)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 8, value, ln=True, border=0)
    
    def _add_table(
        self,
        pdf: 'FPDF',
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[int]] = None
    ):
        """Add a table to PDF."""
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        
        # Headers
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(243, 244, 246)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 8, header, border=1, fill=True)
        pdf.ln()
        
        # Rows
        pdf.set_font('Helvetica', '', 9)
        for row in rows:
            for cell, width in zip(row, col_widths):
                pdf.cell(width, 7, str(cell), border=1)
            pdf.ln()
    
    def generate_training_report(
        self,
        report_data: Dict[str, Any],
        output_path: str
    ):
        """Generate PDF training report."""
        pdf = self._create_pdf()
        
        self._add_header(
            pdf,
            report_data['title'],
            f"Generated: {report_data['timestamp']}"
        )
        
        # Model Configuration
        self._add_section(pdf, "Model Configuration")
        self._add_metric_row(pdf, "Model:", report_data['model_name'])
        
        config = report_data.get('config', {}).get('model', {})
        for key, value in config.items():
            self._add_metric_row(pdf, f"{key}:", str(value))
        pdf.ln(5)
        
        # Training Results
        self._add_section(pdf, "Training Results")
        results = report_data['training_results']
        self._add_metric_row(pdf, "Best Validation Accuracy:", 
                           f"{results.get('best_val_acc', 0):.4f}")
        self._add_metric_row(pdf, "Best Validation Loss:", 
                           f"{results.get('best_val_loss', 0):.4f}")
        self._add_metric_row(pdf, "Epochs Trained:", 
                           str(results.get('total_epochs', 0)))
        self._add_metric_row(pdf, "Training Time:", 
                           f"{(results.get('training_time', 0) / 60):.1f} minutes")
        pdf.ln(5)
        
        # Training Configuration
        self._add_section(pdf, "Training Configuration")
        training_config = report_data.get('config', {}).get('training', {})
        for key, value in training_config.items():
            if not isinstance(value, dict):
                self._add_metric_row(pdf, f"{key}:", str(value))
        
        # Save PDF
        pdf.output(output_path)
    
    def generate_evaluation_report(
        self,
        report_data: Dict[str, Any],
        output_path: str
    ):
        """Generate PDF evaluation report."""
        pdf = self._create_pdf()
        
        self._add_header(
            pdf,
            report_data['title'],
            f"Generated: {report_data['timestamp']}"
        )
        
        # Overall Metrics
        self._add_section(pdf, "Overall Metrics")
        metrics = report_data['metrics']
        self._add_metric_row(pdf, "Accuracy:", 
                           f"{(metrics.get('accuracy', 0) * 100):.2f}%")
        self._add_metric_row(pdf, "Precision:", 
                           f"{(metrics.get('precision', 0) * 100):.2f}%")
        self._add_metric_row(pdf, "Recall:", 
                           f"{(metrics.get('recall', 0) * 100):.2f}%")
        self._add_metric_row(pdf, "F1 Score:", 
                           f"{(metrics.get('f1_score', 0) * 100):.2f}%")
        self._add_metric_row(pdf, "Test Samples:", str(report_data['num_samples']))
        pdf.ln(5)
        
        # Per-Class Performance
        self._add_section(pdf, "Per-Class Performance")
        
        headers = ['Class', 'Precision', 'Recall', 'F1 Score']
        rows = []
        for class_name, class_metrics in report_data.get('per_class_metrics', {}).items():
            rows.append([
                class_name,
                f"{(class_metrics.get('precision', 0) * 100):.2f}%",
                f"{(class_metrics.get('recall', 0) * 100):.2f}%",
                f"{(class_metrics.get('f1', 0) * 100):.2f}%"
            ])
        
        self._add_table(pdf, headers, rows, [50, 46, 47, 47])
        pdf.ln(5)
        
        # Inference Performance
        if report_data.get('inference_time'):
            self._add_section(pdf, "Inference Performance")
            inference = report_data['inference_time']
            self._add_metric_row(pdf, "Mean Inference Time:", 
                               f"{inference.get('mean_time_ms', 0):.2f} ms")
        
        # ROC-AUC
        if report_data.get('roc_auc'):
            self._add_section(pdf, "ROC-AUC Scores")
            for name, value in report_data['roc_auc'].items():
                self._add_metric_row(pdf, f"{name}:", f"{value:.4f}")
        
        # Save PDF
        pdf.output(output_path)
    
    def generate_comparison_report(
        self,
        report_data: Dict[str, Any],
        output_path: str
    ):
        """Generate PDF comparison report."""
        pdf = self._create_pdf()
        
        self._add_header(
            pdf,
            report_data['title'],
            f"Generated: {report_data['timestamp']}"
        )
        
        # Model Comparison Table
        self._add_section(pdf, "Model Comparison")
        
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time (ms)']
        rows = []
        
        for model_name, metrics in report_data['models'].items():
            rows.append([
                model_name,
                f"{(metrics.get('accuracy', 0) * 100):.1f}%",
                f"{(metrics.get('precision', 0) * 100):.1f}%",
                f"{(metrics.get('recall', 0) * 100):.1f}%",
                f"{(metrics.get('f1', 0) * 100):.1f}%",
                f"{metrics.get('inference_time', 0):.1f}"
            ])
        
        self._add_table(pdf, headers, rows, [45, 29, 29, 29, 29, 29])
        
        # Save PDF
        pdf.output(output_path)
