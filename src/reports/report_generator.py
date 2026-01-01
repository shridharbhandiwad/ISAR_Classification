"""Main report generator for ISAR analysis results."""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

from .html_report import HTMLReportGenerator
from .pdf_report import PDFReportGenerator


class ReportGenerator:
    """
    Comprehensive report generator for ISAR classification results.
    
    Generates reports in multiple formats including HTML and PDF.
    """
    
    def __init__(
        self,
        output_dir: str = "reports",
        project_name: str = "ISAR Image Analysis"
    ):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
            project_name: Project name for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        
        self.html_generator = HTMLReportGenerator(project_name)
        self.pdf_generator = PDFReportGenerator(project_name)
    
    def generate_training_report(
        self,
        training_results: Dict[str, Any],
        config: Dict[str, Any],
        model_name: str,
        visualizations: Optional[Dict[str, str]] = None,
        format: str = 'both'
    ) -> str:
        """
        Generate a training report.
        
        Args:
            training_results: Training results dictionary
            config: Training configuration
            model_name: Name of the model
            visualizations: Dictionary of visualization paths
            format: Output format ('html', 'pdf', 'both')
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_data = {
            'title': f'Training Report - {model_name}',
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'config': config,
            'training_results': {
                'best_val_loss': training_results.get('best_val_loss'),
                'best_val_acc': training_results.get('best_val_acc'),
                'total_epochs': training_results.get('epochs_trained'),
                'training_time': training_results.get('total_time'),
            },
            'history': training_results.get('history', []),
            'visualizations': visualizations or {}
        }
        
        paths = []
        
        if format in ['html', 'both']:
            html_path = self.output_dir / f'training_report_{timestamp}.html'
            self.html_generator.generate_training_report(report_data, str(html_path))
            paths.append(str(html_path))
        
        if format in ['pdf', 'both']:
            pdf_path = self.output_dir / f'training_report_{timestamp}.pdf'
            self.pdf_generator.generate_training_report(report_data, str(pdf_path))
            paths.append(str(pdf_path))
        
        # Save JSON data
        json_path = self.output_dir / f'training_data_{timestamp}.json'
        self._save_json(report_data, json_path)
        
        return paths[0] if len(paths) == 1 else paths
    
    def generate_evaluation_report(
        self,
        eval_results: Dict[str, Any],
        model_name: str,
        class_names: List[str],
        visualizations: Optional[Dict[str, str]] = None,
        format: str = 'both'
    ) -> str:
        """
        Generate an evaluation report.
        
        Args:
            eval_results: Evaluation results dictionary
            model_name: Name of the model
            class_names: List of class names
            visualizations: Dictionary of visualization paths
            format: Output format
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_data = {
            'title': f'Evaluation Report - {model_name}',
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'class_names': class_names,
            'num_samples': eval_results.get('num_samples'),
            'metrics': {
                'accuracy': eval_results.get('accuracy'),
                'precision': eval_results.get('metrics', {}).get('precision'),
                'recall': eval_results.get('metrics', {}).get('recall'),
                'f1_score': eval_results.get('metrics', {}).get('f1'),
            },
            'per_class_metrics': eval_results.get('per_class_metrics', {}),
            'confusion_matrix': eval_results.get('confusion_matrix'),
            'roc_auc': eval_results.get('roc_auc', {}),
            'inference_time': eval_results.get('inference', {}),
            'visualizations': visualizations or {}
        }
        
        paths = []
        
        if format in ['html', 'both']:
            html_path = self.output_dir / f'evaluation_report_{timestamp}.html'
            self.html_generator.generate_evaluation_report(report_data, str(html_path))
            paths.append(str(html_path))
        
        if format in ['pdf', 'both']:
            pdf_path = self.output_dir / f'evaluation_report_{timestamp}.pdf'
            self.pdf_generator.generate_evaluation_report(report_data, str(pdf_path))
            paths.append(str(pdf_path))
        
        # Save JSON data
        json_path = self.output_dir / f'evaluation_data_{timestamp}.json'
        self._save_json(report_data, json_path)
        
        return paths[0] if len(paths) == 1 else paths
    
    def generate_comparison_report(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        format: str = 'both'
    ) -> str:
        """
        Generate a model comparison report.
        
        Args:
            comparison_results: Dictionary of model name -> metrics
            format: Output format
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_data = {
            'title': 'Model Comparison Report',
            'timestamp': datetime.now().isoformat(),
            'models': comparison_results
        }
        
        paths = []
        
        if format in ['html', 'both']:
            html_path = self.output_dir / f'comparison_report_{timestamp}.html'
            self.html_generator.generate_comparison_report(report_data, str(html_path))
            paths.append(str(html_path))
        
        if format in ['pdf', 'both']:
            pdf_path = self.output_dir / f'comparison_report_{timestamp}.pdf'
            self.pdf_generator.generate_comparison_report(report_data, str(pdf_path))
            paths.append(str(pdf_path))
        
        return paths[0] if len(paths) == 1 else paths
    
    def _save_json(self, data: Dict[str, Any], path: Path):
        """Save data as JSON, handling non-serializable types."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(path, 'w') as f:
            json.dump(convert(data), f, indent=2)
