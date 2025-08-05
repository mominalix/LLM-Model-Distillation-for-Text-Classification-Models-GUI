"""
Metrics display panel for showing evaluation results and data quality.

This component provides comprehensive visualization of model performance,
data quality metrics, and evaluation results.
"""

import time
from typing import Dict, Any, Optional, List
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from ...config import Config
from ...data import GenerationResult, QualityMetrics
from ...training import EvaluationMetrics


class MetricsPanel(ctk.CTkFrame):
    """Panel for displaying metrics and evaluation results."""
    
    def __init__(self, parent, config: Config, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config
        
        # Current metrics
        self.current_generation_result: Optional[GenerationResult] = None
        self.current_evaluation_result: Optional[EvaluationMetrics] = None
        self.live_metrics: Dict[str, float] = {}
        
        self.grid_columnconfigure(0, weight=1)
        
        # Create metrics sections
        self._create_data_quality_section()
        self._create_model_performance_section()
        self._create_visualization_section()
    
    def _create_data_quality_section(self) -> None:
        """Create data quality metrics section."""
        
        # Data quality frame
        self.data_quality_frame = ctk.CTkFrame(self)
        self.data_quality_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.data_quality_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Section title
        self.data_quality_title = ctk.CTkLabel(
            self.data_quality_frame,
            text="Data Quality Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.data_quality_title.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky="w")
        
        # Quality metrics grid
        metrics_labels = [
            ("Overall Quality:", "quality_score"),
            ("Vocabulary Size:", "vocab_size"),
            ("Avg Length:", "avg_length"),
            ("Class Balance:", "class_balance"),
            ("Duplicate Ratio:", "duplicate_ratio"),
            ("Bias Score:", "bias_score"),
            ("Distinct-2:", "distinct_2"),
            ("Safety Check:", "safety_status")
        ]
        
        self.data_quality_labels = {}
        
        for i, (label_text, metric_key) in enumerate(metrics_labels):
            row = (i // 4) + 1
            col = i % 4
            
            # Metric label
            label = ctk.CTkLabel(
                self.data_quality_frame,
                text=label_text,
                font=ctk.CTkFont(size=12, weight="bold")
            )
            label.grid(row=row*2, column=col, padx=5, pady=2, sticky="w")
            
            # Metric value
            value_label = ctk.CTkLabel(
                self.data_quality_frame,
                text="-",
                font=ctk.CTkFont(size=11)
            )
            value_label.grid(row=row*2+1, column=col, padx=5, pady=2, sticky="w")
            
            self.data_quality_labels[metric_key] = value_label
    
    def _create_model_performance_section(self) -> None:
        """Create model performance metrics section."""
        
        # Model performance frame
        self.model_perf_frame = ctk.CTkFrame(self)
        self.model_perf_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.model_perf_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Section title
        self.model_perf_title = ctk.CTkLabel(
            self.model_perf_frame,
            text="Model Performance",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.model_perf_title.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky="w")
        
        # Performance metrics grid
        perf_metrics = [
            ("Accuracy:", "accuracy"),
            ("F1-Macro:", "f1_macro"),
            ("Precision:", "precision"),
            ("Recall:", "recall"),
            ("Loss:", "loss"),
            ("Inference Time:", "inference_time"),
            ("Model Size:", "model_size"),
            ("Throughput:", "throughput")
        ]
        
        self.model_perf_labels = {}
        
        for i, (label_text, metric_key) in enumerate(perf_metrics):
            row = (i // 4) + 1
            col = i % 4
            
            # Metric label
            label = ctk.CTkLabel(
                self.model_perf_frame,
                text=label_text,
                font=ctk.CTkFont(size=12, weight="bold")
            )
            label.grid(row=row*2, column=col, padx=5, pady=2, sticky="w")
            
            # Metric value
            value_label = ctk.CTkLabel(
                self.model_perf_frame,
                text="-",
                font=ctk.CTkFont(size=11)
            )
            value_label.grid(row=row*2+1, column=col, padx=5, pady=2, sticky="w")
            
            self.model_perf_labels[metric_key] = value_label
        
        # Per-class performance frame
        self.per_class_frame = ctk.CTkFrame(self.model_perf_frame)
        self.per_class_frame.grid(row=5, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        self.per_class_frame.grid_columnconfigure(0, weight=1)
        
        self.per_class_title = ctk.CTkLabel(
            self.per_class_frame,
            text="Per-Class Performance:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.per_class_title.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        
        # Scrollable frame for per-class metrics
        self.per_class_scroll = ctk.CTkScrollableFrame(self.per_class_frame, height=80)
        self.per_class_scroll.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        self.per_class_scroll.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Per-class header
        headers = ["Class", "Precision", "Recall", "F1-Score"]
        for i, header in enumerate(headers):
            header_label = ctk.CTkLabel(
                self.per_class_scroll,
                text=header,
                font=ctk.CTkFont(size=11, weight="bold")
            )
            header_label.grid(row=0, column=i, padx=2, pady=2, sticky="w")
    
    def _create_visualization_section(self) -> None:
        """Create visualization section with charts."""
        
        # Visualization frame
        self.viz_frame = ctk.CTkFrame(self)
        self.viz_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.viz_frame.grid_columnconfigure(0, weight=1)
        
        # Section title
        self.viz_title = ctk.CTkLabel(
            self.viz_frame,
            text="Visualizations",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.viz_title.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Chart notebook (tabs)
        self.chart_tabview = ctk.CTkTabview(self.viz_frame)
        self.chart_tabview.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        # Confusion Matrix tab
        self.confusion_tab = self.chart_tabview.add("Confusion Matrix")
        self.confusion_tab.grid_columnconfigure(0, weight=1)
        
        # Performance Comparison tab
        self.performance_tab = self.chart_tabview.add("Performance")
        self.performance_tab.grid_columnconfigure(0, weight=1)
        
        # Data Quality tab
        self.quality_tab = self.chart_tabview.add("Data Quality")
        self.quality_tab.grid_columnconfigure(0, weight=1)
        
        # Initialize charts
        self._create_confusion_matrix_chart()
        self._create_performance_chart()
        self._create_quality_chart()
        
        # Export button
        self.export_button = ctk.CTkButton(
            self.viz_frame,
            text="Export Metrics",
            command=self._export_metrics
        )
        self.export_button.grid(row=2, column=0, padx=10, pady=5, sticky="e")
    
    def _create_confusion_matrix_chart(self) -> None:
        """Create confusion matrix visualization."""
        
        self.confusion_fig = Figure(figsize=(6, 4), dpi=100)
        self.confusion_fig.patch.set_facecolor('#2b2b2b')
        
        self.confusion_ax = self.confusion_fig.add_subplot(111)
        self.confusion_ax.set_facecolor('#1e1e1e')
        self.confusion_ax.set_title('Confusion Matrix', color='white')
        
        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_fig, self.confusion_tab)
        self.confusion_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Initialize empty matrix
        self._plot_empty_confusion_matrix()
    
    def _create_performance_chart(self) -> None:
        """Create performance metrics visualization."""
        
        self.performance_fig = Figure(figsize=(6, 4), dpi=100)
        self.performance_fig.patch.set_facecolor('#2b2b2b')
        
        self.performance_ax = self.performance_fig.add_subplot(111)
        self.performance_ax.set_facecolor('#1e1e1e')
        self.performance_ax.set_title('Performance Metrics', color='white')
        self.performance_ax.tick_params(colors='white')
        
        self.performance_canvas = FigureCanvasTkAgg(self.performance_fig, self.performance_tab)
        self.performance_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Initialize empty chart
        self._plot_empty_performance_chart()
    
    def _create_quality_chart(self) -> None:
        """Create data quality visualization."""
        
        self.quality_fig = Figure(figsize=(6, 4), dpi=100)
        self.quality_fig.patch.set_facecolor('#2b2b2b')
        
        self.quality_ax = self.quality_fig.add_subplot(111)
        self.quality_ax.set_facecolor('#1e1e1e')
        self.quality_ax.set_title('Data Quality Metrics', color='white')
        self.quality_ax.tick_params(colors='white')
        
        self.quality_canvas = FigureCanvasTkAgg(self.quality_fig, self.quality_tab)
        self.quality_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Initialize empty chart
        self._plot_empty_quality_chart()
    
    def _plot_empty_confusion_matrix(self) -> None:
        """Plot empty confusion matrix placeholder."""
        self.confusion_ax.clear()
        self.confusion_ax.text(
            0.5, 0.5, 'No data available\nTrain a model to see confusion matrix',
            horizontalalignment='center',
            verticalalignment='center',
            transform=self.confusion_ax.transAxes,
            color='white',
            fontsize=12
        )
        self.confusion_ax.set_xticks([])
        self.confusion_ax.set_yticks([])
        self.confusion_canvas.draw()
    
    def _plot_empty_performance_chart(self) -> None:
        """Plot empty performance chart placeholder."""
        self.performance_ax.clear()
        self.performance_ax.text(
            0.5, 0.5, 'No data available\nEvaluate a model to see performance metrics',
            horizontalalignment='center',
            verticalalignment='center',
            transform=self.performance_ax.transAxes,
            color='white',
            fontsize=12
        )
        self.performance_canvas.draw()
    
    def _plot_empty_quality_chart(self) -> None:
        """Plot empty quality chart placeholder."""
        self.quality_ax.clear()
        self.quality_ax.text(
            0.5, 0.5, 'No data available\nGenerate data to see quality metrics',
            horizontalalignment='center',
            verticalalignment='center',
            transform=self.quality_ax.transAxes,
            color='white',
            fontsize=12
        )
        self.quality_canvas.draw()
    
    def update_generation_metrics(self, result: GenerationResult) -> None:
        """Update metrics with generation results."""
        
        self.current_generation_result = result
        quality_metrics = result.quality_metrics
        
        # Update data quality metrics
        self.data_quality_labels["quality_score"].configure(
            text=f"{quality_metrics.overall_quality:.3f}"
        )
        self.data_quality_labels["vocab_size"].configure(
            text=f"{quality_metrics.vocabulary_size:,}"
        )
        self.data_quality_labels["avg_length"].configure(
            text=f"{quality_metrics.average_length:.1f}"
        )
        self.data_quality_labels["class_balance"].configure(
            text=f"{quality_metrics.class_balance:.3f}"
        )
        self.data_quality_labels["duplicate_ratio"].configure(
            text=f"{quality_metrics.duplicate_ratio:.3f}"
        )
        self.data_quality_labels["bias_score"].configure(
            text=f"{quality_metrics.bias_score:.3f}"
        )
        self.data_quality_labels["distinct_2"].configure(
            text=f"{quality_metrics.distinct_2:.3f}"
        )
        
        # Safety status
        safety_status = "Safe" if not quality_metrics.pii_detected else "PII Detected"
        self.data_quality_labels["safety_status"].configure(text=safety_status)
        
        # Update quality chart
        self._plot_quality_metrics(quality_metrics)
    
    def update_training_metrics(self, training_result: Dict[str, Any], evaluation_result: EvaluationMetrics) -> None:
        """Update metrics with training and evaluation results."""
        
        self.current_evaluation_result = evaluation_result
        
        # Update model performance metrics
        self.model_perf_labels["accuracy"].configure(
            text=f"{evaluation_result.accuracy:.4f}"
        )
        self.model_perf_labels["f1_macro"].configure(
            text=f"{evaluation_result.f1_macro:.4f}"
        )
        self.model_perf_labels["precision"].configure(
            text=f"{evaluation_result.precision_macro:.4f}"
        )
        self.model_perf_labels["recall"].configure(
            text=f"{evaluation_result.recall_macro:.4f}"
        )
        
        # Get final loss from training result
        final_loss = 0.0
        if 'training_history' in training_result:
            history = training_result['training_history']
            if history and len(history) > 0:
                final_loss = history[-1].get('train_loss', 0.0)
        
        self.model_perf_labels["loss"].configure(text=f"{final_loss:.4f}")
        
        self.model_perf_labels["inference_time"].configure(
            text=f"{evaluation_result.inference_time_ms:.2f}ms"
        )
        self.model_perf_labels["model_size"].configure(
            text=f"{evaluation_result.model_size_mb:.1f}MB"
        )
        self.model_perf_labels["throughput"].configure(
            text=f"{evaluation_result.throughput_samples_per_sec:.1f}/s"
        )
        
        # Update per-class performance
        self._update_per_class_performance(evaluation_result)
        
        # Update visualizations
        self._plot_confusion_matrix(evaluation_result)
        self._plot_performance_metrics(evaluation_result)
    
    def update_live_metrics(self, metrics: Dict[str, float]) -> None:
        """Update live training metrics."""
        
        self.live_metrics = metrics
        
        # Update displayed values if we have them
        if "train_accuracy" in metrics:
            self.model_perf_labels["accuracy"].configure(
                text=f"{metrics['train_accuracy']:.4f}"
            )
        
        if "train_f1" in metrics:
            self.model_perf_labels["f1_macro"].configure(
                text=f"{metrics['train_f1']:.4f}"
            )
        
        if "train_loss" in metrics:
            self.model_perf_labels["loss"].configure(
                text=f"{metrics['train_loss']:.4f}"
            )
    
    def _update_per_class_performance(self, evaluation_result: EvaluationMetrics) -> None:
        """Update per-class performance display."""
        
        # Clear existing per-class labels (except header)
        for widget in self.per_class_scroll.winfo_children()[4:]:  # Skip header row
            widget.destroy()
        
        # Add per-class metrics
        row = 1
        for class_name in sorted(evaluation_result.per_class_precision.keys()):
            precision = evaluation_result.per_class_precision[class_name]
            recall = evaluation_result.per_class_recall[class_name]
            f1 = evaluation_result.per_class_f1[class_name]
            
            # Class name
            class_label = ctk.CTkLabel(
                self.per_class_scroll,
                text=class_name,
                font=ctk.CTkFont(size=10)
            )
            class_label.grid(row=row, column=0, padx=2, pady=1, sticky="w")
            
            # Precision
            prec_label = ctk.CTkLabel(
                self.per_class_scroll,
                text=f"{precision:.3f}",
                font=ctk.CTkFont(size=10)
            )
            prec_label.grid(row=row, column=1, padx=2, pady=1, sticky="w")
            
            # Recall
            rec_label = ctk.CTkLabel(
                self.per_class_scroll,
                text=f"{recall:.3f}",
                font=ctk.CTkFont(size=10)
            )
            rec_label.grid(row=row, column=2, padx=2, pady=1, sticky="w")
            
            # F1
            f1_label = ctk.CTkLabel(
                self.per_class_scroll,
                text=f"{f1:.3f}",
                font=ctk.CTkFont(size=10)
            )
            f1_label.grid(row=row, column=3, padx=2, pady=1, sticky="w")
            
            row += 1
    
    def _plot_confusion_matrix(self, evaluation_result: EvaluationMetrics) -> None:
        """Plot confusion matrix."""
        
        if evaluation_result.confusion_matrix is None:
            self._plot_empty_confusion_matrix()
            return
        
        self.confusion_ax.clear()
        
        cm = evaluation_result.confusion_matrix
        class_names = list(evaluation_result.per_class_precision.keys())
        
        # Plot heatmap
        im = self.confusion_ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        self.confusion_fig.colorbar(im, ax=self.confusion_ax)
        
        # Add labels
        self.confusion_ax.set_xticks(range(len(class_names)))
        self.confusion_ax.set_yticks(range(len(class_names)))
        self.confusion_ax.set_xticklabels(class_names, rotation=45, ha='right')
        self.confusion_ax.set_yticklabels(class_names)
        self.confusion_ax.set_xlabel('Predicted Label', color='white')
        self.confusion_ax.set_ylabel('True Label', color='white')
        self.confusion_ax.set_title('Confusion Matrix', color='white')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.confusion_ax.text(
                    j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
        
        self.confusion_fig.tight_layout()
        self.confusion_canvas.draw()
    
    def _plot_performance_metrics(self, evaluation_result: EvaluationMetrics) -> None:
        """Plot performance metrics bar chart."""
        
        self.performance_ax.clear()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            evaluation_result.accuracy,
            evaluation_result.precision_macro,
            evaluation_result.recall_macro,
            evaluation_result.f1_macro
        ]
        
        bars = self.performance_ax.bar(metrics, values, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.performance_ax.text(
                bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}',
                ha='center', va='bottom', color='white'
            )
        
        self.performance_ax.set_ylim(0, 1)
        self.performance_ax.set_ylabel('Score', color='white')
        self.performance_ax.set_title('Model Performance Metrics', color='white')
        self.performance_ax.tick_params(colors='white')
        
        # Color bars based on performance
        for bar, value in zip(bars, values):
            if value >= 0.9:
                bar.set_color('#2E8B57')  # Green for excellent
            elif value >= 0.7:
                bar.set_color('#4169E1')  # Blue for good
            elif value >= 0.5:
                bar.set_color('#FF8C00')  # Orange for fair
            else:
                bar.set_color('#DC143C')  # Red for poor
        
        self.performance_fig.tight_layout()
        self.performance_canvas.draw()
    
    def _plot_quality_metrics(self, quality_metrics: QualityMetrics) -> None:
        """Plot data quality metrics."""
        
        self.quality_ax.clear()
        
        metrics = ['Overall\nQuality', 'Lexical\nDiversity', 'Class\nBalance', 'Safety\nScore']
        values = [
            quality_metrics.overall_quality,
            quality_metrics.lexical_diversity,
            quality_metrics.class_balance,
            1.0 - quality_metrics.toxicity_score  # Invert toxicity for safety score
        ]
        
        bars = self.quality_ax.bar(metrics, values, color=['#32CD32', '#4169E1', '#FF8C00', '#9932CC'])
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.quality_ax.text(
                bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}',
                ha='center', va='bottom', color='white'
            )
        
        self.quality_ax.set_ylim(0, 1)
        self.quality_ax.set_ylabel('Score', color='white')
        self.quality_ax.set_title('Data Quality Metrics', color='white')
        self.quality_ax.tick_params(colors='white')
        
        self.quality_fig.tight_layout()
        self.quality_canvas.draw()
    
    def _export_metrics(self) -> None:
        """Export metrics to file."""
        
        from tkinter import filedialog
        import json
        
        # Prepare metrics data
        export_data = {
            'timestamp': time.time(),
            'data_quality': {},
            'model_performance': {},
            'live_metrics': self.live_metrics
        }
        
        # Add generation metrics
        if self.current_generation_result:
            quality = self.current_generation_result.quality_metrics
            export_data['data_quality'] = {
                'overall_quality': quality.overall_quality,
                'vocabulary_size': quality.vocabulary_size,
                'average_length': quality.average_length,
                'class_balance': quality.class_balance,
                'duplicate_ratio': quality.duplicate_ratio,
                'bias_score': quality.bias_score,
                'distinct_2': quality.distinct_2,
                'pii_detected': quality.pii_detected
            }
        
        # Add evaluation metrics
        if self.current_evaluation_result:
            eval_result = self.current_evaluation_result
            export_data['model_performance'] = {
                'accuracy': eval_result.accuracy,
                'f1_macro': eval_result.f1_macro,
                'precision_macro': eval_result.precision_macro,
                'recall_macro': eval_result.recall_macro,
                'inference_time_ms': eval_result.inference_time_ms,
                'model_size_mb': eval_result.model_size_mb,
                'throughput_samples_per_sec': eval_result.throughput_samples_per_sec,
                'per_class_metrics': {
                    'precision': eval_result.per_class_precision,
                    'recall': eval_result.per_class_recall,
                    'f1': eval_result.per_class_f1
                }
            }
        
        # Save to file
        filename = filedialog.asksaveasfilename(
            title="Export Metrics",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                # Show success message
                from tkinter import messagebox
                messagebox.showinfo("Success", f"Metrics exported to {filename}")
                
            except Exception as e:
                from tkinter import messagebox
                messagebox.showerror("Error", f"Failed to export metrics:\n{e}")
    
    def clear_metrics(self) -> None:
        """Clear all displayed metrics."""
        
        # Clear data quality metrics
        for label in self.data_quality_labels.values():
            label.configure(text="-")
        
        # Clear model performance metrics
        for label in self.model_perf_labels.values():
            label.configure(text="-")
        
        # Clear per-class performance
        for widget in self.per_class_scroll.winfo_children()[4:]:
            widget.destroy()
        
        # Clear charts
        self._plot_empty_confusion_matrix()
        self._plot_empty_performance_chart()
        self._plot_empty_quality_chart()
        
        # Clear stored data
        self.current_generation_result = None
        self.current_evaluation_result = None
        self.live_metrics.clear()