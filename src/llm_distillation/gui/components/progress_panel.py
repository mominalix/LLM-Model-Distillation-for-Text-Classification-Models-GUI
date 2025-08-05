"""
Progress monitoring panel for tracking generation and training progress.

This component provides real-time progress bars, charts, and status
updates for data generation and model training processes.
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from ...config import Config


class ProgressPanel(ctk.CTkFrame):
    """Panel for displaying progress information."""
    
    def __init__(self, parent, config: Config, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config
        
        # Progress tracking
        self.generation_start_time: Optional[float] = None
        self.training_start_time: Optional[float] = None
        self.sample_count = 0
        self.total_samples = 0
        
        # Metrics history for charts
        self.metrics_history: Dict[str, deque] = {
            'accuracy': deque(maxlen=100),
            'f1_macro': deque(maxlen=100),
            'loss': deque(maxlen=100),
            'distinct_2': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
        
        self.grid_columnconfigure(0, weight=1)
        
        # Create progress sections
        self._create_generation_progress()
        self._create_training_progress()
        self._create_metrics_chart()
        self._create_status_display()
    
    def _create_generation_progress(self) -> None:
        """Create data generation progress section."""
        
        # Generation progress frame
        self.gen_frame = ctk.CTkFrame(self)
        self.gen_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        self.gen_frame.grid_columnconfigure(1, weight=1)
        
        # Label
        self.gen_label = ctk.CTkLabel(
            self.gen_frame,
            text="Data Generation:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.gen_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Progress bar
        self.gen_progress = ctk.CTkProgressBar(self.gen_frame)
        self.gen_progress.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.gen_progress.set(0)
        
        # Percentage label
        self.gen_percent_label = ctk.CTkLabel(self.gen_frame, text="0%")
        self.gen_percent_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Status and statistics
        self.gen_status_label = ctk.CTkLabel(
            self.gen_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.gen_status_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # Generation stats frame
        self.gen_stats_frame = ctk.CTkFrame(self.gen_frame)
        self.gen_stats_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        self.gen_stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Sample count
        self.sample_count_label = ctk.CTkLabel(
            self.gen_stats_frame,
            text="Samples: 0",
            font=ctk.CTkFont(size=11)
        )
        self.sample_count_label.grid(row=0, column=0, padx=5, pady=2)
        
        # Time elapsed
        self.gen_time_label = ctk.CTkLabel(
            self.gen_stats_frame,
            text="Time: 0s",
            font=ctk.CTkFont(size=11)
        )
        self.gen_time_label.grid(row=0, column=1, padx=5, pady=2)
        
        # Speed
        self.gen_speed_label = ctk.CTkLabel(
            self.gen_stats_frame,
            text="Speed: 0/min",
            font=ctk.CTkFont(size=11)
        )
        self.gen_speed_label.grid(row=0, column=2, padx=5, pady=2)
        
        # Cost
        self.gen_cost_label = ctk.CTkLabel(
            self.gen_stats_frame,
            text="Cost: $0.00",
            font=ctk.CTkFont(size=11)
        )
        self.gen_cost_label.grid(row=0, column=3, padx=5, pady=2)
    
    def _create_training_progress(self) -> None:
        """Create training progress section."""
        
        # Training progress frame
        self.train_frame = ctk.CTkFrame(self)
        self.train_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.train_frame.grid_columnconfigure(1, weight=1)
        
        # Label
        self.train_label = ctk.CTkLabel(
            self.train_frame,
            text="Model Training:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.train_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Progress bar
        self.train_progress = ctk.CTkProgressBar(self.train_frame)
        self.train_progress.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.train_progress.set(0)
        
        # Percentage label
        self.train_percent_label = ctk.CTkLabel(self.train_frame, text="0%")
        self.train_percent_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Status
        self.train_status_label = ctk.CTkLabel(
            self.train_frame,
            text="Waiting for data",
            font=ctk.CTkFont(size=12)
        )
        self.train_status_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # Training stats frame
        self.train_stats_frame = ctk.CTkFrame(self.train_frame)
        self.train_stats_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        self.train_stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Epoch
        self.epoch_label = ctk.CTkLabel(
            self.train_stats_frame,
            text="Epoch: 0/0",
            font=ctk.CTkFont(size=11)
        )
        self.epoch_label.grid(row=0, column=0, padx=5, pady=2)
        
        # Loss
        self.loss_label = ctk.CTkLabel(
            self.train_stats_frame,
            text="Loss: -",
            font=ctk.CTkFont(size=11)
        )
        self.loss_label.grid(row=0, column=1, padx=5, pady=2)
        
        # Accuracy
        self.accuracy_label = ctk.CTkLabel(
            self.train_stats_frame,
            text="Accuracy: -",
            font=ctk.CTkFont(size=11)
        )
        self.accuracy_label.grid(row=0, column=2, padx=5, pady=2)
        
        # F1 Score
        self.f1_label = ctk.CTkLabel(
            self.train_stats_frame,
            text="F1: -",
            font=ctk.CTkFont(size=11)
        )
        self.f1_label.grid(row=0, column=3, padx=5, pady=2)
    
    def _create_metrics_chart(self) -> None:
        """Create real-time metrics chart."""
        
        # Chart frame
        self.chart_frame = ctk.CTkFrame(self)
        self.chart_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.chart_frame.grid_columnconfigure(0, weight=1)
        
        # Chart title
        self.chart_title = ctk.CTkLabel(
            self.chart_frame,
            text="📈 Training Metrics",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.chart_title.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')  # Dark background
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(121)  # Accuracy/F1
        self.ax2 = self.fig.add_subplot(122)  # Loss
        
        # Configure axes
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        
        self.ax1.set_title('Accuracy & F1 Score', color='white')
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Training Loss', color='white')
        self.ax2.grid(True, alpha=0.3)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        # Initialize empty plots
        self.accuracy_line, = self.ax1.plot([], [], 'g-', label='Accuracy', linewidth=2)
        self.f1_line, = self.ax1.plot([], [], 'b-', label='F1-Score', linewidth=2)
        self.loss_line, = self.ax2.plot([], [], 'r-', label='Loss', linewidth=2)
        
        self.ax1.legend(loc='lower right')
        self.ax2.legend(loc='upper right')
        
        self.fig.tight_layout()
    
    def _create_status_display(self) -> None:
        """Create overall status display."""
        
        # Status frame
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        # Overall status
        self.overall_status_label = ctk.CTkLabel(
            self.status_frame,
            text="🔄 System Status: Ready",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.overall_status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Current operation
        self.current_op_label = ctk.CTkLabel(
            self.status_frame,
            text="Current Operation: Idle",
            font=ctk.CTkFont(size=12)
        )
        self.current_op_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
    
    def start_generation_progress(self) -> None:
        """Start tracking generation progress."""
        self.generation_start_time = time.time()
        self.sample_count = 0
        self.gen_progress.set(0)
        self.gen_percent_label.configure(text="0%")
        self.gen_status_label.configure(text="Starting generation...")
        self.overall_status_label.configure(text="🔄 System Status: Generating Data")
        self.current_op_label.configure(text="Current Operation: Data Generation")
        
        # Start update timer
        self._update_generation_stats()
    
    def update_generation_progress(self, progress: float, message: str) -> None:
        """Update generation progress."""
        self.gen_progress.set(progress)
        self.gen_percent_label.configure(text=f"{progress*100:.1f}%")
        self.gen_status_label.configure(text=message)
        
        # Update stats
        self._update_generation_stats()
    
    def complete_generation_progress(self) -> None:
        """Mark generation as complete."""
        self.gen_progress.set(1.0)
        self.gen_percent_label.configure(text="100%")
        self.gen_status_label.configure(text="✓ Generation completed")
        self.overall_status_label.configure(text="✅ System Status: Generation Complete")
    
    def start_training_progress(self) -> None:
        """Start tracking training progress."""
        self.training_start_time = time.time()
        self.train_progress.set(0)
        self.train_percent_label.configure(text="0%")
        self.train_status_label.configure(text="Starting training...")
        self.overall_status_label.configure(text="🔄 System Status: Training Model")
        self.current_op_label.configure(text="Current Operation: Model Training")
        
        # Clear metrics history
        for key in self.metrics_history:
            self.metrics_history[key].clear()
    
    def update_training_progress(self, progress: float, message: str) -> None:
        """Update training progress."""
        self.train_progress.set(progress)
        self.train_percent_label.configure(text=f"{progress*100:.1f}%")
        self.train_status_label.configure(text=message)
    
    def complete_training_progress(self) -> None:
        """Mark training as complete."""
        self.train_progress.set(1.0)
        self.train_percent_label.configure(text="100%")
        self.train_status_label.configure(text="✓ Training completed")
        self.overall_status_label.configure(text="✅ System Status: Training Complete")
    
    def increment_sample_count(self) -> None:
        """Increment the sample count."""
        self.sample_count += 1
        self._update_generation_stats()
    
    def update_live_generation_stats(self, stats: Dict[str, Any]) -> None:
        """Update live generation statistics."""
        
        # Update sample counts
        if 'current_samples' in stats:
            self.sample_count = stats['current_samples']
        
        if 'total_target' in stats:
            self.total_samples = stats['total_target']
        
        # Update cost if available
        if 'current_cost' in stats:
            self.gen_cost_label.configure(text=f"Cost: ${stats['current_cost']:.4f}")
        
        # Update current class being generated
        if 'current_class' in stats:
            current_class = stats['current_class']
            class_progress = stats.get('class_progress', '')
            self.gen_status_label.configure(text=f"Generating {current_class}: {class_progress}")
        
        # Update progress bar if overall progress is provided
        if 'overall_progress' in stats:
            self.gen_progress.set(stats['overall_progress'])
            self.gen_percent_label.configure(text=f"{stats['overall_progress']*100:.1f}%")
        
        # Refresh other stats
        self._update_generation_stats()
    
    def update_training_metrics(self, metrics: Dict[str, float]) -> None:
        """Update training metrics and chart."""
        
        current_time = time.time()
        
        # Extract metrics
        accuracy = metrics.get('train_accuracy', metrics.get('eval_accuracy', 0))
        f1_score = metrics.get('train_f1', metrics.get('eval_f1', 0))
        loss = metrics.get('train_loss', metrics.get('eval_loss', 0))
        epoch = metrics.get('epoch', 0)
        
        # Update labels
        self.accuracy_label.configure(text=f"Accuracy: {accuracy:.3f}")
        self.f1_label.configure(text=f"F1: {f1_score:.3f}")
        self.loss_label.configure(text=f"Loss: {loss:.3f}")
        self.epoch_label.configure(text=f"Epoch: {epoch}")
        
        # Add to history
        self.metrics_history['accuracy'].append(accuracy)
        self.metrics_history['f1_macro'].append(f1_score)
        self.metrics_history['loss'].append(loss)
        self.metrics_history['timestamps'].append(current_time)
        
        # Update chart
        self._update_metrics_chart()
    
    def _update_generation_stats(self) -> None:
        """Update generation statistics."""
        
        if self.generation_start_time is None:
            return
        
        elapsed_time = time.time() - self.generation_start_time
        
        # Update sample count
        self.sample_count_label.configure(text=f"Samples: {self.sample_count}")
        
        # Update time
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.0f}s"
        else:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{minutes}m {seconds}s"
        
        self.gen_time_label.configure(text=f"Time: {time_str}")
        
        # Update speed
        if elapsed_time > 0:
            samples_per_minute = (self.sample_count / elapsed_time) * 60
            self.gen_speed_label.configure(text=f"Speed: {samples_per_minute:.1f}/min")
        
        # Schedule next update
        if self.generation_start_time is not None:
            self.after(1000, self._update_generation_stats)
    
    def _update_metrics_chart(self) -> None:
        """Update the real-time metrics chart."""
        
        if len(self.metrics_history['timestamps']) < 2:
            return
        
        # Convert to numpy arrays
        timestamps = np.array(list(self.metrics_history['timestamps']))
        accuracy = np.array(list(self.metrics_history['accuracy']))
        f1_scores = np.array(list(self.metrics_history['f1_macro']))
        losses = np.array(list(self.metrics_history['loss']))
        
        # Normalize timestamps to start at 0
        timestamps = timestamps - timestamps[0]
        
        # Update line data
        self.accuracy_line.set_data(timestamps, accuracy)
        self.f1_line.set_data(timestamps, f1_scores)
        self.loss_line.set_data(timestamps, losses)
        
        # Update axis limits
        self.ax1.set_xlim(0, max(timestamps[-1], 1))
        self.ax1.set_ylim(0, 1)
        
        self.ax2.set_xlim(0, max(timestamps[-1], 1))
        if len(losses) > 0:
            loss_min, loss_max = np.min(losses), np.max(losses)
            loss_range = loss_max - loss_min
            self.ax2.set_ylim(max(0, loss_min - 0.1 * loss_range), loss_max + 0.1 * loss_range)
        
        # Redraw
        self.canvas.draw()
    
    def reset_progress(self) -> None:
        """Reset all progress indicators."""
        
        # Reset generation
        self.generation_start_time = None
        self.sample_count = 0
        self.gen_progress.set(0)
        self.gen_percent_label.configure(text="0%")
        self.gen_status_label.configure(text="Ready")
        self.sample_count_label.configure(text="Samples: 0")
        self.gen_time_label.configure(text="Time: 0s")
        self.gen_speed_label.configure(text="Speed: 0/min")
        self.gen_cost_label.configure(text="Cost: $0.00")
        
        # Reset training
        self.training_start_time = None
        self.train_progress.set(0)
        self.train_percent_label.configure(text="0%")
        self.train_status_label.configure(text="Waiting for data")
        self.epoch_label.configure(text="Epoch: 0/0")
        self.loss_label.configure(text="Loss: -")
        self.accuracy_label.configure(text="Accuracy: -")
        self.f1_label.configure(text="F1: -")
        
        # Reset status
        self.overall_status_label.configure(text="🔄 System Status: Ready")
        self.current_op_label.configure(text="Current Operation: Idle")
        
        # Clear metrics history
        for key in self.metrics_history:
            self.metrics_history[key].clear()
        
        # Clear chart
        self.accuracy_line.set_data([], [])
        self.f1_line.set_data([], [])
        self.loss_line.set_data([], [])
        self.canvas.draw()
    
    def update_cost_estimate(self, cost: float) -> None:
        """Update cost estimate display."""
        self.gen_cost_label.configure(text=f"Cost: ${cost:.4f}")
    
    def set_total_samples(self, total: int) -> None:
        """Set the total expected number of samples."""
        self.total_samples = total