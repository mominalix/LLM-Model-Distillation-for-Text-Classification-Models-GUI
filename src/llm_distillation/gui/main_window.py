"""
Main application window for the LLM Distillation GUI.

This module provides the primary interface for the application including
all panels, controls, and coordination between components.
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging

import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from ..config import Config, get_config, Language, OpenAIModel, HuggingFaceModel
from ..llm import ModelManager
from ..data import DataGenerator, GenerationTask, DataAugmenter, DataValidator
from ..training import ModelTrainer, TrainingConfig
from ..exceptions import DistillationError
from .components import (
    TaskInputPanel, 
    ClassManagementPanel, 
    GenerationControlsPanel,
    TrainingControlsPanel,
    TestingControlsPanel,
    ProgressPanel,
    MetricsPanel
)
from .components.collapsible_frame import CollapsibleFrame

logger = logging.getLogger(__name__)


class MainWindow:
    """Main application window for LLM Distillation."""
    
    def __init__(self, config: Optional[Config] = None):
        # Load configuration
        self.config = config or get_config()
        
        # Initialize core components
        self.model_manager = ModelManager(self.config)
        self.data_generator = DataGenerator(
            self.config, 
            self.model_manager
        )
        self.data_validator = DataValidator(self.config)
        self.data_augmenter = DataAugmenter(self.config)
        
        # Application state
        self.current_task: Optional[GenerationTask] = None
        self.current_dataset: Optional[List[Dict[str, Any]]] = None
        self.current_model_trainer: Optional[ModelTrainer] = None
        self.generation_thread: Optional[threading.Thread] = None
        self.training_thread: Optional[threading.Thread] = None
        
        # GUI state
        self.class_labels: List[str] = []
        self.generation_in_progress = False
        self.training_in_progress = False
        
        # Setup GUI
        self._setup_gui()
        
        # Apply fullscreen state after all GUI components are created
        self._apply_fullscreen_mode()
        
        self._setup_callbacks()
        
        # Set main window reference for live progress updates
        self.data_generator._main_window_ref = self
        
        logger.info("MainWindow initialized successfully")
    
    def _setup_gui(self) -> None:
        """Initialize the GUI components."""
        
        # Configure CustomTkinter
        ctk.set_appearance_mode(self.config.theme)
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("LLM Model Distillation for Text Classification")
        self.root.minsize(1000, 700)
        
        # Set initial geometry only if not going fullscreen
        # self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        
        # Note: Fullscreen state will be applied after GUI setup is complete
        
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)  # Main container - expandable
        self.root.grid_rowconfigure(1, weight=0)  # Status bar - fixed height
        
        # Create main container with two-column layout
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.main_container.grid_columnconfigure(0, weight=1)  # Left column
        self.main_container.grid_columnconfigure(1, weight=1)  # Right column
        self.main_container.grid_rowconfigure(0, weight=0)     # Title row
        self.main_container.grid_rowconfigure(1, weight=1)     # Content row
        
        # Create title
        self.title_label = ctk.CTkLabel(
            self.main_container,
            text="LLM Model Distillation for Text Classification",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(10, 15), sticky="ew")
        
        # Create left and right columns
        self.left_column = ctk.CTkScrollableFrame(self.main_container)
        self.left_column.grid(row=1, column=0, sticky="nsew", padx=(5, 2), pady=5)
        self.left_column.grid_columnconfigure(0, weight=1)
        
        self.right_column = ctk.CTkScrollableFrame(self.main_container)
        self.right_column.grid(row=1, column=1, sticky="nsew", padx=(2, 5), pady=5)
        self.right_column.grid_columnconfigure(0, weight=1)
        
        # Create status bar first (needed by callbacks)
        self._create_status_bar()
        
        # Create collapsible sections
        self._create_collapsible_sections()
    
    def _apply_fullscreen_mode(self) -> None:
        """Apply fullscreen windowed mode after GUI is fully setup."""
        try:
            # Force window to update and render all components first
            self.root.update()
            
            # Apply fullscreen state
            if self.root.tk.call('tk', 'windowingsystem') == 'win32':
                self.root.state('zoomed')
                logger.info("Applied Windows fullscreen (zoomed) state")
            else:
                self.root.attributes('-zoomed', True)
                logger.info("Applied Unix fullscreen (zoomed) attributes")
                
            # Force another update to ensure the state is applied
            self.root.update_idletasks()
            
        except Exception as e:
            logger.warning(f"Failed to apply fullscreen mode: {e}")
            # Fallback to normal window size if fullscreen fails
            try:
                self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
            except:
                self.root.geometry("1200x800")  # Fallback size
    
    def _create_collapsible_sections(self) -> None:
        """Create all sections using collapsible frames."""
        
        # Left Column Sections
        row_counter = 0
        
        # 1. Synthetic Data Generation Section (merged dataset config + generation)
        self.generation_section = CollapsibleFrame(
            self.left_column,
            title="Synthetic Data Generation",
            collapsed=True
        )
        self.generation_section.grid(row=row_counter, column=0, sticky="ew", padx=5, pady=3)
        row_counter += 1
        
        # Add combined content
        self._create_synthetic_data_content(self.generation_section.get_content_frame())
        
        # 2. Model Training Section
        self.training_section = CollapsibleFrame(
            self.left_column,
            title="Model Training",
            collapsed=True
        )
        self.training_section.grid(row=row_counter, column=0, sticky="ew", padx=5, pady=3)
        row_counter += 1
        
        # Add training content
        self._create_training_content(self.training_section.get_content_frame())
        
        # 3. Model Testing Section
        self.testing_section = CollapsibleFrame(
            self.left_column,
            title="Model Testing & Inference",
            collapsed=True  # Collapsed by default to save space
        )
        self.testing_section.grid(row=row_counter, column=0, sticky="ew", padx=5, pady=3)
        
        # Add testing content
        self._create_testing_content(self.testing_section.get_content_frame())
        
        # Right Column Sections
        row_counter = 0
        
        # 4. Progress Monitoring Section
        self.progress_section = CollapsibleFrame(
            self.right_column,
            title="Progress Monitoring",
            collapsed=False
        )
        self.progress_section.grid(row=row_counter, column=0, sticky="ew", padx=5, pady=3)
        row_counter += 1
        
        # Add progress content
        self._create_progress_content(self.progress_section.get_content_frame())
        
        # 5. Results & Metrics Section
        self.results_section = CollapsibleFrame(
            self.right_column,
            title="Results & Metrics",
            collapsed=False  # Show metrics by default
        )
        self.results_section.grid(row=row_counter, column=0, sticky="ew", padx=5, pady=3)
        
        # Add results content
        self._create_results_content(self.results_section.get_content_frame())
    
    def _create_synthetic_data_content(self, parent) -> None:
        """Create combined synthetic data generation content."""
        parent.grid_columnconfigure(0, weight=1)
        
        # Task Input Panel
        self.task_input = TaskInputPanel(parent, self.config)
        self.task_input.grid(row=0, column=0, sticky="ew", padx=5, pady=3)
        
        # Class Management Panel
        self.class_management = ClassManagementPanel(
            parent, 
            self.config,
            on_classes_changed=self._on_classes_changed
        )
        self.class_management.grid(row=1, column=0, sticky="ew", padx=5, pady=3)
        
        # Language and Settings Panel
        self.settings_frame = ctk.CTkFrame(parent)
        self.settings_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=3)
        self.settings_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Language selection with checkboxes
        language_label = ctk.CTkLabel(self.settings_frame, text="Languages (Multi-select):")
        language_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        # Language checkboxes frame
        self.language_frame = ctk.CTkFrame(self.settings_frame)
        self.language_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.language_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Language selection variables and checkboxes
        self.language_vars = {}
        self.language_checkboxes = {}
        
        language_display = {
            Language.EN: "English",
            Language.ES: "Spanish", 
            Language.FR: "French",
            Language.ZH: "Chinese",
            Language.HI: "Hindi",
            Language.AR: "Arabic"
        }
        
        # Create checkboxes in a grid layout
        for i, (lang_code, lang_name) in enumerate(language_display.items()):
            var = ctk.BooleanVar(value=(lang_code == Language.EN))  # Default English selected
            checkbox = ctk.CTkCheckBox(
                self.language_frame,
                text=lang_name,
                variable=var,
                command=self._on_language_changed
            )
            
            row = i // 3
            col = i % 3
            checkbox.grid(row=row, column=col, sticky="w", padx=5, pady=3)
            
            self.language_vars[lang_code] = var
            self.language_checkboxes[lang_code] = checkbox
        
        # Samples per class
        samples_label = ctk.CTkLabel(self.settings_frame, text="Samples per Class:")
        samples_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        self.samples_var = ctk.StringVar(value=str(self.config.default_samples_per_class))
        self.samples_entry = ctk.CTkEntry(
            self.settings_frame,
            textvariable=self.samples_var,
        )
        self.samples_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Teacher model selection
        model_label = ctk.CTkLabel(self.settings_frame, text="Teacher Model:")
        model_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        self.teacher_model_var = ctk.StringVar(value=self.config.default_teacher_model)
        self.teacher_model_dropdown = ctk.CTkOptionMenu(
            self.settings_frame,
            variable=self.teacher_model_var,
            values=[model.value for model in OpenAIModel]
        )
        self.teacher_model_dropdown.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Generation Controls Panel
        self.generation_controls = GenerationControlsPanel(
            parent, 
            self.config,
            on_start_generation=self._start_generation,
            on_stop_generation=self._stop_generation
        )
        self.generation_controls.grid(row=3, column=0, sticky="ew", padx=5, pady=3)
    
    def _create_training_content(self, parent) -> None:
        """Create model training content."""
        parent.grid_columnconfigure(0, weight=1)
        
        # Training Controls Panel
        self.training_controls = TrainingControlsPanel(
            parent, 
            self.config,
            on_start_training=self._start_training,
            on_export_model=self._export_model
        )
        self.training_controls.grid(row=0, column=0, sticky="ew", padx=5, pady=3)
    
    def _create_testing_content(self, parent) -> None:
        """Create model testing content."""
        parent.grid_columnconfigure(0, weight=1)
        
        # Testing Controls Panel
        self.testing_controls = TestingControlsPanel(
            parent, 
            self.config,
            on_model_loaded=self._on_testing_model_loaded,
            on_inference_complete=self._on_inference_complete
        )
        self.testing_controls.grid(row=0, column=0, sticky="ew", padx=5, pady=3)
    
    def _create_progress_content(self, parent) -> None:
        """Create progress monitoring content."""
        parent.grid_columnconfigure(0, weight=1)
        
        # Progress Panel
        self.progress_panel = ProgressPanel(parent, self.config)
        self.progress_panel.grid(row=0, column=0, sticky="ew", padx=5, pady=3)
    
    def _create_results_content(self, parent) -> None:
        """Create results and metrics content."""
        parent.grid_columnconfigure(0, weight=1)
        
        # Metrics Panel
        self.metrics_panel = MetricsPanel(parent, self.config)
        self.metrics_panel.grid(row=0, column=0, sticky="ew", padx=5, pady=3)

    
    def _create_generation_section(self) -> None:
        """Create the data generation section."""
        
        # Generation section frame
        self.generation_frame = ctk.CTkFrame(self.main_container)
        self.generation_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.generation_frame.grid_columnconfigure(0, weight=1)
        
        # Section title
        gen_title = ctk.CTkLabel(
            self.generation_frame,
            text="Data Generation",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        gen_title.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Generation controls
        self.generation_controls = GenerationControlsPanel(
            self.generation_frame,
            self.config,
            on_start_generation=self._start_generation,
            on_stop_generation=self._stop_generation
        )
        self.generation_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    
    def _create_training_section(self) -> None:
        """Create the model training section."""
        
        # Training section frame
        self.training_frame = ctk.CTkFrame(self.main_container)
        self.training_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.training_frame.grid_columnconfigure(0, weight=1)
        
        # Section title
        train_title = ctk.CTkLabel(
            self.training_frame,
            text="Model Training",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        train_title.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Training controls
        self.training_controls = TrainingControlsPanel(
            self.training_frame,
            self.config,
            on_start_training=self._start_training,
            on_export_model=self._export_model
        )
        self.training_controls.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    
    def _create_progress_section(self) -> None:
        """Create the progress monitoring section."""
        
        # Progress section frame
        self.progress_frame = ctk.CTkFrame(self.main_container)
        self.progress_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.progress_frame.grid_columnconfigure(0, weight=1)
        
        # Section title
        progress_title = ctk.CTkLabel(
            self.progress_frame,
            text="Progress Monitoring",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        progress_title.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Progress panel
        self.progress_panel = ProgressPanel(self.progress_frame, self.config)
        self.progress_panel.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    
    def _create_results_section(self) -> None:
        """Create the results and metrics section."""
        
        # Results section frame
        self.results_frame = ctk.CTkFrame(self.main_container)
        self.results_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Section title
        results_title = ctk.CTkLabel(
            self.results_frame,
            text="Results & Metrics",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_title.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Metrics panel
        self.metrics_panel = MetricsPanel(self.results_frame, self.config)
        self.metrics_panel.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    

    def _create_status_bar(self) -> None:
        """Create the status bar."""
        
        self.status_frame = ctk.CTkFrame(self.root, height=30)
        self.status_frame.grid(row=1, column=0, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
    
    def _setup_callbacks(self) -> None:
        """Setup callbacks for various events."""
        
        # Data generator callbacks
        self.data_generator.progress_callback = self._on_generation_progress
        self.data_generator.sample_callback = self._on_sample_generated
        
        # Window close callback
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _on_classes_changed(self, classes: List[str]) -> None:
        """Handle class labels being changed."""
        self.class_labels = classes
        logger.info(f"Class labels updated: {classes}")
        self._update_status(f"Classes: {', '.join(classes)}")
    
    def _on_language_changed(self) -> None:
        """Handle language selection change."""
        selected_languages = self.get_selected_languages()
        if not selected_languages:
            # Ensure at least one language is selected
            self.language_vars[Language.EN].set(True)
            selected_languages = [Language.EN]
        
        lang_names = [lang.value for lang in selected_languages]
        self._update_status(f"Languages: {', '.join(lang_names)}")
    
    def get_selected_languages(self) -> List[Language]:
        """Get list of selected languages."""
        selected = []
        for lang_code, var in self.language_vars.items():
            if var.get():
                selected.append(lang_code)
        return selected
    
    def _start_generation(self) -> None:
        """Start the data generation process."""
        
        if self.generation_in_progress:
            messagebox.showwarning("Warning", "Data generation is already in progress!")
            return
        
        # Validate inputs
        task_description = self.task_input.get_task_description()
        if not task_description.strip():
            messagebox.showerror("Error", "Please enter a task description!")
            return
        
        if not self.class_labels:
            messagebox.showerror("Error", "Please add at least one class label!")
            return
        
        try:
            samples_per_class = int(self.samples_var.get())
            if samples_per_class <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of samples per class!")
            return
        
        # Get selected languages
        selected_languages = self.get_selected_languages()
        if not selected_languages:
            messagebox.showerror("Error", "Please select at least one language!")
            return

        # Create generation task
        try:
            self.current_task = GenerationTask(
                task_description=task_description,
                class_labels=self.class_labels,
                samples_per_class=samples_per_class,
                languages=selected_languages,
                temperature=0.7,
                quality_threshold=self.config.quality_threshold
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create generation task: {e}")
            return
        
        # Update UI state
        self.generation_in_progress = True
        self.generation_controls.set_generation_state(True)
        self.progress_panel.start_generation_progress()
        self._update_status("Starting data generation...")
        
        # Start generation in background thread
        self.generation_thread = threading.Thread(
            target=self._run_generation,
            daemon=True
        )
        self.generation_thread.start()
    
    def _run_generation(self) -> None:
        """Run data generation in background thread."""
        
        try:
            # Select teacher model
            teacher_model = self.teacher_model_var.get()
            
            # Generate dataset
            result = self.data_generator.generate_dataset(
                task=self.current_task,
                model_name=teacher_model
            )
            
            # Store result
            self.current_dataset = result.generated_samples
            
            # Update UI on main thread
            self.root.after(0, self._on_generation_complete, result)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.root.after(0, self._on_generation_error, str(e))
    
    def _on_generation_complete(self, result) -> None:
        """Handle generation completion (including partial results)."""
        
        self.generation_in_progress = False
        self.generation_controls.set_generation_state(False)
        self.progress_panel.complete_generation_progress()
        
        # Update metrics panel
        try:
            logger.info(f"Updating metrics panel with quality score: {result.quality_metrics.overall_quality}")
            self.metrics_panel.update_generation_metrics(result)
            logger.info("Metrics panel updated successfully")
            
            # Ensure Results & Metrics section is expanded to show the quality metrics
            if self.results_section.collapsed:
                self.results_section.toggle()
                logger.info("Expanded Results & Metrics section to show quality metrics")
            
            # Force UI update to ensure metrics are visible
            self.root.update_idletasks()
                
        except Exception as e:
            logger.error(f"Failed to update metrics panel: {e}")
            self._update_status(f"Metrics update failed: {str(e)}")
        
        # Enable training for both complete and partial results
        self.training_controls.enable_training(True)
        
        # Set the generated dataset path if available
        saved_path = result.metadata.get("saved_path", "")
        if saved_path:
            self.training_controls.set_generated_dataset_path(saved_path)
        
        # Check if this is a partial result
        is_partial = result.metadata.get("is_partial_result", False)
        stop_reason = result.metadata.get("stop_reason", "completed_successfully")
        saved_path = result.metadata.get("saved_path", "")
        
        if is_partial:
            completion_ratio = result.metadata.get("completion_ratio", 0)
            self._update_status(f"Generation stopped: {len(self.current_dataset)} samples saved")
            
            messagebox.showinfo(
                "Generation Stopped", 
                f"Data generation stopped early!\n"
                f"Generated {len(self.current_dataset)} samples ({completion_ratio*100:.1f}% of target)\n"
                f"Quality score: {result.quality_metrics.overall_quality:.3f}\n"
                f"Cost: ${result.total_cost:.4f}\n"
                f"Dataset saved to: {Path(saved_path).name}\n\n"
                f"You can now proceed with training using this partial dataset!"
            )
        else:
            self._update_status(f"Generation complete: {len(self.current_dataset)} samples")
            
            messagebox.showinfo(
                "Success", 
                f"Data generation completed!\n"
                f"Generated {len(self.current_dataset)} samples\n"
                f"Quality score: {result.quality_metrics.overall_quality:.3f}\n"
                f"Cost: ${result.total_cost:.4f}"
            )
    
    def _on_generation_error(self, error_message: str) -> None:
        """Handle generation error."""
        
        self.generation_in_progress = False
        self.generation_controls.set_generation_state(False)
        self.progress_panel.reset_progress()
        
        self._update_status("Generation failed")
        messagebox.showerror("Generation Error", f"Data generation failed:\n{error_message}")
    
    def _on_generation_progress(self, progress: float, message: str) -> None:
        """Handle generation progress updates."""
        
        # Update UI on main thread
        self.root.after(0, lambda: (
            self.progress_panel.update_generation_progress(progress, message),
            self._update_status(f"Generating: {message}")
        ))
    
    def _on_sample_generated(self, sample: Dict[str, Any]) -> None:
        """Handle individual sample generation."""
        
        # Update sample count in progress panel
        self.root.after(0, lambda: self.progress_panel.increment_sample_count())
    
    def _stop_generation(self) -> None:
        """Stop the data generation process."""
        
        if not self.generation_in_progress:
            return
        
        # Request stop
        self.data_generator.stop_generation()
        self._update_status("Stopping generation...")
        
        # Note: The generation thread will handle cleanup
    
    def _load_dataset_from_path(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from a folder path."""
        from pathlib import Path
        import json
        
        dataset_folder = Path(dataset_path)
        samples_file = dataset_folder / "samples.jsonl"
        
        if not samples_file.exists():
            raise FileNotFoundError(f"samples.jsonl not found in {dataset_folder}")
        
        samples = []
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    if 'text' in sample and 'label' in sample:
                        samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        if not samples:
            raise ValueError(f"No valid samples found in {samples_file}")
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_folder}")
        return samples
    
    def _start_training(self) -> None:
        """Start the model training process."""
        
        try:
            logger.info("Start training button clicked")
            self._update_status("Preparing to start training...")
            
            if self.training_in_progress:
                messagebox.showwarning("Warning", "Training is already in progress!")
                return
            
            # Get training configuration (includes dataset path)
            training_config_dict = self.training_controls.get_training_config()
            dataset_path = training_config_dict.get('dataset_path')
            
            # Check for dataset availability (either generated or manually selected)
            dataset_to_use = None
            if self.current_dataset:
                # Use in-memory dataset from generation
                dataset_to_use = self.current_dataset
            elif dataset_path:
                # Load dataset from selected folder
                try:
                    dataset_to_use = self._load_dataset_from_path(dataset_path)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load dataset from {dataset_path}:\n{str(e)}")
                    return
            
            if not dataset_to_use:
                messagebox.showerror("Error", "No dataset available! Please generate data or select a dataset folder.")
                return
            
            # Get student model
            student_model = self.training_controls.get_student_model()
            
            # Create training configuration using values from training controls
            task_name = "dataset_training"  # Default name for manual datasets
            if hasattr(self, 'current_task') and self.current_task:
                task_name = self.current_task.task_description[:50]
            
            # Update config with user settings (for early stopping patience)
            user_epochs = training_config_dict.get('epochs', self.config.num_epochs)
            # Set early stopping patience to reasonable value (25% of epochs, min 2, max 10)
            early_stopping_patience = max(2, min(10, int(user_epochs * 0.25)))
            self.config.early_stopping_patience = early_stopping_patience
            
            training_config = TrainingConfig(
                task_name=task_name,
                model_name=student_model,
                output_dir=str(self.config.output_dir / "training"),
                num_epochs=user_epochs,
                batch_size=training_config_dict.get('batch_size', self.config.batch_size),
                learning_rate=training_config_dict.get('learning_rate', self.config.learning_rate),
                use_distillation=True,
                teacher_model=None,  # Use LLM teacher
                distillation_temperature=training_config_dict.get('distillation_temperature', self.config.distillation_temperature),
                distillation_alpha=training_config_dict.get('distillation_alpha', self.config.distillation_alpha)
            )
            
            # Initialize trainer
            self.current_model_trainer = ModelTrainer(
                self.config,
                training_config,
                self.model_manager
            )
            
            # Store dataset for training thread
            self.dataset_for_training = dataset_to_use
            
            # Update UI state
            self.training_in_progress = True
            self.training_controls.set_training_state(True)
            self.progress_panel.start_training_progress()
            self._update_status("Starting model training...")
            
            # Start training in background thread
            self.training_thread = threading.Thread(
                target=self._run_training,
                daemon=True
            )
            self.training_thread.start()
            logger.info("Training thread started successfully")
            
        except Exception as e:
            logger.error(f"Error in _start_training: {e}")
            self._update_status("Training failed to start")
            messagebox.showerror("Training Error", f"Failed to start training:\n{str(e)}")
    
    def _run_training(self) -> None:
        """Run model training in background thread."""
        
        try:
            # Prepare data
            self.root.after(0, lambda: self._update_status("Preparing training data..."))
            
            # Use the dataset selected for training (either generated or manually selected)
            training_dataset = getattr(self, 'dataset_for_training', self.current_dataset)
            if not training_dataset:
                raise ValueError("No dataset available for training")
            
            dataset = self.current_model_trainer.prepare_data(
                training_dataset,
                validate_data=True
            )
            
            # Train model
            self.root.after(0, lambda: self._update_status("Training model..."))
            
            result = self.current_model_trainer.train(
                progress_callback=self._on_training_progress,
                metrics_callback=self._on_training_metrics
            )
            
            # Evaluate model
            self.root.after(0, lambda: self._update_status("Evaluating model..."))
            
            evaluation_result = self.current_model_trainer.evaluate()
            
            # Update UI on main thread
            self.root.after(0, self._on_training_complete, result, evaluation_result)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.root.after(0, self._on_training_error, str(e))
    
    def _on_training_complete(self, training_result, evaluation_result) -> None:
        """Handle training completion."""
        
        self.training_in_progress = False
        self.training_controls.set_training_state(False)
        self.progress_panel.complete_training_progress()
        
        # Update metrics panel
        self.metrics_panel.update_training_metrics(training_result, evaluation_result)
        
        # Enable model export
        self.training_controls.enable_export(True)
        
        self._update_status("Training complete")
        
        messagebox.showinfo(
            "Success",
            f"Model training completed!\n"
            f"Final F1 Score: {evaluation_result.f1_macro:.4f}\n"
            f"Accuracy: {evaluation_result.accuracy:.4f}\n"
            f"Training time: {training_result.get('total_training_time', 0):.1f}s"
        )
    
    def _on_training_error(self, error_message: str) -> None:
        """Handle training error."""
        
        self.training_in_progress = False
        self.training_controls.set_training_state(False)
        self.progress_panel.reset_progress()
        
        self._update_status("Training failed")
        messagebox.showerror("Training Error", f"Model training failed:\n{error_message}")
    
    def _on_training_progress(self, progress: float, message: str) -> None:
        """Handle training progress updates."""
        
        self.root.after(0, lambda: (
            self.progress_panel.update_training_progress(progress, message),
            self._update_status(f"Training: {message}")
        ))
    
    def _on_training_metrics(self, metrics: Dict[str, float]) -> None:
        """Handle training metrics updates."""
        
        self.root.after(0, lambda: (
            self.metrics_panel.update_live_metrics(metrics),
            self.progress_panel.update_training_metrics(metrics)  # Update real-time charts
        ))
    
    def _export_model(self) -> None:
        """Export the trained model."""
        
        if not self.current_model_trainer:
            messagebox.showerror("Error", "No trained model available!")
            return
        
        # Choose export directory
        export_dir = filedialog.askdirectory(title="Choose Export Directory")
        if not export_dir:
            return
        
        try:
            self._update_status("Exporting model...")
            
            # Export model in multiple formats
            exported_paths = self.current_model_trainer.export_model(
                export_path=Path(export_dir),
                formats=["transformers", "safetensors", "onnx"]
            )
            
            self._update_status("Model exported successfully")
            
            messagebox.showinfo(
                "Success",
                f"Model exported successfully to:\n{export_dir}\n\n"
                f"Formats: {', '.join(exported_paths.keys())}"
            )
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            messagebox.showerror("Export Error", f"Model export failed:\n{e}")
    
    def _on_testing_model_loaded(self, model_path: str) -> None:
        """Handle testing model being loaded."""
        self._update_status(f"Testing model loaded: {Path(model_path).name}")
        logger.info(f"Testing model loaded from: {model_path}")
        
        # Expand testing section if it's collapsed
        if self.testing_section.collapsed:
            self.testing_section.toggle()
    
    def _on_inference_complete(self, results: List[Any]) -> None:
        """Handle inference completion."""
        num_results = len(results)
        avg_confidence = sum(r.confidence for r in results) / num_results if results else 0
        
        self._update_status(f"Inference complete: {num_results} samples, avg confidence: {avg_confidence:.3f}")
        logger.info(f"Inference completed on {num_results} samples")
        
        # Show results section if it's collapsed
        if self.results_section.collapsed:
            self.results_section.toggle()
    



    def _load_dataset(self) -> None:
        """Load dataset from file."""
        
        filetypes = [
            ("JSONL files", "*.jsonl"),
            ("JSON files", "*.json"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Load Dataset",
            filetypes=filetypes
        )
        
        if filename:
            try:
                from ..data import DataProcessor
                processor = DataProcessor(self.config)
                self.current_dataset = processor.load_dataset_from_file(filename)
                
                # Update UI
                self.training_controls.enable_training(True)
                self._update_status(f"Loaded {len(self.current_dataset)} samples from {Path(filename).name}")
                
                messagebox.showinfo("Success", f"Loaded {len(self.current_dataset)} samples")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
    
    def _save_dataset(self) -> None:
        """Save current dataset to file."""
        
        if not self.current_dataset:
            messagebox.showerror("Error", "No dataset to save!")
            return
        
        filetypes = [
            ("JSONL files", "*.jsonl"),
            ("JSON files", "*.json"),
            ("CSV files", "*.csv")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Dataset",
            filetypes=filetypes,
            defaultextension=".jsonl"
        )
        
        if filename:
            try:
                from ..data import DataProcessor
                processor = DataProcessor(self.config)
                
                # Determine format from extension
                ext = Path(filename).suffix.lower()
                format_map = {'.jsonl': 'jsonl', '.json': 'json', '.csv': 'csv'}
                format_name = format_map.get(ext, 'jsonl')
                
                processor.save_samples(self.current_dataset, filename, format_name)
                
                self._update_status(f"Dataset saved to {Path(filename).name}")
                messagebox.showinfo("Success", f"Dataset saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save dataset:\n{e}")
    
    def _load_model(self) -> None:
        """Load a trained model."""
        
        model_dir = filedialog.askdirectory(title="Select Model Directory")
        if not model_dir:
            return
        
        try:
            # Create a dummy training config for loading
            training_config = TrainingConfig(
                model_name="unknown",
                output_dir=model_dir
            )
            
            self.current_model_trainer = ModelTrainer(
                self.config,
                training_config,
                self.model_manager
            )
            
            self.current_model_trainer.load_model(model_dir)
            
            # Enable export
            self.training_controls.enable_export(True)
            
            self._update_status(f"Model loaded from {Path(model_dir).name}")
            messagebox.showinfo("Success", f"Model loaded from {model_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
    
    def _update_status(self, message: str) -> None:
        """Update the status bar."""
        self.status_label.configure(text=message)
        logger.info(f"Status: {message}")
    
    def _on_closing(self) -> None:
        """Handle application closing."""
        
        if self.generation_in_progress or self.training_in_progress:
            if messagebox.askokcancel(
                "Quit", 
                "Operations are in progress. Do you want to quit anyway?"
            ):
                # Stop operations
                if self.generation_in_progress:
                    self.data_generator.stop_generation()
                
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self) -> None:
        """Start the GUI application."""
        
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI error: {e}")
            messagebox.showerror("Application Error", f"An error occurred:\n{e}")


def main() -> None:
    """Main entry point for the GUI application."""
    
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and run application
        app = MainWindow()
        app.run()
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()