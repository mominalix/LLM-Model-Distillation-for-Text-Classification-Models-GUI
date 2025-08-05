"""
Training controls panel for managing model training process.

This component provides controls for model selection, training configuration,
and model export functionality.
"""

from typing import Callable, Optional
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path

from ...config import Config, HuggingFaceModel


class TrainingControlsPanel(ctk.CTkFrame):
    """Panel for model training controls."""
    
    def __init__(
        self,
        parent,
        config: Config,
        on_start_training: Optional[Callable[[], None]] = None,
        on_export_model: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        self.config = config
        self.on_start_training = on_start_training
        self.on_export_model = on_export_model
        
        self.training_active = False
        self.training_enabled = False
        self.export_enabled = False
        
        self.grid_columnconfigure(0, weight=1)
        
        # Model selection frame
        self.model_frame = ctk.CTkFrame(self)
        self.model_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.model_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Student model selection
        self.student_label = ctk.CTkLabel(
            self.model_frame,
            text="Student Model:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.student_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Add "custom" option to model list
        model_options = [model.value for model in HuggingFaceModel] + ["custom"]
        self.student_model_var = ctk.StringVar(value=self.config.default_student_model)
        self.student_dropdown = ctk.CTkOptionMenu(
            self.model_frame,
            variable=self.student_model_var,
            values=model_options,
            command=self._on_model_changed
        )
        self.student_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Custom model URL input (initially hidden)
        self.custom_model_frame = ctk.CTkFrame(self.model_frame)
        self.custom_model_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.custom_model_frame.grid_remove()  # Hide initially
        
        self.custom_url_label = ctk.CTkLabel(
            self.custom_model_frame,
            text="Hugging Face Model URL/Name:",
            font=ctk.CTkFont(size=12)
        )
        self.custom_url_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        
        self.custom_url_var = ctk.StringVar()
        self.custom_url_entry = ctk.CTkEntry(
            self.custom_model_frame,
            textvariable=self.custom_url_var,
            width=300,
            placeholder_text="e.g., distilbert-base-uncased or username/model-name"
        )
        self.custom_url_entry.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        
        # Validation button for custom models
        self.validate_model_button = ctk.CTkButton(
            self.custom_model_frame,
            text="🔍 Validate Model",
            command=self._validate_custom_model,
            height=30,
            fg_color="orange",
            hover_color="darkorange"
        )
        self.validate_model_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Validation status
        self.model_validation_status = ctk.CTkLabel(
            self.custom_model_frame,
            text="",
            font=ctk.CTkFont(size=10)
        )
        self.model_validation_status.grid(row=3, column=0, padx=5, pady=2)
        
        # Custom model upload
        self.custom_frame = ctk.CTkFrame(self.model_frame)
        self.custom_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=5, pady=5)
        self.custom_frame.grid_columnconfigure(0, weight=1)
        
        self.custom_label = ctk.CTkLabel(
            self.custom_frame,
            text="Or Upload Custom Model:",
            font=ctk.CTkFont(size=12)
        )
        self.custom_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.upload_button = ctk.CTkButton(
            self.custom_frame,
            text="📁 Browse Model",
            command=self._browse_custom_model
        )
        self.upload_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.custom_path_label = ctk.CTkLabel(
            self.custom_frame,
            text="No custom model selected",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.custom_path_label.grid(row=2, column=0, padx=5, pady=5)
        
        # Dataset selection frame
        self.dataset_frame = ctk.CTkFrame(self)
        self.dataset_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.dataset_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Dataset selection label
        self.dataset_label = ctk.CTkLabel(
            self.dataset_frame,
            text="📂 Dataset Source:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.dataset_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Current dataset display
        self.current_dataset_var = ctk.StringVar(value="No dataset selected")
        self.current_dataset_label = ctk.CTkLabel(
            self.dataset_frame,
            textvariable=self.current_dataset_var,
            font=ctk.CTkFont(size=12),
            text_color="gray",
            wraplength=400
        )
        self.current_dataset_label.grid(row=1, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        
        # Dataset browser button
        self.browse_dataset_button = ctk.CTkButton(
            self.dataset_frame,
            text="📁 Browse Dataset Folder",
            command=self._on_browse_dataset,
            height=35
        )
        self.browse_dataset_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Use generated data button (for current session data)
        self.use_generated_button = ctk.CTkButton(
            self.dataset_frame,
            text="✨ Use Generated Data",
            command=self._on_use_generated,
            height=35,
            state="disabled"  # Enabled when generation completes
        )
        self.use_generated_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Store dataset path for training
        self.selected_dataset_path: Optional[str] = None
        
        # Training configuration frame
        self.training_config_frame = ctk.CTkFrame(self)
        self.training_config_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.training_config_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Epochs
        self.epochs_label = ctk.CTkLabel(self.training_config_frame, text="Epochs:")
        self.epochs_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.epochs_var = ctk.StringVar(value=str(self.config.num_epochs))
        self.epochs_entry = ctk.CTkEntry(
            self.training_config_frame,
            textvariable=self.epochs_var,
            width=80,

        )
        self.epochs_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Batch size
        self.batch_label = ctk.CTkLabel(self.training_config_frame, text="Batch Size:")
        self.batch_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.batch_var = ctk.StringVar(value=str(self.config.batch_size))
        self.batch_entry = ctk.CTkEntry(
            self.training_config_frame,
            textvariable=self.batch_var,
            width=80,

        )
        self.batch_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Learning rate
        self.lr_label = ctk.CTkLabel(self.training_config_frame, text="Learning Rate:")
        self.lr_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        self.lr_var = ctk.StringVar(value=str(self.config.learning_rate))
        self.lr_entry = ctk.CTkEntry(
            self.training_config_frame,
            textvariable=self.lr_var,
            width=100,

        )
        self.lr_entry.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        
        # Distillation settings frame
        self.distillation_frame = ctk.CTkFrame(self)
        self.distillation_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.distillation_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Distillation temperature
        self.dist_temp_label = ctk.CTkLabel(self.distillation_frame, text="Distillation Temperature:")
        self.dist_temp_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.dist_temp_slider = ctk.CTkSlider(
            self.distillation_frame,
            from_=1.0,
            to=10.0,
            number_of_steps=18,
            command=self._on_temp_changed
        )
        self.dist_temp_slider.set(self.config.distillation_temperature)
        self.dist_temp_slider.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.dist_temp_value = ctk.CTkLabel(
            self.distillation_frame,
            text=f"{self.config.distillation_temperature:.1f}"
        )
        self.dist_temp_value.grid(row=2, column=0, padx=5, pady=5)
        
        # Distillation alpha
        self.dist_alpha_label = ctk.CTkLabel(self.distillation_frame, text="Distillation Weight:")
        self.dist_alpha_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.dist_alpha_slider = ctk.CTkSlider(
            self.distillation_frame,
            from_=0.1,
            to=0.9,
            number_of_steps=8,
            command=self._on_alpha_changed
        )
        self.dist_alpha_slider.set(self.config.distillation_alpha)
        self.dist_alpha_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.dist_alpha_value = ctk.CTkLabel(
            self.distillation_frame,
            text=f"{self.config.distillation_alpha:.1f}"
        )
        self.dist_alpha_value.grid(row=2, column=1, padx=5, pady=5)
        
        # Control buttons frame - compact layout
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=3)
        self.buttons_frame.grid_columnconfigure(0, weight=1)
        
        # Primary button
        self.train_button = ctk.CTkButton(
            self.buttons_frame,
            text="🎓 Start Training",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=35,
            state="disabled",
            command=self._on_train_clicked
        )
        self.train_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Secondary buttons row
        self.secondary_buttons = ctk.CTkFrame(self.buttons_frame)
        self.secondary_buttons.grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        self.secondary_buttons.grid_columnconfigure((0, 1), weight=1)
        
        # Evaluate button - compact
        self.evaluate_button = ctk.CTkButton(
            self.secondary_buttons,
            text="📊 Evaluate",
            font=ctk.CTkFont(size=11),
            height=30,
            state="disabled",
            command=self._on_evaluate_clicked,
            fg_color="purple",
            hover_color="darkmagenta"
        )
        self.evaluate_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        # Export button - compact
        self.export_button = ctk.CTkButton(
            self.secondary_buttons,
            text="💾 Export",
            font=ctk.CTkFont(size=11),
            height=30,
            state="disabled",
            command=self._on_export_clicked,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.export_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        # Status frame
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.training_status_label = ctk.CTkLabel(
            self.status_frame,
            text="Waiting for dataset...",
            font=ctk.CTkFont(size=12)
        )
        self.training_status_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Model info display
        self.model_info_frame = ctk.CTkFrame(self)
        self.model_info_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.model_info_frame.grid_columnconfigure(0, weight=1)
        
        self.model_info_label = ctk.CTkLabel(
            self.model_info_frame,
            text="Select a model to see details",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.model_info_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Initialize model info
        self._update_model_info()
    
    def _on_model_changed(self, model_name: str) -> None:
        """Handle student model selection change."""
        if model_name == "custom":
            self.custom_model_frame.grid()  # Show custom URL input
        else:
            self.custom_model_frame.grid_remove()  # Hide custom URL input
            self.model_validation_status.configure(text="")  # Clear validation status
        self._update_model_info()
    
    def _on_temp_changed(self, value: float) -> None:
        """Handle distillation temperature change."""
        self.dist_temp_value.configure(text=f"{value:.1f}")
    
    def _on_alpha_changed(self, value: float) -> None:
        """Handle distillation alpha change."""
        self.dist_alpha_value.configure(text=f"{value:.1f}")
    
    def _browse_custom_model(self) -> None:
        """Browse for custom model directory."""
        model_dir = filedialog.askdirectory(title="Select Model Directory")
        if model_dir:
            self.custom_path_label.configure(text=f"Selected: {model_dir}")
            # TODO: Validate model directory
    
    def _on_browse_dataset(self) -> None:
        """Browse for dataset folder."""
        from pathlib import Path
        
        # Start browsing from datasets directory if it exists
        initial_dir = self.config.datasets_dir if self.config.datasets_dir.exists() else None
        
        dataset_dir = filedialog.askdirectory(
            title="Select Dataset Folder",
            initialdir=initial_dir
        )
        
        if dataset_dir:
            dataset_path = Path(dataset_dir)
            
            # Debug: Show what files are in the folder
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Selected dataset directory: {dataset_path}")
            try:
                files_in_dir = list(dataset_path.iterdir())
                logger.info(f"Files in directory: {[f.name for f in files_in_dir]}")
            except Exception as e:
                logger.error(f"Could not list directory contents: {e}")
            
            if self._validate_dataset_folder(dataset_path):
                self.selected_dataset_path = str(dataset_path)
                self.current_dataset_var.set(f"📁 {dataset_path.name}")
                self.enable_training(True)
                messagebox.showinfo(
                    "Dataset Selected",
                    f"Dataset folder selected successfully!\n"
                    f"Path: {dataset_path.name}\n"
                    f"✅ Ready to start training"
                )
            else:
                # More detailed error message
                try:
                    files_in_dir = [f.name for f in dataset_path.iterdir() if f.is_file()]
                    messagebox.showerror(
                        "Invalid Dataset",
                        f"The selected folder does not contain a valid dataset.\n\n"
                        f"Expected files: samples.jsonl, metadata.json\n"
                        f"Found files: {', '.join(files_in_dir) if files_in_dir else 'None'}\n\n"
                        f"Selected path: {dataset_path}\n\n"
                        f"Check the logs for more detailed validation information."
                    )
                except Exception as e:
                    messagebox.showerror(
                        "Invalid Dataset",
                        f"The selected folder does not contain a valid dataset.\n"
                        f"Selected: {dataset_path}\n"
                        f"Error: {str(e)}"
                    )
    
    def _on_use_generated(self) -> None:
        """Use the most recently generated dataset."""
        # This will be called when generation completes
        # For now, just find the most recent dataset folder
        from pathlib import Path
        import os
        
        datasets_dir = self.config.datasets_dir
        if not datasets_dir.exists():
            messagebox.showwarning("No Datasets", "No datasets folder found.")
            return
        
        # Find the most recent dataset folder
        dataset_folders = [d for d in datasets_dir.iterdir() if d.is_dir()]
        if not dataset_folders:
            messagebox.showwarning("No Datasets", "No dataset folders found.")
            return
        
        # Sort by creation time, most recent first
        latest_dataset = max(dataset_folders, key=lambda d: d.stat().st_ctime)
        
        if self._validate_dataset_folder(latest_dataset):
            self.selected_dataset_path = str(latest_dataset)
            self.current_dataset_var.set(f"✨ {latest_dataset.name}")
            self.enable_training(True)
            messagebox.showinfo(
                "Dataset Selected",
                f"Latest generated dataset selected!\n"
                f"Path: {latest_dataset.name}\n"
                f"✅ Ready to start training"
            )
        else:
            messagebox.showerror(
                "Invalid Dataset",
                f"The latest dataset folder is not valid.\n"
                f"Path: {latest_dataset}"
            )
    
    def _validate_dataset_folder(self, dataset_path: Path) -> bool:
        """Validate that a folder contains a valid dataset."""
        import logging
        logger = logging.getLogger(__name__)
        
        required_files = ["samples.jsonl", "metadata.json"]
        
        logger.info(f"Validating dataset folder: {dataset_path}")
        
        for file_name in required_files:
            file_path = dataset_path / file_name
            if not file_path.exists():
                logger.warning(f"Required file missing: {file_path}")
                return False
            logger.info(f"Found required file: {file_path}")
        
        # Check if samples.jsonl has content
        samples_file = dataset_path / "samples.jsonl"
        try:
            with open(samples_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    logger.warning(f"samples.jsonl is empty: {samples_file}")
                    return False
                # Try to parse the first line as JSON
                import json
                json.loads(first_line)
                logger.info(f"Dataset folder validation successful: {dataset_path}")
                return True
        except Exception as e:
            logger.warning(f"Failed to parse samples.jsonl: {e}")
            return False
    
    def _validate_custom_model(self) -> None:
        """Validate a custom Hugging Face model for training compatibility."""
        model_name = self.custom_url_var.get().strip()
        
        if not model_name:
            self.model_validation_status.configure(
                text="❌ Please enter a model name or URL",
                text_color="red"
            )
            return
        
        self.validate_model_button.configure(text="🔄 Validating...", state="disabled")
        self.model_validation_status.configure(text="Checking model...", text_color="orange")
        
        def validate_in_thread():
            try:
                # Import here to avoid slow startup
                from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
                
                # Check if model exists and get config
                try:
                    config = AutoConfig.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Check if model supports sequence classification
                    model_type = config.model_type if hasattr(config, 'model_type') else "unknown"
                    
                    # Try to load the model for sequence classification
                    try:
                        # Import the utility function
                        from ...training.distillation import load_model_for_classification
                        
                        # Create dummy label mappings for testing
                        test_label_to_id = {"negative": 0, "positive": 1}
                        test_id_to_label = {0: "negative", 1: "positive"}
                        
                        # Don't actually download, just check if it's possible
                        model = load_model_for_classification(
                            model_name,
                            num_labels=2,
                            label_to_id=test_label_to_id,
                            id_to_label=test_id_to_label
                        )
                        
                        # If we get here, the model is compatible
                        self.root.after(0, lambda: self._validation_success(model_name, model_type))
                        
                    except Exception as model_error:
                        # Check if it's a loading error vs compatibility error
                        error_str = str(model_error).lower()
                        if "not found" in error_str or "does not exist" in error_str:
                            self.root.after(0, lambda: self._validation_error(f"Model not found: {model_name}"))
                        else:
                            self.root.after(0, lambda: self._validation_error(f"Model not compatible with text classification: {str(model_error)[:100]}..."))
                
                except Exception as config_error:
                    error_str = str(config_error).lower()
                    if "not found" in error_str or "does not exist" in error_str:
                        self.root.after(0, lambda: self._validation_error(f"Model not found: {model_name}"))
                    else:
                        self.root.after(0, lambda: self._validation_error(f"Failed to load model config: {str(config_error)[:100]}..."))
                        
            except Exception as e:
                self.root.after(0, lambda: self._validation_error(f"Validation failed: {str(e)[:100]}..."))
        
        # Run validation in background thread
        import threading
        validation_thread = threading.Thread(target=validate_in_thread, daemon=True)
        validation_thread.start()
    
    def _validation_success(self, model_name: str, model_type: str) -> None:
        """Handle successful model validation."""
        self.model_validation_status.configure(
            text=f"✅ {model_name} ({model_type}) - Compatible with text classification",
            text_color="green"
        )
        self.validate_model_button.configure(text="🔍 Validate Model", state="normal")
        
        # Enable training since we have a valid custom model
        self.enable_training(True)
    
    def _validation_error(self, error_message: str) -> None:
        """Handle model validation error."""
        self.model_validation_status.configure(
            text=f"❌ {error_message}",
            text_color="red"
        )
        self.validate_model_button.configure(text="🔍 Validate Model", state="normal")
    
    def _on_train_clicked(self) -> None:
        """Handle start training button click."""
        if not self._validate_training_config():
            return
        
        if self.on_start_training:
            self.on_start_training()
    
    def _on_evaluate_clicked(self) -> None:
        """Handle evaluate button click."""
        # TODO: Implement evaluation
        messagebox.showinfo("Info", "Model evaluation feature coming soon!")
    
    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        if self.on_export_model:
            self.on_export_model()
    
    def _validate_training_config(self) -> bool:
        """Validate training configuration."""
        try:
            epochs = int(self.epochs_var.get())
            if epochs <= 0:
                raise ValueError("Epochs must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of epochs!")
            return False
        
        try:
            batch_size = int(self.batch_var.get())
            if batch_size <= 0:
                raise ValueError("Batch size must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid batch size!")
            return False
        
        try:
            lr = float(self.lr_var.get())
            if lr <= 0:
                raise ValueError("Learning rate must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid learning rate!")
            return False
        
        return True
    
    def _update_model_info(self) -> None:
        """Update model information display."""
        model_name = self.student_model_var.get()
        
        # Model specifications (simplified)
        model_specs = {
            HuggingFaceModel.DISTILBERT_BASE: "66M params, 250MB, Fast inference",
            HuggingFaceModel.TINYBERT_4L: "14M params, 60MB, Ultra fast",
            HuggingFaceModel.ALBERT_BASE: "12M params, 45MB, Memory efficient",
            HuggingFaceModel.BERT_MINI: "11M params, 42MB, Fastest training",
            HuggingFaceModel.SMOLLM_135M: "135M params, 500MB, High quality",
            HuggingFaceModel.SMOLLM_360M: "360M params, 1.4GB, Best performance",
        }
        
        info = model_specs.get(model_name, "Custom model selected")
        self.model_info_label.configure(text=f"📋 {model_name}: {info}")
    
    def enable_training(self, enabled: bool) -> None:
        """Enable or disable training controls."""
        self.training_enabled = enabled
        state = "normal" if enabled else "disabled"
        
        self.train_button.configure(state=state)
        
        if enabled:
            self.training_status_label.configure(text="✓ Ready to train")
        else:
            self.training_status_label.configure(text="Waiting for dataset...")
    
    def enable_export(self, enabled: bool) -> None:
        """Enable or disable model export."""
        self.export_enabled = enabled
        state = "normal" if enabled else "disabled"
        
        self.export_button.configure(state=state)
        self.evaluate_button.configure(state=state)
    
    def set_training_state(self, is_training: bool) -> None:
        """Update UI based on training state."""
        self.training_active = is_training
        
        if is_training:
            self.train_button.configure(state="disabled", text="🔄 Training...")
            self.training_status_label.configure(text="🔄 Training in progress...")
            
            # Disable configuration controls during training
            self.student_dropdown.configure(state="disabled")
            self.epochs_entry.configure(state="disabled")
            self.batch_entry.configure(state="disabled")
            self.lr_entry.configure(state="disabled")
            self.dist_temp_slider.configure(state="disabled")
            self.dist_alpha_slider.configure(state="disabled")
        else:
            self.train_button.configure(
                state="normal" if self.training_enabled else "disabled",
                text="🎓 Start Training"
            )
            self.training_status_label.configure(text="✓ Ready to train" if self.training_enabled else "Waiting for dataset...")
            
            # Re-enable configuration controls
            self.student_dropdown.configure(state="normal")
            self.epochs_entry.configure(state="normal")
            self.batch_entry.configure(state="normal")
            self.lr_entry.configure(state="normal")
            self.dist_temp_slider.configure(state="normal")
            self.dist_alpha_slider.configure(state="normal")
    
    def get_student_model(self) -> str:
        """Get selected student model."""
        if self.student_model_var.get() == "custom":
            return self.custom_url_var.get().strip()
        return self.student_model_var.get()
    
    def get_training_config(self) -> dict:
        """Get current training configuration."""
        return {
            'student_model': self.get_student_model(),
            'epochs': int(self.epochs_var.get()),
            'batch_size': int(self.batch_var.get()),
            'learning_rate': float(self.lr_var.get()),
            'distillation_temperature': self.dist_temp_slider.get(),
            'distillation_alpha': self.dist_alpha_slider.get(),
            'dataset_path': self.selected_dataset_path,
        }
    
    def update_status(self, message: str) -> None:
        """Update training status message."""
        self.training_status_label.configure(text=message)
    
    def enable_generated_data_button(self, enable: bool = True) -> None:
        """Enable or disable the 'Use Generated Data' button."""
        self.use_generated_button.configure(state="normal" if enable else "disabled")
    
    def set_generated_dataset_path(self, dataset_path: str) -> None:
        """Set the path of the most recently generated dataset."""
        self.selected_dataset_path = dataset_path
        dataset_name = Path(dataset_path).name
        self.current_dataset_var.set(f"✨ {dataset_name}")
        self.enable_generated_data_button(True)
        self.enable_training(True)
    
    def reset(self) -> None:
        """Reset the panel to initial state."""
        self.enable_training(False)
        self.enable_export(False)
        self.set_training_state(False)
        self.custom_path_label.configure(text="No custom model selected")