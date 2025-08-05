"""
Testing controls panel for model inference and evaluation.

This component provides controls for loading trained models, performing
inference on individual samples or batches, and evaluating model performance.
"""

import os
import threading
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
import customtkinter as ctk
from tkinter import filedialog, messagebox, StringVar
import time

from ...config import Config
from ...testing import ModelTester, InferenceConfig, InferenceResult, BatchInferenceManager


class TestingControlsPanel(ctk.CTkFrame):
    """Panel for model testing controls."""
    
    def __init__(
        self,
        parent,
        config: Config,
        on_model_loaded: Optional[Callable[[str], None]] = None,
        on_inference_complete: Optional[Callable[[List[InferenceResult]], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        self.config = config
        self.on_model_loaded = on_model_loaded
        self.on_inference_complete = on_inference_complete
        
        # Testing components
        self.model_tester: Optional[ModelTester] = None
        self.batch_manager: Optional[BatchInferenceManager] = None
        
        # UI state
        self.model_loaded = False
        self.inference_in_progress = False
        self.current_results: List[InferenceResult] = []
        
        self.grid_columnconfigure(0, weight=1)
        
        self._create_model_selection()
        self._create_single_inference()
        self._create_batch_inference()
        self._create_results_display()
        
        # Initialize model tester
        self._initialize_tester()
        
        # Load available models on startup
        self._refresh_available_models()
        
        # Placeholder text management
        self.placeholder_text = "Type or paste your text here..."
        self.text_has_placeholder = True
    
    def _initialize_tester(self) -> None:
        """Initialize the model tester."""
        inference_config = InferenceConfig(
            use_gpu=self.config.use_gpu,
            batch_size=self.config.batch_size,
            max_length=self.config.max_sequence_length,
            return_probabilities=True,
            return_confidence=True,
            top_k_predictions=3
        )
        
        self.model_tester = ModelTester(self.config, inference_config)
    
    def _create_model_selection(self) -> None:
        """Create model selection section."""
        # Model Selection Frame
        self.model_frame = ctk.CTkFrame(self)
        self.model_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.model_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self.model_frame,
            text="Model Selection",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(10, 5), sticky="w", padx=10)
        
        # Available models dropdown
        models_label = ctk.CTkLabel(self.model_frame, text="Available Models:")
        models_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.models_var = StringVar(value="Select a model...")
        self.models_dropdown = ctk.CTkOptionMenu(
            self.model_frame,
            variable=self.models_var,
            values=["No models found"],
            command=self._on_model_selected
        )
        self.models_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Refresh button
        self.refresh_btn = ctk.CTkButton(
            self.model_frame,
            text="Refresh",
            width=80,
            command=self._refresh_available_models
        )
        self.refresh_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # Browse button
        browse_label = ctk.CTkLabel(self.model_frame, text="Or browse:")
        browse_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.browse_btn = ctk.CTkButton(
            self.model_frame,
            text="Browse Model Folder",
            command=self._browse_model_folder
        )
        self.browse_btn.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Model info display
        self.model_info_label = ctk.CTkLabel(
            self.model_frame,
            text="No model loaded",
            wraplength=400,
            font=ctk.CTkFont(size=12)
        )
        self.model_info_label.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
    
    def _create_single_inference(self) -> None:
        """Create single sample inference section."""
        # Single Inference Frame
        self.single_frame = ctk.CTkFrame(self)
        self.single_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.single_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self.single_frame,
            text="Single Sample Testing",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Input text area
        input_label = ctk.CTkLabel(self.single_frame, text="Enter text to classify:")
        input_label.grid(row=1, column=0, padx=10, pady=(5, 0), sticky="w")
        
        self.text_input = ctk.CTkTextbox(
            self.single_frame,
            height=80
        )
        # Insert placeholder text manually
        self.text_input.insert("1.0", "Type or paste your text here...")
        self.text_input.bind("<FocusIn>", self._on_text_focus_in)
        self.text_input.bind("<FocusOut>", self._on_text_focus_out)
        self.text_input.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        # Test button
        self.test_btn = ctk.CTkButton(
            self.single_frame,
            text="Test Sample",
            command=self._test_single_sample,
            state="disabled"
        )
        self.test_btn.grid(row=3, column=0, padx=10, pady=5)
        
        # Results area for single inference
        self.single_result_label = ctk.CTkLabel(
            self.single_frame,
            text="Results will appear here...",
            wraplength=400,
            justify="left"
        )
        self.single_result_label.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
    
    def _create_batch_inference(self) -> None:
        """Create batch inference section."""
        # Batch Inference Frame
        self.batch_frame = ctk.CTkFrame(self)
        self.batch_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.batch_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self.batch_frame,
            text="Batch Processing",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # File selection
        file_label = ctk.CTkLabel(self.batch_frame, text="Select input file:")
        file_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.file_path_var = StringVar(value="No file selected")
        self.file_path_label = ctk.CTkLabel(
            self.batch_frame,
            textvariable=self.file_path_var,
            wraplength=300,
            font=ctk.CTkFont(size=11)
        )
        self.file_path_label.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        
        # File selection buttons
        self.file_buttons_frame = ctk.CTkFrame(self.batch_frame)
        self.file_buttons_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self.file_buttons_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.select_txt_btn = ctk.CTkButton(
            self.file_buttons_frame,
            text="Text File",
            command=lambda: self._select_batch_file("txt"),
            state="disabled"
        )
        self.select_txt_btn.grid(row=0, column=0, padx=2, pady=5, sticky="ew")
        
        self.select_csv_btn = ctk.CTkButton(
            self.file_buttons_frame,
            text="CSV File",
            command=lambda: self._select_batch_file("csv"),
            state="disabled"
        )
        self.select_csv_btn.grid(row=0, column=1, padx=2, pady=5, sticky="ew")
        
        self.select_jsonl_btn = ctk.CTkButton(
            self.file_buttons_frame,
            text="JSONL File",
            command=lambda: self._select_batch_file("jsonl"),
            state="disabled"
        )
        self.select_jsonl_btn.grid(row=0, column=2, padx=2, pady=5, sticky="ew")
        
        # Process button
        self.process_btn = ctk.CTkButton(
            self.batch_frame,
            text="Process Batch",
            command=self._process_batch_file,
            state="disabled"
        )
        self.process_btn.grid(row=4, column=0, padx=10, pady=10)
        
        # Progress bar for batch processing
        self.batch_progress = ctk.CTkProgressBar(self.batch_frame)
        self.batch_progress.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
        self.batch_progress.set(0)
        
        self.batch_status_label = ctk.CTkLabel(
            self.batch_frame,
            text="Ready for batch processing",
            font=ctk.CTkFont(size=11)
        )
        self.batch_status_label.grid(row=6, column=0, padx=10, pady=2)
    
    def _create_results_display(self) -> None:
        """Create results display section."""
        # Results Frame
        self.results_frame = ctk.CTkFrame(self)
        self.results_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self.results_frame,
            text="Results Management",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
        
        # Results summary
        self.results_summary_label = ctk.CTkLabel(
            self.results_frame,
            text="No results yet",
            font=ctk.CTkFont(size=12)
        )
        self.results_summary_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Export buttons
        self.export_frame = ctk.CTkFrame(self.results_frame)
        self.export_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.export_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.export_json_btn = ctk.CTkButton(
            self.export_frame,
            text="Export JSON",
            command=lambda: self._export_results("json"),
            state="disabled"
        )
        self.export_json_btn.grid(row=0, column=0, padx=2, pady=5, sticky="ew")
        
        self.export_csv_btn = ctk.CTkButton(
            self.export_frame,
            text="Export CSV",
            command=lambda: self._export_results("csv"),
            state="disabled"
        )
        self.export_csv_btn.grid(row=0, column=1, padx=2, pady=5, sticky="ew")
        
        self.clear_btn = ctk.CTkButton(
            self.export_frame,
            text="Clear Results",
            command=self._clear_results,
            state="disabled"
        )
        self.clear_btn.grid(row=0, column=2, padx=2, pady=5, sticky="ew")
    
    def _refresh_available_models(self) -> None:
        """Refresh the list of available models."""
        try:
            if self.model_tester:
                available_models = self.model_tester.get_available_models()
                
                if available_models:
                    model_names = []
                    for model in available_models:
                        format_info = f" - {model['format']}" if 'format' in model else ""
                        display_name = f"{model['name']}{format_info} ({model.get('task_name', 'Unknown')})"
                        model_names.append(display_name)
                    
                    self.models_dropdown.configure(values=model_names)
                    self.models_var.set("Select a model...")
                    
                    # Store model info for selection
                    self._available_models_info = available_models
                else:
                    self.models_dropdown.configure(values=["No models found"])
                    self.models_var.set("No models found")
                    self._available_models_info = []
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh models: {e}")
    
    def _on_model_selected(self, model_display_name: str) -> None:
        """Handle model selection from dropdown."""
        if model_display_name in ["Select a model...", "No models found"]:
            return
        
        try:
            # Find the corresponding model info
            model_index = None
            for i, model in enumerate(getattr(self, '_available_models_info', [])):
                format_info = f" - {model['format']}" if 'format' in model else ""
                display_name = f"{model['name']}{format_info} ({model.get('task_name', 'Unknown')})"
                if display_name == model_display_name:
                    model_index = i
                    break
            
            if model_index is not None:
                model_info = self._available_models_info[model_index]
                self._load_model(model_info['path'])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load selected model: {e}")
    
    def _browse_model_folder(self) -> None:
        """Browse for a model folder."""
        folder_path = filedialog.askdirectory(
            title="Select Model Folder",
            initialdir=str(self.config.models_dir)
        )
        
        if folder_path:
            self._load_model(folder_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load a model for testing."""
        try:
            success = self.model_tester.load_model(model_path)
            
            if success:
                self.model_loaded = True
                
                # Update UI state
                self.test_btn.configure(state="normal")
                self.select_txt_btn.configure(state="normal")
                self.select_csv_btn.configure(state="normal")
                self.select_jsonl_btn.configure(state="normal")
                
                # Display model info
                model_info = self.model_tester.get_model_info()
                info_text = (
                    f"Model loaded: {model_info.get('model_name', 'Unknown')}\n"
                    f"Classes: {len(model_info.get('class_names', []))} "
                    f"({', '.join(model_info.get('class_names', [])[:3])}{'...' if len(model_info.get('class_names', [])) > 3 else ''})\n"
                    f"Parameters: {model_info.get('num_parameters', 'Unknown'):,}\n"
                    f"Device: {model_info.get('device', 'Unknown')}"
                )
                self.model_info_label.configure(text=info_text)
                
                # Initialize batch manager
                self.batch_manager = BatchInferenceManager(
                    self.model_tester,
                    self.config.output_dir / "inference_results"
                )
                
                if self.on_model_loaded:
                    self.on_model_loaded(model_path)
                    
                messagebox.showinfo("Success", f"Model loaded successfully!\n\nClasses: {', '.join(model_info.get('class_names', []))}")
                
            else:
                messagebox.showerror("Error", "Failed to load the selected model.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def _test_single_sample(self) -> None:
        """Test a single text sample."""
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        text = self.text_input.get("1.0", "end").strip()
        if not text or self.text_has_placeholder:
            messagebox.showerror("Error", "Please enter some text to classify!")
            return
        
        try:
            self.test_btn.configure(state="disabled", text="Testing...")
            
            # Perform inference
            result = self.model_tester.predict_single(text)
            
            # Display result
            result_text = (
                f"Prediction: {result.predicted_label}\n"
                f"Confidence: {result.confidence:.3f}\n"
                f"Time: {result.inference_time_ms:.1f}ms\n\n"
                f"Top predictions:\n"
            )
            
            for i, (label, conf) in enumerate(result.top_k_predictions[:3], 1):
                result_text += f"  {i}. {label}: {conf:.3f}\n"
            
            self.single_result_label.configure(text=result_text)
            
            # Add to current results
            self.current_results.append(result)
            self._update_results_summary()
            
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {e}")
        finally:
            self.test_btn.configure(state="normal", text="Test Sample")
    
    def _select_batch_file(self, file_type: str) -> None:
        """Select a file for batch processing."""
        filetypes = {
            "txt": [("Text files", "*.txt"), ("All files", "*.*")],
            "csv": [("CSV files", "*.csv"), ("All files", "*.*")],
            "jsonl": [("JSONL files", "*.jsonl"), ("JSON files", "*.json"), ("All files", "*.*")]
        }
        
        filename = filedialog.askopenfilename(
            title=f"Select {file_type.upper()} file for batch processing",
            filetypes=filetypes.get(file_type, [("All files", "*.*")])
        )
        
        if filename:
            self.selected_file_path = filename
            self.selected_file_type = file_type
            self.file_path_var.set(f"Selected: {Path(filename).name}")
            self.process_btn.configure(state="normal")
    
    def _process_batch_file(self) -> None:
        """Process the selected batch file."""
        if not hasattr(self, 'selected_file_path'):
            messagebox.showerror("Error", "Please select a file first!")
            return
        
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        # Start batch processing in a separate thread
        self.inference_in_progress = True
        self._update_batch_ui_state(False)
        
        thread = threading.Thread(target=self._run_batch_processing, daemon=True)
        thread.start()
    
    def _run_batch_processing(self) -> None:
        """Run batch processing in background thread."""
        try:
            # Progress callback for UI updates
            def progress_callback(progress: float, message: str):
                self.after(0, lambda: self._update_batch_progress(progress, message))
            
            # Process file based on type
            if self.selected_file_type == "txt":
                results = self.batch_manager.process_text_file(
                    self.selected_file_path,
                    progress_callback=progress_callback
                )
            elif self.selected_file_type == "csv":
                results = self.batch_manager.process_csv_file(
                    self.selected_file_path,
                    progress_callback=progress_callback
                )
            elif self.selected_file_type == "jsonl":
                results = self.batch_manager.process_jsonl_file(
                    self.selected_file_path,
                    progress_callback=progress_callback
                )
            else:
                raise ValueError(f"Unsupported file type: {self.selected_file_type}")
            
            # Update UI on main thread
            self.after(0, lambda: self._on_batch_complete(results))
            
        except Exception as e:
            self.after(0, lambda: self._on_batch_error(str(e)))
    
    def _update_batch_progress(self, progress: float, message: str) -> None:
        """Update batch processing progress."""
        self.batch_progress.set(progress)
        self.batch_status_label.configure(text=message)
    
    def _on_batch_complete(self, results: List[InferenceResult]) -> None:
        """Handle batch processing completion."""
        self.inference_in_progress = False
        self._update_batch_ui_state(True)
        
        # Add results to current results
        self.current_results.extend(results)
        self._update_results_summary()
        
        # Generate summary report
        summary = self.batch_manager.generate_summary_report(results)
        
        # Show completion message
        avg_time = summary['performance']['average_inference_time_ms']
        avg_confidence = summary['predictions']['average_confidence']
        
        messagebox.showinfo(
            "Batch Processing Complete",
            f"Processed {len(results)} samples successfully!\n\n"
            f"Average inference time: {avg_time:.1f}ms\n"
            f"Average confidence: {avg_confidence:.3f}\n"
            f"Throughput: {summary['performance']['throughput_samples_per_sec']:.1f} samples/sec"
        )
        
        if self.on_inference_complete:
            self.on_inference_complete(results)
        
        # Reset progress
        self.batch_progress.set(0)
        self.batch_status_label.configure(text="Batch processing completed")
    
    def _on_batch_error(self, error_message: str) -> None:
        """Handle batch processing error."""
        self.inference_in_progress = False
        self._update_batch_ui_state(True)
        
        messagebox.showerror("Batch Processing Error", f"Failed to process batch file:\n{error_message}")
        
        self.batch_progress.set(0)
        self.batch_status_label.configure(text="Batch processing failed")
    
    def _update_batch_ui_state(self, enabled: bool) -> None:
        """Update UI state during batch processing."""
        state = "normal" if enabled else "disabled"
        
        self.select_txt_btn.configure(state=state)
        self.select_csv_btn.configure(state=state)
        self.select_jsonl_btn.configure(state=state)
        self.process_btn.configure(state=state if hasattr(self, 'selected_file_path') else "disabled")
        self.test_btn.configure(state=state if self.model_loaded else "disabled")
    
    def _update_results_summary(self) -> None:
        """Update the results summary display."""
        if not self.current_results:
            self.results_summary_label.configure(text="No results yet")
            self.export_json_btn.configure(state="disabled")
            self.export_csv_btn.configure(state="disabled")
            self.clear_btn.configure(state="disabled")
            return
        
        num_results = len(self.current_results)
        avg_confidence = sum(r.confidence for r in self.current_results) / num_results
        avg_time = sum(r.inference_time_ms for r in self.current_results) / num_results
        
        summary_text = (
            f"Results: {num_results} samples\n"
            f"Avg. Confidence: {avg_confidence:.3f}\n"
            f"Avg. Time: {avg_time:.1f}ms"
        )
        
        self.results_summary_label.configure(text=summary_text)
        
        # Enable export buttons
        self.export_json_btn.configure(state="normal")
        self.export_csv_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
    
    def _export_results(self, format: str) -> None:
        """Export current results to file."""
        if not self.current_results:
            messagebox.showerror("Error", "No results to export!")
            return
        
        filetypes = {
            "json": [("JSON files", "*.json"), ("All files", "*.*")],
            "csv": [("CSV files", "*.csv"), ("All files", "*.*")]
        }
        
        filename = filedialog.asksaveasfilename(
            title=f"Export Results as {format.upper()}",
            filetypes=filetypes.get(format, [("All files", "*.*")]),
            defaultextension=f".{format}"
        )
        
        if filename:
            try:
                self.batch_manager.save_results(self.current_results, filename, format)
                messagebox.showinfo("Success", f"Results exported to {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")
    
    def _clear_results(self) -> None:
        """Clear all current results."""
        if messagebox.askyesno("Confirm", "Clear all current results?"):
            self.current_results = []
            self._update_results_summary()
            self.single_result_label.configure(text="Results will appear here...")
            messagebox.showinfo("Success", "Results cleared")
    
    def _on_text_focus_in(self, event) -> None:
        """Handle text input focus in event."""
        if self.text_has_placeholder:
            self.text_input.delete("1.0", "end")
            self.text_has_placeholder = False
    
    def _on_text_focus_out(self, event) -> None:
        """Handle text input focus out event."""
        if not self.text_input.get("1.0", "end").strip():
            self.text_input.insert("1.0", self.placeholder_text)
            self.text_has_placeholder = True
    
    def get_testing_state(self) -> Dict[str, Any]:
        """Get current testing state information."""
        return {
            'model_loaded': self.model_loaded,
            'inference_in_progress': self.inference_in_progress,
            'num_results': len(self.current_results),
            'model_info': self.model_tester.get_model_info() if self.model_tester and self.model_loaded else {}
        }