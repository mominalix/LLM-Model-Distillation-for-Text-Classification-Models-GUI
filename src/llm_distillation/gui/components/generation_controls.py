"""
Generation controls panel for managing data generation process.

This component provides controls for starting/stopping data generation
and monitoring generation parameters.
"""

from typing import Callable, Optional
import customtkinter as ctk

from ...config import Config


class GenerationControlsPanel(ctk.CTkFrame):
    """Panel for data generation controls."""
    
    def __init__(
        self,
        parent,
        config: Config,
        on_start_generation: Optional[Callable[[], None]] = None,
        on_stop_generation: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        self.config = config
        self.on_start_generation = on_start_generation
        self.on_stop_generation = on_stop_generation
        
        self.generation_active = False
        
        self.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Controls frame - compact layout
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Primary buttons row
        self.primary_row = ctk.CTkFrame(self.controls_frame)
        self.primary_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self.primary_row.grid_columnconfigure((0, 1), weight=1)
        
        # Start generation button - compact
        self.start_button = ctk.CTkButton(
            self.primary_row,
            text="Generate",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=32,
            command=self._on_start_clicked
        )
        self.start_button.grid(row=0, column=0, padx=2, pady=3, sticky="ew")
        
        # Stop generation button - compact
        self.stop_button = ctk.CTkButton(
            self.primary_row,
            text="Stop",
            font=ctk.CTkFont(size=12),
            height=32,
            state="disabled",
            command=self._on_stop_clicked,
            fg_color="red",
            hover_color="darkred"
        )
        self.stop_button.grid(row=0, column=1, padx=2, pady=3, sticky="ew")
        
        # Secondary buttons row
        self.secondary_row = ctk.CTkFrame(self.controls_frame)
        self.secondary_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self.secondary_row.grid_columnconfigure((0, 1), weight=1)
        
        # Validate button - compact
        self.validate_button = ctk.CTkButton(
            self.secondary_row,
            text="Validate",
            font=ctk.CTkFont(size=11),
            height=28,
            command=self._validate_setup,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.validate_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        # Cost estimation button - compact
        self.estimate_button = ctk.CTkButton(
            self.secondary_row,
            text="Cost",
            font=ctk.CTkFont(size=11),
            height=28,
            command=self._estimate_cost,
            fg_color="orange",
            hover_color="darkorange"
        )
        self.estimate_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        # Advanced settings frame - compact
        self.advanced_frame = ctk.CTkFrame(self)
        self.advanced_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=3)
        self.advanced_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Temperature setting
        self.temp_label = ctk.CTkLabel(self.advanced_frame, text="Temperature:")
        self.temp_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.temp_slider = ctk.CTkSlider(
            self.advanced_frame,
            from_=0.1,
            to=2.0,
            number_of_steps=19,
            command=self._on_temperature_changed
        )
        self.temp_slider.set(0.7)
        self.temp_slider.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.temp_value_label = ctk.CTkLabel(self.advanced_frame, text="0.7")
        self.temp_value_label.grid(row=2, column=0, padx=5, pady=5)
        
        # Quality threshold
        self.quality_label = ctk.CTkLabel(self.advanced_frame, text="Quality Threshold:")
        self.quality_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.quality_slider = ctk.CTkSlider(
            self.advanced_frame,
            from_=0.3,
            to=1.0,
            number_of_steps=7,
            command=self._on_quality_changed
        )
        self.quality_slider.set(self.config.quality_threshold)
        self.quality_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.quality_value_label = ctk.CTkLabel(
            self.advanced_frame, 
            text=f"{self.config.quality_threshold:.1f}"
        )
        self.quality_value_label.grid(row=2, column=1, padx=5, pady=5)
        
        # Diversity settings
        self.diversity_label = ctk.CTkLabel(self.advanced_frame, text="Enable Diversity:")
        self.diversity_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        self.diversity_var = ctk.BooleanVar(value=True)
        self.diversity_checkbox = ctk.CTkCheckBox(
            self.advanced_frame,
            text="Augmentation & Bias Mitigation",
            variable=self.diversity_var
        )
        self.diversity_checkbox.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Status display
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready to generate data",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.cost_label = ctk.CTkLabel(
            self.status_frame,
            text="Estimated cost: $0.00",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.cost_label.grid(row=1, column=0, padx=10, pady=5)
    
    def _on_start_clicked(self) -> None:
        """Handle start generation button click."""
        if self.on_start_generation:
            self.on_start_generation()
    
    def _on_stop_clicked(self) -> None:
        """Handle stop generation button click."""
        if self.on_stop_generation:
            self.on_stop_generation()
    
    def _on_temperature_changed(self, value: float) -> None:
        """Handle temperature slider change."""
        self.temp_value_label.configure(text=f"{value:.1f}")
    
    def _on_quality_changed(self, value: float) -> None:
        """Handle quality threshold slider change."""
        self.quality_value_label.configure(text=f"{value:.1f}")
    
    def _validate_setup(self) -> None:
        """Validate the current setup."""
        # This would be connected to validation logic
        # For now, just show a status message
        self.status_label.configure(text="Setup validation passed")
        
        # Reset status after 3 seconds
        self.after(3000, lambda: self.status_label.configure(text="Ready to generate data"))
    
    def _estimate_cost(self) -> None:
        """Estimate the cost of data generation."""
        # This would calculate actual cost based on current settings
        # For now, show a placeholder
        self.cost_label.configure(text="Estimated cost: $0.15 (placeholder)")
        
        # Reset after 5 seconds
        self.after(5000, lambda: self.cost_label.configure(text="Estimated cost: $0.00"))
    
    def set_generation_state(self, is_generating: bool) -> None:
        """Update UI based on generation state."""
        self.generation_active = is_generating
        
        if is_generating:
            self.start_button.configure(state="disabled", text="Generating...")
            self.stop_button.configure(state="normal")
            self.validate_button.configure(state="disabled")
            self.estimate_button.configure(state="disabled")
            self.status_label.configure(text="Generation in progress...")
        else:
            self.start_button.configure(state="normal", text="Generate Data")
            self.stop_button.configure(state="disabled")
            self.validate_button.configure(state="normal")
            self.estimate_button.configure(state="normal")
            self.status_label.configure(text="Ready to generate data")
    
    def get_generation_config(self) -> dict:
        """Get current generation configuration."""
        return {
            'temperature': self.temp_slider.get(),
            'quality_threshold': self.quality_slider.get(),
            'enable_diversity': self.diversity_var.get(),
        }
    
    def update_status(self, message: str) -> None:
        """Update the status message."""
        self.status_label.configure(text=message)
    
    def update_cost_estimate(self, cost: float) -> None:
        """Update the cost estimate display."""
        self.cost_label.configure(text=f"Estimated cost: ${cost:.4f}")
    
    def reset(self) -> None:
        """Reset the panel to initial state."""
        self.set_generation_state(False)
        self.temp_slider.set(0.7)
        self.quality_slider.set(self.config.quality_threshold)
        self.diversity_var.set(True)
        self._on_temperature_changed(0.7)
        self._on_quality_changed(self.config.quality_threshold)