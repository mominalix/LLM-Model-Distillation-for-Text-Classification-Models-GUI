"""
Task input panel for entering dataset descriptions and requirements.

This component provides an interface for users to describe their
text classification task in natural language.
"""

from typing import Optional
import customtkinter as ctk

from ...config import Config


class TaskInputPanel(ctk.CTkFrame):
    """Panel for task description input."""
    
    def __init__(self, parent, config: Config, **kwargs):
        super().__init__(parent, **kwargs)
        self.config = config
        
        self.grid_columnconfigure(0, weight=1)
        
        # Task description label
        self.task_label = ctk.CTkLabel(
            self,
            text="Task Description:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.task_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        
        # Task description text area
        self.task_textbox = ctk.CTkTextbox(
            self,
            height=100
        )
        # Add placeholder text manually
        placeholder_text = ("Describe the dataset you need...\n\n"
                           "Example: Classify customer feedback for a restaurant into positive, "
                           "negative, or neutral sentiment based on the overall tone and satisfaction "
                           "expressed in the review.")
        self.task_textbox.insert("1.0", placeholder_text)
        self.task_textbox.configure(text_color="gray")
        self.task_textbox.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        # Tips label
        self.tips_label = ctk.CTkLabel(
            self,
            text="Tip: Be specific about your classification goal and provide context about the domain.",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.tips_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))
    
    def get_task_description(self) -> str:
        """Get the current task description."""
        return self.task_textbox.get("1.0", "end-1c").strip()
    
    def set_task_description(self, description: str) -> None:
        """Set the task description."""
        self.task_textbox.delete("1.0", "end")
        self.task_textbox.insert("1.0", description)
    
    def clear(self) -> None:
        """Clear the task description."""
        self.task_textbox.delete("1.0", "end")