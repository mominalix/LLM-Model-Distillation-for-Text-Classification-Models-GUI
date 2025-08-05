"""
Class management panel for adding and removing classification labels.

This component provides dynamic management of class labels with
add/remove functionality and validation.
"""

from typing import List, Callable, Optional
import customtkinter as ctk
from tkinter import messagebox

from ...config import Config


class ClassManagementPanel(ctk.CTkFrame):
    """Panel for managing classification labels."""
    
    def __init__(
        self, 
        parent, 
        config: Config, 
        on_classes_changed: Optional[Callable[[List[str]], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        self.config = config
        self.on_classes_changed = on_classes_changed
        
        self.class_labels: List[str] = []
        self.class_entries: List[ctk.CTkEntry] = []
        
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.classes_label = ctk.CTkLabel(
            self.header_frame,
            text="Class Labels:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.classes_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.add_button = ctk.CTkButton(
            self.header_frame,
            text="+ Add Class",
            width=100,
            command=self._add_class_entry
        )
        self.add_button.grid(row=0, column=1, sticky="e", padx=10, pady=5)
        
        # Scrollable frame for class entries
        self.classes_frame = ctk.CTkScrollableFrame(self, height=120)
        self.classes_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.classes_frame.grid_columnconfigure(0, weight=1)
        
        # Instructions
        self.instructions_label = ctk.CTkLabel(
            self,
            text="💡 Add at least 2 class labels. Use clear, descriptive names (e.g., 'Positive', 'Negative', 'Neutral').",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.instructions_label.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))
        
        # Add initial class entries
        self._add_class_entry("Positive")
        self._add_class_entry("Negative")
        self._add_class_entry("Neutral")
    
    def _add_class_entry(self, initial_value: str = "") -> None:
        """Add a new class entry field."""
        
        entry_frame = ctk.CTkFrame(self.classes_frame)
        entry_frame.grid(
            row=len(self.class_entries), 
            column=0, 
            sticky="ew", 
            padx=5, 
            pady=2
        )
        entry_frame.grid_columnconfigure(1, weight=1)
        
        # Class number label
        class_num_label = ctk.CTkLabel(
            entry_frame,
            text=f"Class {len(self.class_entries) + 1}:",
            width=80
        )
        class_num_label.grid(row=0, column=0, padx=(10, 5), pady=5)
        
        # Class name entry
        class_entry = ctk.CTkEntry(
            entry_frame,

        )
        class_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        if initial_value:
            class_entry.insert(0, initial_value)
        
        # Remove button
        remove_button = ctk.CTkButton(
            entry_frame,
            text="×",
            width=30,
            command=lambda: self._remove_class_entry(entry_frame, class_entry)
        )
        remove_button.grid(row=0, column=2, padx=(5, 10), pady=5)
        
        # Bind events
        class_entry.bind("<KeyRelease>", self._on_class_changed)
        class_entry.bind("<FocusOut>", self._on_class_changed)
        
        self.class_entries.append(class_entry)
        
        # Update class list
        self._update_class_list()
    
    def _remove_class_entry(self, entry_frame: ctk.CTkFrame, class_entry: ctk.CTkEntry) -> None:
        """Remove a class entry."""
        
        if len(self.class_entries) <= 2:
            messagebox.showwarning(
                "Warning",
                "You need at least 2 class labels for classification!"
            )
            return
        
        # Remove from lists
        if class_entry in self.class_entries:
            self.class_entries.remove(class_entry)
        
        # Destroy the frame
        entry_frame.destroy()
        
        # Update numbering for remaining entries
        self._update_entry_numbering()
        
        # Update class list
        self._update_class_list()
    
    def _update_entry_numbering(self) -> None:
        """Update the numbering labels for class entries."""
        
        for i, entry in enumerate(self.class_entries):
            # Find the parent frame and update the label
            parent_frame = entry.master
            if hasattr(parent_frame, 'winfo_children'):
                children = parent_frame.winfo_children()
                if children and isinstance(children[0], ctk.CTkLabel):
                    children[0].configure(text=f"Class {i + 1}:")
    
    def _on_class_changed(self, event=None) -> None:
        """Handle class name changes."""
        self._update_class_list()
    
    def _update_class_list(self) -> None:
        """Update the internal class list and notify callback."""
        
        # Get current class names
        new_class_labels = []
        for entry in self.class_entries:
            class_name = entry.get().strip()
            if class_name:
                new_class_labels.append(class_name)
        
        # Check for duplicates
        seen = set()
        unique_labels = []
        for label in new_class_labels:
            if label.lower() not in seen:
                unique_labels.append(label)
                seen.add(label.lower())
        
        self.class_labels = unique_labels
        
        # Notify callback
        if self.on_classes_changed:
            self.on_classes_changed(self.class_labels)
    
    def get_class_labels(self) -> List[str]:
        """Get the current list of class labels."""
        self._update_class_list()
        return self.class_labels.copy()
    
    def set_class_labels(self, labels: List[str]) -> None:
        """Set the class labels."""
        
        # Clear existing entries
        for entry in self.class_entries:
            entry.master.destroy()
        self.class_entries.clear()
        
        # Add new entries
        for label in labels:
            self._add_class_entry(label)
        
        # Add empty entry if none provided
        if not labels:
            self._add_class_entry()
    
    def validate_classes(self) -> bool:
        """Validate that we have valid class labels."""
        
        valid_labels = [label for label in self.class_labels if label.strip()]
        
        if len(valid_labels) < 2:
            messagebox.showerror(
                "Validation Error",
                "Please provide at least 2 valid class labels!"
            )
            return False
        
        if len(valid_labels) != len(set(label.lower() for label in valid_labels)):
            messagebox.showerror(
                "Validation Error",
                "Class labels must be unique (case-insensitive)!"
            )
            return False
        
        return True
    
    def clear(self) -> None:
        """Clear all class entries."""
        
        # Remove all entries except the first two
        while len(self.class_entries) > 2:
            entry = self.class_entries[-1]
            self._remove_class_entry(entry.master, entry)
        
        # Clear remaining entries
        for entry in self.class_entries:
            entry.delete(0, "end")