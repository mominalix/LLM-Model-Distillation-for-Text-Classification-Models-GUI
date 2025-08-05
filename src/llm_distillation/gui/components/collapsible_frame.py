"""
Collapsible frame component for better UI organization.

This component provides expandable/collapsible sections with headers
for better navigation and space utilization.
"""

import customtkinter as ctk
from typing import Optional, Callable


class CollapsibleFrame(ctk.CTkFrame):
    """A collapsible frame widget with a header and toggle functionality."""
    
    def __init__(
        self, 
        parent, 
        title: str = "Section",
        icon: str = "",
        collapsed: bool = False,
        on_toggle: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        
        self.title = title
        self.icon = icon
        self.collapsed = collapsed
        self.on_toggle = on_toggle
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Create header frame
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.header_frame.grid_columnconfigure(1, weight=1)
        
        # Toggle button
        self.toggle_button = ctk.CTkButton(
            self.header_frame,
            text="▼" if not collapsed else "▶",
            width=30,
            height=30,
            command=self.toggle,
            font=ctk.CTkFont(size=14)
        )
        self.toggle_button.grid(row=0, column=0, padx=(5, 10), pady=5)
        
        # Title label
        display_text = f"{icon} {title}" if icon else title
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text=display_text,
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        self.title_label.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Content frame
        self.content_frame = ctk.CTkFrame(self)
        if not collapsed:
            self.content_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        
        # Configure content frame
        self.content_frame.grid_columnconfigure(0, weight=1)
    
    def toggle(self):
        """Toggle the collapsed state."""
        self.collapsed = not self.collapsed
        
        if self.collapsed:
            self.content_frame.grid_remove()
            self.toggle_button.configure(text="▶")
        else:
            self.content_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
            self.toggle_button.configure(text="▼")
        
        # Call callback if provided
        if self.on_toggle:
            self.on_toggle(self.collapsed)
    
    def expand(self):
        """Expand the section."""
        if self.collapsed:
            self.toggle()
    
    def collapse(self):
        """Collapse the section."""
        if not self.collapsed:
            self.toggle()
    
    def get_content_frame(self) -> ctk.CTkFrame:
        """Get the content frame for adding widgets."""
        return self.content_frame
    
    def set_title(self, title: str, icon: str = None):
        """Update the title and optionally the icon."""
        self.title = title
        if icon is not None:
            self.icon = icon
        display_text = f"{self.icon} {self.title}" if self.icon else self.title
        self.title_label.configure(text=display_text)