"""
GUI utilities and helper functions.

This module provides utility functions for GUI operations including
validation, formatting, and common UI patterns.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import customtkinter as ctk
from tkinter import messagebox


def validate_positive_integer(value: str, min_val: int = 1, max_val: Optional[int] = None) -> Tuple[bool, Optional[int]]:
    """
    Validate that a string represents a positive integer within bounds.
    
    Returns:
        Tuple of (is_valid, parsed_value)
    """
    try:
        num = int(value)
        if num < min_val:
            return False, None
        if max_val is not None and num > max_val:
            return False, None
        return True, num
    except ValueError:
        return False, None


def validate_positive_float(value: str, min_val: float = 0.0, max_val: Optional[float] = None) -> Tuple[bool, Optional[float]]:
    """
    Validate that a string represents a positive float within bounds.
    
    Returns:
        Tuple of (is_valid, parsed_value)
    """
    try:
        num = float(value)
        if num < min_val:
            return False, None
        if max_val is not None and num > max_val:
            return False, None
        return True, num
    except ValueError:
        return False, None


def validate_learning_rate(value: str) -> Tuple[bool, Optional[float]]:
    """
    Validate learning rate format (supports scientific notation).
    
    Returns:
        Tuple of (is_valid, parsed_value)
    """
    # Common learning rate patterns
    lr_pattern = r'^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    
    if not re.match(lr_pattern, value):
        return False, None
    
    try:
        lr = float(value)
        if lr <= 0 or lr > 1.0:
            return False, None
        return True, lr
    except ValueError:
        return False, None


def validate_class_name(name: str) -> bool:
    """
    Validate class name format.
    
    Args:
        name: Class name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name or not name.strip():
        return False
    
    # Check for reasonable length
    if len(name.strip()) > 50:
        return False
    
    # Check for special characters that might cause issues
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\t']
    for char in invalid_chars:
        if char in name:
            return False
    
    return True


def format_file_size(size_bytes: float) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes:.0f} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def format_number(number: float, precision: int = 3) -> str:
    """
    Format number with appropriate precision and scientific notation if needed.
    
    Args:
        number: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if abs(number) >= 1000:
        return f"{number:,.{precision}f}"
    elif abs(number) < 0.001 and number != 0:
        return f"{number:.{precision}e}"
    else:
        return f"{number:.{precision}f}"


def create_tooltip(widget: ctk.CTkBaseClass, text: str) -> None:
    """
    Create a tooltip for a widget.
    
    Args:
        widget: Widget to attach tooltip to
        text: Tooltip text
    """
    # Note: CustomTkinter doesn't have built-in tooltip support
    # This is a placeholder for future implementation
    pass


def show_progress_dialog(parent, title: str, message: str) -> ctk.CTkToplevel:
    """
    Show a progress dialog.
    
    Args:
        parent: Parent window
        title: Dialog title
        message: Progress message
        
    Returns:
        Dialog window
    """
    dialog = ctk.CTkToplevel(parent)
    dialog.title(title)
    dialog.geometry("300x150")
    dialog.transient(parent)
    dialog.grab_set()
    
    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (300 // 2)
    y = (dialog.winfo_screenheight() // 2) - (150 // 2)
    dialog.geometry(f"+{x}+{y}")
    
    # Message label
    message_label = ctk.CTkLabel(dialog, text=message, wraplength=250)
    message_label.pack(pady=20)
    
    # Progress bar
    progress_bar = ctk.CTkProgressBar(dialog, mode="indeterminate")
    progress_bar.pack(pady=10, padx=20, fill="x")
    progress_bar.start()
    
    # Cancel button
    cancel_button = ctk.CTkButton(
        dialog,
        text="Cancel",
        command=dialog.destroy
    )
    cancel_button.pack(pady=10)
    
    return dialog


def show_confirmation_dialog(parent, title: str, message: str) -> bool:
    """
    Show a confirmation dialog.
    
    Args:
        parent: Parent window
        title: Dialog title
        message: Confirmation message
        
    Returns:
        True if confirmed, False otherwise
    """
    return messagebox.askyesno(title, message, parent=parent)


def show_error_dialog(parent, title: str, message: str, details: Optional[str] = None) -> None:
    """
    Show an error dialog with optional details.
    
    Args:
        parent: Parent window
        title: Dialog title
        message: Error message
        details: Optional detailed error information
    """
    if details:
        full_message = f"{message}\n\nDetails:\n{details}"
    else:
        full_message = message
    
    messagebox.showerror(title, full_message, parent=parent)


def show_info_dialog(parent, title: str, message: str) -> None:
    """
    Show an information dialog.
    
    Args:
        parent: Parent window
        title: Dialog title
        message: Information message
    """
    messagebox.showinfo(title, message, parent=parent)


def validate_file_path(path: str, must_exist: bool = True) -> bool:
    """
    Validate file path.
    
    Args:
        path: File path to validate
        must_exist: Whether file must exist
        
    Returns:
        True if valid, False otherwise
    """
    if not path or not path.strip():
        return False
    
    try:
        path_obj = Path(path)
        
        if must_exist and not path_obj.exists():
            return False
        
        # Check if parent directory exists (for new files)
        if not must_exist and not path_obj.parent.exists():
            return False
        
        return True
    except (OSError, ValueError):
        return False


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated_length = max_length - len(suffix)
    return text[:truncated_length] + suffix


def create_labeled_entry(
    parent,
    label_text: str,
    default_value: str = "",
    placeholder: str = "",
    width: Optional[int] = None
) -> Tuple[ctk.CTkLabel, ctk.CTkEntry]:
    """
    Create a labeled entry widget.
    
    Args:
        parent: Parent widget
        label_text: Label text
        default_value: Default entry value
        placeholder: Placeholder text
        width: Entry width
        
    Returns:
        Tuple of (label, entry)
    """
    label = ctk.CTkLabel(parent, text=label_text)
    
    entry_kwargs = {}
    if width:
        entry_kwargs["width"] = width
    
    entry = ctk.CTkEntry(parent, **entry_kwargs)
    
    if default_value:
        entry.insert(0, default_value)
    
    return label, entry


def create_metric_display(
    parent,
    metric_name: str,
    initial_value: str = "-",
    row: int = 0,
    column: int = 0
) -> Tuple[ctk.CTkLabel, ctk.CTkLabel]:
    """
    Create a metric display with name and value labels.
    
    Args:
        parent: Parent widget
        metric_name: Name of the metric
        initial_value: Initial value to display
        row: Grid row
        column: Grid column
        
    Returns:
        Tuple of (name_label, value_label)
    """
    name_label = ctk.CTkLabel(
        parent,
        text=f"{metric_name}:",
        font=ctk.CTkFont(weight="bold")
    )
    name_label.grid(row=row, column=column, padx=5, pady=2, sticky="w")
    
    value_label = ctk.CTkLabel(parent, text=initial_value)
    value_label.grid(row=row+1, column=column, padx=5, pady=2, sticky="w")
    
    return name_label, value_label


def setup_window_icon(window: ctk.CTk, icon_path: Optional[str] = None) -> None:
    """
    Setup window icon if available.
    
    Args:
        window: Window to set icon for
        icon_path: Path to icon file
    """
    if icon_path and Path(icon_path).exists():
        try:
            window.iconbitmap(icon_path)
        except Exception:
            # Ignore icon errors
            pass


def center_window(window: ctk.CTk, width: int, height: int) -> None:
    """
    Center window on screen.
    
    Args:
        window: Window to center
        width: Window width
        height: Window height
    """
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    window.geometry(f"{width}x{height}+{x}+{y}")


class StatusManager:
    """Manager for status updates across the application."""
    
    def __init__(self):
        self.status_callbacks: List[callable] = []
        self.current_status = "Ready"
    
    def add_status_callback(self, callback: callable) -> None:
        """Add a status update callback."""
        self.status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: callable) -> None:
        """Remove a status update callback."""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
    
    def update_status(self, status: str) -> None:
        """Update status across all registered callbacks."""
        self.current_status = status
        for callback in self.status_callbacks:
            try:
                callback(status)
            except Exception as e:
                print(f"Error in status callback: {e}")
    
    def get_current_status(self) -> str:
        """Get current status."""
        return self.current_status


# Global status manager instance
status_manager = StatusManager()