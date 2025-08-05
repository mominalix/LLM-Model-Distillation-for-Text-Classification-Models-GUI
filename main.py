#!/usr/bin/env python3
"""
LLM Model Distillation for Text Classification - Main Entry Point

This is the main entry point for the LLM Model Distillation application.
Run this file to start the GUI application.

Usage:
    python main.py [options]
"""

import sys
from pathlib import Path

# Load .env file first before any other imports (OpenAI recommended pattern)
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path to enable imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    # Import and run the main function from the package
    from llm_distillation.main import main
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"Error: Failed to import llm_distillation module: {e}")
    print("Please ensure you're running from the project root directory.")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    
    if src_path.exists():
        print("Source directory exists. Checking package structure...")
        package_init = src_path / "llm_distillation" / "__init__.py"
        if package_init.exists():
            print("Package structure appears correct.")
            print("Try installing the package in development mode:")
            print("  pip install -e .")
        else:
            print("Package __init__.py not found.")
    else:
        print("Source directory not found.")
    
    sys.exit(1)