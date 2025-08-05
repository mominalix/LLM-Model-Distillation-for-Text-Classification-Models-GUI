"""
Main entry point for the LLM Distillation package.

This module provides the main entry point when the package is installed
and called via the llm-distillation command.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

# Load .env file first before any other imports (OpenAI recommended pattern)
from dotenv import load_dotenv
load_dotenv()

# Download required NLTK data to prevent training errors
try:
    import nltk
    required_nltk_data = ['punkt_tab', 'punkt', 'stopwords', 'wordnet']
    for data_name in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}')
        except (LookupError, OSError):
            try:
                print(f"Downloading NLTK data: {data_name}")
                nltk.download(data_name, quiet=True)
            except Exception as e:
                print(f"Failed to download NLTK {data_name}: {e}")
except ImportError:
    pass

from .config import get_config, Config
from .gui import MainWindow


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup application logging."""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    else:
        # Default log file
        default_log_file = logs_dir / "llm_distillation.log"
        file_handler = logging.FileHandler(default_log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format=log_format
    )
    
    # Suppress verbose logs from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")


def validate_environment() -> bool:
    """Validate the environment and dependencies."""
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        # Check critical dependencies
        critical_deps = [
            'customtkinter',
            'matplotlib', 
            'transformers',
            'torch',
            'openai',
            'datasets',
            'sklearn'
        ]
        
        missing_deps = []
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"Missing required dependencies: {', '.join(missing_deps)}")
            logger.error("Please install dependencies with: pip install -r requirements.txt")
            return False
        
        # Check for GPU availability (optional)
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA available with {torch.cuda.device_count()} device(s)")
            else:
                logger.info("CUDA not available, using CPU")
        except Exception as e:
            logger.warning(f"Could not check CUDA availability: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def check_configuration(config: Config) -> bool:
    """Check configuration validity."""
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check directories
        try:
            config.create_directories()
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration check failed: {e}")
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="LLM Model Distillation for Text Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with default settings
  %(prog)s --log-level DEBUG        # Run with debug logging
  %(prog)s --config my_config.env   # Run with custom config file
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (.env format)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: logs/llm_distillation.log)"
    )
    
    parser.add_argument(
        "--theme",
        choices=["light", "dark", "system"],
        help="GUI theme (overrides config)"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip environment validation (not recommended)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser.parse_args()


def run_gui_mode(config: Config, theme: Optional[str] = None) -> int:
    """Run the application in GUI mode."""
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting GUI application")
        
        # Override theme if specified
        if theme:
            config.theme = theme
        
        # Create and run the main window
        app = MainWindow(config)
        app.run()
        
        logger.info("Application closed normally")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point."""
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("LLM Model Distillation for Text Classification")
    logger.info("Version: 1.0.0")
    logger.info("="*60)
    
    try:
        # Load configuration
        config_overrides = {}
        if args.theme:
            config_overrides['theme'] = args.theme
        
        if args.config:
            # Load from custom config file
            config_path = Path(args.config)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {args.config}")
                return 1
            
            os.environ['CONFIG_FILE'] = str(config_path)
        
        config = get_config(**config_overrides)
        logger.info("Configuration loaded successfully")
        
        # Validate environment
        if not args.no_validation:
            logger.info("Validating environment...")
            if not validate_environment():
                logger.error("Environment validation failed")
                return 1
            
            logger.info("Checking configuration...")
            if not check_configuration(config):
                logger.error("Configuration check failed")
                return 1
        
        # Run application
        return run_gui_mode(config, args.theme)
            
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())