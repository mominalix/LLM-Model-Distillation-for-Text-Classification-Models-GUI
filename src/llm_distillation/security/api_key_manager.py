"""
API Key management and secure credential handling.

This module provides secure storage, rotation, and validation of API keys
and other sensitive credentials with encryption and access controls.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import base64
import secrets

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..config import Config
from ..exceptions import SecurityError, ErrorCodes

logger = logging.getLogger(__name__)


@dataclass
class APIKeyInfo:
    """Information about an API key."""
    name: str
    provider: str
    masked_key: str
    created_at: float
    last_used: Optional[float] = None
    usage_count: int = 0
    is_active: bool = True
    expires_at: Optional[float] = None
    permissions: List[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []


class APIKeyManager:
    """Secure API key management system."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Setup secure storage directory
        self.credentials_dir = config.cache_dir / "credentials"
        self.credentials_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Credentials file
        self.credentials_file = self.credentials_dir / "credentials.enc"
        self.metadata_file = self.credentials_dir / "metadata.json"
        
        # Initialize encryption
        self.cipher = self._initialize_encryption()
        
        # In-memory credential store (encrypted)
        self._credentials: Dict[str, str] = {}
        self._metadata: Dict[str, APIKeyInfo] = {}
        
        # Load existing credentials
        self._load_credentials()
        
        logger.info("API Key Manager initialized")
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption system for secure credential storage."""
        
        # Get or generate master key
        master_key = self._get_master_key()
        
        # Generate Fernet key from master key
        salt = b'llm_distillation_salt'  # In production, use random salt per installation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key))
        
        return Fernet(key)
    
    def _get_master_key(self) -> bytes:
        """Get or generate master encryption key."""
        
        key_file = self.credentials_dir / ".master_key"
        
        if key_file.exists():
            # Load existing key
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to load master key: {e}")
        
        # Generate new master key
        master_key = secrets.token_bytes(32)
        
        try:
            # Save securely
            with open(key_file, 'wb') as f:
                f.write(master_key)
            
            # Set secure permissions
            os.chmod(key_file, 0o600)
            
            logger.info("Generated new master encryption key")
            
        except Exception as e:
            logger.error(f"Failed to save master key: {e}")
            raise SecurityError(
                message="Failed to initialize secure credential storage",
                error_code=ErrorCodes.SECURITY_UNAUTHORIZED_ACCESS,
                original_error=e
            )
        
        return master_key
    
    def _load_credentials(self) -> None:
        """Load encrypted credentials from storage."""
        
        try:
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata_data = json.load(f)
                
                for name, info_dict in metadata_data.items():
                    self._metadata[name] = APIKeyInfo(**info_dict)
            
            # Load encrypted credentials
            if self.credentials_file.exists():
                with open(self.credentials_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self.cipher.decrypt(encrypted_data)
                credentials_data = json.loads(decrypted_data.decode())
                
                self._credentials = credentials_data
                
                logger.info(f"Loaded {len(self._credentials)} stored credentials")
        
        except Exception as e:
            logger.warning(f"Failed to load credentials: {e}")
            # Start with empty credentials if loading fails
            self._credentials = {}
            self._metadata = {}
    
    def _save_credentials(self) -> None:
        """Save encrypted credentials to storage."""
        
        try:
            # Save metadata
            metadata_data = {}
            for name, info in self._metadata.items():
                metadata_data[name] = {
                    'name': info.name,
                    'provider': info.provider,
                    'masked_key': info.masked_key,
                    'created_at': info.created_at,
                    'last_used': info.last_used,
                    'usage_count': info.usage_count,
                    'is_active': info.is_active,
                    'expires_at': info.expires_at,
                    'permissions': info.permissions
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_data, f, indent=2)
            
            # Save encrypted credentials
            credentials_json = json.dumps(self._credentials)
            encrypted_data = self.cipher.encrypt(credentials_json.encode())
            
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set secure permissions
            os.chmod(self.credentials_file, 0o600)
            os.chmod(self.metadata_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise SecurityError(
                message="Failed to save credentials securely",
                error_code=ErrorCodes.SECURITY_UNAUTHORIZED_ACCESS,
                original_error=e
            )
    
    def store_api_key(
        self,
        name: str,
        api_key: str,
        provider: str,
        permissions: Optional[List[str]] = None,
        expires_at: Optional[float] = None
    ) -> bool:
        """
        Securely store an API key.
        
        Args:
            name: Unique name for the API key
            api_key: The actual API key
            provider: API provider (e.g., 'openai', 'anthropic')
            permissions: List of permissions for this key
            expires_at: Expiration timestamp (Unix time)
            
        Returns:
            True if successfully stored
        """
        
        try:
            # Validate API key format
            if not self._validate_api_key_format(api_key, provider):
                raise SecurityError(
                    message=f"Invalid API key format for provider: {provider}",
                    error_code=ErrorCodes.SECURITY_API_KEY_EXPOSED
                )
            
            # Store encrypted key
            self._credentials[name] = api_key
            
            # Store metadata
            self._metadata[name] = APIKeyInfo(
                name=name,
                provider=provider,
                masked_key=self._mask_api_key(api_key),
                created_at=time.time(),
                permissions=permissions or [],
                expires_at=expires_at
            )
            
            # Save to persistent storage
            self._save_credentials()
            
            logger.info(f"Stored API key: {name} for provider: {provider}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store API key {name}: {e}")
            return False
    
    def get_api_key(self, name: str) -> Optional[str]:
        """
        Retrieve an API key by name.
        
        Args:
            name: Name of the API key
            
        Returns:
            The API key if found and active, None otherwise
        """
        
        if name not in self._credentials:
            return None
        
        metadata = self._metadata.get(name)
        if not metadata or not metadata.is_active:
            return None
        
        # Check expiration
        if metadata.expires_at and time.time() > metadata.expires_at:
            logger.warning(f"API key {name} has expired")
            self._deactivate_key(name)
            return None
        
        # Update usage statistics
        metadata.last_used = time.time()
        metadata.usage_count += 1
        
        return self._credentials[name]
    
    def list_api_keys(self) -> List[APIKeyInfo]:
        """List all stored API keys (metadata only)."""
        return list(self._metadata.values())
    
    def delete_api_key(self, name: str) -> bool:
        """
        Delete an API key.
        
        Args:
            name: Name of the API key to delete
            
        Returns:
            True if successfully deleted
        """
        
        try:
            if name in self._credentials:
                del self._credentials[name]
            
            if name in self._metadata:
                del self._metadata[name]
            
            self._save_credentials()
            
            logger.info(f"Deleted API key: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete API key {name}: {e}")
            return False
    
    def rotate_api_key(self, name: str, new_api_key: str) -> bool:
        """
        Rotate an existing API key.
        
        Args:
            name: Name of the API key to rotate
            new_api_key: New API key value
            
        Returns:
            True if successfully rotated
        """
        
        if name not in self._metadata:
            logger.warning(f"Cannot rotate non-existent API key: {name}")
            return False
        
        metadata = self._metadata[name]
        
        # Validate new key format
        if not self._validate_api_key_format(new_api_key, metadata.provider):
            logger.error(f"Invalid API key format for rotation: {name}")
            return False
        
        try:
            # Update key and metadata
            self._credentials[name] = new_api_key
            metadata.masked_key = self._mask_api_key(new_api_key)
            metadata.created_at = time.time()
            metadata.usage_count = 0
            metadata.last_used = None
            
            self._save_credentials()
            
            logger.info(f"Rotated API key: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate API key {name}: {e}")
            return False
    
    def _validate_api_key_format(self, api_key: str, provider: str) -> bool:
        """Validate API key format for specific provider."""
        
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Provider-specific validation
        if provider.lower() == 'openai':
            return api_key.startswith('sk-') and len(api_key) > 20
        elif provider.lower() == 'anthropic':
            return api_key.startswith('sk-ant-') and len(api_key) > 30
        elif provider.lower() == 'huggingface':
            return api_key.startswith('hf_') and len(api_key) > 20
        else:
            # Generic validation
            return len(api_key) > 10 and api_key.isascii()
    
    def _mask_api_key(self, api_key: str) -> str:
        """Create a masked version of the API key for display."""
        if len(api_key) <= 8:
            return "*" * len(api_key)
        
        prefix_len = min(4, len(api_key) // 4)
        suffix_len = min(4, len(api_key) // 4)
        
        prefix = api_key[:prefix_len]
        suffix = api_key[-suffix_len:]
        middle = "*" * (len(api_key) - prefix_len - suffix_len)
        
        return f"{prefix}{middle}{suffix}"
    
    def _deactivate_key(self, name: str) -> None:
        """Deactivate an API key."""
        if name in self._metadata:
            self._metadata[name].is_active = False
            self._save_credentials()
    
    def validate_key_permissions(self, name: str, required_permission: str) -> bool:
        """
        Check if an API key has a specific permission.
        
        Args:
            name: API key name
            required_permission: Permission to check
            
        Returns:
            True if key has permission
        """
        
        metadata = self._metadata.get(name)
        if not metadata or not metadata.is_active:
            return False
        
        # If no permissions specified, allow all
        if not metadata.permissions:
            return True
        
        return required_permission in metadata.permissions
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all API keys."""
        
        stats = {
            'total_keys': len(self._metadata),
            'active_keys': sum(1 for info in self._metadata.values() if info.is_active),
            'keys_by_provider': {},
            'total_usage': sum(info.usage_count for info in self._metadata.values()),
            'expired_keys': 0
        }
        
        current_time = time.time()
        
        for info in self._metadata.values():
            # Count by provider
            provider = info.provider
            if provider not in stats['keys_by_provider']:
                stats['keys_by_provider'][provider] = 0
            stats['keys_by_provider'][provider] += 1
            
            # Count expired keys
            if info.expires_at and current_time > info.expires_at:
                stats['expired_keys'] += 1
        
        return stats
    
    def cleanup_expired_keys(self) -> int:
        """Remove expired API keys."""
        
        current_time = time.time()
        expired_keys = []
        
        for name, info in self._metadata.items():
            if info.expires_at and current_time > info.expires_at:
                expired_keys.append(name)
        
        for name in expired_keys:
            self.delete_api_key(name)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired API keys")
        
        return len(expired_keys)
    
    def export_keys_metadata(self, output_path: Path, include_usage: bool = True) -> None:
        """
        Export API keys metadata (without actual keys) for backup/audit.
        
        Args:
            output_path: Path to export file
            include_usage: Whether to include usage statistics
        """
        
        export_data = {
            'export_timestamp': time.time(),
            'total_keys': len(self._metadata),
            'keys': []
        }
        
        for info in self._metadata.values():
            key_data = {
                'name': info.name,
                'provider': info.provider,
                'masked_key': info.masked_key,
                'created_at': info.created_at,
                'is_active': info.is_active,
                'expires_at': info.expires_at,
                'permissions': info.permissions
            }
            
            if include_usage:
                key_data.update({
                    'last_used': info.last_used,
                    'usage_count': info.usage_count
                })
            
            export_data['keys'].append(key_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported API keys metadata to {output_path}")
    
    def secure_wipe(self) -> None:
        """Securely wipe all stored credentials."""
        
        try:
            # Clear in-memory data
            self._credentials.clear()
            self._metadata.clear()
            
            # Remove files
            if self.credentials_file.exists():
                # Overwrite file before deletion
                with open(self.credentials_file, 'wb') as f:
                    f.write(secrets.token_bytes(1024))
                self.credentials_file.unlink()
            
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            # Remove master key
            key_file = self.credentials_dir / ".master_key"
            if key_file.exists():
                with open(key_file, 'wb') as f:
                    f.write(secrets.token_bytes(64))
                key_file.unlink()
            
            logger.info("Securely wiped all stored credentials")
            
        except Exception as e:
            logger.error(f"Failed to securely wipe credentials: {e}")
            raise SecurityError(
                message="Failed to securely wipe credentials",
                error_code=ErrorCodes.SECURITY_UNAUTHORIZED_ACCESS,
                original_error=e
            )
    
    def __del__(self):
        """Cleanup on object destruction."""
        # Ensure sensitive data is cleared from memory
        if hasattr(self, '_credentials'):
            self._credentials.clear()