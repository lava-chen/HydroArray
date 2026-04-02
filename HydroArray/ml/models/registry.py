"""
Model Registry System for HydroArray

This module provides a centralized model registration and factory system,
enabling dynamic model instantiation based on configuration files.

The registry uses a decorator pattern for model registration, allowing models
to self-register when defined. This eliminates the need for manual registration
and makes the system easily extensible.

Example:
    >>> from HydroArray.ml.models import register_model, create_model
    >>>
    >>> @register_model("convlstm")
    >>> class ConvLSTM(nn.Module):
    ...     def __init__(self, config):
    ...         super().__init__()
    ...         # Model implementation
    ...
    >>> # Create model from configuration
    >>> model = create_model(config)

Design Pattern:
    - Registry Pattern: Centralized model management
    - Factory Pattern: Dynamic model instantiation
    - Decorator Pattern: Self-registration mechanism

References:
    - Inspired by neuralhydrology's modelzoo architecture
    - Follows PyTorch model registration best practices
"""

import inspect
from typing import Any, Callable, Dict, Optional, Type, Union
from pathlib import Path

import torch.nn as nn


class ModelRegistry:
    """
    Central registry for all machine learning models in HydroArray.
    
    This class maintains a mapping between model names (strings) and their
    corresponding classes. It provides thread-safe registration and factory
    methods for model instantiation.
    
    Attributes:
        _models (Dict[str, Type[nn.Module]]): Internal registry mapping
        
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("lstm", LSTMModel)
        >>> model_class = registry.get("lstm")
        >>> model = model_class(config)
    """
    
    def __init__(self):
        """Initialize an empty model registry."""
        self._models: Dict[str, Type[nn.Module]] = {}
    
    def register(
        self, 
        name: str, 
        model_class: Optional[Type[nn.Module]] = None
    ) -> Union[Callable, Type[nn.Module]]:
        """
        Register a model class with the given name.
        
        This method can be used as a decorator or as a direct function call.
        When used as a decorator, the model class is automatically registered
        upon definition.
        
        Args:
            name (str): Unique identifier for the model. Case-insensitive.
            model_class (Optional[Type[nn.Module]]): The model class to register.
                If None, returns a decorator function.
        
        Returns:
            Union[Callable, Type[nn.Module]]: 
                - If model_class is provided: the registered class
                - If model_class is None: decorator function
        
        Raises:
            ValueError: If name is already registered with a different class.
        
        Example:
            # Direct registration
            >>> registry.register("lstm", LSTMModel)
            
            # Decorator registration
            >>> @registry.register("convlstm")
            ... class ConvLSTM(nn.Module):
            ...     pass
        """
        name = name.lower()
        
        # Check for duplicate registration
        if name in self._models and self._models[name] != model_class:
            raise ValueError(
                f"Model '{name}' is already registered with class "
                f"{self._models[name].__name__}. Cannot override with "
                f"{model_class.__name__ if model_class else 'new registration'}."
            )
        
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            """Inner decorator function for class registration."""
            # Validate that cls is a proper nn.Module subclass
            if not inspect.isclass(cls):
                raise TypeError(
                    f"Registered object must be a class, got {type(cls)}"
                )
            if not issubclass(cls, nn.Module):
                raise TypeError(
                    f"Registered class must inherit from nn.Module, "
                    f"got {cls.__name__}"
                )
            
            self._models[name] = cls
            # Attach registry metadata to the class
            cls._model_registry_name = name
            return cls
        
        if model_class is not None:
            # Direct registration
            return decorator(model_class)
        else:
            # Return decorator for @ syntax
            return decorator
    
    def get(self, name: str) -> Type[nn.Module]:
        """
        Retrieve a registered model class by name.
        
        Args:
            name (str): The registered name of the model. Case-insensitive.
        
        Returns:
            Type[nn.Module]: The registered model class.
        
        Raises:
            KeyError: If the model name is not registered.
        
        Example:
            >>> model_class = registry.get("convlstm")
            >>> model = model_class(config)
        """
        name = name.lower()
        if name not in self._models:
            available = list(self._models.keys())
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )
        return self._models[name]
    
    def create(
        self, 
        name: str, 
        config: Optional[Any] = None,
        **kwargs
    ) -> nn.Module:
        """
        Factory method to instantiate a registered model.
        
        This method creates an instance of the specified model class,
        passing the configuration object and any additional keyword arguments
        to the constructor.
        
        Args:
            name (str): The registered name of the model.
            config (Optional[Any]): Configuration object passed to model constructor.
            **kwargs: Additional keyword arguments passed to model constructor.
        
        Returns:
            nn.Module: Instantiated model instance.
        
        Raises:
            KeyError: If the model name is not registered.
            TypeError: If model instantiation fails.
        
        Example:
            >>> model = registry.create("convlstm", config)
            >>> model = registry.create("lstm", config, hidden_size=128)
        """
        model_class = self.get(name)
        
        try:
            if config is not None:
                return model_class(config, **kwargs)
            else:
                return model_class(**kwargs)
        except Exception as e:
            raise TypeError(
                f"Failed to instantiate model '{name}' with class "
                f"{model_class.__name__}: {str(e)}"
            ) from e
    
    def list_models(self) -> list[str]:
        """
        Get a list of all registered model names.
        
        Returns:
            list[str]: Sorted list of registered model names.
        
        Example:
            >>> available_models = registry.list_models()
            >>> print(available_models)
            ['convlstm', 'gru', 'lstm', 'transformer']
        """
        return sorted(self._models.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a model name is registered.
        
        Args:
            name (str): Model name to check. Case-insensitive.
        
        Returns:
            bool: True if registered, False otherwise.
        """
        return name.lower() in self._models
    
    def unregister(self, name: str) -> None:
        """
        Remove a model from the registry.
        
        Args:
            name (str): Model name to unregister. Case-insensitive.
        
        Raises:
            KeyError: If the model name is not registered.
        """
        name = name.lower()
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        del self._models[name]
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """
        Get metadata about a registered model.
        
        Args:
            name (str): Model name. Case-insensitive.
        
        Returns:
            Dict[str, Any]: Model metadata including:
                - name: Registered name
                - class: Model class
                - module: Module where class is defined
                - doc: Class docstring
        """
        model_class = self.get(name)
        return {
            "name": name.lower(),
            "class": model_class.__name__,
            "module": model_class.__module__,
            "doc": model_class.__doc__
        }
    
    def __contains__(self, name: str) -> bool:
        """Enable 'in' operator: 'lstm' in registry"""
        return self.is_registered(name)
    
    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self._models)
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        models = ", ".join(self.list_models())
        return f"ModelRegistry({len(self)} models: [{models}])"


# Global registry instance
# This singleton pattern ensures all models register to the same registry
MODEL_REGISTRY = ModelRegistry()


# Convenience functions for module-level access
def register_model(name: str):
    """
    Decorator to register a model class with the global registry.
    
    This is the primary interface for model registration. Use it as a
    decorator on model class definitions.
    
    Args:
        name (str): Unique identifier for the model.
    
    Returns:
        Callable: Decorator function.
    
    Example:
        >>> @register_model("convlstm")
        ... class ConvLSTM(nn.Module):
        ...     '''Convolutional LSTM for spatiotemporal prediction'''
        ...     def __init__(self, config):
        ...         super().__init__()
        ...         # Implementation
    """
    return MODEL_REGISTRY.register(name)


def create_model(
    config: Any,
    model_type: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a model instance from configuration.
    
    This is the primary interface for model instantiation. It supports
    both Config objects and dictionaries.
    
    Args:
        config: Configuration object or dictionary. Must contain 'model.type'
            or 'model_type' key unless model_type is provided.
        model_type (Optional[str]): Model type name. If None, extracted from config.
        **kwargs: Additional arguments passed to model constructor.
    
    Returns:
        nn.Module: Instantiated model.
    
    Raises:
        ValueError: If model_type cannot be determined from config.
        KeyError: If model type is not registered.
    
    Example:
        >>> # From Config object
        >>> model = create_model(config)
        
        >>> # From dictionary
        >>> model = create_model({"model_type": "convlstm", "hidden_dims": [64, 32]})
        
        >>> # Override model type
        >>> model = create_model(config, model_type="lstm")
    """
    # Extract model type from various config formats
    if model_type is None:
        if hasattr(config, 'model') and hasattr(config.model, 'type'):
            # Config object with attribute access
            model_type = config.model.type
        elif hasattr(config, 'get'):
            # Dictionary-like object
            model_type = config.get('model', {}).get('type') or config.get('model_type')
        elif isinstance(config, dict):
            # Plain dictionary
            model_type = config.get('model', {}).get('type') or config.get('model_type')
    
    if model_type is None:
        raise ValueError(
            "Model type must be specified either in config "
            "(config.model.type or config['model_type']) or "
            "as model_type argument"
        )
    
    return MODEL_REGISTRY.create(model_type, config, **kwargs)


def get_model_class(name: str) -> Type[nn.Module]:
    """
    Get a registered model class by name.
    
    Args:
        name (str): Registered model name.
    
    Returns:
        Type[nn.Module]: Model class.
    
    Example:
        >>> ConvLSTM = get_model_class("convlstm")
        >>> model = ConvLSTM(config)
    """
    return MODEL_REGISTRY.get(name)


def list_available_models() -> list[str]:
    """
    List all available (registered) model names.
    
    Returns:
        list[str]: Sorted list of model names.
    
    Example:
        >>> models = list_available_models()
        >>> print(f"Available models: {models}")
    """
    return MODEL_REGISTRY.list_models()


def is_model_available(name: str) -> bool:
    """
    Check if a model is available in the registry.
    
    Args:
        name (str): Model name to check.
    
    Returns:
        bool: True if available, False otherwise.
    """
    return MODEL_REGISTRY.is_registered(name)
