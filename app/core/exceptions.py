"""
Domain exception hierarchy.

Raising typed exceptions (not strings) gives the API layer
precise control over which HTTP status code to return.
Never raise generic Exception in service code.
"""


class CeilingAIBaseError(Exception):
    """Root for all application errors."""


class ImageValidationError(CeilingAIBaseError):
    """Raised when the uploaded image cannot be decoded or fails constraints."""


class InferenceError(CeilingAIBaseError):
    """Raised when a model forward pass fails unexpectedly."""


class ScaleCalibrationError(CeilingAIBaseError):
    """Raised when pixels_per_meter produces physically impossible measurements."""


class ModelNotLoadedError(CeilingAIBaseError):
    """Raised when a model is accessed before it has been loaded."""
