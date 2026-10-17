class ManualAssistantError(Exception):
    """Base error exposed as a recoverable application failure."""


class CatalogError(ManualAssistantError):
    pass


class IndexNotReadyError(ManualAssistantError):
    pass


class ProviderError(ManualAssistantError):
    pass


class CitationValidationError(ManualAssistantError):
    pass
