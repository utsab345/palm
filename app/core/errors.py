class AppError(Exception):
    status_code = 500
    code = "internal_error"


class ConfigurationError(AppError):
    code = "configuration_error"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class LLMRateLimitError(LLMError):
    status_code = 429
    code = "llm_rate_limited"


class LLMTimeoutError(LLMError):
    status_code = 504
    code = "llm_timeout"


class LLMUnavailableError(LLMError):
    status_code = 503
    code = "llm_unavailable"


class LLMResponseError(LLMError):
    code = "llm_invalid_response"


class EmbeddingError(AppError):
    status_code = 502
    code = "embedding_error"


class EmbeddingDimensionError(EmbeddingError):
    code = "embedding_dimension_mismatch"


class EmbeddingRateLimitError(EmbeddingError):
    status_code = 429
    code = "embedding_rate_limited"


class EmbeddingTimeoutError(EmbeddingError):
    status_code = 504
    code = "embedding_timeout"


class EmbeddingUnavailableError(EmbeddingError):
    status_code = 503
    code = "embedding_unavailable"


class VectorStoreError(AppError):
    status_code = 503
    code = "vector_store_error"


class CollectionDimensionMismatch(VectorStoreError):
    code = "collection_dimension_mismatch"


class DocumentError(AppError):
    status_code = 400
    code = "document_error"


class UnsupportedDocumentError(DocumentError):
    status_code = 415
    code = "unsupported_document_type"


class DocumentTooLargeError(DocumentError):
    status_code = 413
    code = "document_too_large"


class DocumentProcessingError(DocumentError):
    code = "document_processing_error"


class InvalidChunkingConfig(DocumentError):
    status_code = 422
    code = "invalid_chunking_config"


class PersistenceError(AppError):
    status_code = 500
    code = "persistence_error"


class ChatMemoryError(AppError):
    status_code = 503
    code = "chat_memory_error"
