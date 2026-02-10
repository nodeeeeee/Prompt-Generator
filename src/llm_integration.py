import os
import logging
import asyncio
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
import litellm
from litellm import exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

load_dotenv()

# Configure logging to capture errors effectively
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMIntegrationError(Exception):
    """Base exception for LLM integration errors."""
    pass

class LLMContextLimitError(LLMIntegrationError):
    """Raised when context window is exceeded."""
    pass

class LLMRateLimitError(LLMIntegrationError):
    """Raised when rate limit is exceeded."""
    pass

class LLMServiceError(LLMIntegrationError):
    """Raised when the LLM service fails."""
    pass

class LLMClient:
    """
    High-performance, robust client for interacting with various LLM providers.
    Supports both synchronous and asynchronous operations with strict error handling.
    """
    def __init__(self, default_model: str = "gpt-5.2", timeout: int = 30, max_retries: int = 3):
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries

    def _get_retry_decorator(self):
        """
        Returns a tenacity retry decorator configured with exponential backoff.
        """
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((
                exceptions.RateLimitError,
                exceptions.ServiceUnavailableError,
                exceptions.Timeout,
                exceptions.APIConnectionError,
                exceptions.APIError
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )

    def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Synchronous wrapper for generating completion with strict error handling and retries.
        
        Args:
            messages: A list of message dictionaries (role, content).
            model: The model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The content of the response.
            
        Raises:
            LLMContextLimitError: If the prompt exceeds the model's context window.
            LLMRateLimitError: If rate limits are hit even after retries.
            LLMServiceError: For other API failures.
        """
        target_model = model or self.default_model
        
        # Define inner function to apply retry logic dynamically based on instance config
        @self._get_retry_decorator()
        def _call_litellm():
            return litellm.completion(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout
            )

        try:
            response = _call_litellm()
            return response.choices[0].message.content
        except exceptions.ContextWindowExceededError as e:
            logger.error(f"Context window exceeded for model {target_model}: {e}")
            raise LLMContextLimitError(f"Context window exceeded: {e}") from e
        except exceptions.RateLimitError as e:
            logger.error(f"Rate limit exceeded for {target_model}. Retrying via tenacity...")
            raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
        except Exception as e:
            logger.error(f"LLM Call Error ({type(e).__name__}) for {target_model}: {e}")
            raise LLMServiceError(f"LLM service failure: {e}") from e

    async def agenerate_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Asynchronous method with strict timeout and retry logic.
        """
        target_model = model or self.default_model
        effective_timeout = timeout or self.timeout
        
        @self._get_retry_decorator()
        async def _call_litellm_async():
            # Second layer of timeout protection via asyncio.wait_for
            return await asyncio.wait_for(
                litellm.acompletion(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=effective_timeout # LiteLLM internal timeout
                ),
                timeout=effective_timeout + 5 # Safety buffer
            )

        try:
            response = await _call_litellm_async()
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.error(f"Global timeout reached for model {target_model}")
            raise LLMServiceError(f"LLM request timed out after {effective_timeout}s")
        except exceptions.ContextWindowExceededError as e:
            logger.error(f"Context window exceeded for model {target_model}: {e}")
            raise LLMContextLimitError(f"Context window exceeded: {e}") from e
        except exceptions.RateLimitError as e:
            logger.error(f"Rate limit exceeded for {target_model}. Retrying via tenacity...")
            raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
        except Exception as e:
            logger.error(f"LLM Call Error ({type(e).__name__}) for {target_model}: {e}")
            raise LLMServiceError(f"LLM service failure: {e}") from e

    def list_available_models(self) -> List[str]:
        return [
            "gpt-5.2",
            "gemini-3",
            "claude-4.5",
            "o3-mini",
            "o1",
            "gpt-4o",
            "claude-3.5-sonnet",
            "gemini-2.0-flash",
            "deepseek-v3",
            "claude-3-opus",
            "gpt-4-turbo"
        ]
        

        

    