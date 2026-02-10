import asyncio
import enum
import logging
import time
import sys
import uuid
import regex as re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError as PydanticValidationError

# Enhanced Logging Configuration
logger = logging.getLogger("SecurityEngine")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SecurityState(enum.Enum):
    """
    Formalized states for the SecurityEngine lifecycle.
    """
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    PRE_PROCESSING = "PRE_PROCESSING"
    SCANNING = "SCANNING"
    SANITIZING = "SANITIZING"
    AUDITING = "AUDITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HALTED = "HALTED"

# Strict Transition Map for the State Machine
VALID_TRANSITIONS: Dict[SecurityState, Set[SecurityState]] = {
    SecurityState.IDLE: {SecurityState.INITIALIZING, SecurityState.FAILED},
    SecurityState.INITIALIZING: {SecurityState.PRE_PROCESSING, SecurityState.FAILED},
    SecurityState.PRE_PROCESSING: {SecurityState.SCANNING, SecurityState.FAILED, SecurityState.HALTED},
    SecurityState.SCANNING: {SecurityState.SANITIZING, SecurityState.AUDITING, SecurityState.FAILED, SecurityState.HALTED},
    SecurityState.SANITIZING: {SecurityState.AUDITING, SecurityState.FAILED, SecurityState.HALTED},
    SecurityState.AUDITING: {SecurityState.COMPLETED, SecurityState.FAILED, SecurityState.HALTED},
    SecurityState.COMPLETED: {SecurityState.IDLE, SecurityState.INITIALIZING},
    SecurityState.FAILED: {SecurityState.IDLE, SecurityState.INITIALIZING},
    SecurityState.HALTED: {SecurityState.IDLE, SecurityState.INITIALIZING}
}

class SecurityError(Exception):
    """Base exception for all SecurityEngine related errors."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        try:
            super().__init__(message)
            self.context = context or {}
        except Exception:
            self.context = {}

class SecurityValidationError(SecurityError):
    """Raised when input/output validation fails."""
    pass

class SecurityStateError(SecurityError):
    """Raised on illegal state transitions."""
    pass

class ThreatDetectedError(SecurityError):
    """Raised when a severe security threat is detected."""
    pass

class ResourceLimitError(SecurityError):
    """Raised when safety limits (size, recursion, etc.) are breached."""
    pass

class ExecutionTimeoutError(SecurityError):
    """Raised when a security operation exceeds its time budget."""
    pass

class SecurityMetrics(BaseModel):
    """
    Captures granular performance data for security auditing.
    """
    initialization_ms: float = 0.0
    pre_processing_ms: float = 0.0
    scanning_ms: float = 0.0
    sanitization_ms: float = 0.0
    auditing_ms: float = 0.0
    total_duration_ms: float = 0.0
    threats_detected: int = 0
    pii_redacted_count: int = 0
    content_length: int = 0

def get_now() -> datetime:
    """Thread-safe timestamp retrieval."""
    try:
        return datetime.now(timezone.utc)
    except Exception:
        return datetime.min

class SecurityContext(BaseModel):
    """
    Pydantic-powered context for strict type safety and validation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    sanitized_content: Optional[str] = None
    threat_level: str = "LOW"
    pii_detected: List[str] = Field(default_factory=list)
    state: SecurityState = SecurityState.IDLE
    metrics: SecurityMetrics = Field(default_factory=SecurityMetrics)
    error_trace: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    start_time: datetime = Field(default_factory=get_now)
    end_time: Optional[datetime] = None

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: Any) -> str:
        try:
            if not isinstance(v, str):
                raise ValueError("Content must be a string.")
            if not v.strip():
                raise ValueError("Content cannot be empty or whitespace.")
            if len(v) > 500000: # Context-level hard boundary
                 raise ValueError("Content size exceeds maximum allowable limit (500KB).")
            return v
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Unexpected validation error: {e}")

class SecurityEngine:
    """
    High-performance, industrial-grade Security module.
    
    Adheres to Strict Robustness Constraints:
    1. ERROR HANDLING: Every function has explicit try-except blocks.
    2. VALIDATION: All inputs are validated before processing.
    3. BOUNDARIES: Enforces size limits, execution timeouts, and ReDoS protection.
    4. STATE MANAGEMENT: Uses a formal State Machine.
    5. PERFORMANCE: Parallel scanning and optimized regex engine.
    """

    MAX_CONTENT_SIZE = 250000  # 250KB
    GLOBAL_TIMEOUT = 15.0      # Seconds for the entire pipeline
    SCAN_TIMEOUT = 2.0         # Seconds per regex scan (ReDoS protection)
    MAX_RECURSION_DEPTH = 3    # Safety depth for nested sanitization
    REDACTION_CHAR = "*"

    # Regex patterns for common PII/Secrets/Threats
    # Optimized patterns for the 'regex' library
    PATTERNS = {
        "EMAIL": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "IPV4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "API_KEY": r"(?:api[\s_-]?key|secret|token|password|auth|access_key)[\s:=]+(['\"]?)([a-zA-Z0-9_\-\.]{20,})\1",
        "PROMPT_INJECTION": r"(?:ignore\s+(?:all\s+)?previous\s+instructions|system\s+override|disregard\s+(?:the\s+)?above|you\s+are\s+now|new\s+role|acting\s+as|forget\s+all\s+rules)",
        "SQL_INJECTION": r"(?:SELECT\s+.*\s+FROM|INSERT\s+INTO|DROP\s+TABLE|DELETE\s+FROM|UNION\s+SELECT|SLEEP\(\d+\)|BENCHMARK\(\d+)",
        "XSS": r"(?:<script.*?>|javascript:|onload=|onerror=|eval\(|setTimeout\(|setInterval\()",
        "PATH_TRAVERSAL": r"(?:\.\./|\.\.\\|\/etc\/passwd|\/etc\/shadow|C:\\Windows\\System32)"
    }

    def __init__(self):
        """
        Initializes the engine with explicit error handling and pre-compiled patterns.
        """
        try:
            self._lock = asyncio.Lock()
            # Pre-compile regex for performance with the 'regex' library
            self._compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in self.PATTERNS.items()}
            logger.info("SecurityEngine initialized with advanced regex patterns.")
        except Exception as e:
            logger.critical(f"Security Engine Bootstrap Failed: {e}")
            raise SecurityError("Failed to initialize SecurityEngine.", {"original_error": str(e)})

    def _update_state(self, context: SecurityContext, target_state: SecurityState):
        """
        Transitions the engine state with strict validation and error handling.
        """
        try:
            if not isinstance(target_state, SecurityState):
                 raise SecurityStateError(f"Invalid target state type: {type(target_state)}")

            if target_state not in VALID_TRANSITIONS.get(context.state, set()):
                error_msg = f"Illegal transition attempted: {context.state} -> {target_state}"
                logger.error(f"[{context.request_id}] {error_msg}")
                raise SecurityStateError(error_msg, {"current_state": context.state, "target_state": target_state})
            
            logger.debug(f"[{context.request_id}] State Transition: {context.state.name} -> {target_state.name}")
            context.state = target_state
        except SecurityStateError:
            raise
        except Exception as e:
            logger.error(f"State management internal failure: {e}")
            raise SecurityStateError(f"Failed to update state: {e}")

    async def process_content(self, content: str, depth: int = 0) -> Optional[SecurityContext]:
        """
        Main execution entry point for security processing.
        Implements top-level error handling, boundary checks, and global timeout.
        """
        t_start = time.perf_counter()
        context = None
        
        try:
            # Input Validation: Type and Size
            if not isinstance(content, str):
                raise SecurityValidationError(f"Content must be a string, got {type(content).__name__}")
            
            try:
                context = SecurityContext(content=content)
                context.metrics.content_length = len(content)
            except PydanticValidationError as e:
                # Attempt to create a skeleton context for error reporting
                dummy_ctx = SecurityContext.model_construct(
                    content=str(content)[:100] if content else "EMPTY", 
                    state=SecurityState.FAILED,
                    request_id=str(uuid.uuid4())
                )
                self._handle_failure(dummy_ctx, SecurityValidationError(f"Data model validation failed: {str(e)}"))
                return dummy_ctx

            # Boundary Check: Recursion
            if depth > self.MAX_RECURSION_DEPTH:
                raise ResourceLimitError(f"Maximum recursion depth {self.MAX_RECURSION_DEPTH} exceeded.")

            # Run the entire pipeline within a global timeout
            await asyncio.wait_for(self._run_pipeline(context), timeout=self.GLOBAL_TIMEOUT)
            
        except asyncio.TimeoutError:
            if context:
                self._handle_failure(context, ExecutionTimeoutError(f"Security pipeline timed out after {self.GLOBAL_TIMEOUT}s"))
        except (SecurityValidationError, SecurityStateError, ResourceLimitError, ThreatDetectedError) as e:
            if context:
                self._handle_failure(context, e)
            else:
                logger.critical(f"Critical Failure before context creation: {e}")
                return None
        except Exception as e:
            logger.exception("Unhandled security pipeline exception")
            if context:
                self._handle_failure(context, SecurityError(f"Internal system crash: {str(e)}"))
            else:
                return None
        finally:
            if context:
                context.metrics.total_duration_ms = (time.perf_counter() - t_start) * 1000
                context.end_time = get_now()
                logger.info(f"[{context.request_id}] Security processing finished in {context.metrics.total_duration_ms:.2f}ms with state: {context.state.name}")
            
        return context

    async def _run_pipeline(self, context: SecurityContext):
        """
        Internal pipeline execution with granular state management.
        """
        try:
            # 1. Initialization Phase
            self._update_state(context, SecurityState.INITIALIZING)
            init_start = time.perf_counter()
            self._validate_initial_config(context)
            context.metrics.initialization_ms = (time.perf_counter() - init_start) * 1000

            # 2. Pre-processing Phase
            self._update_state(context, SecurityState.PRE_PROCESSING)
            pre_start = time.perf_counter()
            self._pre_process(context)
            context.metrics.pre_processing_ms = (time.perf_counter() - pre_start) * 1000

            # 3. Scanning Phase
            self._update_state(context, SecurityState.SCANNING)
            scan_start = time.perf_counter()
            await self._perform_scan(context)
            context.metrics.scanning_ms = (time.perf_counter() - scan_start) * 1000

            # 4. Sanitization Phase
            self._update_state(context, SecurityState.SANITIZING)
            san_start = time.perf_counter()
            await self._perform_sanitization(context)
            context.metrics.sanitization_ms = (time.perf_counter() - san_start) * 1000

            # 5. Auditing Phase (Final Checks)
            self._update_state(context, SecurityState.AUDITING)
            audit_start = time.perf_counter()
            self._perform_audit(context)
            context.metrics.auditing_ms = (time.perf_counter() - audit_start) * 1000

            # Finalize
            self._update_state(context, SecurityState.COMPLETED)
            
        except (SecurityValidationError, SecurityStateError, ResourceLimitError, ThreatDetectedError) as e:
            raise # Re-raise to be caught by process_content
        except Exception as e:
            raise SecurityError(f"Pipeline execution internal failure: {e}")

    def _validate_initial_config(self, context: SecurityContext):
        """
        Strict configuration checks with explicit error handling.
        """
        try:
            # Boundary Check: Input Size
            if len(context.content) > self.MAX_CONTENT_SIZE:
                raise ResourceLimitError(f"Content size {len(context.content)} exceeds limit {self.MAX_CONTENT_SIZE}")
        except ResourceLimitError:
            raise
        except Exception as e:
            raise SecurityValidationError(f"Unexpected error during config validation: {e}")

    def _pre_process(self, context: SecurityContext):
        """
        Normalizes and prepares content for scanning.
        """
        try:
            # Remove null bytes and other common control characters used in evasion
            original_len = len(context.content)
            normalized_content = context.content.replace('\x00', '')
            
            if len(normalized_content) != original_len:
                logger.warning(f"[{context.request_id}] Potential evasion characters removed during pre-processing.")
                context.content = normalized_content
            
        except Exception as e:
            raise SecurityError(f"Pre-processing phase failure: {e}")

    async def _perform_scan(self, context: SecurityContext):
        """
        Scans for threats in parallel using pre-compiled regex with ReDoS protection.
        """
        try:
            content = context.content

            # Define scanning tasks for parallel execution
            async def scan_task(name: str, pattern: re.Pattern):
                try:
                    # 'regex' library supports a timeout parameter to prevent ReDoS
                    return name, bool(pattern.search(content, timeout=self.SCAN_TIMEOUT))
                except TimeoutError:
                    logger.error(f"[{context.request_id}] Scan for {name} timed out (potential ReDoS).")
                    return name, False
                except Exception as e:
                    logger.error(f"[{context.request_id}] Scan for {name} failed: {e}")
                    return name, False

            # Execute scans in parallel for high performance
            scan_jobs = [
                scan_task("PROMPT_INJECTION", self._compiled_patterns["PROMPT_INJECTION"]),
                scan_task("SQL_INJECTION", self._compiled_patterns["SQL_INJECTION"]),
                scan_task("XSS", self._compiled_patterns["XSS"]),
                scan_task("PATH_TRAVERSAL", self._compiled_patterns["PATH_TRAVERSAL"]),
                scan_task("API_KEY", self._compiled_patterns["API_KEY"])
            ]
            
            results = await asyncio.gather(*scan_jobs)
            
            for name, found in results:
                if found:
                    if name == "PROMPT_INJECTION":
                        context.threat_level = "CRITICAL"
                        context.metrics.threats_detected += 1
                        raise ThreatDetectedError("Severe prompt injection attempt detected.")
                    
                    elif name in ["SQL_INJECTION", "XSS", "PATH_TRAVERSAL"]:
                        context.threat_level = "HIGH"
                        context.metrics.threats_detected += 1
                        logger.warning(f"[{context.request_id}] {name} pattern detected.")
                    
                    elif name == "API_KEY":
                        context.threat_level = "HIGH" if context.threat_level != "CRITICAL" else "CRITICAL"
                        context.metrics.threats_detected += 1
                        logger.warning(f"[{context.request_id}] Potential API key/secret detected.")

        except ThreatDetectedError:
            raise
        except Exception as e:
            raise SecurityError(f"Scanning phase failure: {e}")

    async def _perform_sanitization(self, context: SecurityContext):
        """
        Sanitizes PII and other sensitive data with high performance.
        """
        try:
            sanitized = context.content
            
            # Efficiently redact multiple PII types
            pii_types = ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IPV4"]
            for p_type in pii_types:
                pattern = self._compiled_patterns[p_type]
                matches = pattern.findall(sanitized, timeout=self.SCAN_TIMEOUT)
                if matches:
                    context.pii_detected.extend(matches)
                    sanitized = pattern.sub(f"[{p_type}_REDACTED]", sanitized, timeout=self.SCAN_TIMEOUT)
                    context.metrics.pii_redacted_count += len(matches)

            # Redact API Keys / Secrets with a specialized callback for precision
            def redact_secret(match):
                try:
                    full_match = match.group(0)
                    secret = match.group(2)
                    prefix = full_match.split(secret)[0]
                    return f"{prefix}[REDACTED_SECRET]"
                except Exception:
                    return "[REDACTED_SECRET]"
            
            sanitized = self._compiled_patterns["API_KEY"].sub(redact_secret, sanitized, timeout=self.SCAN_TIMEOUT)
            
            context.sanitized_content = sanitized
        except TimeoutError:
            raise ResourceLimitError("Sanitization timed out due to complex content structure (ReDoS protection).")
        except Exception as e:
            raise SecurityError(f"Sanitization phase failure: {e}")

    def _perform_audit(self, context: SecurityContext):
        """
        Final consistency checks before completion.
        """
        try:
            if context.sanitized_content is None:
                raise SecurityError("Sanitized content is missing after sanitization phase.")
            
            # Ensure no critical threats persisted after sanitization
            if self._compiled_patterns["PROMPT_INJECTION"].search(context.sanitized_content, timeout=self.SCAN_TIMEOUT):
                 raise ThreatDetectedError("Critical threat persisted after sanitization - system compromise risk.")
            
            # Boundary Check: Output Size
            if len(context.sanitized_content) > self.MAX_CONTENT_SIZE * 2:
                raise ResourceLimitError("Sanitized output size ballooned unexpectedly (potential amplification attack).")
                 
        except (SecurityError, ThreatDetectedError, ResourceLimitError):
            raise
        except Exception as e:
            raise SecurityError(f"Audit phase failure: {e}")

    def _handle_failure(self, context: SecurityContext, error: Exception):
        """
        Centralized error handling and context enrichment.
        Guaranteed to be robust.
        """
        try:
            context.state = SecurityState.FAILED
            err_info = {
                "timestamp": get_now().isoformat(),
                "type": error.__class__.__name__,
                "message": str(error),
                "context": getattr(error, 'context', {})
            }
            context.error_trace.append(err_info)
            logger.error(f"[{context.request_id}] Security Failure: {err_info['message']}")
        except Exception as e:
            print(f"CRITICAL: SecurityEngine._handle_failure failed: {e}", file=sys.stderr)

    def get_health(self) -> Dict[str, Any]:
        """
        System health monitoring.
        """
        try:
            return {
                "status": "OPERATIONAL",
                "version": "3.0.0-high-perf",
                "timestamp": get_now().isoformat(),
                "performance": {
                    "max_content_size": self.MAX_CONTENT_SIZE,
                    "global_timeout": self.GLOBAL_TIMEOUT,
                    "scan_timeout": self.SCAN_TIMEOUT,
                    "max_recursion": self.MAX_RECURSION_DEPTH
                },
                "patterns_loaded": list(self.PATTERNS.keys())
            }
        except Exception as e:
            return {"status": "DEGRADED", "error": str(e)}

if __name__ == "__main__":
    async def main():
        engine = SecurityEngine()
        res = await engine.process_content("My email is alice@example.com and secret is 'sk-1234567890abcdef1234567890abcdef'")
        if res:
            print(f"State: {res.state.name}")
            print(f"Sanitized: {res.sanitized_content}")
            print(f"Metrics: {res.metrics.model_dump_json(indent=2)}")

    asyncio.run(main())