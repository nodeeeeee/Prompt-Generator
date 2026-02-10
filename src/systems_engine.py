import asyncio
import enum
import logging
import time
import sys
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError as PydanticValidationError

from src.llm_integration import LLMClient, LLMIntegrationError
from src.clarification_agent import ClarificationAgent
from src.prompt_builder import PromptBuilder
from src.security_engine import SecurityEngine, SecurityState, SecurityContext

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SystemsEngine")

class EngineState(enum.Enum):
    """
    Formalized states for the SystemsEngine lifecycle.
    """
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    SECURITY_SCAN = "SECURITY_SCAN"
    VALIDATING = "VALIDATING"
    CLARIFYING = "CLARIFYING"
    BUILDING = "BUILDING"
    OPTIMIZING = "OPTIMIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HALTED = "HALTED"

# Strict Transition Map for the State Machine
VALID_TRANSITIONS: Dict[EngineState, Set[EngineState]] = {
    EngineState.IDLE: {EngineState.INITIALIZING, EngineState.FAILED},
    EngineState.INITIALIZING: {EngineState.SECURITY_SCAN, EngineState.FAILED},
    EngineState.SECURITY_SCAN: {EngineState.VALIDATING, EngineState.FAILED, EngineState.HALTED},
    EngineState.VALIDATING: {EngineState.CLARIFYING, EngineState.BUILDING, EngineState.FAILED},
    EngineState.CLARIFYING: {EngineState.BUILDING, EngineState.FAILED, EngineState.HALTED},
    EngineState.BUILDING: {EngineState.OPTIMIZING, EngineState.COMPLETED, EngineState.FAILED, EngineState.HALTED},
    EngineState.OPTIMIZING: {EngineState.COMPLETED, EngineState.FAILED},
    EngineState.COMPLETED: {EngineState.IDLE, EngineState.INITIALIZING},
    EngineState.FAILED: {EngineState.IDLE, EngineState.INITIALIZING},
    EngineState.HALTED: {EngineState.IDLE, EngineState.INITIALIZING}
}

class EngineError(Exception):
    """Base exception for all SystemsEngine related errors."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        try:
            super().__init__(message)
            self.context = context or {}
        except Exception:
            # Fallback if even exception initialization fails
            pass

class ValidationError(EngineError):
    """Raised when input/output validation fails."""
    pass

class StateError(EngineError):
    """Raised on illegal state transitions."""
    pass

class ComponentError(EngineError):
    """Raised when an internal component fails to meet its contract."""
    pass

class BoundaryError(EngineError):
    """Raised when safety limits (size, recursion, etc.) are breached."""
    pass

class ExecutionTimeoutError(EngineError):
    """Raised when a pipeline step exceeds its time budget."""
    pass

class PerformanceMetrics(BaseModel):
    """
    Captures granular performance data for optimization analysis.
    """
    initialization_ms: float = 0.0
    validation_ms: float = 0.0
    clarification_ms: float = 0.0
    building_ms: float = 0.0
    optimizing_ms: float = 0.0
    total_duration_ms: float = 0.0
    llm_calls: int = 0
    estimated_tokens: int = 0

def get_now():
    """Thread-safe timestamp retrieval."""
    try:
        return datetime.now(timezone.utc)
    except Exception:
        return datetime.min # Fallback

class SystemContext(BaseModel):
    """
    Pydantic-powered context for strict type safety and validation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intention: str
    model: str = "o3-mini"
    mode: str = "iterative"
    questions: List[str] = []
    answers: List[str] = []
    final_prompt: Optional[str] = None
    state: EngineState = EngineState.IDLE
    metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    error_trace: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    start_time: datetime = Field(default_factory=get_now)
    end_time: Optional[datetime] = None

    @field_validator('intention')
    @classmethod
    def validate_intention_non_empty(cls, v):
        try:
            if not v or not v.strip():
                raise ValueError("Intention must not be empty.")
            if len(v) < 10:
                raise ValueError("Intention is too brief to process effectively.")
            if len(v) > 50000: # Context-level boundary
                 raise ValueError("Intention size exceeds hard limit.")
            return v
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Unexpected validation error: {e}")

class SystemsEngine:
    """
    High-performance, industrial-grade Systems module.
    
    Adheres to Strict Robustness Constraints:
    1. ERROR HANDLING: Every function has explicit try-except blocks.
    2. VALIDATION: All inputs are validated before processing.
    3. BOUNDARIES: Enforces size limits and recursion safety.
    4. STATE MANAGEMENT: Uses a formal State Machine.
    """

    MAX_INTENTION_SIZE = 25000
    MAX_Q_COUNT = 15
    MAX_RECURSION_DEPTH = 5
    STEP_TIMEOUT = 90.0  # Seconds per major step

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initializes the engine with explicit error handling.
        """
        try:
            self.client = llm_client or LLMClient()
            self.clarifier = ClarificationAgent(self.client)
            self.builder = PromptBuilder(self.client)
            self.security = SecurityEngine()
            self._lock = asyncio.Lock()
            logger.info("SystemsEngine initialized with core components including SecurityEngine.")
        except Exception as e:
            logger.critical(f"Engine Bootstrap Failed: {e}")
            raise EngineError("Failed to initialize SystemsEngine components.", {"original_error": str(e)})

    def _update_state(self, context: SystemContext, target_state: EngineState):
        """
        Transitions the engine state with strict validation and error handling.
        """
        try:
            if not isinstance(target_state, EngineState):
                 raise StateError(f"Invalid target state type: {type(target_state)}")

            if target_state not in VALID_TRANSITIONS.get(context.state, set()):
                error_msg = f"Illegal transition attempted: {context.state} -> {target_state}"
                logger.error(f"[{context.request_id}] {error_msg}")
                raise StateError(error_msg, {"current_state": context.state, "target_state": target_state})
            
            logger.debug(f"[{context.request_id}] State Transition: {context.state.name} -> {target_state.name}")
            context.state = target_state
        except StateError:
            raise
        except Exception as e:
            logger.error(f"State management internal failure: {e}")
            raise StateError(f"Failed to update state: {e}")

    async def run_pipeline(
        self, 
        intention: str, 
        model: str = "o3-mini", 
        mode: str = "iterative",
        answers: Optional[List[str]] = None,
        depth: int = 0
    ) -> SystemContext:
        """
        Main execution entry point for the prompt generation pipeline.
        Implements top-level error handling and boundary checks.
        """
        t_start = time.perf_counter()
        context = None
        
        try:
            # Boundary Check: Recursion
            if depth > self.MAX_RECURSION_DEPTH:
                raise BoundaryError(f"Maximum recursion depth {self.MAX_RECURSION_DEPTH} exceeded.")

            # Input Validation: Type and Size
            if not isinstance(intention, str):
                raise ValidationError("Intention must be a string.")
            if not isinstance(model, str):
                raise ValidationError("Model must be a string.")
            
            try:
                context = SystemContext(intention=intention, model=model, mode=mode, answers=answers or [])
            except PydanticValidationError as e:
                # Create a minimal context for error reporting
                dummy_ctx = SystemContext.model_construct(
                    intention=str(intention)[:100], 
                    model=str(model), 
                    mode=str(mode), 
                    state=EngineState.FAILED
                )
                self._handle_failure(dummy_ctx, ValidationError(f"Data model validation failed: {str(e)}"))
                return dummy_ctx

            # 1. Initialization Phase
            self._update_state(context, EngineState.INITIALIZING)
            init_start = time.perf_counter()
            self._validate_initial_config(context)
            context.metrics.initialization_ms = (time.perf_counter() - init_start) * 1000

            # 2. Security Scan Phase
            self._update_state(context, EngineState.SECURITY_SCAN)
            sec_start = time.perf_counter()
            sec_context = await self.security.process_content(context.intention)
            if not sec_context or sec_context.state == SecurityState.FAILED:
                reason = sec_context.error_trace[-1]["message"] if sec_context else "Security engine failure"
                raise ValidationError(f"Security check failed: {reason}")
            
            # Update intention with sanitized version
            context.intention = sec_context.sanitized_content
            context.metadata["security_metrics"] = sec_context.metrics.model_dump()
            
            # 3. Deep Validation Phase
            self._update_state(context, EngineState.VALIDATING)
            val_start = time.perf_counter()
            self._perform_deep_validation(context)
            context.metrics.validation_ms = (time.perf_counter() - val_start) * 1000

            # 3. Clarification Phase
            self._update_state(context, EngineState.CLARIFYING)
            if not context.answers:
                clar_start = time.perf_counter()
                await asyncio.wait_for(self._run_clarification(context), timeout=self.STEP_TIMEOUT)
                context.metrics.clarification_ms = (time.perf_counter() - clar_start) * 1000
            else:
                logger.info(f"[{context.request_id}] Skipping clarification: Pre-answered context provided.")

            # 4. Building Phase
            self._update_state(context, EngineState.BUILDING)
            build_start = time.perf_counter()
            await asyncio.wait_for(self._run_building(context), timeout=self.STEP_TIMEOUT)
            context.metrics.building_ms = (time.perf_counter() - build_start) * 1000

            # 5. Optimization Phase
            self._update_state(context, EngineState.OPTIMIZING)
            context.metrics.optimizing_ms = 0.1 # Placeholder for optimization logic

            # Finalize
            self._update_state(context, EngineState.COMPLETED)
            context.end_time = get_now()
            
        except asyncio.TimeoutError:
            if context:
                self._handle_failure(context, ExecutionTimeoutError(f"Pipeline step timed out after {self.STEP_TIMEOUT}s"))
        except (ValidationError, StateError, BoundaryError, ComponentError) as e:
            if context:
                self._handle_failure(context, e)
            else:
                logger.critical(f"Critical Failure before context creation: {e}")
        except Exception as e:
            logger.exception("Unhandled pipeline exception")
            if context:
                self._handle_failure(context, EngineError(f"Internal system crash: {str(e)}"))
        finally:
            if context:
                context.metrics.total_duration_ms = (time.perf_counter() - t_start) * 1000
                logger.info(f"[{context.request_id}] Pipeline execution finished in {context.metrics.total_duration_ms:.2f}ms with state: {context.state.name}")
            
        return context

    def _validate_initial_config(self, context: SystemContext):
        """
        Strict configuration checks with explicit error handling.
        """
        try:
            valid_modes = {"one-shot", "iterative", "chain-of-thought"}
            if context.mode not in valid_modes:
                raise ValidationError(f"Invalid mode '{context.mode}'. Must be one of {valid_modes}")
            
            # Boundary Check: Input Size
            if len(context.intention) > self.MAX_INTENTION_SIZE:
                raise BoundaryError(f"Intention size {len(context.intention)} exceeds limit {self.MAX_INTENTION_SIZE}")
        except (ValidationError, BoundaryError):
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected error during config validation: {e}")

    def _perform_deep_validation(self, context: SystemContext):
        """
        Advanced semantic validation of inputs.
        """
        try:
            # Check for repetitive garbage or obvious non-instructional input
            words = context.intention.split()
            if not words:
                raise ValidationError("Intention contains no words.")
                
            unique_words_ratio = len(set(words)) / len(words)
            if unique_words_ratio < 0.1 and len(words) > 20:
                 raise ValidationError("Intention appears to be non-instructional or highly repetitive (bot detection).")
            
            if len(words) > 10000:
                raise BoundaryError("Intention contains too many tokens for reliable processing.")
        except (ValidationError, BoundaryError):
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected error during deep validation: {e}")

    async def _run_clarification(self, context: SystemContext):
        """
        Executes clarification with soft-failure resilience and recursion safety.
        """
        try:
            questions = await self.clarifier.generate_questions(context.intention)
            if not isinstance(questions, list):
                raise ComponentError("Clarifier returned malformed output (expected list).")
            
            # Enforce boundary on question count
            context.questions = questions[:self.MAX_Q_COUNT]
            context.metrics.llm_calls += 1
        except Exception as e:
            logger.warning(f"[{context.request_id}] Clarification soft-failed: {e}. Proceeding with best-effort.")
            context.error_trace.append({
                "step": "clarification", 
                "error": str(e), 
                "severity": "WARNING",
                "timestamp": get_now().isoformat()
            })
            context.questions = []

    async def _run_building(self, context: SystemContext):
        """
        Executes the building phase with strict component contract enforcement.
        """
        try:
            result, disc_paths = await self.builder.build_prompt(
                intention=context.intention,
                answers=context.answers,
                questions=context.questions,
                mode=context.mode
            )
            
            if not result or not isinstance(result, str):
                raise ComponentError("PromptBuilder produced empty or invalid string output.")
            
            # Boundary Check: Output Size
            if len(result) > 100000:
                raise BoundaryError("Generated prompt exceeds safety size limit (100KB).")
                
            context.final_prompt = result
            context.metadata["discovered_files"] = disc_paths
            context.metrics.llm_calls += 1
        except LLMIntegrationError as e:
            raise ComponentError(f"LLM connectivity failure during building: {e}")
        except (ComponentError, BoundaryError):
            raise
        except Exception as e:
            raise ComponentError(f"Building phase critical failure: {e}")

    def _handle_failure(self, context: SystemContext, error: Exception):
        """
        Centralized error handling and context enrichment.
        Guaranteed to not raise exceptions itself.
        """
        try:
            context.state = EngineState.FAILED
            err_info = {
                "timestamp": get_now().isoformat(),
                "type": error.__class__.__name__,
                "message": str(error),
                "context": getattr(error, 'context', {})
            }
            context.error_trace.append(err_info)
            logger.error(f"[{context.request_id}] Pipeline Failure: {err_info['message']}")
        except Exception as e:
            # Last resort logging
            print(f"CRITICAL: _handle_failure failed: {e}", file=sys.stderr)

    def get_health(self) -> Dict[str, Any]:
        """
        System health monitoring with explicit error handling.
        """
        try:
            return {
                "status": "OPERATIONAL",
                "version": "2.5.0-robust",
                "timestamp": get_now().isoformat(),
                "components": {
                    "llm_client": "CONNECTED" if self.client else "MISSING",
                    "clarifier": "READY" if self.clarifier else "MISSING",
                    "builder": "READY" if self.builder else "MISSING"
                },
                "limits": {
                    "max_intention_size": self.MAX_INTENTION_SIZE,
                    "max_questions": self.MAX_Q_COUNT,
                    "max_recursion": self.MAX_RECURSION_DEPTH,
                    "step_timeout": self.STEP_TIMEOUT
                }
            }
        except Exception as e:
            return {"status": "DEGRADED", "error": str(e)}

if __name__ == "__main__":
    # Robust Smoke Test
    async def main():
        try:
            engine = SystemsEngine()
            ctx = await engine.run_pipeline("Implement a multi-threaded web scraper in Rust.")
            print(f"Status: {ctx.state.name}")
            if ctx.state == EngineState.COMPLETED:
                print(f"Success! Prompt Length: {len(ctx.final_prompt)}")
            else:
                print(f"Pipeline failed: {ctx.error_trace[-1]['message']}")
        except Exception as e:
            print(f"Test crashed: {e}")

    asyncio.run(main())