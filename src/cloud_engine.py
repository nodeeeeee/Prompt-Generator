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
from src.features.prompt_optimizer import PromptOptimizer
from src.security_engine import SecurityEngine, SecurityState

# Enhanced Logging Configuration for Cloud operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CloudEngine")

class CloudState(enum.Enum):
    """
    Formalized states for the CloudEngine lifecycle.
    """
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    CONTENT_SECURITY_SCAN = "CONTENT_SECURITY_SCAN"
    VALIDATING = "VALIDATING"
    CLARIFYING = "CLARIFYING"
    ARCHITECTING = "ARCHITECTING" # Cloud-specific architecture design
    SECURITY_CHECK = "SECURITY_CHECK" # Cloud-specific security validation
    COST_OPTIMIZATION = "COST_OPTIMIZATION" # Financial/Resource optimization
    BUILDING = "BUILDING"
    OPTIMIZING = "OPTIMIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    HALTED = "HALTED"

# Strict Transition Map for the State Machine
VALID_TRANSITIONS: Dict[CloudState, Set[CloudState]] = {
    CloudState.IDLE: {CloudState.INITIALIZING, CloudState.FAILED},
    CloudState.INITIALIZING: {CloudState.CONTENT_SECURITY_SCAN, CloudState.FAILED},
    CloudState.CONTENT_SECURITY_SCAN: {CloudState.VALIDATING, CloudState.FAILED, CloudState.HALTED},
    CloudState.VALIDATING: {CloudState.CLARIFYING, CloudState.ARCHITECTING, CloudState.FAILED},
    CloudState.CLARIFYING: {CloudState.ARCHITECTING, CloudState.FAILED, CloudState.HALTED},
    CloudState.ARCHITECTING: {CloudState.SECURITY_CHECK, CloudState.FAILED, CloudState.HALTED},
    CloudState.SECURITY_CHECK: {CloudState.COST_OPTIMIZATION, CloudState.BUILDING, CloudState.FAILED, CloudState.HALTED},
    CloudState.COST_OPTIMIZATION: {CloudState.BUILDING, CloudState.FAILED, CloudState.HALTED},
    CloudState.BUILDING: {CloudState.OPTIMIZING, CloudState.COMPLETED, CloudState.FAILED, CloudState.HALTED},
    CloudState.OPTIMIZING: {CloudState.COMPLETED, CloudState.FAILED},
    CloudState.COMPLETED: {CloudState.IDLE, CloudState.INITIALIZING},
    CloudState.FAILED: {CloudState.IDLE, CloudState.INITIALIZING},
    CloudState.HALTED: {CloudState.IDLE, CloudState.INITIALIZING}
}

class CloudEngineError(Exception):
    """Base exception for all CloudEngine related errors."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        try:
            super().__init__(message)
            self.context = context or {}
        except Exception:
            pass

class CloudValidationError(CloudEngineError):
    """Raised when cloud-specific input/output validation fails."""
    pass

class CloudStateError(CloudEngineError):
    """Raised on illegal state transitions in CloudEngine."""
    pass

class CloudComponentError(CloudEngineError):
    """Raised when an internal cloud component fails."""
    pass

class CloudBoundaryError(CloudEngineError):
    """Raised when safety limits for cloud prompts are breached."""
    pass

class CloudExecutionTimeoutError(CloudEngineError):
    """Raised when a cloud pipeline step exceeds its time budget."""
    pass

class CloudPerformanceMetrics(BaseModel):
    """
    Captures granular performance data for cloud-native optimizations.
    """
    initialization_ms: float = 0.0
    validation_ms: float = 0.0
    clarification_ms: float = 0.0
    architecting_ms: float = 0.0
    security_check_ms: float = 0.0
    cost_optimization_ms: float = 0.0
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
        return datetime.min

class CloudContext(BaseModel):
    """
    Pydantic-powered context for Cloud-native tasks.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intention: str
    model: str = "o3-mini"
    mode: str = "iterative"
    cloud_provider: str = "agnostic" # aws, azure, gcp, etc.
    questions: List[str] = []
    answers: List[str] = []
    final_prompt: Optional[str] = None
    state: CloudState = CloudState.IDLE
    metrics: CloudPerformanceMetrics = Field(default_factory=CloudPerformanceMetrics)
    error_trace: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    start_time: datetime = Field(default_factory=get_now)
    end_time: Optional[datetime] = None

    @field_validator('intention')
    @classmethod
    def validate_intention_non_empty(cls, v):
        try:
            if not v or not v.strip():
                raise ValueError("Cloud intention must not be empty.")
            if len(v) < 15:
                raise ValueError("Cloud intention is too brief for architectural analysis.")
            if len(v) > 100000:
                 raise ValueError("Cloud intention size exceeds limit.")
            return v
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Unexpected validation error: {e}")

class CloudEngine:
    """
    High-performance, Cloud-specialized module with strict robustness.
    
    Adheres to:
    1. STRICT ERROR HANDLING: Comprehensive try-except coverage.
    2. CLOUD-NATIVE VALIDATION: Checks for infrastructure-specific constraints.
    3. BOUNDARY MANAGEMENT: Enforces strict limits on resource-intensive prompts.
    4. STATE INTEGRITY: Uses a verified state machine.
    """

    MAX_INTENTION_SIZE = 60000
    MAX_Q_COUNT = 12
    MAX_RECURSION_DEPTH = 3
    STEP_TIMEOUT = 120.0

    def __init__(self, llm_client: Optional[LLMClient] = None):
        try:
            self.client = llm_client or LLMClient()
            self.clarifier = ClarificationAgent(self.client)
            self.builder = PromptBuilder(self.client)
            self.optimizer = PromptOptimizer(self.client)
            self.security = SecurityEngine()
            self._lock = asyncio.Lock()
            logger.info("CloudEngine initialized for high-performance operations with SecurityEngine.")
        except Exception as e:
            logger.critical(f"CloudEngine Bootstrap Failed: {e}")
            raise CloudEngineError("Failed to initialize CloudEngine components.", {"original_error": str(e)})

    def _update_state(self, context: CloudContext, target_state: CloudState):
        try:
            if target_state not in VALID_TRANSITIONS.get(context.state, set()):
                error_msg = f"Illegal cloud state transition: {context.state} -> {target_state}"
                logger.error(f"[{context.request_id}] {error_msg}")
                raise CloudStateError(error_msg, {"current_state": context.state, "target_state": target_state})
            
            logger.debug(f"[{context.request_id}] Cloud State: {context.state.name} -> {target_state.name}")
            context.state = target_state
        except CloudStateError:
            raise
        except Exception as e:
            raise CloudStateError(f"Cloud state management failure: {e}")

    async def run_pipeline(
        self, 
        intention: str, 
        model: str = "o3-mini", 
        mode: str = "iterative",
        cloud_provider: str = "agnostic",
        answers: Optional[List[str]] = None,
        depth: int = 0
    ) -> CloudContext:
        """
        Main execution entry point for the cloud-native prompt generation.
        Implements top-level error handling, boundary checks, and recursion safety.
        """
        t_start = time.perf_counter()
        context = None
        
        try:
            # 0. Boundary Check: Recursion
            if depth > self.MAX_RECURSION_DEPTH:
                raise CloudBoundaryError(f"Maximum cloud recursion depth {self.MAX_RECURSION_DEPTH} exceeded.")

            # 1. Input Validation: Type checks
            if not isinstance(intention, str):
                raise CloudValidationError("Cloud intention must be a string.")
            
            # 2. Context Creation
            try:
                context = CloudContext(
                    intention=intention, 
                    model=model, 
                    mode=mode, 
                    cloud_provider=cloud_provider,
                    answers=answers or []
                )
            except PydanticValidationError as e:
                logger.error(f"Cloud Context Validation Failed: {e}")
                fail_ctx = CloudContext.model_construct(
                    intention=str(intention)[:100], 
                    cloud_provider=str(cloud_provider),
                    state=CloudState.FAILED
                )
                self._handle_failure(fail_ctx, CloudValidationError(f"Data model validation failed: {str(e)}"))
                return fail_ctx

            # 3. Initialization Phase
            self._update_state(context, CloudState.INITIALIZING)
            init_start = time.perf_counter()
            self._validate_cloud_config(context)
            context.metrics.initialization_ms = (time.perf_counter() - init_start) * 1000

            # 4. Content Security Scan Phase
            self._update_state(context, CloudState.CONTENT_SECURITY_SCAN)
            sec_start_time = time.perf_counter()
            sec_context = await self.security.process_content(context.intention)
            if not sec_context or sec_context.state == SecurityState.FAILED:
                reason = sec_context.error_trace[-1]["message"] if sec_context else "Security engine failure"
                raise CloudValidationError(f"Content security check failed: {reason}")
            
            # Update intention with sanitized version
            context.intention = sec_context.sanitized_content
            context.metadata["content_security_metrics"] = sec_context.metrics.model_dump()

            # 5. Deep Validation Phase
            self._update_state(context, CloudState.VALIDATING)
            val_start = time.perf_counter()
            self._perform_cloud_validation(context)
            context.metrics.validation_ms = (time.perf_counter() - val_start) * 1000

            # 5. Clarification Phase (if needed)
            self._update_state(context, CloudState.CLARIFYING)
            if not context.answers:
                clar_start = time.perf_counter()
                await asyncio.wait_for(self._run_clarification(context), timeout=self.STEP_TIMEOUT)
                context.metrics.clarification_ms = (time.perf_counter() - clar_start) * 1000

            # 6. Architecting Phase
            self._update_state(context, CloudState.ARCHITECTING)
            arch_start = time.perf_counter()
            await asyncio.wait_for(self._run_architecting(context), timeout=self.STEP_TIMEOUT)
            context.metrics.architecting_ms = (time.perf_counter() - arch_start) * 1000

            # 7. Security Check Phase
            self._update_state(context, CloudState.SECURITY_CHECK)
            sec_start = time.perf_counter()
            await asyncio.wait_for(self._run_security_check(context), timeout=self.STEP_TIMEOUT)
            context.metrics.security_check_ms = (time.perf_counter() - sec_start) * 1000

            # 8. Cost Optimization Phase (Conditional)
            if context.metadata.get("financial_optimization_requested"):
                self._update_state(context, CloudState.COST_OPTIMIZATION)
                cost_start = time.perf_counter()
                await asyncio.wait_for(self._run_cost_optimization(context), timeout=self.STEP_TIMEOUT)
                context.metrics.cost_optimization_ms = (time.perf_counter() - cost_start) * 1000

            # 9. Building Phase
            self._update_state(context, CloudState.BUILDING)
            build_start = time.perf_counter()
            await asyncio.wait_for(self._run_building(context), timeout=self.STEP_TIMEOUT)
            context.metrics.building_ms = (time.perf_counter() - build_start) * 1000

            # 10. Optimization Phase
            self._update_state(context, CloudState.OPTIMIZING)
            opt_start = time.perf_counter()
            if context.final_prompt:
                try:
                    context.final_prompt = await self.optimizer.optimize_prompt(
                        raw_prompt=context.final_prompt,
                        mode=context.mode,
                        intention=context.intention
                    )
                except Exception as e:
                    logger.warning(f"[{context.request_id}] Cloud optimization soft-failed: {e}")
            
            context.metrics.optimizing_ms = (time.perf_counter() - opt_start) * 1000

            # Finalize
            self._update_state(context, CloudState.COMPLETED)
            context.end_time = get_now()
            
        except asyncio.TimeoutError:
            if context:
                self._handle_failure(context, CloudExecutionTimeoutError(f"Cloud pipeline step timed out after {self.STEP_TIMEOUT}s"))
        except (CloudValidationError, CloudStateError, CloudBoundaryError, CloudComponentError) as e:
            if context:
                self._handle_failure(context, e)
            else:
                logger.critical(f"Critical Cloud Failure: {e}")
        except Exception as e:
            logger.exception("Critical Cloud Engine Error")
            if context:
                self._handle_failure(context, CloudEngineError(f"Internal cloud crash: {str(e)}"))
        finally:
            if context:
                context.metrics.total_duration_ms = (time.perf_counter() - t_start) * 1000
                logger.info(f"[{context.request_id}] Cloud Pipeline finished in {context.metrics.total_duration_ms:.2f}ms")
            
        return context

    def _validate_cloud_config(self, context: CloudContext):
        try:
            if len(context.intention) > self.MAX_INTENTION_SIZE:
                raise CloudBoundaryError(f"Cloud intention exceeds size limit: {len(context.intention)}")
        except Exception as e:
            if isinstance(e, (CloudValidationError, CloudBoundaryError)): raise
            raise CloudValidationError(f"Configuration error: {e}")

    def _perform_cloud_validation(self, context: CloudContext):
        """Checks for cloud-related context, semantics, and input quality."""
        try:
            words = context.intention.split()
            if not words:
                raise CloudValidationError("Cloud intention contains no words.")
            
            # Check for repetitive garbage or obvious non-instructional input
            unique_words_ratio = len(set(words)) / len(words)
            if unique_words_ratio < 0.1 and len(words) > 20:
                 raise CloudValidationError("Cloud intention appears to be non-instructional or highly repetitive.")

            intent_lower = context.intention.lower()
            if "cost" in intent_lower or "budget" in intent_lower or "cheap" in intent_lower:
                context.metadata["financial_optimization_requested"] = True

            if len(words) > 30000:
                raise CloudBoundaryError("Cloud intention contains too many tokens for reliable processing.")
        except (CloudValidationError, CloudBoundaryError):
            raise
        except Exception as e:
            raise CloudValidationError(f"Semantic validation failed: {e}")

    async def _run_clarification(self, context: CloudContext):
        try:
            cloud_intention = f"[Cloud Domain: {context.cloud_provider}] {context.intention}"
            questions = await self.clarifier.generate_questions(cloud_intention)
            if not isinstance(questions, list):
                raise CloudComponentError("Clarifier returned malformed output (expected list).")
            
            context.questions = questions[:self.MAX_Q_COUNT]
            context.metrics.llm_calls += 1
        except Exception as e:
            logger.warning(f"[{context.request_id}] Cloud clarification soft-failed: {e}. Proceeding with best-effort.")
            context.error_trace.append({
                "step": "clarification", 
                "error": str(e), 
                "severity": "WARNING",
                "timestamp": get_now().isoformat()
            })
            context.questions = []

    async def _run_architecting(self, context: CloudContext):
        """Actual LLM-based architecture analysis."""
        try:
            messages = [
                {"role": "system", "content": "You are an expert Cloud Solutions Architect. Analyze the task and identify the most suitable architecture pattern (e.g., Serverless, Microservices, Monolithic, N-Tier, Event-Driven). Provide a brief 1-sentence rationale."},
                {"role": "user", "content": context.intention}
            ]
            response = await self.client.agenerate_completion(messages, temperature=0.3)
            context.metadata["architecture_analysis"] = response
            context.metrics.llm_calls += 1
        except Exception as e:
            logger.error(f"[{context.request_id}] Architecting phase failure: {e}")
            context.metadata["architecture_analysis"] = "Standard N-Tier (Fallback)"
            context.error_trace.append({
                "step": "architecting", 
                "error": str(e), 
                "severity": "WARNING",
                "timestamp": get_now().isoformat()
            })

    async def _run_security_check(self, context: CloudContext):
        """LLM-based security vulnerability assessment."""
        try:
            messages = [
                {"role": "system", "content": "You are a Cloud Security Engineer. Identify potential security risks or compliance requirements (HIPAA, GDPR, PCI-DSS) for the following cloud intention. Provide key security controls to implement."},
                {"role": "user", "content": context.intention}
            ]
            response = await self.client.agenerate_completion(messages, temperature=0.1)
            context.metadata["security_assessment"] = response
            context.metrics.llm_calls += 1
        except Exception as e:
            logger.error(f"[{context.request_id}] Security check phase failure: {e}")
            context.metadata["security_assessment"] = "Apply standard IAM and Encryption (Fallback)"
            context.error_trace.append({
                "step": "security_check", 
                "error": str(e), 
                "severity": "WARNING",
                "timestamp": get_now().isoformat()
            })

    async def _run_cost_optimization(self, context: CloudContext):
        """LLM-based cost and resource optimization."""
        try:
            messages = [
                {"role": "system", "content": "You are a Cloud FinOps specialist. Suggest specific ways to optimize costs for this cloud infrastructure request (e.g., spot instances, auto-scaling, right-sizing)."},
                {"role": "user", "content": context.intention}
            ]
            response = await self.client.agenerate_completion(messages, temperature=0.2)
            context.metadata["cost_optimization"] = response
            context.metrics.llm_calls += 1
        except Exception as e:
            logger.error(f"[{context.request_id}] Cost optimization phase failure: {e}")
            context.error_trace.append({
                "step": "cost_optimization", 
                "error": str(e), 
                "severity": "WARNING",
                "timestamp": get_now().isoformat()
            })

    async def _run_building(self, context: CloudContext):
        try:
            cloud_meta = f"Target Cloud Provider: {context.cloud_provider}\n"
            cloud_meta += f"Architecture: {context.metadata.get('architecture_analysis', 'N/A')}\n"
            cloud_meta += f"Security: {context.metadata.get('security_assessment', 'N/A')}\n"
            if context.metadata.get("cost_optimization"):
                cloud_meta += f"Cost Opt: {context.metadata.get('cost_optimization')}\n"

            result = await self.builder.build_prompt(
                intention=context.intention,
                answers=context.answers,
                questions=context.questions,
                mode=context.mode,
                project_context=cloud_meta
            )
            
            if not result or not isinstance(result, str):
                raise CloudComponentError("PromptBuilder produced empty or invalid string output for cloud prompt.")
            
            # Boundary Check: Output Size
            if len(result) > 120000: # Slightly larger limit for cloud prompts
                raise CloudBoundaryError("Generated cloud prompt exceeds safety size limit (120KB).")
                
            context.final_prompt = result
            context.metrics.llm_calls += 1
        except (CloudComponentError, CloudBoundaryError):
            raise
        except Exception as e:
            raise CloudComponentError(f"Cloud building phase critical failure: {e}")

    def _handle_failure(self, context: CloudContext, error: Exception):
        """
        Centralized error handling and context enrichment.
        Guaranteed to not raise exceptions itself.
        """
        try:
            context.state = CloudState.FAILED
            err_info = {
                "timestamp": get_now().isoformat(),
                "type": error.__class__.__name__,
                "message": str(error),
                "context": getattr(error, 'context', {})
            }
            context.error_trace.append(err_info)
            logger.error(f"[{context.request_id}] Cloud Pipeline Failure: {err_info['message']}")
        except Exception as e:
            # Last resort logging
            print(f"CRITICAL: Cloud _handle_failure internal failure: {e}", file=sys.stderr)

    def get_health(self) -> Dict[str, Any]:
        """
        CloudEngine health monitoring with explicit error handling.
        """
        try:
            return {
                "status": "OPERATIONAL",
                "module": "CloudEngine",
                "version": "1.2.0-robust",
                "timestamp": get_now().isoformat(),
                "capabilities": ["architecture_analysis", "security_assessment", "cost_optimization"],
                "components": {
                    "llm_client": "CONNECTED" if self.client else "MISSING",
                    "clarifier": "READY" if self.clarifier else "MISSING",
                    "builder": "READY" if self.builder else "MISSING",
                    "optimizer": "READY" if self.optimizer else "MISSING"
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
    async def smoke_test():
        engine = CloudEngine()
        ctx = await engine.run_pipeline("Build a cost-optimized AWS EKS cluster with Fargate and RDS.")
        print(f"Status: {ctx.state.name}, LLM Calls: {ctx.metrics.llm_calls}")

    asyncio.run(smoke_test())