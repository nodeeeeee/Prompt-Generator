# CloudEngine Module

## Overview
The CloudEngine is a high-performance module designed for generating complex, cloud-native prompts. It features a robust state machine, Pydantic-based validation, and specialized phases for architecture analysis, security assessment, and cost optimization.

## Key Features
- **Strict Error Handling**: Every phase is wrapped in try-except blocks with centralized failure management.
- **Pydantic Validation**: Uses `CloudContext` for strict type safety and boundary checking.
- **Cloud-Native Phases**:
  - `ARCHITECTING`: Identifies suitable architecture patterns using LLM.
  - `SECURITY_CHECK`: Assesses risks and compliance requirements.
  - `COST_OPTIMIZATION`: Triggered by cost-related keywords to suggest FinOps improvements.
- **Performance Metrics**: Granular tracking of execution time for each phase.

## Usage
```python
from src.cloud_engine import CloudEngine

engine = CloudEngine()
context = await engine.run_pipeline(
    "Deploy a scalable EKS cluster with RDS and S3",
    cloud_provider="aws"
)
if context.state == CloudState.COMPLETED:
    print(context.final_prompt)
```
