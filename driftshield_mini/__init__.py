"""
DriftShield-Mini — Real-time behavioural drift detection for agentic AI systems.

Framework integrations:
    from driftshield_mini import DriftMonitor                    # LangChain / manual
    from driftshield_mini.crewai import DriftCrew                 # CrewAI
    from driftshield_mini.autogen import DriftAutogenAgent        # Microsoft AutoGen
    from driftshield_mini.llama_index import DriftLlamaIndexHandler  # LlamaIndex
    from driftshield_mini.openai_assistants import DriftOpenAIClient # OpenAI Assistants
    from driftshield_mini.semantic_kernel import DriftKernelFilter   # Semantic Kernel
    from driftshield_mini.haystack import DriftHaystackTracer        # Haystack
    from driftshield_mini.google_adk import DriftADKCallbacks        # Google ADK
"""

from driftshield_mini.models import BaselineStats, DetectorType, DriftEvent, Severity, TraceEvent
from driftshield_mini.monitor import DriftMonitor

__version__ = "0.2.2"

__all__ = [
    "DriftMonitor",
    "TraceEvent",
    "DriftEvent",
    "BaselineStats",
    "DetectorType",
    "Severity",
]
