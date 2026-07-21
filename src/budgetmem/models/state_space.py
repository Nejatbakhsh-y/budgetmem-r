"""Official-Mamba adapter with CPU-safe diagonal SSM fallback."""

from budgetmem.baselines.controlled import (
    DiagonalSSMBaseline,
    OfficialMambaOrSSMBaseline,
)

MambaBaseline = OfficialMambaOrSSMBaseline

__all__ = ["DiagonalSSMBaseline", "OfficialMambaOrSSMBaseline", "MambaBaseline"]
