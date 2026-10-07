# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL Reliability — Guards against known multi-agent failure modes.

Provides:
- StateGuard: Hash-based state verification between pipeline stages
- RoleEnforcer: Role boundary validation for agent outputs
- CheckpointManager: Checkpoint/resume for pipeline execution
"""

from shared.reliability.state_guard import StateGuard
from shared.reliability.checkpoint import CheckpointManager
from shared.reliability.role_enforcer import RoleEnforcer

__all__ = ["StateGuard", "CheckpointManager", "RoleEnforcer"]
