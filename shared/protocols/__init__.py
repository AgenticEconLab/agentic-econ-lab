# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
AEL Protocols — A2A + MCP inter-agent communication and discovery.

Provides:
- TeamCard: Agent capability descriptors for pipeline coordination
- A2ATeamServer/A2AClient: A2A protocol compliance (v0.3)
- AELMCPServer/MCPClient: MCP tool exposure and consumption
- MessageBus: In-process pub/sub for inter-team messaging
"""

from shared.protocols.agent_card import TeamCard, get_team_card, list_team_cards

__all__ = [
    "TeamCard",
    "get_team_card",
    "list_team_cards",
]
