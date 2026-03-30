"""Provisioning Agent for IBM Storage Scale fileset operations."""

import logging
from pathlib import Path
from typing import List

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.utils.common import (
    create_langchain_tool_no_confirmation_simple,
    create_langchain_tool_with_confirmation_simple,
    create_mcp_client,
    load_agent_config,
    setup_agent_logging,
)
from src.utils.constants import (
    PROVISIONING_AGENT_SYSTEM_PROMPT,
    PROVISIONING_ALLOWED_TOOLS,
    PROVISIONING_CONFIRMATION_REQUIRED_TOOLS,
)

logger = logging.getLogger(__name__)


class ProvisioningAgent:
    """Agent for managing IBM Storage Scale filesets and snapshots."""

    def __init__(self, config_path: str = "config/agents_settings.ini"):
        self.config = load_agent_config(Path(config_path))
        setup_agent_logging(self.config, "provisioning_log_path", "logs/provisioning_agent.log")

        llm_config = self.config["llm"]
        mcp_config = self.config["mcp"]

        model_name = llm_config["model_name"].replace("ollama_chat/", "")
        self.llm = ChatOllama(model=model_name, temperature=0, verbose=True, timeout=60.0)
        self.mcp_client = create_mcp_client(mcp_config)

        self.agent_executor = None
        self.tools: List = []

    async def initialize(self):
        """Initialize the agent and connect to MCP server."""
        logger.info(f"Allowed tools: {', '.join(PROVISIONING_ALLOWED_TOOLS)}")

        await self.mcp_client._ensure_session()
        logger.info("MCP server connected")

        logger.info("Configuring tools")
        for tool_name in PROVISIONING_ALLOWED_TOOLS:
            if tool_name in PROVISIONING_CONFIRMATION_REQUIRED_TOOLS:
                tool = create_langchain_tool_with_confirmation_simple(tool_name, self.mcp_client)
                logger.debug(f"Configured tool: {tool_name} (with confirmation)")
            else:
                tool = create_langchain_tool_no_confirmation_simple(tool_name, self.mcp_client)
                logger.debug(f"Configured tool: {tool_name}")

            self.tools.append(tool)

        memory = MemorySaver()
        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=memory,
        )

        self.system_prompt = PROVISIONING_AGENT_SYSTEM_PROMPT
        logger.info("Agent initialized")

    async def cleanup(self):
        """Cleanup resources."""
        if self.mcp_client:
            await self.mcp_client.cleanup()
            logger.info("Cleaned up MCP client")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
