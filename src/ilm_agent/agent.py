"""ILM (Information Lifecycle Management) Agent for IBM Storage Scale policy operations."""

import logging
from pathlib import Path
from typing import List

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from src.ilm_agent.workflow_graph import create_ilm_workflow_graph, create_initial_state
from src.utils.common import (
    create_langchain_tool_no_confirmation_simple,
    create_langchain_tool_with_confirmation_simple,
    create_mcp_client,
    load_agent_config,
    setup_logging,
)
from src.utils.constants import (
    ILM_AGENT_SYSTEM_PROMPT,
    ILM_ALLOWED_TOOLS,
    ILM_CONFIRMATION_REQUIRED_TOOLS,
)

logger = logging.getLogger(__name__)


class ILMAgent:
    """Agent for managing IBM Storage Scale ILM policies."""

    def __init__(self, config_path: str = "config/agents_settings.ini"):
        self.config = load_agent_config(Path(config_path))

        # Setup logging from config — use ilm_file_path if present, else fall back to file_path
        logging_config = self.config["logging"] if "logging" in self.config else {}
        setup_logging(
            log_level=logging_config.get("level", "INFO"),
            log_file=logging_config.get("ilm_file_path", logging_config.get("file_path", "logs/ilm_agent.log")),
            log_format=logging_config.get("format", "json"),
            max_bytes=int(logging_config.get("max_bytes", "10485760")),
            backup_count=int(logging_config.get("backup_count", "5")),
        )

        llm_config = self.config["llm"]
        mcp_config = self.config["mcp"]

        model_name = llm_config["model_name"].replace("ollama_chat/", "")
        self.llm = ChatOllama(model=model_name, temperature=0, verbose=True, timeout=60.0)
        self.mcp_client = create_mcp_client(mcp_config)

        self.agent_executor = None
        self.tools: List = []

    async def initialize(self):
        """Initialize the agent and connect to MCP server."""
        logger.info(f"Allowed tools: {', '.join(ILM_ALLOWED_TOOLS)}")

        await self.mcp_client._ensure_session()
        logger.info("MCP server connected")

        logger.info("Configuring tools")
        for tool_name in ILM_ALLOWED_TOOLS:
            if tool_name in ILM_CONFIRMATION_REQUIRED_TOOLS:
                tool = create_langchain_tool_with_confirmation_simple(tool_name, self.mcp_client)
                logger.debug(f"Configured tool: {tool_name} (with confirmation)")
            else:
                tool = create_langchain_tool_no_confirmation_simple(tool_name, self.mcp_client)
                logger.debug(f"Configured tool: {tool_name}")

            self.tools.append(tool)

        memory = MemorySaver()
        
        # Create custom workflow graph with enforced sequencing
        self.agent_executor = create_ilm_workflow_graph(self.llm, self.tools)
        self.agent_executor = self.agent_executor.compile(checkpointer=memory)

        self.system_prompt = ILM_AGENT_SYSTEM_PROMPT
        logger.info("ILM Agent initialized with custom workflow graph")

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

# Made with Bob
