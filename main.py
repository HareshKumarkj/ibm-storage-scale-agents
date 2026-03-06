"""Main entry point for IBM Storage Scale Agents (Provisioning + ILM)."""

import asyncio
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from src.provisioning_agent.agent import ProvisioningAgent
from src.ilm_agent.agent import ILMAgent

logger = logging.getLogger(__name__)


async def route_to_agent(user_input: str) -> str:
    """Use keyword-based routing to determine which agent should handle the request.
    
    Returns:
        "provisioning" or "ilm"
    """
    user_input_lower = user_input.lower()
    
    # ILM Agent keywords (check these first as they're more specific)
    ilm_keywords = [
        'policy', 'policies', 'migrate', 'migration', 'delete files',
        'old files', 'archive', 'lifecycle', 'ilm', 'pool',
        'days old', 'older than', 'not accessed', 'age-based'
    ]
    
    # Provisioning Agent keywords
    provisioning_keywords = [
        'fileset', 'filesets', 'snapshot', 'snapshots',
        'link', 'unlink', 'junction', 'independent', 'dependent'
    ]
    
    # Check for ILM keywords first (more specific)
    for keyword in ilm_keywords:
        if keyword in user_input_lower:
            logger.info(f"Routing to ILM agent (matched keyword: '{keyword}')")
            return "ilm"
    
    # Check for provisioning keywords
    for keyword in provisioning_keywords:
        if keyword in user_input_lower:
            logger.info(f"Routing to Provisioning agent (matched keyword: '{keyword}')")
            return "provisioning"
    
    # Default to provisioning if no clear match
    logger.info("No clear keyword match. Defaulting to Provisioning agent.")
    return "provisioning"


def get_interrupt_value(interrupt_list):
    """Extract interrupt value from interrupt list."""
    if interrupt_list and len(interrupt_list) > 0:
        interrupt_obj = interrupt_list[0]
        return interrupt_obj.value if hasattr(interrupt_obj, "value") else {}
    return {}


def get_user_approval(tool_name, arguments):
    """Display confirmation request and get user approval."""
    print(f"\n{'=' * 70}")
    print(f"⚠️  CONFIRMATION REQUIRED: {tool_name}")
    print(f"{'=' * 70}")
    print("Arguments:")
    print(json.dumps(arguments, indent=2))
    print(f"{'=' * 70}")

    while True:
        approval_input = input("Approve? (yes/no): ").strip().lower()
        if approval_input in ["yes", "y"]:
            return True
        elif approval_input in ["no", "n"]:
            return False
        print("Please enter 'yes' or 'no'")


async def handle_interrupt(agent, event, config):
    """Handle interrupt and resume execution based on user approval."""
    interrupt_value = get_interrupt_value(event["__interrupt__"])
    tool_name = interrupt_value.get("tool_name", "unknown")
    arguments = interrupt_value.get("arguments", {})

    approved = get_user_approval(tool_name, arguments)

    if approved:
        print(f"✓ Approved. Executing {tool_name}...\n")
    else:
        print(f"✗ Cancelled. Operation {tool_name} not executed.\n")

    final_event = None
    async for resume_event in agent.agent_executor.astream(
        Command(resume={"approved": approved}), config=config, stream_mode="values"
    ):
        final_event = resume_event

    return final_event


async def run_agent(agent, user_input: str, config: dict, first_turn: bool = False):
    """Run a single agent turn and handle any interrupts.

    On the first turn of a session the system prompt is injected so the
    MemorySaver checkpointer stores it as the conversation root.
    On subsequent turns only the new HumanMessage is appended — the
    checkpointer already holds the full history.
    """
    interrupted = False
    event = None

    if first_turn:
        messages = [SystemMessage(content=agent.system_prompt), HumanMessage(content=user_input)]
    else:
        messages = [HumanMessage(content=user_input)]

    # For ILM agent, initialize with complete workflow state on first turn
    if first_turn and type(agent).__name__ == 'ILMAgent':
        from src.ilm_agent.workflow_graph import create_initial_state
        initial_state = create_initial_state()
        # Replace messages with the new ones (type: ignore to handle TypedDict strictness)
        initial_state["messages"] = messages  # type: ignore
        input_state = initial_state
    else:
        input_state = {"messages": messages}

    async for event in agent.agent_executor.astream(
        input_state,  # type: ignore
        config=config,
        stream_mode="values",
    ):
        # Display tool calls in real-time
        if event and "messages" in event:
            messages_in_event = event.get("messages", [])
            if messages_in_event:
                last_msg = messages_in_event[-1]
                msg_type = type(last_msg).__name__
                
                # Show tool calls immediately when agent decides to call a tool
                if msg_type == "AIMessage" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        print(f"\n[Agent calling tool: {tool_call.get('name', 'unknown')}]")
        
        if "__interrupt__" in event:
            interrupted = True
            event = await handle_interrupt(agent, event, config)
            # Show final response after interrupt handling
            if event and "messages" in event:
                messages_in_event = event.get("messages", [])
                for msg in messages_in_event:
                    if type(msg).__name__ == "AIMessage" and hasattr(msg, "content") and msg.content:
                        print(f"\nAgent: {msg.content}\n")
                        break

    # Show final response if not interrupted
    if not interrupted and event and "messages" in event:
        messages_in_event = event.get("messages", [])
        for msg in reversed(messages_in_event):
            if type(msg).__name__ == "AIMessage" and hasattr(msg, "content") and msg.content:
                print(f"\nAgent: {msg.content}\n")
                break


async def main():
    """Run interactive session with automatic agent routing."""
    print("IBM Storage Scale Intelligent Agent")
    print("=" * 70)
    print("Automatic routing to:")
    print("  • Provisioning Agent — fileset and snapshot management")
    print("  • ILM Agent          — storage policy management")
    print("=" * 70)

    async with ProvisioningAgent() as provisioning_agent, ILMAgent() as ilm_agent:
        print("Agents ready. Type 'quit' to exit.\n")

        provisioning_config = {"configurable": {"thread_id": "provisioning-main"}, "recursion_limit": 50}
        ilm_config = {"configurable": {"thread_id": "ilm-main"}, "recursion_limit": 25}
        
        # Track which agent is currently active for conversation continuity
        current_agent = None
        current_config = None
        current_label = None
        first_turn = True

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if not user_input:
                    continue

                # Route the request to appropriate agent
                route = await route_to_agent(user_input)
                
                # Determine which agent to use
                if route == "ilm":
                    selected_agent = ilm_agent
                    selected_config = ilm_config
                    selected_label = "ILM Agent"
                else:  # provisioning
                    selected_agent = provisioning_agent
                    selected_config = provisioning_config
                    selected_label = "Provisioning Agent"
                
                # If agent changed, reset first_turn
                if current_agent != selected_agent:
                    if current_agent is not None:
                        print(f"\n[Switching to {selected_label}]\n")
                    else:
                        print(f"[Routing to {selected_label}]\n")
                    first_turn = True
                    current_agent = selected_agent
                    current_config = selected_config
                    current_label = selected_label
                
                # Run the selected agent
                await run_agent(current_agent, user_input, current_config, first_turn=first_turn)
                first_turn = False

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")
                logger.exception("Error in main loop")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

# Made with Bob
