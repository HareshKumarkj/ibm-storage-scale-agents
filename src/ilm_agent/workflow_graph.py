"""Custom LangGraph workflow for ILM policy operations with enforced sequencing."""

import json
import logging
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)


class ILMWorkflowState(TypedDict):
    """State for ILM workflow with enforced sequencing."""
    
    # Standard message history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Workflow tracking
    workflow_step: str  # current step: "initial", "get_policy", "verify_pools", "test_policy", "update_policy", "apply_policy", "completed"
    filesystem: Optional[str]
    
    # Data collected during workflow
    existing_policy: Optional[Dict[str, Any]]
    existing_policy_text: Optional[str]
    storage_pools: Optional[List[str]]
    
    # Validation flags
    policy_retrieved: bool
    pools_verified: bool
    policy_tested: bool
    test_passed: bool
    policy_updated: bool
    
    # Error tracking
    error_message: Optional[str]


def create_initial_state() -> ILMWorkflowState:
    """Create initial workflow state."""
    return ILMWorkflowState(
        messages=[],
        workflow_step="initial",
        filesystem=None,
        existing_policy=None,
        existing_policy_text=None,
        storage_pools=None,
        policy_retrieved=False,
        pools_verified=False,
        policy_tested=False,
        test_passed=False,
        policy_updated=False,
        error_message=None,
    )


def _get_workflow_guidance(state: ILMWorkflowState) -> str:
    """Generate workflow guidance based on current state, including error context."""
    error = state.get("error_message")
    
    # If there's an error, provide context about it
    if error:
        return (
            f"Previous operation encountered an error: {error}\n"
            "You can:\n"
            "1. Retry the operation with corrected parameters\n"
            "2. Try a different approach\n"
            "3. Ask the user for clarification"
        )
    
    # Workflow step guidance mapping
    step = state["workflow_step"]
    guidance_map = {
        "initial": (
            "Starting ILM policy workflow. First step: retrieve existing policy with get_policy "
            "to ensure all existing rules are preserved."
        ),
        "get_policy": (
            "Waiting for get_policy to be called. This is REQUIRED to retrieve existing policy rules "
            "before making any changes."
        ) if not state["policy_retrieved"] else "",
        "verify_pools": (
            f"Policy retrieved for filesystem '{state['filesystem']}'. "
            "Next step: verify storage pools with list_storage_pools to ensure target pool exists."
        ) if not state["pools_verified"] else "",
        "test_policy": (
            "Storage pools verified. Next step: test the new policy with test_policy. "
            "IMPORTANT: Include ALL existing rules plus the new rule in policy_contents."
        ) if not state["policy_tested"] else "",
        "update_policy": (
            "Policy test passed. Ready to update with update_policy. "
            "CRITICAL: Include the COMPLETE policy with ALL rules (existing + new)."
        ) if state["test_passed"] and not state["policy_updated"] else "",
        "apply_policy": (
            "Policy updated successfully. Final step: apply the policy with apply_policy "
            "to execute the policy rules on the filesystem."
        ) if state["policy_updated"] else "",
    }
    
    return guidance_map.get(step, "")


def _parse_tool_content(tool_content: Any) -> Dict[str, Any]:
    """Parse tool content into a dictionary."""
    try:
        return json.loads(tool_content) if isinstance(tool_content, str) else tool_content
    except json.JSONDecodeError:
        return {"text": tool_content}


def _check_tool_error(tool_result: Dict[str, Any], tool_name: str) -> tuple[bool, Optional[str]]:
    """Check if tool result contains an error.
    
    Detects errors from multiple sources by checking for common error keywords
    in various fields of the response.
    """
    if not isinstance(tool_result, dict):
        return False, None
    
    # Common error keywords to check for
    error_keywords = ["error", "failed", "failure", "exception", "no such file"]
    
    # Check isError flag first (explicit error indicator)
    if tool_result.get("isError") is True:
        error_message = tool_result.get("message") or tool_result.get("text") or f"Error in {tool_name}"
        logger.warning(f"Tool {tool_name} returned isError=true: {error_message}")
        return True, str(error_message)
    
    # Check all text fields for error keywords
    fields_to_check = ["status", "text", "result", "message", "output"]
    
    for field in fields_to_check:
        field_value = str(tool_result.get(field, "")).lower()
        if field_value and any(keyword in field_value for keyword in error_keywords):
            error_message = tool_result.get(field, f"Error in {tool_name}")
            logger.warning(f"Tool {tool_name} returned error in {field}: {str(error_message)[:200]}")
            return True, str(error_message)
    
    return False, None


def _check_success_status(tool_result: Any) -> bool:
    """Check if tool result indicates success.
    
    Success is determined by the absence of error indicators.
    Uses the same error detection logic as _check_tool_error.
    """
    if isinstance(tool_result, dict):
        # Use the error detection function - if no error found, it's success
        is_error, _ = _check_tool_error(tool_result, "check_success")
        if is_error:
            return False
        
        # If no error detected, consider it success
        return True
    
    # For non-dict results, check string representation
    result_str = str(tool_result).lower()
    error_keywords = ["error", "failed", "failure", "exception", "no such file"]
    
    # If any error keyword found, it's not success
    if any(keyword in result_str for keyword in error_keywords):
        return False
    
    # No error indicators found = success
    return True


def _extract_filesystem_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    """Extract filesystem parameter from tool calls in messages."""
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "get_policy":
                    return tc.get("args", {}).get("filesystem")
    return None


def _process_tool_results(state: ILMWorkflowState, result: Dict[str, Any]) -> Dict[str, Any]:
    """Process tool execution results and update workflow state.
    
    Handles both successful results and errors, allowing the agent to see errors
    and decide how to recover (retry, adjust parameters, ask user, etc.).
    """
    updates = {"messages": result.get("messages", [])}
    
    # Extract the last tool message
    messages = result.get("messages", [])
    if not messages or not isinstance(messages[-1], ToolMessage):
        return updates
    
    last_message = messages[-1]
    tool_name = last_message.name
    tool_result = _parse_tool_content(last_message.content)
    
    logger.info(f"Processing tool result for: {tool_name}")
    
    # Check for errors in tool result
    is_error, error_message = _check_tool_error(tool_result, tool_name)
    
    if is_error:
        updates["error_message"] = error_message
        logger.info(f"Workflow staying at step '{state['workflow_step']}' due to error - agent can retry")
        return updates
    
    # Clear any previous error on success
    if state.get("error_message"):
        logger.info("Clearing previous error - operation succeeded")
        updates["error_message"] = None
    
    # Tool-specific processing
    if tool_name == "get_policy":
        updates.update({
            "policy_retrieved": True,
            "existing_policy": tool_result,
            "workflow_step": "verify_pools",
            "filesystem": _extract_filesystem_from_messages(state["messages"]),
        })
        
        if isinstance(tool_result, dict) and "policy_contents" in tool_result:
            updates["existing_policy_text"] = tool_result["policy_contents"]
        
        logger.info(f"Policy retrieved for filesystem: {updates.get('filesystem')}")
    
    elif tool_name == "list_storage_pools":
        # Only mark as verified if we actually got pool data (not an error response)
        if isinstance(tool_result, dict) and "pools" in tool_result:
            pools = tool_result["pools"]
            if isinstance(pools, list):
                pool_names = [p.get("poolName") for p in pools if isinstance(p, dict) and p.get("poolName")]
                updates["storage_pools"] = pool_names
                updates["pools_verified"] = True
                updates["workflow_step"] = "test_policy"
                logger.info(f"Storage pools verified: {pool_names}")
            else:
                logger.warning("Pools field is not a list - may need retry")
        else:
            # No pools data - this might be an error response treated as success
            # Don't advance workflow, let agent retry
            logger.warning("No pools data in response - workflow staying at verify_pools step")
    
    elif tool_name == "test_policy":
        test_passed = _check_success_status(tool_result)
        updates["policy_tested"] = True
        updates["test_passed"] = test_passed
        
        if test_passed:
            updates["workflow_step"] = "update_policy"
            logger.info("Policy test PASSED - ready for update")
        else:
            updates["error_message"] = "Policy test failed - please fix errors before updating"
            logger.warning("Policy test FAILED")
    
    elif tool_name == "update_policy" and _check_success_status(tool_result):
        updates.update({
            "policy_updated": True,
            "workflow_step": "apply_policy",
        })
        logger.info("Policy updated successfully - ready to apply")
    
    elif tool_name == "apply_policy" and _check_success_status(tool_result):
        updates["workflow_step"] = "completed"
        logger.info("Policy applied successfully - workflow complete")
    
    return updates


def should_continue(state: ILMWorkflowState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise end
    return "end"


def _get_validation_error(tool_name: str, state: ILMWorkflowState) -> Optional[str]:
    """Get validation error message for a tool call based on workflow state."""
    validation_rules = {
        "test_policy": (
            not state["policy_retrieved"],
            "Cannot test policy without first retrieving existing policy. Please call get_policy first."
        ),
        "update_policy": [
            (not state["policy_retrieved"],
             "Cannot update policy without first retrieving existing policy. CRITICAL: Call get_policy first to preserve all existing rules."),
            (not state["policy_tested"],
             "Cannot update policy without testing it first. Please call test_policy to validate the policy."),
            (not state["test_passed"],
             "Cannot update policy because test_policy did not pass. Please fix the policy errors before updating."),
        ],
        "apply_policy": (
            not state["policy_updated"],
            "Cannot apply policy without updating it first. Please call update_policy to save the policy before applying."
        ),
    }
    
    rules = validation_rules.get(tool_name)
    if not rules:
        return None
    
    # Handle single rule (tuple) or multiple rules (list)
    if isinstance(rules, tuple):
        condition, message = rules
        return message if condition else None
    
    # Multiple rules - return first matching error
    for condition, message in rules:
        if condition:
            return message
    
    return None


def validate_tool_call(state: ILMWorkflowState) -> Dict[str, Any]:
    """Validate that tool calls follow the workflow sequence."""
    messages = state["messages"]
    if not messages:
        return {}
    
    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}
    
    # Check each tool call
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        error = _get_validation_error(tool_name, state)
        
        if error:
            error_msg = ToolMessage(
                content=json.dumps({"status": "error", "message": error}),
                tool_call_id=tool_call.get("id", "error"),
                name=tool_name,
            )
            return {"messages": [error_msg], "error_message": error}
    
    return {}


def create_ilm_workflow_graph(llm, tools):
    """Create the ILM workflow graph with enforced sequencing.
    
    Args:
        llm: Language model instance
        tools: List of available tools
        
    Returns:
        StateGraph ready to be compiled
    """
    # Create closures that capture llm and tools
    async def agent_node_with_context(state: ILMWorkflowState) -> Dict[str, Any]:
        """Agent node with captured LLM and tools."""
        # Build context-aware system message based on workflow state
        workflow_guidance = _get_workflow_guidance(state)
        
        # Create enhanced messages with workflow context
        messages = state["messages"].copy()
        
        # Add workflow guidance as a HumanMessage to prompt action
        if state["workflow_step"] != "initial" and workflow_guidance:
            # Insert guidance as a user instruction to prompt the agent to act
            guidance_msg = HumanMessage(content=f"[Workflow Status]\n{workflow_guidance}\n\nPlease proceed with the next step.")
            messages.append(guidance_msg)
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        # Get LLM response
        response = await llm_with_tools.ainvoke(messages)
        
        return {"messages": [response]}
    
    async def tool_execution_node_with_context(state: ILMWorkflowState) -> Dict[str, Any]:
        """Tool execution node with captured tools."""
        tool_node = ToolNode(tools)
        
        # Execute the tool asynchronously
        result = await tool_node.ainvoke(state)
        
        # Update workflow state based on tool execution
        updates = _process_tool_results(state, result)
        
        return updates
    
    # Create the graph
    workflow = StateGraph(ILMWorkflowState)
    
    # Add nodes with context
    workflow.add_node("agent", agent_node_with_context)
    workflow.add_node("tools", tool_execution_node_with_context)
    workflow.add_node("validate", validate_tool_call)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "validate",
            "end": END,
        },
    )
    
    # After validation, go to tools or back to agent
    workflow.add_edge("validate", "tools")
    workflow.add_edge("tools", "agent")
    
    return workflow

# Made with Bob
