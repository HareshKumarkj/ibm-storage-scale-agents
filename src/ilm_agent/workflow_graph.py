"""Custom LangGraph workflow for ILM policy operations with enforced sequencing."""

import json
import logging
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.ilm_agent.rule_generator import check_rule_redundancy, generate_ilm_rule

logger = logging.getLogger(__name__)


class ILMWorkflowState(TypedDict):
    """State for ILM workflow with enforced sequencing."""
    
    # Standard message history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Workflow tracking
    workflow_step: str  # current step: "initial", "get_policy", "verify_pools", "generate_rule", "test_policy", "update_policy", "apply_policy", "completed", "cancelled"
    filesystem: Optional[str]
    
    # Data collected during workflow
    existing_policy: Optional[Dict[str, Any]]
    existing_policy_text: Optional[str]
    storage_pools: Optional[List[str]]
    
    # Rule generation data
    user_request: Optional[str]  # Original user request for rule generation
    target_pool: Optional[str]  # Target pool for migration
    source_pool: Optional[str]  # Source pool (optional)
    generated_rule: Optional[Dict[str, Any]]  # Generated rule with metadata
    rule_generation_error: Optional[str]  # Error during rule generation

    # Validation flags
    policy_retrieved: bool
    pools_verified: bool
    rule_generated: bool
    policy_tested: bool
    test_passed: bool
    policy_updated: bool
    user_cancelled: bool  # Track if user cancelled an operation
    
    # Error tracking
    error_message: Optional[str]


def create_initial_state() -> ILMWorkflowState:
    """Create initial workflow state."""
    return ILMWorkflowState(
        messages=[],
        workflow_step="get_policy",  # Start with get_policy step to enforce proper sequence
        filesystem=None,
        existing_policy=None,
        existing_policy_text=None,
        storage_pools=None,
        user_request=None,
        target_pool=None,
        source_pool=None,
        generated_rule=None,
        rule_generation_error=None,
        policy_retrieved=False,
        pools_verified=False,
        rule_generated=False,
        policy_tested=False,
        test_passed=False,
        policy_updated=False,
        user_cancelled=False,
        error_message=None,
    )


def _get_workflow_guidance(state: ILMWorkflowState) -> str:
    """Generate workflow guidance based on current state, including error context."""
    # Check if workflow was cancelled
    if state.get("user_cancelled"):
        return (
            "The workflow was cancelled by the user.\n"
            "The policy update operation was not completed.\n"
            "Please inform the user that the operation was cancelled and the policy remains unchanged.\n"
            "Do NOT retry test_policy or update_policy unless the user explicitly requests it."
        )

    error = state.get("error_message")
    
    # If there's an error, provide context about it
    if error:
        # Check if it's a cancellation error
        if "cancelled" in error.lower():
            return (
                f"Operation was cancelled by user: {error}\n"
                "The workflow has been stopped. The user chose not to proceed with this operation.\n"
                "Please inform the user that the operation was cancelled.\n"
                "Do NOT retry the same operation unless the user explicitly requests it with different parameters."
            )
        return (
            f"Previous operation encountered an error: {error}\n"
            "You can:\n"
            "1. Retry the operation with corrected parameters\n"
            "2. Try a different approach\n"
            "3. Ask the user for clarification"
        )
    
    # Workflow step guidance mapping
    step = state["workflow_step"]

    # CRITICAL: For get_policy step, ALWAYS guide to get_policy first
    if step == "get_policy" and not state["policy_retrieved"]:
        return (
            "CRITICAL FIRST STEP: You MUST call get_policy to retrieve the existing policy "
            "before making ANY changes. This ensures all existing rules are preserved. "
            "Do NOT call test_policy, update_policy, list_storage_pools, or any other tool "
            "until you have successfully retrieved the existing policy with get_policy."
        )

    # After get_policy succeeds, verify pools before any modifications
    if step == "verify_pools" and not state["pools_verified"]:
        return (
            f"Policy retrieved for filesystem '{state['filesystem']}'. "
            "CRITICAL NEXT STEP: You MUST call list_storage_pools to verify available storage pools "
            "before generating or modifying any policy rules. This ensures the target pool exists. "
            "Do NOT proceed to test_policy or update_policy without verifying pools first."
        )

    # New step: Rule generation happens automatically after pools are verified
    if step == "generate_rule" and not state["rule_generated"]:
        if state.get("generated_rule"):
            rule_info = state["generated_rule"]
            return (
                f"Rule generated successfully: {rule_info.get('rule_name')}\n"
                f"Rule content:\n{rule_info.get('rule_content')}\n\n"
                "Next step: test the new policy with test_policy."
            )
        return "Generating policy rule based on user request..."

    if step == "test_policy" and not state["policy_tested"]:
        generated_rule = state.get("generated_rule", {})
        rule_content = generated_rule.get("rule_content", "")
        return (
            f"Rule generated. Next step: test the new policy with test_policy.\n"
            f"CRITICAL: Use the EXACT generated rule without ANY modifications:\n"
            f"{rule_content}\n\n"
            f"Combine this EXACT rule with ALL existing rules from the policy.\n"
            f"Do NOT modify the syntax - use it exactly as shown above."
        )

    if step == "update_policy" and state["test_passed"] and not state["policy_updated"]:
        return (
            "Policy test passed. Ready to update with update_policy. "
            "CRITICAL: Include the COMPLETE policy with ALL rules (existing + new)."
        )

    if step == "apply_policy" and state["policy_updated"]:
        return (
            "Policy updated successfully. Final step: apply the policy with apply_policy "
            "to execute the policy rules on the filesystem."
        )
    
    return ""


def generate_rule_node(state: ILMWorkflowState) -> Dict[str, Any]:
    """Node that generates ILM policy rules using structured logic.

    This node extracts information from the user's request and generates
    a properly formatted IBM Storage Scale policy rule without relying on LLM.
    """
    logger.debug("Entering rule generation node")

    # Extract user request from messages
    user_request = state.get("user_request")
    if not user_request:
        # Try to extract from the last human message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break

    if not user_request:
        return {
            "rule_generation_error": "No user request found for rule generation",
            "error_message": "Cannot generate rule without user request",
        }

    # Extract target pool from user request or state
    target_pool = state.get("target_pool")
    if not target_pool:
        # Try to extract from request (e.g., "migrate to archive", "move to cold pool")
        # Pattern excludes common action keywords and handles "pool" suffix
        pool_match = re.search(
            r'(?:to|into)\s+(?:the\s+)?["\']?(\w+?)(?:\s+pool)?["\']?(?:\s|$|,|\.)',
            user_request.lower()
        )
        if pool_match:
            potential_pool = pool_match.group(1)
            # Exclude common action keywords that are not pool names
            action_keywords = {'move', 'migrate', 'transfer', 'copy', 'relocate', 'shift'}
            if potential_pool not in action_keywords:
                target_pool = potential_pool

    # Extract source pool if specified
    source_pool = state.get("source_pool")
    if not source_pool:
        # Pattern handles "from pool" suffix and excludes action keywords
        from_match = re.search(
            r'from\s+(?:the\s+)?["\']?(\w+?)(?:\s+pool)?["\']?(?:\s|$|,|\.)',
            user_request.lower()
        )
        if from_match:
            potential_source = from_match.group(1)
            # Exclude common action keywords
            action_keywords = {'move', 'migrate', 'transfer', 'copy', 'relocate', 'shift'}
            if potential_source not in action_keywords:
                source_pool = potential_source

    if not target_pool:
        return {
            "rule_generation_error": "Target pool not specified in request",
            "error_message": "Please specify the target storage pool (e.g., 'migrate to archive')",
        }

    # Get available pools and existing rules
    storage_pools = state.get("storage_pools", [])
    existing_policy_text = state.get("existing_policy_text", "")

    # Verify storage pools were retrieved
    if not storage_pools:
        return {
            "rule_generation_error": "Storage pools not verified",
            "error_message": "Storage pools must be verified before generating rules. Please call list_storage_pools first.",
            "workflow_step": "verify_pools",
        }

    # Extract existing rule names
    existing_rules = []
    if existing_policy_text:
        for match in re.finditer(r"RULE\s+'([^']+)'", existing_policy_text, re.IGNORECASE):
            existing_rules.append(match.group(1))

    # Generate the rule
    result = generate_ilm_rule(
        user_request=user_request,
        target_pool=target_pool,
        existing_rules=existing_rules,
        available_pools=storage_pools,
        source_pool=source_pool,
    )

    if "error" in result:
        logger.error(f"Rule generation failed: {result['error']}")
        return {
            "rule_generation_error": result["error"],
            "error_message": result["error"],
        }

    # Check for redundancy
    redundancy_check = check_rule_redundancy(
        new_rule_metadata=result["metadata"],
        existing_policy_text=existing_policy_text,
    )

    if redundancy_check.get("is_redundant"):
        logger.warning(f"Rule is redundant: {redundancy_check['reason']}")
        return {
            "rule_generation_error": redundancy_check["reason"],
            "error_message": f"Redundant rule: {redundancy_check['reason']}",
        }

    logger.debug(f"Rule generated successfully: {result['rule_name']}")

    # Add informational message about the generated rule
    info_message = HumanMessage(
        content=f"[Rule Generated]\n"
                f"Rule Name: {result['rule_name']}\n"
                f"Target Pool: {result['metadata']['target_pool']}\n"
                f"Conditions: {result['metadata']['conditions_count']} condition(s)\n\n"
                f"Generated Rule:\n{result['rule_content']}"
    )

    return {
        "messages": [info_message],
        "generated_rule": result,
        "rule_generated": True,
        "workflow_step": "test_policy",
        "user_request": user_request,
        "target_pool": target_pool,
        "source_pool": source_pool,
    }


def _parse_tool_content(tool_content: Any) -> Dict[str, Any]:
    """Parse tool content into a dictionary.

    Args:
        tool_content: Tool response content (string or dict)

    Returns:
        Parsed dictionary

    Raises:
        ValueError: If content is a string but not valid JSON
    """
    if isinstance(tool_content, str):
        try:
            return json.loads(tool_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Tool content is not valid JSON: {str(e)}") from e
    return tool_content


def _check_tool_error(tool_result: Dict[str, Any], tool_name: str) -> tuple[bool, Optional[str]]:
    """Check if tool result contains an error.

    Detects errors from multiple sources by checking for common error keywords
    in various fields of the response. Also detects cancellation as an error.
    """
    if not isinstance(tool_result, dict):
        return False, None

    # Check for cancellation first (user said "no" to confirmation)
    if tool_result.get("cancelled") is True:
        error_message = tool_result.get("message") or f"Operation {tool_name} cancelled by user"
        logger.warning(f"Tool {tool_name} was cancelled by user")
        return True, str(error_message)

    # Common error keywords to check for
    error_keywords = ["error", "failed", "failure", "exception", "no such file", "cancelled"]
    
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
    
    logger.debug(f"Processing tool result for: {tool_name}")

    # Check for errors in tool result
    is_error, error_message = _check_tool_error(tool_result, tool_name)
    
    if is_error:
        # Check if this is a cancellation
        if "cancelled" in str(error_message).lower():
            logger.warning(f"User cancelled {tool_name} - marking workflow as cancelled")
            updates["error_message"] = error_message
            updates["user_cancelled"] = True
            updates["workflow_step"] = "cancelled"
            return updates

        updates["error_message"] = error_message
        logger.debug(f"Workflow staying at step '{state['workflow_step']}' due to error - agent can retry")
        return updates

    # Clear any previous error on success
    if state.get("error_message"):
        logger.debug("Clearing previous error - operation succeeded")
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
        
        logger.debug(f"Policy retrieved for filesystem: {updates.get('filesystem')}")
    
    elif tool_name == "list_storage_pools":
        # Only mark as verified if we actually got pool data (not an error response)
        # MCP server returns "storage_pools" field with pool objects containing "name" field
        if isinstance(tool_result, dict) and "storage_pools" in tool_result:
            pools = tool_result["storage_pools"]
            if isinstance(pools, list):
                pool_names = [p.get("name") for p in pools if isinstance(p, dict) and p.get("name")]
                updates["storage_pools"] = pool_names
                updates["pools_verified"] = True
                # Transition to rule generation step instead of directly to test_policy
                updates["workflow_step"] = "generate_rule"
                logger.debug(f"Storage pools verified: {pool_names}. Moving to rule generation.")
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
            logger.debug("Policy test PASSED - ready for update")
        else:
            updates["error_message"] = "Policy test failed - please fix errors before updating"
            logger.warning("Policy test FAILED")
    
    elif tool_name == "update_policy":
        # Check if the operation was successful
        if _check_success_status(tool_result):
            updates.update({
                "policy_updated": True,
                "workflow_step": "apply_policy",
            })
            logger.debug("Policy updated successfully - ready to apply")
        else:
            # update_policy failed or was cancelled
            # If it was cancelled, the workflow is already set to "cancelled" state above
            # Don't reset test flags to prevent looping back to test_policy
            logger.warning("update_policy failed or was cancelled - workflow will not advance")
            # Keep workflow at update_policy step (or cancelled if user cancelled)
            # This prevents the agent from going back to test_policy
    
    elif tool_name == "apply_policy" and _check_success_status(tool_result):
        updates["workflow_step"] = "completed"
        logger.info("Policy applied successfully - workflow complete")
    
    return updates


def should_continue(state: ILMWorkflowState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    # If workflow was cancelled, end immediately
    if state.get("user_cancelled") or state.get("workflow_step") == "cancelled":
        logger.info("Workflow cancelled - ending execution")
        return "end"

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

        # ALWAYS add workflow guidance when available (including initial step)
        if workflow_guidance:
            logger.debug(f"Workflow Status: {workflow_guidance}")
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
    
    def should_generate_rule(state: ILMWorkflowState) -> Literal["generate_rule", "agent"]:
        """Determine if we should generate a rule or continue with agent."""
        if state["workflow_step"] == "generate_rule" and not state["rule_generated"]:
            return "generate_rule"
        return "agent"

    # Create the graph
    workflow = StateGraph(ILMWorkflowState)

    # Add nodes with context
    workflow.add_node("agent", agent_node_with_context)
    workflow.add_node("tools", tool_execution_node_with_context)
    workflow.add_node("validate", validate_tool_call)
    workflow.add_node("generate_rule", generate_rule_node)
    
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

    # After validation, go to tools
    workflow.add_edge("validate", "tools")

    # After tools, check if we need to generate a rule or go back to agent
    workflow.add_conditional_edges(
        "tools",
        should_generate_rule,
        {
            "generate_rule": "generate_rule",
            "agent": "agent",
        },
    )

    # After rule generation, go back to agent
    workflow.add_edge("generate_rule", "agent")
    
    return workflow
