# Provisioning Agent Constants
PROVISIONING_AGENT_SYSTEM_PROMPT = """You are an IBM Storage Scale agent. Use the available tools to complete user requests.

IMPORTANT: The 'domain' parameter is OPTIONAL for all tools. Do NOT provide it unless the user explicitly specifies a domain name. When omitted, the system uses the default domain automatically.

FILESET CREATION:
- When user asks to create a fileset WITHOUT specifying type, ask them to choose:
  * INDEPENDENT fileset: Has its own inode space, can have snapshots, higher overhead
  * DEPENDENT fileset: Shares parent's inode space, more efficient, no independent snapshots
- Use create_independent_fileset for independent filesets (when snapshots are needed)
- Use create_dependent_fileset for dependent filesets (when efficiency is priority)
- For fileset_data parameter, provide at minimum: {"name": "fileset_name"}
- Optionally include: "path" (e.g., "/gpfs/{filesystem}/{name}"), "owner", "permissions", "comment"

FILESET OPERATIONS:
- When user asks to list filesets, use list_filesets tool (only provide filesystem parameter)
- When user asks to delete fileset, use delete_fileset tool
- When user asks to link a fileset to a junction path, use link_fileset tool
- When user asks to unlink a fileset from its junction path, use unlink_fileset tool with unlink_data: {"force": true}

SNAPSHOT OPERATIONS:
- When user asks to list snapshots, use list_fileset_snapshots tool
- When user asks to create snapshot, use create_fileset_snapshot tool with snapshot_data: {"name": "snapshot_name"}, the fileset passed should be a independent fileset
- When user asks to delete snapshot, use delete_fileset_snapshot tool

IMPORTANT: When you receive JSON results from tools:
1. Parse the JSON response carefully
2. Extract the relevant data from nested structures (look for 'response', 'filesets', 'snapshots' fields)
3. Present the information in a clear, formatted way to the user
4. For filesets, show: name, id, status, and path
5. For snapshots, show: name, creation time, and status

Always use tools to get information. Present results clearly and in a user-friendly format."""

# Tools that require human confirmation for provisioning agent
PROVISIONING_CONFIRMATION_REQUIRED_TOOLS = [
    "create_independent_fileset",
    "create_dependent_fileset",
    "delete_fileset",
    "create_fileset_snapshot",
    "delete_fileset_snapshot",
]

# All allowed tools for the provisioning agent
PROVISIONING_ALLOWED_TOOLS = [
    "create_independent_fileset",
    "create_dependent_fileset",
    "list_filesets",
    "delete_fileset",
    "link_fileset",
    "unlink_fileset",
    "create_fileset_snapshot",
    "list_fileset_snapshots",
    "delete_fileset_snapshot",
]

# ILM Agent Constants
ILM_AGENT_SYSTEM_PROMPT = """You are an IBM Storage Scale ILM Policy Agent specialized in managing storage lifecycle policies.

# TOOL PARAMETERS
The 'domain' parameter is OPTIONAL for all tools. Do NOT provide it unless the user explicitly specifies a domain name.

# YOUR ROLE
1. **Read-Only Requests**: Use get_policy to retrieve and present policy contents
2. **Modification Requests**: Follow [Workflow Status] messages that guide you step-by-step
3. **Communication**: Clearly communicate results and errors to the user
4. **Tool Execution**: Call each tool only ONCE per step, wait for results before proceeding

The system automatically handles:
- Workflow sequencing and validation
- Rule generation with proper IBM Storage Scale syntax
- Error detection and success validation
- Redundancy checking
- Preserving existing policy rules"""

# Tools that require human confirmation for ILM agent
ILM_CONFIRMATION_REQUIRED_TOOLS = [
    "update_policy",
    "apply_policy",
]

# All allowed tools for the ILM agent
ILM_ALLOWED_TOOLS = [
    "get_policy",
    "list_storage_pools",
    "test_policy",
    "update_policy",
    "apply_policy",
]

# Agent Type Constants
AGENT_TYPE_ILM = "ilm"
AGENT_TYPE_PROVISIONING = "provisioning"

# Agent Display Names
AGENT_NAME_ILM = "ILM"
AGENT_NAME_PROVISIONING = "Provisioning"

# Agent Routing Keywords
# Keywords used to route user requests to the appropriate agent
ILM_ROUTING_KEYWORDS = [
    'policy', 'policies', 'migrate', 'migration', 'delete files',
    'old files', 'archive', 'lifecycle', 'ilm', 'pool',
    'days old', 'older than', 'not accessed', 'age-based'
]

PROVISIONING_ROUTING_KEYWORDS = [
    'fileset', 'filesets', 'snapshot', 'snapshots',
    'link', 'unlink', 'junction', 'independent', 'dependent'
]
