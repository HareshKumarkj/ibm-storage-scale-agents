"""IBM Storage Scale ILM Policy Rule Generator.

This module provides structured rule generation following the official IBM Storage Scale
policy syntax: https://www.ibm.com/docs/en/storage-scale/5.2.3?topic=rules-policy-syntax
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Rule generation templates based on IBM Storage Scale policy syntax
RULE_TEMPLATES = {
    "file_patterns": {
        # Temporary and system files
        "temp": "lower(NAME) LIKE '%.tmp'",
        "temporary": "lower(NAME) LIKE '%.tmp'",
        "cache": "lower(NAME) LIKE '%.cache'",
        "swap": "lower(NAME) LIKE '%.swp'",
        
        # Log files
        "log": "lower(NAME) LIKE '%.log'",
        "logs": "lower(NAME) LIKE '%.log'",
        
        # Backup files
        "backup": "lower(NAME) LIKE '%.bak'",
        "bak": "lower(NAME) LIKE '%.bak'",
        
        # Document files
        "text": "lower(NAME) LIKE '%.txt'",
        "txt": "lower(NAME) LIKE '%.txt'",
        "pdf": "lower(NAME) LIKE '%.pdf'",
        "doc": "(lower(NAME) LIKE '%.doc' OR lower(NAME) LIKE '%.docx')",
        "document": "(lower(NAME) LIKE '%.doc' OR lower(NAME) LIKE '%.docx' OR lower(NAME) LIKE '%.pdf')",
        "spreadsheet": "(lower(NAME) LIKE '%.xls' OR lower(NAME) LIKE '%.xlsx' OR lower(NAME) LIKE '%.csv')",
        "presentation": "(lower(NAME) LIKE '%.ppt' OR lower(NAME) LIKE '%.pptx')",
        
        # Data files
        "csv": "lower(NAME) LIKE '%.csv'",
        "json": "lower(NAME) LIKE '%.json'",
        "xml": "lower(NAME) LIKE '%.xml'",
        "yaml": "(lower(NAME) LIKE '%.yaml' OR lower(NAME) LIKE '%.yml')",
        
        # Archive files
        "zip": "lower(NAME) LIKE '%.zip'",
        "tar": "(lower(NAME) LIKE '%.tar' OR lower(NAME) LIKE '%.tar.gz' OR lower(NAME) LIKE '%.tgz')",
        "archive": "(lower(NAME) LIKE '%.zip' OR lower(NAME) LIKE '%.tar' OR lower(NAME) LIKE '%.tar.gz' OR lower(NAME) LIKE '%.tgz' OR lower(NAME) LIKE '%.rar' OR lower(NAME) LIKE '%.7z')",
        "compressed": "(lower(NAME) LIKE '%.gz' OR lower(NAME) LIKE '%.bz2' OR lower(NAME) LIKE '%.xz')",
        
        # Image files
        "image": "(lower(NAME) LIKE '%.jpg' OR lower(NAME) LIKE '%.jpeg' OR lower(NAME) LIKE '%.png' OR lower(NAME) LIKE '%.gif' OR lower(NAME) LIKE '%.bmp' OR lower(NAME) LIKE '%.tiff')",
        "jpg": "(lower(NAME) LIKE '%.jpg' OR lower(NAME) LIKE '%.jpeg')",
        "jpeg": "(lower(NAME) LIKE '%.jpg' OR lower(NAME) LIKE '%.jpeg')",
        "png": "lower(NAME) LIKE '%.png'",
        "gif": "lower(NAME) LIKE '%.gif'",
        "raw": "(lower(NAME) LIKE '%.raw' OR lower(NAME) LIKE '%.cr2' OR lower(NAME) LIKE '%.nef' OR lower(NAME) LIKE '%.arw')",
        
        # Video files
        "video": "(lower(NAME) LIKE '%.mp4' OR lower(NAME) LIKE '%.avi' OR lower(NAME) LIKE '%.mov' OR lower(NAME) LIKE '%.mkv' OR lower(NAME) LIKE '%.wmv' OR lower(NAME) LIKE '%.flv')",
        "mp4": "lower(NAME) LIKE '%.mp4'",
        "avi": "lower(NAME) LIKE '%.avi'",
        "mov": "lower(NAME) LIKE '%.mov'",
        "mkv": "lower(NAME) LIKE '%.mkv'",
        
        # Audio files
        "audio": "(lower(NAME) LIKE '%.mp3' OR lower(NAME) LIKE '%.wav' OR lower(NAME) LIKE '%.flac' OR lower(NAME) LIKE '%.aac' OR lower(NAME) LIKE '%.ogg' OR lower(NAME) LIKE '%.m4a')",
        "mp3": "lower(NAME) LIKE '%.mp3'",
        "wav": "lower(NAME) LIKE '%.wav'",
        "flac": "lower(NAME) LIKE '%.flac'",
        
        # Code files
        "code": "(lower(NAME) LIKE '%.py' OR lower(NAME) LIKE '%.java' OR lower(NAME) LIKE '%.cpp' OR lower(NAME) LIKE '%.c' OR lower(NAME) LIKE '%.js' OR lower(NAME) LIKE '%.go' OR lower(NAME) LIKE '%.rs')",
        "python": "lower(NAME) LIKE '%.py'",
        "java": "lower(NAME) LIKE '%.java'",
        "javascript": "(lower(NAME) LIKE '%.js' OR lower(NAME) LIKE '%.jsx' OR lower(NAME) LIKE '%.ts' OR lower(NAME) LIKE '%.tsx')",
        "cpp": "(lower(NAME) LIKE '%.cpp' OR lower(NAME) LIKE '%.cc' OR lower(NAME) LIKE '%.cxx')",
        "c": "lower(NAME) LIKE '%.c'",
        "go": "lower(NAME) LIKE '%.go'",
        "rust": "lower(NAME) LIKE '%.rs'",
        
        # Database files
        "database": "(lower(NAME) LIKE '%.db' OR lower(NAME) LIKE '%.sqlite' OR lower(NAME) LIKE '%.mdb')",
        "sql": "lower(NAME) LIKE '%.sql'",
        
        # Configuration files
        "config": "(lower(NAME) LIKE '%.conf' OR lower(NAME) LIKE '%.cfg' OR lower(NAME) LIKE '%.ini' OR lower(NAME) LIKE '%.yaml' OR lower(NAME) LIKE '%.yml')",
    },
    "size_units": {
        "kb": 1024,
        "mb": 1048576,
        "gb": 1073741824,
        "tb": 1099511627776,
    },
    "time_units": {
        "days": 1,
        "weeks": 7,
        "months": 30,
        "years": 365,
    }
}


def generate_ilm_rule(
    user_request: str,
    target_pool: str,
    existing_rules: List[str],
    available_pools: List[str],
    source_pool: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate ILM policy rule based on user request using IBM Storage Scale syntax.
    
    File migration rule syntax:
    RULE ['RuleName'] [WHEN TimeBooleanExpression]
      MIGRATE
        [FROM POOL 'FromPoolName']
        [THRESHOLD (HighPercentage[,LowPercentage[,PremigratePercentage]])]
        [WEIGHT (WeightExpression)]
      TO POOL 'ToPoolName'
        [LIMIT (OccupancyPercentage)]
        [FOR FILESET ('FilesetName'[,'FilesetName']...)]
        [SHOW (['String'] SqlExpression)]
        [ACTION (SqlExpression)]
        [WHERE SqlExpression]
    
    Args:
        user_request: User's natural language request
        target_pool: Target storage pool name
        existing_rules: List of existing rule names
        available_pools: List of available pool names
        source_pool: Optional source pool name
        
    Returns:
        Dictionary with rule_name, rule_content, and metadata
    """
    request_lower = user_request.lower()
    
    # Extract file pattern
    file_pattern = None
    file_type = None
    for pattern_key, pattern_value in RULE_TEMPLATES["file_patterns"].items():
        if pattern_key in request_lower:
            file_pattern = pattern_value
            file_type = pattern_key
            break
    
    # Extract directory/path pattern (e.g., "/data/archive", "in /tmp directory")
    path_pattern = None
    path_match = re.search(r'(?:in|from|under)\s+["\']?(/[^\s"\']+)["\']?', request_lower)
    if path_match:
        path_pattern = path_match.group(1)
    
    # Determine time field: ACCESS_TIME (default) or MODIFICATION_TIME
    time_field = "ACCESS_TIME"
    if any(keyword in request_lower for keyword in ["modified", "modification", "changed", "updated"]):
        time_field = "MODIFICATION_TIME"
    elif any(keyword in request_lower for keyword in ["created", "creation"]):
        time_field = "CREATION_TIME"
    
    # Extract age condition with support for multiple time units
    age_days = None
    age_display = None
    
    # Try different time unit patterns
    for unit_name, multiplier in RULE_TEMPLATES["time_units"].items():
        pattern = rf'(\d+)\s*{unit_name[:-1]}s?'  # Match singular or plural
        age_match = re.search(pattern, request_lower)
        if age_match:
            age_value = int(age_match.group(1))
            age_days = age_value * multiplier
            age_display = f"{age_value} {unit_name}"
            break
    
    # Extract size condition (e.g., "100MB", "1GB", "larger than 500MB")
    size_bytes = None
    size_display = None
    size_operator = ">"  # Default: larger than
    
    # Check for size comparison operators
    if any(keyword in request_lower for keyword in ["smaller than", "less than", "under"]):
        size_operator = "<"
    elif any(keyword in request_lower for keyword in ["at least", "minimum"]):
        size_operator = ">="
    
    size_match = re.search(r'(\d+)\s*(kb|mb|gb|tb)', request_lower, re.IGNORECASE)
    if size_match:
        size_value = int(size_match.group(1))
        size_unit = size_match.group(2).lower()
        size_bytes = size_value * RULE_TEMPLATES["size_units"][size_unit]
        size_display = f"{size_value}{size_unit.upper()}"
    
    # Validate target pool
    if target_pool not in available_pools:
        return {
            "error": f"Target pool '{target_pool}' not found in available pools: {', '.join(available_pools)}"
        }
    
    # Validate source pool if specified
    if source_pool and source_pool not in available_pools:
        return {
            "error": f"Source pool '{source_pool}' not found in available pools: {', '.join(available_pools)}"
        }
    
    # Generate descriptive rule name
    rule_name_parts = ["migrate"]
    
    # Add descriptors based on conditions
    if age_days:
        if time_field == "MODIFICATION_TIME":
            rule_name_parts.append("Unmodified")
        elif time_field == "CREATION_TIME":
            rule_name_parts.append("Old")
        else:
            rule_name_parts.append("Unaccessed")
    
    if size_bytes:
        if size_operator == "<":
            rule_name_parts.append("Small")
        elif size_operator == ">=":
            rule_name_parts.append("MinSize")
        else:
            rule_name_parts.append("Large")
    
    if file_type:
        rule_name_parts.append(file_type.capitalize())
    else:
        rule_name_parts.append("Files")
    
    if path_pattern:
        # Extract last directory name for rule name
        path_parts = path_pattern.rstrip('/').split('/')
        if path_parts:
            dir_name = path_parts[-1].capitalize()
            rule_name_parts.append(f"In{dir_name}")
    
    if source_pool:
        rule_name_parts.append(f"From{source_pool.capitalize()}")
    rule_name_parts.append(f"To{target_pool.capitalize()}")
    
    # Ensure unique rule name
    base_rule_name = "".join(rule_name_parts)
    rule_name = base_rule_name
    counter = 1
    while rule_name in existing_rules:
        rule_name = f"{base_rule_name}{counter}"
        counter += 1
    
    # Build WHERE clause conditions
    conditions = []
    
    # Age condition with configurable time field
    if age_days:
        conditions.append(f"DAYS(CURRENT_TIMESTAMP) - DAYS({time_field}) > {age_days}")
    
    # Size condition with configurable operator
    if size_bytes:
        conditions.append(f"FILE_SIZE {size_operator} {size_bytes}")
    
    # File pattern condition
    if file_pattern:
        conditions.append(file_pattern)
    
    # Path/directory pattern condition
    if path_pattern:
        # Escape single quotes in path and use PATH_NAME for directory matching
        escaped_path = path_pattern.replace("'", "''")
        conditions.append(f"PATH_NAME LIKE '{escaped_path}%'")
    
    # Build rule content following IBM Storage Scale syntax
    rule_lines = [f"RULE '{rule_name}'"]
    rule_lines.append("  MIGRATE")
    
    if source_pool:
        rule_lines.append(f"    FROM POOL '{source_pool}'")
    
    rule_lines.append(f"  TO POOL '{target_pool}'")
    
    if conditions:
        where_clause = " AND\n      ".join(conditions)
        rule_lines.append(f"    WHERE {where_clause}")
    
    rule_content = "\n".join(rule_lines)
    
    return {
        "rule_name": rule_name,
        "rule_content": rule_content,
        "metadata": {
            "file_type": file_type,
            "age_days": age_days,
            "age_display": age_display,
            "time_field": time_field,
            "size_bytes": size_bytes,
            "size_display": size_display,
            "size_operator": size_operator,
            "path_pattern": path_pattern,
            "target_pool": target_pool,
            "source_pool": source_pool,
            "conditions_count": len(conditions),
        },
    }


def generate_advanced_ilm_rule(
    user_request: str,
    target_pool: str,
    existing_rules: List[str],
    available_pools: List[str],
    source_pool: Optional[str] = None,
    weight_expression: Optional[str] = None,
    threshold_percentages: Optional[tuple] = None,
    limit_percentage: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate advanced ILM policy rule with WEIGHT, THRESHOLD, and LIMIT clauses.
    
    This function extends generate_ilm_rule with support for:
    - WEIGHT expressions for migration prioritization
    - THRESHOLD for source pool capacity management
    - LIMIT for target pool capacity management
    
    Args:
        user_request: User's natural language request
        target_pool: Target storage pool name
        existing_rules: List of existing rule names
        available_pools: List of available pool names
        source_pool: Optional source pool name
        weight_expression: SQL expression for migration weight/priority
        threshold_percentages: Tuple of (high, low, premigrate) percentages
        limit_percentage: Maximum occupancy percentage for target pool
        
    Returns:
        Dictionary with rule_name, rule_content, and metadata
    """
    # Start with basic rule generation
    basic_rule = generate_ilm_rule(
        user_request=user_request,
        target_pool=target_pool,
        existing_rules=existing_rules,
        available_pools=available_pools,
        source_pool=source_pool,
    )
    
    if "error" in basic_rule:
        return basic_rule
    
    # Parse the basic rule to insert advanced clauses
    rule_lines = basic_rule["rule_content"].split("\n")
    enhanced_lines = []
    
    for line in rule_lines:
        enhanced_lines.append(line)
        
        # Insert THRESHOLD after FROM POOL
        if "FROM POOL" in line and threshold_percentages:
            high, low, premigrate = threshold_percentages
            if premigrate:
                enhanced_lines.append(f"    THRESHOLD ({high},{low},{premigrate})")
            elif low:
                enhanced_lines.append(f"    THRESHOLD ({high},{low})")
            else:
                enhanced_lines.append(f"    THRESHOLD ({high})")
        
        # Insert WEIGHT after THRESHOLD or FROM POOL
        if ("THRESHOLD" in line or ("FROM POOL" in line and not threshold_percentages)) and weight_expression:
            enhanced_lines.append(f"    WEIGHT ({weight_expression})")
        
        # Insert LIMIT after TO POOL
        if "TO POOL" in line and limit_percentage:
            enhanced_lines.append(f"    LIMIT ({limit_percentage})")
    
    enhanced_content = "\n".join(enhanced_lines)
    
    # Update metadata
    enhanced_metadata = basic_rule["metadata"].copy()
    enhanced_metadata.update({
        "weight_expression": weight_expression,
        "threshold_percentages": threshold_percentages,
        "limit_percentage": limit_percentage,
        "is_advanced": True,
    })
    
    return {
        "rule_name": basic_rule["rule_name"],
        "rule_content": enhanced_content,
        "metadata": enhanced_metadata,
    }


def generate_deletion_rule(
    user_request: str,
    existing_rules: List[str],
    source_pool: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate ILM policy rule for file deletion.
    
    Deletion rule syntax:
    RULE ['RuleName']
      DELETE
        [FROM POOL 'PoolName']
        [WHERE SqlExpression]
    
    Args:
        user_request: User's natural language request
        existing_rules: List of existing rule names
        source_pool: Optional source pool to delete from
        
    Returns:
        Dictionary with rule_name, rule_content, and metadata
    """
    request_lower = user_request.lower()
    
    # Extract file pattern
    file_pattern = None
    file_type = None
    for pattern_key, pattern_value in RULE_TEMPLATES["file_patterns"].items():
        if pattern_key in request_lower:
            file_pattern = pattern_value
            file_type = pattern_key
            break
    
    # Extract age condition
    age_days = None
    age_display = None
    time_field = "ACCESS_TIME"
    
    if any(keyword in request_lower for keyword in ["modified", "modification"]):
        time_field = "MODIFICATION_TIME"
    
    for unit_name, multiplier in RULE_TEMPLATES["time_units"].items():
        pattern = rf'(\d+)\s*{unit_name[:-1]}s?'
        age_match = re.search(pattern, request_lower)
        if age_match:
            age_value = int(age_match.group(1))
            age_days = age_value * multiplier
            age_display = f"{age_value} {unit_name}"
            break
    
    # Extract size condition
    size_bytes = None
    size_display = None
    size_operator = ">"
    
    if any(keyword in request_lower for keyword in ["smaller than", "less than"]):
        size_operator = "<"
    
    size_match = re.search(r'(\d+)\s*(kb|mb|gb|tb)', request_lower, re.IGNORECASE)
    if size_match:
        size_value = int(size_match.group(1))
        size_unit = size_match.group(2).lower()
        size_bytes = size_value * RULE_TEMPLATES["size_units"][size_unit]
        size_display = f"{size_value}{size_unit.upper()}"
    
    # Generate rule name
    rule_name_parts = ["delete"]
    
    if age_days:
        rule_name_parts.append("Old")
    if size_bytes:
        rule_name_parts.append("Small" if size_operator == "<" else "Large")
    if file_type:
        rule_name_parts.append(file_type.capitalize())
    else:
        rule_name_parts.append("Files")
    
    if source_pool:
        rule_name_parts.append(f"From{source_pool.capitalize()}")
    
    # Ensure unique rule name
    base_rule_name = "".join(rule_name_parts)
    rule_name = base_rule_name
    counter = 1
    while rule_name in existing_rules:
        rule_name = f"{base_rule_name}{counter}"
        counter += 1
    
    # Build WHERE clause
    conditions = []
    if age_days:
        conditions.append(f"DAYS(CURRENT_TIMESTAMP) - DAYS({time_field}) > {age_days}")
    if size_bytes:
        conditions.append(f"FILE_SIZE {size_operator} {size_bytes}")
    if file_pattern:
        conditions.append(file_pattern)
    
    # Build rule content
    rule_lines = [f"RULE '{rule_name}'"]
    rule_lines.append("  DELETE")
    
    if source_pool:
        rule_lines.append(f"    FROM POOL '{source_pool}'")
    
    if conditions:
        where_clause = " AND\n      ".join(conditions)
        rule_lines.append(f"    WHERE {where_clause}")
    
    rule_content = "\n".join(rule_lines)
    
    return {
        "rule_name": rule_name,
        "rule_content": rule_content,
        "metadata": {
            "rule_type": "DELETE",
            "file_type": file_type,
            "age_days": age_days,
            "age_display": age_display,
            "time_field": time_field,
            "size_bytes": size_bytes,
            "size_display": size_display,
            "size_operator": size_operator,
            "source_pool": source_pool,
            "conditions_count": len(conditions),
        },
    }
    
    if conditions:
        where_clause = " AND\n      ".join(conditions)
        rule_lines.append(f"    WHERE {where_clause}")
    
    rule_content = "\n".join(rule_lines)
    
    return {
        "rule_name": rule_name,
        "rule_content": rule_content,
        "metadata": {
            "file_type": file_type,
            "age_days": age_days,
            "age_display": age_display,
            "time_field": time_field,
            "size_bytes": size_bytes,
            "size_display": size_display,
            "size_operator": size_operator,
            "path_pattern": path_pattern,
            "target_pool": target_pool,
            "source_pool": source_pool,
            "conditions_count": len(conditions),
        },
    }


def check_rule_redundancy(
    new_rule_metadata: Dict[str, Any],
    existing_policy_text: Optional[str],
) -> Dict[str, Any]:
    """Check if a new rule is redundant with existing rules.
    
    Redundancy logic:
    - Rules are redundant only if BOTH file pattern AND target pool match
    - A new rule with MORE restrictive conditions (higher age/size) is redundant
    - A new rule with LESS restrictive conditions is NOT redundant (both should exist)
    - Different target pools = NOT redundant (even with same pattern)
    - Different file patterns = NOT redundant (even with same pool)
    
    Args:
        new_rule_metadata: Dictionary with new rule details (from generate_ilm_rule)
        existing_policy_text: Existing policy content as string
        
    Returns:
        Dictionary with redundancy status and details
    """
    if not existing_policy_text:
        return {"is_redundant": False, "reason": "No existing policy to check"}
    
    # Parse existing MIGRATE rules
    existing_rules = []
    for match in re.finditer(
        r"RULE\s+'([^']+)'.*?MIGRATE.*?(?:FROM\s+POOL\s+'([^']+)')?\s*TO\s+POOL\s+'([^']+)'.*?(?:WHERE\s+(.*?))?(?=RULE|$)",
        existing_policy_text,
        re.DOTALL | re.IGNORECASE
    ):
        rule_name, source_pool, target_pool, where_clause = match.groups()
        
        # Extract conditions from WHERE clause
        age_days = None
        size_bytes = None
        has_file_pattern = False
        
        if where_clause:
            age_match = re.search(r'DAYS\(CURRENT_TIMESTAMP\)\s*-\s*DAYS\(ACCESS_TIME\)\s*>\s*(\d+)', where_clause, re.IGNORECASE)
            if age_match:
                age_days = int(age_match.group(1))
            
            size_match = re.search(r'FILE_SIZE\s*>\s*(\d+)', where_clause, re.IGNORECASE)
            if size_match:
                size_bytes = int(size_match.group(1))
            
            if 'LIKE' in where_clause.upper():
                has_file_pattern = True
        
        existing_rules.append({
            "name": rule_name,
            "source_pool": source_pool,
            "target_pool": target_pool,
            "age_days": age_days,
            "size_bytes": size_bytes,
            "has_file_pattern": has_file_pattern,
            "where_clause": where_clause,
        })
    
    # Check for redundancy with each existing rule
    new_target = new_rule_metadata.get("target_pool")
    new_source = new_rule_metadata.get("source_pool")
    new_age = new_rule_metadata.get("age_days")
    new_size = new_rule_metadata.get("size_bytes")
    new_has_pattern = new_rule_metadata.get("file_type") is not None
    
    for existing in existing_rules:
        # Must have same target pool
        if existing["target_pool"] != new_target:
            continue
        
        # Must have same source pool (or both None)
        if existing["source_pool"] != new_source:
            continue
        
        # Check file pattern presence (simplified check)
        if existing["has_file_pattern"] != new_has_pattern:
            continue
        
        # Check if new rule is more restrictive (redundant)
        is_more_restrictive = False
        
        # For age: higher threshold = more restrictive (fewer files match)
        if existing["age_days"] and new_age:
            if new_age > existing["age_days"]:
                is_more_restrictive = True
        
        # For size: higher threshold = more restrictive (fewer files match)
        if existing["size_bytes"] and new_size:
            if new_size > existing["size_bytes"]:
                is_more_restrictive = True
        
        # If conditions match exactly, also redundant
        if (existing["age_days"] == new_age and 
            existing["size_bytes"] == new_size and
            existing["has_file_pattern"] == new_has_pattern):
            is_more_restrictive = True
        
        if is_more_restrictive:
            return {
                "is_redundant": True,
                "reason": f"Redundant with existing rule '{existing['name']}' - same pattern and pool with more restrictive or identical conditions",
                "existing_rule": existing["name"],
            }
    
    return {"is_redundant": False, "reason": "No redundancy detected"}
