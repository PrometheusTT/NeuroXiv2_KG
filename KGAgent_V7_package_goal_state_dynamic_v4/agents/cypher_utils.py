from __future__ import annotations
import re
from typing import Optional, List, Dict, Any


def sanitize_query(query: str) -> str:
    """
    Sanitize a Cypher query to remove potentially destructive commands.
    """
    # Check for write operations that should be blocked
    write_operations = ["CREATE", "DELETE", "DETACH DELETE", "REMOVE", "SET", "MERGE", "DROP"]

    # Convert to uppercase for case-insensitive matching
    query_upper = query.upper()

    for op in write_operations:
        if re.search(r'\b' + op + r'\b', query_upper):
            raise ValueError(f"Unauthorized operation detected: {op}")

    return query


def extract_json_block(s: str) -> Optional[str]:
    """
    Extract a JSON block from a string, handling both bare and markdown-fenced JSON.
    """
    # Try to find JSON block within markdown code fence
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_block_pattern, s)

    if matches:
        # Return the first valid JSON block
        return matches[0].strip()

    # If no markdown block found, try to find JSON between curly braces
    # This is trickier since we need to handle nested braces
    s = s.strip()
    if s.startswith('{') and s.endswith('}'):
        # Count opening and closing braces to handle nesting
        depth = 0
        for i, char in enumerate(s):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and i == len(s) - 1:
                    return s  # Found balanced JSON

    return None


def validate_cypher_query(query: str) -> Dict[str, Any]:
    """
    Validate a Cypher query for common issues and suggest improvements.
    Returns a dict with validation result.
    """
    result = {
        "valid": True,
        "warnings": [],
        "suggestions": []
    }

    # Check for returning entire nodes (causes unhashable type errors)
    return_clause_pattern = r'\bRETURN\s+([^.]+?)\b(?![.)])'
    returns = re.finditer(return_clause_pattern, query, re.IGNORECASE)

    for match in returns:
        var_name = match.group(1).strip()
        # Skip if already structured or if it's a function like COUNT
        if var_name.startswith('COUNT(') or '(' in var_name or ')' in var_name:
            continue

        # Check if this is a variable not a property
        if var_name and var_name.isalnum() and not any(x in var_name for x in ['.', '(', ')', ' ']):
            result["valid"] = False
            result["warnings"].append(f"Query returns entire node '{var_name}' which may cause errors")
            result["suggestions"].append(f"Change 'RETURN {var_name}' to 'RETURN {var_name}.name, {var_name}.id'")

    # Check for COUNT without GROUP BY
    if "COUNT" in query.upper() and "GROUP BY" not in query.upper():
        count_pattern = r'\bCOUNT\s*\([^)]+\)'
        if re.search(count_pattern, query, re.IGNORECASE):
            # Check if there are other fields in the RETURN clause
            return_clause = re.search(r'\bRETURN\s+(.+?)(?:$|WHERE|LIMIT|ORDER)', query, re.IGNORECASE)
            if return_clause:
                return_items = return_clause.group(1).split(',')
                if len(return_items) > 1:  # More than just COUNT
                    result["valid"] = False
                    result["warnings"].append("Using COUNT with other fields requires GROUP BY")
                    result["suggestions"].append("Add GROUP BY clause for non-aggregated fields")

    # Check for potential syntax issues
    if query.count('(') != query.count(')'):
        result["valid"] = False
        result["warnings"].append("Mismatched parentheses")
        result["suggestions"].append("Check for missing opening or closing parentheses")

    # Check for equality vs CONTAINS for string matching
    if "=" in query and "WHERE" in query.upper():
        equals_pattern = r'WHERE\s+\w+\.\w+\s*=\s*[\'"]'
        if re.search(equals_pattern, query, re.IGNORECASE):
            result["warnings"].append("Using exact equality (=) for string matching")
            result["suggestions"].append("Consider using CONTAINS for partial string matching")

    return result


def improve_cypher_query(query: str) -> str:
    """
    Automatically improve a Cypher query based on validation results.
    """
    validation = validate_cypher_query(query)

    if validation["valid"]:
        return query

    improved_query = query

    # Fix returning entire nodes
    return_clause_pattern = r'\bRETURN\s+([^.]+?)\b(?![.)])'
    matches = list(re.finditer(return_clause_pattern, improved_query, re.IGNORECASE))

    # Process in reverse to avoid offsetting earlier replacements
    for match in reversed(matches):
        var_name = match.group(1).strip()
        # Skip if already structured or if it's a function like COUNT
        if var_name.startswith('COUNT(') or '(' in var_name or ')' in var_name:
            continue

        # Check if this is a variable not a property
        if var_name and var_name.isalnum() and not any(x in var_name for x in ['.', '(', ')', ' ']):
            # Replace with returning the name and labels of the node
            replacement = f"RETURN {var_name}.name as {var_name}_name, labels({var_name}) as {var_name}_labels"
            improved_query = improved_query[:match.start()] + replacement + improved_query[match.end():]

    return improved_query