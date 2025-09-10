# Enhanced neo4j_exec.py with comprehensive schema extraction

from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
import pandas as pd
from .cypher_utils import sanitize_query

try:
    from neo4j import GraphDatabase

    _HAS_NEO4J = True
except Exception:
    _HAS_NEO4J = False


class Neo4jExec:
    """
    Enhanced Neo4j executor with comprehensive schema extraction and error diagnostics.
    """

    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI")
        self.user = os.environ.get("NEO4J_USER")
        self.pw = os.environ.get("NEO4J_PASSWORD")
        self.connected = False
        self._driver = None
        self._cached_schema = None

        if self.uri and self.user and self.pw and _HAS_NEO4J:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pw))
                self.connected = True
            except Exception as e:
                self.connected = False
                self._driver = None
                self._last_error = f"Connection error: {e}"
        else:
            self._last_error = "Neo4j not configured or neo4j driver not installed."

    def get_schema(self) -> Dict[str, Any]:
        """
        Get comprehensive schema information including labels, relationships, and properties.
        """
        if not self.connected:
            return {"ok": False, "error": "NotConnected", "message": self._last_error}

        # Return cached schema if available
        if self._cached_schema:
            return self._cached_schema

        try:
            schema_info = {
                "labels": [],
                "relationships": [],
                "node_properties": {},
                "relationship_properties": {},
                "patterns": [],
                "constraints": [],
                "indexes": []
            }

            with self._driver.session() as sess:
                # Get all node labels
                labels_result = sess.run("CALL db.labels()")
                schema_info["labels"] = [record["label"] for record in labels_result]

                # Get all relationship types
                rel_result = sess.run("CALL db.relationshipTypes()")
                schema_info["relationships"] = [record["relationshipType"] for record in rel_result]

                # Get node properties for each label
                for label in schema_info["labels"]:
                    props_query = f"""
                    MATCH (n:{label})
                    WITH n LIMIT 100
                    UNWIND keys(n) as key
                    RETURN DISTINCT key
                    """
                    try:
                        props_result = sess.run(props_query)
                        props = [record["key"] for record in props_result]
                        if props:
                            schema_info["node_properties"][label] = props
                    except:
                        pass

                # Get relationship properties
                for rel_type in schema_info["relationships"]:
                    rel_props_query = f"""
                    MATCH ()-[r:{rel_type}]->()
                    WITH r LIMIT 100
                    UNWIND keys(r) as key
                    RETURN DISTINCT key
                    """
                    try:
                        props_result = sess.run(rel_props_query)
                        props = [record["key"] for record in props_result]
                        if props:
                            schema_info["relationship_properties"][rel_type] = props
                    except:
                        pass

                # Get sample patterns
                patterns_query = """
                MATCH (n)-[r]->(m)
                WITH labels(n) as from_labels, type(r) as rel_type, labels(m) as to_labels
                RETURN DISTINCT from_labels, rel_type, to_labels
                LIMIT 20
                """
                patterns_result = sess.run(patterns_query)
                for record in patterns_result:
                    pattern = {
                        "from": record["from_labels"],
                        "relationship": record["rel_type"],
                        "to": record["to_labels"]
                    }
                    schema_info["patterns"].append(pattern)

                # Get constraints
                try:
                    constraints_result = sess.run("SHOW CONSTRAINTS")
                    schema_info["constraints"] = [dict(record) for record in constraints_result]
                except:
                    pass

                # Get indexes
                try:
                    indexes_result = sess.run("SHOW INDEXES")
                    schema_info["indexes"] = [dict(record) for record in indexes_result]
                except:
                    pass

            # Format schema as text for easy reading
            schema_text = self._format_schema_text(schema_info)

            result = {
                "ok": True,
                "schema_info": schema_text,
                "raw_schema": schema_info,
                "message": "Schema extracted successfully"
            }

            # Cache the schema
            self._cached_schema = result

            return result

        except Exception as e:
            return {"ok": False, "error": "SchemaError", "message": str(e)}

    def run_cypher(self, q: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Execute Cypher query with enhanced error reporting and warnings capture.
        """
        if params is None:
            params = {}

        # Validate query safety
        try:
            q = sanitize_query(q)
        except Exception as e:
            return {"ok": False, "error": "UnsafeQuery", "message": str(e)}

        if not self.connected:
            return {"ok": False, "error": "NotConnected", "message": self._last_error}

        try:
            with self._driver.session() as sess:
                # Execute query and capture any warnings
                result = sess.run(q, params)

                # Collect results
                rows = []
                warnings = []

                # Process results and capture notifications
                for record in result:
                    rows.append(record.data())

                # Check for any notifications/warnings
                summary = result.consume()
                if summary.notifications:
                    for notification in summary.notifications:
                        warning_msg = f"{notification.title}: {notification.description}"
                        warnings.append(warning_msg)

                        # Parse specific warning types for better diagnostics
                        if "UnknownLabelWarning" in notification.title:
                            warnings.append("HINT: Use 'CALL db.labels()' to see available node labels")
                        if "UnknownPropertyKeyWarning" in notification.title:
                            warnings.append("HINT: Check schema for available properties")

                # Create DataFrame
                df = pd.DataFrame(rows)

                # Build response
                response = {
                    "ok": True,
                    "rows": len(df),
                    "df": df
                }

                # Add warnings if present
                if warnings:
                    response["warnings"] = warnings
                    response["message"] = "Query executed with warnings. Check 'warnings' field."

                # Add helpful diagnostics for empty results
                if len(df) == 0:
                    response["diagnostics"] = self._generate_diagnostics(q)

                return response

        except Exception as e:
            error_msg = str(e)

            # Enhanced error diagnostics
            diagnostics = {
                "query": q,
                "error": error_msg,
                "suggestions": []
            }

            # Provide specific suggestions based on error type
            if "UnknownLabel" in error_msg or "no such label" in error_msg.lower():
                diagnostics["suggestions"].append("Check available labels with 'CALL db.labels()'")
                diagnostics["suggestions"].append("Verify the node label exists in the schema")

            if "UnknownProperty" in error_msg or "no such property" in error_msg.lower():
                diagnostics["suggestions"].append("Check node properties in the schema")
                diagnostics["suggestions"].append("Use 'MATCH (n:Label) RETURN keys(n) LIMIT 1' to see properties")

            if "SyntaxError" in error_msg:
                diagnostics["suggestions"].append("Review Cypher syntax")
                diagnostics["suggestions"].append("Check for missing colons, parentheses, or brackets")

            return {
                "ok": False,
                "error": "QueryError",
                "message": error_msg,
                "diagnostics": diagnostics
            }

    def get_sample_data(self, label: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get sample data for a specific node label.
        """
        if not self.connected:
            return {"ok": False, "error": "NotConnected", "message": self._last_error}

        query = f"MATCH (n:{label}) RETURN n LIMIT {limit}"
        return self.run_cypher(query)

    def explore_relationships(self, label: str, limit: int = 10) -> Dict[str, Any]:
        """
        Explore relationships connected to a specific node label.
        """
        if not self.connected:
            return {"ok": False, "error": "NotConnected", "message": self._last_error}

        query = f"""
        MATCH (n:{label})-[r]-(m)
        RETURN DISTINCT 
            labels(n) as from_labels,
            type(r) as relationship,
            labels(m) as to_labels,
            count(*) as count
        ORDER BY count DESC
        LIMIT {limit}
        """
        return self.run_cypher(query)

    def _format_schema_text(self, schema_info: Dict) -> str:
        """
        Format schema information as readable text.
        """
        lines = []

        lines.append("=== GRAPH SCHEMA ===")
        lines.append("")

        # Node labels
        lines.append("NODE LABELS:")
        for label in schema_info.get("labels", []):
            props = schema_info.get("node_properties", {}).get(label, [])
            if props:
                lines.append(f"  • {label}")
                lines.append(f"    Properties: {', '.join(props)}")
            else:
                lines.append(f"  • {label} (no properties found)")

        lines.append("")

        # Relationships
        lines.append("RELATIONSHIP TYPES:")
        for rel in schema_info.get("relationships", []):
            props = schema_info.get("relationship_properties", {}).get(rel, [])
            if props:
                lines.append(f"  • {rel}")
                lines.append(f"    Properties: {', '.join(props)}")
            else:
                lines.append(f"  • {rel}")

        lines.append("")

        # Patterns
        if schema_info.get("patterns"):
            lines.append("COMMON PATTERNS:")
            for pattern in schema_info["patterns"][:10]:
                from_str = ":".join(pattern["from"]) if pattern["from"] else "?"
                to_str = ":".join(pattern["to"]) if pattern["to"] else "?"
                lines.append(f"  ({from_str})-[:{pattern['relationship']}]->({to_str})")

        return "\n".join(lines)

    def _generate_diagnostics(self, query: str) -> Dict[str, Any]:
        """
        Generate diagnostics for failed or empty queries.
        """
        diagnostics = {
            "query_analysis": [],
            "suggestions": []
        }

        # Check for common issues
        if "=" in query and "CONTAINS" not in query:
            diagnostics["suggestions"].append(
                "Consider using CONTAINS for partial matches instead of exact equality"
            )

        if "WHERE" in query:
            diagnostics["suggestions"].append(
                "Try removing WHERE clause to see if any data exists"
            )

        # Extract labels and relationships from query
        import re

        # Find node labels
        label_pattern = r'\([\w]*:(\w+)[^\)]*\)'
        labels = re.findall(label_pattern, query)
        if labels:
            diagnostics["query_analysis"].append(f"Query references labels: {', '.join(set(labels))}")

        # Find relationships
        rel_pattern = r'\[:(\w+)\]'
        relationships = re.findall(rel_pattern, query)
        if relationships:
            diagnostics["query_analysis"].append(f"Query references relationships: {', '.join(set(relationships))}")

        return diagnostics

    def close(self):
        """
        Close the database connection.
        """
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass