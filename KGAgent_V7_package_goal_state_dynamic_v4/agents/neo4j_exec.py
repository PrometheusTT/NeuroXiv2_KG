# Enhanced neo4j_exec.py with comprehensive schema extraction including relationship properties

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
    Enhanced Neo4j executor with deep schema extraction focusing on relationship properties.
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
        Get comprehensive schema information with special focus on relationship properties.
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
                "sample_patterns": [],  # Added for better pattern understanding
                "constraints": [],
                "indexes": [],
                "key_insights": []  # Added to highlight important findings
            }

            with self._driver.session() as sess:
                # Get all node labels
                labels_result = sess.run("CALL db.labels()")
                schema_info["labels"] = [record["label"] for record in labels_result]

                # Get all relationship types
                rel_result = sess.run("CALL db.relationshipTypes()")
                schema_info["relationships"] = [record["relationshipType"] for record in rel_result]

                # Get node properties for each label with better sampling
                for label in schema_info["labels"]:
                    props_query = f"""
                    MATCH (n:{label})
                    WITH n LIMIT 1000
                    UNWIND keys(n) as key
                    WITH key, count(*) as frequency
                    RETURN DISTINCT key
                    ORDER BY frequency DESC
                    """
                    try:
                        props_result = sess.run(props_query)
                        props = [record["key"] for record in props_result]
                        if props:
                            schema_info["node_properties"][label] = props
                    except:
                        pass

                # Get relationship properties - CRITICAL for optimization
                print("[Neo4j] Extracting relationship properties (critical for optimization)...")
                for rel_type in schema_info["relationships"]:
                    rel_props_query = f"""
                    MATCH ()-[r:{rel_type}]->()
                    WITH r LIMIT 1000
                    UNWIND keys(r) as key
                    WITH key, count(*) as frequency
                    RETURN DISTINCT key
                    ORDER BY frequency DESC
                    """
                    try:
                        props_result = sess.run(rel_props_query)
                        props = [record["key"] for record in props_result]
                        if props:
                            schema_info["relationship_properties"][rel_type] = props

                            # Highlight important properties
                            for prop in props:
                                if any(k in prop.lower() for k in ["pct", "percent", "proportion", "count", "weight"]):
                                    insight = f"Relationship {rel_type} has property '{prop}' - likely a pre-computed metric"
                                    schema_info["key_insights"].append(insight)
                                    print(f"[Neo4j] Found key property: {rel_type}.{prop}")
                    except Exception as e:
                        print(f"[Neo4j] Error getting properties for {rel_type}: {e}")

                # Get detailed sample patterns with counts
                patterns_query = """
                MATCH (n)-[r]->(m)
                WITH labels(n) as from_labels, type(r) as rel_type, labels(m) as to_labels,
                     count(*) as pattern_count
                RETURN from_labels, rel_type, to_labels, pattern_count
                ORDER BY pattern_count DESC
                LIMIT 30
                """
                patterns_result = sess.run(patterns_query)
                for record in patterns_result:
                    pattern = {
                        "from_labels": record["from_labels"],
                        "relationship": record["rel_type"],
                        "to_labels": record["to_labels"],
                        "count": record["pattern_count"]
                    }
                    schema_info["sample_patterns"].append(pattern)

                    # Simplified version for backward compatibility
                    schema_info["patterns"].append({
                        "from": record["from_labels"],
                        "relationship": record["rel_type"],
                        "to": record["to_labels"]
                    })

                # Special query to check HAS_SUBCLASS relationship properties specifically
                # This is critical for Car3 analysis
                if "HAS_SUBCLASS" in schema_info["relationships"]:
                    special_query = """
                    MATCH (r:Region)-[rel:HAS_SUBCLASS]->(s:Subclass)
                    WITH rel, r, s LIMIT 5
                    RETURN r.name as region, s.name as subclass, keys(rel) as rel_props, rel
                    """
                    try:
                        special_result = sess.run(special_query)
                        samples = []
                        for record in special_result:
                            sample = {
                                "region": record["region"],
                                "subclass": record["subclass"],
                                "relationship_properties": record["rel_props"]
                            }
                            # Try to get actual values for key properties
                            rel_data = record["rel"]
                            if rel_data and "pct_cells" in rel_data:
                                sample["pct_cells_value"] = rel_data["pct_cells"]
                            samples.append(sample)

                        if samples:
                            schema_info["has_subclass_samples"] = samples
                            print(f"[Neo4j] HAS_SUBCLASS relationship samples: {samples[:2]}")
                    except Exception as e:
                        print(f"[Neo4j] Error getting HAS_SUBCLASS samples: {e}")

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
                "message": "Schema extracted successfully",
                "key_insights": schema_info.get("key_insights", [])
            }

            # Cache the schema
            self._cached_schema = result

            return result

        except Exception as e:
            return {"ok": False, "error": "SchemaError", "message": str(e)}

    def run_cypher(self, q: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Execute Cypher query with enhanced error reporting and automatic optimization suggestions.
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
                # Log the query for debugging
                print(f"[Neo4j] Executing query: {q[:200]}...")

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
                        print(f"[Neo4j] Warning: {warning_msg}")

                        # Parse specific warning types for better diagnostics
                        if "UnknownPropertyKeyWarning" in str(notification):
                            # Extract the missing property name
                            import re
                            match = re.search(r"missing property name is: (\w+)", notification.description)
                            if match:
                                prop_name = match.group(1)
                                warnings.append(
                                    f"HINT: Property '{prop_name}' doesn't exist. Check schema for available properties.")

                                # Suggest using relationship properties if applicable
                                if prop_name in ["car3_count", "total_count", "proportion"]:
                                    warnings.append(
                                        "HINT: Looking for proportions? Check relationship properties like 'pct_cells' on HAS_SUBCLASS")

                # Create DataFrame
                df = pd.DataFrame(rows)

                # Build response
                response = {
                    "ok": True,
                    "rows": len(df),
                    "df": df,
                    "columns": list(df.columns) if not df.empty else []
                }

                # Add column info for better understanding
                if not df.empty:
                    response["column_info"] = {col: str(df[col].dtype) for col in df.columns}

                    # Add sample data for small result sets
                    if len(df) <= 5:
                        response["sample_data"] = df.to_dict('records')

                # Add warnings if present
                if warnings:
                    response["warnings"] = warnings
                    response["message"] = "Query executed with warnings. Check 'warnings' field."

                # Add helpful diagnostics for empty results
                if len(df) == 0:
                    response["diagnostics"] = self._generate_diagnostics(q)
                    response["suggestions"] = self._generate_query_suggestions(q)

                print(f"[Neo4j] Query returned {len(df)} rows")
                return response

        except Exception as e:
            error_msg = str(e)
            print(f"[Neo4j] Query error: {error_msg}")

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
                # Extract property name if possible
                import re
                match = re.search(r"property[:\s]+(\w+)", error_msg.lower())
                if match:
                    prop = match.group(1)
                    if prop in ["car3_count", "total_count", "proportion", "percentage"]:
                        diagnostics["suggestions"].append(
                            "This property doesn't exist on nodes. Check if it's a relationship property instead."
                        )
                        diagnostics["suggestions"].append(
                            "For proportions, try: MATCH (r:Region)-[rel:HAS_SUBCLASS]->(s:Subclass) RETURN rel.pct_cells"
                        )

                diagnostics["suggestions"].append("Check node/relationship properties in the schema")
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

    def _format_schema_text(self, schema_info: Dict) -> str:
        """
        Format schema information as readable text with emphasis on relationship properties.
        """
        lines = []

        lines.append("=== GRAPH SCHEMA ===")
        lines.append("")

        # Key insights first
        if schema_info.get("key_insights"):
            lines.append("*** KEY INSIGHTS (IMPORTANT!) ***")
            for insight in schema_info["key_insights"]:
                lines.append(f"  ⚠️  {insight}")
            lines.append("")

        # Node labels
        lines.append("NODE LABELS:")
        for label in schema_info.get("labels", []):
            props = schema_info.get("node_properties", {}).get(label, [])
            if props:
                lines.append(f"  • {label}")
                lines.append(f"    Properties: {', '.join(props[:10])}")  # Limit to first 10
            else:
                lines.append(f"  • {label} (no properties found)")

        lines.append("")

        # Relationships with properties emphasized
        lines.append("RELATIONSHIP TYPES (WITH PROPERTIES):")
        for rel in schema_info.get("relationships", []):
            props = schema_info.get("relationship_properties", {}).get(rel, [])
            if props:
                lines.append(f"  • {rel}")
                lines.append(f"    >>> PROPERTIES: {', '.join(props)} <<<")
                # Highlight important properties
                for prop in props:
                    if "pct" in prop.lower() or "percent" in prop.lower():
                        lines.append(f"      ⚠️  {prop} - Pre-computed percentage/proportion!")
            else:
                lines.append(f"  • {rel} (no properties)")

        lines.append("")

        # Sample patterns with counts
        if schema_info.get("sample_patterns"):
            lines.append("COMMON PATTERNS (WITH COUNTS):")
            for pattern in schema_info["sample_patterns"][:15]:
                from_str = ":".join(pattern["from_labels"]) if pattern["from_labels"] else "?"
                to_str = ":".join(pattern["to_labels"]) if pattern["to_labels"] else "?"
                count = pattern.get("count", 0)
                rel = pattern["relationship"]

                pattern_str = f"  ({from_str})-[:{rel}]->({to_str}) [{count} instances]"

                # Add relationship properties if available
                if rel in schema_info.get("relationship_properties", {}):
                    props = schema_info["relationship_properties"][rel][:3]  # First 3 props
                    pattern_str += f" | Props: {', '.join(props)}"

                lines.append(pattern_str)

        # HAS_SUBCLASS samples if available
        if schema_info.get("has_subclass_samples"):
            lines.append("")
            lines.append("*** HAS_SUBCLASS RELATIONSHIP EXAMPLES (CRITICAL FOR Car3 ANALYSIS) ***")
            for sample in schema_info["has_subclass_samples"][:3]:
                lines.append(f"  Region: {sample['region']} -> Subclass: {sample['subclass']}")
                lines.append(f"    Relationship properties: {', '.join(sample['relationship_properties'])}")
                if "pct_cells_value" in sample:
                    lines.append(f"    pct_cells value: {sample['pct_cells_value']}")

        return "\n".join(lines)

    def _generate_diagnostics(self, query: str) -> Dict[str, Any]:
        """
        Generate diagnostics for failed or empty queries with schema-aware suggestions.
        """
        diagnostics = {
            "query_analysis": [],
            "suggestions": [],
            "schema_hints": []
        }

        # Check for common issues
        if "car3_count" in query.lower() or "total_count" in query.lower():
            diagnostics["schema_hints"].append(
                "Properties like 'car3_count' or 'total_count' likely don't exist on nodes."
            )
            diagnostics["suggestions"].append(
                "For Car3 proportions, use: MATCH (r:Region)-[rel:HAS_SUBCLASS]->(s:Subclass) WHERE s.name CONTAINS 'Car3' RETURN rel.pct_cells"
            )

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

            # Check if we have schema info for these relationships
            if self._cached_schema and "raw_schema" in self._cached_schema:
                schema = self._cached_schema["raw_schema"]
                for rel in relationships:
                    if rel in schema.get("relationship_properties", {}):
                        props = schema["relationship_properties"][rel]
                        diagnostics["schema_hints"].append(
                            f"Relationship {rel} has properties: {', '.join(props[:5])}"
                        )

        return diagnostics

    def _generate_query_suggestions(self, failed_query: str) -> List[str]:
        """
        Generate alternative query suggestions based on the failed query and schema.
        """
        suggestions = []

        # If looking for Car3 data
        if "car3" in failed_query.lower():
            suggestions.append(
                "MATCH (r:Region)-[rel:HAS_SUBCLASS]->(s:Subclass) WHERE s.name CONTAINS 'Car3' RETURN r.name, s.name, rel.pct_cells ORDER BY rel.pct_cells DESC LIMIT 10"
            )
            suggestions.append(
                "MATCH (s:Subclass) WHERE s.name CONTAINS 'Car3' RETURN s.name, keys(s) LIMIT 5"
            )
            suggestions.append(
                "MATCH (n) WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS 'Car3') RETURN labels(n), n.name LIMIT 10"
            )

        return suggestions

    def close(self):
        """
        Close the database connection.
        """
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass