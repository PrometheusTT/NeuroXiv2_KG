"""
Enhanced Neo4j executor that properly handles complex queries.
"""

import logging
import re
import time
from typing import Any, Dict, Optional, List

from neo4j import GraphDatabase
try:
    from neo4j.exceptions import TransientError
except Exception:  # pragma: no cover
    class TransientError(Exception):
        pass

logger = logging.getLogger(__name__)


class EnhancedNeo4jExec:
    """
    Enhanced Neo4j executor with proper complex query handling.
    """
    def __init__(self, uri: str, user: str, pwd: str, database: str = "neo4j",
                 timeout_s: int = 25, max_retries: int = 3):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=10)
        self.database = database
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def close(self):
        try:
            self.driver.close()
        except Exception:
            pass

    @staticmethod
    def _is_complex_query(q: str) -> bool:
        """Check if query has complex structures that shouldn't have naive LIMIT appending."""
        q_upper = q.upper()
        complex_patterns = [
            'CALL {',  # Subqueries
            'UNION',   # Union queries
            'WITH ',   # Intermediate processing
            'OPTIONAL MATCH',  # Optional patterns
            'FOREACH',  # Loops
        ]
        return any(pattern in q_upper for pattern in complex_patterns)

    @staticmethod
    def _smart_ensure_limit(q: str, default_limit: int = 100) -> str:
        """Intelligently add LIMIT to queries without breaking complex structures."""
        # If already has LIMIT, don't modify
        if re.search(r"\bLIMIT\s+\d+", q, re.I):
            return q

        # For complex queries, don't auto-add LIMIT
        if EnhancedNeo4jExec._is_complex_query(q):
            logger.debug("Complex query detected, skipping auto-LIMIT")
            return q

        # For simple queries, add LIMIT at the end
        return f"{q.rstrip()}\nLIMIT {default_limit}"

    def explain_ok(self, q: str) -> bool:
        """Test if query is valid with EXPLAIN."""
        try:
            with self.driver.session(database=self.database) as s:
                s.run("EXPLAIN " + q, parameters=None, timeout=self.timeout_s)
            return True
        except Exception as e:
            logger.debug(f"EXPLAIN failed: {e}")
            return False

    def run(self, q: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute query with smart handling of complex structures."""
        original_query = q
        q = self._smart_ensure_limit(q)

        if not self.explain_ok(q):
            # For complex queries, try without ORDER BY
            if self._is_complex_query(q):
                q_no_order = re.sub(r"\bORDER BY .*?(?=LIMIT|UNION|WITH|$)", "", q, flags=re.I | re.S)
                if self.explain_ok(q_no_order):
                    q = q_no_order
                    logger.debug("Removed ORDER BY from complex query")
            else:
                # For simple queries, strip ORDER BY
                q = re.sub(r"\bORDER BY .*?(?=LIMIT|$)", "", q, flags=re.I | re.S)

        for attempt in range(self.max_retries):
            try:
                start = time.time()
                with self.driver.session(database=self.database) as s:
                    res = s.run(q, parameters=(params or {}), timeout=self.timeout_s)
                    data = [r.data() for r in res]
                dur = time.time() - start
                return {
                    "success": True,
                    "rows": len(data),
                    "data": data,
                    "t": dur,
                    "query": q,
                    "original_query": original_query
                }
            except TransientError as e:
                backoff = 2 ** attempt
                logger.warning(f"TransientError; retry in {backoff}s: {e}")
                time.sleep(backoff)
            except Exception as e:
                logger.error(f"Neo4j query failed: {e}")
                break

        return {
            "success": False,
            "rows": 0,
            "data": [],
            "t": 0.0,
            "query": q,
            "original_query": original_query
        }

    def run_direct(self, q: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute query directly without any modifications."""
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                with self.driver.session(database=self.database) as s:
                    res = s.run(q, parameters=(params or {}), timeout=self.timeout_s)
                    data = [r.data() for r in res]
                dur = time.time() - start
                return {
                    "success": True,
                    "rows": len(data),
                    "data": data,
                    "t": dur,
                    "query": q
                }
            except TransientError as e:
                backoff = 2 ** attempt
                logger.warning(f"TransientError; retry in {backoff}s: {e}")
                time.sleep(backoff)
            except Exception as e:
                logger.error(f"Neo4j query failed: {e}")
                break

        return {
            "success": False,
            "rows": 0,
            "data": [],
            "t": 0.0,
            "query": q
        }