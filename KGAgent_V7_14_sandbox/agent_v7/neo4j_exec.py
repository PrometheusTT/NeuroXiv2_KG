# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

class Graph:
    def __init__(self, uri: str, user: str, pwd: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.database = database
        self._driver = None

    def _ensure(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))

    @contextmanager
    def session(self):
        self._ensure()
        sess = self._driver.session(database=self.database) if self.database else self._driver.session()
        try:
            yield sess
        finally:
            sess.close()

    def run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.session() as s:
            try:
                res = s.run(cypher, params or {})
                return [dict(r) for r in res]
            except Exception as e:
                print(f"[neo4j_exec] Cypher failed: {e}\nQuery: {cypher}")
                return []
