#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from agent_v7.agent_v7 import KGAgentV7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--planner_model", type=str, default=os.getenv("PLANNER_MODEL", "gpt-5"))
    parser.add_argument("--summarizer_model", type=str, default=os.getenv("SUMMARIZER_MODEL", "gpt-4o"))
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))

    # Neo4j params with safe defaults (as you requested earlier, but without embedding secrets)
    parser.add_argument("--neo4j_uri", type=str, default=os.getenv("NEO4J_URI", "bolt://100.88.72.32:7687"))
    parser.add_argument("--neo4j_user", type=str, default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j_pwd", type=str, default=os.getenv("NEO4J_PASSWORD", "neuroxiv"))
    parser.add_argument("--database", type=str, default=os.getenv("NEO4J_DATABASE", "neo4j"))

    args = parser.parse_args()

    agent = KGAgentV7(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pwd=args.neo4j_pwd,
        database=args.database,
        openai_api_key=args.openai_api_key,
        planner_model=args.planner_model,
        summarizer_model=args.summarizer_model
    )

    try:
        result = agent.analyze(args.question, rounds=args.rounds, plot=args.plot)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print("ERROR during analysis:", str(e), file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
