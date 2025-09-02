
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KGAgent V7 - runnable entrypoint

Usage:
  python run_v7.py --question "Analyze the projection patterns of Car3 subclass neurons"
  # env: OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""
import argparse
import json
import logging
import os
from agent_v7.agent_v7 import KGAgentV7

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("run_v7")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="Natural language question.")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--planner_model", default=os.getenv("PLANNER_MODEL", "gpt-5"))
    ap.add_argument("--summarizer_model", default=os.getenv("SUMMARIZER_MODEL", "gpt-4o"))
    return ap.parse_args()


def main():
    args = parse_args()
    agent = KGAgentV7(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://100.88.72.32:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        openai_api_key='',
        planner_model=args.planner_model,
        summarizer_model=args.summarizer_model
    )
    out = agent.answer(args.question, max_rounds=args.rounds)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
