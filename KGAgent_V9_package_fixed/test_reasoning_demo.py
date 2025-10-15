#!/usr/bin/env python3
"""
Simple demo script to show CoT reasoning process working step-by-step
"""

import os
import json
from dotenv import load_dotenv
from agent_v7.agent_v7 import KGAgentV7

load_dotenv()

def main():
    print("🧠 Testing Enhanced Agent CoT Reasoning")
    print("=" * 60)

    # Initialize agent with working model
    agent = KGAgentV7(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_pwd=os.getenv('NEO4J_PASSWORD', 'password'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        planner_model='gpt-4o',  # Use a model that actually exists
        summarizer_model='gpt-4o'
    )

    try:
        question = "Find 3 brain regions with highest axonal length"
        print(f"Question: {question}")
        print()

        # Step 1: Show Planning
        print("🔍 Step 1: Planning Phase")
        print("-" * 30)
        plan = agent._plan(question)
        print(f"Analysis Plan: {plan.get('analysis_plan', 'N/A')}")
        print(f"Number of query attempts: {len(plan.get('cypher_attempts', []))}")

        for i, attempt in enumerate(plan.get('cypher_attempts', []), 1):
            print(f"\nPlanned Query {i}:")
            print(f"  Purpose: {attempt.get('purpose', 'N/A')}")
            print(f"  Query: {attempt.get('query', 'N/A')[:100]}...")

        # Step 2: Show Execution
        print(f"\n⚡ Step 2: Query Execution")
        print("-" * 30)
        results = agent._execute_attempts(plan.get('cypher_attempts', []))

        for result in results:
            print(f"\nExecution Result {result.get('idx', 0)}:")
            print(f"  Purpose: {result.get('purpose', 'N/A')}")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Rows: {result.get('rows', 0)}")
            print(f"  Time: {result.get('t', 0):.3f}s")

            if result.get('success') and result.get('data'):
                data = result['data']
                if isinstance(data, list) and data:
                    print(f"  Sample data: {data[0]}")
                elif isinstance(data, dict):
                    print(f"  Data type: {type(data).__name__}")

        # Step 3: Show Reflection
        print(f"\n🤔 Step 3: Reflection")
        print("-" * 30)

        context = {
            'question': question,
            'results': results,
            'round': 1
        }

        reflection = agent._reflect_and_plan_next(context)
        print(f"Continue analysis: {reflection.get('continue', False)}")
        print(f"Analysis: {reflection.get('analysis', 'No analysis provided')[:200]}...")

        if reflection.get('next_attempts'):
            print(f"Next attempts planned: {len(reflection['next_attempts'])}")

        # Step 4: Generate Summary
        print(f"\n📝 Step 4: Final Summary")
        print("-" * 30)

        summary_context = {
            'question': question,
            'all_results': results,
            'total_rounds': 1
        }

        summary = agent._generate_final_summary(summary_context)
        print(summary)

        print(f"\n✅ CoT Reasoning Demo Complete!")
        print(f"The agent successfully demonstrated:")
        print(f"  • LLM-driven planning")
        print(f"  • Query execution with results")
        print(f"  • Reflection and analysis")
        print(f"  • Final summary generation")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        agent.close()

if __name__ == "__main__":
    main()