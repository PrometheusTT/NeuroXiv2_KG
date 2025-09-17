#!/usr/bin/env python3
"""
Enhanced KG Agent V7 with full CoT reasoning, planning, and advanced tool capabilities.
This combines the power of the enhanced tools with the transparent reasoning process of the standard agent.
"""

import argparse
import json
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

from agent_v7.agent_v7 import KGAgentV7

# Configure logging to show the reasoning process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def display_execution_results(results, question: str):
    """Display query execution results in a clear, readable format"""
    print(f"\n🔍 Query Execution Results for: '{question}'")
    print("=" * 80)

    for result in results:
        idx = result.get('idx', 0)
        purpose = result.get('purpose', 'Unknown purpose')
        query = result.get('query', '')
        success = result.get('success', False)
        rows = result.get('rows', 0)
        execution_time = result.get('t', 0)

        status_icon = "✅" if success else "❌"
        print(f"\n{status_icon} Attempt {idx}: {purpose}")
        print(f"   Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"   Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"   Rows returned: {rows}")
        print(f"   Execution time: {execution_time:.3f}s")

        # Show sample data for successful queries
        if success and 'data' in result:
            data = result['data']
            if isinstance(data, dict):
                if 'results' in data and 'pairs' in data['results']:
                    # Enhanced analysis results
                    pairs = data['results']['pairs']
                    if pairs:
                        print(f"   📊 Found {len(pairs)} significant region pairs:")
                        for i, pair in enumerate(pairs[:3], 1):  # Show first 3
                            print(f"      {i}. {pair.get('region1', 'N/A')} ↔ {pair.get('region2', 'N/A')}")
                            print(f"         Morphological similarity: {pair.get('morphological_similarity', 0):.4f}")
                elif isinstance(data, list) and data:
                    # Regular query results
                    print(f"   📋 Sample data (first 3 rows):")
                    for i, row in enumerate(data[:3], 1):
                        row_str = str(row)[:80] + ("..." if len(str(row)) > 80 else "")
                        print(f"      {i}. {row_str}")
            elif isinstance(data, list) and data:
                print(f"   📋 Sample data (first 3 rows):")
                for i, row in enumerate(data[:3], 1):
                    row_str = str(row)[:80] + ("..." if len(str(row)) > 80 else "")
                    print(f"      {i}. {row_str}")


def display_tool_calls(tool_calls):
    """Display tool calls in a readable format"""
    if not tool_calls:
        return

    print(f"\n🛠️  Tool Usage Summary")
    print("=" * 50)

    for i, call in enumerate(tool_calls, 1):
        tool_name = call.get('name', 'Unknown')
        args = call.get('arguments', {})
        result = call.get('result', {})

        print(f"\n{i}. Tool: {tool_name}")
        print(f"   Arguments: {json.dumps(args, indent=2) if args else 'None'}")

        if isinstance(result, dict):
            if 'error' in result:
                print(f"   ❌ Result: ERROR - {result['error']}")
            elif 'value' in result:
                print(f"   ✅ Result: {result['value']}")
            elif 'results' in result:
                print(f"   ✅ Result: Found {len(result.get('results', {}).get('pairs', []))} results")
            else:
                result_str = str(result)[:100] + ("..." if len(str(result)) > 100 else "")
                print(f"   ✅ Result: {result_str}")
        else:
            result_str = str(result)[:100] + ("..." if len(str(result)) > 100 else "")
            print(f"   ✅ Result: {result_str}")


def display_reflection(reflection_content: str, round_num: int):
    """Display reflection content in a readable format"""
    print(f"\n🤔 Reflection & Analysis - Round {round_num}")
    print("=" * 60)
    print(reflection_content)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Enhanced KG Agent V7 with full reasoning")
    parser.add_argument("--question", required=True, help="Question to analyze")
    parser.add_argument("--rounds", type=int, default=3, help="Max reflection rounds (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize the enhanced agent with correct model names
    agent = KGAgentV7(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_pwd=os.getenv('NEO4J_PASSWORD', 'password'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        planner_model='gpt-5',  # Use gpt-5 without temperature parameter
        summarizer_model='gpt-4o'
    )

    try:
        print("🧠 Enhanced KG Agent V7 - Full Reasoning Mode")
        print("=" * 70)
        print(f"Question: {args.question}")
        print(f"Max rounds: {args.rounds}")
        print()

        # Step 1: Planning Phase
        print("📋 Planning Phase")
        print("-" * 30)
        plan = agent._plan(args.question)

        print(f"Analysis Plan: {plan.get('analysis_plan', 'Standard analysis')}")
        print(f"Planned Cypher Attempts: {len(plan.get('cypher_attempts', []))}")

        for i, attempt in enumerate(plan.get('cypher_attempts', []), 1):
            purpose = attempt.get('purpose', 'Unknown')
            query_preview = attempt.get('query', '')[:80] + ('...' if len(attempt.get('query', '')) > 80 else '')
            print(f"  {i}. {purpose}")
            print(f"     Query: {query_preview}")

        # Step 2: Execution Phase
        print("\n⚡ Execution Phase")
        print("-" * 30)
        results = agent._execute_attempts(plan.get('cypher_attempts', []))
        display_execution_results(results, args.question)

        # Step 3: Reflection and Iteration (simplified to avoid timeouts)
        print(f"\n🔄 Reflection & Analysis")
        print("-" * 30)

        all_results = results.copy()  # Start with initial results

        # Only do reflection if we have successful results
        if any(r.get('success', False) for r in results):
            try:
                print("🤔 Analyzing initial results...")

                # Simple reflection context
                context = {
                    "question": args.question,
                    "results": results,
                    "round": 1
                }

                # Try reflection but limit complexity
                reflection = agent._reflect_and_plan_next(context)
                if reflection.get('analysis'):
                    display_reflection(reflection.get('analysis'), 1)

                # Only do one additional round if explicitly requested and results are insufficient
                if args.rounds > 1 and reflection.get('continue', False) and reflection.get('next_attempts'):
                    next_attempts = reflection['next_attempts'][:2]  # Limit to 2 attempts max
                    if next_attempts:
                        print(f"\n⚡ Executing {len(next_attempts)} follow-up queries...")
                        try:
                            new_results = agent._execute_attempts(next_attempts)
                            display_execution_results(new_results, "Follow-up Analysis")
                            all_results.extend(new_results)
                        except Exception as e:
                            print(f"❌ Follow-up execution error: {e}")
                else:
                    print("✅ Initial analysis sufficient - proceeding to summary")

            except Exception as e:
                print(f"❌ Reflection error: {e}")
                print("Proceeding with initial results...")
        else:
            print("⚠️  No successful results to reflect on")

        all_results.extend(results)

        # Step 4: Final Summary
        print(f"\n📝 Final Summary & Interpretation")
        print("=" * 50)

        try:
            # Use the agent's summarizer to create final output
            summary_context = {
                "question": args.question,
                "all_results": all_results,
                "total_rounds": round_num + 1 if 'round_num' in locals() else 1
            }

            final_summary = agent._generate_final_summary(summary_context)
            print(final_summary)

        except Exception as e:
            print(f"❌ Summary generation error: {e}")
            print("\nProviding basic summary based on results:")

            total_successful_queries = sum(1 for r in all_results if r.get('success', False))
            total_rows = sum(r.get('rows', 0) for r in all_results if r.get('success', False))

            print(f"• Successfully executed {total_successful_queries} queries")
            print(f"• Retrieved {total_rows} total data rows")
            print(f"• Completed analysis in {len(all_results)} attempts across multiple rounds")

        print(f"\n✨ Enhanced reasoning analysis completed!")
        print(f"🔧 Tools available: {len(agent.tools)} specialized analysis tools")
        print(f"📊 Enhanced capabilities: Graph metrics, clustering, morphological analysis, molecular profiling")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
    finally:
        agent.close()


if __name__ == "__main__":
    main()