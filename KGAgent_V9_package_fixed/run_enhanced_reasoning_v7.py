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


def display_results(result: Dict[str, Any], verbose: bool = False):
    """Display the agent's results in a clear, readable format"""

    print("\n" + "=" * 70)
    print("📊 ANALYSIS RESULTS")
    print("=" * 70)

    # Display rounds completed
    print(f"\n📈 Reasoning Rounds: {result.get('rounds', 0)}")

    # Display plan
    if 'plan' in result and verbose:
        print(f"\n📋 Analysis Plan:")
        plan = result['plan']
        if 'analysis_plan' in plan:
            print(f"   Strategy: {plan['analysis_plan']}")
        if 'cypher_attempts' in plan:
            print(f"   Planned Queries: {len(plan['cypher_attempts'])}")

    # Display query results
    if 'results' in result:
        print(f"\n⚡ Query Executions:")
        for i, res in enumerate(result['results'], 1):
            status = "✅" if res.get('success') else "❌"
            print(f"   {status} Query {i}: {res.get('purpose', 'Unknown purpose')}")
            print(f"      Rows: {res.get('rows', 0)}, Time: {res.get('t', 0):.3f}s")

            # Show sample data for successful queries
            if res.get('success') and 'data' in res and verbose:
                data = res['data']
                if isinstance(data, dict) and 'results' in data:
                    # Enhanced analysis results
                    if 'pairs' in data['results']:
                        pairs = data['results']['pairs']
                        print(f"      Found {len(pairs)} significant pairs")
                elif isinstance(data, list) and data:
                    print(f"      Sample: {str(data[0])[:100]}...")

    # Display metrics if available
    if 'metrics' in result and result['metrics']:
        print(f"\n📊 Computed Metrics:")
        for metric in result['metrics']:
            print(f"   • {metric.get('type', 'Unknown')}: {metric.get('value', 'N/A')}")

    # Display final answer
    if 'final' in result:
        print(f"\n📝 Final Answer:")
        print("-" * 50)
        # Wrap long text
        final_text = result['final']
        import textwrap
        wrapped = textwrap.fill(final_text, width=68, initial_indent="", subsequent_indent="   ")
        print(wrapped)

    print("\n" + "=" * 70)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Enhanced KG Agent V7 with full reasoning")
    parser.add_argument("--question", required=True, help="Question to analyze")
    parser.add_argument("--rounds", type=int, default=3, help="Max reflection rounds (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--fast", action="store_true", help="Use faster models (gpt-4 instead of gpt-5)")
    parser.add_argument("--provider", choices=["openai", "qwen"], help="Override LLM provider (default from .env)")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    # Override provider if specified
    if args.provider:
        os.environ['LLM_PROVIDER'] = args.provider

    # Initialize the enhanced agent
    current_provider = os.getenv('LLM_PROVIDER', 'qwen')
    if args.fast and current_provider == 'openai':
        planner_model = 'gpt-4'
    elif args.fast and current_provider == 'qwen':
        planner_model = 'qwen2.5-72b-instruct'  # Faster Qwen model
    else:
        planner_model = None  # Use default from LLMClient

    # Use appropriate API key based on provider
    api_key = None
    if current_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
    elif current_provider == 'qwen':
        api_key = os.getenv('DASHSCOPE_API_KEY')

    agent = KGAgentV7(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
        neo4j_pwd=os.getenv('NEO4J_PASSWORD', 'password'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j'),
        openai_api_key=api_key,  # Now supports both OpenAI and Qwen keys
        planner_model=planner_model,
        summarizer_model=os.getenv('SUMMARIZER_MODEL', 'qwen2.5-72b-instruct')
    )

    try:
        print("\n" + "=" * 70)
        print("🧠 ENHANCED KG AGENT V7 - FULL REASONING MODE")
        print("=" * 70)
        print(f"\n❓ Question: {args.question}")
        print(f"⚙️  Max rounds: {args.rounds}")
        print(f"🤖 Provider: {current_provider.upper()}")
        print(f"🧠 Planner Model: {agent.llm.planner_model}")
        print(f"📝 Summarizer Model: {agent.llm.summarizer_model}")
        print(f"🔧 Available tools: {len(agent.tools)}")

        print("\n🚀 Starting analysis...")
        print("-" * 30)

        # Use the main answer method - this is the correct way!
        result = agent.answer(
            question=args.question,
            max_rounds=args.rounds
        )

        # Display results
        display_results(result, verbose=args.verbose)

        print("\n✨ Analysis completed successfully!")

        # Show summary statistics
        if 'results' in result:
            successful_queries = sum(1 for r in result['results'] if r.get('success', False))
            total_rows = sum(r.get('rows', 0) for r in result['results'] if r.get('success', False))
            print(f"\n📊 Summary Statistics:")
            print(f"   • Total queries executed: {len(result['results'])}")
            print(f"   • Successful queries: {successful_queries}")
            print(f"   • Total data rows retrieved: {total_rows}")
            print(f"   • Reasoning rounds: {result.get('rounds', 0)}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        print("\n💡 Tip: Use --debug flag for detailed error information")
    finally:
        print("\n🔒 Closing database connections...")
        agent.close()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()