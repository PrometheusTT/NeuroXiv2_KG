#!/usr/bin/env python3
"""
Comprehensive test and demonstration of the Enhanced Reasoning Agent.
Showcases think-act-observe-reflect with KG guidance, self-evaluation, and dynamic orchestration.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

from agent_v7.enhanced_reasoning_agent import EnhancedReasoningAgent


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'-' * 60}")
    print(f"🔍 {title}")
    print(f"{'-' * 60}")


def demonstrate_enhanced_reasoning():
    """Demonstrate the enhanced reasoning capabilities."""
    load_dotenv()

    print_section("🧠 ENHANCED REASONING AGENT DEMONSTRATION")
    print("This demonstration showcases the advanced CoT + KG reasoning capabilities:")
    print("• Think-Act-Observe-Reflect loops")
    print("• Knowledge Graph guided exploration")
    print("• Self-evaluation and correction")
    print("• Dynamic tool orchestration")
    print("• Memory and pattern learning")

    # Initialize the enhanced agent
    print_subsection("Initializing Enhanced Reasoning Agent")
    try:
        agent = EnhancedReasoningAgent(
            neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
            neo4j_pwd=os.getenv('NEO4J_PASSWORD', 'password'),
            database='neo4j',
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            planner_model="gpt-4",
            summarizer_model="gpt-4"
        )
        print("✅ Enhanced reasoning agent initialized successfully")
        print(f"   • Tools available: {len(agent.tools)}")
        print(f"   • KG knowledge initialized: {agent.kg_reasoning.graph_structure is not None}")
        print(f"   • Self-evaluation enabled: {agent.self_evaluation is not None}")
        print(f"   • Dynamic orchestration active: {agent.orchestrator is not None}")

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return

    # Test questions of increasing complexity
    test_questions = [
        {
            "question": "Find regions with similar morphological characteristics but different molecular profiles.",
            "description": "Basic morphological-molecular analysis",
            "expected_capabilities": ["morphology analysis", "molecular comparison", "KG exploration"]
        },
        {
            "question": "What are the most significant pairs of brain regions that exhibit morphological convergence despite molecular divergence, and what does this tell us about brain organization?",
            "description": "Complex analytical question requiring deep reasoning",
            "expected_capabilities": ["multi-step reasoning", "pattern recognition", "biological interpretation"]
        },
        {
            "question": "Analyze the relationship between regional connectivity patterns and neurotransmitter diversity across the brain, identifying any unexpected associations.",
            "description": "Advanced network analysis with hypothesis generation",
            "expected_capabilities": ["network analysis", "statistical reasoning", "hypothesis formation"]
        }
    ]

    for i, test_case in enumerate(test_questions, 1):
        print_section(f"🎯 TEST CASE {i}: {test_case['description'].upper()}", "=")
        print(f"Question: {test_case['question']}")
        print(f"Expected capabilities: {', '.join(test_case['expected_capabilities'])}")

        try:
            # Run the enhanced reasoning
            print_subsection("Running Enhanced Reasoning Process")
            result = agent.reason_through_question(test_case['question'], max_iterations=8)

            # Display comprehensive results
            print_subsection("Reasoning Summary")
            summary = result['reasoning_summary']
            print(f"📊 Iterations completed: {summary['total_iterations']}")
            print(f"💭 Thoughts generated: {summary['thoughts_generated']}")
            print(f"👁️  Observations made: {summary['observations_made']}")
            print(f"🤔 Reflections completed: {summary['reflections_completed']}")
            print(f"🔧 Corrections applied: {summary['corrections_applied']}")

            if summary['working_hypotheses']:
                print(f"🔬 Working hypotheses: {', '.join(summary['working_hypotheses'][:3])}")
            if summary['confirmed_facts']:
                print(f"✅ Confirmed facts: {', '.join(summary['confirmed_facts'][:3])}")

            print_subsection("Quality Assessment")
            quality = result['quality_assessment']
            print(f"🎯 Overall quality: {quality['overall_quality']:.3f}")
            print(f"🧠 Logical consistency: {quality['logical_consistency']:.3f}")
            print(f"📈 Evidence strength: {quality['evidence_strength']:.3f}")
            print(f"🏗️  Reasoning depth: {quality['reasoning_depth']:.3f}")
            print(f"🎭 Coherence score: {quality['coherence_score']:.3f}")
            print(f"⚖️  Calibrated confidence: {quality['calibrated_confidence']:.3f}")

            if quality['errors_detected'] > 0:
                print(f"⚠️  Errors detected: {quality['errors_detected']} ({', '.join(quality['error_types'])})")

            print_subsection("Methodology Used")
            methodology = result['methodology']
            print(f"🔄 Reasoning approach: {methodology['reasoning_approach']}")
            print(f"🛠️  Tools used: {', '.join(methodology['tools_used'])}")
            print(f"🗺️  KG guidance steps: {methodology['kg_guidance_steps']}")
            print(f"🔧 Self-corrections: {methodology['self_corrections']}")
            print(f"🎼 Orchestration: {methodology['adaptive_orchestration']}")

            print_subsection("Evidence Trail")
            evidence = result['evidence_trail']
            print(f"🔍 Observations summary:")
            for j, obs in enumerate(evidence['observations'][:5], 1):
                status = "✅" if obs['success'] else "❌"
                print(f"   {j}. {status} {len(obs['insights'])} insights, {len(obs['surprises'])} surprises")

            if evidence['key_insights']:
                print(f"💡 Key insights: {', '.join(evidence['key_insights'][:3])}...")

            print_subsection("Final Answer")
            answer = result['answer']
            # Truncate long answers for display
            display_answer = answer[:500] + "..." if len(answer) > 500 else answer
            print(f"📝 {display_answer}")

            print_subsection("Limitations Identified")
            limitations = result['limitations']
            if limitations['knowledge_gaps']:
                print(f"❓ Knowledge gaps: {', '.join(limitations['knowledge_gaps'][:2])}")
            if limitations['reasoning_errors']:
                print(f"⚠️  Reasoning errors: {', '.join(limitations['reasoning_errors'][:2])}")
            print(f"🎯 Confidence level: {limitations['confidence_level']:.3f}")

            # Demonstrate specific enhanced capabilities
            if i == 1:  # First test case
                print_subsection("🔬 Enhanced Capabilities Demonstration")

                # Show KG guidance in action
                if result.get('reasoning_trace', {}).get('kg_guidance_history'):
                    kg_guidance = result['reasoning_trace']['kg_guidance_history'][0]
                    if hasattr(kg_guidance, 'central_entities'):
                        print(f"🗺️  KG identified central entities: {', '.join(kg_guidance.central_entities[:3])}")
                    if hasattr(kg_guidance, 'information_gaps'):
                        print(f"❓ KG identified information gaps: {', '.join(kg_guidance.information_gaps[:2])}")

                # Show self-evaluation in action
                corrections = result['reasoning_trace'].get('corrections_applied', [])
                if corrections:
                    print(f"🔧 Self-evaluation applied corrections:")
                    for correction in corrections[:2]:
                        print(f"   • {correction.get('type', 'Unknown')}: {correction.get('description', 'Applied')}")

                # Show quality progression
                quality_progression = result['reasoning_trace'].get('quality_progression', [])
                if len(quality_progression) > 1:
                    initial_quality = quality_progression[0].overall_quality
                    final_quality = quality_progression[-1].overall_quality
                    improvement = final_quality - initial_quality
                    print(f"📈 Quality improvement: {initial_quality:.3f} → {final_quality:.3f} (+{improvement:.3f})")

        except Exception as e:
            print(f"❌ Error during reasoning: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 80)

    # Demonstrate learning and adaptation
    print_section("🧠 LEARNING AND ADAPTATION DEMONSTRATION")
    print("The enhanced agent learns from each interaction and adapts its approach:")

    try:
        # Show orchestrator learning
        orchestrator = agent.orchestrator
        if orchestrator.execution_history:
            print(f"📚 Total executions learned from: {len(orchestrator.execution_history)}")

            # Show tool reliability adaptation
            print("🛠️  Tool reliability learning:")
            for tool_name, capability in list(orchestrator.tool_capabilities.items())[:5]:
                print(f"   • {tool_name}: reliability {capability.reliability:.3f}")

        # Show memory system
        memory = agent.cot_reasoning.memory
        if memory.episodic_memory:
            print(f"🧠 Episodic memories stored: {len(memory.episodic_memory)}")
            if memory.semantic_memory:
                print(f"🎯 Semantic patterns learned: {len(memory.semantic_memory)}")

    except Exception as e:
        print(f"⚠️  Learning demonstration limited: {e}")

    print_section("✅ DEMONSTRATION COMPLETED")
    print("The Enhanced Reasoning Agent has successfully demonstrated:")
    print("• ✅ Think-Act-Observe-Reflect reasoning loops")
    print("• ✅ Knowledge Graph guided exploration")
    print("• ✅ Self-evaluation and error correction")
    print("• ✅ Dynamic tool orchestration and selection")
    print("• ✅ Quality assessment and confidence calibration")
    print("• ✅ Memory-based learning and adaptation")
    print("• ✅ Comprehensive evidence trail documentation")

    # Clean up
    agent.close()


def run_interactive_demo():
    """Run an interactive demonstration where users can ask questions."""
    load_dotenv()

    print_section("🤖 INTERACTIVE ENHANCED REASONING DEMO")

    try:
        agent = EnhancedReasoningAgent(
            neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
            neo4j_pwd=os.getenv('NEO4J_PASSWORD', 'password'),
            database='neo4j',
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )

        print("🤖 Enhanced Reasoning Agent ready for interactive questions!")
        print("💡 Try questions about brain regions, morphology, molecular profiles, etc.")
        print("🛑 Type 'exit' to quit")

        while True:
            print("\n" + "-" * 60)
            question = input("❓ Your question: ").strip()

            if question.lower() in ['exit', 'quit', 'stop']:
                break

            if not question:
                print("Please enter a question.")
                continue

            print(f"\n🧠 Processing: {question}")
            print("⏳ Running enhanced reasoning...")

            try:
                result = agent.reason_through_question(question, max_iterations=6)

                print(f"\n📊 Reasoning completed:")
                print(f"   • Quality: {result['quality_assessment']['overall_quality']:.3f}")
                print(f"   • Confidence: {result['quality_assessment']['calibrated_confidence']:.3f}")
                print(f"   • Iterations: {result['reasoning_summary']['total_iterations']}")

                print(f"\n📝 Answer:")
                print(result['answer'])

                if result['limitations']['knowledge_gaps']:
                    print(f"\n❓ Knowledge gaps identified: {', '.join(result['limitations']['knowledge_gaps'][:2])}")

            except Exception as e:
                print(f"❌ Error: {e}")

        agent.close()
        print("\n👋 Interactive demo ended.")

    except Exception as e:
        print(f"❌ Failed to start interactive demo: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Reasoning Agent Test")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo",
                        help="Run demonstration or interactive mode")

    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive_demo()
    else:
        demonstrate_enhanced_reasoning()