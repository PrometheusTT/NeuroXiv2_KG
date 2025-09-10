import argparse, json, os, time
from agents.goal_state_agent import GoalDrivenAgent
from agents.llm import LLM, TraceLLM
from agents.workspace import Workspace
from agents.argument_graph import ArgumentGraph
from agents.json_utils import safe_save_json, convert_numpy_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="User question / goal")
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--trace", default=None, help="(Optional) path to a JSON trace file to mock LLM responses")
    parser.add_argument("--out", default="run_output.json")
    args = parser.parse_args()

    # Choose LLM backend
    if args.trace:
        llm = TraceLLM.from_file(args.trace)
    else:
        llm = LLM()

    ws = Workspace()
    ag = ArgumentGraph()

    agent = GoalDrivenAgent(llm=llm, workspace=ws, argument_graph=ag)
    result = agent.run(user_goal=args.question, max_steps=args.max_steps)

    payload = {
        "question": args.question,
        "rounds_used": result.get("steps_used"),
        "history": result.get("history"),
        "final_answer": result.get("final_answer"),
        "workspace": ws.to_meta(),
        "argument_graph": ag.to_meta(),
    }

    # Use safe JSON serialization to handle NumPy/Pandas types
    safe_save_json(payload, args.out)

    # Print the result (also safely converted)
    print(json.dumps(convert_numpy_types(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()