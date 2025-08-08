import argparse
from openai import OpenAI
from eval_backchannel import eval_backchannel
from eval_pause_handling import eval_pause_handling
from eval_smooth_turn_taking import eval_smooth_turn_taking
from eval_user_interruption import eval_user_interruption

# For OpenAI API for user interruption evaluation
organization = "YOUR_ORG_ID"
api_key = "YOUR_API_KEY"


def main():
    parser = argparse.ArgumentParser(description="Run evaluation tasks.")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "backchannel",
            "pause_handling",
            "smooth_turn_taking",
            "user_interruption",
        ],
        help="Evaluation task to perform.",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing data for evaluation.",
    )

    args = parser.parse_args()

    if args.task == "backchannel":
        eval_backchannel(args.root_dir)
    elif args.task == "pause_handling":
        eval_pause_handling(args.root_dir)
    elif args.task == "smooth_turn_taking":
        eval_smooth_turn_taking(args.root_dir)
    elif args.task == "user_interruption":
        client = OpenAI(
            organization=organization,
            api_key=api_key,
        )
        client.models.list()
        eval_user_interruption(args.root_dir, client)


if __name__ == "__main__":
    main()