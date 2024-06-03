import argparse
import json
import os
from workflow_manager import WorkflowManager

def main():
    parser = argparse.ArgumentParser(description="Process privacy policies for story prompting")
    parser.add_argument("input_path", help="Path to the input file or directory")
    parser.add_argument("output_dir", help="Directory to save the output files")
    parser.add_argument("--config", default="config.json", help="Path to the configuration file")
    parser.add_argument("--iterations", type=int, help="Number of iterations")
    parser.add_argument("--level", type=int, help="Prompting level")

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    if args.iterations:
        config["iterations"] = args.iterations
    if args.level:
        config["prompt_levels"]["default"] = args.level

    manager = WorkflowManager(config)

    if os.path.isdir(args.input_path):
        for filename in os.listdir(args.input_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(args.input_path, filename)
                manager.process_file(file_path, config["prompt_levels"]["default"])
    else:
        manager.process_file(args.input_path, config["prompt_levels"]["default"])

if __name__ == "__main__":
    main()
