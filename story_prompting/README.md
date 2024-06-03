
# Story Prompting

## Overview
This project processes privacy policies and generates story prompts using a configurable workflow. The tool is designed to help researchers and annotators identify and categorize privacy-related behaviors in policy texts. 

## Features
- Process single or multiple privacy policy files.
- Generate story prompts based on privacy behavior taxonomy.
- Multiple iterations and prompting levels.
- config JSON files.
- Outputs results to Excel and JSON files.


## Usage
Process a Single File
To process a single privacy policy file:

python main.py <input_file> <output_directory>
Example:

python main.py input/manual_stories/aiart.midjourney.dreamnow_privacy_policy.txt output
Process a Directory of Files
To process all privacy policy files in a directory:

python main.py <input_directory> <output_directory>
Example:

python main.py input/manual_stories/ output
Specify Iterations and Level
To specify the number of iterations and prompting level:

python main.py <input_path> <output_directory> --iterations <iterations> --level <level>
Example:

python main.py input/manual_stories/ output --iterations 5 --level 2
Use a Custom Configuration File
To use a custom configuration file:

python main.py <input_path> <output_directory> --config <config_file>
Example:

python main.py input/manual_stories/ output --config custom_config.json

## Configuration
The configuration file defines parameters like section size, iterations, model name, and prompt levels.

Example config.json
json
{
  "max_section_size": 1000,
  "iterations": 3,
  "model_name": "gpt-4-turbo-2024-04-09",
  "prompt_levels": {
    "1": "simple_story",
    "2": "behavior_analysis",
    "3": "detailed_story"
  }
}

## Code Structure
processor.py
Contains functions to process privacy policy texts into sections.

prompt_builder.py
Builds different types of prompts based on the provided taxonomy and section information.

response_handler.py
Handles responses from the model, ensuring the results are structured and stored correctly.

workflow_manager.py
Manages the workflow, orchestrating the processing, prompt generation, and response handling.

privacy_policy_processor.py
Unified entry point that processes files or directories based on provided configurations.

main.py
Command-line interface that parses arguments and invokes the processing functions.

## Output 
The results are saved in the specified output directory in both Excel and JSON formats. The Excel file contains detailed sections and responses, while the JSON file stores the combined response text and original prompts for further analysis.