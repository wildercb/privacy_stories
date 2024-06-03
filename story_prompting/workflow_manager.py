import json
import os
import pandas as pd

from processing.prompt_processing import load_ontology, ontology_file
from story_prompt_processer import Processor
from story_prompt_builder import PromptBuilder
from story_response_handler import ResponseHandler

class WorkflowManager:
    def __init__(self, config):
        self.processor = Processor(config["max_section_size"])
        self.prompt_builder = PromptBuilder(load_ontology(ontology_file))
        self.response_handler = ResponseHandler(config["model_name"])
        self.iterations = config["iterations"]
        self.output_dir = config["output_dir"]

    def process_file(self, file_path, level):
        sections, section_info = self.processor.process_file(file_path)
        prompts = [self.prompt_builder.build_prompt(section, level) for section in sections]
        return self.run_prompts(prompts, sections, level)

    def run_prompts(self, prompts, sections, level):
        combined_response_text = ""
        section_number = 1
        new_data = []

        for prompt in prompts:
            for iteration in range(1, self.iterations + 1):
                response_text = self.response_handler.get_response(prompt, build_string)
                combined_response_text += response_text
                prompt += "\n" + response_text

            new_data.append({
                'Section': f"Section {section_number}",
                'Level': level,
                'Response': combined_response_text
            })
            section_number += 1

        self.save_results(new_data)
        return combined_response_text

    def save_results(self, data):
        output_file = os.path.join(self.output_dir, "results.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
