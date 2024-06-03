import os
from processing.prompt_processing import process_privacy_policy

class Processor:
    def __init__(self, max_section_size):
        self.max_section_size = max_section_size

    def process_file(self, file_path):
        sections, section_info = process_privacy_policy(file_path, self.max_section_size)
        return sections, section_info
