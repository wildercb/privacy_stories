import os
import re
import json
import argparse

from concurrent.futures import ThreadPoolExecutor

import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

section_size = 1000
model = SentenceTransformer('all-MiniLM-L6-v2')

comparison_policies_directory = 'policies_annotated/annotated_policies'
comparison_sections = []

ontology_file = 'privacy_ontology.json'

SECTION_PATTERN = re.compile(r'{#s(.*?)}(.*?)(?={#s|$)', re.DOTALL)
CLEAN_ANNOTATION_PATTERN = re.compile(r'(\{#s[^/]+/}|(\[#\w+)(.*?)\])')


class SectionBehaviorInformation:
    def __init__(self, section_text, cleaned_annotated_text, details):
        self.section_text = section_text
        self.cleaned_annotated_text = cleaned_annotated_text
        self.details = details
        self.comparison_sections = []  # Initialize comparison sections list
        self.privacy_stories = []  # Initialize privacy stories list

def format_privacy_story(action, dt, purpose):
    return f"We {action} {dt} for the purpose of {purpose}"    

#Section information with privacy stories  

class SectionInformation:
    def __init__(self, section_text, cleaned_annotated_text, details, privacy_stories=None, privacy_behaviors=None):
        self.section_text = section_text
        self.cleaned_annotated_text = cleaned_annotated_text
        self.details = details
        self.privacy_stories = privacy_stories or []
        self.privacy_behaviors = privacy_behaviors or []


class PrivacyPolicyProcessor:
    def __init__(self, input_dir, comparison_dir, model_name='all-MiniLM-L6-v2'):
        self.input_dir = input_dir
        self.comparison_dir = comparison_dir
        self.model = SentenceTransformer(model_name)
        self.section_size = 1000
        self.section_pattern = re.compile(r'{#s(.*?)}(.*?)(?={#s|$)', re.DOTALL)
        self.clean_annotation_pattern = re.compile(r'(\{#s[^/]+/}|(\[#\w+)(.*?)\])')

    def read_policy(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

    def split_text_into_sections(self, text):
        sentences = sent_tokenize(text)
        sections, current_section, current_section_size = [], '', 0
        for sentence in sentences:
            if current_section_size + len(sentence.split()) <= self.section_size:
                current_section += sentence + ' '
                current_section_size += len(sentence.split())
            else:
                sections.append(current_section.strip())
                current_section, current_section_size = sentence + ' ', len(sentence.split())
        if current_section:
            sections.append(current_section.strip())
        return sections

    def extract_and_process_policies(self):
        input_sections = self.split_text_into_sections(self.read_policy(self.input_dir))
        comparison_sections = []
        for root, dirs, files in os.walk(self.comparison_dir):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    comparison_sections.extend(self.split_text_into_sections(self.read_policy(path)))
        
        all_sections = input_sections + comparison_sections
        all_section_embeddings = self.model.encode(all_sections, convert_to_tensor=True)
        
        section_matches = []
        for i, section_embedding in enumerate(all_section_embeddings[:len(input_sections)]):
            similarities = util.pytorch_cos_sim(section_embedding, all_section_embeddings)[0]
            similar_indices = torch.argsort(similarities, descending=True)
            for idx in similar_indices:
                if idx.item() >= len(input_sections):
                    top_match = comparison_sections[idx.item() - len(input_sections)]
                    section_matches.append((input_sections[i], top_match))
                    break

        return section_matches


'''@debug
# Outputting the result
for i, (section, match) in enumerate(section_matches, start=1):
    print(f"Input Section {i}:")
    print(section)
    print("Top Matching Comparison Section:")
    print(match)
    print("\n---\n")
'''


def extract_sections_and_actions(text):
    # Find all sections and associated content
    matches = SECTION_PATTERN.findall(text)

    # Remove all markup annotations efficiently
    cleaned_text = CLEAN_ANNOTATION_PATTERN.sub(lambda m: m.group(3) if m.group(2) else m.group(1), text)

    # Prepare the initial result with the cleaned full text
    result = [{'section': cleaned_text.strip()}]

    # Process each match to extract detailed information
    for section_header, content in matches:
        # Extract annotations without redefining the cleaning patterns
        actions = re.findall(r'\[#a(.*?)\]', section_header)
        dt = re.findall(r'\[#dt(.*?)\]', section_header)
        purposes = re.findall(r'\[#p(.*?)\]', section_header)

        # Append the structured data to the result list
        result.append({
            'sequence': section_header.strip(),
            'actions': [action.strip() for action in actions],
            'dt': [dt_item.strip() for dt_item in dt],
            'purposes': [purpose.strip() for purpose in purposes]
        })

    return result

# Function to load ontology
def load_ontology(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_match_in_ontology(term, ontology):
    # Check for exact and partial matches in the ontology
    for category, items in ontology.items():
        for key, value in items.items():
            if term.lower() in key.lower() or any(term.lower() in synonym.lower() for synonym in value.get('Synonyms', [])):
                return key
    return None


# Prompt templating variables
def ontology_to_string(ontology):
    ontology_string = ""
    for category, items in ontology.items():
        ontology_string += f"Category: {category}\n"
        for key, value in items.items():
            ontology_string += f" - {key}: "
            if 'Synonyms' in value:
                ontology_string += f"Synonyms: {', '.join(value['Synonyms'])}; "
            subcategories = [k for k in value if k != 'Synonyms']
            if subcategories:
                ontology_string += f"Subcategories: {', '.join(subcategories)}; "
            ontology_string += "\n"
    return ontology_string

# Function to apply extraction to each section match and store information
def process_sections(section_matches, ontology):
    section_info_objects = []

    for i, (section, matches) in enumerate(section_matches, start=1):
        # Call the extraction function on each input section
        if not isinstance(matches, list):
            matches = [matches] 
        extracted_info_input = extract_sections_and_actions(section)

        # Store information for the input section
        input_section_info = SectionInformation(section, extracted_info_input[0]['section'], extracted_info_input[1:])

        # Generate privacy stories for the input section
        for entry in input_section_info.details:
            for action in entry['actions']:
                action_match = find_match_in_ontology(action, ontology['Actions'])
                if not action_match:
                    continue

                for dt in entry['dt']:
                    dt_match = find_match_in_ontology(dt, ontology['Data Types'])
                    if not dt_match:
                        continue

                    for purpose in entry['purposes']:
                        purpose_match = find_match_in_ontology(purpose, ontology['Purpose'])
                        if purpose_match:
                            story = format_privacy_story(action_match, dt_match, purpose_match)
                            input_section_info.privacy_stories.append(story)

        # Print cleaned annotated text and its privacy stories for input section
        '''print(f"\nCleaned Annotated Text for Input Section {i}:\n{input_section_info.cleaned_annotated_text}")
        for story in input_section_info.privacy_stories:
            print(f"Privacy Story: {story}")'''

        # Store information for the top matching comparison sections
        section_story_infos = []
        for k, match in enumerate(matches, 1):
            # Call the extraction function on each comparison section
            extracted_info_comparison = extract_sections_and_actions(match)

            # Store information for the comparison section
            section_story_info = SectionInformation(match, extracted_info_comparison[0]['section'], extracted_info_comparison[1:])
            section_story_infos.append(section_story_info)

            # Generate privacy stories for the comparison section
            for entry in section_story_info.details:
                for action in entry['actions']:
                    for dt in entry['dt']:
                        for purpose in entry['purposes']:
                            story = format_privacy_story(action, dt, purpose)
                            section_story_info.privacy_stories.append(story)

            # Print cleaned annotated text and its privacy stories for comparison section
            '''print(f"\nCleaned Annotated Text for Comparison Section {k}:\n{section_story_info.cleaned_annotated_text}")
            for story in section_story_info.privacy_stories:
                print(f"Privacy Story: {story}")'''

        # Add comparison section information to the input section's object
        input_section_info.comparison_sections = section_story_infos

        section_info_objects.append(input_section_info)

    return section_info_objects

def process_section_information(section_matches):
    """
    Takes in list of annotated privacy policy secction
    processes to find sequences and their internal actions,
    data types and purposes. 
    Outputs an list SectionInformation objects with their 
    respective stories and behavriors 
    """
    section_info_objects = []


    for i, (section, matches) in enumerate(section_matches):
        if not isinstance(matches, list):
            matches = [matches]
        # print(f"\nProcessing Input Section {i}:\n{'=' * 30}")
        
        # Call the extraction function on each input section
        extracted_info_input = extract_sections_and_actions(section)
        # Store information for the input section
        input_section_info = SectionInformation(section, section, extracted_info_input[0]['section'])
        section_info_objects.append(input_section_info)

        # Print cleaned annotated text and details for the input section
        '''print(f"\nCleaned Annotated Text for Input Section {i}:\n{input_section_info.cleaned_annotated_text}")
        for j, entry in enumerate(input_section_info.details, 1):
            print(f"\nSubsection {j}:\n{entry['sequence']}")
            print(f"Actions: {entry['actions']}")
            print(f"DT: {entry['dt']}")
            print(f"Purposes: {entry['purposes']}")
            for action in entry['actions']:
                for dt in entry['dt']:
                    for purpose in entry['purposes']:
                        story = format_privacy_story(action, dt, purpose)
                        input_section_info.privacy_stories.append(story)
                        print(f"Privacy Story: {story}")
            print('-' * 20)'''

        # Store information for the top matching comparison sections
        comparison_section_infos = []
        for k, match in enumerate(matches):
            # print(f"\nTop Matching Comparison Section {k}:\n{'=' * 30}")
            # Call the extraction function on each comparison section
            extracted_info_comparison = extract_sections_and_actions(match)

            print(extracted_info_input[0]['section'])

            # Store information for the comparison section
            comparison_section_info = SectionInformation(match, extracted_info_comparison[0]['section'], extracted_info_comparison[1:])
            comparison_section_infos.append(comparison_section_info)

            # Print cleaned annotated text and details for the comparison section
            # print(f"{comparison_section_info.cleaned_annotated_text}")
            for l, entry in enumerate(comparison_section_info.details, 1):
                behavior = ''
                behavior += (f"\nBecuase of:\n{entry['sequence']}")
                actions_str = ', '.join(entry['actions'])
                behavior +=(f"\nActions: {actions_str}")
                dt_str = ', '.join(entry['dt'])
                behavior +=(f"\nDT: {dt_str}")
                purp_str = ', '.join(entry['purposes'])
                behavior +=(f"\nPurposes: {purp_str}")
                # print(behavior)
                comparison_section_info.privacy_behaviors.append(behavior)

            

        # Add comparison section information to the input section's object
        input_section_info.comparison_sections = comparison_section_infos
        # print('-' * 30)
        # print(len(section_info_objects))
    return section_info_objects

def read_policy(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()
def process_new_policy(policy_path):
    input_text = read_policy(policy_path)
    ontology = load_ontology(ontology_file)
    sections_info = process_sections(input_text, ontology)
    return sections_info

############
# Main fx  #
############
def process_privacy_policy(file_path, 
                           max_section_size, 
                           model_name='all-MiniLM-L6-v2',
comparison_policies_directory='policies_annotated/annotated_policies', 
                        ontology_file='privacy_ontology.json'
):
    
    processor = PrivacyPolicyProcessor(file_path, comparison_policies_directory, model_name)
    processor.section_size = max_section_size  # Update section size based on user input
    section_matches = processor.extract_and_process_policies()

    # Load the ontology
    privacy_ontology = load_ontology(ontology_file)
    
    # Process the sections
    sections_stories = process_sections(section_matches, privacy_ontology)
    sections_behaviors = process_section_information(section_matches)

    # Returning results for further usage
    return sections_stories, sections_behaviors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a privacy policy file.')
    parser.add_argument('--file_path', type=str,default=None, help='Path to the privacy policy file')
    parser.add_argument('--max_section_size', type=int,default=1000, help='Maximum section size for processing text')
    parser.add_argument('--policy_path',type=str,default=None, help='Path to the new privacy policy text file')
    args = parser.parse_args()

    if args.policy_path:
        result = process_new_policy(args.policy_path)
        for section_info in result:
            print(f"Section: {section_info.section_text}")
            print("Privacy Stories:")
            for story in section_info.privacy_stories:
                print(f"- {story}")
            print()

    process_privacy_policy(args.file_path, args.max_section_size)