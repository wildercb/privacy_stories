# print(sections_behaviors[0].comparison_sections[1].privacy_behaviors)

from processing.prompt_processing import ontology_to_string, load_ontology, ontology_file

# Import the process_privacy_policy function from your script
from processing.prompt_processing import process_privacy_policy

# Specify the file path and section size
file_path = "input/com.aisecurity privacy_policy.txt"
max_section_size = 1000

# Call the function with the specified parameters
stories, behaviors = process_privacy_policy(file_path, max_section_size)

# Now you can use the returned stories and behaviors in your notebook
print(stories)
print(behaviors)

privacy_taxonomy = load_ontology(ontology_file)

privacy_behavior_taxonomy = ontology_to_string(privacy_taxonomy) 

behaviors_strings = simple_stories_strings = stories_strings = sections = []

build_string = (
        f"User: As an annotator of privacy policies, your task involves identifying and categorizing privacy-related behaviors "
        f"in policy texts. Each behavior should be annotated with the most specific label possible, based on the provided privacy behavior taxonomy. "
)

for i, section in enumerate(stories):
    section_story_information = stories[i]
    section_behavior_information = behaviors[i]

    # Iterate through all comparison sections safely
    for j, nearest_match in enumerate(section_story_information.comparison_sections):
        # Ensure there's a corresponding match in behaviors before accessing
        if j < len(section_behavior_information.comparison_sections):
            nearest_match_behaviors = section_behavior_information.comparison_sections[j]

    # Constructing the prompts
    sections.append(section_story_information.cleaned_annotated_text)
    behaviors_strings.append(
        f"User: This taxonomy serves as your reference for understanding and classifying various privacy practices mentioned in the policies.\n"
        f"Privacy Behavior Taxonomy:\n{privacy_behavior_taxonomy}\n"
        f"User: Privacy policy: {nearest_match.cleaned_annotated_text} "
        f"User: Write the privacy behaviors found within this text\n"
        f"System: Privacy Behaviors: {', '.join(nearest_match_behaviors.privacy_behaviors)}\n"
        f"User: Privacy Policy:\n{section_story_information.cleaned_annotated_text}\n"
        f"User: Write the privacy behaviors found within this text and those which are connected "
        f"reflect a critical understanding of the text. Be mindful of avoiding assumptions or hallucinations that are not supported by the text.\n"
        f"System: Privacy Behaviors:"
    )
    simple_stories_strings.append(
        f"User: This taxonomy serves as your reference for understanding and classifying various privacy practices mentioned in the policies.\n"
        f"Privacy Behavior Taxonomy:\n{privacy_behavior_taxonomy}\n"
        f"User: Privacy Policy:\n{section_story_information.cleaned_annotated_text}\n"
        f"User: Write the privacy behaviors found within this text and those which are connected "
        f"reflect a critical understanding of the text. Be mindful of avoiding assumptions or hallucinations that are not supported by the text.\n"
        f"System: Privacy Behaviors:"
    )
    stories_strings.append(
        f"User: This taxonomy serves as your reference for understanding and classifying various privacy practices mentioned in the policies.\n"
        f"Privacy Behavior Taxonomy:\n{privacy_behavior_taxonomy}\n"
        f"User: Privacy policy: {nearest_match.cleaned_annotated_text} "
        f"User: Write the privacy behaviors found within this text\n"
        f"System: Privacy Behaviors: {', '.join(nearest_match_behaviors.privacy_behaviors)}\n"
        f"User: Write the privacy stories found within this text, connecting related actions, data types and purposes together in the format of we (action) (data type) for the purpose of (purpose)\n"
        f"System: Privacy stories: {', '.join(nearest_match.privacy_stories)}\n"
        f"User: Privacy Policy:\n{section_story_information.cleaned_annotated_text}\n"
        f"User: Write the privacy behaviors found within this text and those which are connected "
        f"reflect a critical understanding of the text. Be mindful of avoiding assumptions or hallucinations that are not supported by the text.\n"
        f"System: Privacy Behaviors:"
    )


system_string_1 = (
    "This taxonomy serves as your reference for understanding and classifying various privacy practices mentioned in the policies.\n\n"
    "Privacy Behavior Taxonomy:\n" + privacy_behavior_taxonomy + "\n"
    "Privacy policy:\n" + nearest_match.cleaned_annotated_text + "\n"
    "Privacy Stories including privacy behaviors from the privacy behavior taxonomy:\n" + "\n".join(nearest_match.privacy_stories) + "\n"
    "Privacy Policy:\n" + section_story_information.cleaned_annotated_text + "\n"
    "write the privacy behaviors found within the privacy behavior taxonomy from the the input section text "
    "reflect a critical understanding of the text. Be mindful of avoiding assumptions or hallucinations that are not supported by the text."
    "\nPrivacy Behaviors:"
)
''' To copy prompt to clipboad
import pyperclip 
behaviors = ''
for behavior in behaviors_strings:
    behaviors += behaviors
pyperclip.copy(behaviors)
print(behaviors)'''

print(behaviors_strings[1])

# Build the template
annotator_template = system_string_1

level_1_prompts = simple_stories_strings
level_2_prompts = behaviors_strings
level_3_prompts = stories_strings
