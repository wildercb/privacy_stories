class PromptBuilder:
    def __init__(self, privacy_behavior_taxonomy):
        self.privacy_behavior_taxonomy = privacy_behavior_taxonomy

    def build_prompt(self, section, level):
        if level == 1:
            return self.build_simple_story_prompt(section)
        elif level == 2:
            return self.build_behavior_analysis_prompt(section)
        elif level == 3:
            return self.build_detailed_story_prompt(section)

    def build_simple_story_prompt(self, section):
        return (
            f"User: Privacy Policy:\n{section.cleaned_annotated_text}\n"
            f"User: Write the privacy behaviors found within this text.\n"
            f"System: Privacy Behaviors:"
        )

    def build_behavior_analysis_prompt(self, section):
        return (
            f"User: Privacy Policy:\n{section.cleaned_annotated_text}\n"
            f"User: Analyze the privacy behaviors found within this text based on the taxonomy.\n"
            f"System: Privacy Behaviors:"
        )

    def build_detailed_story_prompt(self, section):
        return (
            f"User: Privacy Policy:\n{section.cleaned_annotated_text}\n"
            f"User: Write detailed privacy stories including privacy behaviors from the taxonomy.\n"
            f"System: Privacy Stories:"
        )
