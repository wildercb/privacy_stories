# 📜 Privacy Stories
[![Version](https://img.shields.io/badge/version-0.2.7-blue.svg)](https://github.com/your_repo)
[![Python](https://img.shields.io/badge/python-green.svg)](https://www.python.org/)

### Overview

**Privacy Stories** This repository puts together the work I did for privacy stories throughout 2023-2024 , there are several items throughout the repo we will not go over here including apsjs (to get info from the play store) and most of the utils (extra, other functions to get the data required for this project) 
---

## 🔮 Getting Started

### 1. Create Graphs Based on Annotated Privacy Policies

Generate graphs to visualize the connections between actions, data types, and purposes in the annotated privacy stories & their apps data safety sections.

- **Notebook**: [`graphs_stories/graphs.ipynb`](graphs_stories/graphs.ipynb)

### 2. Generate Privacy Stories

To use openai, first create a secrets.py file in the utils directory. Add OPENAI_API_KEY = {your open api key}, 
also provides functionality to run local models with Ollama. 

Upload a privacy policy file to the `input` folder, then use the following notebook to generate privacy stories:

- **Notebook**: [`story_prompting1.ipynb`](story_prompting1.ipynb)
- **Output**: `output/{app_id}_privacy_stories.xlsx` 
  - Leave the output file blank to default to `privacy_stories_1_1.xlsx`.

### 3. Annotate and Answer Questions About Privacy Stories

Annotate / answer questions about the generated privacy stories using the annotation tool.

- **Notebook**: [`annotator/processor.ipynb`](annotator/processor.ipynb) - Load an Excel file for annotation and customize your questions.
- **Run the Annotation App**:
  ```bash
  cd annotator
  python main.py
