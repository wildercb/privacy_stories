# 📜 Privacy Stories v0.2.7

[![Version](https://img.shields.io/badge/version-0.2.7-blue.svg)](https://github.com/your_repo)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

### Overview

**Privacy Stories** is a tool for creating, annotating, and visualizing connections within privacy policies. Our project empowers users to extract and analyze key components of privacy policies, providing insights into data usage, purposes, and actions.

---

## 🚀 Getting Started

### 1. Create Graphs Based on Annotated Privacy Policies

Generate insightful graphs to visualize the connections between actions, data types, and purposes in your annotated privacy stories.

- **Notebook**: [`graphs_stories/graphs.ipynb`](graphs_stories/graphs.ipynb)

### 2. Generate Privacy Stories

Upload a privacy policy file to the `input` folder, then use the following notebook to generate privacy stories:

- **Notebook**: [`story_prompting1.ipynb`](story_prompting1.ipynb)
- **Output**: `output/{app_id}_privacy_stories.xlsx` 
  - Leave the output file blank to default to `privacy_stories_1_1.xlsx`.

### 3. Annotate and Answer Questions About Privacy Stories

Easily annotate and answer questions about your privacy stories using our interactive annotation tool.

- **Notebook**: [`annotator/processor.ipynb`](annotator/processor.ipynb) - Load an Excel file for annotation and customize your questions.
- **Run the Annotation App**:
  ```bash
  cd annotator
  python main.py
