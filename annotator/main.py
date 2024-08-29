from flask import Flask, render_template, request, jsonify
import json
import re

app = Flask(__name__)

# Load JSON data for the web form
def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

@app.route('/')
def index():
    questions_data = load_json('questions.json')

    for section in questions_data:
        section['Responses'] = {
            k: section['Responses'][k] for k in section['Responses']
            if k in ['Response2', 'Response3'] and section['Responses'][k]
        }
        for response_key, response_texts in section['Responses'].items():
            section['Responses'][response_key] = response_texts

    return render_template('index.html', questions=questions_data, len=len, enumerate=enumerate)

@app.route('/submit', methods=['POST'])
def submit_answers():
    questions_data = load_json('questions.json')
    user_answers = request.form.to_dict()

    for idx, section in enumerate(questions_data):
        for response_key, stories in section['Responses'].items():
            if 'UserAnswers' not in section:
                section['UserAnswers'] = {}

            if isinstance(stories, list):
                for story_idx, story in enumerate(stories):
                    answer_key_prefix = f'answer_{response_key}_{idx}_{story_idx}'
                    for q_num in range(1, len(section['Questions']) + 1):
                        question_key = f'{answer_key_prefix}_{q_num}'
                        if question_key in user_answers:
                            if f'Story{story_idx+1}' not in section['UserAnswers']:
                                section['UserAnswers'][f'Story{story_idx+1}'] = {}
                            section['UserAnswers'][f'Story{story_idx+1}'][f'Answer{q_num}'] = user_answers[question_key]

    with open('questions_with_answers.json', 'w', encoding='utf-8') as file:
        json.dump(questions_data, file, indent=4)

    return jsonify({"status": "success", "message": "Answers saved successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
