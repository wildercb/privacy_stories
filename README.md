### Privacy Stories v0.2.7



#### To create graphs based on annotated privacy policies that display connections of actions, data types and purposes -> graphs_stories/graphs.ipynb 


#### To create privacy stories 
upload a privacy policy file to the input folder and use story_prompting1.ipynb
Run through processing and prompting, note: This has not yet been made very efficient, throughout each step please update the relevant app information. 
output -> check output for your new csv 


#### To annotate/answer questions about output privacy stories
Use annotator/processor.ipynb to load an excel file for annotation and modify the questions accordingly. 
navigate to the annotator directory and run
```python main.py```
open the link provided in terminal 
This does rely on the model outputting stories in the proper format of 1. {story}, 2. {story} , etc...



