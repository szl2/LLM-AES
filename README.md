# LLM-AES
Evaluation of Metrics and Models for Automated Essay Scoring and Feedback Generation

- `input/`: dataset subfolder. 
    * Those dataset files are too big to put on github. 
    * Read the input/README.md for details. 

- `utilities.py`: The data pre-processing involves extracting and standardizing review texts, tokenizing feedback, and dividing the data into training, validation, and test sets. 

- `data_analysis.py`: Exploratory data analysis will examine review lengths, common themes, and language use to provide insights for model fine-tuning.

- Comaparative analysis of various LLMs:
    * `T5.py`
    * ...