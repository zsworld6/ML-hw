import os
from kaggle.api.kaggle_api_extended import KaggleApi

from typing import Tuple

def get_variable_from_file(file_path, variable_name):
    global_namespace = {}
    with open(file_path, 'r') as file:
        exec(file.read(), global_namespace)
    return global_namespace.get(variable_name)

def get_prompts(data_path: str) -> str:
    
    example_columns = get_variable_from_file(f"{data_path}/columns.py", "example_columns")
    predict_goal = get_variable_from_file(f"{data_path}/columns.py", "predict_goal")
    
    code = f"""
The dataframe '{data_path}/train.csv' is loaded and in memory. Columns are also named attributes. 
Description of the dataset in 'train.csv':
```
{example_columns}
```
Columns data types are all unknown, and you should carefully think the data type of each column by yourself.
Please choose the best way to implement a model to predict the {predict_goal}. '{data_path}/train.csv' is the training data. '{data_path}/test.csv' is the test data. And you should save your prediction in 'submission.csv'.
Please try your best to avoiding any possible errors when running your code!
    """
    return code

def submit(competition = 'house-prices-advanced-regression-techniques', file_name = 'submission.csv', message = 'Your message') -> Tuple[bool, str]:
    
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'

    api = KaggleApi()
    api.authenticate()
    api.competition_submit(file_name, message, competition)
    
    submissions = api.competition_submissions(competition)
    # for submission in submissions:
    submission = submissions[0]
    is_passing = submission.hasErrorDescription
    return is_passing, f"""Error Description: {submission.errorDescription if submission.hasErrorDescription else 'No errors'}
Status: {submission.status}
Public Score: {submission.publicScore if submission.hasPublicScore else 'N/A'}
f"Private Score: {submission.privateScore if submission.hasPrivateScore else 'N/A'}
"""