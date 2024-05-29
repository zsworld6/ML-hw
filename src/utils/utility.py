import os
from kaggle.api.kaggle_api_extended import KaggleApi

from typing import Tuple

def get_variable_from_file(file_path, variable_name):
    global_namespace = {}
    with open(file_path, 'r') as file:
        exec(file.read(), global_namespace)
    return global_namespace.get(variable_name)

def get_prompts(data_path: str, describe_task: bool) -> str:
    
    example_columns = get_variable_from_file(f"{data_path}/columns.py", "example_columns")
    predict_goal = get_variable_from_file(f"{data_path}/columns.py", "predict_goal")
    train_file = get_variable_from_file(f"{data_path}/columns.py", "train_file")
    if train_file is None:
        train_file = "train.csv"
    test_file = get_variable_from_file(f"{data_path}/columns.py", "test_file")
    if test_file is None:
        test_file = "test.csv"
    if describe_task:
        task_description = get_variable_from_file(f"{data_path}/columns.py", "task_description")
        if task_description is None:
            task_description = ""
    else:
        task_description = ""
    code = f"""        
{task_description}
The dataframe '{data_path}/{train_file}' is loaded and in memory. Columns are also named attributes. 
Examples of the dataset in '{data_path}/{train_file}':
```
{example_columns}
```
Columns data types are all unknown, and you should carefully think the data type of each column by yourself.
Please choose the best way to implement a model to predict the {predict_goal}. '{data_path}/{train_file}' is the training data. '{data_path}/{test_file}' is the test data. And you should save your prediction as a csv file './submissions/submission.csv'.
Please try your best to avoiding any possible errors when running your code!
    """
    return code


def submit(competition = 'house-prices-advanced-regression-techniques', file_name = 'submission.csv', message = 'Your message') -> Tuple[bool, str]:
    
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'

    if not os.path.exists("submission.csv"):
        return False, "FileNotFound: 'submission.csv'"

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