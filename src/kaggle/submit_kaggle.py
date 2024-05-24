import os
from kaggle.api.kaggle_api_extended import KaggleApi
import fire

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

api = KaggleApi()
api.authenticate()

def main(competition = 'house-prices-advanced-regression-techniques', file_name = '../gen/submission.csv', message = 'Your message'):
    api.competition_submit(file_name, message, competition)

if __name__ == '__main__':
    fire.Fire(main)