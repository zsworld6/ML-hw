from kaggle.api.kaggle_api_extended import KaggleApi
import json, fire


def main(competition = 'spaceship-titanic'):
    api = KaggleApi()
    api.authenticate()
    
    leaderboard = api.competition_leaderboard_view(competition)
    # print(leaderboard[0].__dict__)

    # 获取提交历史
    submissions = api.competition_submissions(competition)
    # for submission in submissions:
    submission = submissions[0]
    print(submission.publicScore)
    # print(submission.__dict__)
    # print(f"Submission ID: {submission.ref}")
    # print(f"Filename: {submission.fileName}")
    # print(f"Date: {submission.date}")
    # print(f"Description: {submission.description}")
    # print(f"Error Description: {submission.errorDescription if submission.hasErrorDescription else 'No errors'}")
    # print(f"Status: {submission.status}")
    # print(f"Public Score: {submission.publicScore if submission.hasPublicScore else 'N/A'}")
    # print(f"Private Score: {submission.privateScore if submission.hasPrivateScore else 'N/A'}")
    # print(f"Submitted By: {submission.submittedBy}")
    # print(f"Team Name: {submission.teamName}")
    # print(f"URL: {submission.url}")
    # print(f"Size: {submission.size}")
    # print("------")
    # # print(submission.__dict__)
    # print("------")

    # first_place = leaderboard['submissions'][0]
    # print(f"Rank: {first_place['rank']}")
    # print(f"Team: {first_place['teamName']}")
    # print(f"Score: {first_place['score']}")
    
if __name__ == '__main__':
    fire.Fire(main)