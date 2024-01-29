import time

import requests
import pandas as pd
import yesg


class ESGScores:

    headers = {
        "Authorization": "36ff15c2f7d31276bd411a9e1e7250d2"
    }

    def __init__(self, tickers):
        # for ticker in tickers:
        #     print(f"ESG score of {ticker} is {self.get_esg_score(ticker)}")
        #     time.sleep(60)
        self.get_esg_score("tsla")

    def get_esg_score(self, ticker):
        print(yesg.get_esg_full(ticker).to_string())

        # url = "https://esgapiservice.com/api/Scores/Company/{id}"
        #
        # # Use the ticker to replace the placeholder in the URL
        # url = url.format(id=ticker)
        #
        # esg_response = requests.get(url, headers=self.headers)
        #
        # if esg_response.status_code == 200:
        #     esg_score = esg_response.json()
        #     return esg_score
        # else:
        #     print("Failed to retrieve ESG score. Status Code:", esg_response.status_code)
        #     return None


