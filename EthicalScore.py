import time

import requests
import pandas as pd
import yesg


class ESGScores:

    def __init__(self, tickers):
        # The lower the rating, the more sustainable a company is
        print("ESG scores:")
        for ticker in tickers:
            self.get_esg_score(ticker)
        print()

    def get_esg_score(self, ticker):
        print(ticker + ":")
        scores = yesg.get_historic_esg(ticker).iloc[-1]
        e_score = scores["E-Score"]
        s_score = scores["S-Score"]
        g_score = scores["G-Score"]
        total_score = scores["Total-Score"]
        print(f"E-Score = {e_score}, S-Score = {s_score}, G-Score = {g_score}, Total-Score = {total_score}")



