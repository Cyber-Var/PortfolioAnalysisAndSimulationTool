import yesg


class ESGScores:

    def get_esg_score(self, ticker):
        # The lower the rating, the more ethical and sustainable a company is
        print(ticker + ":")
        scores = yesg.get_historic_esg(ticker).iloc[-1]
        e_score = scores["E-Score"]
        s_score = scores["S-Score"]
        g_score = scores["G-Score"]
        total_score = scores["Total-Score"]
        print(f"E-Score = {e_score}, S-Score = {s_score}, G-Score = {g_score}, Total-Score = {total_score}")
