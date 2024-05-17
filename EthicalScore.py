# import yesg
#
#
# class ESGScores:
#     """
#     Class for fetching ESG scores of companies
#     """
#
#     def get_esg_score(self, ticker):
#         """
#         Method for fetching ESG scores of a company through API
#         :param ticker: ticker of the company
#         :return:
#         """
#
#         # The lower the rating, the more ethical and sustainable a company is
#         print(ticker + ":")
#         scores = yesg.get_historic_esg(ticker).iloc[-1]
#         e_score = scores["E-Score"]
#         s_score = scores["S-Score"]
#         g_score = scores["G-Score"]
#         total_score = scores["Total-Score"]
#         print(f"E-Score = {e_score}, S-Score = {s_score}, G-Score = {g_score}, Total-Score = {total_score}")
#
#         # Return the scores:
#         return total_score, e_score, scores, g_score
