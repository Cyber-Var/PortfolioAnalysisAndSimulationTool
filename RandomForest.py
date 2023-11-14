

class RandomForestAlgorithm:

    def __init__(self, end_date, ticker, data):
        self.end_date = end_date
        self.ticker = ticker
        self.data = data

        self.train_set, self.test_set = self.splitData()

    def splitData(self):
        train_set = self.data.loc[:self.end_date]
        test_set = self.data.loc[self.end_date:]
        return train_set, test_set
