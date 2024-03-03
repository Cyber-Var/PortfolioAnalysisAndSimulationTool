import os

folder = 'parameter tuning'

for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    with open(file_path, 'a+') as file:
        file.write("\n1w, AAPL, 26.02.24\n")
