from scipy import io
import pandas as pd

# This file does some tests on some famous ML datasets to investigate on the matrix coherence topic

# # load enron dataset
# eron_dataset = io.loadmat("dataset/email-Enron.mat")
# eron_problem = eron_dataset['Problem']['A']
# eron_problem
# # load protein dataset
# protein_dataset = io.loadmat("dataset/protein_data.mat")['X'].toarray()

# # reading csv files
# abalone_dataset = pd.read_csv('dataset/Outside Resources/abalone/abalone.data', sep=",")
# print(abalone_dataset.to_numpy()[:, 1:])
#
# reading csv files
wine_dataset = pd.read_csv('dataset/Outside Resources/wine/wine.data', sep=",").to_numpy()


