import numpy as np
import os
import copy
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-contamination_num", help="no of foldout tuples to switch with foldin",
                    type=int, default=1)
args = parser.parse_args()

base_dir = "./"
data_dir = os.path.join(base_dir, "data")
yahoo_data_dir = os.path.join(data_dir, "yahoo-r3")
fname = "data_rvae.pickle"

with open(os.path.join(yahoo_data_dir, fname), 'rb') as handle:
    data_rvae = pickle.load(handle)

contaminated_data = copy.deepcopy(data_rvae)
contaminated_test_data = {}

for user in contaminated_data["test_data"]:
    foldin, foldout = contaminated_data["test_data"][user][0], contaminated_data["test_data"][user][1]
    if len(foldin) > len(foldout):
        foldout_idxs_to_move = np.random.permutation(range(len(foldout)))[:args.contamination_num]
        foldin_idxs_to_move = np.random.permutation(range(len(foldin)))[:len(foldout_idxs_to_move)]

        for i in range(len(foldout_idxs_to_move)):
            idx = foldout_idxs_to_move[i]
            foldout_sample = foldout[idx]
            foldin_idx = foldin_idxs_to_move[i]
            foldout[idx] = foldin[foldin_idx]
            foldin[foldin_idx] = foldout_sample
    else:
        foldin_idxs_to_move = np.random.permutation(range(len(foldin)))[:args.contamination_num]
        foldout_idxs_to_move = np.random.permutation(range(len(foldout)))[:len(foldin_idxs_to_move)]


        for i in range(len(foldin_idxs_to_move)):
            idx = foldin_idxs_to_move[i]
            foldin_sample = foldin[idx]
            foldout_idx = foldout_idxs_to_move[i]
            foldin[idx] = foldout[foldout_idx]
            foldout[foldout_idx] = foldin_sample

    contaminated_test_data[user] = [foldin, foldout, contaminated_data["test_data"][user][2]]

contaminated_data["test_data"] = contaminated_test_data
contaminated_data["contamination"] = args.contamination_num
fname = "contaminated_data_rvae.pickle"

print(f"Contamination Number: {args.contamination_num}")

with open(os.path.join(yahoo_data_dir, fname), 'wb') as handle:
    pickle.dump(contaminated_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


