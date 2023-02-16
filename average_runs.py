import os
import json
import sys

import numpy as np
import pandas as pd

from collections import defaultdict

from search_best_ensemble_delta import compute_final_score
from torch.nn import Sigmoid
import torch

from scipy.stats import wilcoxon, f_oneway


def compute_statistics_test(statistics, dataset):

    p_value = 0.05

    statistics_dict = {0: "global_1", 1: "global_5", 2: "global_10",
                       3: "low_1", 4: "low_5", 5: "low_10",
                       6: "med_1", 7: "med_5", 8: "med_10",
                       9: "high_1", 10: "high_5", 11: "high_10",
                       12: "psi_1", 13: "psi_5", 14: "psi_10"}

    for s, x in enumerate(statistics):
        res = f_oneway(x[0], x[1], x[2], x[3], x[4])
        if res.pvalue > p_value:
            print(res)
            print(f"Differences are not statistically relevant with {dataset} in {statistics_dict[s]}")


def main(algorithm):

    datasets = ["ml-1m", "amzn-ggf", "citeulike-a", "pinterest", "yahoo-r3"]
    seeds = [12121995, 230782, 190291, 81163, 100362]

    model_types = ["baseline", "oversampling", "ensemble", "jannach", "ips", "borattoreweighting", "pd", "low", "usampling"]
    weights = [None, None, None, 2.25, None, 0.7, 0.12, None, None]

    average_results_dict = defaultdict(list)

    best_ensemble_delta_df = pd.read_csv(os.path.join("data", f"{algorithm}_best_ensemble_delta.csv"), sep=":")

    results_path_dict = {"yahoo-r3": "{:.2f}", "ml-1m": "{:.2f}",
                         "citeulike-a": "{:.4f}", "pinterest": "{:.5f}", "amzn-ggf": "{:.5f}"}
    a = 10
    sigmoid = Sigmoid()

    for dataset in datasets:

        globals_1 = [[] for _ in range(len(model_types))]
        globals_5 = [[] for _ in range(len(model_types))]
        globals_10 = [[] for _ in range(len(model_types))]

        lows_1 = [[] for _ in range(len(model_types))]
        lows_5 = [[] for _ in range(len(model_types))]
        lows_10 = [[] for _ in range(len(model_types))]

        meds_1 = [[] for _ in range(len(model_types))]
        meds_5 = [[] for _ in range(len(model_types))]
        meds_10 = [[] for _ in range(len(model_types))]

        highs_1 = [[] for _ in range(len(model_types))]
        highs_5 = [[] for _ in range(len(model_types))]
        highs_10 = [[] for _ in range(len(model_types))]

        for m, model_type in enumerate(model_types):

            if algorithm == "bpr":
                if model_type == "ensemble":
                    model_type = "newensemble"

            for seed in seeds:

                folder_name = f"{model_type}_{str(seed)}"

                weight = weights[m]

                if model_type == "ensemble" or model_type == "newensemble":
                    if algorithm == "bpr":
                        weight = results_path_dict[dataset].format(
                            float(best_ensemble_delta_df[best_ensemble_delta_df["Dataset"] == dataset]["Delta"]))
                    else:
                        weight = "{:.2f}".format(float(best_ensemble_delta_df[best_ensemble_delta_df["Dataset"] == dataset]["Delta"]))

                if weight is not None:
                    folder_name += f"_{str(weight)}"

                path = os.path.join("data", dataset, "results", folder_name, "result_test.json")
                if algorithm == "bpr":
                    path = os.path.join("data", dataset, "bpr", "results", folder_name, "result_test.json")

                with open(path, "r") as f:
                    results_dict = json.load(f)

                average_results_dict["hit_rates@1"].append(results_dict["hit_rate@1"])
                average_results_dict["hit_rates@5"].append(results_dict["hit_rate@5"])
                average_results_dict["hit_rates@10"].append(results_dict["hit_rate@10"])

                hit_rates_by_pop_1 = list(map(float, results_dict["hit_rate_by_pop@1"].split(',')))
                hit_rates_by_pop_5 = list(map(float, results_dict["hit_rate_by_pop@5"].split(',')))
                hit_rates_by_pop_10 = list(map(float, results_dict["hit_rate_by_pop@10"].split(',')))

                average_results_dict["hit_rates_by_pop@1"].append(hit_rates_by_pop_1)
                average_results_dict["hit_rates_by_pop@5"].append(hit_rates_by_pop_5)
                average_results_dict["hit_rates_by_pop@10"].append(hit_rates_by_pop_10)

                globals_1[m].append(results_dict["hit_rate@1"])
                globals_5[m].append(results_dict["hit_rate@5"])
                globals_10[m].append(results_dict["hit_rate@10"])

                lows_1[m].append(hit_rates_by_pop_1[0])
                lows_5[m].append(hit_rates_by_pop_5[0])
                lows_10[m].append(hit_rates_by_pop_10[0])

                meds_1[m].append(hit_rates_by_pop_1[1])
                meds_5[m].append(hit_rates_by_pop_5[1])
                meds_10[m].append(hit_rates_by_pop_10[1])

                highs_1[m].append(hit_rates_by_pop_1[2])
                highs_5[m].append(hit_rates_by_pop_5[2])
                highs_10[m].append(hit_rates_by_pop_10[2])

            average_results_dict["avg_hit_rate@1"] = np.mean(average_results_dict["hit_rates@1"])
            average_results_dict["avg_hit_rate@5"] = np.mean(average_results_dict["hit_rates@5"])
            average_results_dict["avg_hit_rate@10"] = np.mean(average_results_dict["hit_rates@10"])

            average_results_dict["avg_hit_rate_by_pop@1"] = list(np.mean(average_results_dict["hit_rates_by_pop@1"], axis=0))
            average_results_dict["avg_hit_rate_by_pop@5"] = list(np.mean(average_results_dict["hit_rates_by_pop@5"], axis=0))
            average_results_dict["avg_hit_rate_by_pop@10"] = list(np.mean(average_results_dict["hit_rates_by_pop@10"], axis=0))

            average_results_dict["std_hit_rate@1"] = np.std(average_results_dict["hit_rates@1"])
            average_results_dict["std_hit_rate@5"] = np.std(average_results_dict["hit_rates@5"])
            average_results_dict["std_hit_rate@10"] = np.std(average_results_dict["hit_rates@10"])

            average_results_dict["std_hit_rate_by_pop@1"] = list(np.std(average_results_dict["hit_rates_by_pop@1"], axis=0))
            average_results_dict["std_hit_rate_by_pop@5"] = list(np.std(average_results_dict["hit_rates_by_pop@5"], axis=0))
            average_results_dict["std_hit_rate_by_pop@10"] = list(np.std(average_results_dict["hit_rates_by_pop@10"], axis=0))

            avg_results_fn = f"{algorithm}_{model_type}_avg_test_results.json"

            with open(os.path.join("data", dataset, "results", avg_results_fn), "w") as f:
                json.dump(average_results_dict, f)

            average_results_dict.clear()

        psis_1 = [[] for _ in range(len(model_types))]
        psis_5 = [[] for _ in range(len(model_types))]
        psis_10 = [[] for _ in range(len(model_types))]

        for m, model_type in enumerate(model_types):

            if algorithm == "bpr":
                if model_type == "ensemble":
                    model_type = "newensemble"

            for i in range(len(seeds)):
                psi_1 = sigmoid(torch.tensor(compute_final_score(globals_1[0][i], globals_1[m][i], lows_1[0][i], lows_1[m][i], a))).item()
                psis_1[m].append(psi_1)
                psi_5 = sigmoid(torch.tensor(
                    compute_final_score(globals_5[0][i], globals_5[m][i], lows_5[0][i], lows_5[m][i], a))).item()
                psis_5[m].append(psi_5)
                psi_10 = sigmoid(torch.tensor(
                    compute_final_score(globals_10[0][i], globals_10[m][i], lows_10[0][i], lows_10[m][i], a))).item()
                psis_10[m].append(psi_10)

            psis_dict = {"avg_psi@1": round(np.mean(psis_1[m]), 3), "avg_psi@5": round(np.mean(psis_5[m]), 3), "avg_psi@10": round(np.mean(psis_10[m]), 3),
                         "std_psi@1": np.std(psis_1[m]), "std_psi@5": np.std(psis_5[m]), "std_psi@10": np.std(psis_10[m])}

            psi_avg_results_fn = f"{algorithm}_{model_type}_psi_avg_test_results.json"

            with open(os.path.join("data", dataset, "results", psi_avg_results_fn), "w") as f:
                json.dump(psis_dict, f)

        statistics = [globals_1, globals_5, globals_10, lows_1, lows_5, lows_10,
                      meds_1, meds_5, meds_10, highs_1, highs_5, highs_10, psis_1, psis_5, psis_10]

        compute_statistics_test(statistics, dataset)

if __name__ == '__main__':
    algorithm = sys.argv[1] # "rvae"
    main(algorithm)