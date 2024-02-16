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

    statistics_dict = {0: "psi_1", 1: "psi_5", 2: "psi_10",
                       3: "arps_1", 4: "arps_5", 5: "arps_10",
                       6: "aplts_1", 7: "aplts_5", 8: "aplts_10",
                       9: "aclts_1", 10: "aclts_5", 11: "actls_10"}

    for s, x in enumerate(statistics):
        res = f_oneway(x[0], x[1], x[2], x[3], x[4])

        if res.pvalue > p_value:
            print(res)
            print(f"Differences are not statistically relevant with {dataset} in {statistics_dict[s]}")


def main(algorithm):

    datasets = ["ml-1m", "amzn-ggf", "citeulike-a", "pinterest", "yahoo-r3"]
    seeds = [12121995, 230782, 190291, 81163, 100362]

    model_types = ["baseline", "oversampling", "newensemble", "jannach", "ips", "borattoreweighting", "pd", "usampling"]#,"low",  "upperoversampling"]
    weights = [None, None, None, 2.25, None, 0.7, 0.12, None, None, None]

    average_results_dict = defaultdict(list)

    best_ensemble_delta_df = pd.read_csv(os.path.join("data", f"{algorithm}_best_ensemble_delta.csv"), sep=":")

    a = 10
    sigmoid = Sigmoid()

    reference_metric = "ndcg"  #"hit_rate"

    if reference_metric == "ndcg":
        a = 2

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

        arps_1 = [[] for _ in range(len(model_types))]
        arps_5 = [[] for _ in range(len(model_types))]
        arps_10 = [[] for _ in range(len(model_types))]

        aplts_1 = [[] for _ in range(len(model_types))]
        aplts_5 = [[] for _ in range(len(model_types))]
        aplts_10 = [[] for _ in range(len(model_types))]

        reos_1 = [[] for _ in range(len(model_types))]
        reos_5 = [[] for _ in range(len(model_types))]
        reos_10 = [[] for _ in range(len(model_types))]

        for m, model_type in enumerate(model_types):

            for seed in seeds:

                folder_name = f"{model_type}_{str(seed)}"

                weight = weights[m]

                if model_type == "ensemble":
                    if algorithm == "bpr":
                        weight = '{:f}'.format(float(best_ensemble_delta_df[best_ensemble_delta_df["Dataset"] == dataset]["Delta"]))

                    else:
                        weight = "{:.2f}".format(float(best_ensemble_delta_df[best_ensemble_delta_df["Dataset"] == dataset]["Delta"]))

                if weight is not None:
                    folder_name += f"_{str(weight)}"

                path = os.path.join("data", dataset, "results", folder_name, "result_test.json")
                if algorithm == "bpr":
                    path = os.path.join("data", dataset, "bpr", "results", folder_name, "result_test.json")

                with open(path, "r") as f:
                    results_dict = json.load(f)

                average_results_dict["arps@1"].append(results_dict["arp@1"])
                average_results_dict["arps@5"].append(results_dict["arp@5"])
                average_results_dict["arps@10"].append(results_dict["arp@10"])

                average_results_dict["positive_arps@1"].append(results_dict["positive_arp@1"])
                average_results_dict["positive_arps@5"].append(results_dict["positive_arp@5"])
                average_results_dict["positive_arps@10"].append(results_dict["positive_arp@10"])

                average_results_dict["negative_arps@1"].append(results_dict["negative_arp@1"])
                average_results_dict["negative_arps@5"].append(results_dict["negative_arp@5"])
                average_results_dict["negative_arps@10"].append(results_dict["negative_arp@10"])

                print(dataset, model_type, seed)
                average_results_dict["aplts@1"].append(results_dict["aplt@1"])
                average_results_dict["aplts@5"].append(results_dict["aplt@5"])
                average_results_dict["aplts@10"].append(results_dict["aplt@10"])

                average_results_dict["aclts@1"].append(results_dict["aclt@1"])
                average_results_dict["aclts@5"].append(results_dict["aclt@5"])
                average_results_dict["aclts@10"].append(results_dict["aclt@10"])

                average_results_dict["reos@1"].append(results_dict["reo@1"])
                average_results_dict["reos@5"].append(results_dict["reo@5"])
                average_results_dict["reos@10"].append(results_dict["reo@10"])

                average_results_dict[f"{reference_metric}s@1"].append(results_dict[f"{reference_metric}@1"])
                average_results_dict[f"{reference_metric}s@5"].append(results_dict[f"{reference_metric}@5"])
                average_results_dict[f"{reference_metric}s@10"].append(results_dict[f"{reference_metric}@10"])

                reference_metric_by_pop_1 = list(map(float, results_dict[f"{reference_metric}_by_pop@1"].split(',')))
                reference_metric_by_pop_5 = list(map(float, results_dict[f"{reference_metric}_by_pop@5"].split(',')))
                reference_metric_by_pop_10 = list(map(float, results_dict[f"{reference_metric}_by_pop@10"].split(',')))

                average_results_dict[f"{reference_metric}s_by_pop@1"].append(reference_metric_by_pop_1)
                average_results_dict[f"{reference_metric}s_by_pop@5"].append(reference_metric_by_pop_5)
                average_results_dict[f"{reference_metric}s_by_pop@10"].append(reference_metric_by_pop_10)

                globals_1[m].append(results_dict[f"{reference_metric}@1"])
                globals_5[m].append(results_dict[f"{reference_metric}@5"])
                globals_10[m].append(results_dict[f"{reference_metric}@10"])

                lows_1[m].append(reference_metric_by_pop_1[0])
                lows_5[m].append(reference_metric_by_pop_5[0])
                lows_10[m].append(reference_metric_by_pop_10[0])

                meds_1[m].append(reference_metric_by_pop_1[1])
                meds_5[m].append(reference_metric_by_pop_5[1])
                meds_10[m].append(reference_metric_by_pop_10[1])

                highs_1[m].append(reference_metric_by_pop_1[2])
                highs_5[m].append(reference_metric_by_pop_5[2])
                highs_10[m].append(reference_metric_by_pop_10[2])

                arps_1[m].append(results_dict["arp@1"])
                arps_5[m].append(results_dict["arp@5"])
                arps_10[m].append(results_dict["arp@10"])

                aplts_1[m].append(results_dict["aplt@1"])
                aplts_5[m].append(results_dict["aplt@5"])
                aplts_10[m].append(results_dict["aplt@10"])

                reos_1[m].append(results_dict["reo@1"])
                reos_5[m].append(results_dict["reo@5"])
                reos_10[m].append(results_dict["reo@10"])

            average_results_dict["avg_arp@1"] = np.mean(average_results_dict["arps@1"])
            average_results_dict["avg_arp@5"] = np.mean(average_results_dict["arps@5"])
            average_results_dict["avg_arp@10"] = np.mean(average_results_dict["arps@10"])

            average_results_dict["avg_positive_arp@1"] = np.mean(average_results_dict["positive_arps@1"])
            average_results_dict["avg_positive_arp@5"] = np.mean(average_results_dict["positive_arps@5"])
            average_results_dict["avg_positive_arp@10"] = np.mean(average_results_dict["positive_arps@10"])

            average_results_dict["avg_negative_arp@1"] = np.mean(average_results_dict["negative_arps@1"])
            average_results_dict["avg_negative_arp@5"] = np.mean(average_results_dict["negative_arps@5"])
            average_results_dict["avg_negative_arp@10"] = np.mean(average_results_dict["negative_arps@10"])

            average_results_dict["avg_aplt@1"] = np.mean(average_results_dict["aplts@1"])
            average_results_dict["avg_aplt@5"] = np.mean(average_results_dict["aplts@5"])
            average_results_dict["avg_aplt@10"] = np.mean(average_results_dict["aplts@10"])

            average_results_dict["avg_aclt@1"] = np.mean(average_results_dict["aclts@1"])
            average_results_dict["avg_aclt@5"] = np.mean(average_results_dict["aclts@5"])
            average_results_dict["avg_aclt@10"] = np.mean(average_results_dict["aclts@10"])

            average_results_dict["avg_reo@1"] = np.mean(average_results_dict["reos@1"])
            average_results_dict["avg_reo@5"] = np.mean(average_results_dict["reos@5"])
            average_results_dict["avg_reo@10"] = np.mean(average_results_dict["reos@10"])

            average_results_dict[f"avg_{reference_metric}@1"] = np.mean(average_results_dict[f"{reference_metric}s@1"])
            average_results_dict[f"avg_{reference_metric}@5"] = np.mean(average_results_dict[f"{reference_metric}s@5"])
            average_results_dict[f"avg_{reference_metric}@10"] = np.mean(average_results_dict[f"{reference_metric}s@10"])

            average_results_dict[f"avg_{reference_metric}_by_pop@1"] = list(np.mean(average_results_dict[f"{reference_metric}s_by_pop@1"], axis=0))
            average_results_dict[f"avg_{reference_metric}_by_pop@5"] = list(np.mean(average_results_dict[f"{reference_metric}s_by_pop@5"], axis=0))
            average_results_dict[f"avg_{reference_metric}_by_pop@10"] = list(np.mean(average_results_dict[f"{reference_metric}s_by_pop@10"], axis=0))

            avg_results_fn = f"{algorithm}_{model_type}_{reference_metric}_avg_test_results.json"

            with open(os.path.join("data", dataset, "results", avg_results_fn), "w") as f:
                json.dump(average_results_dict, f)

            average_results_dict.clear()

        psis_1 = [[] for _ in range(len(model_types))]
        psis_5 = [[] for _ in range(len(model_types))]
        psis_10 = [[] for _ in range(len(model_types))]

        for m, model_type in enumerate(model_types):

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

            psi_avg_results_fn = f"{algorithm}_{model_type}_{reference_metric}_psi_avg_test_results.json"

            with open(os.path.join("data", dataset, "results", psi_avg_results_fn), "w") as f:
                json.dump(psis_dict, f)

        statistics = [psis_1, psis_5, psis_10, arps_1, arps_5, arps_10,
                      aplts_1, aplts_5, aplts_10, reos_1, reos_5, reos_10]

        compute_statistics_test(statistics, dataset)

if __name__ == '__main__':
    algorithm = sys.argv[1] # "rvae"
    main(algorithm)