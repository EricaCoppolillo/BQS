
import os
import numpy as np
import json
import sys


def compute_final_score(global_b, global_e, low_b, low_e, a):
    # − (𝑎 𝜙@𝑘 − 1/2𝑎 )^2 + 1/4𝑎^2

    diff_globals = global_e - global_b
    diff_lows = low_e - low_b

    globals_score = diff_globals
    if diff_globals < 0:
        globals_score = -(a*diff_globals)**2 + diff_globals# -(diff_globals*a-1/(2*a))**2 + 1/(4*a**2)

    lows_score = diff_lows
    if diff_lows < 0:
        lows_score = -(a * diff_lows)**2 + diff_lows# -(diff_lows*a-1/(2*a))**2 + 1/(4*a**2)

    return globals_score + lows_score

def main(algorithm, a):
    datasets = ["citeulike-a", "ml-1m", "pinterest", "yahoo-r3", "amzn-ggf"]
    seeds = [12121995, 230782, 190291, 81163, 100362]

    avg_test_results_path = os.path.join("data", f"{algorithm}_best_ensemble_delta.csv")

    if os.path.exists(avg_test_results_path):
        os.remove(avg_test_results_path)

    with open(avg_test_results_path, "a") as f:
        f.write("Dataset:Delta:Avg Low HR@1\n")

    deltas = np.arange(0.05, 1.05, 0.05)
    deltas_dict = {"yahoo-r3": np.arange(0.05, 1.05, 0.05), "ml-1m": np.arange(0.05, 1.05, 0.05),
                         "citeulike-a": np.arange(0.0005, 0.005, 0.0005), "pinterest": np.arange(0.00005, 0.0005, 0.00005), "amzn-ggf": np.arange(0.00005, 0.0005, 0.00005)}
    results_path_dict = {"yahoo-r3": "{:.2f}", "ml-1m": "{:.2f}",
                         "citeulike-a": "{:.4f}", "pinterest": "{:.5f}", "amzn-ggf": "{:.5f}"}

    epsilon = 10e-5
    top_ks = [1, 5, 10]

    for dataset in datasets:

        avg_test_results = []

        if algorithm == "bpr":
            deltas = deltas_dict[dataset]

        for i in deltas:

            delta_test_results = []

            for seed in seeds:
                results_path = os.path.join("data", dataset, "results", "ensemble_" + str(seed) + "_" + "{:.2f}".format(i))
                results_path = os.path.join(results_path, "result_val.json")

                baseline_results_path = os.path.join("data", dataset, "results", "baseline_" + str(seed))
                baseline_results_path = os.path.join(baseline_results_path, "result_val.json")

                if algorithm == "bpr":
                    results_path = os.path.join("data", dataset, "bpr", "results",
                                                    "newensemble_" + str(seed) + "_" + results_path_dict[dataset].format(i))
                    print(results_path)
                    results_path = os.path.join(results_path, "result_val.json")

                    baseline_results_path = os.path.join("data", dataset, "bpr", "results", "baseline_" + str(seed))
                    baseline_results_path = os.path.join(baseline_results_path, "result_val.json")

                with open(results_path, "r") as f:
                    results_dict = json.load(f)

                with open(baseline_results_path, "r") as f:
                    baseline_results_dict = json.load(f)

                final_scores = []

                for top_k in top_ks:

                    global_e = float(results_dict[f"hit_rate@{top_k}"])
                    global_b = float(baseline_results_dict[f"hit_rate@{top_k}"])

                    low_e = float(results_dict[f"hit_rate_by_pop@{top_k}"].split(",")[0])
                    low_b = float(baseline_results_dict[f"hit_rate_by_pop@{top_k}"].split(",")[0])

                    if low_e == 0.0:
                        low_e += epsilon

                    if low_b == 0.0:
                        low_b += epsilon

                    final_score = compute_final_score(global_b, global_e, low_b, low_e, a)

                    # print((global_e-global_b/global_b), (low_e-low_b)/low_b)
                    # final_score = (global_e-global_b)/global_b + (low_e-low_b)/low_b  # np.log(global_e-global_b+1) + np.log(low_e-low_b+1)
                    final_scores.append(final_score)

                print(final_scores)
                final_scores = [(1/x) if x != 0.0 else epsilon for x in final_scores]
                print(final_scores)

                delta_test_results.append(len(top_ks)/np.sum(final_scores))

            avg_test_results.append(np.mean(delta_test_results))

        # print(avg_test_results)

        with open(avg_test_results_path, "a") as f:
            if algorithm == "bpr" and dataset != "yahoo-r3":
                f.write(dataset + ":" + str(deltas[np.argmax(avg_test_results)].round(5)) + ":" + str(
                    max(avg_test_results).round(3)) + "\n")
            else:
                f.write(dataset + ":" + str(deltas[np.argmax(avg_test_results)].round(2)) + ":" + str(max(avg_test_results).round(3)) + "\n")


if __name__ == '__main__':
    algorithm = sys.argv[1]  # "rvae"
    if algorithm == "rvae":
        a = 25
    elif algorithm == "bpr":
        a = 25
    main(algorithm, a)
