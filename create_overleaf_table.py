import os
import json
import sys
from torch.nn import Sigmoid
import torch
import numpy as np
from search_best_ensemble_delta import compute_final_score


def write_global_low_heatmap(avg_arps_1, avg_arps_5, avg_arps_10,
                             avg_positive_arps_1, avg_positive_arps_5, avg_positive_arps_10,
                             avg_negative_arps_1, avg_negative_arps_5, avg_negative_arps_10,
                             avg_aplts_1, avg_aplts_5, avg_aplts_10,
                             avg_reos_1, avg_reos_5, avg_reos_10,
                             avg_hit_rates_1, avg_hit_rates_5, avg_hit_rates_10,
                             avg_hit_rates_by_pop_1, avg_hit_rates_by_pop_5, avg_hit_rates_by_pop_10,
                             avg_pop_metric_by_pop_1, avg_pop_metric_by_pop_5, avg_pop_metric_by_pop_10,
                             model_types_dict, overleaf_heatmap_fn, overleaf_heatmap_positive_negative_fns,
                             overleaf_all_metrics_fns, dataset):
    overleaf_heatmap_positive_negative_1_fn, overleaf_heatmap_positive_negative_5_fn, overleaf_heatmap_positive_negative_10_fn = overleaf_heatmap_positive_negative_fns
    overleaf_all_metrics_1_fn, overleaf_all_metrics_5_fn, overleaf_all_metrics_10_fn = overleaf_all_metrics_fns

    with open(os.path.join("data", overleaf_heatmap_fn), "a") as f, \
            open(os.path.join("data", overleaf_heatmap_positive_negative_1_fn), "a") as pos_neg_1, \
            open(os.path.join("data", overleaf_heatmap_positive_negative_5_fn), "a") as pos_neg_5, \
            open(os.path.join("data", overleaf_heatmap_positive_negative_10_fn), "a") as pos_neg_10, \
            open(os.path.join("data", overleaf_all_metrics_1_fn), "a") as all_1, \
            open(os.path.join("data", overleaf_all_metrics_5_fn), "a") as all_5, \
            open(os.path.join("data", overleaf_all_metrics_10_fn), "a") as all_10:

        psi_best_1 = np.max(avg_pop_metric_by_pop_1)  # np.argmax(avg_pop_metric_by_pop_1[1:])
        psi_second_best_1 = np.partition(list(set(avg_pop_metric_by_pop_1)), -2)[-2]

        psi_best_5 = np.max(avg_pop_metric_by_pop_5)  # np.argmax(avg_pop_metric_by_pop_5)
        psi_second_best_5 = np.partition(list(set(avg_pop_metric_by_pop_5)), -2)[
            -2]  # np.argsort(avg_pop_metric_by_pop_5)[-2]

        psi_best_10 = np.max(avg_pop_metric_by_pop_10)  # np.argmax(avg_pop_metric_by_pop_10)
        psi_second_best_10 = np.partition(list(set(avg_pop_metric_by_pop_10)), -2)[
            -2]  # np.argsort(avg_pop_metric_by_pop_10)[-2]

        max_avg_arps_1, min_avg_arps_1 = str(max(avg_arps_1)), str(min(avg_arps_1))
        max_avg_arps_5, min_avg_arps_5 = str(max(avg_arps_5)), str(min(avg_arps_5))
        max_avg_arps_10, min_avg_arps_10 = str(max(avg_arps_10)), str(min(avg_arps_10))

        max_avg_aplts_1, min_avg_aplts_1 = str(max(avg_aplts_1)), str(min(avg_aplts_1))
        max_avg_aplts_5, min_avg_aplts_5 = str(max(avg_aplts_5)), str(min(avg_aplts_5))
        max_avg_aplts_10, min_avg_aplts_10 = str(max(avg_aplts_10)), str(min(avg_aplts_10))

        max_avg_reos_1, min_avg_reos_1 = str(max(avg_reos_1)), str(min(avg_reos_1))
        max_avg_reos_5, min_avg_reos_5 = str(max(avg_reos_5)), str(min(avg_reos_5))
        max_avg_reos_10, min_avg_reos_10 = str(max(avg_reos_10)), str(min(avg_reos_10))

        arp_best_1, aplt_best_1, reo_best_1 = float(min_avg_arps_1), float(max_avg_aplts_1), float(min_avg_reos_1)
        arp_second_best_1, aplt_second_best_1, reo_second_best_1 = np.sort(list(set(avg_arps_1)))[1], \
            np.sort(list(set(avg_aplts_1)))[-2] if len(set(avg_aplts_1)) > 1 else avg_aplts_1[0], \
            np.sort(list(set(avg_reos_1)))[1]

        arp_best_5, aplt_best_5, reo_best_5 = float(min_avg_arps_5), float(max_avg_aplts_5), float(min_avg_reos_5)
        arp_second_best_5, aplt_second_best_5, reo_second_best_5 = np.sort(list(set(avg_arps_5)))[1], \
            np.sort(list(set(avg_aplts_5)))[-2] if len(set(avg_aplts_5)) > 1 else avg_aplts_5[0], \
            np.sort(list(set(avg_reos_5)))[1]

        arp_best_10, aplt_best_10, reo_best_10 = float(min_avg_arps_10), float(max_avg_aplts_10), float(min_avg_reos_10)
        arp_second_best_10, aplt_second_best_10, reo_second_best_10 = np.sort(list(set(avg_arps_10)))[1], \
            np.sort(list(set(avg_aplts_10)))[-2] if len(set(avg_aplts_10)) > 1 else avg_aplts_10[0], \
            np.sort(list(set(avg_reos_10)))[1]

        max_avg_positive_arps_1, min_avg_positive_arps_1 = str(max(avg_positive_arps_1)), str(min(avg_positive_arps_1))
        max_avg_positive_arps_5, min_avg_positive_arps_5 = str(max(avg_positive_arps_5)), str(min(avg_positive_arps_5))
        max_avg_positive_arps_10, min_avg_positive_arps_10 = str(max(avg_positive_arps_10)), str(
            min(avg_positive_arps_10))

        max_avg_negative_arps_1, min_avg_negative_arps_1 = str(max(avg_negative_arps_1)), str(min(avg_negative_arps_1))
        max_avg_negative_arps_5, min_avg_negative_arps_5 = str(max(avg_negative_arps_5)), str(min(avg_negative_arps_5))
        max_avg_negative_arps_10, min_avg_negative_arps_10 = str(max(avg_negative_arps_10)), str(
            min(avg_negative_arps_10))

        max_avg_hit_rates_1, min_avg_hit_rates_1 = str(max(avg_hit_rates_1)), str(min(avg_hit_rates_1))
        max_avg_hit_rates_5, min_avg_hit_rates_5 = str(max(avg_hit_rates_5)), str(min(avg_hit_rates_5))
        max_avg_hit_rates_10, min_avg_hit_rates_10 = str(max(avg_hit_rates_10)), str(min(avg_hit_rates_10))

        max_low_1, min_low_1 = str(max(avg_hit_rates_by_pop_1[0])), str(min(avg_hit_rates_by_pop_1[0]))
        max_low_5, min_low_5 = str(max(avg_hit_rates_by_pop_5[0])), str(min(avg_hit_rates_by_pop_5[0]))
        max_low_10, min_low_10 = str(max(avg_hit_rates_by_pop_10[0])), str(min(avg_hit_rates_by_pop_10[0]))

        max_pop_1, min_pop_1 = str(max(avg_pop_metric_by_pop_1)), str(min(avg_pop_metric_by_pop_1))
        max_pop_5, min_pop_5 = str(max(avg_pop_metric_by_pop_5)), str(min(avg_pop_metric_by_pop_5))
        max_pop_10, min_pop_10 = str(max(avg_pop_metric_by_pop_10)), str(min(avg_pop_metric_by_pop_10))

        for i in range(len(avg_hit_rates_1)):

            model_type = list(model_types_dict.keys())[i]

            to_write_1 = ""
            to_write_5 = ""
            to_write_10 = ""
            to_write_all_metrics_1 = ""
            to_write_all_metrics_5 = ""
            to_write_all_metrics_10 = ""
            to_write_positive_negative_1 = ""
            to_write_positive_negative_5 = ""
            to_write_positive_negative_10 = ""

            prefix = ""
            suffix = ""

            to_write_1 += "& $\mathit{" + model_types_dict[model_type] + "}$ & " + prefix + "\gradientcell{" + str(
                avg_hit_rates_1[i]) + "}{" + min_avg_hit_rates_1 + "}{" + max_avg_hit_rates_1 + "}{low}{high}{\opacity}" \
                          + suffix

            to_write_1 += "& " + prefix + "\gradientcell{" + str(
                avg_hit_rates_by_pop_1[0][i]) + "}{" + min_low_1 + "}{" + max_low_1 + "}{low}{high}{\opacity}" + suffix \

            if avg_arps_1[i] == arp_best_1:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_arps_1[i] == arp_second_best_1:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_1 += "& " + prefix + "\gradientcell{" + str(
                avg_arps_1[i]) + "}{" + min_avg_arps_1 + "}{" + max_avg_arps_1 + "}{low}{high}{\opacity}" \
                          + suffix

            to_write_all_metrics_1 += to_write_1  # arp

            prefix = ""
            suffix = ""

            if avg_aplts_1[i] == aplt_best_1:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_aplts_1[i] == aplt_second_best_1:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_all_metrics_1 += "& " + prefix + "\gradientcell{" + str(
                avg_aplts_1[i]) + "}{" + min_avg_aplts_1 + "}{" + max_avg_aplts_1 + "}{low}{high}{\opacity}" \
                                      + suffix

            prefix = ""
            suffix = ""

            if avg_reos_1[i] == reo_best_1:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_reos_1[i] == reo_second_best_1:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_all_metrics_1 += "& " + prefix + "\gradientcell{" + str(
                avg_reos_1[i]) + "}{" + min_avg_reos_1 + "}{" + max_avg_reos_1 + "}{low}{high}{\opacity}" \
                                      + suffix

            to_write_positive_negative_1 += to_write_1

            prefix = ""
            suffix = ""

            if avg_pop_metric_by_pop_1[i] == psi_best_1 and (
                    algorithm == "rvae" or (algorithm == "bpr" and dataset != "yahoo-r3")):
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_pop_metric_by_pop_1[i] == psi_second_best_1 and (
                    algorithm == "rvae" or (algorithm == "bpr" and dataset != "yahoo-r3")):
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_1 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_1[i]) + "}{" + min_pop_1 + "}{" + max_pop_1 + "}{low}{high}{\opacity}" + suffix

            to_write_positive_negative_1 += "& \gradientcell{" + str(
                avg_positive_arps_1[
                    i]) + "}{" + min_avg_positive_arps_1 + "}{" + max_avg_positive_arps_1 + "}{low}{high}{\opacity}"

            to_write_positive_negative_1 += "& \gradientcell{" + str(
                avg_negative_arps_1[
                    i]) + "}{" + min_avg_negative_arps_1 + "}{" + max_avg_negative_arps_1 + "}{low}{high}{\opacity}"

            to_write_positive_negative_1 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_1[
                    i]) + "}{" + min_pop_1 + "}{" + max_pop_1 + "}{low}{high}{\opacity}" + suffix + "\\\\"

            to_write_all_metrics_1 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_1[
                    i]) + "}{" + min_pop_1 + "}{" + max_pop_1 + "}{low}{high}{\opacity}" + suffix + "\\\\"

            prefix = ""
            suffix = ""

            to_write_5 += "& " + prefix + "\gradientcell{" + str(
                avg_hit_rates_5[
                    i]) + "}{" + min_avg_hit_rates_5 + "}{" + max_avg_hit_rates_5 + "}{low}{high}{\opacity}" + suffix

            to_write_5 += "& " + prefix + "\gradientcell{" + str(
                avg_hit_rates_by_pop_5[0][i]) + "}{" + min_low_5 + "}{" + max_low_5 + "}{low}{high}{\opacity}" + suffix \

            if avg_arps_5[i] == arp_best_5:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_arps_5[i] == arp_second_best_5:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_5 += "& " + prefix + "\gradientcell{" + str(
                avg_arps_5[i]) + "}{" + min_avg_arps_5 + "}{" + max_avg_arps_5 + "}{low}{high}{\opacity}" \
                          + suffix

            to_write_all_metrics_5 += "& $\mathit{" + model_types_dict[model_type] + "}$" + to_write_5  # arp

            to_write_positive_negative_5 += "& $\mathit{" + model_types_dict[model_type] + "}$" + to_write_5

            to_write_positive_negative_5 += "& \gradientcell{" + str(
                avg_positive_arps_5[
                    i]) + "}{" + min_avg_positive_arps_5 + "}{" + max_avg_positive_arps_5 + "}{low}{high}{\opacity}"

            to_write_positive_negative_5 += "& \gradientcell{" + str(
                avg_negative_arps_5[
                    i]) + "}{" + min_avg_negative_arps_5 + "}{" + max_avg_negative_arps_5 + "}{low}{high}{\opacity}"

            prefix = ""
            suffix = ""

            if avg_aplts_5[i] == aplt_best_5:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_aplts_5[i] == aplt_second_best_5:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_all_metrics_5 += "& " + prefix + "\gradientcell{" + str(
                avg_aplts_5[i]) + "}{" + min_avg_aplts_5 + "}{" + max_avg_aplts_5 + "}{low}{high}{\opacity}" \
                                      + suffix

            prefix = ""
            suffix = ""

            if avg_reos_5[i] == reo_best_5:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_reos_5[i] == reo_second_best_5:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_all_metrics_5 += "& " + prefix + "\gradientcell{" + str(
                avg_reos_5[i]) + "}{" + min_avg_reos_5 + "}{" + max_avg_reos_5 + "}{low}{high}{\opacity}" \
                                      + suffix

            prefix = ""
            suffix = ""

            if avg_pop_metric_by_pop_5[i] == psi_best_5 and (
                    algorithm == "rvae" or (algorithm == "bpr" and dataset != "yahoo-r3")):
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_pop_metric_by_pop_5[i] == psi_second_best_5 and (
                    algorithm == "rvae" or (algorithm == "bpr" and dataset != "yahoo-r3")):
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_5 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_5[i]) + "}{" + min_pop_5 + "}{" + max_pop_5 + "}{low}{high}{\opacity}" + suffix

            to_write_positive_negative_5 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_5[
                    i]) + "}{" + min_pop_5 + "}{" + max_pop_5 + "}{low}{high}{\opacity}" + suffix + "\\\\"

            to_write_all_metrics_5 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_5[
                    i]) + "}{" + min_pop_5 + "}{" + max_pop_5 + "}{low}{high}{\opacity}" + suffix + "\\\\"

            prefix = ""
            suffix = ""

            to_write_10 += "& " + prefix + "\gradientcell{" + str(
                avg_hit_rates_10[
                    i]) + "}{" + min_avg_hit_rates_10 + "}{" + max_avg_hit_rates_10 + "}{low}{high}{\opacity}" + suffix

            to_write_10 += "& " + prefix + "\gradientcell{" + str(
                avg_hit_rates_by_pop_10[0][
                    i]) + "}{" + min_low_10 + "}{" + max_low_10 + "}{low}{high}{\opacity}" + suffix

            if avg_arps_10[i] == arp_best_10:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_arps_10[i] == arp_second_best_10:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_10 += "& " + prefix + "\gradientcell{" + str(
                avg_arps_10[i]) + "}{" + min_avg_arps_10 + "}{" + max_avg_arps_10 + "}{low}{high}{\opacity}" \
                           + suffix

            to_write_positive_negative_10 += "& $\mathit{" + model_types_dict[model_type] + "}$" + to_write_10

            to_write_positive_negative_10 += "& \gradientcell{" + str(
                avg_positive_arps_10[
                    i]) + "}{" + min_avg_positive_arps_10 + "}{" + max_avg_positive_arps_10 + "}{low}{high}{\opacity}"

            to_write_positive_negative_10 += "& \gradientcell{" + str(
                avg_negative_arps_10[
                    i]) + "}{" + min_avg_negative_arps_10 + "}{" + max_avg_negative_arps_10 + "}{low}{high}{\opacity}"

            to_write_all_metrics_10 += "& $\mathit{" + model_types_dict[model_type] + "}$ " + to_write_10  # arp

            prefix = ""
            suffix = ""

            if avg_aplts_10[i] == aplt_best_10:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_aplts_10[i] == aplt_second_best_10:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_all_metrics_10 += "& " + prefix + "\gradientcell{" + str(
                avg_aplts_10[i]) + "}{" + min_avg_aplts_10 + "}{" + max_avg_aplts_10 + "}{low}{high}{\opacity}" \
                                       + suffix

            prefix = ""
            suffix = ""

            if avg_reos_10[i] == reo_best_10:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_reos_10[i] == reo_second_best_10:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_all_metrics_10 += "& " + prefix + "\gradientcell{" + str(
                avg_reos_10[i]) + "}{" + min_avg_reos_10 + "}{" + max_avg_reos_10 + "}{low}{high}{\opacity}" \
                                       + suffix

            prefix = ""
            suffix = ""

            if avg_pop_metric_by_pop_10[i] == psi_best_10:
                prefix = "$\mathbf{"
                suffix = "}$"

            if avg_pop_metric_by_pop_10[i] == psi_second_best_10:
                prefix = "$\\underline{"
                suffix = "}$"

            to_write_10 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_10[
                    i]) + "}{" + min_pop_10 + "}{" + max_pop_10 + "}{low}{high}{\opacity}" + suffix + \
                           "\\\\"

            to_write_positive_negative_10 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_10[
                    i]) + "}{" + min_pop_10 + "}{" + max_pop_10 + "}{low}{high}{\opacity}" + suffix + "\\\\"

            to_write_all_metrics_10 += "& " + prefix + "\gradientcell{" + str(
                avg_pop_metric_by_pop_10[
                    i]) + "}{" + min_pop_10 + "}{" + max_pop_10 + "}{low}{high}{\opacity}" + suffix + "\\\\"

            to_write = (to_write_1 + to_write_5 + to_write_10)  # .replace("& &", "&")

            if model_type == "baseline" or model_type == "ensemble":
                to_write += "\n\hhline{~|-|-|-|-|-|-|-|-|-|-|-|-|-|}\n"
                to_write_positive_negative_1 += "\n\hhline{~|-|-|-|-|-|-|-|}\n"
                to_write_positive_negative_5 += "\n\hhline{~|-|-|-|-|-|-|-|}\n"
                to_write_positive_negative_10 += "\n\hhline{~|-|-|-|-|-|-|-|}\n"
                to_write_all_metrics_1 += "\n\hhline{~|-|-|-|-|-|-|-|}\n"
                to_write_all_metrics_5 += "\n\hhline{~|-|-|-|-|-|-|-|}\n"
                to_write_all_metrics_10 += "\n\hhline{~|-|-|-|-|-|-|-|}\n"

            elif model_type == "pd":
                to_write += "\n\midrule\n"
                to_write_positive_negative_1 += "\n\midrule\n"
                to_write_positive_negative_5 += "\n\midrule\n"
                to_write_positive_negative_10 += "\n\midrule\n"
                to_write_all_metrics_1 += "\n\midrule\n"
                to_write_all_metrics_5 += "\n\midrule\n"
                to_write_all_metrics_10 += "\n\midrule\n"

            f.write(to_write)
            pos_neg_1.write(to_write_positive_negative_1)
            pos_neg_5.write(to_write_positive_negative_5)
            pos_neg_10.write(to_write_positive_negative_10)
            all_1.write(to_write_all_metrics_1)
            all_5.write(to_write_all_metrics_5)
            all_10.write(to_write_all_metrics_10)


def main(algorithm):
    datasets = ["ml-1m", "amzn-ggf", "citeulike-a", "pinterest", "yahoo-r3"]
    datasets_dict = {"ml-1m": "MovieLens-1M", "amzn-ggf": "Amazon-GGF", "citeulike-a": "Citeulike-a",
                     "pinterest": "Pinterest", "yahoo-r3": "Yahoo-r3"}

    upper_algorithm = algorithm.upper()

    model_types_dict = {"baseline": upper_algorithm, "oversampling": "{" + upper_algorithm + "}^{S}",
                        "usampling": "{" + upper_algorithm + "}^{S,u}",
                        "ensemble": "{" + upper_algorithm + "}^{E}",
                        "jannach": "{" + upper_algorithm + "}^{Jan}", "ips": "{" + upper_algorithm + "}^{IPS}",
                        "borattoreweighting": "{" + upper_algorithm + "}^{b(r)}",
                        "pd": "{" + upper_algorithm + "}^{PD}"}

    model_types = ["baseline", "oversampling", "usampling", "ensemble", "jannach", "ips",
                   "borattoreweighting", "pd"]

    reference_metric = "ndcg"  # "hit_rate" #

    overleaf_all_metrics_fns = []
    for k in [1, 5, 10]:

        f = f"overleaf_{algorithm}_{reference_metric}_heatmap_all_metrics_{k}.txt"
        overleaf_all_metrics_fns.append(f)
        if os.path.exists(os.path.join("data", f)):
            os.remove(os.path.join("data", f))

    metric = "HR"
    if reference_metric == "ndcg":
        metric = "nDCG"


    for i, k in enumerate([1, 5, 10]):
        with open(os.path.join("data", overleaf_all_metrics_fns[i]), "a") as f:
            to_write = "\\resizebox{!}{0.7\columnwidth}{\\begin{tabular}{|p{0.2cm}|c|cc|c|c|c|c|}\\toprule {} & \multirow{1}{*}{Model} & " \
                       "\multicolumn{2}{c|}{$" + metric + "@" + str(k) + "$} & \multicolumn{1}{c|}{$\mathit{ARP}@" + str(
                k) + "$} & " \
                     "\multicolumn{1}{c|}{$\mathit{APLT}@" + str(k) + "$} &" + \
                       "\multicolumn{1}{c|}{$\mathit{REO}@" + str(k) + "$} &" + \
                       "\multicolumn{1}{c|}{$\mathit{BQS}@" + str(k) + "$} " \
                                                                       "\\\\\cmidrule{3-4} {} & {} & Global & Low & {} & {} & {} & {} \\\\\midrule\n"
            f.write(to_write)

    for dataset in datasets:

        for i, k in enumerate([1, 5, 10]):
            with open(os.path.join("data", overleaf_all_metrics_fns[i]), "a") as f:
                f.write(
                    "\multirow{2}{*}{\\rotatebox[origin = c]{90}{\hspace{-20.5mm} " + datasets_dict[dataset] + "}} \n")

        avg_arps_1 = []
        avg_arps_5 = []
        avg_arps_10 = []

        avg_positive_arps_1 = []
        avg_positive_arps_5 = []
        avg_positive_arps_10 = []

        avg_negative_arps_1 = []
        avg_negative_arps_5 = []
        avg_negative_arps_10 = []

        avg_aplts_1 = []
        avg_aplts_5 = []
        avg_aplts_10 = []

        avg_reos_1 = []
        avg_reos_5 = []
        avg_reos_10 = []

        avg_hit_rates_1 = []
        avg_hit_rates_5 = []
        avg_hit_rates_10 = []

        avg_hit_rates_by_pop_1 = [[] for _ in range(3)]
        avg_hit_rates_by_pop_5 = [[] for _ in range(3)]
        avg_hit_rates_by_pop_10 = [[] for _ in range(3)]

        avg_pop_metric_by_pop_1 = []
        avg_pop_metric_by_pop_5 = []
        avg_pop_metric_by_pop_10 = []

        for i, model_type in enumerate(model_types):

            avg_results_fn = f"{algorithm}_{model_type}_{reference_metric}_avg_test_results.json"
            print(avg_results_fn)

            with open(os.path.join("data", dataset, "results", avg_results_fn), "r") as f:
                average_results_dict = json.load(f)

            write_table(average_results_dict, reference_metric, model_types_dict, model_type, overleaf_table_fn)

            avg_arp_1 = round(average_results_dict["avg_arp@1"], 1)
            avg_arp_5 = round(average_results_dict["avg_arp@5"], 1)
            avg_arp_10 = round(average_results_dict["avg_arp@10"], 1)

            avg_arps_1.append(avg_arp_1)
            avg_arps_5.append(avg_arp_5)
            avg_arps_10.append(avg_arp_10)

            avg_aplt_1 = round(average_results_dict["avg_aplt@1"], 3)
            avg_aplt_5 = round(average_results_dict["avg_aplt@5"], 3)
            avg_aplt_10 = round(average_results_dict["avg_aplt@10"], 3)

            avg_aplts_1.append(avg_aplt_1)
            avg_aplts_5.append(avg_aplt_5)
            avg_aplts_10.append(avg_aplt_10)

            avg_reo_1 = round(average_results_dict["avg_reo@1"], 3)
            avg_reo_5 = round(average_results_dict["avg_reo@5"], 3)
            avg_reo_10 = round(average_results_dict["avg_reo@10"], 3)

            avg_reos_1.append(avg_reo_1)
            avg_reos_5.append(avg_reo_5)
            avg_reos_10.append(avg_reo_10)

            avg_positive_arp_1 = round(average_results_dict["avg_positive_arp@1"], 1)
            avg_positive_arp_5 = round(average_results_dict["avg_positive_arp@5"], 1)
            avg_positive_arp_10 = round(average_results_dict["avg_positive_arp@10"], 1)

            avg_positive_arps_1.append(avg_positive_arp_1)
            avg_positive_arps_5.append(avg_positive_arp_5)
            avg_positive_arps_10.append(avg_positive_arp_10)

            avg_negative_arp_1 = round(average_results_dict["avg_negative_arp@1"], 1)
            avg_negative_arp_5 = round(average_results_dict["avg_negative_arp@5"], 1)
            avg_negative_arp_10 = round(average_results_dict["avg_negative_arp@10"], 1)

            avg_negative_arps_1.append(avg_negative_arp_1)
            avg_negative_arps_5.append(avg_negative_arp_5)
            avg_negative_arps_10.append(avg_negative_arp_10)

            avg_hit_rates_1.append(round(average_results_dict[f"avg_{reference_metric}@1"], 4))
            avg_hit_rates_5.append(round(average_results_dict[f"avg_{reference_metric}@5"], 4))
            avg_hit_rates_10.append(round(average_results_dict[f"avg_{reference_metric}@10"], 4))

            avg_hit_rates_by_pop_1[0].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@1"][0], 2))
            avg_hit_rates_by_pop_1[1].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@1"][1], 2))
            avg_hit_rates_by_pop_1[2].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@1"][2], 2))

            avg_hit_rates_by_pop_5[0].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@5"][0], 2))
            avg_hit_rates_by_pop_5[1].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@5"][1], 2))
            avg_hit_rates_by_pop_5[2].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@5"][2], 2))

            avg_hit_rates_by_pop_10[0].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@10"][0], 2))
            avg_hit_rates_by_pop_10[1].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@10"][1], 2))
            avg_hit_rates_by_pop_10[2].append(round(average_results_dict[f"avg_{reference_metric}_by_pop@10"][2], 2))

            if model_type != "baseline":
                psi_results_fn = f"{algorithm}_{model_type}_{reference_metric}_psi_avg_test_results.json"

                with open(os.path.join("data", dataset, "results", psi_results_fn), "r") as f:
                    psi_results_dict = json.load(f)

                avg_pop_metric_by_pop_1.append(psi_results_dict["avg_psi@1"])
                avg_pop_metric_by_pop_5.append(psi_results_dict["avg_psi@5"])
                avg_pop_metric_by_pop_10.append(psi_results_dict["avg_psi@10"])

        avg_pop_metric_by_pop_1.insert(0, 0.5)
        avg_pop_metric_by_pop_5.insert(0, 0.5)
        avg_pop_metric_by_pop_10.insert(0, 0.5)

        write_global_low_heatmap(avg_arps_1, avg_arps_5, avg_arps_10,
                                 avg_positive_arps_1, avg_positive_arps_5, avg_positive_arps_10,
                                 avg_negative_arps_1, avg_negative_arps_5, avg_negative_arps_10,
                                 avg_aplts_1, avg_aplts_5, avg_aplts_10,
                                 avg_reos_1, avg_reos_5, avg_reos_10,
                                 avg_hit_rates_1, avg_hit_rates_5, avg_hit_rates_10,
                                 avg_hit_rates_by_pop_1, avg_hit_rates_by_pop_5, avg_hit_rates_by_pop_10,
                                 avg_pop_metric_by_pop_1, avg_pop_metric_by_pop_5, avg_pop_metric_by_pop_10,
                                 model_types_dict, overleaf_global_low_fn, overleaf_positive_negative_fns,
                                 overleaf_all_metrics_fns, dataset)


    for i, k in enumerate([1, 5, 10]):
        with open(os.path.join("data", overleaf_all_metrics_fns[i]), "a") as f:
            f.write("\end{tabular} \n}")


if __name__ == '__main__':
    algorithm = sys.argv[1]  # "rvae", "bpr", "simgcl"
    main(algorithm)
