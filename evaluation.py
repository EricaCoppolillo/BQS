import collections
import numpy as np
from sklearn.metrics import ndcg_score

class SimpleMetric:
    def __init__(self):
        self.recall = 0
        self.recall_den = 0
        self.weighted_recall = 0
        self.weighted_recall_den = 0
        self.hitrate = 0
        self.weighted_hitrate = 0

        self.ndcg = 0

        self.hitrate_by_pop = np.zeros(3)
        self.hitrate_by_pop_users = np.zeros(3)

        self.ndcg_by_pop = np.zeros(3)

        self.recalled_by_pop = np.zeros(3)
        self.positives_by_pop = np.zeros(3)
        self._num_users = 0

        self.users_arp = []
        self.users_positive_arp = []
        self.users_negative_arp = []

        self.users_aplt = []
        self.users_aclt = []

        self.users_p_r_low, self.users_p_r_med, self.users_p_r_high = [], [], []

        self.arp = 0
        self.positive_arp = 0
        self.negative_arp = 0

        self.aplt = 0
        self.aclt = 0

        self.ps_rs_low, self.ps_rs_med, self.ps_rs_high = 0, 0, 0

        self.reo = 0

        self.luciano_stat_by_pop = np.zeros(3)
        self.luciano_weighted_stat = 0
        self.luciano_occurencies = None
        self.luciano_guessed_items = None

    def metric_names(self):
        return ('recall',
                'weighted_recall',
                'hitrate',
                'weighted_hitrate',
                'hitrate_by_pop',
                'recalled_by_pop',
                'positives_by_pop',
                'luciano_weighted_stat',
                'luciano_stat',
                'luciano_stat_by_pop',
                'luciano_recalled_by_pop',
                'arp',
                'positive_arp',
                'negative_arp',
                'aplt',
                'aclt',
                'reo',
                'ndcg',
                'ndcg_by_pop')

    def __getitem__(self, item):
        v = getattr(self, item)
        if isinstance(v, np.ndarray):
            return ','.join([f'{j:.2f}' for j in v])
        return v

    def __str__(self):
        return f'recall = {self.recall:.2f}, weighted_recall = {self.weighted_recall:.2f}, ' \
               f'hitrate = {self.hitrate:.2f}, weighted_hitrate = {self.weighted_hitrate:.2f} ' \
               f'hitrate_by_pop = {[f"{x:.2f}" for x in self.hitrate_by_pop]}'

    def __repr__(self):
        return self.__str__()


class MetricAccumulator:
    def __init__(self):
        self.data = collections.defaultdict(SimpleMetric)

    def reset(self):
        self.data.clear()

    def get_top_k(self):
        return sorted(self.data.keys())

    def get_metrics(self, top_k=None):
        result = {}

        with np.errstate(divide='ignore', invalid='ignore'):
            k_list = [top_k] if top_k else self.data.keys()
            for k in k_list:
                acc = self.data[k]

                computed_acc = SimpleMetric()
                computed_acc.recall = acc.recall / acc.recall_den
                computed_acc.weighted_recall = acc.weighted_recall / acc.weighted_recall_den
                computed_acc.hitrate = acc.hitrate / acc._num_users
                computed_acc.weighted_hitrate = acc.weighted_hitrate / acc._num_users

                computed_acc.ndcg = acc.ndcg / acc._num_users

                computed_acc.hitrate_by_pop = acc.hitrate_by_pop / acc.hitrate_by_pop_users

                computed_acc.ndcg_by_pop = acc.ndcg_by_pop / acc.hitrate_by_pop_users  # they are equal

                computed_acc.ndcg_by_pop[np.isnan(computed_acc.ndcg_by_pop)] = 0

                computed_acc.recalled_by_pop = acc.recalled_by_pop
                computed_acc.positives_by_pop = acc.positives_by_pop

                # Luciano's metrics
                computed_acc.luciano_recalled_by_pop = acc.luciano_stat_by_pop
                computed_acc.luciano_stat_by_pop = acc.luciano_stat_by_pop / acc.positives_by_pop
                computed_acc.luciano_stat = acc.luciano_stat_by_pop.sum() / acc.positives_by_pop.sum()
                # computed_acc.weighted_luciano_stat = acc.weighted_luciano_stat / acc.positives_by_pop.sum()
                nz = np.nonzero(acc.luciano_occurencies)[0]

                computed_acc.luciano_weighted_stat = np.average(acc.luciano_guessed_items[nz] / acc.luciano_occurencies[nz])

                computed_acc.arp = acc.arp / acc._num_users
                computed_acc.positive_arp = acc.positive_arp / acc._num_users
                computed_acc.negative_arp = acc.negative_arp / acc._num_users

                computed_acc.aplt = acc.aplt / acc._num_users
                computed_acc.aclt = acc.aclt / acc._num_users

                computed_acc.reo = np.std((acc.ps_rs_low, acc.ps_rs_med, acc.ps_rs_high)) / \
                                   np.mean((acc.ps_rs_low, acc.ps_rs_med, acc.ps_rs_high))

                result[k] = computed_acc

        return result[top_k] if top_k else result

    def compute_metric(self, x, y, pos, neg, popularity, popularity_thresholds, top_k=10, test_users=None):
        """
        Compute metric:
        - recall
        - weighted recall
        - hitrate
        - weighted hit rate
        - hitrate per popularity

        :param x: user preferences, BS x items, if None then no warmup data is used
        :param y: user predictions, BS x items
        :param pos: list of positives
        :param neg: list of negatives
        :param popularity: dict-type of item popularity (normalized)
        :param popularity_thresholds: tuple of popularity thresholds
        :param top_k: top k items to select ranked by score
        :return: recall, weighted_recall, hitrate, weighted_hr, popularity_hitrate, total_positives(low,medium,high)
        """

        assert min([len(r) for r in neg]) >= top_k and top_k > 0, f"fail with top_k = {top_k} and neg = {[len(r) for r in neg]}"
        assert len(popularity) == y.shape[-1], f'{len(popularity)} != {y.shape[-1]}'

        for user in range(y.shape[0]):
            score = y[user, :]  # score degli oggetti

            if x is not None:
                input_idx = np.where(x[user, :] == 1)[0]  # indice degli oggetti in input per l'utente user
                viewed_item = set(input_idx)
                score[input_idx] = - np.inf  # hide viewed
            else:
                viewed_item = set()

            positive_items = set(pos[user])

            predicted_item = positive_items - viewed_item  # si considerano solo gli oggetti non forniti in input
            # TODO: assert positive_items intersection viewed_item is 0

            ranked_idx = np.argsort(-score)  # indici degli items ordinati per score in modo decrescente
            ranked_top_k_idx = ranked_idx[:top_k]  # indici dei top K elementi (ordinati per score in modo decrescente)

            H_u = [i for i in predicted_item if i in ranked_top_k_idx]  # elementi che piacciono all'utente nei top K

            low_popular_positive_h_u = [x for x in H_u if popularity[x] <= popularity_thresholds[0]]
            med_popular_positive_h_u = [x for x in H_u if popularity_thresholds[0] < popularity[x] <= popularity_thresholds[1]]
            high_popular_positive_h_u = [x for x in H_u if popularity[x] > popularity_thresholds[1]]

            low_popular_test_set = [x for x in predicted_item if popularity[x] <= popularity_thresholds[0]]
            med_popular_test_set = [x for x in predicted_item if popularity_thresholds[0] < popularity[x] <= popularity_thresholds[1]]
            high_popular_test_set = [x for x in predicted_item if popularity[x] > popularity_thresholds[1]]

            if not len(low_popular_test_set):
                p_r_low = -1
            else:
                p_r_low = len(low_popular_positive_h_u) / len(low_popular_test_set)
            if not len(med_popular_test_set):
                p_r_med = -1
            else:
                p_r_med = len(med_popular_positive_h_u) / len(med_popular_test_set)
            if not len(high_popular_test_set):
                p_r_high = -1
            else:
                p_r_high = len(high_popular_positive_h_u) / len(high_popular_test_set)

            P_u_c = sorted(list(predicted_item), key=lambda i: -score[i])[:top_k]

            low_popular_suggested_items = [x for x in ranked_top_k_idx if popularity[x] <= popularity_thresholds[0]]

            negative_recs = list(set(ranked_top_k_idx).difference(P_u_c))  # wrong suggestions

            popularities = [popularity[i] for i in ranked_top_k_idx]
            absolute_popularities = np.array(popularities) * test_users

            negative_popularities = [popularity[i] for i in negative_recs]
            negative_absolute_popularities = np.array(negative_popularities) * test_users

            positive_popularities = [popularity[i] for i in H_u]
            positive_absolute_popularities = np.array(positive_popularities) * test_users

            weights_H_u = sum([1 - popularity[i] for i in H_u])  # Numeratore
            weights_P_u_c = sum([1 - popularity[i] for i in P_u_c])  # Denominatore

            accumulator = self.data[top_k]
            if accumulator.luciano_occurencies is None:
                accumulator.luciano_occurencies = np.zeros(len(popularity))
                accumulator.luciano_guessed_items = np.zeros(len(popularity))

            accumulator.users_arp.append(np.sum(absolute_popularities)/min(len(ranked_top_k_idx), top_k))
            accumulator.users_positive_arp.append(np.sum(positive_absolute_popularities)/min(len(ranked_top_k_idx), top_k))
            accumulator.users_negative_arp.append(np.sum(negative_absolute_popularities)/min(len(ranked_top_k_idx), top_k))

            accumulator.users_aplt.append(len(low_popular_suggested_items)/min(len(ranked_top_k_idx), top_k))
            accumulator.users_aclt.append(len(low_popular_suggested_items))

            accumulator.users_p_r_low.append(p_r_low)
            accumulator.users_p_r_med.append(p_r_med)
            accumulator.users_p_r_high.append(p_r_high)

            accumulator._num_users += 1

            accumulator.recall += len(H_u)  # ok
            accumulator.recall_den += min(len(predicted_item), top_k)  # ok
            accumulator.weighted_recall += weights_H_u  # ok
            accumulator.weighted_recall_den += weights_P_u_c  # ok

            accumulator.hitrate += len(H_u) / min(top_k, len(predicted_item))  # ok
            accumulator.weighted_hitrate += weights_H_u / weights_P_u_c  # ok

            # NEW
            n_pos = min(top_k, len(predicted_item))
            n_neg = top_k - n_pos

            ideal_relevance_score = [1]*n_pos + [0]*n_neg

            idcg_at_k = np.sum([ideal_relevance_score[i]/np.log2(i+2) for i in range(top_k)])  # ideal

            actual_relevance_score = [1 if i in predicted_item else 0 for i in ranked_top_k_idx]

            dcg_at_k = np.sum([actual_relevance_score[i]/np.log2(i+2) for i in range(top_k)])

            ndcg_at_k = dcg_at_k/idcg_at_k

            if np.isnan(ndcg_at_k):
                print("nDCG is nan")
                exit(0)

            if ndcg_at_k > 1:
                print("nDCG > 1")
                exit(0)

            accumulator.ndcg += ndcg_at_k

            pop_dict = {0: [], 1: [], 2: []}

            for i in ranked_top_k_idx:
                current_popularity = popularity[i]

                if current_popularity <= popularity_thresholds[0]:
                    idx = 0
                elif popularity_thresholds[0] < current_popularity <= popularity_thresholds[1]:
                    idx = 1
                else:
                    idx = 2

                pop_dict[idx].append(i)

            for idx, split_list in pop_dict.items():

                H_u = [i for i in predicted_item if i in split_list]
                ranked_in_split_list = [i for i in ranked_top_k_idx if i in split_list]

                if idx == 0:
                    pos_test = low_popular_test_set
                elif idx == 1:
                    pos_test = med_popular_test_set
                else:
                    pos_test = high_popular_test_set

                n_split_list = len(split_list)

                n_pos = min(n_split_list, len(pos_test))
                n_neg = n_split_list - n_pos

                ideal_relevance_score = [1] * n_pos + [0] * n_neg

                if len(split_list) > 0:
                    accumulator.hitrate_by_pop[idx] += len(H_u) / min(n_split_list, top_k)
                    accumulator.recalled_by_pop[idx] += len(H_u)

                    idcg_at_k = np.sum([ideal_relevance_score[i] / np.log2(i + 2) for i in range(n_split_list)])  # ideal

                    if idcg_at_k == 0:
                        ndcg_at_k = 0
                    else:
                        actual_relevance_score = [1 if i in pos_test else 0 for i in ranked_in_split_list]
                        dcg_at_k = np.sum([actual_relevance_score[i] / np.log2(i + 2) for i in range(n_split_list)])
                        ndcg_at_k = dcg_at_k / idcg_at_k

                    if ndcg_at_k > 1:
                        print("nDCG by pop > 1")
                        exit(0)

                    accumulator.ndcg_by_pop[idx] += ndcg_at_k
                    accumulator.hitrate_by_pop_users[idx] += 1


            # TODO Old metrics
            # print('user:', user)
            # print('pos:', pos[user])
            # print('neg:', neg[user])
            neg_scores = sorted(score[neg[user]].tolist(), reverse=True)
            score_top_k = neg_scores[top_k - 1]

            for pos_item in predicted_item:
                score_pos = score[pos_item]
                current_popularity = popularity[pos_item]

                if current_popularity <= popularity_thresholds[0]:
                    idx = 0
                elif popularity_thresholds[0] < current_popularity <= popularity_thresholds[1]:
                    idx = 1
                else:
                    idx = 2

                accumulator.positives_by_pop[idx] += 1
                accumulator.luciano_occurencies[pos_item] += 1
                if score_pos > score_top_k:
                    accumulator.luciano_stat_by_pop[idx] += 1
                    # accumulator.weighted_luciano_stat += (1 - current_popularity)
                    # accumulator.luciano_weighted_stat += (1 / current_popularity)
                    accumulator.luciano_guessed_items[pos_item] += 1

        p_rs_low = np.sum([x for x in accumulator.users_p_r_low if x >= 0])
        p_rs_med = np.sum([x for x in accumulator.users_p_r_med if x >= 0])
        p_rs_high = np.sum([x for x in accumulator.users_p_r_high if x >= 0])

        accumulator.ps_rs_low += p_rs_low
        accumulator.ps_rs_med += p_rs_med
        accumulator.ps_rs_high += p_rs_high

        accumulator.arp += np.sum(accumulator.users_arp)
        accumulator.positive_arp += np.sum(accumulator.users_positive_arp)
        accumulator.negative_arp += np.sum(accumulator.users_negative_arp)

        accumulator.aplt += np.sum(accumulator.users_aplt)
        accumulator.aclt += np.sum(accumulator.users_aclt)


if __name__ == '__main__':
    acc = MetricAccumulator()

    x = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape((2, 5))
    y = np.array([.7, .9, .1, .1, .1, .9, .9, .9, .1, .1]).reshape((2, 5))
    pos = [[0, 1], [0, 1, 2]]
    neg = [[3, 4], [3, 4]]
    popularity_thresholds = [.4, .8]
    popularity = [.1, .1, .9, .4, .5]

    acc.compute_metric(x, y, pos, neg, popularity, popularity_thresholds, 1)
    acc.compute_metric(x, y, pos, neg, popularity, popularity_thresholds, 2)

    print(acc.get_metrics())

    for k, values in acc.get_metrics().items():
        for v in values.metric_names():
            print(f'{v}@{k} = {values[v]}')