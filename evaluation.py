import collections
import numpy as np


class SimpleMetric:
    def __init__(self):
        self.recall = 0
        self.recall_den = 0
        self.weighted_recall = 0
        self.weighted_recall_den = 0
        self.hitrate = 0
        self.weighted_hitrate = 0
        self.hitrate_by_pop = np.zeros(3)
        self.hitrate_by_pop_users = np.zeros(3)
        self.recalled_by_pop = np.zeros(3)
        self.positives_by_pop = np.zeros(3)
        self._num_users = 0

        self.new_stat_by_pop = np.zeros(3)
        self.new_weighted_stat = 0
        self.new_occurencies = None
        self.new_guessed_items = None

    def metric_names(self):
        return ('recall',
                'weighted_recall',
                'hitrate',
                'weighted_hitrate',
                'hitrate_by_pop',
                'recalled_by_pop',
                'positives_by_pop',
                'new_weighted_stat',
                'new_stat',
                'new_stat_by_pop',
                'new_recalled_by_pop')

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
                computed_acc.hitrate_by_pop = acc.hitrate_by_pop / acc.hitrate_by_pop_users
                computed_acc.recalled_by_pop = acc.recalled_by_pop
                computed_acc.positives_by_pop = acc.positives_by_pop

                computed_acc.new_recalled_by_pop = acc.new_stat_by_pop
                computed_acc.new_stat_by_pop = acc.new_stat_by_pop / acc.positives_by_pop
                computed_acc.new_stat = acc.new_stat_by_pop.sum() / acc.positives_by_pop.sum()
                nz = np.nonzero(acc.new_occurencies)[0]

                computed_acc.new_weighted_stat = np.average(acc.new_guessed_items[nz] / acc.new_occurencies[nz])

                result[k] = computed_acc

        return result[top_k] if top_k else result

    def compute_metric(self, x, y, pos, neg, popularity, popularity_thresholds, top_k=10):
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

            # TODO (Verificare)
            # C'e' un errore nell'istruzione seguente!
            # si usa come valore per l'ordinamento l'indice e non lo score!
            # -------------------------------------------------------------------------
            # P_u_c = sorted(list(predicted_item), key=lambda i: ranked_idx[i])[:top_k]
            # -------------------------------------------------------------------------

            P_u_c = sorted(list(predicted_item), key=lambda i: -score[i])[:top_k]

            weights_H_u = sum([1 - popularity[i] for i in H_u])  # Numeratore
            weights_P_u_c = sum([1 - popularity[i] for i in P_u_c])  # Denominatore

            accumulator = self.data[top_k]
            if accumulator.new_occurencies is None:
                accumulator.new_occurencies = np.zeros(len(popularity))
                accumulator.new_guessed_items = np.zeros(len(popularity))

            accumulator._num_users += 1

            accumulator.recall += len(H_u)  # ok
            accumulator.recall_den += min(len(predicted_item), top_k)  # ok
            accumulator.weighted_recall += weights_H_u  # ok
            accumulator.weighted_recall_den += weights_P_u_c  # ok

            accumulator.hitrate += len(H_u) / min(top_k, len(predicted_item))  # ok
            accumulator.weighted_hitrate += weights_H_u / weights_P_u_c  # ok

            pop_dict = {0: [], 1: [], 2: []}

            # TODO possibile errore nel prossimo for:
            # non dobbiamo considerare solo i topK predetti e non TUTTI gli oggetti?
            # --------------------
            # for i in ranked_idx:
            # --------------------

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

                # TODO
                # ------------------------------------------------------------
                # H_u = [i for i in predicted_item if i in split_list[:top_k]]
                # ------------------------------------------------------------

                H_u = [i for i in predicted_item if i in split_list]

                # print(f'H_u_{idx} / P_u = {len(H_u)} / {min(len(split_list), top_k)} = {len(H_u) / min(len(split_list), top_k)}')
                if len(split_list) > 0:
                    accumulator.hitrate_by_pop[idx] += len(H_u) / min(len(split_list), top_k)
                    accumulator.recalled_by_pop[idx] += len(H_u)

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
                accumulator.new_occurencies[pos_item] += 1
                if score_pos > score_top_k:
                    accumulator.new_stat_by_pop[idx] += 1
                    accumulator.new_guessed_items[pos_item] += 1


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