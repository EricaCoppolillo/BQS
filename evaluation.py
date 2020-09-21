import collections
import numpy as np

class SimpleMetric:
    def __init__(self):
        self.recall = 0
        self.weighted_recall = 0
        self.hitrate = 0
        self.weighted_hitrate = 0
        self.recall_by_pop = np.zeros(3)

        self._total_positives = np.zeros(3)
        self._num_users = 0
        self._total_weights = .0

    def metric_names(self):
        return ('recall', 'weighted_recall',
                'hitrate', 'weighted_hitrate',
                'recall_by_pop')

    def __getitem__(self, item):
        v = getattr(self, item)
        if isinstance(v, np.ndarray):
            return ','.join([f'{j:.2f}' for j in v])
        return v

    def __str__(self):
        return f'recall = {self.recall:.2f}, weighted_recall = {self.weighted_recall:.2f}, ' \
               f'hitrate = {self.hitrate:.2f}, weighted_hitrate = {self.weighted_hitrate:.2f} ' \
               f'recall_by_pop = {[f"{x:.2f}" for x in self.recall_by_pop]}'

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
                computed_acc.recall = acc.recall / acc._total_positives.sum()
                computed_acc.weighted_recall = acc.weighted_recall / acc._total_weights
                computed_acc.hitrate = acc.hitrate / acc._num_users
                computed_acc.weighted_hitrate = acc.weighted_hitrate / acc._num_users
                computed_acc.recall_by_pop = acc.recall_by_pop / acc._total_positives
                computed_acc.recall_by_pop[np.isnan(computed_acc.recall_by_pop)] = 0

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

        :param x: user preferences, BS x items
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

        for i in range(y.shape[0]):
            input_idx = np.where(x[i, :] == 1)[0]
            score = y[i, :]

            viewed_item = set(input_idx)
            positive_items = set(pos[i])
            negative_items = neg[i]
            neg_scores = sorted(score[negative_items].tolist(), reverse=True)

            # Tutti i positivi predetti meno quelli visti
            predicted_item = positive_items - viewed_item
            recall = 0
            weighted_recall = 0
            hit = 0
            weighted_hit = 0
            weights = .0
            hit_pop = [0, 0, 0]
            hit_pop_tot = [0, 0, 0]

            for pos_item in predicted_item:
                score_pos = score[pos_item]
                current_popularity = popularity[pos_item]
                score_top_k = neg_scores[top_k - 1]

                weights += 1 - current_popularity

                if score_pos > score_top_k:
                    recall += 1
                    weighted_recall += 1 - current_popularity
                    hit += 1
                    weighted_hit += 1 - current_popularity

                # popularity
                if current_popularity <= popularity_thresholds[0]:

                    hit_pop_tot[0] += 1
                    if score_pos > score_top_k:
                        hit_pop[0] += 1

                elif popularity_thresholds[0] < current_popularity <= popularity_thresholds[1]:

                    hit_pop_tot[1] += 1
                    if score_pos > score_top_k:
                        hit_pop[1] += 1

                else:  # current_popularity > popularity_thresholds[1]

                    hit_pop_tot[2] += 1
                    if score_pos > score_top_k:
                        hit_pop[2] += 1

                if recall == top_k:
                    break

            assert hit <= len(predicted_item), f'{hit} / {len(predicted_item)}'
            assert sum(hit_pop) == hit, f'hit count error {hit} != {hit_pop}'
            # assert len(predicted_item) == sum(hit_pop_tot), f'hit count error {len(predicted_item)} != {hit_pop_tot}'

            accumulator = self.data[top_k]
            accumulator._total_positives += np.array(hit_pop_tot)
            accumulator._num_users += 1
            accumulator._total_weights += weights
            accumulator.recall += recall
            accumulator.weighted_recall += weighted_recall
            accumulator.hitrate += hit / min(top_k, len(predicted_item))
            accumulator.weighted_hitrate += weighted_hit / weights
            accumulator.recall_by_pop += np.array(hit_pop)


if __name__ == '__main__':
    acc = MetricAccumulator()

    x = np.array([1,0,0,0,0,1,0,0,0,0]).reshape((2, 5))
    y = np.array([.7,.9,.1,.1,.1,.9,.9,.9,.1,.1]).reshape((2, 5))
    pos = [[0,1], [0,1,2]]
    neg = [[3,4], [3,4]]
    popularity_thresholds = [.4, .8]
    popularity = [.1, .1, .9, .4, .5]

    acc.compute_metric(x, y, pos, neg, popularity, popularity_thresholds, 1)
    acc.compute_metric(x, y, pos, neg, popularity, popularity_thresholds, 2)

    print(acc.get_metrics())

    for k, values in acc.get_metrics().items():
        for v in values.metric_names():
            print(f'{v}@{k} = {values[v]}')
