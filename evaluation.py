import numpy as np

def compute_metric(x, y, pos, neg, popularity, popularity_thresholds, top_k=10):
    """
    Compute metric:
    - hitrate
    - hitrate per popularity
    - weighted hit rate

    :param x: user preferences, BS x items
    :param y: user predictions, BS x items
    :param mask: user selected preferences, BS x 100
    :param popularity: dict of item popularity (normalized)
    :param top_k: top k items to select ranked by score
    :return: hitrate, popularity hitrate, total_positives(low,medium,high), weighted_hr
    """
    assert min([len(r) for r in neg]) >= top_k and top_k > 0, f"fail with top_k = {top_k} and neg = {[len(r) for r in neg]}"
    assert len(popularity) == y.shape[-1], f'{len(popularity)} != {y.shape[-1]}'
    avg_hr = 0
    weighted_hr = 0
    total_positives = np.zeros(3)
    avg_hits = np.zeros(3)

    for i in range(y.shape[0]):
        input_idx = np.where(x[i, :] == 1)[0]
        score = y[i, :]

        viewed_item = set(input_idx)
        # print('LC > viewed_item:',viewed_item)
        positive_items = set(pos[i])
        negative_items = neg[i]
        neg_scores = sorted(score[negative_items].tolist(), reverse=True)

        # Tutti i positivi predetti meno quelli visti
        predicted_item = positive_items - viewed_item
        hit = 0
        weight_hit = 0
        hit_pop = [0, 0, 0]
        hit_pop_tot = [0, 0, 0]

        for pos_item in predicted_item:
            score_pos = score[pos_item]
            current_popularity = popularity[pos_item]
            score_top_k = neg_scores[top_k - 1]

            if score_pos > score_top_k:
                hit += 1
                weight_hit += 1 - current_popularity

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

        assert hit <= len(predicted_item), f'{hit} / {len(predicted_item)}'
        assert sum(hit_pop) == hit, f'hit count error {hit} != {hit_pop}'
        assert len(predicted_item) == sum(hit_pop_tot), f'hit count error {len(predicted_item)} != {hit_pop_tot}'
        # avg_hr += hit / len(predicted_item)
        # avg_hits += np.array([a / b if a > 0 else 0 for a, b in zip(hit_pop, hit_pop_tot)])
        avg_hr += hit
        weighted_hr += weight_hit
        avg_hits += np.array(hit_pop)
        total_positives += np.array(hit_pop_tot)

    return avg_hr, avg_hits, total_positives, weighted_hr
