import numpy as np


def compute_metrics(testset, recom_list, popularity, popularity_thresholds, top_k=5):
    """
    Compute metric:
    - hitrate
    - hitrate popularity based (low, medium, high)
    
    :param testset: user preferences as dict(user_id=list), BS x items
    :param recom_list: recommendation list as dict(user_id=list), i.e. the model output ordered by score desc
    :param popularity: dict of item popularity (normalized)
    :param popularity_thresholds: popularity thresholds tuple
    :param top_k: top k items to select ranked by score
    :return: hitrate, hitrate_{low, medium, high}_array
    """
    assert len(testset) == len(recom_list), "recommendation size must be equals to testset"
    assert len(popularity) > 0, "popularity must be provided"
    assert len(popularity_thresholds) == 2 and popularity_thresholds[0] < popularity_thresholds[1], "2 thresholds must be provided"

    n_users = len(testset)
    n_users_pop = [0, 0, 0]
    avg_hr = 0
    abs_hits = [0, 0, 0]

    for user_id, positive_items in testset.items():
        hr = 0
        hits = [0, 0, 0]
        flag = [False, False, False]

        # cnt users by popularity category
        for current_item in positive_items:
            current_popularity = popularity[current_item]

            if current_popularity <= popularity_thresholds[0]:
                flag[0] = True
            elif popularity_thresholds[0] < current_popularity <= popularity_thresholds[1]:
                flag[1] = True
            else:  # current_popularity > popularity_thresholds[1]
                flag[2] = True

            if np.all(flag):
                break

        for i, f in enumerate(flag):
            if f:
                n_users_pop[i] += 1

        # Get the top n indices
        arg_index = recom_list[user_id]
        if len(arg_index) == 0:
            print('NO recommendations for user', user_id)
            continue

        arg_index = arg_index[:min(top_k, len(arg_index))]

        for current_position, current_item in enumerate(arg_index):
            # global
            if current_item in positive_items:
                current_popularity = popularity[current_item]

                if current_popularity <= popularity_thresholds[0]:
                    hits[0] += 1
                elif popularity_thresholds[0] < current_popularity <= popularity_thresholds[1]:
                    hits[1] += 1
                else:  # current_popularity > popularity_thresholds[1]
                    hits[2] += 1

                hr += 1
                assert (hr == hits[0] + hits[1] + hits[2])

        avg_hr += hr / min(top_k, len(positive_items))

        for i in range(3):
            abs_hits[i] += hits[i] / min(top_k, len(positive_items))
            assert 0 <= hits[i] / min(top_k, len(positive_items)) <= 1
            assert hits[i] == 0 or (hits[i] > 0 and flag[i])

    avg_hr /= n_users
    for i in range(3):
        abs_hits[i] = abs_hits[i] / n_users_pop[i] if n_users_pop[i] > 0 else 0

    return avg_hr, abs_hits


def compute_thresholds(popularity):
    t = popularity.copy()
    t.sort()
    a = t[int(t.shape[0] * .33)], t[int(t.shape[0] * .66)]
    print('thresholds', a)

    first, third = t[t < a[0]].shape[0], t[t >= a[1]].shape[0]
    print('instances', first, popularity.shape[0] - first - third, third)

    return a


def build_recom_list(positives):
    part1 = np.random.choice(positives, (recom_size + 1) // 2)
    part2 = np.random.randint(1, n_items, (recom_size + 1) // 2)

    return np.concatenate((part1, part2))


n_items = 1000 + 1
n_users = 10
positive_items_number = 20
recom_size = 10

popularity = np.linspace(1, n_items * 10, n_items)
popularity /= popularity.sum()

pop_threasholds = compute_thresholds(popularity)

testset = {i + 1: np.random.randint(1, n_items, positive_items_number + 1) for i in range(n_users)}

recom_list = {i + 1: build_recom_list(testset[i + 1]) for i in range(n_users)}

compute_metrics(testset, recom_list, popularity, pop_threasholds)

popularity = [.1, .1, .1, .2, .2, .2, .2, .4, .4, .4]

pop_threasholds = (.2, .3)

testset = {1: [1, 2, 3, 4, 5], 2: [1, 2, 3, 4, 5], 3: [8, 9]}

recom_list = {1: [1], 2: [6], 3: [9]}

target_hr = 2 / 3
target_pop = np.array([.5, 0, 1.])

hr, hr_pop = compute_metrics(testset, recom_list, popularity, pop_threasholds, top_k=1)

print(hr, hr_pop)

assert hr == target_hr, "fail HR test"
assert np.all(hr_pop == target_pop), "fail HR-sPop test"
