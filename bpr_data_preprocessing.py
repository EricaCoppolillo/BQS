import os
import pickle
import copy
import tqdm

base_dir = "./"
data_dir = os.path.join(base_dir, "data")
datasets = [elem for elem in os.listdir(data_dir) if not elem.startswith(".")]

for IDX in range(len(datasets)):

    if not os.path.exists(os.path.join(data_dir, datasets[IDX], "data_rvae.pickle")):
        print(f"Skipping: {datasets[IDX]}")
        continue

    with open(os.path.join(data_dir, datasets[IDX], "data_rvae.pickle"), "rb") as f:
        data = pickle.load(f)

    # checking the three splits do not share any user
    val_users = set(list(data["validation_data"].keys()))
    tr_users = set(list(data["training_data"].keys()))
    te_users = set(list(data["test_data"].keys()))
    assert len(val_users.intersection(tr_users)) == 0
    assert len(val_users.intersection(te_users)) == 0
    assert len(te_users.intersection(tr_users)) == 0

    new_data = {}
    POSITIVES_TR_IDX, NEGATIVES_IDX = 0, 2
    # goal: adding rvae fold-in into bpr training set
    new_data["training_data"] = {**data["training_data"],
                                 **{user_id: [data[split_type][user_id][POSITIVES_TR_IDX],
                                              data[split_type][user_id][NEGATIVES_IDX]]
                                    for split_type in ["validation_data", "test_data"] for user_id in data[split_type]}}

    for key in ["items", "users", "thresholds", "validation_data", "test_data", "popularity"]:
        new_data[key] = data[key]

    popularity_dict = {}
    for split_type in tqdm.tqdm(["training_data", "validation_data", "test_data"]):
        pop = [0] * new_data["items"]
        for user_id in new_data[split_type]:
            for item_id in new_data[split_type][user_id][0]:
                pop[item_id] += 1
        popularity_dict[split_type[:-5]] = list(map(lambda x: x / len(new_data[split_type]), pop))

    new_data["popularity_dict"] = popularity_dict
    new_data["original_training_size"] = len(data["training_data"])
    new_data["original_val_size"] = len(data["validation_data"])
    with open(os.path.join(data_dir, datasets[IDX], "data_bpr.pickle"), 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
