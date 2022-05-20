from tqdm import tqdm
import os
import pickle

DEBUG = True

base_dir = "./"
data_dir = os.path.join(base_dir, "data")
datasets = [elem for elem in os.listdir(data_dir) if not elem.startswith(".")]
print(datasets)

for idx in range(len(datasets)):
    print(f"Currently analyzing: {datasets[idx]}")
    oldversion=os.path.join(data_dir, datasets[idx], "old_data_rvae.pickle")

    if not os.path.exists(os.path.join(data_dir, datasets[idx], "data_rvae.pickle")):
       print('DATASET NOT VALID', os.path.join(data_dir, datasets[idx], "data_rvae.pickle"))
       continue

    if DEBUG:
        print('PROCESS', oldversion)
        continue

    if not os.path.exists(oldversion):
        print('rename old version')
        os.rename(os.path.join(data_dir, datasets[idx], "data_rvae.pickle"), oldversion)

    with open(oldversion, "rb") as f:
        data = pickle.load(f)
    f = data["popularity"]
    popularity_dict = {}
    train_popularity = [0] * len(f)
    for key in tqdm(data["training_data"]):
        for item_id in data["training_data"][key][0]:
            train_popularity[item_id] += 1
    # normalization
    popularity_dict["training"] = [elem / len(data["training_data"]) for elem in train_popularity]

    val_popularity = [0] * len(f)
    for key in tqdm(data["validation_data"]):
        for item_id in data["validation_data"][key][0]:
            val_popularity[item_id] += 1
    # normalization
    popularity_dict["validation"] = [elem / len(data["validation_data"]) for elem in val_popularity]

    test_popularity = [0] * len(f)
    for key in tqdm(data["test_data"]):
        for item_id in data["test_data"][key][0]:
            test_popularity[item_id] += 1
    # normalization
    popularity_dict["test"] = [elem / len(data["test_data"]) for elem in test_popularity]
    data["popularity_dict"] = popularity_dict

    with open(os.path.join(data_dir, datasets[idx], "data_rvae.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
