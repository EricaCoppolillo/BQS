import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm as tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-contamination_perc", help="percentage of foldout tuples to switch with foldin",
                    type=float, default=0.1)
parser.add_argument("-seed", help="random seed",
                    type=int, default=12121995)
args = parser.parse_args()
seed = args.seed
np.random.seed(seed)
import random
random.seed(seed)

base_dir = "./"
data_dir = os.path.join(base_dir, "data")
yahoo_data_dir = os.path.join(data_dir, "yahoo-r3")

train_file_name = "ydata-ymusic-rating-study-v1_0-train.txt"
test_file_name = "ydata-ymusic-rating-study-v1_0-test.txt"

train_df = pd.read_csv(os.path.join(yahoo_data_dir, train_file_name), sep="\t", index_col=None, header=None)
train_df.columns = ["user", "item", "rating"]
test_df = pd.read_csv(os.path.join(yahoo_data_dir, test_file_name), sep="\t", index_col=None, header=None)
test_df.columns = ["user", "item", "rating"]
train_df.head()


def check_train_contains_test_users_and_items(train_df, test_df):
    # are all items in test present in train
    assert len(set(test_df.user.unique()) - set(train_df.user.unique())) == test_df.user.nunique(), len(
        set(test_df.user.unique()) - set(train_df.user.unique()))
    assert len(set(test_df.item.unique()) - set(train_df.item.unique())) == 0, len(
        set(test_df.item.unique()) - set(train_df.item.unique()))


def check_users_and_item_frequency_constraint(train_df, min_users=5, min_items=5):
    assert train_df.groupby("item").count()['rating'].min() >= min_items, train_df.groupby("item").count()[
        'rating'].min()
    assert train_df.groupby("user").count()['rating'].min() >= min_users, train_df.groupby("user").count()[
        'rating'].min()


def print_basic_statistics(train_df, test_df):
    train_nusers = train_df.user.nunique()
    test_nusers = test_df.user.nunique()
    train_nitems = train_df.item.nunique()
    test_nitems = test_df.item.nunique()

    print(f"Training set consists of {train_nusers} users and {train_nitems} items")
    print(f"Test set consists of {test_nusers} users and {test_nitems} items")
    print(f"Number of ratings in the training set: {len(train_df.index)} "
          f"(sparsity: {round(len(train_df.index) / (train_nusers * train_nitems), 4)})")
    print(f"Number of ratings in the test set: {len(test_df.index)} "
          f"(sparsity: {round(len(test_df.index) / (test_nusers * test_nitems), 4)})")


print_basic_statistics(train_df, test_df)

# from explicit to implicit feedback, a rating > 3 is considered as a positive
POSITIVE_TH = 3
train_implicit_df = train_df[train_df.rating > POSITIVE_TH]
test_implicit_df = test_df[test_df.rating > POSITIVE_TH]

print_basic_statistics(train_implicit_df, test_implicit_df)

# 1st assumption: training and test sets must not share any user
tr_users = train_implicit_df.user.unique()
te_users = test_implicit_df.user.unique()
old_tr_nratings = len(train_implicit_df.index)
old_te_nratings = len(test_implicit_df.index)

shared_users = set(te_users).intersection(set(tr_users))
test_implicit_df = pd.concat([train_implicit_df[train_implicit_df.user.isin(shared_users)], test_implicit_df])
train_implicit_df = train_implicit_df.drop(train_implicit_df[train_implicit_df.user.isin(shared_users)].index)
tr_nratings = len(train_implicit_df.index)
te_nratings = len(test_implicit_df.index)
tr_users = train_implicit_df.user.unique()
te_users = test_implicit_df.user.unique()

# adding a flag for ratings moved from train to test
moved_records_flag = np.full(shape=te_nratings, fill_value=False)
moved_records_flag[:(te_nratings - old_te_nratings)] = True
test_implicit_df["moved_records"] = moved_records_flag

assert tr_nratings + te_nratings == old_tr_nratings + old_te_nratings
assert len(set(te_users).intersection(set(tr_users))) == 0

# 2nd assumption: training set must contain users interacting with at least 5 items and items interacting
# with at least 5 users.
# deleting items interacting less than 5 times

items_interacting_less_than_5_times = train_implicit_df.groupby("item").count().index \
    [train_implicit_df.groupby("item").count()['rating'] < 5]
train_implicit_df = train_implicit_df[~train_implicit_df.item.isin(items_interacting_less_than_5_times)]

# do the same for users
users_interacting_less_than_5_times = train_implicit_df.groupby("user").count().index \
    [train_implicit_df.groupby("user").count()['rating'] < 5]
train_implicit_df = train_implicit_df[~train_implicit_df.user.isin(users_interacting_less_than_5_times)]

# repeat for the items
items_interacting_less_than_5_times = train_implicit_df.groupby("item").count().index \
    [train_implicit_df.groupby("item").count()['rating'] < 5]
train_implicit_df = train_implicit_df[~train_implicit_df.item.isin(items_interacting_less_than_5_times)]
check_users_and_item_frequency_constraint(train_implicit_df, min_users=5, min_items=5)

# 3rd assumption: test must contain users interacting with at least 5 items (RVAE history reconstruction)
users_interacting_less_than_5_times = test_implicit_df.groupby("user").count().index \
    [test_implicit_df.groupby("user").count()['rating'] < 5]
test_implicit_df = test_implicit_df[~test_implicit_df.user.isin(users_interacting_less_than_5_times)]
check_users_and_item_frequency_constraint(test_implicit_df, min_users=5, min_items=0)

# 4th assumption: train must contains all items in the test set.
items_not_present_in_train = set(test_implicit_df.item.unique()) - set(train_implicit_df.item.unique())
test_implicit_df = test_implicit_df[~test_implicit_df.item.isin(items_not_present_in_train)]

check_train_contains_test_users_and_items(train_implicit_df, test_implicit_df)

# iterating the last two chunks for a time
# 3rd assumption: test must contain users interacting with at least 5 items (RVAE history reconstruction)
users_interacting_less_than_5_times = test_implicit_df.groupby("user").count().index \
    [test_implicit_df.groupby("user").count()['rating'] < 5]
test_implicit_df = test_implicit_df[~test_implicit_df.user.isin(users_interacting_less_than_5_times)]
check_users_and_item_frequency_constraint(test_implicit_df, min_users=5, min_items=0)

# 4th assumption: train must contains all items in the test set.
items_not_present_in_train = set(test_implicit_df.item.unique()) - set(train_implicit_df.item.unique())
test_implicit_df = test_implicit_df[~test_implicit_df.item.isin(items_not_present_in_train)]

check_train_contains_test_users_and_items(train_implicit_df, test_implicit_df)

print_basic_statistics(train_implicit_df, test_implicit_df)
check_train_contains_test_users_and_items(train_implicit_df, test_implicit_df)
check_users_and_item_frequency_constraint(train_implicit_df, min_users=5, min_items=5)
check_users_and_item_frequency_constraint(test_implicit_df, min_users=5, min_items=0)

# contamination
train_implicit_df.reset_index(inplace=True)
test_implicit_df.reset_index(inplace=True)

amount_of_contamination = args.contamination_perc
if amount_of_contamination > 0:
    unbiased_records = test_implicit_df[~test_implicit_df.moved_records]
    no_of_unbiased_records = len(unbiased_records.index)
    number_of_unbiased_records_to_sample = int(no_of_unbiased_records * amount_of_contamination)
    unbiased_idxs_to_switch = unbiased_records.sample(number_of_unbiased_records_to_sample, random_state=seed).index
    biased_idxs_to_switch = test_implicit_df[test_implicit_df.moved_records] \
        .sample(number_of_unbiased_records_to_sample, random_state=seed).index

    assert all(test_implicit_df.loc[biased_idxs_to_switch].moved_records.values.tolist())
    assert all(test_implicit_df[test_implicit_df.moved_records].moved_records.values.tolist())
    assert not all(test_implicit_df.loc[unbiased_idxs_to_switch].moved_records.values.tolist())
    assert not all(test_implicit_df[~test_implicit_df.moved_records].moved_records.values.tolist())

    # {True, False} -> {Biased, Unbiased}, hence to contaminate the data we switch them for a part of the records
    test_implicit_df.loc[unbiased_idxs_to_switch, "moved_records"] = True
    test_implicit_df.loc[biased_idxs_to_switch, "moved_records"] = False

    assert not all(test_implicit_df.loc[biased_idxs_to_switch].moved_records.values.tolist())
    assert all(test_implicit_df.loc[unbiased_idxs_to_switch].moved_records.values.tolist())

tr_nusers = train_implicit_df.user.nunique()

# remap userId and movieId to 0...N values
tr_user_new_mapping = {val: idx for idx, val in enumerate(train_implicit_df.user.unique())}
item_new_mapping = {val: idx for idx, val in enumerate(train_implicit_df.item.unique())}
train_implicit_df["user"] = train_implicit_df["user"].map(tr_user_new_mapping)
train_implicit_df["item"] = train_implicit_df["item"].map(item_new_mapping)

te_user_new_mapping = {val: idx + tr_nusers for idx, val in enumerate(test_implicit_df.user.unique())}
test_implicit_df["user"] = test_implicit_df["user"].map(te_user_new_mapping)
test_implicit_df["item"] = test_implicit_df["item"].map(item_new_mapping)

data_rvae = dict()
data_rvae["users"] = train_implicit_df["user"].nunique() + test_implicit_df["user"].nunique()
data_rvae["items"] = len(train_implicit_df["item"].unique())


def select_negatives(positive_items, all_items, N=100):
    negative_items = all_items - positive_items
    return np.random.permutation(list(negative_items))[:N]


all_items = set(train_implicit_df.item.unique())

print("Processing training data...")
data_rvae["training_data"] = {user_id: [train_implicit_df[train_implicit_df.user == user_id].item.unique().tolist(),
                                        select_negatives(
                                            set(train_implicit_df[train_implicit_df.user == user_id].item.unique()),
                                            all_items)]
                              for user_id in train_implicit_df.user.unique()}


# def fold_in_out_split(history, fold_out_size=.3):
#    fold_in, fold_out, _, _ = train_test_split(history, history, test_size=fold_out_size, random_state=1)
#    return [fold_in, fold_out]

def fold_in_out_split(history, fold_out_size=.3):
    fold_in = history[history.moved_records].item.unique().tolist()
    fold_out = history[~history.moved_records].item.unique().tolist()
    return [fold_in, fold_out]


print("Processing validation data...")
NVAL = test_implicit_df.user.nunique() // 2  # number of users to include in validation set
validation_users = np.random.permutation(test_implicit_df.user.unique())[:NVAL]
validation_implicit_df = test_implicit_df[test_implicit_df.user.isin(validation_users)]
data_rvae["validation_data"] = {
    user_id: fold_in_out_split(validation_implicit_df[validation_implicit_df.user == user_id]) +
             [select_negatives(set(validation_implicit_df[validation_implicit_df.user == user_id].item.unique()),
                               all_items)]
    for user_id in validation_users}
print("Processing test data...")
test_implicit_df = test_implicit_df[~test_implicit_df.user.isin(validation_users)]
data_rvae["test_data"] = {user_id: fold_in_out_split(test_implicit_df[test_implicit_df.user == user_id]) +
                                   [select_negatives(
                                       set(test_implicit_df[test_implicit_df.user == user_id].item.unique()),
                                       all_items)]
                          for user_id in test_implicit_df.user.unique()}

train_pop = [0] * data_rvae["items"]
train_count = train_implicit_df.groupby("item").count()["user"]
for idx in tqdm(train_count.index, "train"):
    train_pop[idx] = train_count.loc[idx]

val_pop = [0] * data_rvae["items"]
val_count = validation_implicit_df.groupby("item").count()["user"]
for idx in tqdm(val_count.index, "val"):
    val_pop[idx] = val_count.loc[idx]

test_pop = [0] * data_rvae["items"]
test_count = test_implicit_df.groupby("item").count()["user"]
for idx in tqdm(test_count.index, "test"):
    test_pop[idx] = test_count.loc[idx]

print("Computing popularity for each item")
joined_data = pd.concat([train_implicit_df, validation_implicit_df, test_implicit_df])
pop = joined_data.groupby("item").count()["user"].values.tolist()
assert joined_data.groupby("item").count().index.tolist() == list(range(data_rvae["items"]))
data_rvae["popularity"] = pop

print("Computing popularity for each item - by split")
data_rvae["popularity_dict"] = {"training": train_pop,
                                "validation": val_pop,
                                "test": test_pop
                                }

th_low, th_high = 20, 285  # amount of exposures which delimits low, medium and high popular items
data_rvae["thresholds"] = [th_low / data_rvae["users"], th_high / data_rvae["users"]]
# normalize popularities
data_rvae["popularity"] = list(map(lambda x: x / data_rvae["users"], data_rvae["popularity"]))
data_rvae["popularity_dict"]["training"] = list(
    map(lambda x: x / len(data_rvae["training_data"]), data_rvae["popularity_dict"]["training"]))
data_rvae["popularity_dict"]["validation"] = list(
    map(lambda x: x / len(data_rvae["validation_data"]), data_rvae["popularity_dict"]["validation"]))
data_rvae["popularity_dict"]["test"] = list(
    map(lambda x: x / len(data_rvae["test_data"]), data_rvae["popularity_dict"]["test"]))

data_rvae["contamination"] = int(100 * args.contamination_perc)
with open(os.path.join(yahoo_data_dir, "data_rvae.pickle"), 'wb') as handle:
    pickle.dump(data_rvae, handle, protocol=pickle.HIGHEST_PROTOCOL)
