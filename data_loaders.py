import os
import gc
import json

from tqdm import tqdm
from math import ceil, modf, floor
import scipy.stats as ss
from scipy.sparse import csr_matrix, vstack
from scipy.sparse import save_npz, load_npz
from pathlib import Path
import pickle
import sqlite3
from util import *
from models import *
from config import Config

model_types = Config("./model_type_info.json")

def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__, reverse=True)

def rankdata(a):
    n = len(a)
    ivec=rank_simple(a)
    svec=[a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0]*n
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

class DataLoader:
    def __init__(self, file_tr, seed, decreasing_factor, model_type, pos_neg_ratio=4, negatives_in_test=100, alpha=None,
                        gamma=None):

        dataset = load_dataset(file_tr)
        if "bpr" in file_tr:
            self.discount_idxs = {"validation": dataset["original_training_size"],
                                  "test": dataset["original_training_size"]+dataset["original_val_size"]}

        self.n_users = dataset["users"]
        self.item_popularity = dataset['popularity']
        self.item_popularity_dict = dataset['popularity_dict']

        self.n_users_dict = {"training": len(dataset["training_data"]), "validation": len(dataset["validation_data"]),
                             "test": len(dataset["test_data"])}
        self.absolute_item_popularity_dict = {split_type: [elem * self.n_users_dict[split_type]
                                                           for elem in dataset['popularity_dict'][split_type]]
                                                            for split_type in dataset['popularity_dict']
                                              }
        self.thresholds = dataset['thresholds']
        self.absolute_thresholds = list(map(lambda x:x*self.n_users_dict["training"], self.thresholds))

        self.pos_neg_ratio = pos_neg_ratio
        self.negatives_in_test = negatives_in_test
        self.model_type = model_type

        # IMPROVEMENT
        self.low_pop = len([i for i in self.item_popularity_dict["training"] if i <= self.thresholds[0]])
        self.med_pop = len([i for i in self.item_popularity_dict["training"] if self.thresholds[0] < i <= self.thresholds[1]])
        self.high_pop = len([i for i in self.item_popularity_dict["training"] if self.thresholds[1] < i])

        # compute the Beta parameter according to the item popularity
        if self.model_type == model_types.REWEIGHTING:
            self.absolute_item_popularity = np.array(self.absolute_item_popularity_dict["training"])
            # Beta is defined as the average popularity of the medium-popular class of items
            self.beta = self.absolute_item_popularity[(self.absolute_item_popularity >= self.absolute_thresholds[0]) &
                                                 (self.absolute_item_popularity < self.absolute_thresholds[1])].mean()
            assert self.beta > 0, self.absolute_thresholds
            self.gamma = gamma
            self.alpha = alpha

        self.n_items = len(self.item_popularity)
        self.use_popularity = self.model_type in (model_types.LOW, model_types.MED, model_types.HIGH,
                                                  model_types.OVERSAMPLING)
        self.sorted_item_popularity = sorted(self.item_popularity)
        limit = 1
        self.max_popularity = self.sorted_item_popularity[-limit]
        self.min_popularity = self.sorted_item_popularity[0]

        self.decreasing_factor = decreasing_factor
        n = self.pos_neg_ratio
        self.frequencies_dict = {}
        self.freq_decimal_part_dict = {}
        self.counter_for_decimal_part_dict = {}
        self.slots_available_for_decimal_part_dict = {}
        for split_type in ["training", "validation", "test"]:
            frequencies = []
            freq_decimal_part = []
            item_popularity = self.item_popularity_dict[split_type]
            # computing item ranking
            item_ranking = rankdata(item_popularity)
            max_popularity = max(item_popularity)
            nusers = self.n_users_dict[split_type]
            for idx in range(len(item_popularity)):
                f_i = item_popularity[idx]
                # d_i = ceil(np.log2(ceil((item_ranking[idx] / self.high_pop) + 1) + 1))
                d_i = ceil((item_ranking[idx]/self.high_pop) + 1)
                # d_i = self.decreasing_factor
                if f_i > 0:
                    n_i_decimal, n_int = modf(n * (max_popularity / (d_i * f_i)))
                    frequencies.append(int(n_int))
                    freq_decimal_part.append(n_i_decimal)
                else:
                    frequencies.append(0)
                    freq_decimal_part.append(0)
            counter_for_decimal_part = [int(int(item_popularity[idx]*nusers)*freq_decimal_part[idx]) for idx in range(len(item_popularity))] # K_i
            slots_available_for_decimal_part = [int(f_i*nusers) for f_i in item_popularity] # K

            self.frequencies_dict[split_type] = frequencies
            self.freq_decimal_part_dict[split_type] = freq_decimal_part
            self.counter_for_decimal_part_dict[split_type] = counter_for_decimal_part
            self.slots_available_for_decimal_part_dict[split_type] = slots_available_for_decimal_part

        self.item_visibility_dict = {split_type: [0]*len(self.item_popularity_dict[split_type])
                                for split_type in ["training", "validation", "test"]}


        self.max_width = -1


        print('phase 1: Loading data...')
        self._initialize()
        # checking if data have already been computed
        path = Path(file_tr)
        par_dir = path.parent.absolute()
        if "bpr" in file_tr:
            preprocessed_data_dir = os.path.join(par_dir, "preprocessed_data", "bpr", self.model_type
                                                 , f"decreasing_factor_{decreasing_factor}", str(seed))
        else:
            preprocessed_data_dir = os.path.join(par_dir, "preprocessed_data", self.model_type
                                                 , f"decreasing_factor_{decreasing_factor}", str(seed))

        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)

        def _dir_does_not_contain_files(input_path):
            return len(os.listdir(input_path)) == 0

        if _dir_does_not_contain_files(preprocessed_data_dir):
            print("Generating pre-processed data from scratch")
            self._load_data(dataset, preprocessed_data_dir)
        else:
            print("Loading pre-processed data from disk")
            tags = ["train", "validation", "test"]
            for tag in tags:
                with open(os.path.join(preprocessed_data_dir, f'pos_{tag}.pkl'), 'rb') as f:
                    self.pos[tag] = pickle.load(f)
                with open(os.path.join(preprocessed_data_dir, f'neg_{tag}.pkl'), 'rb') as f:
                    self.neg[tag] = pickle.load(f)
                self.data[tag] = np.load(os.path.join(preprocessed_data_dir, f"data_{tag}.npy"))
                self.size[tag] = self.data[tag].shape[0]
                if tag in tags[1:]:  # all except train
                    with open(os.path.join(preprocessed_data_dir, f'pos_rank_{tag}.pkl'), 'rb') as f:
                        self.pos_rank[tag] = pickle.load(f)
                    with open(os.path.join(preprocessed_data_dir, f'neg_rank_{tag}.pkl'), 'rb') as f:
                        self.neg_rank[tag] = pickle.load(f)

            with open(os.path.join(preprocessed_data_dir, f'max_width.pkl'), 'rb') as f:
                self.max_width = pickle.load(f)

        print("phase 2: converting the pos/neg list of lists to a sparse matrix for future indexing")

        def _converting_to_csr_matrix(x, input_shape, desc):
            rows = []
            cols = []
            vals = []
            for i, elem in tqdm(enumerate(x), desc=desc):  # users loop
                for j in range(len(elem)):  # pos/neg loop
                    rows.append(i)
                    cols.append(j)
                    vals.append(elem[j])
            return csr_matrix((vals, (rows, cols)), shape=input_shape, dtype=np.int32)


        if model_type == model_types.REWEIGHTING:
            def _creating_csr_mask(x, input_shape, desc):
                print("Currently using masks with reweighting")
                rows = []
                cols = []
                vals = []

                if self.alpha is not None:
                    def inverse_sigmoid_weight(item_pop, _alpha=0.01, _beta=0.002, _gamma=100):
                        return (_gamma+1)*((_gamma * (1 + np.exp(_alpha * (item_pop - _beta - 1))) ** -1 + 1) / (
                                            _gamma * (1 + np.exp(-_alpha * _beta)) ** -1 + 1))

                    w_i = [inverse_sigmoid_weight(elem, _alpha=self.alpha, _beta=self.beta, _gamma=self.gamma)
                           for elem in self.absolute_item_popularity]
                else:
                    w_i = [1/elem for elem in self.absolute_item_popularity]

                for i, elem in tqdm(enumerate(x), desc=desc):
                    for j in range(len(elem)):
                        rows.append(i)
                        cols.append(j)
                        # elem[j] è l'item ID dell'oggetto
                        vals.append(w_i[elem[j]])
                return csr_matrix((vals, (rows, cols)), shape=input_shape, dtype=np.float)
        else:
            def _creating_csr_mask(x, input_shape, desc):
                rows = []
                cols = []
                vals = []
                for i, elem in tqdm(enumerate(x), desc=desc):
                    for j in range(len(elem)):
                        rows.append(i)
                        cols.append(j)
                        vals.append(1)
                return csr_matrix((vals, (rows, cols)), shape=input_shape, dtype=np.uint8)

        # check if sparse matrices have already been computed
        if "bpr" in file_tr:
            dir_for_sparse_matrices = os.path.join(par_dir, "sparse_matrices", "bpr", self.model_type,
                                                   f"decreasing_factor_{decreasing_factor}", str(seed))
        else:
            dir_for_sparse_matrices = os.path.join(par_dir, "sparse_matrices", self.model_type,
                                                   f"decreasing_factor_{decreasing_factor}", str(seed))

        if not os.path.exists(dir_for_sparse_matrices):
            os.makedirs(dir_for_sparse_matrices)

        self.pos_sparse = dict()
        self.neg_sparse = dict()
        self.mask_sparse = dict()

        if _dir_does_not_contain_files(dir_for_sparse_matrices):
            print("Generating pos/neg/masks from scratch")
            for tag in self.pos:
                shape = [self.size[tag], self.max_width]
                self.pos_sparse[tag] = _converting_to_csr_matrix(self.pos[tag], input_shape=shape,
                                                                 desc="Positive Items")
                self.neg_sparse[tag] = _converting_to_csr_matrix(self.neg[tag], input_shape=shape,
                                                                 desc="Negative Items")
                self.mask_sparse[tag] = _creating_csr_mask(self.pos[tag], input_shape=shape, desc="Mask Items")
            # save matrices
            for tag in self.pos:
                save_npz(os.path.join(dir_for_sparse_matrices, f"pos_{tag}.npz"), self.pos_sparse[tag])
                save_npz(os.path.join(dir_for_sparse_matrices, f"neg_{tag}.npz"), self.neg_sparse[tag])
                save_npz(os.path.join(dir_for_sparse_matrices, f"mask_{tag}.npz"), self.mask_sparse[tag])
        else:
            # load matrices
            print("Loading stored pos/neg/mask matrices")
            for tag in self.pos:
                self.pos_sparse[tag] = load_npz(
                    os.path.join(dir_for_sparse_matrices, f"pos_{tag}.npz"))
                self.neg_sparse[tag] = load_npz(
                    os.path.join(dir_for_sparse_matrices, f"neg_{tag}.npz"))
                self.mask_sparse[tag] = load_npz(
                    os.path.join(dir_for_sparse_matrices, f"mask_{tag}.npz"))

        print('phase 3: generating test masks...')

        for tag in ('validation', 'test'):
            self._generate_mask_te(tag)

        print('Done.')

    def _generate_mask_te(self, tag):
        self.mask_rank[tag] = np.zeros((self.size[tag], self.n_items), dtype=np.int8)

        for row in np.arange(self.size[tag]):
            pos = self.pos_rank[tag][row]
            self.mask_rank[tag][row, pos] = 1

    def _initialize(self):
        self._count_positive = {'train': None, 'validation': None, 'test': None}
        self._count_positive_user = {'train': None, 'validation': None, 'test': None}

        self.data = dict()
        self.pos = dict()  # history of the user
        self.neg = dict()
        self.mask = dict()
        self.pos_rank = dict()  # items to predict
        self.neg_rank = dict()
        self.mask_rank = dict()
        self.size = dict()

        for tag in ('train', 'validation', 'test'):
            self.pos[tag] = []
            self.neg[tag] = []

        for tag in ('validation', 'test'):
            self.pos_rank[tag] = []
            self.neg_rank[tag] = []

    def _load_data(self, dataset, preprocessed_data_dir):
        train = []
        validation = []
        test = []

        training_data = dataset['training_data']
        validation_data = dataset['validation_data']
        test_data = dataset['test_data']

        print('LEN TEST:', len(test_data.keys()))

        for user_id in training_data:
            if user_id % 1000 == 0:
                print("{}/{}".format(user_id, len(training_data)))

            pos, neg = training_data[user_id]
            assert len(neg) >= 100, f'fail train neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[pos] = 1

            pos, neg = self._generate_pairs(pos, tag="training")

            train.append(items_np)
            self.pos['train'].append(pos)
            self.neg['train'].append(neg)

        print('self.max_width:', self.max_width)

        self.data['train'] = np.array(train, dtype=np.int8)
        self.data['train'][:] = train[:]
        self.size['train'] = len(train)

        for user_id in validation_data:
            if user_id % 1000 == 0:
                print("{}/{}".format(user_id, len(validation_data)))
            positives_tr, positives_te, negatives_sampled = validation_data[user_id]
            if len(positives_te) == 0:
                continue

            assert len(negatives_sampled) >= 100, f'fail valid neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[positives_tr] = 1
            items_np[positives_te] = 1

            pos, neg = self._generate_pairs(positives_tr, tag="validation")

            validation.append(items_np)

            self.pos['validation'].append(pos)
            self.neg['validation'].append(neg)

            self.pos_rank['validation'].append(positives_te)
            self.neg_rank['validation'].append(negatives_sampled)

        if len(validation) > 0:
            self.data['validation'] = np.array(validation, dtype=np.int8)
            self.data['validation'][:] = validation[:]
            self.size['validation'] = len(validation)

        for user_id in test_data:
            if user_id % 1000 == 0:
                print("{}/{}".format(user_id, len(test_data)))
            positives_tr, positives_te, negatives_sampled = test_data[user_id]
            if len(positives_te) == 0:
                continue

            assert len(negatives_sampled) >= 100, f'fail test neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[positives_tr] = 1
            items_np[positives_te] = 1

            pos, neg = self._generate_pairs(positives_tr, tag="test")

            test.append(items_np)

            self.pos['test'].append(pos)
            self.neg['test'].append(neg)

            self.pos_rank['test'].append(positives_te)
            self.neg_rank['test'].append(negatives_sampled)

        if len(test) > 0:
            self.data['test'] = np.array(test, dtype=np.int8)
            self.data['test'][:] = test[:]
            self.size['test'] = len(test)

        # saving artifacts on disk
        tags = ["train", "validation", "test"]
        for tag in tags:
            with open(os.path.join(preprocessed_data_dir, f'pos_{tag}.pkl'), 'wb') as f:
                pickle.dump(self.pos[tag], f)
            with open(os.path.join(preprocessed_data_dir, f'neg_{tag}.pkl'), 'wb') as f:
                pickle.dump(self.neg[tag], f)
            np.save(os.path.join(preprocessed_data_dir, f"data_{tag}.npy"), self.data[tag])
            if tag in tags[1:]:  # all except train
                with open(os.path.join(preprocessed_data_dir, f'pos_rank_{tag}.pkl'), 'wb') as f:
                    pickle.dump(self.pos_rank[tag], f)
                with open(os.path.join(preprocessed_data_dir, f'neg_rank_{tag}.pkl'), 'wb') as f:
                    pickle.dump(self.neg_rank[tag], f)

        with open(os.path.join(preprocessed_data_dir, f'max_width.pkl'), 'wb') as f:
            pickle.dump(self.max_width, f)

    def _sample_negatives(self, pos, size):
        all_items = set(range(len(self.item_popularity)))
        all_negatives = list(all_items - set(pos))

        return random.choices(all_negatives, k=size)

    def _generate_pairs(self, pos, tag=""):
        # IMPROVEMENT
        positives = []
        for item in pos:
            if self.use_popularity:
                frequency = self.frequencies_dict[tag][item]
            else:
                frequency = self.pos_neg_ratio
            if self.counter_for_decimal_part_dict[tag][item] > 0:
                if self.slots_available_for_decimal_part_dict[tag][item]-self.counter_for_decimal_part_dict[tag][item]<=0 or random.random() < self.freq_decimal_part_dict[tag][item]:
                    frequency += 1
                    self.counter_for_decimal_part_dict[tag][item] -= 1
                self.slots_available_for_decimal_part_dict[tag][item] -= 1

            self.item_visibility_dict[tag][item] += frequency
            positives[0:0] = [item] * frequency  # append at the beginning (pre-pend)

        negatives = self._sample_negatives(pos, len(positives))
        self.max_width = max(self.max_width, len(negatives))

        return positives, negatives

    def iter(self, batch_size=256, tag='train', model_type="rvae"):
        """
        Iter on data

        :param batch_size: size of the batch
        :param tag: tag in {train, validation, test} tells you from which sample to extract data
        :param model_type: if "bpr" then you also need the user indexes
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        assert (tag in ('train', 'validation', 'test'))

        idxlist = np.arange(self.data[tag].shape[0])
        np.random.shuffle(idxlist)
        N = idxlist.shape[0]
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            raw_idxs = idxlist[start_idx:end_idx]
            x = self.data[tag][raw_idxs]
            mask = self.mask_sparse[tag][raw_idxs].A
            pos = self.pos_sparse[tag][raw_idxs].A
            neg = self.neg_sparse[tag][raw_idxs].A
            if model_type=="rvae":
                yield x, pos, neg, mask
            else:
                yield x, pos, neg, mask, raw_idxs

    def iter_test(self, batch_size=256, tag='test', model_type="rvae"):
        """
        Iter on data

        mask_loss

        :param batch_size: size of the batch
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        assert (tag in ('validation', 'test'))

        N = self.size[tag]
        idxlist = np.arange(self.data[tag].shape[0])
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)

            raw_idxs = idxlist[start_idx:end_idx]

            x = self.data[tag][start_idx:end_idx]

            # generating pos, neg, mask on-the-fly
            mask = self.mask_sparse[tag][raw_idxs].A
            pos = self.pos_sparse[tag][raw_idxs].A
            neg = self.neg_sparse[tag][raw_idxs].A

            pos_te = self.pos_rank[tag][start_idx:end_idx]
            neg_te = self.neg_rank[tag][start_idx:end_idx]
            mask_pos_te = self.mask_rank[tag][start_idx:end_idx]

            if model_type=="rvae":
                yield x, pos, neg, mask, pos_te, neg_te, mask_pos_te
            else:
                yield x, pos, neg, mask, pos_te, neg_te, mask_pos_te, raw_idxs

    def get_items_counts_by_cat(self, tag):
        """
        Return the number of positive items in test set
        :return: #less, #middle, #most
        """

        assert (tag in ('validation', 'test'))

        if self._count_positive[tag] is None:
            self._count_positive[tag] = [0, 0, 0]

            for pos in self.pos_rank[tag]:
                for item in pos:
                    if self.item_popularity[item] <= self.thresholds[0]:
                        self._count_positive[tag][0] += 1
                    elif self.thresholds[0] < self.item_popularity[item] <= self.thresholds[1]:
                        self._count_positive[tag][1] += 1
                    else:
                        self._count_positive[tag][2] += 1

        return self._count_positive[tag]

    def get_users_counts_by_cat(self, tag):
        """
        Return the number of user in popularity category in test set
        :return: #less, #middle, #most
        """

        assert (tag in ('validation', 'test'))

        if self._count_positive_user[tag] is None:
            self._count_positive_user[tag] = [0, 0, 0]

            for pos in self.pos_rank[tag]:
                match = [False, False, False]
                for item in pos:
                    if self.item_popularity[item] <= self.thresholds[0]:
                        match[0] = True
                    elif self.thresholds[0] < self.item_popularity[item] <= self.thresholds[1]:
                        match[1] = True
                    else:
                        match[2] = True

                    if match[0] and match[1] and match[2]:
                        break

                for i, v in enumerate(match):
                    if v:
                        self._count_positive_user[tag][i] += 1

        return self._count_positive_user[tag]


class CachedDataLoader(DataLoader):
    """
    Version of data loader that use a sqlite cache to store data
    """
    def __init__(self, file_tr, seed, decreasing_factor, model_type, pos_neg_ratio=4,
                 negatives_in_test=100, alpha=None, gamma=None, clean_cache=False):
        """

        :param clean_cache: if True rebuild cache from scratch
        """
        self.use_popularity = model_type in (model_types.LOW, model_types.MED, model_types.HIGH,
                                                  model_types.OVERSAMPLING)

        cache_file = f'{file_tr}_{1 if self.use_popularity else 0}.db'

        if clean_cache and os.path.exists(cache_file):
            os.remove(cache_file)

        init_db = not os.path.exists(cache_file)

        self.max_width = 0  # unused
        self.pos_neg_ratio = pos_neg_ratio
        self.negatives_in_test = negatives_in_test
        self.model_type = model_type
        self.size = {}

        self._db = sqlite3.connect(cache_file)

        if init_db:
            dataset = load_dataset(file_tr)
            self._init_cache(dataset, decreasing_factor, alpha, gamma)
        else:
            cur = self._db.cursor()

            self._init_size_struct(cur)

            cur.execute('SELECT data FROM config WHERE id = 1')
            self.n_users, self.item_popularity, self.thresholds, self.item_popularity_dict, self.n_users_dict = json.loads(cur.fetchone()[0])
            cur.close()

            self.item_popularity = np.array(self.item_popularity)
            self._init_item_struct(decreasing_factor, alpha, gamma)

        if model_type == model_types.REWEIGHTING:
            def _creating_csr_mask(x, input_shape):
                rows = []
                cols = []
                vals = []

                if self.alpha is not None:
                    def inverse_sigmoid_weight(item_pop, _alpha=0.01, _beta=0.002, _gamma=100):
                        return (_gamma+1)*((_gamma * (1 + np.exp(_alpha * (item_pop - _beta - 1))) ** -1 + 1) / (
                                _gamma * (1 + np.exp(-_alpha * _beta)) ** -1 + 1))

                    w_i = [inverse_sigmoid_weight(elem, _alpha=self.alpha, _beta=self.beta, _gamma=self.gamma)
                           for elem in self.absolute_item_popularity]
                else:
                    w_i = [1/elem for elem in self.absolute_item_popularity]

                for i, elem in enumerate(x):
                    for j in range(len(elem)):
                        rows.append(i)
                        cols.append(j)
                        # elem[j] è l'item ID dell'oggetto
                        vals.append(w_i[elem[j]])
                return csr_matrix((vals, (rows, cols)), shape=input_shape, dtype=np.float)
        else:
            def _creating_csr_mask(x, input_shape):
                rows = []
                cols = []
                vals = []
                for i, elem in enumerate(x):
                    for j in range(len(elem)):
                        rows.append(i)
                        cols.append(j)
                        vals.append(1)
                return csr_matrix((vals, (rows, cols)), shape=input_shape, dtype=np.uint8)

        self.creating_csr_mask = _creating_csr_mask

        print('Done.')

    def __del__(self):
        self._db.close()

    def _init_item_struct(self, decreasing_factor, alpha, gamma):
        self.n_items = len(self.item_popularity)
        self.absolute_thresholds = list(map(lambda x: x * self.n_users, self.thresholds))
        self.absolute_item_popularity = np.array(list(map(lambda x: x * self.n_users, self.item_popularity)))

        # IMPROVEMENT
        self.low_pop = len([i for i in self.item_popularity if i <= self.thresholds[0]])
        self.med_pop = len([i for i in self.item_popularity if self.thresholds[0] < i <= self.thresholds[1]])
        self.high_pop = len([i for i in self.item_popularity if self.thresholds[1] < i])

        # compute the Beta parameter according to the item popularity
        if self.model_type == model_types.REWEIGHTING:
            # Beta is defined as the average popularity of the medium-popular class of items
            self.beta = self.absolute_item_popularity[(self.absolute_item_popularity >= self.absolute_thresholds[0]) &
                                                      (self.absolute_item_popularity < self.absolute_thresholds[
                                                          1])].mean()
            assert self.beta > 0, self.absolute_thresholds
            self.gamma = gamma
            self.alpha = alpha

        self.sorted_item_popularity = sorted(self.item_popularity)
        limit = 1
        self.max_popularity = self.sorted_item_popularity[-limit]
        self.min_popularity = self.sorted_item_popularity[0]

        self.decreasing_factor = decreasing_factor
        n = self.pos_neg_ratio
        self.frequencies_dict = {}
        self.freq_decimal_part_dict = {}
        self.counter_for_decimal_part_dict = {}
        self.slots_available_for_decimal_part_dict = {}
        for split_type in ["training", "validation", "test"]:
            frequencies = []
            freq_decimal_part = []
            item_popularity = self.item_popularity_dict[split_type]
            max_popularity = max(item_popularity)
            nusers = self.n_users_dict[split_type]
            for idx in range(len(item_popularity)):
                f_i = item_popularity[idx]
                if f_i > 0:
                    n_i_decimal, n_int = modf(n * (max_popularity / (self.decreasing_factor * f_i)))
                    frequencies.append(int(n_int))
                    freq_decimal_part.append(n_i_decimal)
                else:
                    frequencies.append(0)
                    freq_decimal_part.append(0)
            counter_for_decimal_part = [int(int(item_popularity[idx] * nusers) * freq_decimal_part[idx]) for idx in
                                        range(len(item_popularity))]  # K_i
            slots_available_for_decimal_part = [int(f_i * nusers) for f_i in item_popularity]  # K

            self.frequencies_dict[split_type] = frequencies
            self.freq_decimal_part_dict[split_type] = freq_decimal_part
            self.counter_for_decimal_part_dict[split_type] = counter_for_decimal_part
            self.slots_available_for_decimal_part_dict[split_type] = slots_available_for_decimal_part

        self.item_visibility_dict = {split_type: [0] * len(self.item_popularity_dict[split_type])
                                     for split_type in ["training", "validation", "test"]}


    def _create_vector(self, item_data, shape, dtype=np.int32):
        """
        Return a csr matrix 1xD where
        :param item_data: is a list of integer, the id of the items
        :param shape: number of the columns D
        :param dtype:
        """
        cols = list(range(len(item_data)))
        rows = [0] * len(item_data)
        vals = item_data
        items_np = csr_matrix((vals, (rows, cols)), shape=(1, shape), dtype=dtype)

        return items_np

    def _create_vector_one_hot(self, item_data, shape=None, dtype=np.int8):
        """
        Return a csr matrix 1xD where
        :param item_data: is a list of integer, the id of the items
        :param shape: number of the columns D
        :param dtype:
        """
        if shape is None:
            shape = self.n_items
        vals = [1] * len(item_data)
        rows = [0] * len(item_data)
        cols = item_data
        items_np = csr_matrix((vals, (rows, cols)), shape=(1, shape), dtype=dtype)

        return items_np

    def _load_data(self, dataset, cur):
        batch = []

        training_data = dataset['training_data']
        validation_data = dataset['validation_data']
        test_data = dataset['test_data']

        print('LEN TEST:', len(test_data.keys()))

        for i, user_id in enumerate(training_data):
            if i % 1000 == 0:
                print("{}/{}".format(i, len(training_data)))

            pos, neg = training_data[user_id]
            assert len(neg) >= 100, f'fail train neg for user {user_id}'

            items_np = self._create_vector_one_hot(pos)
            pos, neg = self._generate_pairs(pos, tag="training")
            pos = self._create_vector(pos, len(pos))
            neg = self._create_vector(neg, len(neg))

            batch.append((i+1,
                          pickle.dumps(items_np),
                          pickle.dumps(pos),
                          pickle.dumps(neg)
                          ))

            if (i + 1) % 1000 == 0 or i + 1 == len(training_data):
                cur.executemany('INSERT INTO training VALUES(?,?,?,?)', batch)
                self._db.commit()
                batch.clear()

        # VALIDATION
        assert len(batch) == 0
        for i, user_id in enumerate(validation_data):
            if i % 1000 == 0:
                print("{}/{}".format(i, len(validation_data)))
            positives_tr, positives_te, negatives_sampled = validation_data[user_id]
            if len(positives_te) == 0:
                continue

            assert len(negatives_sampled) >= 100, f'fail valid neg for user {user_id}'

            all_pos = positives_tr + positives_te
            items_np = self._create_vector_one_hot(all_pos)

            pos, neg = self._generate_pairs(positives_tr, tag="validation")
            pos = self._create_vector(pos, len(pos))
            neg = self._create_vector(neg, len(neg))

            batch.append((i + 1,
                          pickle.dumps(items_np),
                          pickle.dumps(pos),
                          pickle.dumps(neg),
                          pickle.dumps(positives_te),  # simple list
                          pickle.dumps(negatives_sampled)  # simple list
                          ))

            if (i + 1) % 1000 == 0 or i + 1 == len(validation_data):
                cur.executemany('INSERT INTO validation VALUES(?,?,?,?,?,?)', batch)
                self._db.commit()
                batch.clear()

        # TEST
        assert len(batch) == 0
        for i, user_id in enumerate(test_data):
            if i % 1000 == 0:
                print("{}/{}".format(i, len(test_data)))
            positives_tr, positives_te, negatives_sampled = test_data[user_id]
            if len(positives_te) == 0:
                continue

            assert len(negatives_sampled) >= 100, f'fail test neg for user {user_id}'

            all_pos = positives_tr + positives_te
            items_np = self._create_vector_one_hot(all_pos)

            pos, neg = self._generate_pairs(positives_tr, tag="test")
            pos = self._create_vector(pos, len(pos))
            neg = self._create_vector(neg, len(neg))

            batch.append((i + 1,
                          pickle.dumps(items_np),
                          pickle.dumps(pos),
                          pickle.dumps(neg),
                          pickle.dumps(positives_te),  # simple list
                          pickle.dumps(negatives_sampled)  # simple list
                          ))

            if (i + 1) % 1000 == 0 or i + 1 == len(validation_data):
                cur.executemany('INSERT INTO testset VALUES(?,?,?,?,?,?)', batch)
                self._db.commit()
                batch.clear()

    def _init_cache(self, dataset, decreasing_factor, alpha, gamma):
        cur = self._db.cursor()

        ### TABLE
        # the id are 1-based
        cur.executescript('''
CREATE TABLE config (
  "id" integer NOT NULL,
  "data" TEXT,
  PRIMARY KEY ("id")
);
CREATE TABLE training (
  "id" integer NOT NULL,
  "data_x" blob,
  "data_pos" blob,
  "data_neg" blob,
  PRIMARY KEY ("id")
);
CREATE TABLE validation (
  "id" integer NOT NULL,
  "data_x" blob,
  "data_pos" blob,
  "data_neg" blob,
  "data_pos_te" blob,
  "data_neg_te" blob,
  PRIMARY KEY ("id")
);
CREATE TABLE testset (
  "id" integer NOT NULL,
  "data_x" blob,
  "data_pos" blob,
  "data_neg" blob,
  "data_pos_te" blob,
  "data_neg_te" blob,
  PRIMARY KEY ("id")
);
        ''')

        self._db.commit()

        ### DATA
        n_users = dataset['users']
        item_popularity = dataset['popularity']
        thresholds = dataset['thresholds']
        item_popularity_dict = dataset['popularity_dict']
        n_users_dict = {"training": len(dataset["training_data"]), "validation": len(dataset["validation_data"]),
                             "test": len(dataset["test_data"])}

        self.item_popularity_dict = item_popularity_dict
        self.n_users_dict = n_users_dict

        self.n_items = len(item_popularity)

        self.n_users, self.item_popularity, self.thresholds = n_users, item_popularity, thresholds

        cur.execute('INSERT INTO config VALUES (?, ?)',
                    (1, json.dumps((n_users, item_popularity, thresholds, item_popularity_dict, n_users_dict))))
        self._db.commit()

        self._init_item_struct(decreasing_factor, alpha, gamma)

        # pre process data
        self._load_data(dataset, cur)
        self._init_size_struct(cur)

        # CLOSE
        cur.close()

    def _generate_mask_te(self, pos_rank):
        """
        Generate a 1-based mask with shape len(pos_rank) x n_items

        :return: numpy array
        """
        mask_rank = np.zeros((len(pos_rank), self.n_items), dtype=np.int8)

        for i, pos in enumerate(pos_rank):
            mask_rank[i, pos] = 1

        return mask_rank

    def iter(self, batch_size=256, tag='train'):
        """
        Iter on data

        :param batch_size: size of the batch
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        assert (tag in ('train', 'validation', 'test'))

        tablename = 'validation'
        if tag == 'train':
            tablename = 'training'
        elif tag == 'test':
            tablename = 'testset'

        cur = self._db.cursor()
        n_users = cur.execute(f'SELECT COUNT(*) AS cnt FROM {tablename}').fetchone()[0]

        idxlist = np.arange(1, n_users + 1)
        np.random.shuffle(idxlist)
        N = idxlist.shape[0]
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            raw_idxs = idxlist[start_idx:end_idx]

            x = []
            pos = []
            neg = []
            mask = []

            max_column = 0
            id_params = ",".join([str(i) for i in raw_idxs])
            for row in cur.execute(f'SELECT * FROM {tablename} WHERE id IN ({id_params})'):
                x.append(pickle.loads(row[1]))
                pos.append(pickle.loads(row[2]))
                neg.append(pickle.loads(row[3]))
                mask.append(pos[-1].data)

                max_column = max(max_column, pos[-1].shape[1])

            for m1, m2 in zip(pos, neg):
                m1.resize(1, max_column)
                m2.resize(1, max_column)

            assert len(x) > 0
            x = vstack(x).A
            mask = self.creating_csr_mask(mask, (x.shape[0], max_column)).A
            pos = vstack(pos).A
            neg = vstack(neg).A
            yield x, pos, neg, mask

        cur.close()

    def iter_test(self, batch_size=256, tag='test'):
        """
        Iter on data

        mask_loss

        :param batch_size: size of the batch
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        assert (tag in ('validation', 'test'))

        tablename = 'validation' if tag != 'test' else 'testset'

        cur = self._db.cursor()
        n_users = cur.execute(f'SELECT COUNT(*) AS cnt FROM {tablename}').fetchone()[0]

        idxlist = np.arange(1, n_users + 1)
        N = idxlist.shape[0]

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            raw_idxs = idxlist[start_idx:end_idx]

            x = []
            pos = []
            neg = []
            mask = []
            pos_te = []
            neg_te = []

            max_column = 0
            id_params = ",".join([str(i) for i in raw_idxs])
            for row in cur.execute(f'SELECT * FROM {tablename} WHERE id IN ({id_params})'):
                x.append(pickle.loads(row[1]))
                pos.append(pickle.loads(row[2]))
                neg.append(pickle.loads(row[3]))
                mask.append(pos[-1].data)
                pos_te.append(pickle.loads(row[4]))
                neg_te.append(pickle.loads(row[5]))

                max_column = max(max_column, pos[-1].shape[1])

            for m1, m2 in zip(pos, neg):
                m1.resize(1, max_column)
                m2.resize(1, max_column)

            assert len(x) > 0
            x = vstack(x).A
            pos = vstack(pos).A
            neg = vstack(neg).A
            mask = self.creating_csr_mask(mask, (x.shape[0], max_column)).A
            # pos_te is list
            # neg_te is list
            mask_te = self._generate_mask_te(pos_te)

            yield x, pos, neg, mask, pos_te, neg_te, mask_te

        cur.close()

    def _init_size_struct(self, cur):
        for name, tablename in (('train', 'training'), ('validation', 'validation'), ('test', 'testset')):
            self.size[name] = cur.execute(f'SELECT COUNT(*) AS cnt FROM {tablename}').fetchone()[0]


class EnsembleDataLoader(DataLoader):
    def __init__(self, data_dir, p_dims, seed, decreasing_factor, model_type=model_types.BASELINE,
                 pos_neg_ratio=4, negatives_in_test=100, alpha=None,
                 gamma=None, device='cpu'):
        super().__init__(os.path.join(data_dir, 'data_rvae'), seed, decreasing_factor, model_type,
                       pos_neg_ratio, negatives_in_test, alpha, gamma)

        # loading models
        print('Loading ensemble models...')
        first_model, second_model = model_types.BASELINE, model_types.LOW
        baseline_dir = os.path.join(data_dir, first_model)
        popularity_dir = os.path.join(data_dir, second_model)

        baseline_file_model = os.path.join(baseline_dir, 'best_model.pth')
        popularity_file_model = os.path.join(popularity_dir, 'best_model.pth')

        p_dims.append(self.n_items)
        self.baseline_model = MultiVAE(p_dims)
        self.baseline_model.load_state_dict(torch.load(baseline_file_model, map_location=device))
        print(f"Loaded {first_model} model")
        self.popularity_model = MultiVAE(p_dims)
        self.popularity_model.load_state_dict(torch.load(popularity_file_model, map_location=device))
        print(f"Loaded {second_model} model")
        self.baseline_model.to(device)
        self.popularity_model.to(device)
        self.baseline_model.eval()
        self.popularity_model.eval()
        print('ensemble models loaded!')

    def iter_ensemble(self, batch_size=256, tag='train', device="cpu"):
        for batch_idx, (x, pos, neg, mask) in enumerate(self.iter(batch_size=batch_size, tag=tag)):
            x_tensor = naive_sparse2tensor(x).to(device)

            y_a, _, _ = self.baseline_model(x_tensor, True)
            y_b, _, _ = self.popularity_model(x_tensor, True)

            yield x, pos, neg, mask, y_a, y_b

    def iter_test_ensemble(self, batch_size=256, tag='test', device="cpu"):
        for batch_idx, (x, pos, neg, mask, pos_te, neg_te, mask_pos_te) in enumerate(
                self.iter_test(batch_size=batch_size, tag=tag)):
            x_tensor = naive_sparse2tensor(x).to(device)
            mask_te_tensor = naive_sparse2tensor(mask_pos_te).to(device)

            x_input = x_tensor * (1 - mask_te_tensor)

            y_a, _, _ = self.baseline_model(x_input, True)
            y_b, _, _ = self.popularity_model(x_input, True)

            yield x, pos, neg, mask, pos_te, neg_te, mask_pos_te, y_a, y_b


class EnsembleDataLoaderOld:
    def __init__(self, data_dir, p_dims, seed, decreasing_factor, pos_neg_ratio=4, negatives_in_test=100,
                  chunk_size=1000, device="cpu"):

        dataset_file = os.path.join(data_dir, 'data_rvae')
        dataset = load_dataset(dataset_file)

        self.n_users = dataset['users']
        self.item_popularity = dataset['popularity_dict']["test"]
        self.thresholds = dataset['thresholds']
        self.pos_neg_ratio = pos_neg_ratio
        self.negatives_in_test = negatives_in_test
        self.chunk_size = chunk_size

        # IMPROVEMENT
        self.low_pop = len([i for i in self.item_popularity if i <= self.thresholds[0]])
        self.med_pop = len([i for i in self.item_popularity if self.thresholds[0] < i <= self.thresholds[1]])
        self.high_pop = len([i for i in self.item_popularity if self.thresholds[1] < i])

        # limit = self.high_pop
        limit = 1

        self.n_items = len(self.item_popularity)
        self.use_popularity = True
        self.sorted_item_popularity = sorted(self.item_popularity)
        self.max_popularity = self.sorted_item_popularity[-limit]
        self.min_popularity = self.sorted_item_popularity[0]

        self.decreasing_factor = decreasing_factor
        n = self.pos_neg_ratio
        self.frequencies = [n * ceil(self.max_popularity / (self.decreasing_factor * f_i)) for f_i in self.item_popularity]

        self.max_width = -1


        # loading models
        print('Loading ensemble models...')
        first_model, second_model = model_types.BASELINE, model_types.LOW
        baseline_dir = os.path.join(data_dir, first_model)
        popularity_dir = os.path.join(data_dir, second_model)

        baseline_file_model = os.path.join(baseline_dir, 'best_model.pth')
        popularity_file_model = os.path.join(popularity_dir, 'best_model.pth')

        p_dims.append(self.n_items)
        self.baseline_model = MultiVAE(p_dims)
        self.baseline_model.load_state_dict(torch.load(baseline_file_model, map_location=device))
        print(f"Loaded {first_model} model")
        self.popularity_model = MultiVAE(p_dims)
        self.popularity_model.load_state_dict(torch.load(popularity_file_model, map_location=device))
        print(f"Loaded {second_model} model")
        self.baseline_model.to(device)
        self.popularity_model.to(device)
        self.baseline_model.eval()
        self.popularity_model.eval()
        print('ensemble models loaded!')

        print('phase 1: Loading data...')
        self._initialize()
        # checking if data have already been computed
        preprocessed_data_dir = os.path.join(data_dir, "preprocessed_data", "low", f"decreasing_factor_{decreasing_factor}",
                                             str(seed))

        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)

        def _dir_does_not_contain_files(input_path):
            return len(os.listdir(input_path)) == 0

        if _dir_does_not_contain_files(preprocessed_data_dir):
            print("Generating pre-processed data from scratch")
            self._load_data(dataset, preprocessed_data_dir)
        else:
            print("Loading pre-processed data from disk")
            tags = ["train", "validation", "test"]
            for tag in tags:
                with open(os.path.join(preprocessed_data_dir, f'pos_{tag}.pkl'), 'rb') as f:
                    self.pos[tag] = pickle.load(f)
                with open(os.path.join(preprocessed_data_dir, f'neg_{tag}.pkl'), 'rb') as f:
                    self.neg[tag] = pickle.load(f)
                self.data[tag] = np.load(os.path.join(preprocessed_data_dir, f"data_{tag}.npy"))
                self.size[tag] = self.data[tag].shape[0]
                if tag in tags[1:]:  # all except train
                    with open(os.path.join(preprocessed_data_dir, f'pos_rank_{tag}.pkl'), 'rb') as f:
                        self.pos_rank[tag] = pickle.load(f)
                    with open(os.path.join(preprocessed_data_dir, f'neg_rank_{tag}.pkl'), 'rb') as f:
                        self.neg_rank[tag] = pickle.load(f)

            with open(os.path.join(preprocessed_data_dir, f'max_width.pkl'), 'rb') as f:
                self.max_width = pickle.load(f)

        print("phase 2: converting the pos/neg list of lists to a sparse matrix for future indexing")

        def _converting_to_csr_matrix(x, input_shape, desc):
            rows = []
            cols = []
            vals = []
            for i, elem in tqdm(enumerate(x), desc=desc):  # users loop
                for j in range(len(elem)):  # pos/neg loop
                    rows.append(i)
                    cols.append(j)
                    vals.append(elem[j])
            return csr_matrix((vals, (rows, cols)), shape=input_shape, dtype=np.int32)

        def _creating_csr_mask(x, input_shape, desc):
            rows = []
            cols = []
            vals = []
            for i, elem in tqdm(enumerate(x), desc=desc):
                for j in range(len(elem)):
                    rows.append(i)
                    cols.append(j)
                    vals.append(1)
            return csr_matrix((vals, (rows, cols)), shape=input_shape, dtype=np.uint8)

        # check if sparse matrices have already been computed
        dir_for_sparse_matrices = os.path.join(data_dir, "sparse_matrices", "low", f"decreasing_factor_{decreasing_factor}"
                                               , str(seed))
        if not os.path.exists(dir_for_sparse_matrices):
            os.makedirs(dir_for_sparse_matrices)

        self.pos_sparse = dict()
        self.neg_sparse = dict()
        self.mask_sparse = dict()

        def _dir_does_not_contain_files(input_path):
            return len(os.listdir(input_path)) == 0

        if _dir_does_not_contain_files(dir_for_sparse_matrices):
            print("Generating pos/neg/masks from scratch")
            for tag in self.pos:
                shape = [self.size[tag], self.max_width]
                self.pos_sparse[tag] = _converting_to_csr_matrix(self.pos[tag], input_shape=shape, desc="Positive Items")
                self.neg_sparse[tag] = _converting_to_csr_matrix(self.neg[tag], input_shape=shape, desc="Positive Items")
                self.mask_sparse[tag] = _creating_csr_mask(self.pos[tag], input_shape=shape, desc="Mask Items")
            # save matrices
            for tag in self.pos:
                save_npz(os.path.join(dir_for_sparse_matrices, f"pos_{tag}.npz"), self.pos_sparse[tag])
                save_npz(os.path.join(dir_for_sparse_matrices, f"neg_{tag}.npz"), self.neg_sparse[tag])
                save_npz(os.path.join(dir_for_sparse_matrices, f"mask_{tag}.npz"), self.mask_sparse[tag])
        else:
            # load matrices
            print("Loading stored pos/neg/mask matrices")
            for tag in self.pos:
                self.pos_sparse[tag] = load_npz(
                    os.path.join(dir_for_sparse_matrices, f"pos_{tag}.npz"))
                self.neg_sparse[tag] = load_npz(
                    os.path.join(dir_for_sparse_matrices, f"neg_{tag}.npz"))
                self.mask_sparse[tag] = load_npz(
                    os.path.join(dir_for_sparse_matrices, f"mask_{tag}.npz"))

        print('phase 3: generating test masks...')

        for tag in ('validation', 'test'):
            self._generate_mask_te(tag)

        print('Done.')

    def _generate_mask_tr(self, tag):

        # IMPROVED VERSION
        # TODO replace self.max_width with self.chunk_size
        print('self.max_width:', self.max_width)
        print('self.chunk_size:', self.chunk_size)

        # self.mask[tag] = lil_matrix((self.size[tag], self.chunk_size))
        # pos_temp = lil_matrix((self.size[tag], self.chunk_size))
        # neg_temp = lil_matrix((self.size[tag], self.chunk_size))

        self.mask[tag] = np.zeros((self.size[tag], self.max_width))
        pos_temp = np.zeros((self.size[tag], self.max_width))
        neg_temp = np.zeros((self.size[tag], self.max_width))

        for row in range(self.size[tag]):

            if row % 1000 == 0:
                print('generate_mask_tr: user {}/{}'.format(row, self.size[tag]))

            self.mask[tag][row, :len(self.pos[tag][row])] = [1] * len(self.pos[tag][row])
            if row % 1000 == 0:
                # print('Updating pos...')
                pass
            pos_temp[row, :len(self.pos[tag][row])] = self.pos[tag][row]
            if row % 1000 == 0:
                # print('Updating neg...')
                pass
            neg_temp[row, :len(self.neg[tag][row])] = self.neg[tag][row]

        print('generate_mask_tr almost completed...', tag)
        self.pos[tag] = pos_temp
        self.neg[tag] = neg_temp
        print('generate_mask_tr completed!', tag)

    def _generate_mask_te(self, tag):
        self.mask_rank[tag] = np.zeros((self.size[tag], self.n_items), dtype=np.int8)

        for row in np.arange(self.size[tag]):
            if row % 1000 == 0:
                print('generate_mask_te: user {}/{}'.format(row, self.size[tag]))
            pos = self.pos_rank[tag][row]
            self.mask_rank[tag][row, pos] = 1
        print('generate_mask_te completed!', tag)

    def _initialize(self):
        self._count_positive = {'train': None, 'validation': None, 'test': None}
        self._count_positive_user = {'train': None, 'validation': None, 'test': None}

        self.data = dict()
        self.pos = dict()  # history of the user
        self.neg = dict()
        self.mask = dict()
        self.pos_rank = dict()  # items to predict
        self.neg_rank = dict()
        self.mask_rank = dict()
        self.size = dict()

        for tag in ('train', 'validation', 'test'):
            self.pos[tag] = []
            self.neg[tag] = []

        for tag in ('validation', 'test'):
            self.pos_rank[tag] = []
            self.neg_rank[tag] = []

    def _load_data(self, dataset, preprocessed_data_dir):
        train = []
        validation = []
        test = []

        training_data = dataset['training_data']
        validation_data = dataset['validation_data']
        test_data = dataset['test_data']

        for user_id in training_data:
            if user_id % 1000 == 0:
                print("{}/{}".format(user_id, len(training_data)))

            pos, neg = training_data[user_id]
            assert len(neg) >= 100, f'fail train neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[pos] = 1

            pos, neg = self._generate_pairs(pos)

            train.append(items_np)
            self.pos['train'].append(pos)
            self.neg['train'].append(neg)

        print('self.max_width:', self.max_width)

        self.data['train'] = np.array(train, dtype=np.int8)
        self.data['train'][:] = train[:]
        self.size['train'] = len(train)

        gc.collect()

        for user_id in validation_data:
            if user_id % 1000 == 0:
                print("{}/{}".format(user_id, len(validation_data)))
            positives_tr, positives_te, negatives_sampled = validation_data[user_id]
            if len(positives_te) == 0:
                continue

            assert len(negatives_sampled) >= 100, f'fail valid neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[positives_tr] = 1
            items_np[positives_te] = 1

            pos, neg = self._generate_pairs(positives_tr)
            validation.append(items_np)

            self.pos['validation'].append(pos)
            self.neg['validation'].append(neg)

            self.pos_rank['validation'].append(positives_te)
            self.neg_rank['validation'].append(negatives_sampled)

        if len(validation) > 0:
            self.data['validation'] = np.array(validation, dtype=np.int8)
            self.data['validation'][:] = validation[:]
            self.size['validation'] = len(validation)
            gc.collect()

        for user_id in test_data:
            if user_id % 1000 == 0:
                print("{}/{}".format(user_id, len(test_data)))
            positives_tr, positives_te, negatives_sampled = test_data[user_id]
            if len(positives_te) == 0:
                continue

            assert len(negatives_sampled) >= 100, f'fail test neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[positives_tr] = 1
            items_np[positives_te] = 1

            pos, neg = self._generate_pairs(positives_tr)
            test.append(items_np)

            self.pos['test'].append(pos)
            self.neg['test'].append(neg)

            self.pos_rank['test'].append(positives_te)
            self.neg_rank['test'].append(negatives_sampled)

        if len(test) > 0:
            self.data['test'] = np.array(test, dtype=np.int8)
            self.data['test'][:] = test[:]
            self.size['test'] = len(test)
            gc.collect()

        # saving artifacts on disk
        tags = ["train", "validation", "test"]
        for tag in tags:
            with open(os.path.join(preprocessed_data_dir, f'pos_{tag}.pkl'), 'wb') as f:
                pickle.dump(self.pos[tag], f)
            with open(os.path.join(preprocessed_data_dir, f'neg_{tag}.pkl'), 'wb') as f:
                pickle.dump(self.neg[tag], f)
            np.save(os.path.join(preprocessed_data_dir, f"data_{tag}.npy"), self.data[tag])
            if tag in tags[1:]:  # all except train
                with open(os.path.join(preprocessed_data_dir, f'pos_rank_{tag}.pkl'), 'wb') as f:
                    pickle.dump(self.pos_rank[tag], f)
                with open(os.path.join(preprocessed_data_dir, f'neg_rank_{tag}.pkl'), 'wb') as f:
                    pickle.dump(self.neg_rank[tag], f)

        with open(os.path.join(preprocessed_data_dir, f'max_width.pkl'), 'wb') as f:
            pickle.dump(self.max_width, f)

    def _sample_negatives(self, pos, size):
        all_items = set(range(len(self.item_popularity)))
        all_negatives = list(all_items - set(pos))

        return random.choices(all_negatives, k=size)

    def _generate_pairs(self, pos):
        # IMPROVEMENT
        positives = []
        for item in pos:
            if self.use_popularity:
                frequency = self.frequencies[item]
            else:
                frequency = self.pos_neg_ratio
            positives[0:0] = [item] * frequency

        negatives = self._sample_negatives(pos, len(positives))
        self.max_width = max(self.max_width, len(negatives))

        return positives, negatives

    def iter_ensemble(self, batch_size=256, tag='train', device="cpu"):
        for batch_idx, (x, pos, neg, mask) in enumerate(self.iter(batch_size=batch_size, tag=tag)):
            x_tensor = naive_sparse2tensor(x).to(device)

            y_a, _, _ = self.baseline_model(x_tensor, True)
            y_b, _, _ = self.popularity_model(x_tensor, True)

            yield x, pos, neg, mask, y_a, y_b

    def iter(self, batch_size=256, tag='train'):
        """
        Iter on data

        :param batch_size: size of the batch
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        assert (tag in ('train', 'validation', 'test'))

        idxlist = np.arange(self.data[tag].shape[0])
        np.random.shuffle(idxlist)

        N = idxlist.shape[0]

        idx = np.argsort(self.frequencies)

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)

            x = self.data[tag][idxlist[start_idx:end_idx]]

            pos = self.pos[tag][idxlist[start_idx:end_idx]]
            neg = self.neg[tag][idxlist[start_idx:end_idx]]
            mask = self.mask[tag][idxlist[start_idx:end_idx]]

            yield x, pos, neg, mask

    def iter_test_ensemble(self, batch_size=256, tag='train', device="cpu"):
        if tag == 'train':
            for batch_idx, (x, pos, neg, mask) in enumerate(
                    self.iter(batch_size=batch_size, tag=tag)):

                y_a, _, _ = self.baseline_model(x, True)
                y_b, _, _ = self.popularity_model(x, True)

                yield x, pos, neg, mask, y_a, y_b
        else:
            for batch_idx, (x, pos, neg, mask, pos_te, neg_te, mask_te) in enumerate(
                    self.iter_test(batch_size=batch_size, tag=tag)):
                x_tensor = naive_sparse2tensor(x).to(device)
                mask_te_tensor = naive_sparse2tensor(mask_te).to(device)

                x_input = x_tensor * (1 - mask_te_tensor)

                y_a, _, _ = self.baseline_model(x_input, True)
                y_b, _, _ = self.popularity_model(x_input, True)

                yield x, pos, neg, mask, pos_te, neg_te, mask_te, y_a, y_b

    def iter_test(self, batch_size=256, tag='test'):
        """
        Iter on data

        mask_loss

        :param batch_size: size of the batch
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        assert (tag in ('validation', 'test'))

        N = self.size[tag]
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)

            x = self.data[tag][start_idx:end_idx]

            pos = self.pos_sparse[tag][start_idx:end_idx].A
            neg = self.neg_sparse[tag][start_idx:end_idx].A
            mask = self.mask_sparse[tag][start_idx:end_idx].A

            pos_te = self.pos_rank[tag][start_idx:end_idx]
            neg_te = self.neg_rank[tag][start_idx:end_idx]

            mask_pos_te = self.mask_rank[tag][start_idx:end_idx]

            # print('>>> pos_te:',pos_te)
            # print('>>> neg_te:', neg_te)
            # print('>>> mask_pos_te:', mask_pos_te)

            yield x, pos, neg, mask, pos_te, neg_te, mask_pos_te

    def get_items_counts_by_cat(self, tag):
        """
        Return the number of positive items in test set
        :return: #less, #middle, #most
        """

        assert (tag in ('validation', 'test'))

        if self._count_positive[tag] is None:
            self._count_positive[tag] = [0, 0, 0]

            for pos in self.pos_rank[tag]:
                for item in pos:
                    if self.item_popularity[item] <= self.thresholds[0]:
                        self._count_positive[tag][0] += 1
                    elif self.thresholds[0] < self.item_popularity[item] <= self.thresholds[1]:
                        self._count_positive[tag][1] += 1
                    else:
                        self._count_positive[tag][2] += 1

        return self._count_positive[tag]

    def get_users_counts_by_cat(self, tag):
        """
        Return the number of user in popularity category in test set
        :return: #less, #middle, #most
        """

        assert (tag in ('validation', 'test'))

        if self._count_positive_user[tag] is None:
            self._count_positive_user[tag] = [0, 0, 0]

            for pos in self.pos_rank[tag]:
                match = [False, False, False]
                for item in pos:
                    if self.item_popularity[item] <= self.thresholds[0]:
                        match[0] = True
                    elif self.thresholds[0] < self.item_popularity[item] <= self.thresholds[1]:
                        match[1] = True
                    else:
                        match[2] = True

                    if match[0] and match[1] and match[2]:
                        break

                for i, v in enumerate(match):
                    if v:
                        self._count_positive_user[tag][i] += 1

        return self._count_positive_user[tag]
