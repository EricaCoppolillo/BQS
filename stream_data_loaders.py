import torch
import numpy as np

from tqdm import tqdm
from math import ceil, modf
from scipy.sparse import csr_matrix
from scipy.special import softmax

from util import *
from models import *
from config import Config

model_types = Config("./model_type_info.json")


def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__, reverse=True)


def rankdata(a):
    n = len(a)
    ivec = rank_simple(a)
    svec = [a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0] * n
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i == n - 1 or svec[i] != svec[i + 1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i - dupcount + 1, i + 1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray


class StreamDataLoader:
    def __init__(self, file_tr, seed, decreasing_factor, model_type, pos_neg_ratio=4, negatives_in_test=100, alpha=None,
                 gamma=None):

        self.dataset = load_dataset(file_tr)

        if "contamination" in self.dataset:
            self.contamination = self.dataset["contamination"]

        self.file_tr = file_tr
        if "bpr" in file_tr:
            self.discount_idxs = {"validation": self.dataset["original_training_size"],
                                  "test": self.dataset["original_training_size"] + self.dataset["original_val_size"]}

        self.n_users = self.dataset["users"]
        self.item_popularity = self.dataset['popularity']
        self.item_popularity_dict = self.dataset['popularity_dict']

        self.n_users_dict = {"training": len(self.dataset["training_data"]),
                             "validation": len(self.dataset["validation_data"]),
                             "test": len(self.dataset["test_data"])}
        self.absolute_item_popularity_dict = {split_type: [elem * self.n_users_dict[split_type]
                                                           for elem in self.dataset['popularity_dict'][split_type]]
                                              for split_type in self.dataset['popularity_dict']
                                              }
        self.thresholds = self.dataset['thresholds']
        self.absolute_thresholds = list(map(lambda x: x * self.n_users_dict["training"], self.thresholds))

        self.pos_neg_ratio = pos_neg_ratio
        self.negatives_in_test = negatives_in_test
        self.model_type = model_type

        # IMPROVEMENT
        self.low_pop = len([i for i in self.item_popularity_dict["training"] if i <= self.thresholds[0]])
        self.med_pop = len(
            [i for i in self.item_popularity_dict["training"] if self.thresholds[0] < i <= self.thresholds[1]])
        self.high_pop = len([i for i in self.item_popularity_dict["training"] if self.thresholds[1] < i])

        # compute the Beta parameter according to the item popularity
        if self.model_type == model_types.REWEIGHTING:
            self.absolute_item_popularity = np.array(self.absolute_item_popularity_dict["training"])
            # Beta is defined as the average popularity of the medium-popular class of items
            self.beta = self.absolute_item_popularity[(self.absolute_item_popularity >= self.absolute_thresholds[0]) &
                                                      (self.absolute_item_popularity < self.absolute_thresholds[
                                                          1])].mean()
            assert self.beta > 0, self.absolute_thresholds
            self.gamma = gamma
            self.alpha = alpha
            assert alpha > 0 and gamma > 0, alpha

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
            nusers = self.n_users_dict[split_type]
            # computing item ranking
            item_ranking = rankdata(item_popularity)
            max_popularity = max(item_popularity)
            h = 1
            if "ml-20m" in self.file_tr:
                h = 0.12
            if "netflix" in self.file_tr:
                h = 0.14
            self.h = h

            for idx in range(len(item_popularity)):
                f_i = item_popularity[idx]

                d_i = ceil((item_ranking[idx] / (self.high_pop * h)) + 1)
                if f_i > 0:
                    if self.model_type == model_types.OVERSAMPLING:
                        n_i_decimal, n_int = modf(n * (max_popularity / (d_i * f_i)))
                    else:
                        d_i = self.decreasing_factor
                        n_i_decimal, n_int = modf(n * (max_popularity / (d_i * f_i)))
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

        self.max_width = -1

    def _init_batches(self):
        print('phase 1: Loading data...')
        self._initialize()

        print("Generating pre-processed data from scratch")
        self._load_data(self.dataset)

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

        if self.model_type == model_types.REWEIGHTING:
            def _creating_csr_mask(x, input_shape, desc):
                print("Currently using masks with reweighting")
                rows = []
                cols = []
                vals = []

                if self.alpha is not None:
                    def inverse_sigmoid_weight(item_pop, _alpha=0.01, _beta=0.002, _gamma=100):
                        return (_gamma + 1) * ((_gamma * (1 + np.exp(_alpha * (item_pop - _beta - 1))) ** -1 + 1) / (
                                _gamma * (1 + np.exp(-_alpha * _beta)) ** -1 + 1))

                    w_i = [inverse_sigmoid_weight(elem, _alpha=self.alpha, _beta=self.beta, _gamma=self.gamma)
                           for elem in self.absolute_item_popularity]
                else:
                    w_i = [1 / elem for elem in self.absolute_item_popularity]

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

        self.pos_sparse = dict()
        self.neg_sparse = dict()
        self.mask_sparse = dict()

        print("Generating pos/neg/masks from scratch")
        for tag in self.pos:
            shape = [self.size[tag], self.max_width]
            self.pos_sparse[tag] = _converting_to_csr_matrix(self.pos[tag], input_shape=shape,
                                                             desc="Positive Items")
            self.neg_sparse[tag] = _converting_to_csr_matrix(self.neg[tag], input_shape=shape,
                                                             desc="Negative Items")
            self.mask_sparse[tag] = _creating_csr_mask(self.pos[tag], input_shape=shape, desc="Mask Items")

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

    def _load_data(self, dataset):
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
                condition1 = self.slots_available_for_decimal_part_dict[tag][item] - self.counter_for_decimal_part_dict[tag][item] <= 0
                condition2 = random.random() < self.freq_decimal_part_dict[tag][item]
                if condition1 or condition2:
                    frequency += 1
                    self.counter_for_decimal_part_dict[tag][item] -= 1
                self.slots_available_for_decimal_part_dict[tag][item] -= 1

            self.item_visibility_dict[tag][item] += frequency
            positives[0:0] = [item] * frequency  # append at the beginning (pre-pend)

        negatives = self._sample_negatives(pos, len(positives))
        self.max_width = max(self.max_width, len(positives))

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
            if model_type == "rvae":
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

            if model_type == "rvae":
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


BETA_VOLUMN_CONST = 0.2228655673209082014292
RVAE_EMBEDDING_SIZE = 200


class TwoStagesNegativeSamplingDataLoader(StreamDataLoader):
    def __init__(self, device, model_class, model, beta_sampling, d=RVAE_EMBEDDING_SIZE, num_neg_candidates=300, *args, **kwargs):
        self.init_negative_distr = False
        self.beta_sampling = beta_sampling
        self.model = model
        self.model_class = model_class
        self.device = device
        self.d=d
        self.num_neg_candidates = num_neg_candidates
        super(TwoStagesNegativeSamplingDataLoader, self).__init__(*args, **kwargs)

    def _sample_negatives(self, pos, frequencies):
        size = sum(frequencies)
        if not self.init_negative_distr:
            popularity = self.absolute_item_popularity_dict["training"]
            popularity_total_mass = sum(popularity)
            self.negative_distr = np.array([elem / popularity_total_mass for elem in popularity])
            self.init_negative_distr = True

        # select negatives based on the popularity
        all_items = set(range(len(self.item_popularity)))
        neg = list(all_items - set(pos))
        popularity = np.array(self.absolute_item_popularity_dict["training"])[neg]
        popularity = popularity ** self.beta_sampling

        # 1st stage: select C candidates by popularity
        neg_candidates = np.random.choice(neg, size=min(self.num_neg_candidates, len(neg)), replace=False,
                                          p=softmax(popularity))

        # 2nd stage: select @size negatives from the previous set of candidates
        with torch.no_grad():
            # TODO: to be model-transparent you should pass the embeddings as input
            # in the RVAE model we don't have the concept of item embedding
            # we can arrange it by passing a one-hot vector where the 1 is placed
            # on the item index
            if self.model_class == "rvae":
                pos_input = torch.FloatTensor(len(pos), self.n_items)  # num_pos x num_items
                pos_input.zero_()
                pos_items = torch.LongTensor(pos).view(-1, 1)  # it has to be 2D to use scatter
                pos_input.scatter_(1, pos_items, 1)
                pos_item_embeddings, _ = self.model.encode(pos_input.to(self.device))
                neg_input = torch.FloatTensor(len(neg_candidates), self.n_items)  # num_neg x num_items
                neg_input.zero_()
                neg_items = torch.LongTensor(neg_candidates).view(-1, 1)  # it has to be 2D to use scatter
                neg_input.scatter_(1, neg_items, 1)
                neg_item_embeddings, _ = self.model.encode(neg_input.to(self.device))
                pos_item_embeddings = pos_item_embeddings.cpu().numpy()
                neg_item_embeddings = neg_item_embeddings.cpu().numpy()
            elif self.model_class == "bpr":
                pos_item_embeddings = self.model.embed_item(torch.LongTensor(pos).to(self.device)).cpu().numpy()
                neg_item_embeddings = self.model.embed_item(torch.LongTensor(neg_candidates).to(self.device)).cpu().numpy()
            else:
                raise Exception(f"{self.model_class} model class does not exist")

        item_spreadout_distances = np.linalg.multi_dot([pos_item_embeddings,
                                                        neg_item_embeddings.T])
        negatives = []
        for idx in range(len(pos)):
            weights = self._calculate_weights(item_spreadout_distances[idx])
            weights_sum = np.sum(weights)
            REPLACE_FLAG = frequencies[idx] > len(neg_candidates) or (sum(weights > 0) < frequencies[idx])
            if weights_sum > 0:
                weights = weights / weights_sum
            if weights_sum > 0:
                neg_samples = np.random.choice(neg_candidates,
                                               size=frequencies[idx],
                                               p=weights,
                                               replace=REPLACE_FLAG)
            else:
                neg_samples = np.random.choice(neg_candidates,
                                               size=frequencies[idx],
                                               replace=REPLACE_FLAG)

            negatives[0:0] = neg_samples.tolist()

        return negatives

    def _calculate_weights(self, spreadout_distance):
        mask = spreadout_distance > 0
        log_weights = (1.0 - (float(self.d) - 1) / 2) * np.log(
            1.0 - np.square(spreadout_distance) + 1e-8) + np.log(
            BETA_VOLUMN_CONST)
        weights = np.exp(log_weights)
        weights[np.isnan(weights)] = 0.
        weights[~mask] = 0.
        weights_sum = np.sum(weights)

        if weights_sum > 0:
            weights = weights / weights_sum
        return weights

    def _generate_pairs(self, pos, tag=""):
        # IMPROVEMENT
        positives = []
        frequencies = []
        for item in pos:
            if self.use_popularity:
                frequency = self.frequencies_dict[tag][item]
            else:
                frequency = self.pos_neg_ratio
            if self.counter_for_decimal_part_dict[tag][item] > 0:
                if self.slots_available_for_decimal_part_dict[tag][item] - self.counter_for_decimal_part_dict[tag][
                    item] <= 0 or random.random() < self.freq_decimal_part_dict[tag][item]:
                    frequency += 1
                    self.counter_for_decimal_part_dict[tag][item] -= 1
                self.slots_available_for_decimal_part_dict[tag][item] -= 1

            self.item_visibility_dict[tag][item] += frequency
            positives[0:0] = [item] * frequency  # append at the beginning (pre-pend)
            frequencies[0:0] = [frequency]

        negatives = self._sample_negatives(pos, frequencies)
        self.max_width = max(self.max_width, len(positives))

        return positives, negatives
