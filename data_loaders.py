import os
import gc

from util import *
from models import *


class DataLoader:
    def __init__(self, file_tr, pos_neg_ratio=4, negatives_in_test=100, use_popularity=False):

        dataset = load_dataset(file_tr)

        self.n_users = dataset['users']
        self.item_popularity = dataset['popularity']
        self.thresholds = dataset['thresholds']
        self.pos_neg_ratio = pos_neg_ratio
        self.negatives_in_test = negatives_in_test

        # IMPROVEMENT
        self.low_pop = len([i for i in self.item_popularity if i <= self.thresholds[0]])
        self.med_pop = len([i for i in self.item_popularity if self.thresholds[0] < i <= self.thresholds[1]])
        self.high_pop = len([i for i in self.item_popularity if self.thresholds[1] < i])

        # limit = self.high_pop
        limit = 1

        self.n_items = len(self.item_popularity)
        self.use_popularity = use_popularity
        self.sorted_item_popularity = sorted(self.item_popularity)
        self.max_popularity = self.sorted_item_popularity[-limit]
        self.min_popularity = self.sorted_item_popularity[0]

        gamma = self.pos_neg_ratio
        # gamma = 1
        self.frequencies = [int(round((self.max_popularity * (gamma / min(p, self.max_popularity))))) for p in
                            self.item_popularity]

        self.max_width = -1

        print('DATASET STATS ------------------------------')
        print('users:', self.n_users)
        print('items:', self.n_items)
        print('low_pop:', self.low_pop)
        print('med_pop:', self.med_pop)
        print('high_pop:', self.high_pop)
        print('thresholds:', self.thresholds)
        print('max_popularity:', self.max_popularity)
        print('min_popularity:', self.min_popularity)
        print('max_frequency:', max(self.frequencies))
        print('min_frequency:', min(self.frequencies))
        print('num(max_popularity):', sum(self.item_popularity == self.max_popularity))
        print('num(min_popularity):', sum(self.item_popularity == self.min_popularity))
        print('sorted(self.sorted_item_popularity)[:100]:', sorted(self.sorted_item_popularity[:10]))
        print('sorted(self.sorted_item_popularity)[-100:]:', sorted(self.sorted_item_popularity[-10:]))
        print('sorted(frequencies):', sorted(self.frequencies)[:10])

        print('phase 1: Loading data...')
        self._initialize()
        self._load_data(dataset)

        print('phase 2: Generating training masks...')

        for tag in ('train', 'validation', 'test'):
            self._generate_mask_tr(tag)
            print('type(self.data[tag])', type(self.data[tag]))
            print('SET {}, shape {}, positives {}'.format(tag.upper(), self.data[tag].shape, self.data[tag].sum()))

        print('phase 3: generating test masks...')

        for tag in ('validation', 'test'):
            self._generate_mask_te(tag)

        print('Done.')

    def _generate_mask_tr(self, tag):
        self.mask[tag] = np.zeros((self.size[tag], self.max_width))
        pos_temp = np.zeros((self.size[tag], self.max_width))
        neg_temp = np.zeros((self.size[tag], self.max_width))

        for row in range(self.size[tag]):
            self.mask[tag][row, :len(self.pos[tag][row])] = [1] * len(self.pos[tag][row])
            pos_temp[row, :len(self.pos[tag][row])] = self.pos[tag][row]
            neg_temp[row, :len(self.neg[tag][row])] = self.neg[tag][row]

        self.pos[tag] = pos_temp
        self.neg[tag] = neg_temp

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

        # print('LEN TEST:',test_data)
        '''
        for user_id in training_data:
            pos, neg = training_data[user_id]
            assert len(neg) >= 100, f'fail train neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[pos] = 1
            train.append(items_np)

            pos, neg = self._generate_training_pairs(pos)

            # print('pos:',pos)
            # print('neg:',neg)
            self.pos['train'].append(pos)
            self.neg['train'].append(neg)
        '''

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

    def iter(self, batch_size=256, tag='train'):
        """
        Iter on data

        :param batch_size: size of the batch
        :return: batch_idx, x, pos, neg, mask_pos, mask_neg
        """

        assert (tag in ('train', 'validation', 'test'))

        # idxlist = np.arange(self._N)
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

            pos = self.pos[tag][start_idx:end_idx]
            neg = self.neg[tag][start_idx:end_idx]

            mask = self.mask[tag][start_idx:end_idx]

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


class EnsembleDataLoader:
    def __init__(self, data_dir, p_dims, pos_neg_ratio=4, negatives_in_test=100, use_popularity=False, chunk_size=1000
                 , device="cpu"):
        dataset_file = os.path.join(data_dir, 'data_rvae')
        dataset = load_dataset(dataset_file)

        self.n_users = dataset['users']
        self.item_popularity = dataset['popularity']
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
        self.use_popularity = use_popularity
        self.sorted_item_popularity = sorted(self.item_popularity)
        self.max_popularity = self.sorted_item_popularity[-limit]
        self.min_popularity = self.sorted_item_popularity[0]

        gamma = self.pos_neg_ratio
        # gamma = 1
        self.frequencies = [int(round((self.max_popularity * (gamma / min(p, self.max_popularity))))) for p in
                            self.item_popularity]

        # self.frequencies = [int(round(sqrt(self.max_popularity * (gamma / min(p, self.max_popularity))))) for p in self.item_popularity]
        # self.double_frequencies = [self.max_popularity * (self.pos_neg_ratio / min(p, self.max_popularity)) for p in self.item_popularity]

        self.max_width = -1

        print('DATASET STATS ------------------------------')
        print('users:', self.n_users)
        print('items:', self.n_items)
        print('low_pop:', self.low_pop)
        print('med_pop:', self.med_pop)
        print('high_pop:', self.high_pop)
        print('thresholds:', self.thresholds)
        print('max_popularity:', self.max_popularity)
        print('min_popularity:', self.min_popularity)
        print('max_frequency:', max(self.frequencies))
        print('min_frequency:', min(self.frequencies))
        print('num(max_popularity):', sum(self.item_popularity == self.max_popularity))
        print('num(min_popularity):', sum(self.item_popularity == self.min_popularity))
        print('sorted(self.sorted_item_popularity)[:100]:', sorted(self.sorted_item_popularity[:10]))
        print('sorted(self.sorted_item_popularity)[-100:]:', sorted(self.sorted_item_popularity[-10:]))
        print('sorted(frequencies):', sorted(self.frequencies)[:10])

        # loading models
        print('loading ensemble models...')
        baseline_dir = os.path.join(data_dir, 'baseline')
        popularity_dir = os.path.join(data_dir, 'popularity_low')

        baseline_file_model = os.path.join(baseline_dir, 'best_model.pth')
        popularity_file_model = os.path.join(popularity_dir, 'best_model.pth')

        p_dims.append(self.n_items)
        self.baseline_model = MultiVAE(p_dims)
        self.baseline_model.load_state_dict(torch.load(baseline_file_model, map_location=device))
        self.popularity_model = MultiVAE(p_dims)
        self.popularity_model.load_state_dict(torch.load(popularity_file_model, map_location=device))
        self.baseline_model.to(device)
        self.popularity_model.to(device)
        self.baseline_model.eval()
        self.popularity_model.eval()
        print('ensemble models loaded!')

        print('phase 1: Loading data...')
        self._initialize()
        self._load_data(dataset)

        print('phase 2: Generating training masks...')

        for tag in ('train', 'validation', 'test'):
            # print('LC > tag:', tag)
            # print('LC > self.neg:', self.neg)
            # print('LC > self.neg[tag]:', self.neg[tag])
            # print('LC > self.size[tag]:', self.size[tag])

            self._generate_mask_tr(tag)
            print('type(self.data[tag])', type(self.data[tag]))
            print('SET {}, shape {}, positives {}'.format(tag.upper(), self.data[tag].shape, self.data[tag].sum()))

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

    def _load_data(self, dataset):
        train = []
        validation = []
        test = []

        training_data = dataset['training_data']
        validation_data = dataset['validation_data']
        test_data = dataset['test_data']
        '''
        for user_id in training_data:
            pos, neg = training_data[user_id]
            assert len(neg) >= 100, f'fail train neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[pos] = 1
            train.append(items_np)

            pos, neg = self._generate_training_pairs(pos)

            # print('pos:',pos)
            # print('neg:',neg)
            self.pos['train'].append(pos)
            self.neg['train'].append(neg)
        '''

        for user_id in training_data:
            if user_id % 1000 == 0:
                print("{}/{}".format(user_id, len(training_data)))

            pos, neg = training_data[user_id]
            assert len(neg) >= 100, f'fail train neg for user {user_id}'

            items_np = np.zeros(self.n_items, dtype=np.int8)
            items_np[pos] = 1

            pos, neg = self._generate_pairs(pos)

            # print('pos:',pos)
            # print('neg:',neg)

            # for start_idx in range(0, len(pos), self.chunk_size):
            #   end_idx = min(start_idx + self.chunk_size, len(pos))
            #  temp_pos = pos[start_idx:end_idx]
            # temp_neg = neg[start_idx:end_idx]

            # print('temp_pos:', temp_pos)
            # print('temp_neg:', temp_neg)

            train.append(items_np)
            self.pos['train'].append(pos)
            self.neg['train'].append(neg)

        print('self.max_width:', self.max_width)

        self.data['train'] = np.array(train, dtype=np.int8)
        # self.data['train'] = np.memmap('train_memmapped.dat', dtype=np.int8, mode='w+', shape=(len(train), self.n_items))
        self.data['train'][:] = train[:]
        self.size['train'] = len(train)

        # del train
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

            # for start_idx in range(0, len(pos), self.chunk_size):
            #   end_idx = min(start_idx + self.chunk_size, len(pos))
            #  temp_pos = pos[start_idx:end_idx]
            # temp_neg = neg[start_idx:end_idx]

            validation.append(items_np)

            self.pos['validation'].append(pos)
            self.neg['validation'].append(neg)

            self.pos_rank['validation'].append(positives_te)
            self.neg_rank['validation'].append(negatives_sampled)

        if len(validation) > 0:
            self.data['validation'] = np.array(validation, dtype=np.int8)
            # self.data['validation'] = np.memmap('validation_memmapped.dat', dtype=np.int8, mode='w+', shape=(len(validation), self.n_items))
            self.data['validation'][:] = validation[:]
            self.size['validation'] = len(validation)
            # del validation
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

            # for start_idx in range(0, len(pos), self.chunk_size):
            #   end_idx = min(start_idx + self.chunk_size, len(pos))
            #  temp_pos = pos[start_idx:end_idx]
            # temp_neg = neg[start_idx:end_idx]

            test.append(items_np)

            self.pos['test'].append(pos)
            self.neg['test'].append(neg)

            self.pos_rank['test'].append(positives_te)
            self.neg_rank['test'].append(negatives_sampled)

        if len(test) > 0:
            self.data['test'] = np.array(test, dtype=np.int8)
            # self.data['test'] = np.memmap('test_memmapped.dat', dtype=np.int8, mode='w+', shape=(len(test), self.n_items))
            self.data['test'][:] = test[:]
            self.size['test'] = len(test)
            # del test
            gc.collect()

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
                # frequency = self.pos_neg_ratio
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

        # idxlist = np.arange(self._N)
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

            pos = self.pos[tag][start_idx:end_idx]
            neg = self.neg[tag][start_idx:end_idx]

            mask = self.mask[tag][start_idx:end_idx]

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
