import glob
import logging
import os
import random

import tensorflow as tf
from tqdm import tqdm

import settings
from settings import *
from util.cmn import CollaborativeMemoryNetwork
from util.data import Dataset
from util.evaluation import evaluate_model, get_eval, get_model_scores

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_paths(dataset_str, loss_type, folder):
    dataset_path = 'data/preprocess/' + dataset_str + '/data.pickle'
    if folder is not None:
        model_path = 'result/' + dataset_str + '/loss_' + str(loss_type) + '/' + folder
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    else:
        model_path = 'result/' + dataset_str + '/loss_' + str(loss_type)

    return dataset_path, model_path


def run_training(dataset_str, loss_type, limit, folder, batch_size, learning_rate, init_min_validation_loss):
    tf.reset_default_graph()

    np.random.seed(settings.SEED)
    random.seed(settings.SEED)
    os.environ['PYTHONHASHSEED'] = str(settings.SEED)

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(settings.SEED)
    session = tf.Session(graph=tf.get_default_graph(), config=config)

    print('TRAINING ---------------------------------------------------------------------------------')

    dataset_path, model_path = get_paths(dataset_str, loss_type, folder)
    dataset = Dataset(dataset_path, limit=limit)

    # parameters
    epochs, embeddings_scale, \
    use_popularity, less_popular_items_first, \
    k, k_trainable, a, b, neg_items = get_train_parameters(dataset_str, loss_type)
    # ------------------------------------------

    pretrain_file_path = 'pretrain/' + dataset_str + '.npz'

    min_validation_loss = init_min_validation_loss

    print('INFO:')
    print('dataset:' + str(dataset_str))
    print('loss_type:', str(loss_type))
    print('learning_rate:', str(learning_rate))
    print('batch_size:', str(batch_size))
    print('base_directory:', str(model_path))

    # -------------------------------------------------------
    Config.logdir = model_path
    Config.save_directory = model_path
    Config.filename = dataset_path
    Config.embed_size = 50
    Config.batch_size = batch_size
    Config.hops = 2
    Config.l2 = 0.1
    Config.user_count = dataset.user_count
    Config.item_count = dataset.item_count
    Config.optimizer = 'rmsprop'
    Config.tol = 1e-5
    Config.neg_count = neg_items
    Config.optimizer_params = {'learning_rate': learning_rate, 'decay': 0.9, 'momentum': 0.9}
    Config.grad_clip = 5.0
    Config.decay_rate = 0.9
    Config.learning_rate = learning_rate
    Config.pretrain = pretrain_file_path
    Config.max_neighbors = dataset.max_user_neighbors
    Config.normalized_popularity = dataset.normalized_popularity
    Config.loss_type = loss_type
    Config.a = a
    Config.b = b
    Config.k = k
    Config.k_trainable = k_trainable
    Config.low_popularity_threshold = dataset.thresholds[0]
    Config.high_popularity_threshold = dataset.thresholds[1]
    Config.less_popular_items_first = less_popular_items_first
    # -------------------------------------------------------

    config = Config()

    tf.logging.info("\n\n%s\n\n" % config)

    model = CollaborativeMemoryNetwork(config)

    session.graph._unsafe_unfinalize()

    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())

    load_pretrained_embeddings = True

    if load_pretrained_embeddings:
        try:
            print('Loading Pretrained Embeddings from %s' % pretrain_file_path)
            pretrain = np.load(pretrain_file_path)

            session.run([
                model.user_memory.embeddings.assign(pretrain['user'] * embeddings_scale),
                model.item_memory.embeddings.assign(pretrain['item'] * embeddings_scale)
            ])

            print('Embeddings loaded!')
        except IOError as e:
            print('Embeddings not found!')

    # Train Loop
    results = []

    best_model_found = False

    for epoch in range(epochs):
        print('Epoch n. {}'.format(epoch))

        progress = tqdm(enumerate(dataset.get_data(batch_size, True, neg_items, use_popularity=use_popularity)),
                        dynamic_ncols=True, total=(dataset.train_size * neg_items) // batch_size)

        loss = []

        for k, example in progress:
            ratings, pos_neighborhoods, pos_neighborhood_length, neg_neighborhoods, neg_neighborhood_length = example

            feed = {
                model.input_users: ratings[:, 0],

                model.input_items: ratings[:, 1],
                model.input_positive_items_popularity: config.normalized_popularity[ratings[:, 1]],  # Added by LC

                model.input_items_negative: ratings[:, 2],
                model.input_negative_items_popularity: config.normalized_popularity[ratings[:, 2]],  # Added by LC

                model.input_neighborhoods: pos_neighborhoods,
                model.input_neighborhood_lengths: pos_neighborhood_length,
                model.input_neighborhoods_negative: neg_neighborhoods,
                model.input_neighborhood_lengths_negative: neg_neighborhood_length
            }

            verbose = False
            if verbose:
                print('\nBATCH ----------------------------')
                print('input_users:\n{}'.format(ratings[:, 0]))
                print('input_items:\n{}'.format(ratings[:, 1]))
                print('input_items_negative:\n{}'.format(ratings[:, 2]))
                print('model.input_positive_items_popularity:\n{}'.format(config.normalized_popularity[ratings[:, 1]]))
                print('pos_neighborhoods:\n{}'.format(pos_neighborhoods))
                print('pos_neighborhood_length:\n{}'.format(pos_neighborhood_length))
                print('neg_neighborhoods:\n{}'.format(neg_neighborhoods))
                print('neg_neighborhood_length:\n{}'.format(neg_neighborhood_length))

            if Config.loss_type == IMPROVED_LOSS:
                batch_loss, _, parameter_k = session.run([model.loss, model.train, model.k], feed)
            else:
                parameter_k = 0
                batch_loss, _ = session.run([model.loss, model.train], feed)

            loss.append(batch_loss)
            progress.set_description(u"[{}] Loss (type={}, lr={}, bs={}): {:,.4f} » » » » ".format(epoch, Config.loss_type, learning_rate, batch_size, batch_loss))

        print("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(epoch, np.mean(loss)))

        hrs, \
        ndcgs, \
        hits_list, \
        normalized_hits_list, \
        validation_loss, \
        hrs_low, \
        hrs_medium, \
        hrs_high = evaluate_model(session,
                                  dataset.validation_data,
                                  dataset.item_users_list,
                                  model.input_users,
                                  model.input_items,
                                  model.input_neighborhoods,
                                  model.input_neighborhood_lengths,
                                  model.dropout,
                                  model.score,
                                  config.max_neighbors,
                                  model)

        results.append([np.mean(loss),
                        validation_loss,
                        hrs[0],
                        hrs_low[0],
                        hrs_medium[0],
                        hrs_high[0],
                        hrs[1],
                        hrs_low[1],
                        hrs_medium[1],
                        hrs_high[1],
                        hrs[2],
                        hrs_low[2],
                        hrs_medium[2],
                        hrs_high[2]
                        ])

        print('______________________________________________________________________________________________________________________')
        print('RESULTS AT Epoch {} ({} - path: {}):'.format(epoch, dataset_str, model_path))
        print('Ep.'
              '\tHR@1\tHR_L@1\tHR_M@1\tHR_H@1'
              '\tHR@5\tHR_L@5\tHR_M@5\tHR_H@5'
              '\tHR@10\tHR_L@10\tHR_M@10\tHR_H@10'
              '\tLoss\tV. Loss')

        for row in range(len(results)):
            row_string = '{:02d}'.format(row) + '\t' + \
                         format(results[row][2], '.4f') + '\t' + \
                         format(results[row][3], '.4f') + '\t' + \
                         format(results[row][4], '.4f') + '\t' + \
                         format(results[row][5], '.4f') + '\t' + \
                         format(results[row][6], '.4f') + '\t' + \
                         format(results[row][7], '.4f') + '\t' + \
                         format(results[row][8], '.4f') + '\t' + \
                         format(results[row][9], '.4f') + '\t' + \
                         format(results[row][10], '.4f') + '\t' + \
                         format(results[row][11], '.4f') + '\t' + \
                         format(results[row][12], '.4f') + '\t' + \
                         format(results[row][13], '.4f') + '\t' + \
                         format(results[row][0], '.4f') + '\t' + \
                         format(results[row][1], '.4f')

            print(row_string)
        print('______________________________________________________________________________________________________________________')

        if min_validation_loss is None or validation_loss < min_validation_loss:
            print("Saving model in {}".format(model_path))
            print('Found a better model!')
            min_validation_loss = validation_loss
            config.optimizer_params = {'learning_rate': learning_rate, 'decay': 0.9, 'momentum': 0.9}
            config.best_model_found_at_epoch = epoch
            config.save()
            saver.save(session, model_path + "/best_model.ckpt")

            best_model_found = True

    # Saving pictures!
    if best_model_found:
        print("Saving pictures in {}".format(model_path))
        save_pictures(model_path, results)

    # remove useless files
    fileList = glob.glob(model_path + '/events*')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except IOError as E:
            print("Error while deleting file : ", filePath)

    return min_validation_loss


def run_test(dataset_str=CITEULIKE, loss_type=BASELINE_LOSS, folder='grid_search'):
    print('TESTING ----------------------------------------------------------------------------------')

    dataset_path, model_path = get_paths(dataset_str, loss_type, folder)
    dataset = Dataset(dataset_path)

    print('Restoring best model...')
    tf.reset_default_graph()

    config = Config()

    config.save_directory = model_path
    config.load()

    config.optimizer_params = eval(config.optimizer_params)
    Config.normalized_popularity = dataset.normalized_popularity

    model = CollaborativeMemoryNetwork(config)
    saver = tf.train.Saver()
    session = tf.Session()
    saver.restore(session, model_path + "/best_model.ckpt")
    print("Bst Model restored!")

    print('Evaluating model on the test set...')
    EVAL_AT = range(1, 11)
    hrs, ndcgs, hits_list, normalized_hits_list = [], [], [], []
    s = ""
    scores, items, out, loss, user_items, user_scores = get_model_scores(session,
                                                                         dataset.test_data,
                                                                         dataset.item_users_list,
                                                                         model.input_users,
                                                                         model.input_items,
                                                                         model.input_neighborhoods,
                                                                         model.input_neighborhood_lengths,
                                                                         model.dropout,
                                                                         model.score,
                                                                         config.max_neighbors,
                                                                         model,
                                                                         return_scores=True)

    for k in EVAL_AT:
        hr, ndcg, hits, normalized_hits, hr_low, hr_medium, hr_high, n_pop = get_eval(scores, items, len(scores[0]) - 1, k)
        hrs.append(hr)
        hits_list.append(hits)
        normalized_hits_list.append(normalized_hits)
        ndcgs.append(ndcg)

        s += "{:<10} {:<3.4f} " \
             "{:<10} {:<3.4f} " \
             "{:<10} {:<3.4f} " \
             "{:<10} {:<3.4f} " \
             "{:<10} {} " \
             "{:<10} {} \n". \
            format('HR@%s' % k, hr,
                   'HR_LOW@%s' % k, hr_low,
                   'HR_MED@%s' % k, hr_medium,
                   'HR_HIGH@%s' % k, hr_high,
                   'HITS@%s' % k, str(hits),
                   'N_POP@%s' % k, str(n_pop))

    s += "Avg Loss on Test Set (each loss value is computed on (user, pos, [neg_1, ..., neg_100])): " + str(loss)
    print(s)

    with open("{}/final_results.txt".format(config.logdir), 'w') as fout:
        fout.write(s)


if __name__ == "__main__":
    # Example:
    # nohup /home/caroprese/CMN/venv/bin/python3 -u /home/caroprese/CMN/train.py > result/pinterest/loss_0/console.txt 2>&1 &

    # ACTIONS --------------------------------------
    TRAIN = True
    TEST = True
    # -----------------------------------------------

    gpu = '0'
    dataset_str = EPINIONS
    loss_type = IMPROVED_LOSS  # IMPROVED_LOSS
    limit = 0.05

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('Tensorflow Ver.', tf.__version__)

    if TRAIN:
        learning_rates = [0.001, 0.0001]
        batch_sizes = [128, 256]
        grid_search = False

        if grid_search:
            current_min_validation_loss = None
            best_learning_rate = None
            best_batch_size = None

            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    updated_min_validation_loss = run_training(dataset_str, loss_type, limit, 'grid_search', batch_size, learning_rate, current_min_validation_loss)

                    if current_min_validation_loss is None or updated_min_validation_loss < current_min_validation_loss:
                        best_learning_rate = learning_rate
                        best_batch_size = batch_size
        else:
            best_learning_rate = 0.00001
            best_batch_size = 256

        print('GRID SEARCH COMPETED!')
        print('(best) batch size:', best_batch_size)
        print('(best) learning rate:', best_learning_rate)

        print('\nFinal Training...')
        limit_2 = None  # 0.01
        run_training(dataset_str, loss_type, limit_2, 'old', best_batch_size, best_learning_rate, None)

        print('Training completed!')

    if TEST:
        run_test(dataset_str, loss_type, None)

    print('Done!')
