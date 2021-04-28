import random

import tensorflow as tf
from tqdm import tqdm
import logging
import os

import settings
from settings import *
from util.data import Dataset
from util.gmf import PairwiseGMF

datasets_to_pretrain = [NETFLIX_SAMPLE]
epochs = 15
gpu = '0'

# -----------------------------------------------
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

if not settings.RANDOMNESS:
    np.random.seed(settings.SEED)
    random.seed(settings.SEED)
    tf.set_random_seed(settings.SEED)

for ds in datasets_to_pretrain:

    tf.reset_default_graph()
    tf.Graph().as_default()

    print('PRETRAINING ----------------------------------------------------')
    print(ds)

    Config.loss_type = 0
    Config.filename = 'data/preprocess/' + ds + '/data.pickle'
    Config.embed_size = 50
    Config.batch_size = 256
    Config.l2 = 0.001
    Config.user_count = -1
    Config.item_count = -1
    Config.optimizer = 'adam'
    Config.neg_count = 4
    Config.learning_rate = 0.001

    dataset = Dataset(Config.filename, limit=None)
    config = Config()

    config.normalized_popularity = dataset.normalized_popularity
    config.item_count = dataset.item_count
    config.user_count = dataset.user_count

    tf.logging.info("\n\n%s\n\n" % config)

    model = PairwiseGMF(config)
    sv = tf.train.Supervisor(logdir=None, save_model_secs=0, save_summaries_secs=0)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)))

    for i in range(epochs):
        if sv.should_stop():
            break

        progress = tqdm(enumerate(dataset.get_data(config.batch_size, True, config.neg_count, use_popularity=False)), dynamic_ncols=True, total=(dataset.train_size * config.neg_count) // config.batch_size)
        loss = []
        for k, example in progress:
            ratings, pos_neighborhoods, pos_neighborhood_length, neg_neighborhoods, neg_neighborhood_length = example
            feed = {
                model.input_users: ratings[:, 0],
                model.input_items: ratings[:, 1],
                model.input_items_negative: ratings[:, 2],
                model.input_positive_items_popularity: config.normalized_popularity[ratings[:, 1]],  # Added by LC
                model.input_negative_items_popularity: config.normalized_popularity[ratings[:, 2]]  # Added by LC
            }
            batch_loss, _ = sess.run([model.loss, model.train], feed)

            loss.append(batch_loss)
            progress.set_description(u"[{}] Loss (type={}): {:,.4f} » » » » ".format(i, Config.loss_type, batch_loss))

        print("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(i, np.mean(loss)))

    user_embed, item_embed, v = sess.run([model.user_memory.embeddings, model.item_memory.embeddings, model.v.w])
    save_path = './pretrain/' + ds + '.npz'
    np.savez(save_path, user=user_embed, item=item_embed, v=v)

    print('Saving to: %s' % save_path)

    sv.request_stop()

print('Done!')
