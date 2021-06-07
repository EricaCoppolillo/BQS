import matplotlib.pyplot as plt
import numpy as np

RANDOMNESS = False
SEED = 0

# LOSS TYPE -------------------------------------
BASELINE_LOSS = 0
IMPROVED_LOSS = 1

# DATASETS --------------------------------------
CITEULIKE = 'citeulike-a'
EPINIONS = 'epinions'
MOVIELENS_1M = 'ml-1m'
MOVIELENS_20M = 'ml-20m'
NETFLIX = 'netflix'
NETFLIX_SAMPLE = 'netflix_sample'
PINTEREST = 'pinterest'


class Config(BaseConfig):
    logdir = None
    filename = None
    embed_size = None
    batch_size = None
    hops = None
    l2 = None
    user_count = None
    item_count = None
    optimizer = None
    tol = None
    neg_count = None
    optimizer_params = None
    grad_clip = None
    decay_rate = None
    learning_rate = None
    pretrain = None
    max_neighbors = None

    # new parameters
    normalized_popularity = None
    loss_type = None  # 0 original, 1 CNR first version, 2 CNR second version
    a = 100
    b = 100
    k = 10000
    k_trainable = False
    low_popularity_threshold = 0.05
    high_popularity_threshold = 0.25
    less_popular_items_first = True
    best_model_found_at_epoch = None


def get_train_parameters(dataset_str, loss_type):
    epochs = 30
    embeddings_scale = 0.5

    k = 50
    k_trainable = False

    if loss_type == BASELINE_LOSS:
        a = b = 0
        neg_items = 4
        use_popularity = False  # [Evaluation phase] If use_popularity==True, a negative item N wrt a positive item P, can be a positive item with a lower popularity than P
        less_popular_items_first = True
    else:
        a = b = 0
        neg_items = 2
        use_popularity = True  # [Evaluation phase] If use_popularity==True, a negative item N wrt a positive item P, can be a positive item with a lower popularity than P
        less_popular_items_first = True

    return epochs, \
           embeddings_scale, \
           use_popularity, less_popular_items_first, \
           k, k_trainable, \
           a, b, \
           neg_items


def save_pictures(path, results):
    results_array = np.array(results)

    fig, axes = plt.subplots(4, 3, figsize=(20, 20))

    axes = axes.ravel()
    i = 0

    for r, r_name in enumerate('HR HR_LOW HR_MED HR_HIGH'.split()):
        for c, c_name in enumerate([1, 5, 10]):
            column = c * 4 + 2 + r
            hitRate = results_array[:, column]

            ax = axes[i]
            i += 1

            ax.plot(hitRate)
            ax.set_title(r_name + '@' + str(c_name))

    # plt.show();
    plt.savefig(path + '/metrics.png')

    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    ax = axes[0]
    ax.plot(results_array[:, 0])
    ax.set_title('Training Loss')

    ax = axes[1]
    ax.plot(results_array[:, 1])
    ax.set_title('Validation Loss')

    plt.savefig(path + '/train_validation_loss.png')
