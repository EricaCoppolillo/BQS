import time
import numpy as np

from data_loaders import model_types, CachedDataLoader


if __name__ == '__main__':

    MAX_BATCH = 10
    BATCH_SIZE = 256

    model_type_param = model_types.LOW
    # model_type_param = model_types.REWEIGHTING
    # model_type_param = model_types.BASELINE
    dataset_name = 'ml-1m'
    dataset_name = 'netflix_sample'
    fname = f'./data/{dataset_name}/data_rvae'

    start = time.time()
    dataloader = CachedDataLoader(fname, 123, decreasing_factor=500, model_type=model_type_param,
                                  clean_cache=True)
    end = time.time()

    batch_time = [time.time()]
    for batch_idx, (x, pos, neg, mask) in enumerate(dataloader.iter(batch_size=BATCH_SIZE)):
        batch_time[-1] = time.time() - batch_time[-1]
        print('-' * 20)
        print('x', x.shape)
        print('pos', pos.shape)
        print('neg', neg.shape)
        print('mask', mask.shape)
        print('*' * 20)

        batch_time.append(time.time())
        if batch_idx > MAX_BATCH:
            break

    batch_time_val = [time.time()]
    for tag in ('validation', 'test'):
        for batch_idx, (x, pos, neg, mask, pos_te, neg_te, mask_te) in enumerate(
                dataloader.iter_test(batch_size=BATCH_SIZE, tag=tag)):
            batch_time_val[-1] = time.time() - batch_time_val[-1]
            print('-' * 20)
            print('x', x.shape)
            print('pos', pos.shape)
            print('neg', neg.shape)
            print('mask', mask.shape)
            print('pos_te', len(pos_te))
            print('neg_te', len(neg_te))
            print('mask_te', mask_te.shape)
            print('*' * 20)

            batch_time_val.append(time.time())
            if batch_idx > MAX_BATCH:
                break

    # STATS
    print(f'\n\ninit time is {end - start} secs')
    print(f'train mean batch time is {np.mean(batch_time[:-1])} secs')
    print(f'valid/test mean batch time is {np.mean(batch_time_val[:-1])} secs')
