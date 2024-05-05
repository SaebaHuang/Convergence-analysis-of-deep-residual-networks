# encoding=UTF-8
import numpy as np
import sys
from os.path import exists


def generate_doubly_stochastic_matrix(m=1):
    ds_matrix = np.random.random(size=(m, m)) + 1e-5 * np.eye(m)

    while True:
        ds_matrix /= ds_matrix.sum(0)
        ds_matrix /= ds_matrix.sum(1)[:, np.newaxis]

        row_sum = ds_matrix.sum(1)
        col_sum = ds_matrix.sum(0)

        if np.any(np.abs(row_sum - 1) > 1e-8):
            continue
        if np.any(np.abs(col_sum - 1) > 1e-8):
            continue
        break

    return ds_matrix


def generate_mask(m=1):
    negative_positive = np.array([-1, 1])
    mask_matrix = np.random.choice(negative_positive, size=(m, m), replace=True)
    return mask_matrix


def generate_normalized_norm_matrix(m=1):
    ds_matrix = generate_doubly_stochastic_matrix(m)
    mask_matrix = generate_mask(m)
    normalized_norm_matrix = np.multiply(mask_matrix, ds_matrix)
    assert np.abs(np.linalg.norm(normalized_norm_matrix, ord=1) - 1.0) < 1e-8
    assert np.abs(np.linalg.norm(normalized_norm_matrix, ord=np.inf) - 1.0) < 1e-8
    return normalized_norm_matrix


def generate_normalized_norm_vector(m=1):
    random_vector = np.random.randn(m)
    normalized_random_vector = random_vector / np.linalg.norm(random_vector, ord=np.inf)
    return normalized_random_vector


def logger(msg, txt_dir):
    default_std_out = sys.stdout
    if not exists(txt_dir):
        experiment_results_logger = open(txt_dir, 'x')
    else:
        experiment_results_logger = open(txt_dir, 'a')
    sys.stdout = experiment_results_logger
    print(msg)
    sys.stdout = default_std_out
    experiment_results_logger.close()
    print(msg)
    return
