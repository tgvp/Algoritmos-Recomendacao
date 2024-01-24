import pandas as pd
import numpy as np
import cupy as cp


def calculate_pearson_correlation_gpu(v1: cp.ndarray, v2: cp.ndarray, verbose: bool = False) -> float:
    """Calculates the Pearson correlation coefficient between two vectors on GPU.

    Args:
        v1 (cupy.ndarray): Vector 1
        v2 (cupy.ndarray): Vector 2
        verbose (bool, optional): Print intermediate results. Defaults to False.

    Returns:
        float: Pearson correlation coefficient
    """
    # Pearson's correlation coefficient - to resolve data sparseness problem
    common_indices = cp.logical_and(v1 > 0, v2 > 0)

    # means of items rated (not zeros)
    v1_mean = cp.sum(v1) / cp.count_nonzero(v1)
    v2_mean = cp.sum(v2) / cp.count_nonzero(v2)

    # numerator of person correlation
    numerator = cp.sum((v1[common_indices] - v1_mean) * (v2[common_indices] - v2_mean))

    # denominator
    denominator = cp.sqrt(cp.sum((v1[common_indices] - v1_mean)**2)) * cp.sqrt(cp.sum((v2[common_indices] - v2_mean)**2))

    if verbose:
        print(f'Common indices: {common_indices}')
        print(f'Means: {v1_mean}, {v2_mean}')
        print(f'Numerator: {numerator}')
        print(f'Denominator: {denominator}')

    # zero division
    if denominator == 0:
        return 0

    return cp.asnumpy(numerator / denominator)

def calculate_similarity_gpu(user1_ratings, user2_ratings, similarity='cosine'):
    if similarity == 'cosine':
        # Cosine similarity
        magnitudes = cp.linalg.norm(user1_ratings) * cp.linalg.norm(user2_ratings)
        
        # avoiding zero division
        if magnitudes == 0:
            magnitudes += 1e-6
            
        return cp.dot(user1_ratings, user2_ratings) / magnitudes
    elif similarity == 'pearson':
        # Pearson's correlation coefficient - to resolve data sparseness problem
        return calculate_pearson_correlation_gpu(user1_ratings, user2_ratings)
    else:
        raise ValueError('Invalid similarity!')

def calculate_similarity_matrix_gpu(user_item_matrix: np.ndarray, similarity: str = 'cosine', verbose: bool = False):
    """Calculates the similarity matrix for all users on GPU.

    Args:
        user_item_matrix (numpy.ndarray): numpy.ndarray with user-item interactions
        similarity (str, optional): Similarity measure. Defaults to 'cosine'.

    Returns:
        numpy.ndarray: Similarity matrix
    """
    num_users = user_item_matrix.shape[0]

    # Transfer user_item_matrix to GPU
    user_item_matrix_gpu = cp.asarray(user_item_matrix)

    similarity_matrix = cp.zeros((num_users, num_users))

    # Calculates full matrix and sets diagonal to 1
    for i in range(num_users):
        for j in range(i+1, num_users):
            similarity_matrix[i, j] = calculate_similarity_gpu(user_item_matrix_gpu[i], user_item_matrix_gpu[j], similarity)
            similarity_matrix[j, i] = similarity_matrix[i, j]

    for i in range(num_users):
        similarity_matrix[i, i] = 1

    # Transfer the result back to CPU if necessary
    similarity_matrix_cpu = cp.asnumpy(similarity_matrix)

    if verbose:
        print(f'Similarity matrix shape: {similarity_matrix_cpu.shape}')
        print(f'Num users: {num_users}')

    return similarity_matrix_cpu