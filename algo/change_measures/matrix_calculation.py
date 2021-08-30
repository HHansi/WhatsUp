# Created by Hansi at 6/30/2021
import os

import numpy as np
import pandas as pd
from orderedset import OrderedSet

from algo.utils.file_utils import create_folder_if_not_exist


def get_absolute_matrix_diff(matrix1, matrix2):
    """
    Method to get absolute difference matrix

    parameters
    -----------
    :param matrix1: matrix
    :param matrix2: matrix
    :return: matrix
        |matrix1 - matrix2|
    """
    return np.absolute(matrix1 - matrix2)


def get_upper_triangular_matrix(matrix, index):
    """
    Method to get upper triangular

    parameters
    -----------
    :param matrix: matrix
    :param index: int
        Elements below the `index`-th diagonal will be zeroed.
        (0-include diagonal, 1-without diagonal)
    :return: matrix
    """
    return np.triu(matrix, index)


def save_matrix(file_path, matrix):
    """
    Method to save a matrix to a binary file in NumPy (.npy format)

    parameters
    -----------
    :param file_path: str
        Path to file (if extension is not given, it will be added during saving)
    :param matrix: matrix
    :return: {0, 1}
        0 - save is unsuccessful
        1 - save is successful
    """
    try:
        np.save(file_path, matrix)
        return 1
    except:
        return 0


def load_matrix(file_path):
    """
    Method to load saved matrix in .npy file

    parameters
    -----------
    :param file_path: str
        Path to file
    :return: matrix
    """
    if os.path.splitext(file_path)[1] != '.npy':
        file_path = file_path + '.npy'
    return np.load(file_path)


def get_upper_triangular_as_list(matrix, index=1, descending=True):
    """
    Method to get upper triangular of a matrix as a list of elements

    parameters
    -----------
    :param matrix: matrix
    :param index: int, optional
        0- include diagonal, 1-without diagonal
    :param descending: boolean, optional
        Boolean to indicate the list sorting order
    :return: list
        Upper triangular as a list
    """
    matrix_length = len(matrix)
    lst = list(matrix[np.triu_indices(matrix_length, k=index)])
    lst.sort(reverse=descending)
    return lst


def get_sorted_matrix_labels(label_list, matrix, descending=False, file_path=None, word_pair_limit=None,
                             non_zeros_only=False):
    """
    Method to get list of matrix labels and corresponding values
    While adding the labels, row label is added prior to corresponding column label. Same value is repeatedly added for
    both row and column label.

    parameters
    -----------
    :param label_list: list of str
        List of labels
    :param matrix: matrix
    :param descending: boolean, optional
        Boolean to indicate the sorting order of matrix cell values in final nd-array
    :param file_path: str, optional
        .csv file path to save cell values with row and column labels for analysis purpose
    :param word_pair_limit: int, optional
        Limits the number of sorted cells need to be considered.
    :param non_zeros_only: boolean, optional
        Boolean to indicate the inclusion of non-zero values
    :return: list of str, list of float
        List of sorted matrix labels
        List of values corresponding to the sorted labels
    """
    sorted_labels = OrderedSet()
    label_count = 0
    values = []

    # sort matrix
    if descending:
        sorted = np.argsort(matrix, axis=None)[::-1]
    else:
        sorted = np.argsort(matrix, axis=None)
    rows, cols = np.unravel_index(sorted, matrix.shape)
    matrix_sorted = matrix[rows, cols]

    if file_path:
        df = pd.DataFrame(columns=['value', 'row', 'column'])

    i = 0
    for r, c, v in zip(rows, cols, matrix_sorted):
        i = i + 1
        if non_zeros_only and v == 0:
            continue

        sorted_labels.add(label_list[r])
        if len(sorted_labels) > label_count:  # if the label is added newly, update values
            label_count = len(sorted_labels)
            values.append(v)
        sorted_labels.add(label_list[c])
        if len(sorted_labels) > label_count:
            label_count = len(sorted_labels)
            values.append(v)
        if file_path:
            df.loc[i] = [v, label_list[r], label_list[c]]

        if word_pair_limit and i == word_pair_limit:
            break

    if file_path:
        create_folder_if_not_exist(file_path, is_file_path=True)
        df.to_csv(file_path, index=False)
    return list(sorted_labels), values


def get_proportion(array, positive=True):
    """
    Return proportion of signed values

    parameters
    -----------
    :param array: array of numbers
    :param positive: boolean, optional
        True-consider positive value proportion, False-consider negative value proportion
    :return: float, list
        Number of signed elements/total number of elements
        List of signed values
    """
    array = np.array(array)
    if positive:
        filtered = array[array > 0]
    else:
        filtered = array[array < 0]
    proportion = len(filtered) / len(array)
    return proportion, filtered


def get_ut_matrix_values(label_list, matrix, non_zeros_only=False):
    """
    Return upper triangular values of the matrix

    parameters
    -----------
    :param label_list: list of str
        List of labels
    :param matrix: matrix
    :param non_zeros_only: boolean, optional
        Boolean to indicate the inclusion of non-zero values
    :return: list
        List of [value, [label1, label2]
    """
    results = []
    row_count, col_count = matrix.shape

    r = -1
    while r < row_count - 1:
        r += 1
        c = r
        while c < col_count - 1:
            c += 1
            v = matrix[r, c]
            word_pair = [label_list[r], label_list[c]]
            word_pair.sort()  # added to support comparison of word pairs

            if non_zeros_only:
                if v != 0:
                    results.append([v, word_pair])
            else:
                results.append([v, word_pair])

    return results
