# Created by Hansi at 6/30/2021
import multiprocessing

import numpy as np

from algo.change_measures.dendrogram_level_calculation import get_normalized_dendrogram_level_similarity, \
    get_local_dendrogram_level_similarity
from algo.utils.word_embedding_utils import get_embedding


def get_vectors_for_words(words, model):
    """
    Get embeddings of given words, if embeddings are available in the model

    parameters
    -----------
    :param words: list
        List of words
    :param model: object
        Word embedding model
    :return: list, list
        List of words which have embeddings
        List of embeddings
    """
    word_list = []
    vector_list = []
    for w in words:
        try:
            vector_list.append(get_embedding(w, model))
            word_list.append(w)
        except KeyError:
            pass
    return word_list, vector_list


def get_cosine_similarity_for_words(word, words, model):
    """
    Get cosine similarity between given word and each word in the given word list

    parameters
    -----------
    :param word: str
    :param words: list
        List of words
    :param model: object
        Word embedding model
    :return: list of float
        List of cosine similarity values
    """
    result_array = []
    for w in words:
        try:
            similarity = model.wv.similarity(word, w)

        except KeyError:
            similarity = 0
        result_array.append(similarity)
    return result_array


def get_dendrogram_level_similarity_for_words(word, words, label_codes, max_level_count):
    """
    Get DL similarity between given word and each word in the given word list

    parameters
    -----------
    :param word: str
    :param words: list
        List of words
    :param label_codes: object
        Dictionary of level codes (label:level_code) which is returned by generate_dendrogram_level_codes
    :param max_level_count: int
        Maximum number of levels in the dendrogram
    :return: list of float
        List of DL similarity values
    """
    result_array = []
    for w in words:
        try:
            similarity, common_code_length, max_length = get_normalized_dendrogram_level_similarity(word, w,
                                                                                                    label_codes,
                                                                                                    max_level_count)

        except KeyError:
            similarity = 0
        result_array.append(similarity)
    return result_array


def get_local_dendrogram_level_similarity_for_words(word, words, label_codes):
    """
    Get LDL similarity between given word and each word in the given word list

    parameters
    -----------
    :param word: str
    :param words: list
        List of words
    :param label_codes: object
        Dictionary of level codes (label:level_code) which is returned by generate_dendrogram_level_codes
    :return: list of float
        List of LDL similarity values
    """
    result_array = []
    for w in words:
        try:
            similarity, common_code_length, max_length = get_local_dendrogram_level_similarity(word, w, label_codes)

        except KeyError:
            similarity = 0
        result_array.append(similarity)
    return result_array


def get_cosine_similarity_matrix_for_words(model, words, workers=1):
    """
    Generate cosine similarity matrix (included parallel implementation)

    parameters
    -----------
    :param model: object
        Word embedding model
    :param words: list of str
        List of words
    :param workers: int, optional
        Number of worker threads to use with matrix generation.
    :return: matrix
        Similarity matrix of given words
    """
    pool = multiprocessing.Pool(workers)

    inputs = []
    for w1 in words:
        inputs.append([w1, words, model])

    similarity_array = pool.starmap(get_cosine_similarity_for_words, inputs)
    pool.close()
    pool.join()
    similarity_matrix = np.asmatrix(similarity_array)
    return similarity_matrix


def get_dendrogram_level_similarity_matrix_for_words(label_codes, words, max_level_count, workers=1):
    """
    Generate DL similarity matrix (included parallel implementation)

    parameters
    -----------
    :param label_codes: object
        Dictionary of level codes (label:level_code) which is returned by generate_dendrogram_level_codes
    :param words: list of str
        List of words
    :param max_level_count: int
        Maximum number of levels in the dendrogram
    :param workers: int, optional
        Number of worker threads to use with matrix generation.
    :return: matrix
        Similarity matrix of given words
    """
    pool = multiprocessing.Pool(workers)

    inputs = []
    for w1 in words:
        inputs.append([w1, words, label_codes, max_level_count])

    similarity_array = pool.starmap(get_dendrogram_level_similarity_for_words, inputs)
    pool.close()
    pool.join()
    similarity_matrix = np.asmatrix(similarity_array)
    return similarity_matrix


def get_local_dendrogram_level_similarity_matrix_for_words(label_codes, words, workers=1):
    """
    Generate LDL similarity matrix (included parallel implementation)

    parameters
    -----------
    :param label_codes: object
        Dictionary of level codes (label:level_code) which is returned by generate_dendrogram_level_codes
    :param words: list of str
        List of words
    :param workers: int, optional
        Number of worker threads to use with matrix generation.
    :return: matrix
        Similarity matrix of given words
    """
    pool = multiprocessing.Pool(workers)

    inputs = []
    for w1 in words:
        inputs.append([w1, words, label_codes])

    similarity_array = pool.starmap(get_local_dendrogram_level_similarity_for_words, inputs)
    pool.close()
    pool.join()
    similarity_matrix = np.asmatrix(similarity_array)
    return similarity_matrix
