# Created by Hansi at 6/30/2021
import csv

import numpy as np


def save_vocabulary(file_path, vocab):
    """
    Method to save a vocabulary(or list of tokens) to a binary file in NumPy (.npy format)

    parameters
    -----------
    :param file_path: str
        Path to file (if extension is not given, it will be added during saving)
    :param vocab: list
    :return: {0, 1}
        0 - save is unsuccessful
        1 - save is successful
    """
    try:
        np.save(file_path, vocab)
        return 1
    except:
        return 0


def load_wordcounts(filepath):
    """
    Method to load word counts saved in .tsv file to a dictionary.
    File format [token \t count]

    parameters
    -----------
    :param filepath: str
        .tsv file path
    :return: dictionary
        Dictionary of words and their corresponding counts (word:count)
    """
    word_counts = dict()
    csv_file = open(filepath, encoding='utf-8')
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        word_counts[row[0]] = int(row[1])
    return word_counts


def get_word_diff(words1, words2, assume_unique=False):
    """
    Get additional words in words2 compared to words1

    parameters
    -----------
    :param words1: list of str tokens
    :param words2: list of str tokens
    :param assume_unique: boolean, optional
    :return: int, list
        Number of additional words in words2
        List of additional words in words2
    """
    word_diff = np.setdiff1d(words2, words1, assume_unique)
    return len(word_diff), word_diff


def filter_vocabulary_by_frequency(words, word_freq, frequency):
    """
    Method to remove less frequent words in vocabulary

    parameters
    -----------
    :param words: list
    :param word_freq: dictionary
        Dictionary of words and their corresponding counts (word:count)
    :param frequency: int
        Frequency threshold
        Words with less count than the threshold will be removed.
    :return: list
        List of words with high frequency
    """
    if frequency == 0:
        return words

    filtered_words = []
    for word in words:
        if word in word_freq and (int(word_freq[word]) >= frequency):
            filtered_words.append(word)
    return filtered_words


def filter_vocabulary_by_frequency_diff(words, word_freq1, word_freq2, threshold):
    """
    Filter words if their freq. diff over t2 and t1 >= threshold

    parameters
    -----------
    :param words: list
    :param word_freq1: dictionary
        Dictionary of words and their corresponding counts (word:count)
    :param word_freq2: dictionary
        Dictionary of words and their corresponding counts (word:count)
    :param threshold: int
    :return: list, dictionary
        List of words with frequency diff above the threshold
        Dictionary of words and their freq, diffs (word:freq_diff)

    """
    dict_word_diff = dict()
    filtered = []
    for word in words:
        f1 = word_freq1[word] if word in word_freq1 else 0
        f2 = word_freq2[word] if word in word_freq2 else 0
        diff = f2 - f1
        dict_word_diff[word] = diff
        if diff >= threshold:
            filtered.append(word)
    return filtered, dict_word_diff
