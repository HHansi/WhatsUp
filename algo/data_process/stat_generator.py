# Created by Hansi at 4/8/2021
import collections
import logging
import os

import numpy as np
import pandas as pd

from algo.event_identification.event_word_extractor import load_events
from algo.utils.file_utils import read_list_from_text_file, delete_create_folder, read_text_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


def save_counter(counter, filepath):
    """
    Method to save counter to a .tsv file

    parameters
    -----------
    :param counter: collections.Counter
    :param filepath: str (path to .tsv file)
    :return: int
        Number of elements in the counter
    """
    with open(filepath, 'w+', encoding="utf8") as f:
        i = 0
        for k, v in counter:
            if k.strip():
                if k != '\"':
                    f.write("{}\t{}\n".format(k, v))
                    i += 1
    return i


def get_word_frequencies(df, counter_file_path):
    """
    Method to count token frequencies within given dataframe.
    Used split by spaces to tokenise text.

    parameters
    -----------
    :param df: dataframe
        Only the text column dataframe
    :param counter_file_path: str (.tsv file path)
        File path to save frequency details
    :return: int, int
        Total number of words found.
        Number of distinct words  found.
    """
    results = list()
    i = 0
    for text in df:
        i += 1
        results.extend(text.split())

    all_words = len(results)

    word_counter = collections.Counter(results)
    word_counter = word_counter.most_common()

    distinct_words = save_counter(word_counter, counter_file_path)

    return all_words, distinct_words


def generate_stats(input_folder_path, output_folder_path):
    """
    Generate statistical details of text in each file in the input_folder.
    As stat. details, frequency of tokens will be measured.

    parameters
    -----------
    :param input_folder_path: str
        Path to input data folder
    :param output_folder_path: str
        Path to save statistical details.
        For each file in the input_folder, separate file with same name will be created. This file will contain the
        distinct tokens and their frequencies.
        Another file named 'WordCounts.tsv' will be created which contains the total number of distinct words found in
        each time window.
    :return:
    """
    # delete if there already exist a folder and create new folder
    delete_create_folder(output_folder_path)

    output_file_path = os.path.join(output_folder_path, 'WordCounts.tsv')
    df_output = pd.DataFrame(columns=['File_Name', 'Word_Count', 'Distinct_Word_Count'])

    for root, dirs, files in os.walk(input_folder_path):
        for file in files:
            file_path = os.path.join(input_folder_path, file)
            file_name = os.path.splitext(file_path)[0]

            data = read_text_column(file_path)
            count_path = os.path.join(output_folder_path, file)
            all_words, distinct_words = get_word_frequencies(data, count_path)
            df_output = df_output.append(
                {'File_Name': file_name, 'Word_Count': all_words, 'Distinct_Word_Count': distinct_words},
                ignore_index=True)

        df_output.to_csv(output_file_path, sep='\t', mode='a', index=False, encoding='utf-8')


def count_words(folder_path, result_file_path=None):
    """
    Count words in the files in given folder_path

    parameters
    -----------
    :param folder_path:
    :param result_file_path:
    :return:
    """
    if result_file_path:
        df = pd.DataFrame(columns=['file', 'count'])

    i = 0
    counts = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            words = read_list_from_text_file(os.path.join(folder_path, file))
            counts.append(len(words))
            if result_file_path:
                df.loc[i] = [os.path.splitext(file)[0], len(words)]
                i += 1

    average = np.mean(counts)
    if result_file_path:
        df.loc[i] = ['average', average]
        df.to_csv(result_file_path, index=False)
    logger.info(f'average: {average}')


def get_event_counts(folder_path, result_file_path=None):
    """
    Count events in the .txt files in the given folder

    :param folder_path: str
        Folder path which contain .txt files of saved events
    :param result_file_path: str
        Path to .csv file to save event counts
    :return:
    """
    type = "emerging"
    if result_file_path:
        df = pd.DataFrame(columns=['file', 'count'])

    i = 0
    counts = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            events = load_events(os.path.join(folder_path, dir, type + '.txt'))
            counts.append(len(events))
            if result_file_path:
                df.loc[i] = [dir, len(events)]
                i += 1

    average = np.mean(counts)
    if result_file_path:
        df.loc[i] = ['average', average]
        df.to_csv(result_file_path, index=False)
    logger.info(f'average: {average}')
