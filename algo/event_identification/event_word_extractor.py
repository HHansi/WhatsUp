# Created by Hansi at 6/30/2021
import logging
import os

import numpy as np
import pandas as pd

from algo.change_measures.matrix_calculation import get_sorted_matrix_labels, get_upper_triangular_matrix, \
    get_proportion
from algo.data_process.data_preprocessor import remove_emoji_and_non_alphanumeric
from algo.utils.file_utils import write_list_to_text_file, create_folder_if_not_exist
from algo.utils.word_embedding_utils import get_similar_words, load_model

logger = logging.getLogger(__name__)


class Event:
    def __init__(self, words, novelty):
        self.words = words
        self.novelty = novelty


def save_events(events, file_path, analysis_mode=False):
    """
    Save events to a .txt file

    parameters
    -----------
    :param events: list
        List of events (list of list of tokens)
    :param file_path: str
        File path (.txt) to save events
    :param analysis_mode: boolean, optional
        If true, additional new line will be print after each event to ease analysis
    :return:
    """
    create_folder_if_not_exist(file_path, is_file_path=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for event in events:
            event_str = ", ".join(event)
            if not analysis_mode:
                f.write("%s\n" % event_str)
            else:
                f.write("%s\n\n" % event_str)


def load_events(file_path):
    """
    Load events saved to a .txt file

    parameters
    -----------
    :param file_path: str
        Path to .txt file
    :return: dictionary
        Dictionary of event words- [incrementing ID: [event words]]
    """
    event_words = dict()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            i = 0
            for line in f.readlines():
                line = line.strip()
                event_words[i] = line.split(', ')
                i += 1
        return event_words
    except FileNotFoundError:
        return None


def save_event_objs(events, file_path):
    """
    Save detailed events to a .txt file

    parameters
    -----------
    :param events:  list
        list of Event objects
    :param file_path: str
        File path (.txt) to save events
    :return:
    """
    create_folder_if_not_exist(file_path, is_file_path=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for event in events:
            event_str = ", ".join(event.words)
            f.write("%s - " % event_str)
            f.write("%.4f\n" % event.novelty)


def get_fd(f_t1, f_t2, args):
    """
    Get frequency difference

    parameters
    -----------
    :param f_t1: int
        frequency at t1
    :param f_t2: int
        frequency at t2
    :param args:
    :return:
    """
    if args['token_similarity_change'] == 'abs':
        fd = abs(f_t2 - f_t1) / max(f_t2, f_t1)
    else:
        fd = (f_t2 - f_t1) / max(f_t2, f_t1)
    return fd


def sort_words(event_window, args, result_file=None):
    """
    Sort words based on the assigned weight

    parameters
    -----------
    :param event_window: object
        EventWindow object
    :param args: JSON
        JSON of arguments
    :param result_file: str, optional
        File path to save weight details words (for analysis purpose)
    :return:
    """
    df = pd.DataFrame(columns=['word', 'sd', 'fd', 'weight'])

    if args['token_weight'] == 'fd':
        for i in range(len(event_window.vocab)):
            # measure frequency change of words
            f_t1 = event_window.word_counts_t1[event_window.vocab[i]]
            f_t2 = event_window.word_counts_t2[event_window.vocab[i]]
            fd = get_fd(f_t1, f_t2, args)

            df.loc[i] = [event_window.vocab[i], 0, fd, fd]

    else:
        matrix = event_window.diff_matrix
        ut_matrix = get_upper_triangular_matrix(matrix, 1)

        if args['token_similarity_change'] == 'abs':
            matrix = np.absolute(event_window.diff_matrix)
            ut_matrix = np.absolute(ut_matrix)

        # using word pair similarity difference - psd
        if 'psd' in args['token_weight']:
            words, values = get_sorted_matrix_labels(event_window.vocab, np.asarray(ut_matrix),
                                                     descending=True, non_zeros_only=True)
            for i in range(len(words)):
                psd = values[i]
                if args['token_weight'] == 'psd':
                    fd = 0
                    weight = psd
                else:
                    # measure frequency change of words
                    f_t1 = event_window.word_counts_t1[event_window.vocab[i]]
                    f_t2 = event_window.word_counts_t2[event_window.vocab[i]]
                    fd = get_fd(f_t1, f_t2, args)

                    if args['token_weight'] == 'avg_psd_fd':  # average of pairwise similarity diff. and frequency diff.
                        weight = (psd + fd) / 2
                    elif args['token_weight'] == 'max_psd_fd':  # max of pairwise similarity diff. and frequency diff.
                        weight = max(psd, fd)
                    else:
                        raise KeyError('Unknown token weighting method found')
                df.loc[i] = [words[i], psd, fd, weight]
        # using average similarity difference - asd
        elif 'asd' in args['token_weight']:
            for i in range(len(event_window.vocab)):
                # calculate average similarity difference of the token
                similarity_values = matrix[i]
                if args['token_similarity_change'] == 'pos':
                    pos_proportion, filtered = get_proportion(similarity_values, positive=True)
                    if pos_proportion == 0:  # handle not return of nan values as asd
                        asd = 0
                    else:
                        asd = np.mean(filtered) * pos_proportion
                else:
                    asd = np.mean(similarity_values)

                if args['token_weight'] == 'asd':  # average of similarity differences
                    fd = 0
                    weight = asd
                else:
                    # measure frequency change of words
                    f_t1 = event_window.word_counts_t1[event_window.vocab[i]]
                    f_t2 = event_window.word_counts_t2[event_window.vocab[i]]
                    fd = get_fd(f_t1, f_t2, args)

                    if args['token_weight'] == 'avg_asd_fd':  # average of average similarity diff. and frequency diff.
                        weight = (asd + fd) / 2
                    elif args['token_weight'] == 'max_asd_fd':  # max of average similarity diff. and frequency diff.
                        weight = max(asd, fd)
                    else:
                        raise KeyError('Unknown token weighting method found')
                df.loc[i] = [event_window.vocab[i], asd, fd, weight]

        else:
            raise KeyError('Unknown token weighting method found')

    if args['token_weight'] == 'psd':
        sorted_words = list(df['word'])  # values are already sorted
    else:
        df.sort_values('weight', ascending=False, inplace=True)
        sorted_words = list(df['word'])

    if result_file:
        create_folder_if_not_exist(result_file, is_file_path=True)
        df.to_csv(result_file, index=False)

    return sorted_words, df


def get_event_keywords(dict_event_windows, result_folder, args, analysis_mode=False):
    """
    Get event keywords in each event window

    parameters
    -----------
    :param dict_event_windows: dictionary
        Dictionary of list of EventWindows
        key - frequency_threshold-change_threshold
    :param result_folder: str
        Folder path to save event keywords
    :param args: JSON
        JSON of arguments
    :param analysis_mode: boolean, optional
        If true, calculated word weights will be saved (for analysis purpose) to a .csv file.
        Files will be generated per each window at path- result_folder/key{-weights}/time_window.csv
    :return:
    """
    for key in dict_event_windows:
        logger.info(f'Processing {key}')
        for event_window in dict_event_windows[key]:
            time_window = event_window.time_window
            logger.info(f'Processing words in {time_window}')

            file_path = None
            if analysis_mode:
                file_path = os.path.join(result_folder, key + "-weights", time_window + '.csv')
            sorted_words, df_word_weights = sort_words(event_window, args, result_file=file_path)

            # Ignore filtering if n is found as a boolean
            if args['n'] and not isinstance(args['n'], bool):
                keywords = sorted_words[:args['n']]
            else:
                keywords = sorted_words

            result_file_path = os.path.join(result_folder, key, time_window + '.txt')
            create_folder_if_not_exist(result_file_path, is_file_path=True)
            write_list_to_text_file(keywords, result_file_path)


def get_similar_words_by_matrix(word, matrix, vocab, count_threshold=None, sim_threshold=None, analysis_mode=False):
    """
    Filter similar words according to given count or similarity threshold.
    Priority is given to count_threshold, if both the thresholds are provided. If none of the thresholds are provided
    reversely sorted similar words of whole vocabulary will be returned.

    parameters
    -----------
    :param word: str
    :param matrix: matrix
        Similarity matrix
    :param vocab: list of str
    :param count_threshold: int, optional
    :param sim_threshold: float, optional
    :param analysis_mode: boolean, optional
        If true, a detailed_output: a list of [word, similarity] will be returned.
        Otherwise an empty list will be returned as detailed_output
    :return: list, list
        list of similar words matched with the given criteria
        detailed_output
    """
    index = vocab.index(word)
    similarities = np.array(matrix[index])
    nd_array = np.column_stack((vocab, similarities.astype(np.object)))

    # sort by similarity
    nd_array = sorted(nd_array, key=lambda x: x[1], reverse=True)

    if count_threshold is not None:
        filtered = nd_array[:count_threshold]
    elif sim_threshold is not None:
        filtered = [row for row in nd_array if row[1] >= sim_threshold]
    else:
        filtered = nd_array

    filtered_words = [row[0] for row in filtered]
    detailed_output = []
    if analysis_mode:
        detailed_output = [f"({row[0]},{row[1]})" for row in filtered]
    return filtered_words, detailed_output


def calculate_event_novelty(list_event_words, df_word_weights, sort_words=True):
    """
    Calculate novelty of event

    parameters
    -----------
    :param list_event_words: list
        List of list of words (event words)
    :param df_word_weights: dataframe
        Dataframe which has columns: 'word' and 'weight'.
    :param sort_words: boolean, optional
        If true, sort words in the event by weight
    :return: list of Event objects
    """
    dict_word_weights = dict()
    for index, row in df_word_weights.iterrows():
        dict_word_weights[row['word']] = row['weight']

    events = []
    for words in list_event_words:
        weights = [dict_word_weights[word] for word in words]

        if sort_words:
            nd_array = np.column_stack((words, weights))
            # sort by weight
            nd_array = sorted(nd_array, key=lambda x: x[1], reverse=True)
            words = [row[0] for row in nd_array]

        novelty = np.mean(weights)
        events.append(Event(words, novelty))

    return events


def group_event_words(dict_event_windows, result_folder, embedding_folder, args, analysis_mode=False):
    """
    Generate event word groups

    parameters
    -----------
    :param dict_event_windows: dictionary
        Dictionary of list of EventWindows
        key - frequency_threshold-change_threshold
    :param result_folder: str
        Folder path to save events
    :param embedding_folder: str
        Folder path which contains embedding models
    :param args: JSON
        JSON of arguments
    :param analysis_mode: boolean, optional
        If true,
        Word weights will be saved to files at path- result_folder/key{-weights}/time_window.csv,
        Event objects including novelty will be saved to files at path- result_folder/key/time_window/emerging-objs.txt,
        and Detailed outputs which contain word similarities will be saved to files at path-
        result_folder/key/time_window/emerging-analysis.txt.
    :return:
    """
    for key in dict_event_windows:
        logger.info(f'Processing {key}')
        for event_window in dict_event_windows[key]:
            time_window = event_window.time_window
            logger.info(f'Processing words in {time_window}')

            # load model
            model = load_model(os.path.join(embedding_folder, time_window + '.model'), args['model_type'])

            file_path = None
            if analysis_mode:
                file_path = os.path.join(result_folder, key + "-weights", time_window + '.csv')
            words, df_word_weights = sort_words(event_window, args, result_file=file_path)

            # only consider words with positive temporal change
            df_positive_word_weights = df_word_weights[df_word_weights['weight'] > 0]
            words = list(df_positive_word_weights['word'])

            # Ignore filtering if n is found as a boolean
            if args['n'] and not isinstance(args['n'], bool):
                keywords = words[:args['n']]
            else:
                keywords = words

            # remove emojis and non-alphanumeric characters in keywords
            keywords = remove_emoji_and_non_alphanumeric(keywords)

            events = []
            detailed_outputs = []  # for analysis purpose
            grouped_words = []
            for word in keywords:
                if word not in grouped_words:
                    if args['word_grouping'] == 'we':
                        event_words, detailed_output = get_similar_words(model, [word], count_threshold=args['m'],
                                                                         sim_threshold=args['s'],
                                                                         vocab=event_window.vocab,
                                                                         analysis_mode=analysis_mode)
                    elif args['word_grouping'] == 'matrix':
                        event_words, detailed_output = get_similar_words_by_matrix(word, event_window.sim_matrix_t2,
                                                                                   event_window.vocab,
                                                                                   count_threshold=args['m'],
                                                                                   sim_threshold=args['s'],
                                                                                   analysis_mode=analysis_mode)
                    else:
                        raise KeyError("Unknown word grouping method found!")

                    events.append(event_words)
                    grouped_words.extend(event_words)

                    if analysis_mode:
                        detailed_outputs.append(detailed_output)

            # calculate event novelties
            event_objs = calculate_event_novelty(events, df_word_weights, sort_words=False)
            event_objs.sort(key=lambda x: x.novelty, reverse=True)

            if args['novelty'] and not isinstance(args['novelty'], bool):
                filtered_event_objs = []
                for event_obj in event_objs:
                    if event_obj.novelty >= args['novelty']:
                        filtered_event_objs.append(event_obj)
                event_objs = filtered_event_objs
                events = [event_obj.words for event_obj in event_objs]

            event_folder_path = os.path.join(result_folder, key, time_window)
            create_folder_if_not_exist(event_folder_path)
            if len(events) > 0:
                save_events(events, os.path.join(event_folder_path, 'emerging.txt'))

                if analysis_mode:
                    save_event_objs(event_objs, os.path.join(event_folder_path, 'emerging-objs.txt'))

                    analysis_path = os.path.join(event_folder_path, 'emerging-analysis.txt')
                    save_events(detailed_outputs, analysis_path, analysis_mode=analysis_mode)
