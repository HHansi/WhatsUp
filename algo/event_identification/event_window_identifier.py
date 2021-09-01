# Created by Hansi at 3/16/2020
import logging
import os

import numpy as np
import pandas as pd

from algo.change_measures.dendrogram_level_calculation import generate_dendrogram_level_codes
from algo.change_measures.matrix_calculation import get_upper_triangular_as_list, \
    get_upper_triangular_matrix, get_ut_matrix_values, get_proportion
from algo.change_measures.vocabulary_calculation import load_wordcounts, filter_vocabulary_by_frequency, get_word_diff, \
    filter_vocabulary_by_frequency_diff
from algo.change_measures.word_similarity_calculation import get_vectors_for_words, \
    get_dendrogram_level_similarity_matrix_for_words, \
    get_local_dendrogram_level_similarity_matrix_for_words, get_cosine_similarity_matrix_for_words
from algo.data_process.data_preprocessor import preprocess_vocabulary
from algo.utils.file_utils import save_row, get_sorted_filenames, create_folder_if_not_exist
from algo.utils.word_embedding_utils import load_model, get_vocab

logger = logging.getLogger(__name__)


class EventWindow:
    def __init__(self, time_window, overall_change, diff_matrix, vocab, word_counts_t1, word_counts_t2, sim_matrix_t2):
        self.time_window = time_window
        self.overall_change = overall_change
        self.diff_matrix = diff_matrix
        self.vocab = vocab
        self.word_counts_t1 = word_counts_t1
        self.word_counts_t2 = word_counts_t2
        self.sim_matrix_t2 = sim_matrix_t2


def save_word_pairs(word_pair_values, file_path):
    """
    save word pairs and their similarity to .csv file
    :param word_pair_values: list of [similarity, [word1, word2]]
    :param file_path: .csv file path
    :return:
    """
    i = 0
    df_t = pd.DataFrame(columns=['similarity', 'word1', 'word2'])
    for word_pair in word_pair_values:
        df_t.loc[i] = [word_pair[0], word_pair[1][0], word_pair[1][1]]
        i += 1
    df_t.to_csv(file_path, index=False)


def get_similarity_matrices(model1, model2, vocab1, vocab2, common_vocab, args, result_folder=None):
    """
    Generate similarity matrices for T and T+1

    parameters
    -----------
    :param model1: object
        Word embedding model at T
    :param model2: object
        Word embedding model at T+1
    :param vocab1: list
        Vocabulary at T
    :param vocab2: list
        Vocabulary at T+1
    :param common_vocab: list
        Vocabulary to use with matrices
    :param args: JSON
        JSON of arguments
    :param result_folder: str, optional
        Folder path to save matrices (for analysis purpose)
    :return: matrix, matrix
        Similarity matrix for T
        Similarity matrix for T+1
    """
    if args['relative']:
        word_list1, vector_list1 = get_vectors_for_words(vocab2, model1)
    else:  # use preprocessed vocab1 for t1 and vocab2 for t2
        word_list1, vector_list1 = get_vectors_for_words(vocab1, model1)

    word_list2, vector_list2 = get_vectors_for_words(vocab2, model2)
    label_codes1, max_level_count1 = generate_dendrogram_level_codes(vector_list1, word_list1, args['affinity'],
                                                                     args['linkage'])
    label_codes2, max_level_count2 = generate_dendrogram_level_codes(vector_list2, word_list2, args['affinity'],
                                                                     args['linkage'])

    if args['similarity_type'] == 'dl':
        sim_matrix1 = get_dendrogram_level_similarity_matrix_for_words(label_codes1, common_vocab, max_level_count1,
                                                                       workers=args['workers'])
        sim_matrix2 = get_dendrogram_level_similarity_matrix_for_words(label_codes2, common_vocab, max_level_count2,
                                                                       workers=args['workers'])
    elif args['similarity_type'] == 'ldl':
        sim_matrix1 = get_local_dendrogram_level_similarity_matrix_for_words(label_codes1, common_vocab,
                                                                             args['workers'])
        sim_matrix2 = get_local_dendrogram_level_similarity_matrix_for_words(label_codes2, common_vocab,
                                                                             args['workers'])
    else:
        raise KeyError('Unknown similarity type!')

    sim_matrix1 = np.array(sim_matrix1)
    sim_matrix2 = np.array(sim_matrix2)

    if result_folder:
        create_folder_if_not_exist(result_folder)

        # save word pair values to a .csv files
        sim_ut_matrix1 = get_upper_triangular_matrix(sim_matrix1, 1)  # ut matrix without diagonal
        sim_ut_matrix2 = get_upper_triangular_matrix(sim_matrix2, 1)
        word_pair_values1 = get_ut_matrix_values(common_vocab, sim_ut_matrix1)
        word_pair_values2 = get_ut_matrix_values(common_vocab, sim_ut_matrix2)

        save_word_pairs(word_pair_values1, os.path.join(result_folder, "t1.csv"))
        save_word_pairs(word_pair_values2, os.path.join(result_folder, "t2.csv"))

    return sim_matrix1, sim_matrix2


def get_event_windows(embedding_folder_path: str, stat_folder_path: str, args, result_folder: str = None) -> object:
    """
    Method to identify event occurred time windows

    parameters
    -----------
    :param embedding_folder_path: folder path
        Path to folder which contains word embedding models
    :param stat_folder_path: folder path
        Path to folder which contains stat details of data
    :param args: json object
    :param result_folder: folder path, optional
        Folder path to save time window and overall change details.
    :return: dictionary
        Dictionary of list of EventWindows
        key - frequency_threshold-change_threshold
    """

    time_frames = get_sorted_filenames(embedding_folder_path, ext=".model")
    logger.info(f'length of time frames {len(time_frames)}')

    dict_event_windows = dict()

    for index in range(0, len(time_frames)):
        if index < (len(time_frames) - 1):
            t1 = time_frames[index]
            t2 = time_frames[index + 1]
            info_label = t1 + '-' + t2
            logger.info(f'processing {info_label}')

            # load word embedding models
            model1 = load_model(os.path.join(embedding_folder_path, t1 + '.model'), args['model_type'])
            model2 = load_model(os.path.join(embedding_folder_path, t2 + '.model'), args['model_type'])
            # load stats
            word_counts1 = load_wordcounts(os.path.join(stat_folder_path, t1 + '.tsv'))
            word_counts2 = load_wordcounts(os.path.join(stat_folder_path, t2 + '.tsv'))
            # get vocabularies
            vocab1_all = get_vocab(model1)
            vocab2_all = get_vocab(model2)

            if args['preprocess']:
                vocab1_all = preprocess_vocabulary(vocab1_all, args['preprocess'])
                vocab2_all = preprocess_vocabulary(vocab2_all, args['preprocess'])

            for frequency_threshold in args['frequency_threshold']:
                vocab1 = filter_vocabulary_by_frequency(vocab1_all, word_counts1, frequency_threshold)
                vocab2 = filter_vocabulary_by_frequency(vocab2_all, word_counts2, frequency_threshold)
                vocab1.sort()
                vocab2.sort()
                common_vocab = vocab2

                # get word counts correspond to the common vocabulary words
                filtered_word_counts1 = {key: word_counts1[key] if key in word_counts1 else 0 for key in common_vocab}
                filtered_word_counts2 = {key: word_counts2[key] if key in word_counts2 else 0 for key in common_vocab}

                # get similarity matrices
                sim_matrix1, sim_matrix2 = get_similarity_matrices(model1, model2, vocab1, vocab2, common_vocab, args)
                diff_matrix = sim_matrix2 - sim_matrix1
                # diff_ut_matrix, diff_ut_values = get_upper_triangular_as_list(diff_matrix)
                diff_ut_values = get_upper_triangular_as_list(diff_matrix)

                diff_ut_values = np.array(diff_ut_values)

                if args['similarity_change'] == 'normal':
                    cluster_change = np.mean(diff_ut_values)
                elif args['similarity_change'] == 'abs':
                    diff_ut_values = np.absolute(diff_ut_values)
                    cluster_change = np.mean(diff_ut_values)
                elif args['similarity_change'] == 'pos':
                    pos_proportion, filtered = get_proportion(diff_ut_values, positive=True)
                    if pos_proportion == 0:  # handle not return of nan values as cluster_change
                        cluster_change = 0
                    else:
                        cluster_change = np.mean(filtered) * pos_proportion
                else:
                    raise KeyError('Unknown similarity change method found')

                n_words_diff, words_diff = get_word_diff(vocab1, vocab2)
                if args['use_freq_diff']:
                    filtered, _ = filter_vocabulary_by_frequency_diff(words_diff, word_counts1, word_counts2,
                                                                      frequency_threshold)
                    vocab_change = len(filtered) / len(vocab2)
                else:
                    vocab_change = len(words_diff) / len(vocab2)

                if args['aggregation_method'] is None:
                    overall_change = cluster_change
                else:
                    if 'max' == args['aggregation_method']:
                        overall_change = max(cluster_change, vocab_change)
                    elif 'avg' == args['aggregation_method']:
                        overall_change = (cluster_change + vocab_change) / 2
                    else:
                        raise KeyError('Unknown aggregation method found')

                for change_threshold in args['change_threshold']:
                    key = f"{change_threshold}-{frequency_threshold}"
                    # logger.info(f'processing {key}')
                    if overall_change > change_threshold:
                        if key in dict_event_windows.keys():
                            dict_event_windows[key].append(EventWindow(t2, overall_change, diff_matrix,
                                                                       common_vocab, filtered_word_counts1,
                                                                       filtered_word_counts2, sim_matrix2))
                        else:
                            dict_event_windows[key] = [EventWindow(t2, overall_change, diff_matrix,
                                                                   common_vocab, filtered_word_counts1,
                                                                   filtered_word_counts2, sim_matrix2)]
                    if result_folder:
                        create_folder_if_not_exist(os.path.join(result_folder, key))
                        save_row([t2, cluster_change, vocab_change, overall_change], os.path.join(result_folder, key +
                                                                                                  '-change.tsv'))
    return dict_event_windows
