# Created by Hansi at 3/29/2021
import os

RANDOM_SEED = 157

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER_PATH = os.path.join(BASE_PATH, 'data')
EMBEDDING_FOLDER_PATH = os.path.join(BASE_PATH, 'embedding')
STAT_FOLDER_PATH = os.path.join(BASE_PATH, 'stat')
RESULTS_FOLDER_PATH = os.path.join(BASE_PATH, 'results')
GT_FOLDER_PATH = os.path.join(BASE_PATH, 'ground_truth')

TEMP_FOLDER = os.path.join(BASE_PATH, 'temp')

we_args = {
    # word embedding configs
    'model_type': 'w2v',  # supported types: w2v, ft
    'sg': 1,  # 1-skipgram, 0-cbow
    'min_word_count': 1,
    'vector_size': 100,
    'context_size': 5,
    'we_workers': 1,  # For reproducible results it is recommended to use single worker for word embedding generation.
    # additional parameters for fastText
    'min_n': 3,
    'max_n': 6,
    'word_ngrams': 1
}

args = {
    # Data process
    'model_type': 'w2v',
    'preprocess': ['rm-punct', 'rm-stop_words'],

    # Event window identification
    'affinity': 'cosine',
    'linkage': 'average',
    'similarity_type': 'ldl',  # 'dl', 'ldl'
    'relative': False,
    'aggregation_method': 'avg',  # None, 'avg', 'max'
    'workers': 1,
    'use_freq_diff': True,  # True-consider tokens for vocabulary change calculation, if freq. diff >= freq. threshold
    'similarity_change': 'pos',  # 'normal', 'abs', 'pos'

    # Event word extraction
    'word_grouping': 'matrix',  # 'we', 'matrix'
    'token_weight': 'avg_asd_fd',  # psd, asd, fd, avg_psd_fd, avg_asd_fd, max_psd_fd, max_asd_fd
    'token_similarity_change': 'pos',  # 'normal', 'abs', 'pos' (default = 'normal')
                            # 'pos' is only applicable for asd-based weights. Otherwise it will be treated as 'normal'.
    'analysis_mode': False,

    # Args to set by end-user
    # Data process
    'from_time': '2019_10_20_15_28_00',
    'to_time': '2019_10_20_17_23_59',
    'time_window_length': 2,  # in minutes

    # Event window identification
    'change_threshold': [0.13],  # alpha
    'frequency_threshold': [20],  # beta

    # Event word extraction (Either one of m or s should be provided.)
    'm': 25,  # int-number of keywords per event or count threshold
    's': None,  # float-similarity threshold

    # Cluster pruning (Either one of n or s should be provided.)
    'n': 15,  # int-number of top keywords to consider during event detection
    'novelty': None,  # float-novelty threshold
}

