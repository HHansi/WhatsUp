# Created by Hansi at 7/1/2021
import logging
import os
import time

from algo.data_process.data_preprocessor import clean_bulk
from algo.data_process.stat_generator import generate_stats
from algo.data_process.stream_chunker import filter_documents_by_time_bulk
from algo.embedding.model_learner import learn_embeddings_bulk
from algo.event_identification.event_window_identifier import get_event_windows
from algo.event_identification.event_word_extractor import group_event_words

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


def full_flow(data_file_path: str, args: dict, we_args: dict, output_folder_path: str):
    """
    WhatsUp full flow

    parameters
    -----------
    :param data_file_path: str (.tsv file path)
        There should be at least 3 columns in the file with the column names: 'id', 'timestamp' and 'text'.
        The timestamp values need to be formatted as %Y-%m-%d %H:%M:%S
    :param args: JSON object
        JSON of arguments for event detection
    :param we_args: JSON object
        JSON of arguments to learn word embeddings
    :param output_folder_path: str
        Folder path to save outputs:
        - cleaned data: output_folder_path/cleaned.tsv
        - time window data: output_folder_path/time-windows
        - leaned embedding models: output_folder_path/word-embedding
        - extracted statistical information: output_folder_path/stat
        - detected events: output_folder_path/results
    :return:
    """
    start_time_full_process = time.time()

    # # Data cleaning
    cleaned_data_path = os.path.join(output_folder_path, 'cleaned.tsv')
    clean_bulk(data_file_path, cleaned_data_path)

    # # Data Preprocessing
    # Stream Chunking
    logger.info('Separating data stream into chunks..')
    start_time = time.time()
    data_chunk_folder_path = os.path.join(output_folder_path, 'time-windows')
    filter_documents_by_time_bulk(args['from_time'], args['to_time'], args['time_window_length'], cleaned_data_path,
                                  data_chunk_folder_path)
    end_time = time.time()
    logger.info(f'Completed data stream separation in {int(end_time - start_time)} seconds \n')

    # Word Embedding Learning
    logger.info('Learning word embeddings..')
    start_time = time.time()
    embedding_folder_path = os.path.join(output_folder_path, 'word-embedding')
    learn_embeddings_bulk(data_chunk_folder_path, embedding_folder_path, we_args)
    end_time = time.time()
    logger.info(f'Completed word embedding learning in {int(end_time - start_time)} seconds \n')

    # Statistical Information Extraction
    logger.info('Extracting statistical information..')
    start_time = time.time()
    stat_folder_path = os.path.join(output_folder_path, 'stat')
    generate_stats(data_chunk_folder_path, stat_folder_path)
    end_time = time.time()
    logger.info(f'Completed statistical information extraction in {int(end_time - start_time)} seconds \n')

    # # Event Window Identification
    logger.info('Identifying event windows')
    start_time = time.time()
    results_folder_path = os.path.join(output_folder_path, 'results')
    dict_event_windows = get_event_windows(embedding_folder_path, stat_folder_path, args, results_folder_path)
    end_time = time.time()
    logger.info(f'Completed event window identification in {int(end_time - start_time)} seconds \n')

    # # Event Cluster Detection
    logger.info('Detecting event clusters')
    start_time = time.time()
    group_event_words(dict_event_windows, results_folder_path, embedding_folder_path, args,
                      analysis_mode=args['analysis_mode'])
    end_time = time.time()
    logger.info(f'Completed event cluster detection in {int(end_time - start_time)} seconds \n')

    end_time_full_process = time.time()
    logger.info('Process completed')
    logger.info(f'Full process completed in {int(end_time_full_process - start_time_full_process)} seconds')
