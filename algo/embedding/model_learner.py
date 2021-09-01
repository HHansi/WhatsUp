# Created by Hansi at 3/29/2021
import logging
import os
import time

import gensim

from algo.utils.file_utils import create_folder_if_not_exist, delete_create_folder, read_text_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)

RANDOM_SEED = 157


def set_seed(seed):
    global RANDOM_SEED
    RANDOM_SEED = seed


def format_data(data):
    """
    Convert data into list of list of tokens.

    parameters
    -----------
    :param data: list of str
    :return: list of list of tokens
    """
    formated_data = []
    for i in data:
        temp = []
        for j in i.split():
            temp.append(j)
        formated_data.append(temp)
    return formated_data


def build_word2vec(data, model_path, args):
    """
    Train and save a Word2Vec model

    parameters
    -----------
    :param data: list of str
    :param model_path: str
        Path to save trained model without extension(.model)
    :param args: JSON
        JSON of arguments required for the model
    :return: object
        Word2Vec model
    """
    formated_data = format_data(data)
    model = gensim.models.Word2Vec(formated_data, min_count=args['min_word_count'], size=args['vector_size'],
                                   window=args['context_size'], sg=args['sg'],
                                   seed=RANDOM_SEED, workers=args['we_workers'])

    # create folder if not exist
    create_folder_if_not_exist(model_path, is_file_path=True)

    # save model
    model_path = model_path + ".model"
    model.save(model_path)
    return model


def build_fasttext(data, model_path, args):
    """
    Train and save fastText model

    parameters
    -----------
    :param data: list of str
    :param model_path: str
        Path to save trained model without extension(.model)
    :param args: JSON
        JSON of arguments required for the model
    :return: object
        fastText model
    """
    formated_data = format_data(data)
    model = gensim.models.FastText(formated_data, min_count=args['min_word_count'], size=args['vector_size'],
                                       window=args['context_size'], sg=args['sg'],
                                       word_ngrams=args['word_ngrams'], min_n=args['min_n'], max_n=args['max_n'],
                                       seed=RANDOM_SEED, workers=args['we_workers'])

    # create folder if not exist
    create_folder_if_not_exist(model_path, is_file_path=True)

    # save model
    model_path = model_path + ".model"
    model.save(model_path)
    return model


def learn_embeddings_bulk(data_folder_path, model_folder_path, args):
    """
    Method to train and save embedding models correspond to the all files in data_folder_path

    parameters
    -----------
    :param data_folder_path: str
        Path to folder which contain .tsv files per window. format - [id, timestamp, text]
    :param model_folder_path: str
        Folder path to save embedding models
    :param args: JSON
        JSON of arguments
    :return:
    """
    delete_create_folder(model_folder_path)
    for root, dirs, files in os.walk(data_folder_path):
        for file in files:
            file_path = os.path.join(data_folder_path, file)
            file_name = os.path.splitext(file)[0]
            model_path = os.path.join(model_folder_path, file_name)
            logger.info(f'learning word embeddings- {file_name}')
            start_time = time.time()
            data = read_text_column(file_path)

            if args['model_type'] == 'w2v':
                build_word2vec(data, model_path, args)
            elif args['model_type'] == 'ft':
                build_fasttext(data, model_path, args)
            else:
                raise KeyError("Unknown model type is given to learn embedding!")
            end_time = time.time()
            # logger.info(f'Completed learning in {int(end_time - start_time)} seconds')
