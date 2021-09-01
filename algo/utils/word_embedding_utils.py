# Created by Hansi at 6/28/2021

import numpy as np
from gensim.models import Word2Vec, FastText

from algo.data_process.data_preprocessor import remove_punctuations, remove_stopwords


def load_model(model_path, type):
    """
    Method to load word embedding model.

    parameters
    -----------
    :param model_path: str
        Path to model
    :param type: {'w2v', 'ft'}
        Type of the saved model
        w2v - word2vec
        ft - fastText
    :return: object
        Loaded model
    """
    if type == 'w2v':
        return Word2Vec.load(model_path)
    elif type == 'ft':
        return FastText.load(model_path)
    else:
        raise KeyError("Not supported model type is given")


def get_embedding(word, model):
    """
    Method to get embedding of given word

    parameters
    -----------
    :param word: str
    :param model: object
        Word embedding model
    :return: vector
    """
    if word in model.wv.vocab:
        return model.wv[word]
    else:
        raise KeyError


def get_similarity(word1, word2, model):
    try:
        return model.wv.similarity(word1, word2)
    except KeyError:
        return 0


def get_vocab(model):
    """
    Method to get vocabulary of a word embedding model

    parameters
    -----------
    :param model: object
        Word embedding model
    :return: list
        Vocabulary as a list of words
    """
    return list(model.wv.vocab)


def get_similar_words(model, positive, count_threshold=None, sim_threshold=None, preprocess=None, vocab=None,
                      analysis_mode=False):
    """
    Filter similar words according to given count or similarity threshold.
    Priority is given to count_threshold, if both the thresholds are provided. If none of the thresholds are provided
    reversely sorted similar words of whole vocabulary will be returned.

    parameters
    -----------
    :param model: object
        word embedding model
    :param positive: list of tokens
    :param count_threshold: int, optional
    :param sim_threshold: float, optional
    :param preprocess: list of string, optional
        Each string represents the preprocessing step
    :param vocab: list of tokens, optional
    :param analysis_mode: boolean, optional
        If true, a detailed_output: a list of [word, similarity to positive] will be returned.
        Otherwise an empty list will be returned as detailed_output
    :return: list, list
        list of similar words matched with the given criteria
        detailed_output
    """
    # topn is set to a very high value to get all similar words
    similar_words = model.wv.most_similar(positive=positive, topn=1000000)
    dict_word_similarity = dict((row[0], row[1]) for row in similar_words)
    words = np.array(similar_words)[:, 0]
    if vocab:
        # words = list(set(words).intersection(vocab))
        intersection = [x for x in words if x in vocab]  # get intersection while keeping the order
        words = intersection
    if preprocess:
        for step in preprocess:
            if 'rm-punct' == step:
                words = remove_punctuations(words)
            if 'rm-stop_words' == step:
                words = remove_stopwords(words)

    if count_threshold is not None:
        filtered = positive + words[:count_threshold - len(positive)]
    elif sim_threshold is not None:
        filtered = [word for word in words if dict_word_similarity[word] >= sim_threshold]
        filtered = positive + filtered
    else:
        filtered = positive + words

    detailed_output = []
    if analysis_mode:
        for word in filtered:
            if word in dict_word_similarity.keys():
                detailed_output.append(f"({word},{dict_word_similarity[word]})")
            else:
                detailed_output.append(f"({word},na)")
        # detailed_output = [f"({word},{similarity_dict[word]})" for word in words]

    return filtered, detailed_output
