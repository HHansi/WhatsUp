# Created by Hansi at 3/16/2020
import csv
import re
import emoji

from nltk import TweetTokenizer
from nltk.corpus import stopwords

from algo.utils.file_utils import create_folder_if_not_exist

en_stopwords = stopwords.words('english')

# Clean the punctuation marks
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', '..', '...', '…']


def update_stopwords(words, remove=True):
    if remove:
        for word in words:
            en_stopwords.remove(word)
    else:
        en_stopwords.extend(words)


def remove_punctuations(token_list):
    """
    Method to remove punctuation marks in the given token list

    parameters
    -----------
    :param token_list: list of str
        List of tokens
    :return: list
        Filtered list of tokens without punctuation
    """
    filtered_list = []
    for token in token_list:
        if token not in puncts:
            filtered_list.append(token)
    return filtered_list


def remove_stopwords(word_list):
    """
     Method to remove stopwords in the given token list

    parameters
    -----------
    :param word_list: list of str
        List of tokens
    :return: list
        Filtered list of tokens without stopwords
    """
    filtered_list = []
    for word in word_list:
        if word not in en_stopwords:
            filtered_list.append(word)
    return filtered_list


def remove_retweet_notations(sentence):
    """
    Method to remove retweet notations in the given text

    parameters
    -----------
    :param sentence: str
    :return: str
        String without retweet notations
    """
    updated_sentence = re.sub(r'RT @[a-zA-Z0-9_/-]*:', '', sentence)
    return updated_sentence.strip()


def tokenize_text(tokenizer, text, return_sentence=False):
    """
    Method to tokenise text using given tokenizer

    parameters
    -----------
    :param tokenizer: object
        NLTK tokenizer
    :param text: str
        String which need to be tokenised
    :param return_sentence: boolean, optional
        Boolean to indicate the output type.
        True - return the tokenised text as a sentence/string. Tokens are appended using spaces.
        False - returns tokens as a list
    :return: str or list
        Tokenised text
        Return type depends on the 'return_sentence' argument. Default is a list.
    """
    tokens = tokenizer.tokenize(text)
    if return_sentence:
        return " ".join(tokens)
    else:
        return tokens


def remove_links(sentence, substitute=''):
    """
    Method to remove links in the given text

    parameters
    -----------
    :param sentence: str
    :param substitute: str
        which to replace link
    :return: str
        String without links
    """
    sentence = re.sub(r'https?:\/\/\S+', substitute, sentence, flags=re.MULTILINE)
    return sentence.strip()


def remove_symbol(text, symbol):
    """
    Method to remove given symbol in the text. All the symbol occurrences will be replaced by "".

    parameters
    -----------
    :param text: str
    :param symbol: str
        Symbol which need to be removed (e.g., '#')
    :return: str
        Symbol removed text
    """
    return text.replace(symbol, "")


def preprocessing_flow(text):
    """
    Preprocessing flow defined to process text.
    1. Remove retweet notations (e.g., RT @abc:)
    2. Tokenize using TweetTokenizer without preserving case and with length reduction
    3. Remove links
    4. Remove hash symbol

    parameters
    -----------
    :param text: str
    :return: str
        preprocessed text
    """
    text = remove_retweet_notations(text)
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    text = tokenize_text(tknzr, text, return_sentence=True)
    text = remove_links(text)
    text = remove_symbol(text, '#')

    # remove white spaces at the beginning and end of the text
    text = text.strip()
    # remove extra whitespace
    text = ' '.join(text.split())
    return text


def preprocess_bulk(input_file_path, output_file_path):
    """
    Preprocess data in input_file and save to the output_file

    parameters
    -----------
    :param input_file_path: str (.tsv file path)
        Path to input data file
        There should be at least 3 columns in the file corresponding to id, timestamp and text with the column names. If
        no column names are provided, following order is considered. 0-id, 1-timestamp, 2-text
    :param output_file_path: str (.tsv file path)
        Path to output/preprocessed data file
        Output file will be formatted as three-column ([id, timestamp, text-content]) file without column names.
    :return:
    """
    # create folder if not exists
    create_folder_if_not_exist(output_file_path, is_file_path=True)

    input_file = open(input_file_path, encoding='utf-8')
    input_reader = csv.reader(input_file, delimiter='\t')

    output_file = open(output_file_path, 'w', newline='', encoding='utf-8')
    output_writer = csv.writer(output_file, delimiter='\t')

    header = next(input_reader)
    try:
        id_column_index = header.index('id')
        date_column_index = header.index('timestamp')
        text_column_index = header.index('text')
    except ValueError:  # If headers are not provided, consider the following order
        id_column_index = 0
        date_column_index = 1
        text_column_index = 2

    for row in input_reader:
        text = row[text_column_index]
        if text != '_na_':
            processed_text = preprocessing_flow(text)
            output_writer.writerow([row[id_column_index], row[date_column_index], processed_text])


def preprocess_vocabulary(words, preprocess):
    for step in preprocess:
        if 'rm-punct' == step:
            words = remove_punctuations(words)
        if 'rm-stop_words' == step:
            words = remove_stopwords(words)
    return words


def remove_emojis(word_list):
    """
    Remove emojis in the given word list

    parameters
    -----------
    :param word_list: list of str
    :return: list of str
    """
    filtered_list = [word for word in word_list if word not in emoji.UNICODE_EMOJI]
    return filtered_list


def remove_non_alphanumeric(word_list):
    """
    Remove non-alphanumeric tokens (e.g. :, @) in the given word list

    parameters
    -----------
    :param word_list: list of str
    :return: list of str
    """
    filtered_list = [word for word in word_list if not re.match('^\W+$', word)]
    return filtered_list


def remove_emoji_and_non_alphanumeric(word_list):
    """
    Remove both emojis and non-alphanumeric tokens in the given word list

    parameters
    -----------
    :param word_list: list of str
    :return: list of str
    """
    filtered_list = [word for word in word_list if word not in emoji.UNICODE_EMOJI and not re.match('^\W+$', word)]
    return filtered_list
