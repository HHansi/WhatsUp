# Created by Hansi at 3/16/2020
import csv
import os
import shutil

import pandas as pd


def create_folder_if_not_exist(path, is_file_path=False):
    """
    Method to create folder if it does not exist

    parameters
    -----------
    :param path: str
        Path to folder or file
    :param is_file_path: boolean, optional
        Boolean to indicate whether given path is a file path or a folder path
    :return:
    """
    if is_file_path:
        folder_path = os.path.dirname(os.path.abspath(path))
    else:
        folder_path = path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def delete_create_folder(path):
    """
    Method to create a folder. If the folder already exists, it will be deleted before the creation.

    parameters
    -----------
    :param path: str
        Path to folder
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_sorted_filenames(folder_path, ext=None):
    """
    Sort names of the files within the given folder

    parameters
    -----------
    :param folder_path: str
        Path to folder
    :param ext: str, optional
        Extension of the files which should sort (e.g. '.model')
    :return: list of str
        Sorted file names (in ascending order)
    """
    names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # if an ext is given only consider the files with that extension
            if ext and os.path.splitext(file)[1] == ext:
                file_name = os.path.splitext(file)[0]
                names.append(file_name)
    names.sort()
    return names


def write_list_to_text_file(list, file_path):
    """
    Method to write list of str to a .txt file.

    parameters
    -----------
    :param list: list
    :param file_path: str (.txt file path)
    :return:
    """
    create_folder_if_not_exist(file_path, is_file_path=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in list:
            f.write("%s\n" % item)


def read_list_from_text_file(file_path):
    """
    Read string lines in .txt file into a list

    parameters
    -----------
    :param file_path: str (.txt file path)
    :return: list
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = f.read().splitlines()
    return data_list


def read_text_column(file_path):
    """
    Method to extract text column of a .tsv file with columns formatted as ['id', 'timestamp', 'text'] without column
    names.

    parameters
    -----------
    :param file_path: str (.tsv file path)
    :return: dataframe with text column
    """
    data = pd.read_csv(file_path, sep='\t', engine='python', encoding='utf-8',
                       names=['id', 'timestamp', 'text'])
    data = data[data['text'] != '_na_']
    data = data['text']
    return data


def save_row(result, result_file_path):
    """
    Method to append a row to a .tsv file.

    parameters
    -----------
    :param result: list
        Row need to append
    :param result_file_path: str (.tsv file path)
    :return:
    """
    result_file = open(result_file_path, 'a', newline='', encoding='utf-8')
    result_writer = csv.writer(result_file, delimiter='\t')
    result_writer.writerow(result)
    result_file.close()
