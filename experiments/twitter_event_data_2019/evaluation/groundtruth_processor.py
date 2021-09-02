# Created by Hansi at 3/16/2020

import os

from algo.data_process.data_preprocessor import data_cleaning_flow
from algo.utils.file_utils import delete_create_folder


def extract_gt_tokens(text):
    """
    Given GT string, method to extract GT labels.
    GT string should be formatted as Twitter-Event-Data-2019.

    parameters
    -----------
    :param text: str
    :return: list
        List of GT labels corresponding to a single event
        Since there can be duplicate definitions for a single event, this list contains separate label lists for each
        duplicate definition.
    """
    duplicates = []

    for element in text.split("|"):
        labels = []
        for subelement in element.split("["):
            if subelement:
                subelement = subelement.replace("\n", "")
                subelement = subelement.replace("]", "")
                tokens = subelement.split(",")
                labels.append(tokens)
        duplicates.append(labels)
    return duplicates


def load_gt(folder_path):
    """
    Method to read GT data into a dictionary formatted as (time-window: labels)

    parameters
    -----------
    :param folder_path: str
        Path to folder which contains GT data
    :return: object
        Dictionary of GT data
    """
    gt = dict()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name = os.path.splitext(file)[0]
            f = open(os.path.join(folder_path, file), 'r', encoding='utf-8')
            events = []
            for line in f:
                tokens = extract_gt_tokens(line)
                events.append(tokens)
            gt[file_name] = events
            f.close()
    return gt


def generate_gt_string(tokens):
    """
    Given a list of GT labels corresponding to a single event, convert them to a string formatted according to
    Twitter-Event-Data-2019 GT format.

    parameters
    -----------
    :param tokens: list
    :return: str
    """
    str = ""
    for duplicate in tokens:
        if str and str[-1] == "]":
            str = str + "|"
        for label in duplicate:
            str = str + "["
            for element in label:
                if str[-1] == "[":
                    str = str + element
                else:
                    str = str + "," + element
            str = str + "]"
    return str


def get_combined_gt(gt):
    """
    Combine the GT labels of multiple events available at a time frame into single event representation.

    parameters
    -----------
    :param gt: object
        Dictionary of GT returned by load_GT
    :return: object
        Dictionary of combined GT
    """
    combined_gt = dict()
    for time_frame in gt.keys():
        gt_events = gt[time_frame]
        combined_gt_event = gt_events[0]

        for event in gt_events[1:]:
            temp = []
            for duplicate in event:
                for combined_event in combined_gt_event:
                    temp.append(combined_event + duplicate)
            combined_gt_event = temp

        # even though there is 1 event, it is added to a list to preserve consistency with general evaluation_v2 methods
        events = [combined_gt_event]
        combined_gt[time_frame] = events
    return combined_gt


def preprocess_gt(input_filepath, output_filepath):
    """
    Preprocess ground truth data in input_file and save to the output_file

    parameters
    -----------
    :param input_filepath: str (.txt file path)
        Ground truth file formatted as Twitter-Event-Data-2019
    :param output_filepath: str (.txt file path)
    :return:
    """
    input_file = open(input_filepath, 'r')
    output_file = open(output_filepath, 'a', encoding='utf-8')

    events = []
    for line in input_file:
        tokens = extract_gt_tokens(line)
        events.append(tokens)

    # update tokens
    new_events = []
    for event in events:
        new_duplicates = []
        for duplicate in event:
            new_labels = []
            for label in duplicate:
                new_elements = []
                for element in label:
                    new_label = data_cleaning_flow(element)
                    new_elements.append(new_label)
                new_labels.append(new_elements)
            new_duplicates.append(new_labels)
        new_events.append(new_duplicates)

    for event in new_events:
        str = generate_gt_string(event)
        output_file.write(str)
        output_file.write("\n")
    output_file.close()


def preprocess_gt_bulk(input_folder_path, output_folder_path):
    """
    Preprocess ground truth data in all files in input_folder and save to the output_folder

    parameters
    -----------
    :param input_folder_path: str
        Path to folder which contains GT data files
    :param output_folder_path: str
        Path to folder to save preprocessed GT data
    :return:
    """
    # delete if there already exist a folder and create new folder
    delete_create_folder(output_folder_path)

    for root, dirs, files in os.walk(input_folder_path):
        for file in files:
            input_filepath = os.path.join(input_folder_path, file)
            output_filepath = os.path.join(output_folder_path, file)
            preprocess_gt(input_filepath, output_filepath)