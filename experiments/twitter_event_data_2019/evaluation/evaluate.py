# Created by Hansi at 7/6/2021
import logging
import os

from experiments.twitter_event_data_2019.args import GT_FOLDER_PATH, RESULTS_FOLDER_PATH, TEMP_FOLDER
from experiments.twitter_event_data_2019.evaluation.event_evaluate import get_eval_measures
from experiments.twitter_event_data_2019.evaluation.general_methods import calculate_recall, calculate_precision, \
    calculate_f1
from experiments.twitter_event_data_2019.evaluation.groundtruth_processor import load_gt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_events(file_path):
    """
    Load events saved to a .txt file

    parameters
    -----------
    :param file_path: str
        Path to .txt file
    :return: dictionary
        Dictionary of event words- {incrementing ID: [event words]}
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


def evaluate_results(result_folder_path, groundtruth_folder_path, type='emerging', eval_result_folder_path=None):
    """
    Method to compute evaluation measures by comparing results with ground truth (GT).
    Files in results folder and GT folder need to be named by time windows formatted similarly.

    Keyword Recall(Micro-averaged), Event Recall, Event Precision, Event F1, Timeframe Recall, Timeframe Precision and
    Timeframe F1 will be calculated and logged.

    parameters
    -----------
    :param result_folder_path: str
        Path to folder which contains extracted events
        In this folder there should be separate folders for each time window. Within each time window, there should be
        separate .txt files for each event type (e.g. emerging.txt).
        Event words need to be written per line of corresponding .txt files (Words are separated by ', ').
    :param groundtruth_folder_path: str
        Path to GT data folder.
    :param type: str, optional
        Type of the events need to be evaluated ('emerging')
    :param eval_result_folder_path: str, optional
        Folder path to save matched event words in each time window.
    :return:
    """
    groundtruth = load_gt(os.path.join(groundtruth_folder_path, type))

    dict_event_words = dict()

    for root, dirs, files in os.walk(result_folder_path):
        for dir in dirs:
            event_words = load_events(os.path.join(result_folder_path, dir, type + '.txt'))
            if event_words is not None:
                dict_event_words[dir] = event_words

    eval_measures, dict_time_frame_measures = get_eval_measures(dict_event_words, groundtruth,
                                                                exact_match=True, coverage_n=1,
                                                                one_cluster_m_events=False,
                                                                eval_result_folder_path=eval_result_folder_path)

    logger.info(f"time_window:[event_n,event_tp,cluster_n,cluster_tp,keyword_n,keyword_tp]")
    logger.info(dict_time_frame_measures)
    dict_results = dict()

    # keyword evaluation
    micro_keyword_recall = calculate_recall(eval_measures['keyword_tp'], eval_measures['keyword_n'])
    dict_results['keyword recall'] = micro_keyword_recall
    logger.info(f'Micro Keyword Recall: {micro_keyword_recall}')

    # event evaluation
    # event recall = ratio of the events successfully detected among the ground-truth events
    event_recall = calculate_recall(eval_measures['event_tp'], eval_measures['event_n'])
    # event precision/relevance = ratio of the events(clusters) matched to some groundtruth topic among the events found
    # by a method.
    event_precision = calculate_recall(eval_measures['cluster_tp'], eval_measures['cluster_n'])
    dict_results['event recall'] = event_recall
    dict_results['event relevance'] = event_precision
    logger.info(f'Event Recall: {event_recall}')
    logger.info(f'event relevance: {event_precision}')

    # timeframe evaluation
    timeframe_recall = calculate_recall(eval_measures['timeframe_tp'], eval_measures['timeframe_n'])
    timeframe_precision = calculate_precision(eval_measures['timeframe_tp'], eval_measures['timeframe_fp'])
    timeframe_f1 = calculate_f1(timeframe_recall, timeframe_precision)
    dict_results['timeframe recall'] = timeframe_recall
    dict_results['timeframe precision'] = timeframe_precision
    dict_results['timeframe f1'] = timeframe_f1
    logger.info(f'Timeframe Recall: {timeframe_recall}')
    logger.info(f'Timeframe Precision: {timeframe_precision}')
    logger.info(f'Timeframe F1: {timeframe_f1}')

    return dict_results


if __name__ == '__main__':
    groundtruth_folder_path = os.path.join(GT_FOLDER_PATH, "munliv")
    results_folder = os.path.join(TEMP_FOLDER, "munliv-15.28-17.23-2/results/0.13-20")
    type = 'emerging'
    evaluate_results(results_folder, groundtruth_folder_path, type=type, eval_result_folder_path=None)
