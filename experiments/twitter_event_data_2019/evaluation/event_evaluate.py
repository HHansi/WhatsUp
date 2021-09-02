# Created by Hansi at 7/6/2021
import os

from algo.utils.file_utils import write_list_to_text_file
from experiments.twitter_event_data_2019.evaluation.keyword_evaluate import eval_keywords


def get_eval_measures(dict_clusters, dict_groundtruth, exact_match=True, cluster_word_limit=None,
                      coverage_percentage=None, coverage_n=None, one_cluster_m_events=False,
                      eval_result_folder_path=None):
    """
    Compute time window, event and keyword-based evaluation measures.

    parameters
    -----------
    :param dict_clusters: object
        Dictionary of clusters {time window:[[cluster key]:[cluster words]]}
    :param dict_groundtruth: object
        Dictionary of GT data (time window: GT data)
    :param exact_match: boolean, optional
        Boolean to indicate match type.
        True - exact match
        False - match if edit distance < 2
    :param cluster_word_limit: None or int, optional
        If given, only consider top n words in the cluster (n=cluster_word_limit).
    :param coverage_percentage: None or float, optional
        Percentage of GT labels need to be covered by a cluster to consider it as a match.
    :param coverage_n: None or int, optional
        Number of GT labels need to be contained in a cluster to consider it as a match.
        Given a coverage_percentage, coverage_n will be ignored.
        If no coverage_percentage or coverage_n is given, 100% coverage will be considered for a match.
    :param one_cluster_m_events: boolean, optional
        Indicate the allowance to match one cluster with many events
        True - one cluster can be matched with many events
    :param eval_result_folder_path: str, optional
        Folder path to save matched event keywords
    :return: dictionary, dictionary
        dictionary of evaluation measures {eval_measure_name:value}
        dictionary of summary for each time window {time window:[event_n, event_tp, cluster_n, cluster_tp, keyword_n,
        keyword_tp]}
    """
    timeframe_tp = 0
    timeframe_fp = 0
    timeframe_n = len(dict_groundtruth.keys())  # actual time frame count (TP + FN)

    event_tp = 0  # #events detected by clusters
    event_n = sum([len(dict_groundtruth[time_frame]) for time_frame in dict_groundtruth.keys()])

    cluster_tp = 0  # #clusters matched to GT events
    cluster_n = 0  # total clusters

    keyword_tp = 0
    keyword_n = 0  # actual keyword(synonym set) count (TP + FN)

    dict_time_frame_measures = dict()
    for time_frame in dict_clusters.keys():
        clusters = dict_clusters[time_frame]

        temp_timeframe_tp = 0
        temp_timeframe_fp = 0

        temp_event_tp = 0
        temp_event_n = 0

        temp_cluster_tp = 0
        temp_cluster_n = len(clusters)

        temp_keyword_tp = 0
        temp_keyword_n = 0

        if time_frame not in dict_groundtruth:
            temp_timeframe_fp = 1
        else:
            temp_event_n = len(dict_groundtruth[time_frame])
            dict_gt_events = dict()  # {event_k: GT event}
            for event_index in range(temp_event_n):
                dict_gt_events[event_index] = dict_groundtruth[time_frame][event_index]

            temp_keyword_n, temp_keyword_tp, dict_matched_clusters, dict_event_cluster = eval_keywords(clusters,
                                                                                                       dict_gt_events,
                                                                                                       exact_match,
                                                                                                       cluster_word_limit,
                                                                                                       coverage_percentage=coverage_percentage,
                                                                                                       coverage_n=coverage_n,
                                                                                                       one_cluster_m_events=one_cluster_m_events)

            if eval_result_folder_path:
                # Only the matched words of best matched cluster (which used for keyword measures) will be saved.
                event_outputs = []
                for event_k, cluster_k in dict_event_cluster.items():
                    event_outputs.append(dict_matched_clusters[event_k][cluster_k][1].matched_words)
                write_list_to_text_file(event_outputs, os.path.join(eval_result_folder_path, time_frame + '.txt'))

            # calculate cluster TP - #clusters matched to GT events
            matched_cluster_keys = set()
            for k, v in dict_matched_clusters.items():
                matched_cluster_keys.update(v.keys())
            temp_cluster_tp = len(matched_cluster_keys)

            # calculate event TP - #events detected by clusters
            temp_event_tp = len(dict_event_cluster.keys())

            # if all events belong to the time frame are identified, count time frame as a TP
            if temp_event_tp == temp_event_n:
                temp_timeframe_tp = 1
            else:
                temp_timeframe_fp = 1

        dict_time_frame_measures[time_frame] = [temp_event_n, temp_event_tp, temp_cluster_n, temp_cluster_tp,
                                                temp_keyword_n, temp_keyword_tp]
        timeframe_tp += temp_timeframe_tp
        timeframe_fp += temp_timeframe_fp

        event_tp += temp_event_tp

        cluster_tp += temp_cluster_tp
        cluster_n += temp_cluster_n

        keyword_tp += temp_keyword_tp
        keyword_n += temp_keyword_n

    eval_measures = dict()
    eval_measures['timeframe_n'] = timeframe_n
    eval_measures['timeframe_tp'] = timeframe_tp
    eval_measures['timeframe_fp'] = timeframe_fp
    eval_measures['event_n'] = event_n
    eval_measures['event_tp'] = event_tp
    eval_measures['cluster_n'] = cluster_n
    eval_measures['cluster_tp'] = cluster_tp
    eval_measures['keyword_n'] = keyword_n
    eval_measures['keyword_tp'] = keyword_tp

    return eval_measures, dict_time_frame_measures
