# Created by Hansi at 7/6/2021
import editdistance
import numpy as np
from scipy.optimize import linear_sum_assignment


class MatchedCluster:
    def __init__(self, cluster_words, gt_event_words, matched_words, keyword_n, keyword_tp):
        self.cluster = cluster_words
        self.gt_event = gt_event_words
        self.matched_words = matched_words  # matched event words
        self.keyword_n = keyword_n  # total of GT words used for evaluation_v2
        self.keyword_tp = keyword_tp  # total of true positives


def find_best_match(cluster, groundtruth_event, exact_match=True, cluster_n=None):
    """
    Find best match (event duplicate) for the event cluster wrt the given event

    parameters
    -----------
    :param cluster: list
        List of words belong to a cluster
    :param groundtruth_event: list
        Ground truth tokens correspond to an event
    :param exact_match: boolean, optional
        True- exact match between cluster word and gt label
        False- match if edit distance between cluster word and gt label is less than 2
    :param cluster_n: None or int, optional
        If given, only consider top n words in the cluster.
    :return: float, object
        missed_proportion(#misses/n), MatchedCluster
    """
    # if cluster_n is mentioned, prune cluster to only consider top n words
    if cluster_n and len(cluster) > cluster_n:
        cluster = cluster[:cluster_n]

    matches = []  # array of [missed_proportion, MatchedCluster]

    # sample ground truth_event:
    # [[[1st half,first half,first-half,half time,half-time,halftime,half,ht][1-0]],
    #  [[1st half,first half,first-half,half time,half-time,halftime,half,ht][1][0]]]
    for keywords in groundtruth_event:  # iterate through the keyword combinations of the given event
        matched_labels = []
        keyword_tp = 0
        keyword_n = len(keywords)

        # sample keywords: [[1st half,first half,first-half,half time,half-time,halftime,half,ht][1-0]]
        for synset in keywords:
            matched_elements = []
            # sample syn set: [1st half,first half,first-half,half time,half-time,halftime,half,ht]
            for element in synset:
                splits = element.split()
                if len(splits) > 1:  # if more than one token available, all tokens need to be matched
                    splits_n = len(splits)
                    matched = []
                    for split in splits:
                        for word in cluster:
                            if exact_match:
                                if split == word:
                                    matched.append(word)
                                    break
                            else:
                                if editdistance.eval(split, word) < 2:
                                    matched.append(word)
                                    break
                    if len(matched) == splits_n:
                        matched_elements.append(" ".join(matched))
                else:
                    for word in cluster:
                        if exact_match:
                            if element == word:
                                matched_elements.append(word)
                                break
                        else:
                            if editdistance.eval(element, word) < 2:
                                matched_elements.append(word)
                                break

            # count TPs in event duplicate, if at least 1 token in the syn set is matched.
            if len(matched_elements) > 0:
                keyword_tp += 1
            matched_labels.append(matched_elements)

        match = MatchedCluster(cluster, keywords, matched_labels, keyword_n, keyword_tp)
        missed_proportion = (keyword_n - keyword_tp) / keyword_n  # missed_proportion = #misses/n
        matches.append([missed_proportion, match])

    # sort matches by missed_proportion in ascending order
    matches = sorted(matches, key=lambda x: x[0])

    # best match is the one with minimum missed_proportion
    return matches[0][0], matches[0][1]


def eval_keywords(dict_clusters, dict_gt_events, exact_match=True, cluster_n=None, coverage_percentage=None,
                  coverage_n=None, one_cluster_m_events=False):
    """
    Evaluate word clusters with ground truth

    parameters
    -----------
    :param dict_clusters: dictionary
        Dictionary of clusters {cluster_key: list of cluster words}
    :param dict_gt_events: dictionary
        Dictionary of GT events {event_key: event words}
        Sample event words:
            [[[1st half,first half,first-half,half time,half-time,halftime,half,ht][1-0]],
             [[1st half,first half,first-half,half time,half-time,halftime,half,ht][1][0]]]
    :param exact_match: boolean, optional
        Boolean to indicate match type.
        True - exact match
        False - match if edit distance < 2
    :param cluster_n: None or int, optional
        If given, only consider top n words in the cluster.
    :param coverage_percentage: None or float, optional
        Percentage of GT labels need to be covered by a cluster to consider it as a match.
    :param coverage_n: None or int, optional
        Number of GT labels need to be contained in a cluster to consider it as a match.
        Given a coverage_percentage, coverage_n will be ignored.
        If no coverage_percentage or coverage_n is given, 100% coverage will be considered for a match.
    :param one_cluster_m_events: boolean, optional
        Indicate the allowance to match one cluster with many events
        True - one cluster can be matched with many events
    :return: int, int, dictionary, dictionary
        Total GT labels (keywords)
        Total true positives (keywords)
        Dictionary of {event_k: {cluster_k: [#miss/n (missed_proportion), MatchedCluster]}}
            Summarises the all matched clusters to events. {cluster_k: [#miss/n (missed_proportion), MatchedCluster]} is
            sorted by missed_proportion in ascending order.
        Dictionary of {event_k: cluster_k}
            Summarises best cluster match to events
    """
    dict_matched_clusters = dict()  # {event_k: {cluster_k: [#miss/n (missed_proportion), MatchedCluster]}}
    dict_event_cluster = dict()  # {event_k: cluster_k}

    for event_k in dict_gt_events.keys():
        temp_dict_matched_clusters = dict()  # {cluster_k: [#miss/n (missed_proportion), MatchedCluster]}

        for cluster_k in dict_clusters.keys():
            # find best match (event duplicate) for the cluster
            missed_proportion, matched_cluster = find_best_match(dict_clusters[cluster_k], dict_gt_events[event_k],
                                                                 exact_match, cluster_n)
            # if coverage percentage is given
            if coverage_percentage:
                if matched_cluster.keyword_tp / matched_cluster.keyword_n >= coverage_percentage:
                    temp_dict_matched_clusters[cluster_k] = [missed_proportion, matched_cluster]
            # if coverage count is given
            elif coverage_n:
                if matched_cluster.keyword_tp >= coverage_n:
                    temp_dict_matched_clusters[cluster_k] = [missed_proportion, matched_cluster]
            else:
                if matched_cluster.keyword_tp == matched_cluster.keyword_n:
                    temp_dict_matched_clusters[cluster_k] = [missed_proportion, matched_cluster]

        # if any cluster match found for the GT event
        if len(temp_dict_matched_clusters.keys()) > 0:
            # sort matched clusters by missed_proportion in ascending order
            temp_dict_matched_clusters = dict(sorted(temp_dict_matched_clusters.items(), key=lambda e: e[1][0]))

            dict_matched_clusters[event_k] = temp_dict_matched_clusters

    # If a single cluster can be matched with multiple events
    if one_cluster_m_events:
        for k, v in dict_matched_clusters.items():
            dict_event_cluster[k] = list(v.keys())[0]
    # If a single cluster cannot be matched with multiple events
    else:
        cluster_ks = dict_clusters.keys()

        cost_matrix = []  # cost = missed_proportion, rows = events, columns = clusters
        for event_k in dict_gt_events:
            event_costs = []
            for cluster_k in cluster_ks:
                if event_k in dict_matched_clusters and cluster_k in dict_matched_clusters[event_k]:
                    event_costs.append(dict_matched_clusters[event_k][cluster_k][0])
                else:
                    event_costs.append(1)  # no cluster words matched with the event keywords
            cost_matrix.append(event_costs)

        cost_matrix = np.array(cost_matrix)
        # Hungarian algorithm (Kuhn-Munkres) for solving the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i in range(len(row_ind)):
            # only consider clusters matched to events
            # no keyword matches -> missed_proportion = 1
            if cost_matrix[row_ind[i], col_ind[i]] < 1:
                dict_event_cluster[row_ind[i]] = col_ind[i]

    total_keyword_tp = 0
    total_keyword_n = 0
    for event_k, cluster_k in dict_event_cluster.items():
        total_keyword_n += dict_matched_clusters[event_k][cluster_k][1].keyword_n
        total_keyword_tp += dict_matched_clusters[event_k][cluster_k][1].keyword_tp

    return total_keyword_n, total_keyword_tp, dict_matched_clusters, dict_event_cluster
