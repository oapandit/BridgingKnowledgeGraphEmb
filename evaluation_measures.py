from bridging_utils import *


def get_accurate_pred_gold_numbers(mention_ops,pred_pairs):
    """
    measure how many accurate predictions
    measure number of predictions and number of gold pairs
    :param gold_pairs:
    :param pred_pairs:
    :return:
    """
    logger.debug(15*"***")
    logger.debug("evaluating {}".format(mention_ops.doc_key))
    logger.debug("predictions {}".format(pred_pairs))
    # pred_pairs = list(set([tuple(sorted(cluster)) for cluster in pred_pairs]))
    # logger.debug("changed predictions {}".format(pred_pairs))

    accurate_pred = 0
    pred_pair_labels = [] # for each pair we will assign label, 0 for inaccurate 1 for accurate
    for ante,ana in pred_pairs:
        logger.debug(5*"----")
        if not isinstance(ana,tuple):
            ana = tuple(ana)
        if not isinstance(ante,tuple):
            ante = tuple(ante)
        correct_antecedents_list = mention_ops.ana_to_antecedents_map.get(ana,None)
        assert len(correct_antecedents_list)>0,"for anaphor {} correct antecdent list {}".format(ana,correct_antecedents_list)
        assert type(ante) is type(correct_antecedents_list[0]),"ante is of {} and correct ante of {}".format(type(ante),type(correct_antecedents_list[0]))
        logger.debug("gold antecendets {}".format(mention_ops.gold_ana_to_antecedent_map[ana]))
        logger.debug("correct antecedents for {} ana {}".format(ana,correct_antecedents_list))
        logger.debug("prediction {}".format(ante))
        logger.debug("coref clusters {}".format(mention_ops.coref_clusters))
        if ante in correct_antecedents_list:
            accurate_pred += 1
            pred_pair_labels.append(1)
            logger.debug("found in correct")
        else:
            pred_pair_labels.append(0)
            logger.debug("not found")
    # print(accurate_pred)
    assert len(pred_pairs) == len(pred_pair_labels)
    total_pred = len(pred_pairs)
    total_gold = len(mention_ops.bridging_clusters)
    assert total_pred == total_gold
    return accurate_pred,total_pred,total_gold,pred_pair_labels

def get_prec_rec_f1_scores(accurate_pred,total_pred,total_gold):
    precision = accurate_pred / total_pred
    recall = accurate_pred / total_gold
    if precision!=0 or recall!=0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1

def get_accuracy(cum_acc_pred, cum_total_pred, cum_total_gold):
    assert cum_total_pred==cum_total_gold
    return cum_acc_pred/cum_total_gold

def evaluate_anaphor_antecedent_pairs(mention_ops,pred_pairs):
    """
    1. how many pairs are correctly calculated i.e. number of pairs in prediction which are
       accurate. this is a precision
       #accurate_pairs/#total_pred
    2. recall will be how many pairs from the gold pairs are correctly calculated
       #accurate_pairs/#total_gold
    3. f1 = 2 * (precision * recall) / (precision + recall)
    :param gold_pairs:
    :param pred_pairs:
    :return:
    """
    accurate_pred, total_pred, total_gold = get_accurate_pred_gold_numbers(mention_ops,pred_pairs)
    precision,recall,f1 = get_prec_rec_f1_scores(accurate_pred, total_pred, total_gold)
    logger.debug("precision {} , recall {} and f1 {}".format(precision, recall, f1))
    return precision,recall,f1,accurate_pred,total_pred,total_gold

def error_analysis(mention_ops,pred_pair,pred_pair_labels):
    logger.debug(5 * "---")
    logger.debug("doc {}".format(mention_ops.doc_key))
    assert len(pred_pair_labels) == len(pred_pair)
    for i,cl in enumerate(pred_pair):
        m1, m2 = cl
        logger.debug(5*"---")
        logger.debug("sent 1 : {}".format(mention_ops.get_span_sentences([m1])[0]))
        logger.debug("sent 2 : {}".format(mention_ops.get_span_sentences([m2])[0]))
        logger.debug("mention 1 : {}".format(mention_ops.get_span_words([m1])[0]))
        logger.debug("mention 2 : {}".format(mention_ops.get_span_words([m2])[0]))
        logger.debug("pairing {}".format("inaccurate" if pred_pair_labels[i]==0 else "accurate"))