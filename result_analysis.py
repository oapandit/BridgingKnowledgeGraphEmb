import os
from bridging_utils import *
from bridging_json_extract import BridgingJsonDocInformationExtractor
from evaluation_measures import *
from stat_significance_tests import mcNemar_test,wilcoxon_test
class ResultAnalysis:
    def __init__(self,pred_jsonl_paths):
        self.pred_jsonl_paths = pred_jsonl_paths
        pass

    def get_accuracy_and_sentences(self,jsonl):
        jsonl_path = os.path.join(self.pred_jsonl_paths,jsonl)
        json_objects = read_jsonlines_file(jsonl_path)
        total_pred_pair_labels = []
        total_gold = 0
        accurate_pred = 0
        pred_sentence_pairs = []
        true_sentence_pairs = []
        for single_json_object in json_objects:
            mention_ops = BridgingJsonDocInformationExtractor(single_json_object,logger)
            pred_pairs = single_json_object.get('predicted_pairs')
            _acc_pred,_,_gold,pred_pair_labels = get_accurate_pred_gold_numbers(mention_ops,pred_pairs)
            total_pred_pair_labels += pred_pair_labels
            total_gold += _gold
            accurate_pred += _acc_pred

            pred_sentence_pairs += mention_ops.get_pairwise_sentences(pred_pairs)
            true_sentence_pairs += mention_ops.get_pairwise_sentences(mention_ops.bridging_clusters)
        assert total_gold == len(total_pred_pair_labels) == len(pred_sentence_pairs)==\
               len(true_sentence_pairs) == 663
        print("total accuracy : {}".format(accurate_pred/total_gold))
        print("\n\n\n")
        print(pred_sentence_pairs[0:5])
        print("\n\n\n")
        print(true_sentence_pairs[0:5])
        return total_pred_pair_labels,true_sentence_pairs,pred_sentence_pairs

if __name__ == '__main__':
    bert_path2vec_file = "svm_ranking_multi_isnotes_data_sem_heads_bert_emb_pp_hou_like_shp_avg_path2vec_train_2_test_2_sen_wind_size_C_1000_gamma__kernel_linear_27092020-081344_pred.pairs.jsonl"
    bert_file = "svm_ranking_multi_isnotes_data_sem_heads_bert_emb_pp_hou_like_train_2_test_2_sen_wind_size_C__gamma__kernel__27092020-080517_pred.pairs.jsonl"
    ra = ResultAnalysis(pred_jsons_path)
    bp_total_pred_pair_labels,bp_true_sentence_pairs,bp_pred_sentence_pairs = ra.get_accuracy_and_sentences(bert_path2vec_file)
    b_total_pred_pair_labels, b_true_sentence_pairs, b_pred_sentence_pairs = ra.get_accuracy_and_sentences(
        bert_file)
    mcNemar_test(bp_total_pred_pair_labels,b_total_pred_pair_labels)
    wilcoxon_test(bp_total_pred_pair_labels,b_total_pred_pair_labels)

    for i,(a,b) in enumerate(zip(bp_total_pred_pair_labels,b_total_pred_pair_labels)):
        if a == 0 and b == 1:

            logger.critical(5*"+++++")
            logger.critical("CORRECT LINK")
            logger.critical(bp_true_sentence_pairs[i])
            logger.critical("\n PRED with graph embeddings\n")
            logger.critical(bp_pred_sentence_pairs[i])
            logger.critical("\n PRED with bert\n")
            logger.critical(b_pred_sentence_pairs[i])

