import os
from bridging_json_extract import BridgingJsonDocInformationExtractor
import numpy as np
from bridging_utils import *



class GenerateKGLearningData:
    def __init__(self, json_objects_path, jsonlines,dataset_name):
        self.json_objects_path = json_objects_path
        json_path = os.path.join(self.json_objects_path, jsonlines)
        self.json_objects = read_jsonlines_file(json_path)
        self.dataset_name = dataset_name
        self.number_anaphors = 0

    def get_anaphor_cand_antecedents_labels_for_doc(self,json_object):
        """
        For a document get anaphor and its all possible candidate antecedents. If candidate antecedent
        is true antecedent then labelled as bridging otherwise non-bridging.

        Anaphor and candidate antecedents are normalized to its semantic heads so that they can be
        further used in the KG entity matching.
        :param json_object:
        :return:
        """
        mention_ops = BridgingJsonDocInformationExtractor(json_object, logger)
        if mention_ops.is_file_empty:
            return None
        ana_ante_pair_label_list = []
        for i,ana in enumerate(mention_ops.anaphors):
            self.number_anaphors +=1
            is_true_antecedent_present = False
            current_cand_ante = mention_ops.get_cand_antecedents(ana,-1,is_salient=True,is_apply_hou_rules=False)

            true_antecedents = mention_ops.ana_to_antecedents_map[ana]
            logger.debug("true antecedents {}".format(true_antecedents))

            current_cand_ante += current_cand_ante + true_antecedents
            current_cand_ante = list(set(current_cand_ante))
            logger.debug("candidate {}".format(current_cand_ante))

            ana_word = mention_ops.get_words_for_start_end_indices(ana)
            ana_head,_ = mention_ops.get_span_head(ana,is_semantic_head=True)

            current_cand_ante_word = mention_ops.get_span_words(current_cand_ante)
            current_cand_ante_head,_ = mention_ops.get_spans_head(current_cand_ante, is_semantic_head=True)

            assert len(current_cand_ante_word) == len(current_cand_ante_head) == len(current_cand_ante)
            for can_ante,cand_word,cand_head in zip(current_cand_ante,current_cand_ante_word,current_cand_ante_head):
                curr_pair_label_list = []
                logger.debug(3*"---")
                logger.debug("current ante {}".format(can_ante))
                curr_pair_label_list.append(ana_word)
                curr_pair_label_list.append(cand_word)
                curr_pair_label_list.append(ana_head)
                curr_pair_label_list.append(cand_head)
                if can_ante in true_antecedents:
                    curr_pair_label_list.append("Y")
                    is_true_antecedent_present = True
                else:
                    curr_pair_label_list.append("N")
                ana_ante_pair_label_list.append(curr_pair_label_list)
            assert is_true_antecedent_present
        return ana_ante_pair_label_list

    def write_tsv_mention_pairs(self,ana_ante_pair_label_list):
        tsv_file_name = os.path.join(is_notes_data_path,self.dataset_name+"_ana_antecedents_labels.tsv")
        tsv_file = open(tsv_file_name, mode='w')
        tsv_writer = csv.writer(tsv_file,delimiter="\t")
        tsv_writer.writerow(["Anaphor","Antecedent","Anaphor Head","Antecedent Head", "Label"])
        for ana_ante_pair_label in ana_ante_pair_label_list:
            tsv_writer.writerow(ana_ante_pair_label)
        logger.debug("tsv file creation completed.")
        tsv_file.close()

    def get_anaphor_cand_antecedents_labels_for_dataset(self):
        """
        For each document in the dataset get the anaphor antecedent pairs with labels.
        :return:
        """
        ana_ante_pair_label_list = []
        for json_object in self.json_objects:
            ana_ante_pair_label_list_per_doc = self.get_anaphor_cand_antecedents_labels_for_doc(json_object)
            if ana_ante_pair_label_list_per_doc is None:
                continue
            else:
                for pair in ana_ante_pair_label_list_per_doc:
                    assert len(pair) == 5
                print(5*"-----")
                print(ana_ante_pair_label_list_per_doc[0:5])
                ana_ante_pair_label_list += ana_ante_pair_label_list_per_doc
        self.write_tsv_mention_pairs(ana_ante_pair_label_list)
        print("number of anaphors",self.number_anaphors)
        return ana_ante_pair_label_list



if __name__ == '__main__':
    ac = GenerateKGLearningData(json_objects_path=bashi_data_path, jsonlines=bashi_jsonlines,dataset_name="bashi")
    # ac = GenerateKGLearningData(json_objects_path=is_notes_data_path,jsonlines=is_notes_jsonlines,dataset_name="isnotes")
    ac.get_anaphor_cand_antecedents_labels_for_dataset()