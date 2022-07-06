import os
import sys

from bridging_json_extract import BridgingJsonDocInformationExtractor
import numpy as np
from bridging_utils import *
from word_embeddings import Path2VecEmbeddings

from data_read import BridgingVecDataReader

wordnet_mapped_alberto = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/wordnet/wn_mapped_nltk.nt"

from nltk.corpus import wordnet as wn

def is_synset(words):
    try:
        if "." in words:
            w,p,n = words.split(".")
            n = int(n)
            return True
    except:
        logger.critical("Error while dealing with word {}".format(words))
        # sys.exit()
    return False

def get_synset(offset_id):
    if not is_synset(offset_id):
        offset_id_num,pos = offset_id.split("-")
        p = str(pos)
        id = int(offset_id_num)
        try:
            synset = wn.synset_from_pos_and_offset(p,id)
        except:
            logger.critical("======")
            logger.critical("error while getting synset for : {}".format(offset_id))
            return offset_id
        return synset
    else:
        return offset_id

# def clean_wordnet_triplet_file(wordnet_trip_path=wordnet_mapped_alberto):
#     """
#     the file given by Alberto contains wordnet synsets as well as offset ids.
#     So to get the singal format changing offset ids to synsets.
#     :param wordnet_trip_path:
#     :return:
#     """
#     wordnet_data = read_text_file(wordnet_trip_path)
#     args_to_rel_dict = {}
#     for line in wordnet_data:
#         arg1, rel, arg2 = line.split()
#         p = str()
#         id = int()
#         wn.synset_from_pos_and_offset('n',4543158)

def load_wordnet_triplets_as_dict(wordnet_trip_path=wordnet_mapped_alberto):
    """
    Wordnet triplets are the two senses and relation between them. These are provided in a file,
    we read it and convert into dictionary format.
    A key is <sense_1>_<sense_2> and value is <rel>
    :param wordnet_trip_path:
    :return:
    """
    wn_rels = []
    wordnet_data = read_text_file(wordnet_trip_path)
    args_to_rel_dict = {}
    i = 1
    print("total lines {}".format(len(wordnet_data)))
    for line in wordnet_data:
        # print("line : {}".format(line))
        arg1,rel,arg2 = line.split()
        # arg1 = get_synset(arg1)
        # arg2 = get_synset(arg2)
        args = "{}_{}".format(arg1,arg2)
        args_to_rel_dict[args] = rel
        wn_rels.append(rel)
        if i%2000:
            print("{} - {} : {}".format(arg1,arg2,rel))
        i += 1
    i = 1
    # for k in args_to_rel_dict.keys():
    #     if i%20:
    #         break
    #     print("{} - {}".format(k,args_to_rel_dict[k]))
    #     i+=1
    print("length of dict {}".format(len(args_to_rel_dict)))
    print("wordnet rels {}".format(set(wn_rels)))
    return args_to_rel_dict


class BridgingWordNetAnalysis:
    def __init__(self, json_objects_path, jsonlines):
        self.json_objects_path = json_objects_path
        json_path = os.path.join(self.json_objects_path, jsonlines)
        self.json_objects = read_jsonlines_file(json_path)
        self.wordnet = Path2VecEmbeddings("", logger, 0)

        self.corpus_vec_data = BridgingVecDataReader(json_objects_path=json_objects_path, vec_path=is_notes_vec_path,
                                 is_read_head_vecs=True, is_sem_head=True,
                                 is_all_words=False, path2vec_sim=2,
                                 is_hou_like=False,wn_vec_alg=PATH2VEC)
        self.corpus_vec_data.is_soon = False
        self.corpus_vec_data.is_positive_surr = False
        self.sen_window_size  = 2
        self.is_consider_saleint = True
        self.is_training = False
        self.args_to_rel_dict = None
        self.rel_not_found = 0
        self.total_pairs = 0

    def get_wordnet_rel(self,ana_sense,cand_ante_sense):
        ana_sense = ana_sense.lower()
        cand_ante_sense = cand_ante_sense.lower()
        assert self.args_to_rel_dict is not None
        args1 = "{}_{}".format(ana_sense,cand_ante_sense)
        args2 = "{}_{}".format(cand_ante_sense,ana_sense)
        return self.args_to_rel_dict.get(args1,self.args_to_rel_dict.get(args2,None))

    def get_words_heads_senses_for_mentions(self,bridging_json,is_first_sense):
        interesting_entities_words = bridging_json.get_span_words(bridging_json.interesting_entities)
        interesting_entities_sem_head_words, sem_head_spans = bridging_json.get_spans_head(
            bridging_json.interesting_entities, True)
        if is_first_sense :
            interesting_entities_context = None
        else:
            interesting_entities_context = bridging_json.get_surrounding_sentences_for_span_list(bridging_json.interesting_entities,l_context=2,r_context=1)
        interesting_entities_senses = self.wordnet.get_most_probable_sense_for_words(words=interesting_entities_sem_head_words,sents=interesting_entities_context)
        assert len(interesting_entities_words) == len(interesting_entities_sem_head_words) == len(interesting_entities_senses)

        return interesting_entities_words,interesting_entities_sem_head_words,interesting_entities_senses

    def generate_ana_cand_ant_sense_wordnet_rels_pos_neg(self,bridging_json):
        ana_to_antecedent_map = bridging_json.get_ana_to_anecedent_map_from_selected_data(
            True, False, False, True)
        cand_ante_per_anaphor = self.corpus_vec_data.generate_cand_ante_by_sentence_window(bridging_json, self.sen_window_size,self.is_consider_saleint, self.is_training, ana_to_antecedent_map)
        assert len(cand_ante_per_anaphor) == len(ana_to_antecedent_map.keys())
        anaphors = list(ana_to_antecedent_map)
        antecedents = [ana_to_antecedent_map[ana] for ana in anaphors]

        interesting_entities_words, interesting_entities_sem_head_words, interesting_entities_senses = self.get_words_heads_senses_for_mentions(bridging_json,is_first_sense=False)
        ana_words,cand_ante_words,ana_head_words,cand_ante_head_words,ana_senses,cand_ante_senses,is_bridg = [],[],[],[],[],[],[]
        for i in range(len(anaphors)):
            ana = anaphors[i]
            int_ent_ana_ind = bridging_json.interesting_entities.index(ana)
            for cand_ant_ind in cand_ante_per_anaphor[i]:
                if cand_ant_ind in antecedents[i]:
                    is_present = 1
                else:
                    is_present = 0
                int_ent_cand_ant_ind = bridging_json.interesting_entities.index(cand_ant_ind)

                ana_words.append(interesting_entities_words[int_ent_ana_ind])
                cand_ante_words.append(interesting_entities_words[int_ent_cand_ant_ind])

                ana_head_words.append(interesting_entities_sem_head_words[int_ent_ana_ind])
                cand_ante_head_words.append(interesting_entities_sem_head_words[int_ent_cand_ant_ind])

                ana_senses.append(interesting_entities_senses[int_ent_ana_ind])
                cand_ante_senses.append(interesting_entities_senses[int_ent_cand_ant_ind])

                is_bridg.append(is_present)

        assert len(ana_words) == len(cand_ante_words) == len(ana_head_words) == len(cand_ante_head_words) == len(ana_senses) == len(cand_ante_senses) == len(is_bridg)
        ana_cand_ante_rels = self.get_wordnet_rel_for_ana_cands(ana_senses,cand_ante_senses)
        return ana_words,cand_ante_words,ana_head_words,cand_ante_head_words,ana_senses,cand_ante_senses,is_bridg,ana_cand_ante_rels

    def get_wordnet_rel_for_ana_cands(self,ana_senses,cand_ante_senses):
        ana_cand_ante_rels = []
        for ana_sense,cand_ante_sense in zip(ana_senses,cand_ante_senses):
            rel = self.get_wordnet_rel(ana_sense,cand_ante_sense)
            if rel is None:
                self.rel_not_found += 1
            ana_cand_ante_rels.append(rel)
        return ana_cand_ante_rels


    def get_info_for_corpus(self):
        self.args_to_rel_dict = load_wordnet_triplets_as_dict()
        corp_ana_words, corp_cand_ante_words, corp_ana_head_words, corp_cand_ante_head_words, corp_ana_senses, corp_cand_ante_senses, corp_is_bridg, corp_ana_cand_ante_rels = ["Anaphors"], ["Candidate Antecedents"], ["Anaphor Heads"], ["Candidate Antecedent Heads"], ["Anaphor Sense"], ["Candidate Antecedent Sense"], ["Bridging"], ["Wordnet Relation"]
        for json_object in self.json_objects:
            bridging_json= BridgingJsonDocInformationExtractor(json_object, logger)
            if bridging_json.is_file_empty : continue
            ana_words, cand_ante_words, ana_head_words, cand_ante_head_words, ana_senses, cand_ante_senses, is_bridg, ana_cand_ante_rels = self.generate_ana_cand_ant_sense_wordnet_rels_pos_neg(bridging_json)

            corp_ana_words += ana_words
            corp_cand_ante_words += cand_ante_words
            corp_ana_head_words += ana_head_words
            corp_cand_ante_head_words += cand_ante_head_words
            corp_ana_senses += ana_senses
            corp_cand_ante_senses += cand_ante_senses
            corp_is_bridg += is_bridg
            corp_ana_cand_ante_rels += ana_cand_ante_rels

            assert len(corp_ana_words) == len(corp_cand_ante_words) == len(corp_ana_head_words) == len(
                corp_cand_ante_head_words) == len(corp_ana_senses) == len(corp_cand_ante_senses) == len(corp_is_bridg)
        print("out of {}, for {} pairs relation is not found".format(len(corp_ana_words),self.rel_not_found))
        return corp_ana_words, corp_cand_ante_words, corp_ana_head_words, corp_cand_ante_head_words, corp_ana_senses, corp_cand_ante_senses, corp_is_bridg, corp_ana_cand_ante_rels


    def write_csv(self,corp_ana_words, corp_cand_ante_words, corp_ana_head_words, corp_cand_ante_head_words, corp_ana_senses, corp_cand_ante_senses, corp_is_bridg, corp_ana_cand_ante_rels):
        data = list(zip(corp_ana_words, corp_cand_ante_words, corp_ana_head_words, corp_cand_ante_head_words, corp_ana_senses, corp_cand_ante_senses, corp_is_bridg, corp_ana_cand_ante_rels))
        write_csv(bridging_ana_cand_ante_wordnet_rel_file,data)


if __name__ == '__main__':
    bwn_ana = BridgingWordNetAnalysis(json_objects_path=is_notes_data_path, jsonlines=is_notes_jsonlines)
    corp_ana_words, corp_cand_ante_words, corp_ana_head_words, corp_cand_ante_head_words, corp_ana_senses, corp_cand_ante_senses, corp_is_bridg, corp_ana_cand_ante_rels = bwn_ana.get_info_for_corpus()
    bwn_ana.write_csv(corp_ana_words, corp_cand_ante_words, corp_ana_head_words, corp_cand_ante_head_words,
              corp_ana_senses, corp_cand_ante_senses, corp_is_bridg, corp_ana_cand_ante_rels)