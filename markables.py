from bridging_utils import *
import random

class Word:
    def __init__(self, id, word, word_pos_in_doc):
        self.id = id
        self.word = word
        self.word_pos_in_doc = word_pos_in_doc

    def __str__(self):
        return "{} : {}".format(self.id, self.word)


class Bridg:
    def __init__(self, id1, id2):
        self.id1 = get_id_number(id1)
        self.id2 = get_id_number(id2)

    def get_tuple_loc(self,entities_id_to_mark_map):
        tuple1 = entities_id_to_mark_map[self.id1].get_ent_pos_in_doc()
        tuple2 = entities_id_to_mark_map[self.id2].get_ent_pos_in_doc()
        return [tuple1,tuple2]

    def __str__(self):
        return "{} -- {}".format(self.id1, self.id2)

class Anaphors_bashi:
    def __init__(self,doc_name,id,ana_type,ana_sent,ana_start,ana_end,ante_sent,ante_start,ante_end):
        self.doc_name = doc_name
        self.id = id

        self.ana_type = ana_type
        self.ana_sent = ana_sent
        self.ana_start = ana_start
        self.ana_end = ana_end

        self.ante_sent = ante_sent
        self.ante_start = ante_start
        self.ante_end = ante_end

        self.anaphor_ind = None
        self.antecedent_ind = None

    def _get_start_end_ind(self,sent,sent_start_pos,sent_end_pos,sentences_to_word_doc_pos):
        sentence = sentences_to_word_doc_pos[int(sent)]
        return [sentence[int(sent_start_pos)],sentence[int(sent_end_pos)]]

    def set_start_end(self,sentences_to_word_doc_pos):
        self.anaphor_ind = self._get_start_end_ind(self.ana_sent,self.ana_start,self.ana_end,sentences_to_word_doc_pos)
        self.antecedent_ind = self._get_start_end_ind(self.ante_sent,self.ante_start,self.ante_end,sentences_to_word_doc_pos)

    def get_bridging_pair(self):
        assert self.anaphor_ind is not None and self.antecedent_ind is not None
        return [self.antecedent_ind,self.anaphor_ind]

    def __str__(self):
        return "{} : {}".format(self.doc_name,self.id)

class Entity:
    def __init__(self, id, start_word_id, end_word_id):
        self.id = id
        self.start_word_id = get_id_number(start_word_id)
        self.end_word_id = get_id_number(end_word_id)

    def set_ent_start_end_pos_in_doc(self, word_id_to_mark_map):
        self.start_word_pos_in_doc = word_id_to_mark_map[self.start_word_id].word_pos_in_doc
        self.end_word_pos_in_doc = word_id_to_mark_map[self.end_word_id].word_pos_in_doc

    def get_ent_pos_in_doc(self):
        return (self.start_word_pos_in_doc,self.end_word_pos_in_doc)

    def __str__(self):
        return "{} : {}-{}".format(self.id, self.start_word_id, self.end_word_id)


class Sentence:
    def __init__(self, id, start_word, end_word):
        self.id = id
        self.start_word_id = get_id_number(start_word)
        self.end_word_id = get_id_number(end_word)

    def __str__(self):
        return "{} : {}-{}".format(self.id, self.start_word_id, self.end_word_id)


class ARRAUMarkable:
    def __init__(self,id,start_pos,head_pos,coref_chain_id,is_bridging,antecedent_id=None,antecedent_chain=None,bridg_rel=None):
        self.id = id
        self.start_pos = start_pos
        self.head_pos = head_pos
        self.end_pos = None
        self.coref_chain_id = coref_chain_id
        self.is_bridging_anaphor = is_bridging
        self.antecedent_id = antecedent_id
        self.antecedent_chain = antecedent_chain
        self.bridg_rel = bridg_rel

    def close_markable(self,last_tok_nm):
        assert self.end_pos is None
        self.end_pos = last_tok_nm
        logger.debug("markable id : {}start : {}, end : {} and head : {}".format(self.id,self.start_pos,self.end_pos,self.head_pos))
        assert self.start_pos <= self.end_pos
        if self.head_pos is None:
            self.head_pos = random.randint(self.start_pos,self.end_pos)
        try:
            assert self.start_pos <= self.head_pos
            assert self.head_pos <= self.end_pos
        except:
            self.head_pos = self.end_pos

    def get_start_end(self):
        assert self.end_pos is not None
        return [self.start_pos,self.end_pos]

    def get_head(self):
        assert self.head_pos is not None
        return self.head_pos

    def __str__(self):
        return "markable is : {} \n start : {} \n end : {} \n head : {} \n coref chain {}\n" \
               "is bridg : {} \n" \
               "ante id : {} \n" \
               "ante coref chain : {}".format(self.id,self.start_pos,self.end_pos,self.head_pos,self.coref_chain_id,
                                              self.is_bridging_anaphor,self.antecedent_id,self.antecedent_chain)

