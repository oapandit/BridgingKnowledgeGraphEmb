import os
from common_code import *
from bridging_json_extract import BridgingJsonDocInformationExtractor
import numpy as np
from bridging_utils import *
import random
from sklearn.model_selection import KFold
from random_fourier_features import IRFF

class BridgingVecDataReader:

    def __init__(self, json_objects_path, vec_path,is_read_head_vecs,is_sem_head,is_all_words,wn_vec_alg,path2vec_sim,is_hou_like,is_consider_arti_good_pairs=False,is_consider_arti_bad_pairs=False,is_consider_arti_ugly_pairs=False,max_antecedents=None):
        self.json_objects_path = json_objects_path
        self.vec_path = vec_path
        self.is_consider_arti_good_pairs = is_consider_arti_good_pairs
        self.is_consider_arti_bad_pairs = is_consider_arti_bad_pairs
        self.is_consider_arti_ugly_pairs = is_consider_arti_ugly_pairs
        self.num_anaphors = 0
        self.num_ana_cand_antecedent_has_true=0
        self.is_read_head_vecs = is_read_head_vecs
        self.is_sem_head = is_sem_head
        self.is_hou_like = is_hou_like
        self.is_all_words=is_all_words

        assert not (self.is_hou_like and self.is_all_words) or not (self.is_hou_like and self.is_sem_head)

        if max_antecedents is None:
            self.max_ante = max_ante
        else:
            self.max_ante = max_antecedents
        assert path2vec_sim <4
        self.pe_extn = path2vec_sims_file_extns[path2vec_sim]
        self.irff = IRFF()
        create_dir(is_notes_svm_ranking_data_path)
        assert wn_vec_alg in wn_vec_alg_options
        self.wn_vec_alg = wn_vec_alg

    def set_params(self,exp_params):
        """
        exp_params is object of class ExptParams which has all the params to be set for
        reading data. We read those param values and set for reading bridging data.
        :param exp_params:
        :return:
        """
        self.is_wsd = exp_params.is_wsd
        self.is_avg = exp_params.is_avg
        self.is_w2vec = exp_params.is_w2vec
        self.is_glove = exp_params.is_glove
        self.is_bert = exp_params.is_bert
        self.is_path2vec = exp_params.is_path2vec
        self.is_emb_pp = exp_params.is_emb_pp
        self.word2vec_dim = exp_params.w2vec_dim
        self.glove_vec_dim = exp_params.glove_dim

        self.is_consider_arti_good_pairs = exp_params.is_consider_arti_good_pairs
        self.is_consider_arti_bad_pairs = exp_params.is_consider_arti_bad_pairs
        self.is_consider_arti_ugly_pairs = exp_params.is_consider_arti_ugly_pairs

        self.is_soon = exp_params.is_soon
        self.is_positive_surr = exp_params.is_positive_surr
        self.is_balanced = exp_params.is_balanced
        self.surr_window = exp_params.surr_window

        self.train_sen_window_size = exp_params.train_sen_window_size
        self.test_sen_window_size = exp_params.test_sen_window_size


    def get_json_objects(self, jsonlines):
        if isinstance(jsonlines,str):
            json_path = os.path.join(self.json_objects_path, jsonlines)
            json_objects = read_jsonlines_file(json_path)
            # random.shuffle(json_objects)
        elif isinstance(jsonlines,list):
            json_objects = jsonlines
        else:
            raise NotImplementedError
        return json_objects

    def get_arrau_json_objects_split(self, jsonlines,dataset_name):
        """
        In ARRAU jsonline contains json objects for RST, PEAR and TRAINS dataset.
        Also, these are divided as train / dev and test.
        This function returns train, dev and test object of the given dataset.
        :param jsonlines:
        :return:
        """
        train_objects,dev_objects,test_objects = [],[],[]
        json_objects = self.get_json_objects(jsonlines)
        print("number of objects : {}".format(len(json_objects)))
        if dataset_name == RST:
            num_obj = num_rst_docs
        elif dataset_name == TRAINS :
            num_obj = num_trains_docs
        elif dataset_name == PEAR :
            num_obj = num_pear_docs
        else:
            raise NotImplementedError
        for jo in json_objects:
            if jo['dataset_name'] == dataset_name:
                if jo['dataset_type'] == 'train':
                    train_objects.append(jo)
                elif jo['dataset_type'] == 'dev':
                    dev_objects.append(jo)
                elif jo['dataset_type'] == 'test':
                    test_objects.append(jo)
                else:
                    raise NotImplementedError
        assert num_obj == len(train_objects) + len(dev_objects) + len(test_objects),"num obj : {} and total train/dev/test: {}".format(num_obj,len(train_objects) + len(dev_objects) + len(test_objects))
        return train_objects,dev_objects,test_objects

    def get_k_fold_json_objects_split(self, jsonlines,k,is_dev=False,num_dev_docs=None):
        """
        divide given json objects as train test and if is_dev is set then as well as
        dev. number of dev docs should be provided if we want to devide as train dev and test.

        k defines fold
        :param jsonlines:
        :param k:
        :param is_dev:
        :return:
        """
        json_objects = self.get_json_objects(jsonlines)
        X = list(range(len(json_objects)))
        random.shuffle(X)
        kf = KFold(n_splits=k, shuffle=True)
        kf.get_n_splits(X)
        train_objects = []
        dev_objects = []
        test_objects = []
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index)
            print("TEST:", test_index)
            if is_dev:
                assert num_dev_docs is not None
                random.shuffle(train_index)
                dev_index = train_index[-num_dev_docs - 1:-1]
                train_index = train_index[0:-num_dev_docs]
                _dev = [json_objects[i] for i in dev_index]
                dev_objects.append(_dev)
            _train = [json_objects[i] for i in train_index]
            _test = [json_objects[i] for i in test_index]
            train_objects.append(_train)
            test_objects.append(_test)
        # print("train {} and test {}".format(len(_train),len(_test)))
        if is_dev:
            return train_objects, dev_objects, test_objects
        else:
            return train_objects,test_objects

    def _get_cand_ants_between_true_ante_anaphor(self,curr_anaphor,true_antecedents,cand_antecedents):
        new_cand_antecedents = []
        true_ant_starts = [int(ant[0]) for ant in true_antecedents]
        logger.debug("for true antecedents {} starts are {}".format(true_antecedents, true_ant_starts))
        first_true_start = min(true_ant_starts)
        curr_anaphor_start = curr_anaphor[0]
        for ant in cand_antecedents:
            s = ant[0]
            if first_true_start < s and s < curr_anaphor_start:
                new_cand_antecedents.append(ant)
        logger.debug("for anaphor {} and candidate antedents {} filtered candidates are {}".format(curr_anaphor,
                                                                                                   cand_antecedents,
                                                                                                   new_cand_antecedents))
        return new_cand_antecedents

    def remove_true_from_cand_ante(self,true_antecedents,cand_antecedents):
        new_cand_antecedents = [ant for ant in cand_antecedents if ant not in true_antecedents]
        return new_cand_antecedents

    def filter_candidates(self,curr_anaphor,true_antecedents,cand_antecedents):
        assert (self.is_positive_surr and not self.is_soon and not self.is_balanced) or (self.is_soon and not self.is_positive_surr and not self.is_balanced) or (self.is_balanced and not self.is_positive_surr and not self.is_soon)
        new_cand_antecedents = []
        cand_antecedents.sort()
        if self.is_soon:
            new_cand_antecedents = self._get_cand_ants_between_true_ante_anaphor(curr_anaphor,true_antecedents,cand_antecedents)
        elif self.is_positive_surr:
            for true_ant in true_antecedents:
                logger.debug(4*"---")
                _new_cand_antecedents = []
                r_cand_ant = []
                l_cand_ant = []
                s_true_ant, e_true_ant = true_ant
                for cand_ant in cand_antecedents:
                    s, e = cand_ant
                    if s < s_true_ant and e < s_true_ant:
                        l_cand_ant.append(cand_ant)
                    if s > s_true_ant and e > e_true_ant:
                        r_cand_ant.append(cand_ant)
                logger.debug(
                    "for true ant {} and candidate antedents {} left cand {} right candidate {}".format(
                        true_ant,
                        cand_antecedents, l_cand_ant,
                        r_cand_ant))

                for i in range(self.surr_window):
                    try:
                        _new_cand_antecedents.append(l_cand_ant[-1 - i])
                        _new_cand_antecedents.append(r_cand_ant[i])
                    except IndexError:
                        pass

                logger.debug("for window {} chosen surrounding candidates are {}".format(self.surr_window,_new_cand_antecedents))
                new_cand_antecedents += _new_cand_antecedents
                new_cand_antecedents.sort()
            logger.debug("candidate antedents {} surrounding candidates are {}".format(cand_antecedents,new_cand_antecedents))
        elif self.is_balanced:
            new_cand_antecedents = self.remove_true_from_cand_ante(true_antecedents,cand_antecedents)[-len(true_antecedents):]
            assert len(new_cand_antecedents) == len(true_antecedents),"new candidate {} and true ants {}".format(new_cand_antecedents,true_antecedents)
            logger.debug("blanced data candidates {}".format(new_cand_antecedents))
        else:
            raise NotImplementedError
        return new_cand_antecedents

    def generate_cand_ante_by_sentence_window(self, mention_ops, sen_window_size,is_consider_saleint, is_training, ana_to_antecedent_map,anaphors=None):
        """
        Generate mention pairs by selecting current mention and all the mention which occur in the
        window of size provided.

        If the falg is_positive_surr is set, then we are going to select negative samples in the surrounding of the positive samples, i.e. 2 candidates left and right to positive samples.

        If the flag is_soon set then we are going to generate negative pairs as mentioned Soon 2001 coreference paper.
        negative samples will be considered from the anaphor and positive samples.

        :return: list of tuples
        """
        if anaphors is None:
            anaphors = mention_ops.anaphors
        log_start("generating candidate antecedents for anaphor")
        cand_ante_per_anaphor = []
        if is_training and ana_to_antecedent_map is None:
            ana_to_antecedent_map = mention_ops.get_ana_to_anecedent_map_from_selected_data(self.is_consider_arti_good_pairs,self.is_consider_arti_bad_pairs,self.is_consider_arti_ugly_pairs,is_clean_art_data=True)
        for men_ind, curr_anaphor in enumerate(anaphors):
            logger.debug(5 * "---")
            curr_cand_ante = []
            if is_training:
                assert ana_to_antecedent_map is not None
                true_antecedents = ana_to_antecedent_map[curr_anaphor]
                curr_cand_ante += true_antecedents
                if self.is_positive_surr or self.is_soon:
                    cand_antecedents = mention_ops.get_cand_antecedents(curr_anaphor,-1,is_consider_saleint,is_apply_hou_rules=False)
                    cand_antecedents = self.filter_candidates(curr_anaphor,true_antecedents,cand_antecedents)
                else:
                    cand_antecedents = mention_ops.get_cand_antecedents(curr_anaphor, sen_window_size, is_consider_saleint,is_apply_hou_rules=False)
            else:
                mention_ops.set_interesting_entities_sem_heads(self.vec_path)
                cand_antecedents = mention_ops.get_cand_antecedents(curr_anaphor,sen_window_size,is_consider_saleint,is_apply_hou_rules=False)
            curr_cand_ante += cand_antecedents
            curr_cand_ante.sort(reverse=True)
            logger.debug("cand ana sorted {}".format(curr_cand_ante))
            cand_ante_per_anaphor.append(curr_cand_ante)
        log_finish()
        cand_ante_per_anaphor = [list(set(cand_ante_per_anaphor[i])) for i in range(len(cand_ante_per_anaphor))]
        for cand_ in cand_ante_per_anaphor:
            cand_.sort(reverse=True)
        return cand_ante_per_anaphor

    def _pad_words(self,cont_vec):
        word_emb = []
        max_word = 0
        num_ent = cont_vec.shape[0]
        vec_dim = cont_vec[0].shape[1]
        logger.debug("shape {} vec_dim {}".format(cont_vec[0].shape,vec_dim))
        for i in range(num_ent):
            emb = cont_vec[0]
            num_words = emb.shape[0]
            max_word = num_words if num_words>max_word else max_word
            if num_words<max_words_in_ent:
                pad_num_words = max_words_in_ent - num_words
                pad_vec = np.zeros((pad_num_words,vec_dim))
                emb_final = np.concatenate([emb,pad_vec])
            else:
                emb_final = emb[0:max_words_in_ent]
            logger.debug("emb shape {}".format(emb_final.shape))
            assert emb_final.shape[0] == max_words_in_ent
            assert emb_final.shape[1] == vec_dim
            word_emb.append(np.expand_dims(emb_final,axis=0))
        word_emb = np.concatenate(word_emb)
        logger.debug("final all ent emb {}".format(word_emb.shape))
        assert word_emb.shape[0] == num_ent
        assert word_emb.shape[1] == max_words_in_ent
        assert word_emb.shape[2] == vec_dim
        return word_emb

    def read_vecs(self, mention_ops):
        doc_name = mention_ops.doc_key
        cont_vec = None
        path2vec_vec = None
        emb_pp_ana_vec = None
        emb_pp_cand_ant_vec = None

        assert self.is_glove or self.is_w2vec or self.is_bert or self.is_path2vec or self.is_emb_pp
        assert (not (self.is_glove and self.is_bert)) and (not (self.is_w2vec and self.is_bert)) and (not (self.is_w2vec and self.is_glove))

        assert not (self.is_read_head_vecs and self.is_all_words)
        w2vec_dim_extn = "_{}d_".format(self.word2vec_dim)
        glove_dim_extn = "_{}d_".format(self.glove_vec_dim)

        emb_pp_extn = sem_head_extn
        if self.is_read_head_vecs and not self.is_all_words:
            _head_extn = sem_head_extn if self.is_sem_head else head_extn
            emb_pp_extn = _head_extn
        else:
            _head_extn = ""
        _extn = all_extn if self.is_all_words else ""
        if self.is_hou_like:
            _hou_extn = hou_like_emb_extn
            emb_pp_extn = ""
        else:
            _hou_extn = ""

        if self.is_bert:
            cont_vec = read_pickle(os.path.join(self.vec_path, doc_name +_extn+ _head_extn+ bert_vec_file_extn))
        if self.is_w2vec:
            cont_vec = read_pickle(os.path.join(self.vec_path, doc_name +_extn+w2vec_dim_extn+ _hou_extn + _head_extn+ w2vec_vec_file_extn))
        if self.is_glove:
            cont_vec = read_pickle(os.path.join(self.vec_path, doc_name +_extn+glove_dim_extn+_hou_extn + _head_extn+ glove_vec_file_extn))
        if self.is_emb_pp:
            emb_pp_ana_vec = read_pickle(os.path.join(self.vec_path, doc_name + _hou_extn + emb_pp_extn + embeddings_pp_ana_vec_file_extn))
            emb_pp_cand_ant_vec = read_pickle(os.path.join(self.vec_path, doc_name + _hou_extn + emb_pp_extn + embeddings_pp_cand_ant_vec_file_extn))

            assert emb_pp_ana_vec.shape[0] == len(mention_ops.gold_and_art_anaphors)
            assert emb_pp_cand_ant_vec.shape[0] == len(mention_ops.interesting_entities)
            assert emb_pp_cand_ant_vec.shape[1] == emb_pp_ana_vec.shape[1] == embeddings_pp_dim

        if cont_vec is not None:
            logger.debug("contextual embeddings shape {} and number of interesting entities {}".format(cont_vec.shape,len(mention_ops.interesting_entities)))
            assert cont_vec.shape[0]==len(mention_ops.interesting_entities)

            if self.is_all_words:
                cont_vec = self._pad_words(cont_vec)
        if self.is_path2vec:
            if self.wn_vec_alg == PATH2VEC:
                if self.is_wsd:
                    path2vec_vec = read_pickle(os.path.join(self.vec_path, doc_name + self.pe_extn + path2vec_all_sense_vec_file_extn))
                elif self.is_avg:
                    path2vec_vec = read_pickle(os.path.join(self.vec_path, doc_name + self.pe_extn + path2vec_sense_avg_vec_file_extn))
                else:
                    path2vec_vec = read_pickle(os.path.join(self.vec_path, doc_name + self.pe_extn + path2vec_sense_lesk_vec_file_extn))
                logger.debug("path2vec embeddings shape {} and number of interesting entities {}".format(path2vec_vec.shape,
                                                                                                       len(mention_ops.interesting_entities)))
            elif self.wn_vec_alg == BRWN2VEC:
                if self.is_wsd:
                    path2vec_vec = read_pickle(os.path.join(self.vec_path, doc_name + br_wn_vec_all_sense_vec_file_extn))
                elif self.is_avg:
                    path2vec_vec = read_pickle(os.path.join(self.vec_path, doc_name + br_wn_vec_sense_avg_vec_file_extn))
                else:
                    path2vec_vec = read_pickle(os.path.join(self.vec_path, doc_name + br_wn_vec_sense_lesk_vec_file_extn))
                logger.debug("path2vec embeddings shape {} and number of interesting entities {}".format(path2vec_vec.shape,
                                                                                                       len(mention_ops.interesting_entities)))
            elif self.wn_vec_alg == WN2VEC:
                path2vec_vec = read_pickle(
                    os.path.join(self.vec_path, doc_name + sem_head_extn + wn2vec_file_extn))
            elif self.wn_vec_alg == RWWN2VEC:
                path2vec_vec = read_pickle(
                    os.path.join(self.vec_path, doc_name +sem_head_extn + rw_wn_vec_file_extn))
            else:
                raise NotImplementedError
            assert path2vec_vec.shape[0] == len(mention_ops.interesting_entities)
        return cont_vec, path2vec_vec,emb_pp_ana_vec,emb_pp_cand_ant_vec

    def get_vecs_for_mentions(self, mention_ops, mentions, interesting_entities_vecs,anaphors_vecs=None,is_mention_anaphor=None):
        if is_mention_anaphor is not None:
            assert anaphors_vecs is not None
            assert anaphors_vecs.shape[0] == len(mention_ops.gold_and_art_anaphors)
            assert len(mentions) == len(is_mention_anaphor)
        assert interesting_entities_vecs.shape[0] == len(mention_ops.interesting_entities)
        vecs = []
        for j,mention in enumerate(mentions):
            if is_mention_anaphor is not None and is_mention_anaphor[j]:
                ind = mention_ops.gold_and_art_anaphors.index(mention)
                vec_ = np.expand_dims(anaphors_vecs[ind], axis=0)
            else:
                ind = mention_ops.interesting_entities.index(mention)
                vec_ = np.expand_dims(interesting_entities_vecs[ind], axis=0)
            vecs.append(vec_)
        vecs = np.concatenate(vecs)
        assert vecs.shape[0] == len(mentions)
        assert vecs.shape[1] == interesting_entities_vecs.shape[1]
        return vecs

    def get_cand_ante_per_anaphor_and_vecs(self,mention_ops, sen_window_size, is_consider_saleint, is_training, ana_to_antecdents_map_selected_data=None, anaphors=None):
    # def get_cand_ante_per_anaphor_and_vecs(self, mention_ops,is_wsd,is_avg, is_bert, is_path2vec, is_w2vec, is_glove,is_embeddings_pp, sen_window_size,is_consider_saleint, is_training,ana_to_antecdents_map_selected_data=None,is_balanced=False,is_positive_surr=False,is_soon=False,surr_window=2,anaphors=None):
        context_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec = self.read_vecs(mention_ops)
        cand_ante_per_anaphor = self.generate_cand_ante_by_sentence_window(mention_ops, sen_window_size,is_consider_saleint, is_training,ana_to_antecdents_map_selected_data,anaphors)
        return cand_ante_per_anaphor,context_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec


    def generate_vecs_for_pairs(self,is_cont_emb, cont_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec,cand_ante_per_anaphor, cand_ante_per_anaphor_labels, mention_ops,anaphors=None):
        """
        for given list pairs (tuples of index of mentions) return vecs corrsponding to it from
        bert or/and path2vec
        :return:
        """
        vec_m1, vec_m2, vec_cont_m1, vec_ext_m1, vec_cont_m2, vec_ext_m2, vec_em_pp_m1, vec_em_pp_m2, label = None, None, None, None, None, None, None, None, None
        if self.is_wsd:
            vec_cont_m1 = []
            vec_ext_m1 = []

            vec_cont_m2 = []
            vec_ext_m2 = []
        else:
            vec_m1 = []
            vec_m2 = []

        label = []
        if emb_pp_ana_vec is not None:
            vec_em_pp_m1 = []
            vec_em_pp_m2 = []
            assert emb_pp_cand_ant_vec is not None

        if anaphors is None:
            anaphors = mention_ops.anaphors
        assert len(cand_ante_per_anaphor) == len(cand_ante_per_anaphor_labels) == len(anaphors)
        for ana_ind,cand_antecedents in enumerate(cand_ante_per_anaphor):
            curr_ana = anaphors[ana_ind]
            cur_labels = cand_ante_per_anaphor_labels[ana_ind]
            for cand_ante,cur_label in zip(cand_antecedents,cur_labels):
                if emb_pp_ana_vec is not None:
                    vec_em_pp_m1.append(self._get_vec_mention(True,emb_pp_ana_vec, None, mention_ops,
                                                             curr_ana,is_emb_pp_ana=True))
                    vec_em_pp_m2.append(self._get_vec_mention(True,emb_pp_cand_ant_vec, None, mention_ops,
                                                             cand_ante))
                if self.is_wsd:
                    if is_cont_emb:
                        vec_cont_m1.append(self._get_vec_mention(is_cont_emb, cont_vecs, None, mention_ops,
                                         curr_ana))
                        vec_cont_m2.append(self._get_vec_mention(is_cont_emb, cont_vecs, None, mention_ops,
                                         cand_ante))

                    vec_ext_m1.append(
                        self._get_vec_mention(False,None, path2vec_vecs, mention_ops,
                                              curr_ana))
                    vec_ext_m2.append(
                        self._get_vec_mention(False,None, path2vec_vecs, mention_ops,
                                              cand_ante))
                else:
                    if is_cont_emb or self.is_path2vec:
                        vec_m1.append(self._get_vec_mention(is_cont_emb, cont_vecs, path2vec_vecs, mention_ops,
                                         curr_ana))
                        vec_m2.append(
                            self._get_vec_mention(is_cont_emb, cont_vecs, path2vec_vecs, mention_ops,
                                              cand_ante))
                label.append(cur_label)
        label = np.array(label)

        if emb_pp_ana_vec is not None:
            vec_em_pp_m1 = np.concatenate(vec_em_pp_m1)
            vec_em_pp_m2 = np.concatenate(vec_em_pp_m2)

        if self.is_wsd:
            if is_cont_emb:
                vec_cont_m1 = np.concatenate(vec_cont_m1)
                vec_cont_m2 = np.concatenate(vec_cont_m2)

                if emb_pp_ana_vec is not None:
                    vec_cont_m1 = np.concatenate([vec_cont_m1,vec_em_pp_m1],axis=-1)
                    vec_cont_m2 = np.concatenate([vec_cont_m2,vec_em_pp_m2],axis=-1)

            else:
                vec_cont_m1 = vec_em_pp_m1
                vec_cont_m2 = vec_em_pp_m2


            vec_ext_m1 = np.concatenate(vec_ext_m1)
            vec_ext_m2 = np.concatenate(vec_ext_m2)

            logger.debug("cont_vec_m1 shape {}".format(vec_cont_m1.shape))
            logger.debug("cont_vec_m2 shape {}".format(vec_cont_m2.shape))
            logger.debug("path2vec_vec_m1 shape {}".format(vec_ext_m1.shape))
            logger.debug("path2vec_vec_m2 shape {}".format(vec_ext_m2.shape))
            logger.debug("label shape {}".format(label.shape))

            assert vec_cont_m1.shape == vec_cont_m2.shape
            assert vec_ext_m1.shape == vec_ext_m2.shape
            assert vec_cont_m1.shape[0] == vec_cont_m2.shape[0] == vec_ext_m1.shape[0] == vec_ext_m2.shape[
                0] == label.shape[0]

        else:
            if is_cont_emb or self.is_path2vec:
                vec_m1 = np.concatenate(vec_m1)
                vec_m2 = np.concatenate(vec_m2)

            if emb_pp_ana_vec is not None:
                if is_cont_emb or self.is_path2vec:
                    vec_m1 = np.concatenate([vec_m1, vec_em_pp_m1], axis=-1)
                    vec_m2 = np.concatenate([vec_m2, vec_em_pp_m2], axis=-1)

                else:
                    vec_m1 = vec_em_pp_m1
                    vec_m2 = vec_em_pp_m2

            logger.debug("m1 shape {}".format(vec_m1.shape))
            logger.debug("m2 shape {}".format(vec_m2.shape))
            logger.debug("label shape {}".format(label.shape))
            assert vec_m1.shape == vec_m2.shape
            assert vec_m1.shape[0] == vec_m2.shape[0] == label.shape[0]

        return vec_m1, vec_m2,vec_cont_m1, vec_ext_m1, vec_cont_m2, vec_ext_m2, label

    def _get_vec_mention(self,is_cont_emb, cont_vecs, path2vec_vecs,mention_ops,mention,is_emb_pp_ana=False):
        if is_emb_pp_ana:
            assert cont_vecs.shape[0] == len(mention_ops.gold_and_art_anaphors)
            ind = mention_ops.gold_and_art_anaphors.index(mention)
        else:
            if is_cont_emb:
                assert cont_vecs.shape[0] == len(mention_ops.interesting_entities)
            if path2vec_vecs is not None:
                assert self.is_path2vec
                assert path2vec_vecs.shape[0] == len(mention_ops.interesting_entities)
            ind = mention_ops.interesting_entities.index(mention)
        logger.debug("mention {} at {} in the big vector".format(mention,ind))
        vec_c,vec_p = None,None
        if is_cont_emb:
            vec_c = np.expand_dims(cont_vecs[ind], axis=0)
        if path2vec_vecs is not None:
            vec_p = np.expand_dims(path2vec_vecs[ind], axis=0)
        if vec_c is not None and vec_p is not None and not self.is_wsd:
            vec = np.concatenate([vec_c,vec_p],axis=1)
            return vec
        elif vec_c is not None:
            return vec_c
        elif vec_p is not None:
            return vec_p
        else:
            raise NotImplementedError

    def _pad_vectors(self, curr_cand_vecs):
        zero_vec = np.zeros_like(curr_cand_vecs[0])
        while len(curr_cand_vecs) < self.max_ante:
            curr_cand_vecs.append(zero_vec)
        curr_cand_vecs = np.concatenate(curr_cand_vecs)
        logger.debug("current candidate vecs {}".format(curr_cand_vecs.shape))
        assert curr_cand_vecs.shape[0] == self.max_ante
        return curr_cand_vecs

    def _pad_labels(self, cur_labels):
        while len(cur_labels) < self.max_ante:
            cur_labels.append(-1)
        assert len(cur_labels) == self.max_ante
        return cur_labels

    def balance_cand_antecedents_for_ranking(self, cand_antecedents, cur_labels,is_training):
        '''
        if candidate antecedents are more than maximum allowed antecedents, then truncate the list.
        make sure that at least one of the candidate is false antecedent.
        :param cand_antecedents:
        :param cur_labels:
        :return:
        '''
        assert len(cand_antecedents)==len(cur_labels),"candi ante {} and labels {}".format(len(cand_antecedents),len(cur_labels))
        if len(cand_antecedents)>self.max_ante:
            trunc_labels = cur_labels[0:self.max_ante]
            trunc_antecedents = cand_antecedents[0:self.max_ante]
            if 0 not in trunc_labels and is_training:
                neg_ind = cur_labels.index(0)
                trunc_antecedents[-1] = cand_antecedents[neg_ind]
                trunc_labels[-1] = 0
            if 1 not in trunc_labels and is_training:
                pos_ind = cur_labels.index(1)
                trunc_antecedents[-1] = cand_antecedents[pos_ind]
                trunc_labels[-1] = 1
            cand_antecedents,cur_labels = trunc_antecedents,trunc_labels
            if is_training:
                assert 0 in cur_labels and 1 in cur_labels,"cur labels {}".format(cur_labels)
        assert len(cand_antecedents)==len(cur_labels) <= self.max_ante
        return cand_antecedents,cur_labels

    def generate_vecs_for_ranking(self,is_cont_emb,cont_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec,is_training, cand_ante_per_anaphor, cand_ante_per_anaphor_labels, mention_ops,anaphors=None):
        vec_m1, vec_m2,vec_cont_m1, vec_ext_m1, vec_cont_m2, vec_ext_m2,vec_em_pp_m1,vec_em_pp_m2, label = None, None, None, None, None, None, None,None,None

        if self.is_wsd:
            assert is_cont_emb or emb_pp_ana_vec is not None
            vec_cont_m1 = []
            vec_ext_m1 = []

            vec_cont_m2 = []
            vec_ext_m2 = []
        else:
            vec_m1 = []
            vec_m2 = []

        label = []
        if emb_pp_ana_vec is not None:
            vec_em_pp_m1 = []
            vec_em_pp_m2 = []
            assert emb_pp_cand_ant_vec is not None
        if anaphors is None:
            anaphors = mention_ops.anaphors
        assert len(cand_ante_per_anaphor) == len(cand_ante_per_anaphor_labels) == len(anaphors)
        for ana_ind,cand_antecedents in enumerate(cand_ante_per_anaphor):
            curr_ana = anaphors[ana_ind]
            cur_labels = cand_ante_per_anaphor_labels[ana_ind]
            cand_antecedents,cur_labels = self.balance_cand_antecedents_for_ranking(cand_antecedents, cur_labels,is_training)
            assert len(cand_antecedents)==len(cur_labels) <=self.max_ante,"maximum antecedents {} but candidate antecedents {}".format(self.max_ante,len(cand_antecedents))
            if emb_pp_ana_vec is not None:
                vec_em_pp_m1.append(self._get_vec_mention(True, emb_pp_ana_vec, None,
                                                          mention_ops, curr_ana,is_emb_pp_ana=True))
            if self.is_wsd:
                if is_cont_emb:
                    vec_cont_m1.append(self._get_vec_mention(is_cont_emb,cont_vecs, None,
                                                    mention_ops, curr_ana))

                vec_ext_m1.append(self._get_vec_mention(False, None, path2vec_vecs,
                                                         mention_ops, curr_ana))
            else:
                if is_cont_emb or self.is_path2vec:
                    vec_m1.append(self._get_vec_mention(is_cont_emb, cont_vecs, path2vec_vecs,mention_ops,curr_ana))

            curr_cand_vecs = []
            curr_cand_cont_vecs = []
            curr_cand_ext_vecs = []
            curr_cand_em_pp_vecs = []
            for cand_ind,cand_ante in enumerate(cand_antecedents):
                if emb_pp_ana_vec is not None:
                    curr_cand_em_pp_vecs.append(self._get_vec_mention(True,emb_pp_cand_ant_vec, None, mention_ops,
                                              cand_ante))
                if self.is_wsd:
                    if is_cont_emb:
                        curr_cand_cont_vecs.append(self._get_vec_mention(is_cont_emb,cont_vecs, None,
                                                                 mention_ops, cand_ante))
                    curr_cand_ext_vecs.append(self._get_vec_mention(False,None, path2vec_vecs,
                                                            mention_ops, cand_ante))
                else:
                    if is_cont_emb or self.is_path2vec:
                        curr_cand_vecs.append(self._get_vec_mention(is_cont_emb,cont_vecs, path2vec_vecs,mention_ops,cand_ante))

            cur_labels = self._pad_labels(cur_labels)
            if emb_pp_ana_vec is not None:
                curr_cand_em_pp_vecs = self._pad_vectors(curr_cand_em_pp_vecs)
                vec_em_pp_m2.append(np.expand_dims(curr_cand_em_pp_vecs, axis=0))

            if self.is_wsd:
                if is_cont_emb:
                    curr_cand_cont_vecs = self._pad_vectors(curr_cand_cont_vecs)
                    vec_cont_m2.append(np.expand_dims(curr_cand_cont_vecs, axis=0))

                curr_cand_ext_vecs= self._pad_vectors(curr_cand_ext_vecs)
                vec_ext_m2.append(np.expand_dims(curr_cand_ext_vecs, axis=0))

            if (is_cont_emb or self.is_path2vec) and not self.is_wsd:
                curr_cand_vecs = self._pad_vectors(curr_cand_vecs)
                vec_m2.append(np.expand_dims(curr_cand_vecs,axis=0))
            label.append(cur_labels)

        if emb_pp_ana_vec is not None:
            vec_em_pp_m1 = np.concatenate(vec_em_pp_m1)
            vec_em_pp_m2 = np.concatenate(vec_em_pp_m2)

        if self.is_wsd:
            if is_cont_emb:
                vec_cont_m1 = np.concatenate(vec_cont_m1)
                vec_cont_m2 = np.concatenate(vec_cont_m2)

                if emb_pp_ana_vec is not None:
                    vec_cont_m1 = np.concatenate([vec_cont_m1,vec_em_pp_m1],axis=-1)
                    vec_cont_m2 = np.concatenate([vec_cont_m2, vec_em_pp_m2], axis=-1)
            else:
                vec_cont_m1 = vec_em_pp_m1
                vec_cont_m2 = vec_em_pp_m2

            vec_ext_m1 = np.concatenate(vec_ext_m1)
            vec_ext_m2 = np.concatenate(vec_ext_m2)

        else:
            if is_cont_emb or self.is_path2vec:
                vec_m1 = np.concatenate(vec_m1)
                vec_m2 = np.concatenate(vec_m2)

                if emb_pp_ana_vec is not None:
                    vec_m1 = np.concatenate([vec_m1,vec_em_pp_m1],axis=-1)
                    vec_m2 = np.concatenate([vec_m2, vec_em_pp_m2], axis=-1)
            else:
                vec_m1 = vec_em_pp_m1
                vec_m2 = vec_em_pp_m2

        label = np.array(label)

        assert label.shape[1] == self.max_ante
        if self.is_wsd:
            logger.debug("cont_vec_m1 shape {}".format(vec_cont_m1.shape))
            logger.debug("cont_vec_m2 shape {}".format(vec_cont_m2.shape))
            logger.debug("path2vec_vec_m1 shape {}".format(vec_ext_m1.shape))
            logger.debug("path2vec_vec_m2 shape {}".format(vec_ext_m2.shape))
            logger.debug("label shape {}".format(label.shape))

            assert vec_ext_m1.shape[1] == vec_ext_m2.shape[2] ==max_num_sense
            assert vec_ext_m2.shape[1] == self.max_ante
            assert vec_cont_m1.shape[0] == vec_cont_m2.shape[0] == vec_ext_m1.shape[0] == vec_ext_m2.shape[
                0] == label.shape[0]
            assert vec_ext_m1.shape[2] == vec_ext_m2.shape[3] == path2vec_dim

        else:
            logger.debug("m1 shape {}".format(vec_m1.shape))
            logger.debug("m2 shape {}".format(vec_m2.shape))
            logger.debug("label shape {}".format(label.shape))
            assert vec_m2.shape[1] == self.max_ante
            assert vec_m1.shape[0] == vec_m2.shape[0] == label.shape[0]

        return vec_m1, vec_m2,vec_cont_m1, vec_ext_m1, vec_cont_m2, vec_ext_m2, label


    # def generate_vec_data_for_doc(self, single_json_object, is_wsd,is_avg, is_bert, is_path2vec, is_w2vec, is_glove,is_embeddings_pp,
    #                               sen_window_size,is_consider_saleint,is_training,is_ranking_format,is_balanced,is_positive_surr,is_soon,surr_window):
    def generate_vec_data_for_doc(self, single_json_object,sen_window_size,is_consider_saleint, is_training, is_ranking_format):

        """
        for a given json object document, generate BERT or/and path2vec vectors from file
        1. generate mention pairs with window size, use sentence to mention map
        4. add positive pairs depending on the flag set at the class level
        5. return vecs with labels - if bridging 1 otherwise 0
        :param single_json_object:
        :return:
        """
        mention_ops = BridgingJsonDocInformationExtractor(single_json_object, logger)
        if mention_ops.is_file_empty : return True,None, None, None, None, None,None, None, None
        if is_training:
            ana_to_antecdents_map_selected_data = mention_ops.get_ana_to_anecedent_map_from_selected_data(self.is_consider_arti_good_pairs,self.is_consider_arti_bad_pairs,self.is_consider_arti_ugly_pairs,is_clean_art_data=True)
        else:
            ana_to_antecdents_map_selected_data = mention_ops.get_ana_to_anecedent_map_from_selected_data(
                False, False, False,True)
        if self.is_consider_arti_ugly_pairs and is_training:
            print("assertion from #######")
            assert len(mention_ops.anaphors) <= len(ana_to_antecdents_map_selected_data.keys())
            anaphors = list(ana_to_antecdents_map_selected_data.keys())
        else:
            try:
                assert len(mention_ops.anaphors) == len(ana_to_antecdents_map_selected_data.keys()),"anaphors : {} \n ana ante map {}".format(mention_ops.anaphors,ana_to_antecdents_map_selected_data)
            except AssertionError:
                logger.error("----")
                logger.error("ERROR : For doc {} \n anaphors : {} \n and anphor antecedent map : {} \n are not equal ".format(mention_ops.doc_key,mention_ops.anaphors,ana_to_antecdents_map_selected_data))
                logger.error("----")
            anaphors = mention_ops.anaphors
        cand_ante_per_anaphor, context_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec = self.get_cand_ante_per_anaphor_and_vecs(mention_ops, sen_window_size,is_consider_saleint, is_training, ana_to_antecdents_map_selected_data,anaphors)
        assert len(cand_ante_per_anaphor) == len(anaphors)
        cand_ante_per_anaphor_labels = []
        # cand_ante_per_anaphor = [list(set(cand_ante_per_anaphor[i])) for i in range(len(cand_ante_per_anaphor))]
        # for cand_ in cand_ante_per_anaphor:
        #     cand_.sort(reverse=True)
        #
        # cand_ante_per_anaphor = [cand_.sort(reverse=True) for cand_ in cand_ante_per_anaphor]
        for ana_ind,ana in enumerate(anaphors):
            logger.debug(5*"+++++")
            current_ana_cand_ante = cand_ante_per_anaphor[ana_ind]
            logger.debug("for anaphor {}, candidate antecedents are {}".format(ana,current_ana_cand_ante))
            curr_labels = []
            true_antecedents = mention_ops.ana_to_antecedents_map[ana]
            logger.debug("true antecedents {}".format(true_antecedents))
            num_ones = 0
            num_zeroes = 0
            for can_ante in current_ana_cand_ante:
                logger.debug(3*"---")
                logger.debug("current ante {}".format(can_ante))
                if can_ante in true_antecedents:
                    curr_labels.append(1)
                    logger.debug("postive")
                    num_ones +=1
                else:
                    curr_labels.append(0)
                    logger.debug("negtive")
                    num_zeroes +=1

            if is_training:
                assert 1 in curr_labels,"no ones in labels {}".format(curr_labels)
                #TODO : Uncomment
                # if is_ranking_format:
                #     assert 0 in curr_labels, "no zero in labels {}".format(curr_labels)
            cand_ante_per_anaphor_labels.append(curr_labels)

        is_cont_emb = self.is_glove or self.is_w2vec or self.is_bert
        if is_ranking_format:
            m1_vec, m2_vec,m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, label = self.generate_vecs_for_ranking(is_cont_emb,context_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec,is_training,cand_ante_per_anaphor,
                                            cand_ante_per_anaphor_labels, mention_ops,anaphors)
        else:
            m1_vec, m2_vec,m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, label= self.generate_vecs_for_pairs(is_cont_emb, context_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec,cand_ante_per_anaphor,
                                            cand_ante_per_anaphor_labels, mention_ops,anaphors)
        return False,m1_vec, m2_vec, m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, label,cand_ante_per_anaphor

    def concatenate_additional_data(self,curr_data_vecs,additional_data_vecs):
        vecs = []
        for curr_vec,additional_vec in zip(curr_data_vecs,additional_data_vecs):
            if not isinstance(curr_vec,list):
                assert not isinstance(additional_vec,list)
                conc_vec = np.concatenate([curr_vec,additional_vec])
                vecs.append(conc_vec)
            else:
                vecs.append([])
        return vecs

    def latent_svm_compliatnt_data(self,m1_ext_vec,m2_ext_vec, label,is_ranking_format):
        def only_one_sense(x):
            if len(x.shape) == 4:
                x[:, :, 1:5, :] = -1
            else:
                x[:, 1:5, :] = -1
            return x

        num_senses = m1_ext_vec.shape[-2]
        vec_dim = m1_ext_vec.shape[-1]
        _zero_minus_1 = np.concatenate([np.zeros((1, 1, vec_dim)), np.ones((1, num_senses-1, vec_dim)) * -1], axis=1)
        zeros = np.zeros((1, num_senses, vec_dim))

        replace_zeroes_with_minus_one = lambda x: np.where(x == zeros, _zero_minus_1, x)

        m1_ext_vec = replace_zeroes_with_minus_one(m1_ext_vec)
        m2_ext_vec = replace_zeroes_with_minus_one(m2_ext_vec)

        m1_ext_vec = only_one_sense(m1_ext_vec)
        m2_ext_vec = only_one_sense(m2_ext_vec)

        if len(m2_ext_vec.shape)==4:
            assert np.all(np.any(np.any(m2_ext_vec != -1, -1), -1))
        if not is_ranking_format:
            label = np.where(label==0, -1, label)

        return m1_ext_vec,m2_ext_vec, label

    def generate_vecs_for_dataset(self, jsonlines,is_training,sen_window_size,is_consider_saleint=False,is_ranking_format=False,additional_data_vecs = None,is_latent_svm=False):
        """
        read jsonlines and for each json object create vectors
        :param jsonlines:
        :return:
        """
        assert self.is_bert or self.is_path2vec or self.is_glove or self.is_w2vec or self.is_emb_pp
        m1_vec, m2_vec,m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, label = [], [],[], [], [], [], []
        json_objects = self.get_json_objects(jsonlines)
        if is_ranking_format:
            assert not (self.is_positive_surr or self.is_soon)
        for single_json_object in json_objects:
            is_empty,_m1_vec, _m2_vec,_m1_cont_vec, _m1_ext_vec, _m2_cont_vec, _m2_ext_vec, _label,_ = self.generate_vec_data_for_doc(
                single_json_object,sen_window_size,is_consider_saleint, is_training, is_ranking_format)

            if not is_empty:
                assert (self.is_wsd and _m1_cont_vec is not None and _m1_vec is None) or (not self.is_wsd and _m1_cont_vec is None and _m1_vec is not None)

            if _m1_cont_vec is not None:
                m1_cont_vec.append(_m1_cont_vec)
                m1_ext_vec.append(_m1_ext_vec)

                m2_cont_vec.append(_m2_cont_vec)
                m2_ext_vec.append(_m2_ext_vec)

                label.append(_label)
            if _m1_vec is not None:
                m1_vec.append(_m1_vec)
                m2_vec.append(_m2_vec)
                label.append(_label)

        label = np.concatenate(label)
        if self.is_wsd:
            m1_cont_vec = np.concatenate(m1_cont_vec)
            m1_ext_vec = np.concatenate(m1_ext_vec)

            m2_cont_vec = np.concatenate(m2_cont_vec)
            m2_ext_vec = np.concatenate(m2_ext_vec)

            # m1_cont_vec = np.random.randn(*m1_cont_vec.shape)
            # m1_ext_vec = np.random.randn(*m1_ext_vec.shape)
            #
            # m2_cont_vec = np.random.randn(*m2_cont_vec.shape)
            # m2_ext_vec = np.random.randn(*m2_ext_vec.shape)
            # num_samples = 30
            # m1_cont_vec = m1_cont_vec[0:num_samples]
            # m1_ext_vec = m1_ext_vec[0:num_samples]
            #
            # m2_cont_vec = m2_cont_vec[0:num_samples]
            # m2_ext_vec = m2_ext_vec[0:num_samples]
            #
            # label = label[0:num_samples]

            logger.debug("m1 cont shape {}".format(m1_cont_vec.shape))
            logger.debug("m1 ext shape {}".format(m1_ext_vec.shape))
            logger.debug("m2 cont shape {}".format(m2_cont_vec.shape))
            logger.debug("m2 ext shape {}".format(m2_ext_vec.shape))
            logger.debug("label shape {}".format(label.shape))

            print("m1 cont shape {}".format(m1_cont_vec.shape))
            print("m1 ext shape {}".format(m1_ext_vec.shape))
            print("m2 cont shape {}".format(m2_cont_vec.shape))
            print("m2 ext shape {}".format(m2_ext_vec.shape))
            print("label shape {}".format(label.shape))

            if is_ranking_format:
                # assert len(m1_cont_vec.shape) == 2
                # assert len(m1_ext_vec.shape) == 3
                #
                # assert len(m2_cont_vec.shape) == 3
                # assert len(m2_ext_vec.shape) == 4

                assert m1_ext_vec.shape[1] == m2_ext_vec.shape[2]==max_num_sense
                assert m2_ext_vec.shape[1] == self.max_ante

                assert m1_ext_vec.shape[2] == m2_ext_vec.shape[3]

                assert m1_cont_vec.shape[0]==m1_ext_vec.shape[0]==m2_cont_vec.shape[0]==m2_ext_vec.shape[0]

            else:
                assert m1_ext_vec.shape == m2_ext_vec.shape
                assert m1_cont_vec.shape == m2_cont_vec.shape

            assert m1_cont_vec.shape[0] == label.shape[0]

            # write_pickle("./m1_cont_vec.pkl",m1_cont_vec)
            # write_pickle("./m1_ext_vec.pkl", m1_ext_vec)
            # write_pickle("./m2_cont_vec.pkl",m2_cont_vec)
            # write_pickle("./m2_ext_vec.pkl", m2_ext_vec)

            label = label.reshape(m1_cont_vec.shape[0], -1)

        else:

            m1_vec = np.concatenate(m1_vec)
            m2_vec = np.concatenate(m2_vec)

            # for i in range(m1_vec.shape[0]):
            #     logger.debug("m1 : {} : {}".format(i,np.all(m1_vec[i]==0)))
            #     logger.debug("m2 : {} : {}".format(i, np.all(m2_vec[i] == 0)))

            # m1_vec = np.random.randn(*m1_vec.shape)
            # m2_vec = np.random.randn(*m2_vec.shape)
            label = label.reshape(m1_vec.shape[0], -1)

            logger.debug("m1 shape {}".format(m1_vec.shape))
            logger.debug("m2 shape {}".format(m2_vec.shape))
            logger.debug("label shape {}".format(label.shape))

            print("m1 shape {}".format(m1_vec.shape))
            print("m2 shape {}".format(m2_vec.shape))
            print("label shape {}".format(label.shape))

            # write_pickle("./m1_vec.pkl",m1_vec)
            # write_pickle("./m2_vec.pkl", m2_vec)
            #
            # sys.exit()
            if is_ranking_format:
                m2_vec.shape[1] == self.max_ante
            else:
                assert m1_vec.shape == m2_vec.shape
            assert m1_vec.shape[0] == m2_vec.shape[0] == label.shape[0]
            if is_training and self.is_balanced and not is_ranking_format:
                m1_vec, m2_vec,label = down_sample(label,m1_vec,m2_vec)
                print("m1 shape {}".format(m1_vec.shape))
                print("m2 shape {}".format(m2_vec.shape))
                print("label shape {}".format(label.shape))
        if additional_data_vecs is not None:
            curr_data_vecs = m1_vec, m2_vec, m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, label
            m1_vec, m2_vec, m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, label = self.concatenate_additional_data(curr_data_vecs,additional_data_vecs)
        if is_latent_svm:
            m1_ext_vec,m2_ext_vec, label = self.latent_svm_compliatnt_data(m1_ext_vec,m2_ext_vec, label,is_ranking_format)
        return m1_vec, m2_vec, m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, label

    def write_svm_ranking_dat_file(self,x,y,dat_file,is_ignore_dummy):
        if y is not None:
            assert x.shape[0] == y.shape[0]
            assert x.shape[1] == y.shape[1]
            assert x.shape[2] >= 0

            lines = []
            for i in range(x.shape[0]):
                qid = i+1
                for j in range(x.shape[1]):
                    rank = int(y[i][j])
                    # print(rank)
                    if rank == -1 and is_ignore_dummy: break
                    if is_ignore_dummy:
                        assert rank == 1 or rank == 0,"rank is {}".format(rank)
                    else:
                        assert rank == -1 or rank == 1 or rank == 0, "rank is {}".format(rank)
                    line = "{} qid:{}".format(rank,qid)
                    for k in range(x.shape[2]):
                        line += " {}:{}".format(k+1,x[i][j][k])
                    lines.append(line)
                    # logger.debug(line)
            if not is_ignore_dummy:
                assert len(lines) % self.max_ante == 0
                assert len(lines) == x.shape[0]* x.shape[1]

        else:
            lines = []
            for i in range(x.shape[0]):
                qid = i+1
                line = "{} qid:{}".format(0, qid)
                for j in range(x.shape[1]):
                    line += " {}:{}".format(j+1,x[i][j])
                lines.append(line)
                # print(line)

        write_text_file(dat_file,lines)
        return None,None


    def generate_svm_data(self, jsonlines,is_classification, is_training, dat, sen_window_size, is_consider_saleint=True, is_concat=False, is_ignore_dummy=True, is_return_vecs=False, additional_data=None, is_fourier_features=False, is_fit_features=False, is_normalize=False):
        assert not (is_fourier_features and is_normalize)
        anaphors_vec_train, antecedents_vec_train, anaphors_cont_vec_train, anaphors_ext_vec_train, antecedents_cont_vec_train, antecedents_ext_vec_train, bridging_labels_vec_train = self.generate_vecs_for_dataset(
            jsonlines,sen_window_size=sen_window_size,
            is_training=is_training,
            is_ranking_format=not is_classification,
            is_consider_saleint=is_consider_saleint)
        print("labels {}".format(bridging_labels_vec_train.shape))

        if not is_classification:
            arr = anaphors_vec_train.reshape(anaphors_vec_train.shape[0], 1, anaphors_vec_train.shape[1])
            anaphors_vec_train = np.repeat(arr, self.max_ante, axis=1)
        if is_concat:
            X = np.concatenate([anaphors_vec_train, antecedents_vec_train], axis=-1)
        else:
            X = np.multiply(anaphors_vec_train, antecedents_vec_train)
        print("x shape {}".format(X.shape))
        print("y shape {}".format(bridging_labels_vec_train.shape))

        if is_normalize or is_fourier_features:
            for i in range(5):
                print(X[i][0][0:5])

        if is_normalize:
            X = normalize(X)
            print("after normalizing")
            for i in range(5):
                print(X[i][0][0:5])

        if is_fourier_features:
            # irff = IRFF()
            if is_fit_features:
                self.irff.fit(X)
                print("data fitted")
            # X = self.irff.compute_kernel(X)
            X = self.irff.transform(X)

            print("after rff transformation")
            for i in range(5):
                print(X[i][0][0:5])

        if additional_data is not None:
            X_add,labels_add = additional_data
            print("additional data : x shape {}, y shape {}".format(X_add.shape,labels_add.shape))
            X = np.concatenate([X,X_add])
            bridging_labels_vec_train = np.concatenate([bridging_labels_vec_train,labels_add])
            print("after adding data : x shape {}, y shape {}".format(X.shape,bridging_labels_vec_train.shape))

        if is_return_vecs or is_classification:
            return X,bridging_labels_vec_train
        else:
            return self.write_svm_ranking_dat_file(X,bridging_labels_vec_train,dat,is_ignore_dummy=is_ignore_dummy)


    def get_data_for_autoencoder(self,jsonlines,sen_window_size,additional_data_vecs = None):
        anaphors_vec, antecedents_vec, _, _, _, _, _ = self.generate_vecs_for_dataset(
            jsonlines, is_training=True,is_consider_saleint=True,is_ranking_format= False,
            sen_window_size=sen_window_size,additional_data_vecs=additional_data_vecs)

        assert anaphors_vec.shape == antecedents_vec.shape
        autoencoder_input = np.concatenate([anaphors_vec, antecedents_vec])
        print("input to autoencoder shape {}".format(autoencoder_input.shape))
        return autoencoder_input






if __name__ == '__main__':
    pass
    # # is_bert = True
    # # is_path2vec = True
    #
    # gvd = GenerateVecData(json_objects_path=is_notes_data_path, vec_path=is_notes_vec_path)
    # gvd.generate_vecs_for_all_docs(is_notes_jsonlines)
    # # gvd.generate_sem_heads_files(is_notes_jsonlines)
    #
    # gvd = GenerateVecData(json_objects_path=bashi_data_path, vec_path=bashi_vec_path)
    # gvd.generate_vecs_for_all_docs(bashi_jsonlines,is_bashi=True)
    # # gvd.generate_sem_heads_files(bashi_jsonlines)
    #
    # # bd = BridgingData(json_objects_path=is_notes_data_path, vec_path=is_notes_vec_path)
    # # bd.generate_vecs_for_dataset(is_notes_jsonlines,is_training=True, is_wsd=False, is_bert=is_bert, is_path2vec=is_path2vec,
    # #                              sen_window_size=sen_window_size,is_ranking_format=False)
