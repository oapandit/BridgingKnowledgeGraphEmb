from bridging_json_extract import BridgingJsonDocInformationExtractor
import numpy as np
from bridging_utils import *
from word_embeddings import Path2VecEmbeddings
from nltk.probability import FreqDist
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import SpectralEmbedding
from word_embeddings import Word2vecEmbeddings
from word_embeddings import GloveEmbeddings
from word_embeddings import Path2VecEmbeddings
from word_embeddings import BERTEmbeddings
from word_embeddings import EmbeddingsPP
from data_generation import BridgingData

vec_path = os.path.join(is_notes_data_path, "is_notes_tsv_vecs")
create_dir(vec_path)

bert_vec_extn = "bert.pkl"
w2v_extn = "300d_w2vec.pkl"
glove_vec_extn = "300d_glove_vec.pkl"
emb_pp_vec_extn = "embeddings_pp_vec.pkl"
p2v_lesk_vec_extn = "path2vec_lesk_vec.pkl"
p2v_avg_vec_extn = "path2vec_avg_vec.pkl"

class NounPhrase:
    def __init__(self,id,index,words,sem_head,synt_head,doc_name,sent_num,is_anaphor,antecedents):
        self.id = id
        self.index = index
        self.words = words
        self.sem_head = sem_head
        self.synt_head = synt_head
        self.doc_name = doc_name
        self.sent_num = sent_num
        self.is_anaphor = is_anaphor
        self.antecedents = antecedents

    def map_antecedents_to_id(self,index_to_id_map):
        ant_ids = []
        for ant in self.antecedents:
            if index_to_id_map.get(ant,None) is not None:
                id = index_to_id_map[ant]
                if id < self.id:
                    ant_ids.append(id)
                    if id > 1000:
                        print(self.doc_name,self.antecedents)
        ant_ids.sort()
        self.antecedents = ant_ids

    def get_info(self):
        return self.doc_name,self.id,self.words,self.sem_head,self.synt_head,self.sent_num,self.is_anaphor,self.antecedents



class AnalyzeDoc:
    def __init__(self, json_object=None):
        if json_object is not None:
            self.json_info = BridgingJsonDocInformationExtractor(json_object, logger)
            self.is_file_empty = False
            if self.json_info.is_file_empty:
                self.is_file_empty = True
                return None
            self.json_object = json_object
            self.total_ana = 0
            self.total_ana_ant_in_cand = 0
            self.total_cand_antecedents = 0
            self.num_sense_vectors = []
            self.total_spans =0
            self.ext_info_with_word = 0
            self.ext_info_with_head_word = 0
            self.anaphors = self.json_info.get_span_words(self.json_info.anaphors)
            self.sem_head_ana = []
            self.synt_head_ana = []

    def _gen_data(self,doc_key,entities_words,entities_words_clean,entities_sents,entities_context,vec_extn,is_anaphor):
        self.bert_embeddings.generate_bert_emb_for_mentions_from_sentences(entities_sents, doc_key + vec_extn + bert_vec_extn)
        self.word2vec_embeddings.generate_emb_file_for_word_list(entities_words, doc_key + vec_extn + w2v_extn)
        self.glove_embeddings.generate_emb_file_for_word_list(entities_words,doc_key + vec_extn + glove_vec_extn)
        self.embeddings_pp_sem_head.generate_emb_file_for_word_list(entities_words,
                                                                    doc_key + vec_extn + emb_pp_vec_extn,
                                                                    is_force=True, is_anaphor=is_anaphor)
        self.path2vec_embeddings.generate_emb_file_for_word_list(entities_words_clean,
                                                                 doc_key + vec_extn + p2v_lesk_vec_extn,
                                                                 entities_words,entities_context)
        self.path2vec_embeddings.generate_emb_file_for_word_list(entities_words_clean,
                                                                 doc_key + vec_extn + p2v_avg_vec_extn,
                                                                 entities_words,is_avg=True)

    def cand_ante_heads(self):
        self.cand_antecedents = self.json_info.get_span_words(self.json_info.noun_phrase_list)
        self.sem_head_cand_ante, sem_head_spans = self.json_info.get_spans_head(self.json_info.noun_phrase_list, True)
        self.synt_head_cand_ante, _ = self.json_info.get_spans_head(self.json_info.noun_phrase_list, False)
        self.doc_name_cand_ante = len(self.synt_head_cand_ante) * [self.json_info.doc_key]
        self.cand_ante_sents_number = self.json_info.get_span_to_sentence_list(self.json_info.noun_phrase_list)
        entities_sents = self.json_info.get_span_sentences(sem_head_spans)
        entities_words_clean = self.json_info.remove_pronouns_and_clean_words(self.cand_antecedents)
        entities_context = self.json_info.get_surrounding_sentences_for_span_list(self.json_info.noun_phrase_list,
                                                                                 l_context=2, r_context=1)

        assert len(self.sem_head_cand_ante) == len(self.synt_head_cand_ante) == len(self.doc_name_cand_ante) == len(
            self.cand_antecedents) == len(self.cand_ante_sents_number) ==len(entities_sents) == len(entities_words_clean) == len(entities_context)

        # self._gen_data(self.mention_ops.doc_key,self.sem_head_cand_ante,entities_words_clean,entities_sents,entities_context,"_noun_phrases_sem_head_",False)


    def anaphor_head_words(self):
        self.present = []
        self.cand_ante_words_per_ana = []
        self.cand_ante_sem_head_words_per_ana = []
        self.true_ante_words_per_ana = []
        self.true_ante_sem_head_words_per_ana = []
        anaphors_sem_head_words, sem_head_spans = self.json_info.get_spans_head(self.json_info.anaphors, True)
        anaphors_collins_head_words, _ = self.json_info.get_spans_head(self.json_info.anaphors, False)
        self.sem_head_ana = anaphors_sem_head_words
        self.synt_head_ana = anaphors_collins_head_words
        self.doc_name = len(anaphors_collins_head_words)*[self.json_info.doc_key]
        self.diff = 0
        self.anaphors_sents_number = self.json_info.get_span_to_sentence_list(self.json_info.anaphors)
        assert len(self.sem_head_ana) == len(self.synt_head_ana) == len(self.doc_name) == len(self.anaphors) == len(self.anaphors_sents_number)
        for an in self.json_info.anaphors:
            cand_ante = self.json_info.get_cand_antecedents(an,2,is_salient=True)
            for ca in cand_ante:
                assert ca in self.json_info.noun_phrase_list,"ca {} and nps {}".format(ca,self.json_info.noun_phrase_list)
            cand_ante_words = self.json_info.get_span_words(cand_ante)
            cand_ante_sem_head_words, _ = self.json_info.get_spans_head(cand_ante, True)
            assert len(cand_ante) == len(cand_ante_words) == len(cand_ante_sem_head_words)
            true_antecedents = self.json_info.ana_to_antecedents_map[an]
            true_ante_words = self.json_info.get_span_words(true_antecedents)
            true_ante_sem_head_words, _ = self.json_info.get_spans_head(true_antecedents, True)
            assert len(true_ante_sem_head_words) == len(true_antecedents) == len(true_ante_words)
            no_found = True
            for ant in true_antecedents:
                if ant in cand_ante:
                    no_found = False
                    break
            if no_found:
                self.present.append("NO")
            else:
                self.present.append("YES")
            self.cand_ante_words_per_ana.append(cand_ante_words)
            self.cand_ante_sem_head_words_per_ana.append(cand_ante_sem_head_words)
            self.true_ante_words_per_ana.append(true_ante_words)
            self.true_ante_sem_head_words_per_ana.append(true_ante_sem_head_words)

        entities_sents = self.json_info.get_span_sentences(sem_head_spans)
        entities_words_clean = self.json_info.remove_pronouns_and_clean_words(self.anaphors)
        entities_context = self.json_info.get_surrounding_sentences_for_span_list(self.json_info.anaphors,
                                                                                 l_context=2, r_context=1)
        assert len(self.cand_ante_words_per_ana) == len(self.true_ante_words_per_ana) == len(self.doc_name) == len(self.anaphors) == len(
            self.present)==len(entities_sents) == len(entities_words_clean) == len(entities_context) \
               ==len(self.cand_ante_sem_head_words_per_ana) == len(self.true_ante_sem_head_words_per_ana)


        # self._gen_data(self.mention_ops.doc_key, self.sem_head_ana, entities_words_clean,entities_sents,entities_context, "_anaphors_sem_head_", True)

    def _get_fnames(self,doc_key,vec_extn):
        bert_file = doc_key + vec_extn + bert_vec_extn
        w2v_file = doc_key + vec_extn + w2v_extn
        glove_file = doc_key + vec_extn + glove_vec_extn
        embeddings_pp_file = doc_key + vec_extn + emb_pp_vec_extn
        p2v_lesk_file = doc_key + vec_extn + p2v_lesk_vec_extn
        p2v_avg_file =  doc_key + vec_extn + p2v_avg_vec_extn
        return [bert_file,w2v_file,glove_file,p2v_lesk_file,p2v_avg_file,embeddings_pp_file]

    def _get_same_words_ind_cluster(self,list_of_words):
        word_to_ind = {}
        for ind,w in enumerate(list_of_words):
            cur_word_ind_list = word_to_ind.get(w,[])
            cur_word_ind_list.append(ind)
            word_to_ind[w] = cur_word_ind_list
        #test
        for w in word_to_ind.keys():
            indices = [i for i, x in enumerate(list_of_words) if x == w]
            assert sorted(indices) == sorted(word_to_ind[w])
        return word_to_ind

    def test_files_dim(self):
        vec_dims = [768,300,300,100,300,300]
        doc_key = self.json_info.doc_key
        vec_extns = ["_anaphors_sem_head_","_noun_phrases_sem_head_"]
        num_ana = len(self.json_info.anaphors)
        num_np = len(self.json_info.noun_phrase_list)
        num_words = [num_ana,num_np]
        sem_head_noun_phrase, _ = self.json_info.get_spans_head(self.json_info.noun_phrase_list, True)
        word_to_ind = self._get_same_words_ind_cluster(sem_head_noun_phrase)
        for vec_extn,num_word in zip(vec_extns,num_words):
            files = self._get_fnames(doc_key,vec_extn)
            for f,vec_dim in zip(files,vec_dims):
                vec = read_pickle(os.path.join(vec_path,f))
                vec.shape[0] == num_word,"numb words {} != vec shape {}".format(num_word,vec.shape[0])
                vec.shape[1] == vec_dim, "vec dim {} != vec shape {}".format(vec_dim, vec.shape[1])
                if vec_extn == "_noun_phrases_sem_head_" and not (vec_dim == 768 or "path2vec" in f):
                    for w in word_to_ind.keys():
                        if len(word_to_ind[w])>1:
                            ref_ind = word_to_ind[w][0]
                            for ind in word_to_ind[w]:
                                try:
                                    assert np.array_equal(vec[ind],vec[ref_ind]),"array not equal for word {}, indices are {} and {},vecs are {} , {}".format(w,ind,ref_ind,vec[ind][0:5],vec[ref_ind][0:5])
                                except:
                                    print("for {} vectors are not same".format(w))


    def get_noun_phrases(self,data):
        # for ana in self.mention_ops.anaphors:
        #     try:
        #         assert ana in self.mention_ops.noun_phrase_list,"{} not in {}, words : {}".format(ana,self.mention_ops.noun_phrase_list,self.mention_ops.get_span_words([ana])[0])
        #     except AssertionError:
        #         logger.critical("doc : {}".format(self.mention_ops.doc_key))
        #         logger.critical("anaphor - index : {}, word : {}".format(ana,self.mention_ops.get_span_words([ana])[0]))
        #         logger.critical("parse : {}".format(self.mention_ops.parse_bit[ana[0]]))
        #         logger.critical("noun phrases : {}".format(self.mention_ops.noun_phrase_list))
        #         logger.critical(2*"----")
        np_ana = list(set(self.json_info.noun_phrase_list + self.json_info.anaphors))
        np_ana.sort()
        logger.debug(np_ana)
        for ana in self.json_info.anaphors:
            assert ana in np_ana,"{} not in {}, words : {}".format(ana,self.json_info.noun_phrase_list,self.json_info.get_span_words([ana])[0])

        self.noun_phrases_words = self.json_info.get_span_words(np_ana)
        self.sem_head_np, _ = self.json_info.get_spans_head(np_ana, True)
        self.synt_head_np, _ = self.json_info.get_spans_head(np_ana, False)
        self.np_sents_number = self.json_info.get_span_to_sentence_list(np_ana)

        self.np_ana_markables_list = []
        num_anaphor = 0
        index_to_id_map = {}
        is_anaphor_list = []
        for id,index in enumerate(np_ana):
            words = self.noun_phrases_words[id]
            sem_head = self.sem_head_np[id]
            synt_head = self.synt_head_np[id]
            sent_num = self.np_sents_number[id]
            is_anaphor = False
            antecedents = ""
            if index in self.json_info.anaphors:
                num_anaphor += 1
                is_anaphor = True
                antecedents = self.json_info.ana_to_antecedents_map[index]

            is_anaphor_list.append(is_anaphor)
            np_ana_markable = NounPhrase(id,index,words,sem_head,synt_head,self.json_info.doc_key,sent_num,is_anaphor,antecedents)
            self.np_ana_markables_list.append(np_ana_markable)
            index_to_id_map[index] = id

        for np_ana_markable in self.np_ana_markables_list:
            np_ana_markable.map_antecedents_to_id(index_to_id_map)

        bert_vec, w2vec_vec, glove_vec, path2vec_avg, path2vec_lesk, emb_pp_ana_vec, emb_pp_cand_ant_vec = data.read_all_vecs(self.json_info)
        interesting_ent_vecs = [bert_vec, w2vec_vec, glove_vec, path2vec_avg, path2vec_lesk]
        vec_shapes = [768,300,300,300,300]
        vec_extn = "_"
        files = self._get_fnames(self.json_info.doc_key,vec_extn)

        for j,f in enumerate(files[:-1]):
            vecs = data.get_vecs_for_mentions(self.json_info,np_ana,interesting_ent_vecs[j])
            assert vecs.shape[0] == len(np_ana)
            assert vecs.shape[1] == vec_shapes[j]
            write_pickle(os.path.join(vec_path,f),vecs)

        vecs = data.get_vecs_for_mentions(self.json_info, np_ana, emb_pp_cand_ant_vec,emb_pp_ana_vec,is_anaphor_list)
        assert vecs.shape[0] == len(np_ana)
        assert vecs.shape[1] == 100
        write_pickle(os.path.join(vec_path, files[-1]), vecs)


    def test_pickled_np(self,doc_key,num_word):
        vec_dims = [768,300,300,300,300,100]
        vec_extn = "_"
        files = self._get_fnames(doc_key,vec_extn)
        for f, vec_dim in zip(files, vec_dims):
            vec = read_pickle(os.path.join(vec_path, f))
            assert vec.shape[0] == num_word, "numb words {} != vec shape {}".format(num_word, vec.shape[0])
            assert vec.shape[1] == vec_dim, "vec dim {} != vec shape {}".format(vec_dim, vec.shape[1])
            print("shape : {}".format(vec.shape))
        print("----")

class AnalyzeCorpus:
    def __init__(self, json_objects_path, jsonlines):
        self.json_objects_path = json_objects_path
        json_path = os.path.join(self.json_objects_path, jsonlines)
        self.json_objects = read_jsonlines_file(json_path)
        self.bert_embeddings = BERTEmbeddings(vec_path, logger)
        self.word2vec_embeddings = Word2vecEmbeddings(300,vec_path, logger)
        self.glove_embeddings = GloveEmbeddings(300,vec_path, logger)
        self.path2vec_embeddings = Path2VecEmbeddings(vec_path, logger)
        self.embeddings_pp_sem_head = EmbeddingsPP(100, vec_path, logger)
        self.data = BridgingData(json_objects_path=is_notes_data_path, vec_path=is_notes_vec_path,
                                 is_read_head_vecs=True, is_sem_head=True,
                                 is_all_words=False)
        self.data.word2vec_dim = 300
        self.data.glove_vec_dim = 300

    def analyze_heads_anaphors(self):
        sem_head_ana = []
        synt_head_ana = []
        doc_name = []
        anaphors = []
        diff = 0
        sent_nums = []
        present = []
        cand_ante_words_per_ana = []
        true_ante_words_per_ana = []
        cand_ante_sem_head_words_per_ana = []
        true_ante_sem_head_words_per_ana = []

        sem_head_cand_ante = []
        synt_head_cand_ante = []
        doc_name_cand_ante = []
        cand_ante_sents_number = []
        cand_antecedents = []

        for single_json_object in self.json_objects:
            doc = AnalyzeDoc(single_json_object)
            if doc.is_file_empty: continue

            doc.bert_embeddings = self.bert_embeddings
            doc.word2vec_embeddings = self.word2vec_embeddings
            doc.glove_embeddings = self.glove_embeddings
            doc.path2vec_embeddings = self.path2vec_embeddings
            doc.embeddings_pp_sem_head = self.embeddings_pp_sem_head

            doc.anaphor_head_words()

            sem_head_ana += doc.sem_head_ana
            synt_head_ana += doc.synt_head_ana
            doc_name += doc.doc_name
            anaphors += doc.anaphors
            sent_nums += doc.anaphors_sents_number
            present += doc.present
            cand_ante_words_per_ana += doc.cand_ante_words_per_ana
            cand_ante_sem_head_words_per_ana += doc.cand_ante_sem_head_words_per_ana
            true_ante_sem_head_words_per_ana += doc.true_ante_sem_head_words_per_ana
            true_ante_words_per_ana += doc.true_ante_words_per_ana

            doc.cand_ante_heads()

            cand_antecedents += doc.cand_antecedents
            sem_head_cand_ante += doc.sem_head_cand_ante
            synt_head_cand_ante += doc.synt_head_cand_ante
            doc_name_cand_ante += doc.doc_name_cand_ante
            cand_ante_sents_number += doc.cand_ante_sents_number

            for cand_ante_head in doc.cand_ante_sem_head_words_per_ana:
                for h in cand_ante_head:
                    assert h in doc.sem_head_cand_ante,"{} not in {}".format(h,doc.sem_head_cand_ante)

        assert len(cand_antecedents) == len(sem_head_cand_ante) == len(synt_head_cand_ante) == len(doc_name_cand_ante)== len(cand_ante_sents_number)
        assert len(sem_head_ana) == len(synt_head_ana) == len(doc_name) == len(anaphors) == len(sent_nums) \
               == len(present) == len(cand_ante_words_per_ana) == len(true_ante_words_per_ana) \
               == len(cand_ante_sem_head_words_per_ana) == len(true_ante_sem_head_words_per_ana)


        tsv_file_name = os.path.join(vec_path, "anaphors_info.tsv")
        tsv_file = open(tsv_file_name, mode='w')
        tsv_writer = csv.writer(tsv_file,delimiter="\t")
        tsv_writer.writerow(["Doc Name","Anaphor","Semantic Head", "Syntactic Head","Sentence Number","Candidate Antecedents","Candidate Antecedents Sem Heads","True Antecedents","True Antecedents Sem Heads","Is_present"])
        for dn,ana,sem,synt,sn,cand_ant,cand_ant_head,true_ant,true_ant_head,p in zip(doc_name,anaphors,sem_head_ana,synt_head_ana,sent_nums,cand_ante_words_per_ana,cand_ante_sem_head_words_per_ana,true_ante_words_per_ana,true_ante_sem_head_words_per_ana,present):
            cand_ants = "],[".join(cand_ant)
            cand_ants = "[["+cand_ants+"]]"

            cand_ants_head = "],[".join(cand_ant_head)
            cand_ants_head = "[["+cand_ants_head+"]]"

            true_ants = "],[".join(true_ant)
            true_ants = "[["+true_ants+"]]"

            true_ants_head = "],[".join(true_ant_head)
            true_ants_head = "[["+true_ants_head+"]]"

            tsv_writer.writerow([dn,ana,sem,synt,sn,cand_ants,cand_ants_head,true_ants,true_ants_head,p])
        logger.debug("tsv file creation completed.")
        tsv_file.close()

        tsv_file_name = os.path.join(vec_path, "noun_phrases_info.tsv")
        tsv_file = open(tsv_file_name, mode='w')
        tsv_writer = csv.writer(tsv_file, delimiter="\t")
        tsv_writer.writerow(
            ["Doc Name", "Noun Phrases", "Semantic Head", "Syntactic Head", "Sentence Number"])
        for dn, cand_ant, sem, synt, sn in zip(doc_name_cand_ante, cand_antecedents, sem_head_cand_ante, synt_head_cand_ante,
                                                                 cand_ante_sents_number):

            tsv_writer.writerow([dn, cand_ant, sem, synt, sn])
        logger.debug("tsv file creation completed.")
        tsv_file.close()

    def test_files(self):
        for single_json_object in self.json_objects:
            doc = AnalyzeDoc(single_json_object)
            if doc.is_file_empty: continue
            doc.test_files_dim()

    # def test_tsv(self):
    #     ana_tsv = os.path.join(vec_path, "anaphors_info.tsv")
    #     np_tsv = os.path.join(vec_path, "noun_phrases_info.tsv")
    #     ana_tsv_cont = read_tsv(ana_tsv)
    #     assert len(ana_tsv_cont) == 664


    def generate_single_tsv_and_vecs(self):
        np_ana_markables_list = []
        for single_json_object in self.json_objects:
            doc = AnalyzeDoc(single_json_object)
            if doc.is_file_empty: continue
            doc.get_noun_phrases(self.data)
            np_ana_markables_list += doc.np_ana_markables_list

        tsv_file_name = os.path.join(vec_path, "ana_np_info.tsv")
        tsv_file = open(tsv_file_name, mode='w')
        tsv_writer = csv.writer(tsv_file,delimiter="\t")
        tsv_writer.writerow(["Doc Name","ID","Noun Phrase","Semantic Head", "Syntactic Head","Sentence Number","Is Anaphor","Antecedents"])
        num_ana= 0
        num_ana_without_ant = 0
        for np_ana_markable in np_ana_markables_list:
            doc_name,id,words, sem_head, synt_head, sent_num, is_anaphor, antecedents = np_ana_markable.get_info()
            if is_anaphor:
                num_ana += 1
                if len(antecedents) == 0:
                    num_ana_without_ant += 1
                logger.debug("ante {}".format(antecedents))
            antecedents = "NON_NP_ANT" if len(antecedents) == 0 and is_anaphor else ",".join(map(str, antecedents))
            if is_anaphor:
                logger.debug("ante {}".format(antecedents))
                logger.debug("======")
            is_anaphor = "YES" if is_anaphor else "NO"
            tsv_writer.writerow([doc_name,id,words, sem_head, synt_head, sent_num, is_anaphor, antecedents])

        logger.debug("tsv file creation completed.")
        tsv_file.close()
        logger.debug("number of total anaphors {} and antecedents not found {}, {}%".format(num_ana,num_ana_without_ant,num_ana_without_ant*100.0/num_ana))
        assert num_ana == 663

    def test_tsv(self):
        ana_tsv = os.path.join(vec_path, "ana_np_info.tsv")
        ana_tsv_cont = read_tsv(ana_tsv)
        curr_doc = ana_tsv_cont[1][0]
        curr_nps = 0
        doc = AnalyzeDoc()
        test = 0
        total_words = 0
        for num,line in enumerate(ana_tsv_cont[1:]):
            # print(line)
            # if num == 5:
            #     break
            doc_name, id, words, sem_head, synt_head, sent_num, is_anaphor, antecedents = line
            if is_anaphor == "YES":
                id_ = int(id)
                if antecedents != "NON_NP_ANT":
                    ants_list = antecedents.split(',')
                    for ant in ants_list:
                        assert int(ant)<id_,"id : {}, antecedents :{}".format(id_,antecedents)
            if curr_doc != line[0] or num == len(ana_tsv_cont)-2:
                print(num)
                print("test : {}".format(curr_doc))
                if num == len(ana_tsv_cont)-2:
                    curr_nps +=1
                try:
                    doc.test_pickled_np(curr_doc,curr_nps)
                except AssertionError:
                    print(curr_doc)
                    print(line)
                    print("ERROR")
                total_words += curr_nps
                curr_nps = 1
                curr_doc = line[0]
                test +=1
            else:
                curr_nps += 1
        print("tested {} docs, total words {}".format(test,total_words))

if __name__ == '__main__':
    ac = AnalyzeCorpus(json_objects_path=is_notes_data_path,jsonlines=is_notes_jsonlines)
    # ac.analyze_heads_anaphors()
    # ac.test_files()
    ac.test_tsv()
    # ac.generate_single_tsv_and_vecs()