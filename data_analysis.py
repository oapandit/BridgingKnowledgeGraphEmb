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
from prettytable import PrettyTable

from data_read import BridgingVecDataReader
class AnalyzeDoc:
    def __init__(self, json_object,corpus):
        self.mention_ops = BridgingJsonDocInformationExtractor(json_object, logger)
        self.is_file_empty = False
        if self.mention_ops.is_file_empty:
            print("{} is empty".format(self.mention_ops.doc_key))
            self.is_file_empty = True
            return None
        self.json_object = json_object
        self.corpus = corpus
        self.sen_window_size  = -1
        self.is_consider_saleint = True
        self.is_training = False

    def analyze_ana_ante_cands(self):
        ana_to_antecedent_map = self.mention_ops.get_ana_to_anecedent_map_from_selected_data(
            True, False, False, True)
        ana_ante_sent_dist = self.mention_ops.get_distance_between_ana_ante(ana_to_antecedent_map)
        cand_ante_per_anaphor = self.corpus.corpus_vec_data.generate_cand_ante_by_sentence_window(self.mention_ops, self.sen_window_size,self.is_consider_saleint, self.is_training, ana_to_antecedent_map)

        try:
            assert len(cand_ante_per_anaphor) == len(ana_to_antecedent_map.keys()),"cand ante per ana :{},ana to anecdent map :{}".format(len(cand_ante_per_anaphor),len(ana_to_antecedent_map.keys()))
        except AssertionError:
            print("doc : {}".format(self.mention_ops.doc_key))
            print("cand ante per ana :{},ana to anecdent map :{}".format(len(cand_ante_per_anaphor),len(ana_to_antecedent_map.keys())))
            print("")
            raise AssertionError

        anaphors_ = ana_to_antecedent_map.keys()
        anaphors_sents_numb = self.mention_ops.get_span_to_sentence_list(anaphors_)
        anaphors = self.mention_ops.get_span_words(anaphors_)
        anaphors_heads, _ = self.mention_ops.get_spans_head(anaphors_, True)

        antecedents_ = [ana_to_antecedent_map[ana] for ana in anaphors_]
        antecedents_sents_numb = [self.mention_ops.get_span_to_sentence_list(ant) for ant in antecedents_]
        antecedents = [self.mention_ops.get_span_words(ant) for ant in antecedents_]
        antecedents_heads = [(self.mention_ops.get_spans_head(ant, True))[0] for ant in antecedents_]

        candidate_antecedents_sents_numb = [self.mention_ops.get_span_to_sentence_list(cand) for cand in cand_ante_per_anaphor]
        candidate_antecedents = [self.mention_ops.get_span_words(cand) for cand in cand_ante_per_anaphor]
        candidate_antecedents_heads = [(self.mention_ops.get_spans_head(cand, True))[0] for cand in cand_ante_per_anaphor]

        antecedent_present_in_cand = []
        assert len(anaphors_) == len(anaphors) == len(anaphors_heads) == len(ana_ante_sent_dist) ==\
               len(antecedents_) == len(antecedents) == len(antecedents_heads) == \
               len(candidate_antecedents) == len(candidate_antecedents_heads) == \
               len(anaphors_sents_numb) == len(antecedents_sents_numb) == len(candidate_antecedents_sents_numb)

        logger.critical(3*"====**====")
        for i in range(len(anaphors)):
            is_present = False
            logger.critical("******")
            logger.critical("anaphor : [{}] - [{}]".format(anaphors[i],anaphors_heads[i]))
            logger.critical("antecedent : [{}] - [{}]".format(antecedents[i],antecedents_heads[i]))
            logger.critical("distance is : {}".format(ana_ante_sent_dist[i]))
            logger.critical("candidate antecedents : ")
            for can_ant,hs,cand_ant_ind in zip(candidate_antecedents[i],candidate_antecedents_heads[i],cand_ante_per_anaphor[i]):
                logger.critical("[{}] - [{}] : {}".format(can_ant,hs,cand_ant_ind in antecedents_[i]))
                if cand_ant_ind in antecedents_[i]:
                    is_present = True
                # logger.debug("+++")
            if ana_ante_sent_dist[i] < 3 and not is_present:
                logger.critical("ERROR IN SELECTING CANDIDATE ANTECEDENTS")
                is_present_in_ent = False
                for an in antecedents_[i]:
                    if an in self.mention_ops.entities:
                        is_present_in_ent = True
                logger.critical(" antecedents : {} ,  present in entities {}".format(antecedents_[i],is_present_in_ent))
                # if not is_present_in_ent:
                logger.critical("entities {}".format(self.mention_ops.entities))
                logger.critical("NP : {}".format(self.mention_ops.noun_phrase_list))
            antecedent_present_in_cand.append(is_present)
        logger.critical(3 * "====**====")

        self.corpus.num_cand_antes_per_ana += [len(c) for c in candidate_antecedents]

        self.corpus.corpus_ana_ante_sent_dist += ana_ante_sent_dist
        self.corpus.num_anaphors_per_doc.append(len(anaphors_))
        self.corpus.corpus_anaphors += anaphors
        self.corpus.corpus_antecedents += antecedents
        self.corpus.corpus_anaphors_heads += anaphors_heads
        self.corpus.corpus_antecedents_heads += antecedents_heads
        self.corpus.corpus_candidate_antecedents += candidate_antecedents
        self.corpus.corpus_candidate_antecedents_heads += candidate_antecedents_heads

        self.corpus.corpus_anaphors_sents_numb += anaphors_sents_numb
        self.corpus.corpus_antecedents_sents_numb += antecedents_sents_numb
        self.corpus.corpus_candidate_antecedents_sents_numb += candidate_antecedents_sents_numb

        self.corpus.corpus_antecedent_present_in_cand += antecedent_present_in_cand
    def ext_info_analysis(self, path2vec_embeddings):
        all_spans_interested = []
        salients = self.mention_ops.get_salients()
        all_spans_interested += salients
        for an, ant in zip(self.mention_ops.anaphors, self.mention_ops.antecedents):
            all_spans_interested.append(an)
            all_spans_interested.append(ant)
            cand_ante = self.mention_ops.get_cand_antecedents(an, 3)
            all_spans_interested += cand_ante
        logger.debug("sample all spans - {}".format(all_spans_interested[0:5]))
        all_spans_interested = list(set(all_spans_interested))
        span_words = self.mention_ops.get_span_words(all_spans_interested)
        span_words_clean = self.mention_ops.remove_pronouns_and_clean_words(span_words)
        span_head_words = self.mention_ops.get_span_head_word(all_spans_interested)
        assert len(span_words_clean) == len(span_head_words) == len(span_words)
        for w, hw in zip(span_words_clean, span_head_words):
            self.total_spans += 1
            logger.debug(3 * "---")
            logger.debug("word {}".format(w))
            num_sense_vectors = path2vec_embeddings.get_num_sense_vectors(w)
            if num_sense_vectors > 0:
                self.num_sense_vectors.append(num_sense_vectors)
                logger.debug("--- FOUND SENSE WITH WORD --")
                self.ext_info_with_word += 1
            else:
                logger.debug("head word {}".format(hw))
                num_sense_vectors = path2vec_embeddings.get_num_sense_vectors(hw)
                if num_sense_vectors > 0:
                    self.num_sense_vectors.append(num_sense_vectors)
                    logger.debug("--- FOUND SENSE WITH HEAD WORD --")
                    self.ext_info_with_head_word += 1
                else:
                    logger.error("no external information for word : {}".format(w))

    def analyze_artificial_links(self):
        self.num_good_links = len(self.mention_ops.art_bridging_pairs_good)
        self.num_bad_links = len(self.mention_ops.art_bridging_pairs_bad)
        self.num_ugly_links = len(self.mention_ops.art_bridging_pairs_ugly)

        self.valid_np_good = 0
        for ante, ana in self.mention_ops.art_bridging_pairs_good:
            if self.mention_ops.is_valid_np(ante[0], ante[1]):
                self.valid_np_good += 1

        self.valid_np_bad = 0
        for ante, ana in self.mention_ops.art_bridging_pairs_bad:
            if self.mention_ops.is_valid_np(ante[0], ante[1]):
                self.valid_np_bad += 1

        self.valid_np_ugly = 0
        for ante, ana in self.mention_ops.art_bridging_pairs_ugly:
            if self.mention_ops.is_valid_np(ante[0], ante[1]) and self.mention_ops.is_valid_np(ana[0], ana[1]):
                self.valid_np_ugly += 1

        # logger.debug("\n \n artificial good links \n \n")
        # self.mention_ops.print_words(self.mention_ops.art_bridging_pairs_good,is_print_sent=True)
        # logger.debug("\n\n")
        #
        # logger.debug("\n \n artificial bad links \n \n")
        # self.mention_ops.print_words(self.mention_ops.art_bridging_pairs_bad,is_print_sent=True)
        # logger.debug("\n\n")
        #
        # logger.debug("\n \n artificial ugly links \n \n")
        # self.mention_ops.print_words(self.mention_ops.art_bridging_pairs_ugly,is_print_sent=True)
        # logger.debug("\n\n")

    def anaphor_head_words(self):
        anaphors_sem_head_words, _ = self.mention_ops.get_spans_head(self.mention_ops.anaphors, True)
        anaphors_collins_head_words, _ = self.mention_ops.get_spans_head(self.mention_ops.anaphors, False)
        self.sem_head_ana = anaphors_sem_head_words
        self.synt_head_ana = anaphors_collins_head_words
        self.doc_name = len(anaphors_collins_head_words) * [self.mention_ops.doc_key]
        self.diff = 0
        self.anaphors_sents_number = self.mention_ops.get_span_to_sentence_list(self.mention_ops.anaphors)
        assert len(self.sem_head_ana) == len(self.synt_head_ana) == len(self.doc_name) == len(self.anaphors) == len(
            self.anaphors_sents_number)
        for e, y in zip(anaphors_sem_head_words, anaphors_collins_head_words):
            if e != y:
                self.diff += 1


class AnalyzeCorpus:
    def __init__(self, json_objects_path, jsonlines):
        self.corpus_vec_data = BridgingVecDataReader(json_objects_path=json_objects_path, vec_path=is_notes_vec_path,
                                 is_read_head_vecs=True, is_sem_head=True,
                                 is_all_words=False, path2vec_sim=2,
                                 is_hou_like=False,wn_vec_alg=PATH2VEC)
        self.corpus_vec_data.is_soon = False
        self.corpus_vec_data.is_positive_surr = False
        self.json_objects_path = json_objects_path
        json_path = os.path.join(self.json_objects_path, jsonlines)
        self.json_objects = read_jsonlines_file(json_path)
        self.num_anaphors_per_doc = None

    def analyze_by_doc(self):
        self.num_anaphors_per_doc =[]
        self.num_cand_antes_per_ana = []
        self.corpus_ana_ante_sent_dist = []
        self.corpus_anaphors = []
        self.corpus_antecedents = []
        self.corpus_anaphors_heads = []
        self.corpus_antecedents_heads = []
        self.corpus_candidate_antecedents = []
        self.corpus_candidate_antecedents_heads = []
        self.corpus_anaphors_sents_numb = []
        self.corpus_antecedents_sents_numb = []
        self.corpus_candidate_antecedents_sents_numb = []

        self.corpus_antecedent_present_in_cand = []

        present = []
        for single_json_object in self.json_objects:
            doc = AnalyzeDoc(single_json_object,self)
            if doc.is_file_empty: continue
            doc.analyze_ana_ante_cands()

            assert len(self.corpus_anaphors) == len(self.corpus_anaphors_heads) == \
                   len(self.corpus_antecedents) == len(self.corpus_antecedents_heads) ==\
                    len(self.corpus_antecedents) == len(self.corpus_antecedents_heads) ==\
                    len(self.num_cand_antes_per_ana) == len(self.corpus_ana_ante_sent_dist) == \
                   len(self.corpus_anaphors_sents_numb) == len(self.corpus_antecedents_sents_numb) == \
                   len(self.corpus_candidate_antecedents_sents_numb) == len(self.corpus_antecedent_present_in_cand)

            # sys.exit()
        num_dist_fq = FreqDist(self.corpus_ana_ante_sent_dist)
        # print("{}".format(num_dist_fq.most_common(30)))
        # two_sent_window = [-1,0,1,2]
        two_sent_anap = 0
        total = 0
        for i in range(-1,100,1):
            if i<3:
                two_sent_anap += num_dist_fq[i]
            total += num_dist_fq[i]
        print("out of total {} anaphors, {} are two sentence antecedent and {} are away. Out of two sentence anaphors actually {} have antecedents in candidates.".format(total,two_sent_anap,total-two_sent_anap,sum(self.corpus_antecedent_present_in_cand)))
        print("total candidate antecedents {} and average {}".format(sum(self.num_cand_antes_per_ana),sum(self.num_cand_antes_per_ana)/total))



    def analyze_ext_info_by_doc(self):
        self.path2vec_embeddings = Path2VecEmbeddings("", logger)
        num_sense_vectors = []
        total_spans = 0
        ext_info_with_word = 0
        ext_info_with_head_word = 0
        for single_json_object in self.json_objects:
            doc = AnalyzeDoc(single_json_object)
            if doc.is_file_empty: continue
            doc.ext_info_analysis(self.path2vec_embeddings)
            num_sense_vectors += doc.num_sense_vectors
            total_spans += doc.total_spans
            ext_info_with_word += doc.ext_info_with_word
            ext_info_with_head_word += doc.ext_info_with_head_word
        logger.debug("total spans {}".format(total_spans))
        logger.debug(
            "found ext info with words {}, {}%".format(ext_info_with_word, ext_info_with_word * 100 / total_spans))
        logger.debug("found ext info with head words {}, {}%".format(ext_info_with_head_word,
                                                                     ext_info_with_head_word * 100 / total_spans))
        no_info = total_spans - ext_info_with_word - ext_info_with_head_word
        logger.debug("no info for {}, {}% words ".format(no_info, no_info * 100 / total_spans))
        num_sense_fq = FreqDist(num_sense_vectors)
        logger.debug("{}".format(num_sense_fq.most_common(30)))

    def analyze_art_links(self):
        num_good_links = 0
        num_bad_links = 0
        num_ugly_links = 0

        valid_np_good = 0
        valid_np_bad = 0
        valid_np_ugly = 0
        for single_json_object in self.json_objects:
            doc = AnalyzeDoc(single_json_object)
            if doc.is_file_empty: continue
            doc.analyze_artificial_links()

            num_good_links += doc.num_good_links
            num_bad_links += doc.num_bad_links
            num_ugly_links += doc.num_ugly_links

            valid_np_good += doc.valid_np_good
            valid_np_bad += doc.valid_np_bad
            valid_np_ugly += doc.valid_np_ugly
        logger.debug(
            "total good {} valid {} - {}%".format(num_good_links, valid_np_good, valid_np_good * 100 / num_good_links))
        logger.debug(
            "total good {} valid {} - {}%".format(num_bad_links, valid_np_bad, valid_np_bad * 100 / num_bad_links))
        logger.debug(
            "total good {} valid {} - {}%".format(num_ugly_links, valid_np_ugly, valid_np_ugly * 100 / num_ugly_links))

    def tsne_vis(self):
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(X)
        print('Cumulative explained variation for 50 principal components: {}'.format(
            np.sum(pca_50.explained_variance_ratio_)))

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
        tsne_results = tsne.fit_transform(pca_result_50)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
        print("tsne shape {}".format(tsne_results.shape))

        data = {'tsne-2d-one': tsne_results[:, 0],
                'tsne-2d-two': tsne_results[:, 1],
                'y': bridging_labels_vec_train.reshape(-1)
                }

        df_subset = pd.DataFrame(data, columns=['tsne-2d-one', 'tsne-2d-two', 'y'])

        plt.figure(figsize=(16, 10))
        sns_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 2),
            data=df_subset,
            legend="full",
            alpha=0.3
        )
        plt.savefig("tsne_red_dummy_output.png")

        tsne = SpectralEmbedding(n_components=2)
        tsne_results = tsne.fit_transform(X)

        data = {'tsne-2d-one': tsne_results[:, 0],
                'tsne-2d-two': tsne_results[:, 1],
                'y': bridging_labels_vec_train.reshape(-1)
                }

        df_subset = pd.DataFrame(data, columns=['tsne-2d-one', 'tsne-2d-two', 'y'])

        plt.figure(figsize=(16, 10))
        sns_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 2),
            data=df_subset,
            legend="full",
            alpha=0.3
        )
        plt.savefig("spec_emb_output.png")

    def analyze_heads_anaphors(self):
        sem_head_ana = []
        synt_head_ana = []
        doc_name = []
        anaphors = []
        diff = 0
        sent_nums = []
        for single_json_object in self.json_objects:
            doc = AnalyzeDoc(single_json_object)
            if doc.is_file_empty: continue
            doc.anaphor_head_words()

            sem_head_ana += doc.sem_head_ana
            synt_head_ana += doc.synt_head_ana
            doc_name += doc.doc_name
            anaphors += doc.anaphors
            diff += doc.diff
            sent_nums += doc.anaphors_sents_number
        assert len(sem_head_ana) == len(synt_head_ana) == len(doc_name) == len(anaphors) == len(sent_nums)
        tsv_file_name = os.path.join(is_notes_data_path, "ana_sem_synt.tsv")
        tsv_file = open(tsv_file_name, mode='w')
        tsv_writer = csv.writer(tsv_file, delimiter="\t")
        tsv_writer.writerow(["Doc Name", "Anaphor", "Semantic Head", "Syntactic Head", "Sentence Number"])
        for dn, ana, sem, synt, sn in zip(doc_name, anaphors, sem_head_ana, synt_head_ana, sent_nums):
            tsv_writer.writerow([dn, ana, sem, synt, sn])
        logger.debug("tsv file creation completed.")
        tsv_file.close()
        print("difference in {} anaphors".format(diff))

    def compare_with_hou(self,prob_path):
        json_data_2sents_obj = read_jsonlines_file(prob_path)[0]
        if self.num_anaphors_per_doc is None:
            self.analyze_by_doc()
        ana_2sent_dict = {sentence["anaphor"]: counter for counter, sentence in enumerate(json_data_2sents_obj)}
        same_num_ante = 0
        found_ana = 0
        num_cand_onkar = 0
        num_cand_hou = 0
        num_cand_match = 0
        num_ana_ant_not_found = 0
        num_ana_ante_out_of_boundary = 0
        num_sel_error = 0
        num_yufang_ana_missed = 0
        missed_cands = []
        for i,anaphor in enumerate(self.corpus_anaphors):
            is_found = False
            # total_ana += 1
            is_ante_in_cand = False
            logger.debug(2 * "******************")
            logger.debug("anaphor : [{}] - [{}] : {}".format(anaphor, self.corpus_anaphors_heads[i],
                                                             self.corpus_anaphors_sents_numb[i]))
            logger.debug(
                "antecedent : [{}] - [{}] : [{}]".format(self.corpus_antecedents[i], self.corpus_antecedents_heads[i],
                                                         self.corpus_antecedents_sents_numb[i]))
            # logger.debug("distance is : {}".format(ana_ante_sent_dist[i]))
            logger.debug("candidate antecedents : ")
            for can_ant, hs, sen_num in zip(self.corpus_candidate_antecedents[i],
                                            self.corpus_candidate_antecedents_heads[i],
                                            self.corpus_candidate_antecedents_sents_numb[i]):
                logger.debug("[{}] - [{}] : {}".format(can_ant, hs, sen_num))
                if can_ant in self.corpus_antecedents[i]:
                    is_ante_in_cand = True
                num_cand_onkar += 1
            # assert is_ante_in_cand
            if not is_ante_in_cand:
                num_ana_ant_not_found += 1
                ana_sent = self.corpus_anaphors_sents_numb[i]
                cand_sents = list(range(ana_sent - 2, ana_sent + 1))
                cand_sents.append(0)
                if len([_s for _s in self.corpus_antecedents_sents_numb[i] if int(_s) in cand_sents])==0:
                    logger.debug("out of range error : {} - {}".format(self.corpus_antecedents_sents_numb[i],cand_sents))
                    num_ana_ante_out_of_boundary += 1
                else:
                    logger.debug("selection error : {} - {}".format(self.corpus_antecedents_sents_numb[i], cand_sents))
                    num_sel_error += 1
            if ana_2sent_dict.get(anaphor,None) is not None:
                is_found = True
                if not is_ante_in_cand:
                    num_yufang_ana_missed += 1
                    logger.debug("ERROR : NO ANTE IN CAND")
                found_ana += 1
                logger.debug(2*"+++++")
                logger.debug("found in Yufang's candi")
                logger.debug("+++++")
                sentence_loc_2sent = ana_2sent_dict[anaphor]
                sent = json_data_2sents_obj[sentence_loc_2sent]
                antes = [a['mentionStr'] for a in sent["antecedents"]]
                antes_heads = [a['head'] for a in sent["antecedents"]]
                logger.debug("anaphor : [{}]".format(sent["anaphor"]))
                logger.debug("antecedent : [{}] - [{}]".format(antes,antes_heads))
                # logger.debug("\n")
                logger.debug("candidate antecedents : ")
                ops = sent["options"]
                for op in ops:
                    _is_match = False
                    if op['mentionStr'] in self.corpus_candidate_antecedents[i]:
                        _is_match = True
                        num_cand_match += 1
                    else:
                        missed_cands.append(op['mentionStr'])
                    logger.debug("{} : [{}] - [{}]".format(_is_match,op['mentionStr'], op['head']))
                    num_cand_hou += 1
                if len(sent["options"]) == len(self.corpus_candidate_antecedents[i]):
                    same_num_ante += 1
            if not is_found:
                logger.debug("NOT FOUND IN YUFANG, THOUGH IN RANGE")
        print("total ana : {} ".format(len(self.corpus_anaphors)))
        print("total ana in Yufang {}, found : {} and ante not in cand antes for {}".format(len(json_data_2sents_obj),found_ana,num_yufang_ana_missed))
        print("out of total {} anaphors not found antecedents, {} are because of selection error and {} are because ante are not in range".format(num_ana_ant_not_found ,num_sel_error,num_ana_ante_out_of_boundary))
        print("same candidates {} ".format(same_num_ante))
        print("cand ante - onkar : {} , Hou : {} , matched : {} - {}%".format(num_cand_onkar,num_cand_hou,num_cand_match,num_cand_match*100.0/num_cand_hou))
        # logger.debug("------")
        # logger.debug("missed candidates")
        # for m in missed_cands:
        #     logger.debug("{}".format(m))

class ARRAUData:
    def __init__(self,data_path,jsonlines):
        self.data_path= data_path
        self.jsonlines = jsonlines
        json_path = os.path.join(self.data_path, jsonlines)
        self.json_objects = read_jsonlines_file(json_path)

    def basic_analysis(self):
        num_file_empty = 0
        corpus_anaphors = ["Anaphor"]
        corpus_anaphor_heads = ["Anaphor Head"]
        corpus_anaphor_sents = ["Anaphor Sentence"]

        corpus_antecedents = ["Antecedent"]
        corpus_antecedent_heads = ["Antecedent Head"]
        corpus_antecedent_sents = ["Antecedent Sentence"]

        corpus_data_set = ["Dataset"]
        corpus_train_ana = 0
        corpus_dev_ana = 0
        corpus_test_ana = 0

        corpus_rst_ana =0
        corpus_pear_ana = 0
        corpus_trains_ana = 0

        rst = 0
        trains = 1
        pear = 2

        train = 0
        dev = 1
        test = 2

        corpus_data_name_type = [[0] * 4 for i in range(3)]

        for single_json_object in self.json_objects:
            mention_ops = BridgingJsonDocInformationExtractor(single_json_object, logger)
            if mention_ops.is_file_empty:
                num_file_empty += 1
                print("{} is empty".format(mention_ops.doc_key))
                continue

            anaphors = mention_ops.get_span_words(mention_ops.anaphors)
            anaphor_heads, _ = mention_ops.get_spans_head(mention_ops.anaphors, True)
            anaphor_sents = mention_ops.get_span_sentences(mention_ops.anaphors)

            antecedents = mention_ops.get_span_words(mention_ops.antecedents)
            antecedent_heads, _ = mention_ops.get_spans_head(mention_ops.antecedents, True)
            antecedent_sents = mention_ops.get_span_sentences(mention_ops.antecedents)

            corpus_anaphors += anaphors
            corpus_anaphor_heads += anaphor_heads
            corpus_anaphor_sents += anaphor_sents

            corpus_antecedents += antecedents
            corpus_antecedent_heads += antecedent_heads
            corpus_antecedent_sents += antecedent_sents

            # _data_name = None
            # _data_type = None
            if mention_ops.json_object['dataset_name'] == RST:
                _data_set = RST
                _data_name = rst
                # corpus_rst_ana += len(anaphors)
            elif mention_ops.json_object['dataset_name'] == PEAR:
                _data_set = PEAR
                _data_name = pear
                # corpus_pear_ana += len(anaphors)
            elif mention_ops.json_object['dataset_name'] == TRAINS:
                _data_set = TRAINS
                _data_name = trains
                # corpus_trains_ana += len(anaphors)
            else:
                raise NotImplementedError

            corpus_data_set += len(anaphors) * [_data_set]
            if mention_ops.json_object['dataset_type'] == 'train':
                _data_type = train
                # corpus_train_ana += len(anaphors)
            elif mention_ops.json_object['dataset_type'] == 'dev':
                _data_type = dev
                # corpus_dev_ana += len(anaphors)
            elif mention_ops.json_object['dataset_type'] == 'test':
                _data_type = test
                # corpus_test_ana += len(anaphors)
            else:
                raise NotImplementedError

            corpus_data_name_type[_data_name][_data_type] += len(anaphors)
            corpus_data_name_type[_data_name][3] += 1

        print("---Analysis---")
        print("total documents : {}".format(len(self.json_objects)))
        print("total empty docs : {}".format(num_file_empty))

        # print("Total anaphors : {} from them RST : {}, PEAR : {} and TRAINS : {}".format(len(corpus_anaphors),corpus_rst_ana,corpus_pear_ana,corpus_trains_ana))
        # print("Total anaphors : {} for training : {}, dev : {} and test : {}".format(len(corpus_anaphors),corpus_train_ana,corpus_dev_ana,corpus_test_ana))

        data_details = PrettyTable()

        data_details.field_names = ["", "#doc", "Train", "Dev","Train/Dev", "Test","Empty","Total"]
        empty_anaphors = [72,26,7]
        _train = 0
        _dev = 0
        _test = 0
        _doc = 0
        for i in range(3):
            _train += corpus_data_name_type[i][0]
            _dev += corpus_data_name_type[i][1]
            _test += corpus_data_name_type[i][2]
            _doc += corpus_data_name_type[i][3]

            _total = sum(corpus_data_name_type[i][0:3]) + empty_anaphors[i]
            _train_dev = corpus_data_name_type[i][0] + corpus_data_name_type[i][1]
            _current_data_details = [corpus_data_name_type[i][3],corpus_data_name_type[i][0],corpus_data_name_type[i][1],_train_dev,corpus_data_name_type[i][2],empty_anaphors[i],_total]
            data_details.add_row([datasets[3+i]]+_current_data_details)
        _train_dev = _train + _dev
        _total = _train_dev + _test + sum(empty_anaphors)
        data_details.add_row(["TOTAL",_doc,_train,_dev,_train_dev,_test,sum(empty_anaphors),_total])
        print(data_details)

        # data = [corpus_anaphors,corpus_anaphor_heads,corpus_anaphor_sents,
        #         corpus_antecedents,corpus_antecedent_heads,corpus_antecedent_sents]

        data = zip(corpus_data_set,corpus_anaphors,corpus_anaphor_heads,corpus_anaphor_sents,
                corpus_antecedents,corpus_antecedent_heads,corpus_antecedent_sents)

        arrau_detail_csv = os.path.join(self.data_path,"arrau_details.csv")
        write_csv(arrau_detail_csv,data)



if __name__ == '__main__':
    # prob_data_2sents_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi_2Sent"
    # prob_data_2sents_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi"
    # ac = AnalyzeCorpus(json_objects_path=bashi_data_path, jsonlines=bashi_jsonlines)
    # ac = AnalyzeCorpus(json_objects_path=is_notes_data_path, jsonlines=is_notes_jsonlines)
    # ac.analyze_by_doc()
    # ac.compare_with_hou(prob_data_2sents_path)
    # ac.analyze_ext_info_by_doc()
    # ac.analyze_art_links()
    # ac.analyze_heads_anaphors()

    # ac = AnalyzeCorpus(json_objects_path=arrau_data_path, jsonlines=arrau_jsonlines)
    # ac.analyze_by_doc()

    arr = ARRAUData(data_path=arrau_data_path, jsonlines=arrau_jsonlines)
    arr.basic_analysis()