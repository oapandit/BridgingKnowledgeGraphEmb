from common_code import *
from json_info_extractor import JsonDocInformationExtractor
from common_constants import *
from common_utils import *
from bridging_constants import *
import re

class BridgingJsonDocInformationExtractor(JsonDocInformationExtractor):
    def __init__(self, json_object, logger):
        super().__init__(json_object, logger)
        if self.is_file_empty:
            return None
        self.logger = logger
        self.json_object = json_object
        self.anaphors = self.get_anaphors()
        self.antecedents = self.get_gold_antecedents()
        self.noun_phrase_list = self.get_noun_phrases()
        self.interesting_entities = self.get_all_interesting_entities()
        self.all_bridging_links = self.get_all_bridging_links()
        self.art_anaphors = self.get_art_anaphors()
        self.gold_and_art_anaphors = self.anaphors + self.art_anaphors
        self.anaphors_and_noun_phrases_to_sent_map = self.get_span_to_sentence_map(self.noun_phrase_list+self.anaphors+self.art_anaphors)
        self.sentence_to_anaphors_and_noun_phrases_map = self.get_sentence_to_span_map(self.anaphors_and_noun_phrases_to_sent_map)
        self.ana_to_antecedents_map = self.get_anaphor_to_antecedents_map()
        self.gold_ana_to_antecedent_map = self.get_gold_ana_to_antecedent_map()
        self.salients = self.get_salients()
        self.clean_art_bridging_pairs_good, self.clean_art_bridging_pairs_bad, self.clean_art_bridging_pairs_ugly = self.get_cleaned_artificial_data()
        self.interesting_ent_head_map = self.entities_to_head_span_map
        # self.anaphors = self.get_non_comp_anaphors()

    def get_gold_antecedents(self):
        assert self.bridging_clusters is not None
        antecedents = [cl[0] for cl in self.bridging_clusters]
        return antecedents

    def get_anaphors(self):
        """
        In the ISNotes data, in the cluster information the second element is of anaphor.
        Anaphor is represented as tuple containing start and end position in the document.
        This method returns list of all the anaphors of the document.
        :param mention_ops:
        :return: list of tuples containing start and end pos  e.g. [(1,2),(2,3),(4,5)]
        """
        assert self.bridging_clusters is not None
        anaphors = [cl[1] for cl in self.bridging_clusters]
        self.logger.debug("anaphors {}".format(anaphors))
        assert len(anaphors)==len(self.bridging_clusters),"not equal : anaphors {} clusters {}".format(anaphors,self.bridging_clusters)
        return anaphors

    def get_art_anaphors(self):
        '''
        when artificially generated links are added, new anphors are also generated becuase of coreference linking.
        return those anaphors.
        :return:
        '''
        assert self.art_bridging_pairs_ugly is not None
        anaphors = [cl[1] for cl in self.art_bridging_pairs_ugly]
        self.logger.debug("artificial anaphors {}".format(anaphors))
        assert len(anaphors)==len(self.art_bridging_pairs_ugly),"not equal : anaphors {} clusters {}".format(anaphors,self.art_bridging_pairs_ugly)
        return anaphors

    def _get_current_parse(self,parse_bit_):
        parse_bit_list = []
        pos = ""
        for ch in parse_bit_:
            if ch == "(" or ch == ")":
                if len(pos) > 0:
                    parse_bit_list.append(pos)
                    pos = ""
            else:
                pos += ch
        if len(pos) > 0:
            parse_bit_list.append(pos)
        return parse_bit_list.pop()


    def is_valid_np(self, start_pos, end_pos, is_filter_pronouns=True):
        is_np_parse = False
        is_not_pronoun = True

        #TODO : This is the old code, if needed revert
        if self.parse_bit is not None:
            current_parse = self._get_current_parse(self.parse_bit[start_pos])
            if current_parse == "NP*" or current_parse == "NML*":
                is_np_parse = True
        else: # In ARRAU dataset we do not parsed bits for words
            is_np_parse = True

        # is_np_parse = True
        # string_representation = self.get_tree_rep(start_pos, end_pos)
        # is_np_parse = string_representation[1] == "N"

        if is_np_parse and is_filter_pronouns:
            words = self.get_words_for_start_end_indices((start_pos, end_pos))
            if len(words.split())==1 and words.lower() in remove_these_words:
                is_not_pronoun = False

        return is_np_parse and is_not_pronoun


    def get_noun_phrases(self,is_filter_pronouns=False):
        # TODO:signature of the function changed, default was true earlier but because some of the antecedents are pronouns only,
        # it does not make sense to filter them
        """
        Entities in the jsonlines contains all the entities of the document. These are not limited to being
        NP but can be of any pos tag example VP, ADJP etc.

        We consider only NPs as the candidate antecedents so we filter out those.
        This method returns list of candidate antecedents represented by their start and end position in the doc.
        :param mention_ops:
        :return:
        """
        noun_phrase_list = []
        assert self.entities is not None
        assert len(self.entities) > 0
        self.logger.critical(5*"==**==")
        self.logger.critical("List of entities which are filtered")
        self.logger.critical(5 * "==**==")
        for start_pos, end_pos in self.entities:
            if self.is_valid_np(start_pos, end_pos,is_filter_pronouns):
                noun_phrase_list.append((start_pos, end_pos))
            else:
                words = self.get_words_for_start_end_indices((start_pos, end_pos))
                self.logger.critical("{}".format(words))
                # self.logger.critical("{}".format(self.get_tree_rep(start_pos, end_pos)))
                self.logger.critical(3*"----")
        self.logger.critical(5 * "==**==")
        self.logger.debug(
            "number of entities {} and candidate antecedents {}".format(len(self.entities), len(noun_phrase_list)))
        assert len(noun_phrase_list) > 0
        return noun_phrase_list

    def get_all_interesting_entities(self, is_filter_pronoun_nps=True):
        '''
        the entities from bridging clusters, all noun phrases, all the entities present in the document
        should be included in the interesting entities list.

        They will be repeating so set operation is done to get only unique entities

        :return:
        '''
        assert self.art_bridging_pairs_good is not None and self.art_bridging_pairs_bad is not None and self.art_bridging_pairs_ugly is not None
        if is_filter_pronoun_nps:
            nps = self.noun_phrase_list
        else:
            nps = self.get_noun_phrases(False)
        interesting_entities = list(set(flatten(self.bridging_clusters) + nps + flatten(self.art_bridging_pairs_good + self.art_bridging_pairs_bad + self.art_bridging_pairs_ugly)))
        return interesting_entities

    def get_all_bridging_links(self):
        assert self.art_bridging_pairs_good is not None and self.art_bridging_pairs_bad is not None and self.art_bridging_pairs_ugly is not None
        all_bridging_links = self.bridging_clusters + self.art_bridging_pairs_good + self.art_bridging_pairs_bad + self.art_bridging_pairs_ugly
        self.logger.debug("total bridging links {}".format(len(all_bridging_links)))
        assert len(all_bridging_links) == len(self.bridging_clusters) + len(self.art_bridging_pairs_good) + len(self.art_bridging_pairs_bad) + len(self.art_bridging_pairs_ugly)
        return all_bridging_links

    def get_anaphor_to_antecedents_map(self,bridging_links=None):
        """
        if bridging_links are not given then
        this method considers all the lists which have the information of anaphors and antecedents
            - bridging clusters
            - arti good
            - arti bad
            - arti ugly
        otherwise given bridging links is considered
        from these lists, we create a map from anaphor to list of antecedent, as it may happen that an anaphor
        can be linked multiple antecedents because of coreference.
        :return:
        """
        if bridging_links is None:
            bridging_links = self.all_bridging_links
        ana_to_antecedents_map = {}
        for cl in bridging_links:
            # self.logger.debug("cluster {}".format(cl))
            ante,ana = cl
            # self.logger.debug("ante {} and ana {}".format(ante, ana))
            curr_ante_list = ana_to_antecedents_map.get(ana, [])
            curr_ante_list.append(ante)
            ana_to_antecedents_map[ana] = curr_ante_list
        self.logger.debug("map {}".format(ana_to_antecedents_map))
        return ana_to_antecedents_map

    def get_gold_ana_to_antecedent_map(self):
        gold_ana_to_antecedent_map = {cl[1]:cl[0] for cl in self.bridging_clusters}
        return gold_ana_to_antecedent_map


    def get_ana_to_anecedent_map_from_selected_data(self,is_consider_arti_good_pairs,is_consider_arti_bad_pairs,is_consider_arti_ugly_pairs,is_clean_art_data):
        true_pairs = []
        true_pairs += self.bridging_clusters

        if is_consider_arti_good_pairs:
            if is_clean_art_data:
                true_pairs += self.clean_art_bridging_pairs_good
            else:
                true_pairs += self.art_bridging_pairs_good
        if is_consider_arti_bad_pairs:
            if is_clean_art_data:
                true_pairs += self.clean_art_bridging_pairs_bad
            else:
                true_pairs += self.art_bridging_pairs_bad
        if is_consider_arti_ugly_pairs:
            if is_clean_art_data:
                true_pairs += self.clean_art_bridging_pairs_ugly
            else:
                true_pairs += self.art_bridging_pairs_ugly

        ana_to_antecdents_map_selected_data = self.get_anaphor_to_antecedents_map(true_pairs)
        return ana_to_antecdents_map_selected_data


    def print_words(self,word_start_end_ind,is_print_sent=False):
        words = self.get_span_words(word_start_end_ind)
        sents = self.get_span_sentences(word_start_end_ind)
        pos = self.get_spans_pos(word_start_end_ind)
        parse_bits = self.get_spans_parse_bits(word_start_end_ind)
        for w,s,p,pb in zip(words,sents,pos,parse_bits):
            self.logger.debug(3*"----")
            self.logger.debug("word : {}".format(w))
            self.logger.debug("pos tag: {}".format(p))
            self.logger.debug("parse bits: {}".format(pb))
            if is_print_sent:
                self.logger.debug("sentence : {}".format(s))

    def get_pairwise_sentences(self,list_of_span_pairs):
        """
        Returns list of lists containing two sentences, each for antecedent and anaphor.
        :param list_of_span_pairs:
        :return:
        """
        list_pairwise_sentences = []
        for ante,ana in list_of_span_pairs:
            # ante,ana = pair
            # print("ante : {}, ana : {}".format(ante,ana))
            ante_sents = self.add_marker_to_sent(ante)
            ana_sents = self.add_marker_to_sent(ana)
            _sents = [ante_sents,ana_sents]
            list_pairwise_sentences.append(_sents)
        return list_pairwise_sentences

    def filter_anaphors(self,spans):
        anaphors = self.anaphors + self.art_anaphors
        all_cand_ante = [ca for ca in spans if ca not in anaphors]
        return all_cand_ante

    def filter_cand_antecedents(self, anaphor, all_cand_ante,is_filter_other_anaphors=True):
        """
        Checks if given candidate antecedents really lie before anaphor, if not neglect that.
        Also remove those NPs which are anaphors, anaphors are less likely to be antecedent.
        :param anaphor: tuple containing start, end info e.g. (7,11)
        :param all_cand_ante: list of tuples start, end info e.g. [(1,2),(2,3),(4,5)]
        :return: list of tuples
        """
        self.logger.debug("anaphor {} and candidate antecedents {}".format(anaphor, all_cand_ante))
        all_cand_ante = list(set(all_cand_ante))
        start_anaphor, end_anaphor = anaphor
        all_cand_ante = [(s, e) for s, e in all_cand_ante if e <= start_anaphor]
        self.logger.debug("after removing candidate antecedents which are after anaphor {}".format(all_cand_ante))
        #TODO : Uncomment
        # assert len(all_cand_ante)>0
        if is_filter_other_anaphors:
            all_cand_ante = self.filter_anaphors(all_cand_ante)
            assert len(all_cand_ante) > 0,"no antecedent after removing anaphors {}".format(all_cand_ante)
        self.logger.debug("after removing candidate antecedents which are anaphors {}".format(all_cand_ante))
        return all_cand_ante

    def _test_ante_present_in_cands(self,all_cand_ante,antecedents,is_ante_present):
        if len([c for c in all_cand_ante if c in antecedents]) == 0 and is_ante_present:
            self.logger.debug("ERROR : antecedent is removed after this operation")
            is_ante_present = False
        return is_ante_present

    def filter_cand_antecedents_hou_rules(self,anaphor,all_cand_ante):
        ana_to_antecedent_map = self.get_ana_to_anecedent_map_from_selected_data(
            True, False, False, True)
        antecedents = ana_to_antecedent_map[anaphor]
        is_ante_present = self._test_ante_present_in_cands(all_cand_ante,antecedents,True)
        assert self.interesting_ent_head_map is not None
        ana_head = self.interesting_ent_head_map[anaphor]
        self.logger.debug("candidate antecedents {}".format(all_cand_ante))
        all_cand_ante = [cand for cand in all_cand_ante if self.interesting_ent_head_map[cand] != ana_head]
        self.logger.debug("after removing same head as anaphor candidate antecedents {}".format(all_cand_ante))
        is_ante_present = self._test_ante_present_in_cands(all_cand_ante, antecedents, is_ante_present)
        remove_cands = []
        for cand_i in all_cand_ante:
            for cand_j in all_cand_ante:
                cand_i_head = self.interesting_ent_head_map[cand_i]
                cand_j_head = self.interesting_ent_head_map[cand_j]
                if (cand_i != cand_j) and (cand_i[0] == cand_j[0]) and (cand_i_head==cand_j_head): # start ind and head same
                    if cand_i[1]<cand_j[1]:
                        remove_cands.append(cand_i)
                        self.logger.debug("removing candidate :{} because of {}".format(cand_i,cand_j))
        self.logger.debug("remove these candidate antecedents {}".format(remove_cands))
        all_cand_ante = [cand for cand in all_cand_ante if cand not in remove_cands]
        self.logger.debug("after removing same start candidate antecedents {}".format(all_cand_ante))
        is_ante_present = self._test_ante_present_in_cands(all_cand_ante, antecedents, is_ante_present)
        remove_cands = []
        all_anaphors = self.anaphors + self.art_anaphors
        for cand in all_cand_ante:
            start_cand,end_cand = cand
            for ana in all_anaphors:
                if (ana[0] <= start_cand and end_cand <= ana[1]) or (start_cand <= ana[0] and ana[1] <= end_cand):
                    remove_cands.append(cand)
                    self.logger.debug("removing candidate antecedents {} because of {}".format(cand,ana))
                    break
        self.logger.debug("remove these candidate antecedents {}".format(remove_cands))
        all_cand_ante = [cand for cand in all_cand_ante if cand not in remove_cands]
        self.logger.debug("after removing candidate antecedents which contain anaphors {}".format(all_cand_ante))
        is_ante_present = self._test_ante_present_in_cands(all_cand_ante, antecedents, is_ante_present)
        remove_cands=[]
        for cand in all_cand_ante:
            cand_head = self.interesting_ent_head_map[cand]
            if (not re.match(time_reg,ana_head.lower())) and re.match(time_reg,cand_head.lower()):
                remove_cands.append(cand)
        self.logger.debug("remove these candidate antecedents {}".format(remove_cands))
        all_cand_ante = [cand for cand in all_cand_ante if cand not in remove_cands]
        self.logger.debug("after removing candidate antecedents which are time and anaphor is not time {}".format(all_cand_ante))
        self._test_ante_present_in_cands(all_cand_ante, antecedents, is_ante_present)
        return all_cand_ante


    def get_distance_between_ana_ante(self,ana_to_antecedent_map=None):
        """
        We calculate how many sentences are there in between anaphor and antecedent.
        If there are multiple instances of antecedent then nearest antecedent is considered.
        :return:
        """
        ana_ante_sent_dist = []
        if ana_to_antecedent_map is None:
            ana_to_antecedent_map = self.get_ana_to_anecedent_map_from_selected_data(
                True, False, False, True)
        for ana in ana_to_antecedent_map.keys():
            ana_sent = self.anaphors_and_noun_phrases_to_sent_map[ana]
            antes = ana_to_antecedent_map[ana]
            ante_sents = self.get_span_to_sentence_list(antes)
            distances = [-1 if ante_sent==0 else -ante_sent+ana_sent for ante_sent in ante_sents]
            ana_ante_sent_dist.append(min(distances))
        return ana_ante_sent_dist

    def get_cand_antecedents(self,anaphor,sent_window,is_salient,is_apply_hou_rules):
        """
        For a given anaphor return the list of candidate antecedents which lie in the window of 'sent_window' sentences.
        The candidate antecedents are chosen from the NPs.

        In the sentence window can lie NPs which are occuring after anaphor, filter out those NPs.
        :param anaphor: tuple containing start,end info e.g (1,2)
        :return:
        """
        # is_apply_hou_rules = False
        cand_antecedents = []
        ana_sent = self.anaphors_and_noun_phrases_to_sent_map[anaphor]
        if sent_window == -1:
            cand_sent_end = 0
        else:
            cand_sent_end = ana_sent - sent_window
        self.logger.debug("anaphor sentence {} and sentence window {}".format(ana_sent,sent_window))
        self.logger.debug("will be searching for cand ant in {} to {} window".format(ana_sent,
                                                                                cand_sent_end))
        for cand_sent in range(ana_sent,cand_sent_end-1, -1):
            if self.sentence_to_anaphors_and_noun_phrases_map.get(cand_sent, None) is not None:
                cand_antecedents += self.sentence_to_anaphors_and_noun_phrases_map[cand_sent]

        if is_salient:
            cand_antecedents += self.salients

        self.logger.debug("anaphor {} and candidates antecedents from window {}".format(anaphor,cand_antecedents))
        ana_to_antecedent_map = self.get_ana_to_anecedent_map_from_selected_data(
            True, False, False, True)
        antecedents = ana_to_antecedent_map[anaphor]
        is_ante_present = self._test_ante_present_in_cands(cand_antecedents,antecedents,True)

        cand_antecedents = self.filter_cand_antecedents(anaphor, cand_antecedents,is_filter_other_anaphors=False)
        self.logger.debug("after filtering anaphors and candidates which are after anaphor {}".format(cand_antecedents))
        is_ante_present = self._test_ante_present_in_cands(cand_antecedents, antecedents, is_ante_present)
        if is_apply_hou_rules:
            cand_antecedents = self.filter_cand_antecedents_hou_rules(anaphor, cand_antecedents)
            self.logger.debug(
                "after filtering candidates with hou's rule {}".format(cand_antecedents))
            is_ante_present = self._test_ante_present_in_cands(cand_antecedents, antecedents, is_ante_present)
        assert len(cand_antecedents)!=0
        # if len(cand_antecedents)==0:
        #     return self.get_cand_antecedents(anaphor, sent_window+1,is_salient,is_apply_hou_rules)
        # self.logger.debug("for anaphor {} ".format(self.get_words_for_start_end_indices(anaphor)))
        # self.logger.debug("candidate antecedents are - ")
        # self.print_words(cand_antecedents)
        # assert len(cand_antecedents)>0,"candidate antecedents {}".format(cand_antecedents)
        cand_antecedents.sort(reverse=True)
        return cand_antecedents

    def get_salients(self):
        '''
        consider all the NPs from the first sentence of the doc, ignore anaphors.
        :return:
        '''
        if self.sentence_to_anaphors_and_noun_phrases_map.get(0,None) is None:
            return []
        init_nps = self.sentence_to_anaphors_and_noun_phrases_map[0]
        init_nps = self.filter_anaphors(init_nps)
        return init_nps

    def _filter_pairs(self,pairs):
        cleaned_pairs = []
        for pair in pairs:
            ant,ana = pair
            if self.is_valid_np(*ant) and self.is_valid_np(*ana):
                cleaned_pairs.append(pair)
        return cleaned_pairs
        # return pairs

    def get_cleaned_artificial_data(self):
        '''
        remove those entries which are not NPs.
        :return:
        '''
        clean_art_bridging_pairs_good = self._filter_pairs(self.art_bridging_pairs_good)
        clean_art_bridging_pairs_bad = self._filter_pairs(self.art_bridging_pairs_bad)
        clean_art_bridging_pairs_ugly = self._filter_pairs(self.art_bridging_pairs_ugly)
        return clean_art_bridging_pairs_good,clean_art_bridging_pairs_bad,clean_art_bridging_pairs_ugly

    def set_interesting_entities_sem_heads(self,vec_path):
        if self.interesting_ent_head_map is None:
        #     interesting_entities_sem_head_words,_ = self.get_spans_head(self.interesting_entities,True)
        #     self.interesting_ent_head_map = {ent:head for ent,head in zip(self.interesting_entities,interesting_entities_sem_head_words)}

            # TODO : earlier We were reading semantic heads from the file, but now generating them
            # on the fly, may be revert back to earlier logic
            f_path = os.path.join(vec_path, self.doc_key + int_head_map_extn)
            self.interesting_ent_head_map = read_pickle(f_path)
            # for k in self.interesting_ent_head_map.keys():
            #     v = self.interesting_ent_head_map[k]
                # print("{}:{}".format(k,v))

    def get_non_comp_anaphors(self):
        if self.json_object.get('bridg_type', None) is not None:
            non_comp_anaphors = []
            types = self.json_object.get('bridg_type', None)
            assert len(self.anaphors) == len(types)
            for i,ana in enumerate(self.anaphors):
                t = types[i]
                if anaphor_types[2] in t and anaphor_types[0] not in t:
                    non_comp_anaphors.append(ana)
            return non_comp_anaphors
        else:
            return self.anaphors


    def get_hou_like_mention_modifiers(self,mentions,is_bashi):
        if mentions is None: mentions = self.interesting_entities
        mentions_sem_head_words,_ = self.get_spans_head(mentions,True)
        mentions_words = self.get_span_words(mentions)
        mentions_pos = self.get_spans_pos(mentions)
        modifiers = bashi_modifiers if is_bashi else is_notes_modifiers
        assert len(mentions) == len(mentions_sem_head_words) == len(mentions_words) == len(mentions_pos)
        mention_mods = []
        for m,mh,mw,mp in zip(mentions,mentions_sem_head_words,mentions_words,mentions_pos):
            self.logger.debug(5*"---")
            self.logger.debug("ind : {}, head : {}, word : {}, pos : {}".format(m,mh,mw,mp))
            assert mh in mw,"head {}, not in word {}".format(mh,mw)
            m_words_list = mw.split()
            mp = mp.split()
            assert len(mp) == len(m_words_list)
            content_simple = []
            for w,pos_tag in zip(m_words_list,mp):
                if re.match(modifiers, pos_tag.upper()):
                    content_simple.append(w)
            context_after_head = []
            if mh != mw:
                head_ind = m_words_list.index(mh)
                if head_ind+1<len(m_words_list) and m_words_list[head_ind+1] == "of":
                    i = head_ind + 2
                    while i<len(mp):
                        context_after_head.append(m_words_list[i])
                        if re.match("(NN|NNS)", mp[i].upper()):
                            break
                        i+=1
            if mh in content_simple and len(content_simple) != 1:
                    content_simple = content_simple[0:content_simple.index(mh)]
                    content_simple += [mh]
                    if len(context_after_head) !=0:
                        content_simple += context_after_head
                    content_simple = " ".join(content_simple)
            else:
                content_simple = mh
            self.logger.debug("content simple : {}".format(content_simple))
            assert isinstance(content_simple,str)
            mention_mods.append(content_simple)

        assert len(mentions) == len(mention_mods)
        return mention_mods

if __name__ == '__main__':
    from bridging_utils import *
    json_objects = read_jsonlines_file(is_notes_json_f_path)
    for single_json_object in json_objects:
        mention_ops = BridgingJsonDocInformationExtractor(single_json_object,logger)
        if mention_ops.is_file_empty: continue
        mention_ops.get_hou_like_mention_modifiers()
        # sys.exit()
