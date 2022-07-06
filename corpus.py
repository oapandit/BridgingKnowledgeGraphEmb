import os
from bridging_constants import *
from bridging_utils import *
from common_code import *
from common_constants import *
import shutil
from markables import *
from lxml import etree as ET
from bridging_json_extract import BridgingJsonDocInformationExtractor


def generate_artificial_bridging_links(bridging_clusters, coref_clusters):
    """
    We generate extra artificial bridging links with the use of coreference links.
    We assume that if the bridging anaphor or antecedent share coreference relationship with other mention
    then there exist bridging relation between the coreferring mention as well.

    There can be two cases 1. antecedent is coreferring with some mention 2. anaphor is coreferring
        1. if antecedent is coreferring with some mention
            - if that mention lies before the anaphor then we term this as "good" linking
            - else we term it as "bad" linking
        2. if anaphor corefers with some mention
            - then we consider the mention as new anaphor and create all the links which the anaphor has
              including gold, "good" and "bad"
            - we term all the links created in this way as "ugly"
    :return:
    """
    orig_bridging_pairs = 0
    ante_after_ana = 0
    num_ana_coref_ent = 0
    art_bridging_pairs_good = []
    art_bridging_pairs_bad = []
    art_bridging_pairs_ugly = []

    orig_bridging_pairs += len(bridging_clusters)
    # men_to_cluster_map = {}
    logger.debug("coref clusters {}".format(coref_clusters))
    men_to_coref_cluster_map = {tuple(men): cl for cl in coref_clusters for men in cl}
    ana_to_bridging_map = {tuple(cl[1]): cl for cl in bridging_clusters}
    for ante, ana in bridging_clusters:
        logger.debug(7 * "-----")
        logger.debug("ante {} ana {}".format(ante, ana))
        ante = tuple(ante)
        curr_ana_bridges = []
        # check is there any is coreference link for antecedent
        if men_to_coref_cluster_map.get(ante, None) is not None:
            st_ana, end_ana = ana
            curr_ana_bridges.append([ante, ana])
            for men in men_to_coref_cluster_map.get(ante, None):
                logger.debug("men {}".format(men))
                if tuple(men) != ante:
                    curr_ana_bridges.append([men, ana])
                    if men[1] > st_ana:  # check if anaphor lies after mention
                        ante_after_ana += 1
                        art_bridging_pairs_bad.append([men, ana])
                        logger.debug("occurs after ana {}".format(ana))
                    else:
                        art_bridging_pairs_good.append([men, ana])
        # check is there any coreference link for anaphor
        if men_to_coref_cluster_map.get(tuple(ana), None) is not None:
            for men in men_to_coref_cluster_map.get(tuple(ana), None):
                if ana_to_bridging_map.get(tuple(men), None) is None:  # to not add same link of ana
                    logger.debug("men {} corefer {} ana".format(men, ana))
                    num_ana_coref_ent += 1
                    for ante, _ in curr_ana_bridges:
                        art_bridging_pairs_ugly.append([ante, men])
                        logger.debug("ugly pair {} added ".format([ante, men]))

    logger.debug("original bridging pairs {}".format(orig_bridging_pairs))
    logger.debug("artificial good bridging pairs {}".format(len(art_bridging_pairs_good)))
    logger.debug("artificial bad bridging pairs {}".format(len(art_bridging_pairs_bad)))
    logger.debug("artificial ugly bridging pairs {}".format(len(art_bridging_pairs_ugly)))
    return art_bridging_pairs_good, art_bridging_pairs_bad, art_bridging_pairs_ugly


class ISNotes:
    """
    This class builds ISNotes bridging anaphora corpus with use of annotations, jars and ontological files.
    """

    def __init__(self):
        self.dir_original = os.path.join(is_notes_data_path, "OntoNotesOriginal")
        self.dir_target = os.path.join(is_notes_data_path, "IS")
        self.dir_target_clean = os.path.join(is_notes_data_path, "ISClean")
        self.dir_resource_file = os.path.join(is_notes_data_path, "MMAXResourceFile")
        self.dir_is_annotation_trace = os.path.join(is_notes_data_path, "ISAnnotationWithTrace")
        self.dir_is_annotation_without_trace = os.path.join(is_notes_data_path, "ISAnnotationWithoutTrace")

        self.wsj_files_numbers = "1004|1123|1200|1327|1428|1017|1137|1215|" \
                                 "1353|1435|1041|1146|1232|1367|1436|1044|" \
                                 "1148|1264|1379|1443|1053|1150|1284|1387|" \
                                 "1448|1066|1153|1298|1388|1450|1094|1160|" \
                                 "1310|1397|1455|1100|1163|1313|1404|1465|" \
                                 "1101|1172|1315|1423|1473|1121|1174|1324|1424|2454"

    def copy_files_from_onto_notes_to_is_notes(self):
        """
        To construct ISNotes dataset we need to take some files from ontonotes.
        These are wsj files with extension .onf, we copy those files from ontonotes to ISNotes directory.
        :return:
        """
        wsj_numbers_list = self.wsj_files_numbers.split('|')
        wsj_onf_files = ["wsj_{}.onf".format(f) for f in wsj_numbers_list]

        assert len(wsj_numbers_list) == len(wsj_onf_files) == 50
        create_dir(self.dir_original)
        for f in wsj_onf_files:
            logger.debug(5 * "---")
            logger.debug("copying file {}".format(f))
            src_path = search_and_get_file_path(onto_notes_wsj_files, f)
            dst_path = os.path.join(self.dir_original, f)
            shutil.copy2(src_path, dst_path)

        logger.debug("copying completed")

    def execute_jars(self):
        """
        execute mma jar to create data files
        :return:
        """

        command = 'cd {}; java -jar OntoNotes2MMAX.jar {} {} {} {}'.format(is_notes_data_path, self.dir_original,
                                                                           self.dir_target,
                                                                           self.dir_resource_file,
                                                                           self.dir_is_annotation_trace)
        logger.debug("executing command {}".format(command))
        os.system(command)

        command = 'cd {}; java -jar OntoNotes2MMAXWOTrace.jar {} {} {} {}'.format(is_notes_data_path, self.dir_original,
                                                                                  self.dir_target_clean,
                                                                                  self.dir_resource_file,
                                                                                  self.dir_is_annotation_without_trace)
        logger.debug("executing command {}".format(command))
        os.system(command)

    def build_is_note_data_set(self):
        """
        execute all the steps one-by-one to generate dataset as mentioned in the reademe
        :return:
        """
        self.copy_files_from_onto_notes_to_is_notes()
        self.execute_jars()

    def _get_words_id_to_markable_map(self, words_file_path):
        '''
        read all words and corrosponding id from file words_file_path and create dictionary
        mapping from word id to object of class Word. The object contains information about word.

        In the words file word id is word_<number> we remove word and keep <number> as its id.
        :param words_file_path:
        :return:
        '''
        logger.debug("reading word file {}".format(words_file_path))
        xml_content = ET.parse(words_file_path)
        word_elems = xml_content.getroot().findall('word')
        word_id_to_mark_map = {}
        for word_pos_in_doc, w in enumerate(word_elems):
            logger.debug(5 * "----")
            id = w.get('id')
            word = w.text
            logger.debug("id {}, word {}, word position {}".format(id, word, word_pos_in_doc))
            assert word_pos_in_doc == get_id_number(id) - 1
            word_id_to_mark_map[get_id_number(id)] = Word(id, word, word_pos_in_doc)
        return word_id_to_mark_map

    def _get_sents_id_to_markable_map(self, sents_file_path):
        '''
        read all sentences and corrosponding id, spans from file sents_file_path and create dictionary
        mapping from sent id to object of class Word. The object contains information about start word index and
        end word index.
        :param sents_file_path:
        :return:
        '''
        logger.debug("reading sentence file {}".format(sents_file_path))
        xml_content = ET.parse(sents_file_path)
        sent_elems = xml_content.getroot().findall(xmlns_sent + 'markable')
        sent_id_to_mark_map = {}
        for sent_pos_in_doc, s in enumerate(sent_elems):
            logger.debug(5 * "----")
            id = s.get('id')
            try:
                start_word, end_word = s.get('span').split('..')
            except ValueError:
                logger.error("current sentence contains only one word in it. {}".format(s.get('span')))
                prev_end = sent_id_to_mark_map[sent_pos_in_doc - 1].end_word_id
                logger.error("previous sentence end is {}".format(prev_end))
                if get_id_number(s.get('span')) > prev_end + 1:
                    start_word = "word_{}".format(prev_end + 1)
                    end_word = s.get('span')
            logger.debug("id {}, start word {}, end word {}".format(id, start_word, end_word))
            assert sent_pos_in_doc == get_id_number(id)
            sent_id_to_mark_map[get_id_number(id)] = Sentence(id, start_word, end_word)
        return sent_id_to_mark_map

    def _get_entities_id_to_markable_map(self, entities_file_path):
        '''
        read all entities and create map from entity id to Entity object.
        Entity object contains its start and end word position of entity i.e. span of entity.
        :param entities_file_path:
        :return:
        '''
        logger.debug("reading entity file {}".format(entities_file_path))
        xml_content = ET.parse(entities_file_path)
        ent_elems = xml_content.getroot().findall(xmlns_ent + 'markable')
        entities_id_to_mark_map = {}
        for e in ent_elems:
            logger.debug(5 * "----")
            id = e.get('id')
            start_end_id = e.get('span').split('..')
            if len(start_end_id) == 2:
                start_word, end_word = start_end_id
            else:
                start_word, end_word = start_end_id[0], start_end_id[0]
            logger.debug("id {}, start word {}, end word {}".format(id, start_word, end_word))
            ent_pos_in_doc = get_id_number(id)
            entities_id_to_mark_map[ent_pos_in_doc] = Entity(id, start_word, end_word)
        return entities_id_to_mark_map

    def _get_coref_ids(self, coref_file_path):
        '''
        read all coreference entities and create a list of Entity objects containing coref information
        each Entity object will contain information -
            - id : id of cluster, this is arbitrary id assigned to all the entities from the same cluster.
            - start : start pos of the word
            - end : end position of the word
        :param coref_file_path:
        :return:
        '''
        logger.debug("reading coref file {}".format(coref_file_path))
        xml_content = ET.parse(coref_file_path)
        ent_elems = xml_content.getroot().findall(xmlns_coref + 'markable')
        coref_entities = []
        coref_set_to_id_map = {}
        id = 0
        for e in ent_elems:
            logger.debug(5 * "----")
            coref_set = e.get('coref_set')
            logger.debug("coref set {}".format(coref_set))
            if coref_set_to_id_map.get(coref_set, None) is None:
                coref_set_to_id_map[coref_set] = id
                id += 1
            curr_id = coref_set_to_id_map[coref_set]
            start_end_id = e.get('span').split('..')
            if len(start_end_id) == 2:
                start_word, end_word = start_end_id
            else:
                start_word, end_word = start_end_id[0], start_end_id[0]
            logger.debug("current id {}, start word {}, end word {}".format(curr_id, start_word, end_word))
            coref_entities.append(Entity(curr_id, start_word, end_word))
        return coref_entities

    def _set_entity_start_end_pos_in_doc(self, entities_id_to_mark_map, word_id_to_mark_map):
        '''
        :param entities_id_to_mark_map:
        :return:
        '''
        if isinstance(entities_id_to_mark_map, list):
            entities_id_to_mark_map = [e.set_ent_start_end_pos_in_doc(word_id_to_mark_map) for e in
                                       entities_id_to_mark_map]
        if isinstance(entities_id_to_mark_map, dict):
            for k in entities_id_to_mark_map.keys():
                entities_id_to_mark_map[k].set_ent_start_end_pos_in_doc(word_id_to_mark_map)

    def _get_bridges_id_cluster(self, entities_file_path):
        '''
        read entities file and return list of bridges.
        Each bridge will be list containing all the connected bridges.
        :param entities_file_path:
        :return:
        '''
        xml_content = ET.parse(entities_file_path)
        ent_elems = xml_content.getroot().findall(xmlns_ent + 'markable')
        bridges = []
        for e in ent_elems:
            id2 = e.get('id')
            if e.get('information_status') == 'mediated' and e.get('mediated_type') == 'bridging':
                logger.debug(5 * "----")
                id1 = e.get('bridged_from')
                logger.debug("{} is bridged to {}".format(id2, id1))
                bridges.append(Bridg(id1, id2))
        return bridges

    def get_sentence_list(self, sent_id_to_mark_map, word_id_to_mark_map):
        '''
        Method generates list of sentences which are present in the document.
        It accepts two dictionaries to generate that.
        :param sent_id_to_mark_map: it contains sentence's order number as key and Sentence class object as value.
                                    the object contains start aned end id of word in sentence.
        :param word_id_to_mark_map: this is dictionary where word id is a key Word object is value of each key
        :return: list of lists
        '''
        sentences = []
        for k in range(len(sent_id_to_mark_map)):
            start_word_id = sent_id_to_mark_map[k].start_word_id
            end_word_id = sent_id_to_mark_map[k].end_word_id

            assert end_word_id > start_word_id
            sentence = []
            for word_id in range(start_word_id, end_word_id + 1):
                sentence.append(word_id_to_mark_map[word_id].word)
            logger.debug("sentence is {}".format(sentence))
            sentences.append(sentence)
            assert end_word_id == len(flatten(sentences))
        return sentences

    def get_bridges_by_start_end_word_pos_in_doc(self, bridges_markable, entities_id_to_mark_map):
        bridges = []
        for bridge in bridges_markable:

            if isinstance(bridge, list):
                tuple_list = []
                for b in bridge:
                    tuple_list.append(entities_id_to_mark_map[b].get_ent_pos_in_doc())
            elif isinstance(bridge, Bridg):
                tuple_list = bridge.get_tuple_loc(entities_id_to_mark_map)
            else:
                raise NotImplementedError
            bridges.append(tuple_list)
        return bridges

    def _get_object_with_doc_name(self, json_objects, doc_name):
        for single_json_object in json_objects:
            if doc_name in single_json_object['doc_key']:
                return single_json_object
        return None

    def get_entities_start_end_tuple_list(self, entities_id_to_mark_map):
        """
        Get the list of tuples containing start,end position of entities.

        Accepts dictionary containing key as the order number of the entity in the document
        and value as its entity object. The object contains info about the entities start and end pos.
        :param entities_id_to_mark_map:
        :return:
        """
        return [entities_id_to_mark_map[id].get_ent_pos_in_doc() for id in entities_id_to_mark_map]

    def get_coref_cluster_list(self, coref_entity_obj_list):
        """
        Get the list of clusters containing coreferring entities.
        Each entity is a tuples containing start,end position of entity.

        Accepts list of Entity markable objects which contains coreference cluster information
        the 'id' in the Entity object refers to the id of coreference cluster. That entity is added to
        the same cluster.
        :param coref_entity_obj_list:
        :return:
        """
        id_to_cluster_map = {}
        for entity in coref_entity_obj_list:
            curr_id = entity.id
            curr_cluster = id_to_cluster_map.get(curr_id, [])
            curr_cluster.append(entity.get_ent_pos_in_doc())
            id_to_cluster_map[curr_id] = curr_cluster
        cluster_list = []
        for id in id_to_cluster_map:
            cluster_list.append(id_to_cluster_map[id])
        return cluster_list

    def build_json_file(self):
        '''
        we are creating jsonl file which contains information of all the files related to briding.

        This jsonl contains json object per line. The json object is nothing but a dictionary containing
        following keys
        1. doc_name - this will be wsj file name
        2. sentences - all the sentences from the file. This will be list of lists. each list is a sentence
        3. bridging_pairs - The list of lists. Eeach list will be containing two tuples showing information
                            bridging references.
        :return:
        '''
        # jsonlines_file_path = os.path.join(coref_data_path,train_jsonlines)
        jsonlines_file_path = os.path.join(coref_data_path, dev_jsonlines)
        coref_json_objects = read_jsonlines_file(jsonlines_file_path)
        jsonlines_file_path = os.path.join(coref_data_path, train_jsonlines)
        coref_json_objects += read_jsonlines_file(jsonlines_file_path)
        print(len(coref_json_objects))
        data = []
        # get all the files with mmax extension
        file_list = get_list_of_files_with_extn_in_dir(self.dir_target_clean, mmax)

        basedata_dir = os.path.join(self.dir_target_clean, "Basedata")
        markable_dir = os.path.join(self.dir_target_clean, "markables")
        num_json = 0
        for f in file_list:
            doc_name = f.split(".")[0]
            logger.debug("doc name {}".format(doc_name))
            coref_json_object = self._get_object_with_doc_name(coref_json_objects, doc_name)
            logger.debug("coref obj {}".format(coref_json_object))
            words_file_name = doc_name + words_file_extn
            words_file_path = os.path.join(basedata_dir, words_file_name)
            word_id_to_mark_map = self._get_words_id_to_markable_map(words_file_path)

            sents_file_name = doc_name + sents_file_extn
            sents_file_path = os.path.join(markable_dir, sents_file_name)
            sent_id_to_mark_map = self._get_sents_id_to_markable_map(sents_file_path)
            sentences = self.get_sentence_list(sent_id_to_mark_map, word_id_to_mark_map)

            entities_file_name = doc_name + entity_file_extn
            entities_file_path = os.path.join(markable_dir, entities_file_name)
            entities_id_to_mark_map = self._get_entities_id_to_markable_map(entities_file_path)
            self._set_entity_start_end_pos_in_doc(entities_id_to_mark_map, word_id_to_mark_map)
            entities = self.get_entities_start_end_tuple_list(entities_id_to_mark_map)
            logger.debug("entities {}".format(entities))

            coref_file_name = doc_name + coref_file_extn
            coref_file_path = os.path.join(markable_dir, coref_file_name)
            coref_entities = self._get_coref_ids(coref_file_path)
            self._set_entity_start_end_pos_in_doc(coref_entities, word_id_to_mark_map)
            coref_clusters = self.get_coref_cluster_list(coref_entities)
            logger.debug("coref clusters {}".format(coref_clusters))

            # if coref_json_object is not None:
            assert len(sentences) == len(coref_json_object['sentences'])
            logger.debug("coref clusters form coref json {}".format(coref_json_object['clusters']))
            # assert len(coref_clusters) == len(coref_json_object['clusters'])

            bridges_markable = self._get_bridges_id_cluster(entities_file_path)
            bridges = self.get_bridges_by_start_end_word_pos_in_doc(bridges_markable, entities_id_to_mark_map)
            logger.debug("bridges {}".format(bridges))

            art_bridging_pairs_good, art_bridging_pairs_bad, art_bridging_pairs_ugly = generate_artificial_bridging_links(
                bridging_clusters=bridges, coref_clusters=coref_clusters)

            json_object = {}
            json_object['doc_key'] = "ISNotes_" + doc_name
            json_object['sentences'] = sentences
            json_object['bridging'] = bridges
            json_object['clusters'] = coref_clusters
            json_object['entities'] = entities
            json_object['pos'] = coref_json_object['pos']
            json_object['parse_bit'] = coref_json_object['parse_bit']
            json_object['art_bridging_pairs_good'] = art_bridging_pairs_good
            json_object['art_bridging_pairs_bad'] = art_bridging_pairs_bad
            json_object['art_bridging_pairs_ugly'] = art_bridging_pairs_ugly
            data.append(json_object)
            logger.debug(5 * "---")

        json_lines_file_path = os.path.join(is_notes_data_path, is_notes_jsonlines)
        write_jsonlines_file(data, json_lines_file_path)

    def get_train_test_files_names(self):
        '''
        we are going to devide files into training files and test files. Do this division and return names
        of files
        :return:
        '''
        file_list = get_list_of_files_with_extn_in_dir(self.dir_target_clean, mmax)
        file_list = file_list[:-1]  # so as to remove last file for which we dont have json object
        file_list = [f.split(".")[0] for f in file_list]
        assert len(file_list) == 49
        # now split
        train_files_name = ["ISNotes_" + f + ".txt" for f in file_list if f in train_files]
        test_files_name = ["ISNotes_" + f + ".txt" for f in file_list if f not in train_files]
        return train_files_name, test_files_name

    def generate_separte_jsonlines_train_dev_test(self):
        '''
        read ISNotes jsonlines file and separate objects of training and test.
        create two separate files for this.
        :return:
        '''
        json_lines_file_path = os.path.join(is_notes_data_path, is_notes_jsonlines)
        json_objects = read_jsonlines_file(json_lines_file_path)

        train_data = []
        dev_data = []
        test_data = []

        for single_json_object in json_objects:
            doc_name = single_json_object['doc_key']
            if doc_name[8:] in dev_files:
                dev_data.append(single_json_object)
            elif doc_name[8:] in test_files:
                test_data.append(single_json_object)
            else:
                train_data.append(single_json_object)

        logger.debug("number of total objects {}".format(len(json_objects)))
        logger.debug("train objects {} and test objects {}".format(len(train_data), len(dev_data + test_data)))
        assert len(json_objects) == len(train_data) + len(test_data) + len(dev_data)

        train_json_lines_file_path = os.path.join(is_notes_data_path, is_notes_train_jsonlines)
        # dev_json_lines_file_path = os.path.join(is_notes_data_path, is_notes_dev_jsonlines)
        test_json_lines_file_path = os.path.join(is_notes_data_path, is_notes_test_jsonlines)
        # TODO write different code
        test_data = dev_data + test_data
        write_jsonlines_file(train_data, train_json_lines_file_path)
        # write_jsonlines_file(dev_data, dev_json_lines_file_path)
        write_jsonlines_file(test_data, test_json_lines_file_path)

    def analyse_dataset(self):
        """
        See what kind of mentions are linked to each other by bridging.
        general analysis of the dataset to get the idea of the dataset.
        :return:
        """

        def _is_men_in_entity(men, entity):
            s, e = men
            # men = tuple(men)
            return [s, e + 1] in entity

        json_lines_file_path = os.path.join(is_notes_data_path, is_notes_jsonlines)
        json_objects = read_jsonlines_file(json_lines_file_path)
        # print(len(json_objects))
        ment_pos_dict = {}
        ent_pos_dict = {}
        missing_parse = {}
        missing_mentions = 0
        total_mentions = 0
        total_pases = {}
        total_cand_ant = 0
        total_ana_ante_pairs = 0
        for single_json_object in json_objects:
            mention_ops = MentionFileOperations(single_json_object)
            logger.debug("")
            if not mention_ops.is_file_empty:
                logger.debug("mention indices {}".format(mention_ops.mention_indices))
                # for men_ind,(m,mh,ssent) in enumerate(zip(mention_ops.mention_words_list,mention_ops.mention_heads,mention_ops.mention_sents)):
                #     logger.debug(5*"----")
                #     logger.debug("mention : {}".format(m))
                #     logger.debug("head : {}".format(mh))
                #     ssent = remove_markers(ssent)
                #     logger.debug("sentence : {}".format(ssent))
                #     men_sent = mention_ops.mention_to_sentence_map[men_ind]
                #     logger.debug("mention is present in {} th sentence".format(men_sent))
                #     sentence_from_map = " ".join(mention_ops.sentences[men_sent])
                #     logger.debug("sentence from map is {}".format(sentence_from_map))
                #     assert mention_ops.sentences[men_sent] == ssent
                # for sent_cluster in mention_ops.mention_cluster_with_context:
                #     logger.debug("sent 1: {}".format(sent_cluster[0]))
                #     logger.debug("sent 2: {}".format(sent_cluster[1]))

                logger.debug("clusters{}".format(mention_ops.clusters))
                logger.debug("entities {}".format(mention_ops.entities))
                anaphors = []
                antecedents = []
                total_ana_ante_pairs += len(mention_ops.clusters)
                for cl in mention_ops.clusters:
                    # logger.debug("anaphor {} in entity {}".format(cl[1],_is_men_in_entity(cl[1],mention_ops.entities)))
                    # logger.debug("antecedent {} in entity {}".format(cl[0],_is_men_in_entity(cl[0], mention_ops.entities)))
                    assert _is_men_in_entity(cl[1], mention_ops.entities), "anaphor {} not in entity".format(cl[1])
                    assert _is_men_in_entity(cl[0], mention_ops.entities), "antecedent {} not in entity".format(cl[0])
                    anaphors.append(cl[1])
                    antecedents.append(cl[0])
                logger.debug("antecedents {}".format(antecedents))
                logger.debug("anaphors {}".format(anaphors))
                cand_ante = {}
                for ent_ind, ent in enumerate(mention_ops.entities_words_list):
                    # logger.debug(5 * "----")
                    # logger.debug("entity : {}".format(ent))
                    ent_sent = mention_ops.entity_to_sentence_map[ent_ind]
                    sentence_from_map = " ".join(mention_ops.sentences[ent_sent])
                    # logger.debug("sentence from map is {}".format(sentence_from_map))
                    start_pos = mention_ops.entities[ent_ind][0]
                    # logger.debug("pos parse bit of the start word {}".format(mention_ops.parse_bit_list[start_pos]))
                    # logger.debug("pos of the start word {}".format(mention_ops.pos_list[start_pos]))
                    ent_pos_dict[mention_ops.parse_bit_list[start_pos]] = 1
                    current_parse = self._get_current_parse(mention_ops.parse_bit_list[start_pos])
                    if current_parse == "NP*" or current_parse == "NML*":
                        cand_ante[tuple(mention_ops.entities[ent_ind])] = mention_ops.entity_to_sentence_map[ent_ind]
                    total_pases[current_parse] = total_pases.get(current_parse, 0) + 1
                # for men_start_ind in mention_ops.mention_indices:
                logger.debug("candidate antecedents {}".format(cand_ante))
                for men_start_ind in antecedents:
                    total_mentions += 1
                    logger.debug('men ind {}'.format(men_start_ind))
                    men_start_ind[-1] += 1
                    if tuple(men_start_ind) not in cand_ante:
                        last_parse_word = self._get_current_parse(mention_ops.parse_bit_list[men_start_ind[0]])
                        missing_parse[last_parse_word] = missing_parse.get(last_parse_word, 0) + 1
                        missing_mentions += 1
                    # ment_pos_dict[mention_ops.parse_bit_list[men_start_ind[0]]] = 1
                total_cand_ant += len(cand_ante)
        # cluster_stat.get_statistics_for_dataset(json_objects=json_objects)
        # logger.debug("mention pos {}".format(ment_pos_dict.keys()))
        # logger.debug("entity pos {}".format(ent_pos_dict.keys()))
        logger.debug("total anaphor antecedent pairs {}".format(total_ana_ante_pairs))
        logger.debug("total parses {}".format(total_pases))
        logger.debug("missing parses {}".format(missing_parse))
        logger.debug("total candidate antecedents {}".format(total_cand_ant))
        logger.debug("total mentions {} and missing mentions {}".format(total_mentions, missing_mentions))

    def _get_current_parse(self, parse_bit):
        parse_bit_list = []
        pos = ""
        for ch in parse_bit:
            if ch == "(" or ch == ")":
                if len(pos) > 0:
                    parse_bit_list.append(pos)
                    pos = ""
            else:
                pos += ch
        if len(pos) > 0:
            parse_bit_list.append(pos)
        # logger.debug("parse bit {} and last parse {}".format(parse_bit,parse_bit_list[-1]))
        return parse_bit_list.pop()


class BASHI:
    def __init__(self):
        self.bashi_conll = bashi_conll_file_path

    def get_object_with_doc_name(self, json_objects, doc_name):
        for single_json_object in json_objects:
            if doc_name == single_json_object['doc_key']:
                return single_json_object
        return None

    def sentences_to_doc_pos(self, sentences):
        word_counter = 0
        sentences_to_word_doc_pos = []
        for sentence in sentences:
            current_sentence_word_pos = []
            for word in sentence:
                current_sentence_word_pos.append(word_counter)
                word_counter += 1
            sentences_to_word_doc_pos.append(current_sentence_word_pos)
        assert len(sentences_to_word_doc_pos) == len(sentences)
        return sentences_to_word_doc_pos

    def get_bridging_pairs(self, bashi_anaphors_markables, sentences_to_word_doc_pos):
        bridges = []
        bridg_type = []
        for m in bashi_anaphors_markables:
            if anaphor_types[2] in m.ana_type and anaphor_types[0] not in m.ana_type:
                m.set_start_end(sentences_to_word_doc_pos)
                bridges.append(m.get_bridging_pair())
                bridg_type.append(m.ana_type)
            # print(m)
        return bridges, bridg_type

    def build_json_file(self):
        '''

        :return:
        '''
        jsonlines_file_path = os.path.join(coref_data_path, dev_jsonlines)
        coref_json_objects = read_jsonlines_file(jsonlines_file_path)
        jsonlines_file_path = os.path.join(coref_data_path, train_jsonlines)
        coref_json_objects += read_jsonlines_file(jsonlines_file_path)
        print(len(coref_json_objects))
        data = []
        conll_bashi_data = read_text_file(self.bashi_conll)
        total_num_ana = 0
        for line_num, line in enumerate(conll_bashi_data):

            if line.startswith(begin_doc):
                line_list = line.split()
                assert " ".join(line_list[0:2]) == begin_doc
                doc_name = line_list[2][1:-2] + "_0"

                json_object = self.get_object_with_doc_name(coref_json_objects, doc_name)
                assert json_object is not None, "doc name is {}".format(doc_name)

                doc_name = "BASHI_" + line_list[2][1:-2].split("/")[-1]
                sentences = json_object['sentences']
                coref_clusters = json_object['clusters']
                sentences_to_word_doc_pos = self.sentences_to_doc_pos(sentences)

                bashi_anaphors_markables = []
                curr_doc_sentence_num = 0
                num_anaphors = 0
                start_of_anaphor = -1
                logger.debug(3 * "====")
                logger.debug("current doc {}".format(doc_name))

            elif line.startswith(end_doc):
                assert num_anaphors == len(bashi_anaphors_markables), "number of anaphors {} and markables {}".format(
                    num_anaphors, len(bashi_anaphors_markables))
                assert curr_doc_sentence_num == len(sentences)

                bridges, bridg_type = self.get_bridging_pairs(bashi_anaphors_markables, sentences_to_word_doc_pos)
                num_ana = len(bridges)
                total_num_ana += num_ana
                logger.debug("bridging anaphors {}".format(bridges))
                art_bridging_pairs_good, art_bridging_pairs_bad, art_bridging_pairs_ugly = generate_artificial_bridging_links(
                    bridging_clusters=bridges, coref_clusters=coref_clusters)

                # json_object = {}
                json_object['doc_key'] = doc_name
                json_object['bridging'] = bridges
                json_object['bridg_type'] = bridg_type
                json_object['num_ana'] = num_ana
                json_object['art_bridging_pairs_good'] = art_bridging_pairs_good
                json_object['art_bridging_pairs_bad'] = art_bridging_pairs_bad
                json_object['art_bridging_pairs_ugly'] = art_bridging_pairs_ugly
                data.append(json_object)
                logger.debug(5 * "---")

            elif line == "\n":
                assert line.strip() == ""
                assert curr_word + 1 == len(sentences[curr_doc_sentence_num])
                curr_doc_sentence_num += 1
                assert start_of_anaphor == -1

            elif len(line.split()) >= 5:
                if len(line.split()) == 5:
                    line += " -"
                line_split = line.split()
                curr_word = int(line_split[2])
                if line_split[3][0] == "(" or line_split[4][0] == "(" or line_split[5][0] == "(":
                    # start of anaphor
                    assert start_of_anaphor == -1
                    num_anaphors += 1
                    start_of_anaphor = curr_word
                    ana_sentence = curr_doc_sentence_num
                    anaphor_type = []
                    bridging_info = []
                    if line_split[3] != "-":  # GEN type anaphor
                        anaphor_type.append(anaphor_types[0])
                        bridging_info.append(line_split[3])
                    if line_split[4] != "-":  # INDEF type anaphor
                        anaphor_type.append(anaphor_types[1])
                        bridging_info.append(line_split[4])
                    if line_split[5] != "-":  # COMP type anaphor
                        anaphor_type.append(anaphor_types[2])
                        bridging_info.append(line_split[5])

                    anaphor_type = "_".join(anaphor_type)
                    assert len(set(bridging_info)) == 1
                    bridging_info = bridging_info[0]
                    bri_info = bridging_info.split('$')
                    ana_id = bri_info[1]
                    ante_sent, ante_start, ante_end = bri_info[2].split('-')
                    if ante_end[-1] == ')': ante_end = ante_end[0:-1]
                    logger.debug("line num : {}, id {}, ante sente : {}, start : {}, end : {}".format(line_num, ana_id,
                                                                                                      ante_sent,
                                                                                                      ante_start,
                                                                                                      ante_end))

                if line_split[3][-1] == ")" or line_split[4][-1] == ")" or line_split[5][-1] == ")":
                    # end of anaphor
                    try:
                        assert start_of_anaphor != -1
                    except:
                        print("anaphor ending at but not started anywhere line num {} ".format(line_num))
                        continue
                    end_of_anaphor = curr_word
                    bridging_end_info = []
                    if line_split[3] != "-":  # GEN type anaphor
                        assert anaphor_types[0] in anaphor_type
                        bridging_end_info.append(line_split[3])
                    if line_split[4] != "-":  # INDEF type anaphor
                        assert anaphor_types[1] in anaphor_type
                        bridging_end_info.append(line_split[4])
                    if line_split[5] != "-":  # COMP type anaphor
                        assert anaphor_types[2] in anaphor_type
                        bridging_end_info.append(line_split[5])
                    assert len(set(bridging_end_info)) == 1

                    bridging_end_info = bridging_end_info[0][0:-1]
                    bridging_end_ana_id = bridging_end_info.split('$')[1]

                    assert bridging_end_ana_id == ana_id

                    bashi_anaphors_markables.append(
                        Anaphors_bashi(doc_name, ana_id, anaphor_type, ana_sentence, start_of_anaphor, end_of_anaphor,
                                       ante_sent, ante_start, ante_end))
                    logger.debug(
                        "line num : {}, id {}, anaphor type : {}, sente : {}, start : {}, end : {}, ante sente : {}, start : {}, end : {}".format(
                            line_num, ana_id, anaphor_type, ana_sentence, start_of_anaphor, end_of_anaphor, ante_sent,
                            ante_start, ante_end))
                    start_of_anaphor = -1

            else:
                print("line length {} and line is {}".format(len(line.split()), line))
        json_lines_file_path = os.path.join(bashi_data_path, bashi_jsonlines)
        assert len(data) == 50
        assert total_num_ana == 349, "total ana {}".format(total_num_ana)
        write_jsonlines_file(data, json_lines_file_path)

    def bashi_data(self):
        bashi_json_objects = read_jsonlines_file(os.path.join(bashi_data_path, bashi_jsonlines))
        for single_json_object in bashi_json_objects:
            json_info_extr = BridgingJsonDocInformationExtractor(single_json_object, logger)


class ARRAU:
    train_data_folder = "ARRAU_Train_v4"
    test_data_folder = "ARRAU_Test_Gold_v2"

    pear = "Pear_Stories"
    rst = "RST_DTreeBank"
    trains = "Trains_93"

    def __init__(self, arrau_data_path, target_jsonlines):
        self.arrau_data_path = arrau_data_path
        self.target_jsonlines = target_jsonlines

        self.num_empty_anaphors = 0
        self.num_multiple_antecedents_anaphors = 0
        self.num_docs = [0,0,0]
        self.num_empty_anaphors_dataset = [0, 0, 0]

    def _get_mmax_path(self, conll_file_path):
        if ARRAU.train_data_folder in conll_file_path:
            return arrau_train_mmax_path
        else:
            return arrau_test_mmax_path

    def _set_dataset_name_and_type(self, conll_file_path):
        if ARRAU.pear in conll_file_path:
            self.dataset_name = PEAR
            self.num_docs[2] += 1
            self._data_set_ind = 2
        elif ARRAU.rst in conll_file_path:
            self.dataset_name = RST
            self.num_docs[0] += 1
            self._data_set_ind = 0
        elif ARRAU.trains in conll_file_path:
            self.dataset_name = TRAINS
            self.num_docs[1] += 1
            self._data_set_ind = 1
        else:
            raise NotImplementedError

        if ARRAU.train_data_folder in conll_file_path:
            self.data_type = "train"
            if "dev" in conll_file_path:
                self.data_type = "dev"
        else:
            self.data_type = "test"

    def _get_actual_filename(self, fn):
        # remove .CONLL extension
        return fn.replace(".CONLL", "")

    def read_conll_file(self, file_path):
        """
        The file contains four columns -
            1. tokens
            2. indicator for token being mention and then corresponding coreference chain set
            3. if mention then its head word
            4. information bridging reference if its anaphor
        After reading the file, returns 4 lists containing all the content of columns
        :param file_path:
        :return:
        """
        conll_data = read_text_file(file_path)
        tokens = []
        mention_info = []
        head_info = []
        bridging_info = []
        for line_num, line in enumerate(conll_data):
            if line_num == 0: continue
            line_content = line.split("\t")
            assert len(line_content) == 4, "file {} - line number {} does not have 4 columns {}".format(file_path,
                                                                                                        line_num,
                                                                                                        line_content)
            token, mention, head, bridg = line.split("\t")
            tokens.append(token)
            mention_info.append(mention)
            head_info.append(head)
            bridging_info.append(bridg)
        assert len(tokens) == len(mention_info) == len(head_info) == len(bridging_info)
        return tokens, mention_info, head_info, bridging_info

    def is_token_markable(self, tok_nm):
        if len(self.mention_info[tok_nm]) > 0:
            assert "markable" in self.mention_info[tok_nm], "markable word not in {} at line {}".format(
                self.mention_info[tok_nm], tok_nm)
            return True
        return False

    def _get_actual_id_from_marakable_id(self, markable_id):
        if markable_id.startswith("B-") or markable_id.startswith("I-"):
            return markable_id[2:]
        return markable_id

    def _is_markable_begin(self, markable_id):
        if markable_id[0] == 'B':
            is_markable_begin = True
        elif markable_id[0] == 'I':
            is_markable_begin = False
        else:
            is_markable_begin = None
        return is_markable_begin

    def get_markable_id_coref_set_id(self, markable):
        markable_id, coref_set = markable.split("=")
        is_markable_begin = True if markable_id[0] == 'B' else False
        markable_id = self._get_actual_id_from_marakable_id(markable_id)
        return is_markable_begin, markable_id, coref_set

    def get_all_markables_at_tok_num(self, tok_nm):
        markables = self.mention_info[tok_nm].split("@")
        head_words = self.head_info[tok_nm].split("@")
        bridgings = self.bridging_info[tok_nm].strip().split("@")

        assert len(markables) == len(head_words), "markables {} and heads {} are not of same length".format(markables,
                                                                                                            head_words)
        logger.debug(
            "at line {}, markable {} , head {} and bridging {}".format(tok_nm, markables, head_words, bridgings))
        return markables, head_words, bridgings

    def get_bridging_pairs(self, brdgs):
        # brdgs = brdg.split("&")
        if len(brdgs) == 1 and brdgs[0] == '': return {}
        brd_ana_id_to_ante = {}
        rels = ["==", "=subset=", "=subset-inv=", "=element=", "=element-inv=", "=other=", "=other-inv=",
                "=undersp-rel=", "=poss=", "=poss-inv=", "=unmarked="]

        for b_pairs in brdgs:
            logger.debug("bridging pair {}".format(b_pairs))
            is_not_found_rel = True
            for i, rel in enumerate(rels):
                if rel in b_pairs:
                    anaphor_markable_id, antecedent_markable = b_pairs.split(rel)
                    bridg_rel = bridg_rels[i]
                    is_not_found_rel = False
                    break
            if is_not_found_rel:
                print("{} : rel not found".format(b_pairs))
                raise AssertionError

            is_ana_begin = self._is_markable_begin(anaphor_markable_id)
            anaphor_markable_id = self._get_actual_id_from_marakable_id(anaphor_markable_id)
            _, ante_markable_id, ante_coref_set = self.get_markable_id_coref_set_id(antecedent_markable)
            assert brd_ana_id_to_ante.get(anaphor_markable_id, None) is None
            if ante_markable_id == 'empty':
                if is_ana_begin:
                    logger.error("ERROR : EMPTY ANAPHOR : {}".format(b_pairs))
                    self.num_empty_anaphors += 1
                    self.num_empty_anaphors_dataset[self._data_set_ind] +=1
                continue
            # TODO anaphor is having two antecedents, handle that case differently
            if ";" in ante_markable_id:
                if is_ana_begin:
                    self.num_multiple_antecedents_anaphors += 1
                    logger.error("ERROR : MULTIPLE ANTE : {}".format(b_pairs))
                ante_markable_id = ante_markable_id.split(";")[0]
            brd_ana_id_to_ante[anaphor_markable_id] = [ante_markable_id, ante_coref_set, bridg_rel]
        return brd_ana_id_to_ante

    def _get_head_pos(self, hw):
        if '..' in hw:
            # few times head words are two, connected with '..'
            # we consider the first one
            logger.error("{} contains two head words".format(hw))
            hw = hw.split("..")[0]
        try:
            w, id = hw.split("_")
        except ValueError:
            print("ERROR : Unable to split : {} with '_'. Head word number not found.".format(hw))
            return None
        assert w == "word", "{} no word in it".format(hw)
        id = int(id)
        id = id - 1
        return id

    def close_current_markables(self, current_markables, tok_nm, markables=None):
        dont_close_ids = []
        last_tok_nm = tok_nm - 1
        if markables is not None:
            for markable in markables:
                _, markable_id, _ = self.get_markable_id_coref_set_id(markable)
                dont_close_ids.append(markable_id)
        for cm in list(current_markables):
            if cm not in dont_close_ids:
                current_markables[cm].close_markable(last_tok_nm)
                del current_markables[cm]
        return current_markables

    def get_sents_from_tokens(self, sents_file_path, tokens):
        '''
        read all sentences and corrosponding id, spans from file sents_file_path and create dictionary
        mapping from sent id to object of class Word. The object contains information about start word index and
        end word index.
        :param sents_file_path:
        :return:
        '''
        logger.debug("reading sentence file {}".format(sents_file_path))
        xml_content = ET.parse(sents_file_path)
        sent_elems = xml_content.getroot().findall(xmlns_sent + 'markable')
        sentences = []
        for sent_pos_in_doc, s in enumerate(sent_elems):
            logger.debug(5 * "----")
            span = s.get('span')
            if '..' in span:
                start_word, end_word = span.split('..')
            else:
                start_word = span
                end_word = span
            start_word = get_id_number(start_word) - 1
            end_word = get_id_number(end_word)
            sentences.append(tokens[start_word:end_word])
        return sentences

    def add_to_current_markables(self, tok_nm, mk, hw, brd_ana_id_to_ante, current_markables, markables_str_to_obj,
                                 coref_id_to_markables):
        is_markable_begin, markable_id, coref_set = self.get_markable_id_coref_set_id(mk)
        hw = self._get_head_pos(hw)
        if is_markable_begin:
            # check coreference chain information
            if coref_id_to_markables.get(coref_set, None) is not None:
                logger.debug("chain for {} already exists {}".format(coref_set, coref_id_to_markables[coref_set]))
                assert markable_id not in coref_id_to_markables.get(coref_set,
                                                                    None), "{} already present in coref {}".format(
                    markable_id, coref_id_to_markables.get(coref_set, None))
                coref_id_to_markables[coref_set].append(markable_id)
            else:
                logger.debug("creating new chain for {}".format(coref_set))
                coref_id_to_markables[coref_set] = [markable_id]
            assert coref_id_to_markables.get(coref_set, None) is not None
            logger.debug("new markable {} added to coref set {}, now coref chain is {}".format(markable_id, coref_set,
                                                                                               coref_id_to_markables[
                                                                                                   coref_set]))
            # create markable object
            assert markables_str_to_obj.get(markable_id, None) is None, "{} already present".format(markable_id)
            assert current_markables.get(markable_id, None) is None

            is_bridg, antecedent_id, antecedent_chain, bridg_rel = False, None, None, None

            if brd_ana_id_to_ante.get(markable_id, None) is not None:
                is_bridg = True
                antecedent_id, antecedent_chain, bridg_rel = brd_ana_id_to_ante.get(markable_id, None)
            mark_obj = ARRAUMarkable(markable_id, tok_nm, hw, coref_set, is_bridg, antecedent_id, antecedent_chain,
                                     bridg_rel)

            current_markables[markable_id] = mark_obj
            markables_str_to_obj[markable_id] = mark_obj

        else:
            assert markables_str_to_obj.get(markable_id, None) is not None
            assert markable_id in coref_id_to_markables[coref_set], "{} is not present in coref {}".format(
                markable_id, coref_id_to_markables[coref_set])
            assert current_markables.get(markable_id,
                                         None) is not None, "{} markable id not present in the current".format(
                markable_id)
            mark_obj = current_markables[markable_id]

            assert mark_obj.id == markable_id
            assert mark_obj.start_pos < tok_nm
            assert mark_obj.head_pos == hw
            assert mark_obj.coref_chain_id == coref_set

        return current_markables, markables_str_to_obj, coref_id_to_markables

    def process_conll_file(self, conll_file_path):
        logger.debug(5 * "****")
        logger.debug("Processing : {}".format(conll_file_path))
        logger.debug(5 * "****")

        self._set_dataset_name_and_type(conll_file_path)
        conll_fn = os.path.basename(conll_file_path)
        actual_fn = self._get_actual_filename(conll_fn)
        # conll_file_path = os.path.join(self.arrau_conll_path,conll_fn)
        logger.critical("----")
        logger.critical(
            "conll path :{} \n conll fn :{} \n actual fn :{} \n dataset :{} \n data type :{}".format(conll_file_path,
                                                                                                     conll_fn,
                                                                                                     actual_fn,
                                                                                                     self.dataset_name,
                                                                                                     self.data_type))
        logger.critical("----")

        self.tokens, self.mention_info, self.head_info, self.bridging_info = self.read_conll_file(conll_file_path)
        num_tokens = len(self.tokens)

        current_markables, markables_str_to_obj, coref_id_to_markables = {}, {}, {}
        for tok_nm in range(num_tokens):
            if self.is_token_markable(tok_nm):
                markables, head_words, bridgings = self.get_all_markables_at_tok_num(tok_nm)
                brd_ana_id_to_ante = self.get_bridging_pairs(bridgings)
                for mk, hw in zip(markables, head_words):
                    current_markables, markables_str_to_obj, coref_id_to_markables = self.add_to_current_markables(
                        tok_nm, mk, hw, brd_ana_id_to_ante, current_markables, markables_str_to_obj,
                        coref_id_to_markables)

                current_markables = self.close_current_markables(current_markables, tok_nm, markables)
            else:
                current_markables = self.close_current_markables(current_markables, tok_nm)
                assert len(current_markables) == 0, "non-zero current markables {}".format(current_markables)
        # assert len(current_markables) <= 1
        # if len(current_markables) >= 1:
        self.close_current_markables(current_markables, num_tokens)

        # search for sentence file and provide to function to get sentence boundaries over tokens
        arrau_mmax_path = self._get_mmax_path(conll_file_path)
        sents_file_path = os.path.join(arrau_mmax_path, actual_fn + sents_file_extn)
        sentences = self.get_sents_from_tokens(sents_file_path, self.tokens)

        clusters = []
        for coref_id in coref_id_to_markables.keys():
            markables = coref_id_to_markables[coref_id]
            cluster = [markables_str_to_obj[m].get_start_end() for m in markables]
            clusters.append(cluster)

        bridging = []
        bridging_rels = []
        entities = []
        entity_heads = []
        for markable in markables_str_to_obj.keys():
            logger.debug(2 * "--")
            mark_obj = markables_str_to_obj[markable]
            logger.debug("{}".format(mark_obj))
            entities.append(mark_obj.get_start_end())
            entity_heads.append(mark_obj.get_head())
            if mark_obj.is_bridging_anaphor:
                anaphor_position = mark_obj.get_start_end()
                ante_mark = mark_obj.antecedent_id
                assert markables_str_to_obj.get(ante_mark, None), "{} no object".format(ante_mark)
                ante_position = markables_str_to_obj[ante_mark].get_start_end()
                bridging.append([ante_position, anaphor_position])
                bridging_rels.append(mark_obj.bridg_rel)

        art_bridging_pairs_good, art_bridging_pairs_bad, art_bridging_pairs_ugly = generate_artificial_bridging_links(
            bridging, clusters)

        json_object = {}
        json_object['doc_key'] = "ARRAU_" + actual_fn
        json_object['dataset_name'] = self.dataset_name
        json_object['dataset_type'] = self.data_type
        json_object['sentences'] = sentences
        json_object['bridging'] = bridging
        json_object['bridging_rels'] = bridging_rels
        json_object['clusters'] = clusters
        json_object['entities'] = entities
        json_object['entity_heads'] = entity_heads
        # json_object['pos'] = coref_json_object['pos']
        # json_object['parse_bit'] = coref_json_object['parse_bit']
        json_object['art_bridging_pairs_good'] = art_bridging_pairs_good
        json_object['art_bridging_pairs_bad'] = art_bridging_pairs_bad
        json_object['art_bridging_pairs_ugly'] = art_bridging_pairs_ugly
        return json_object

    def generate_jsonlines_from_all(self):
        dirname = "CRAC18_Task2"
        crac_2018_task_2_dirs = get_paths_of_dir_names(self.arrau_data_path, dirname)
        conll_file_paths = []
        for arrau_conll_path in crac_2018_task_2_dirs:
            conll_file_paths += get_list_of_file_paths_with_extn_in_dir(arrau_conll_path, ".CONLL")

        data = []
        for counter, conll_file_path in enumerate(conll_file_paths):
            if counter % 20 == 0:
                print("remaining : {}".format(len(conll_file_paths) - counter))
                print("-----")
            data.append(self.process_conll_file(conll_file_path))
        print("Number of CONLL files : {}".format(len(conll_file_paths)))
        print("RST : {}, TRAINS : {} and PEAR : {}".format(*self.num_docs))
        print("total empty anaphors : {} and total multiple antecedents anaphors : {}".format(self.num_empty_anaphors,
                                                                                              self.num_multiple_antecedents_anaphors))
        print("Empty anaphors : RST : {}, TRAINS : {} and PEAR : {}, total : {}".format(*self.num_empty_anaphors_dataset,sum(self.num_empty_anaphors_dataset)))
        write_jsonlines_file(data, self.target_jsonlines)


if __name__ == '__main__':
    # is_note = ISNotes()
    # is_note.build_is_note_data_set()
    # is_note.build_json_file()
    # is_note.generate_separte_jsonlines_train_dev_test()
    # is_note.analyse_dataset()
    # bashi = BASHI()
    # bashi.build_json_file()
    # bashi.bashi_data()

    # arrau_train_json_path = os.path.join(arrau_train_data_path,arrau_jsonlines)
    # arr = ARRAU(arrau_train_conll_path,arrau_train_mmax_path,arrau_train_json_path)
    # arr.generate_jsonlines_from_all()

    arrau_json_path = os.path.join(arrau_data_path, arrau_jsonlines)
    # arr = ARRAU(arrau_test_conll_path,arrau_test_mmax_path,arrau_test_json_path)
    arr = ARRAU(arrau_data_path, arrau_json_path)
    arr.generate_jsonlines_from_all()
