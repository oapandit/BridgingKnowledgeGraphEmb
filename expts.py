from coref_code import *
from bridging_constants import *
from bridging_utils import *
from mention_operations import MentionFileOperations
from data import Data
from ffnn_expts import FFNNExpts
from corpus import ISNotes

if __name__ == '__main__':
    # json_objects = read_jsonlines_file(is_notes_json_f_path)
    # print(len(json_objects))
    # num_obj = 0
    # for json_object in json_objects:
    #     mfo = MentionFileOperations(json_object)
    #     if not mfo.is_file_empty:
    #         logger.debug("mention indices {}".format(mfo.mention_indices))
    #         for m,mh,ssent in zip(mfo.mention_words_list,mfo.mention_heads,mfo.mention_sents):
    #             logger.debug(5*"----")
    #             logger.debug("mention : {}".format(m))
    #             logger.debug("head : {}".format(mh))
    #             logger.debug("sentence : {}".format(ssent))
    #         num_obj+=1
    # print(num_obj)

    data = Data(json_objects_path=is_notes_data_path,wordnet_vector_data=is_notes_vec_path,emb_data_path=is_notes_vec_path)
    # data.generate_path2vec_and_bert_vectors(is_notes_train_jsonlines, path2vec_paths[0], is_one_sense=False, is_force=True,is_test=False)
    # data.generate_path2vec_and_bert_vectors(is_notes_test_jsonlines, path2vec_paths[0], is_one_sense=False,
    #                                         is_force=True, is_test=True)
    # data.generate_path2vec_and_bert_vectors(is_notes_jsonlines,path2vec_paths[0])
    data.generate_word2vec_glove_data(is_notes_jsonlines)

    # ffnn = FFNNExpts(json_objects_path=is_notes_data_path, wordnet_vector_data=is_notes_vec_path, save_model_path=result_path, t_json=is_notes_train_jsonlines,d_json=is_notes_test_jsonlines)
    # ffnn.simple_ffnn_wordnet_vec_expt()
    # ffnn.simple_ffnn_bert_vec_expt()
    # ffnn.simple_ffnn_pathvec_bert_vec_expt()

    # data = Data(json_objects_path=is_notes_data_path, wordnet_vector_data=is_notes_vec_path)
    # data.get_stats_mention(is_notes_jsonlines, path2vec_paths[0])
