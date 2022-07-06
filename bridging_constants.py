import os

# PATHS

base_dir = os.environ['base_dir']

# directories
data_path = os.path.join(base_dir, "Data/bridging")
result_path = os.path.join(base_dir, "Results/bridging")
log_files_path = os.path.join(base_dir, "Logs/bridging")
tf_trained_models_path = os.path.join(result_path,"tf_trained_models")
auto_encoder_trained_models_path = os.path.join(result_path,"ae_trained_models")
bert_analysis_path = os.path.join(result_path,"bert_analysis")
tf_board_path = os.path.join(log_files_path,"tensorboards")
pred_jsons_path = os.path.join(result_path,"pred_jsons")
train_plots_path = os.path.join(result_path,"train_plots")

#datasets
onto_notes_wsj_files = os.path.join(base_dir,"Data/coref_data/ontonotes-release-5.0/data/files/data/english/annotations/nw/wsj")
is_notes_data_path = os.path.join(data_path,"ISNotes1.0-master")
is_notes_jsonlines = "ISNotes.jsonlines"
is_notes_train_jsonlines = "ISNotes_train.jsonlines"
is_notes_dev_jsonlines = "ISNotes_dev.jsonlines"
is_notes_test_jsonlines = "ISNotes_test.jsonlines"
is_notes_json_f_path = os.path.join(is_notes_data_path,is_notes_jsonlines)
is_notes_vec_path = os.path.join(is_notes_data_path,"vecs")
is_notes_raw_path = os.path.join(is_notes_data_path,"raw")
bert_vec_path = os.path.join(is_notes_vec_path, "bert/vecs")
is_notes_svm_ranking_data_path = os.path.join(is_notes_data_path,"svm_rank")

bashi_data_path = os.path.join(data_path,"BASHI")
bashi_conll_file_path = os.path.join(bashi_data_path,"bashi-bridging-column.conll")
bashi_jsonlines = "BASHI.jsonlines"
bashi_vec_path = os.path.join(bashi_data_path,"vecs")
bashi_svm_ranking_data_path = os.path.join(bashi_data_path,"svm_rank")

arrau_data_path = os.path.join(data_path,"ARRAU")

arrau_train_data_path = os.path.join(arrau_data_path,"ARRAU_Train_v4")
arrau_train_conll_path = os.path.join(arrau_train_data_path,"TXT")
arrau_train_mmax_path = os.path.join(arrau_train_data_path,"MMAX")

arrau_test_data_path =os.path.join(arrau_data_path,"ARRAU_Test_Gold_v2")
arrau_test_conll_path = os.path.join(arrau_test_data_path,"TXT")
arrau_test_mmax_path = os.path.join(arrau_test_data_path,"MMAX")

arrau_jsonlines = "ARRAU.jsonlines"
arrau_vec_path = os.path.join(arrau_data_path,"vecs")

bridg_rels = ["NA", "Subset", "Subset-inv", "Element", "Element-inv", "Other", "Other-inv",
              "Undersp-rel", "Poss", "Poss-inv", "Unmarked"]
#extensions
mmax = ".mmax"
words_file_extn = "_words.xml"
sents_file_extn = "_sentence_level.xml"
entity_file_extn = "_entity_level.xml"
coref_file_extn = "_coref_level.xml"



#misc
xmlns_ent="{www.comp.leeds.ac.uk/markert/entity}"
xmlns_sent="{www.eml.org/NameSpaces/sentence}"
xmlns_coref="{www.eml.org/NameSpaces/coref}"

path2vec_base_path = os.path.join(base_dir,"Data/wordnet/path2vec/")
path2vec_paths = [os.path.join(path2vec_base_path,"lch_embeddings.vec.gz"),
                  os.path.join(path2vec_base_path,"jcn-semcor_embeddings.vec.gz"),
                  os.path.join(path2vec_base_path,"shp_embeddings.vec.gz"),
                  os.path.join(path2vec_base_path,"wup_embeddings.vec.gz")]
path2vec_sims = ["lch", "jcn-semcor", "shp", "wup"]

#original hou rule based training files
# train_files = ["wsj_1101", "wsj_1123", "wsj_1094", "wsj_1100", "wsj_1121", "wsj_1367", "wsj_1428", "wsj_1200", "wsj_1423", "wsj1353"]
dev_files = ["wsj_1101",  "wsj_1094",  "wsj_1121",  "wsj_1428",  "wsj_1423"]
test_files = ["wsj_1123","wsj_1100","wsj_1367","wsj_1200", "wsj1353"]


span_start_indi = '$ss'
span_end_indi = '$es'
time_reg = "second|seconds|time|month|months|year|years|hour|hours|week|weeks|day|days|monday|tuesday|wednesday|thursday|friday|saturday|sunday"

max_seq_len = 256
max_num_sense = 5
path2vec_dim = 300
bert_dim = 768
sen_window_size_ = 3
word2vec_dim = 300
glove_vec_dim = 300
max_ante = 20
embeddings_pp_dim = 100
max_words_in_ent = 50
bashi_anaphors = 349
is_notes_anaphors = 663
is_notes_modifiers = "(CD|NN|NNS|JJ|JJR|JJS|VB|VBD|VBG|VBP|VBZ)"
bashi_modifiers = "(NN|NNS|VB|VBD|VBG|VBP|VBZ)"

#vec file extns
w2vec_vec_file_extn = "_ana_cand_w2vec.pkl"
glove_vec_file_extn = "_ana_cand_glove.pkl"
bert_vec_file_extn = "_ana_cand_bert.pkl"

path2vec_all_sense_vec_file_extn = "_ana_cand_path2vec_all_sense.pkl"
path2vec_sense_lesk_vec_file_extn = "_ana_cand_path2vec_lesk_sense.pkl"
path2vec_sense_avg_vec_file_extn = "_ana_cand_path2vec_avg_sense.pkl"
path2vec_sims_file_extns = ["_lch", "_jcn", "_shp", "_wup"]
wn2vec_file_extn = "_ana_cand_wn2vec.pkl"
rw_wn_vec_file_extn = "_ana_cand_rw_wn_vec.pkl"
br_wn_vec_file_extn = "_ana_cand_br_wn_vec.pkl"

br_wn_vec_all_sense_vec_file_extn = "_ana_cand_br_wn_all_sense.pkl"
br_wn_vec_sense_lesk_vec_file_extn = "_ana_cand_br_wn_lesk_sense.pkl"
br_wn_vec_sense_avg_vec_file_extn = "_ana_cand_br_wn_avg_sense.pkl"

embeddings_pp_ana_vec_file_extn = "_ana_embeddings_pp.pkl"
embeddings_pp_cand_ant_vec_file_extn = "_cand_embeddings_pp.pkl"


head_extn = "_synt_head_"
sem_head_extn = "_sem_head_"
int_ent_sem_head_pkl_extn = "_ant_cand_sem_head_words_and_span.pkl"
ana_sem_head_pkl_extn = "_ana_sem_head_words_and_span.pkl"
int_head_map_extn = "_int_ent_head_map.pkl"
all_extn = "_all_words"
hou_like_emb_extn = "_hou_like"


#SVM settings
kernels = ['linear','poly','rbf']

#BASHI constants
# anaphor_types = ["GEN","INDEF","COMP"]
anaphor_types = ["COMP","INDEF","GEN"]
begin_doc = "#begin document"
end_doc = "#end document"

wn_vec_alg_options = ["wn2vec","path2vec","rw_wn_vec","br_wn_vec"]
WN2VEC = wn_vec_alg_options[0]
PATH2VEC = wn_vec_alg_options[1]
RWWN2VEC = wn_vec_alg_options[2]
BRWN2VEC = wn_vec_alg_options[3]

bridging_ana_cand_ante_wordnet_rel_file = os.path.join(data_path,"ana_cand_wn_rels.csv")

datasets = ["ISNotes","BASHI","ARRAU","RST","TRAINS","PEAR"]
ISNotes = datasets[0]
BASHI = datasets[1]
ARRAU = datasets[2]
RST = datasets[3]
PEAR = datasets[5]
TRAINS = datasets[4]

num_rst_docs = 413
num_pear_docs = 20
num_trains_docs = 98