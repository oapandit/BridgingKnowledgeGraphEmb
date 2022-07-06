from common_code import *
from tf_base import *
from data_read import BridgingVecDataReader
from bridging_utils import *
from evaluation_measures import *
import itertools
from operator import xor

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class ExptParams:
    def __init__(self):
        self.is_all_words = False
        self.is_read_head_vecs = False
        self.is_sem_head = False
        self.path2vec_sim = 0
        # self.is_bashi = True
        self.dataset_name = ISNotes
        self.sub_dataset_name = PEAR # applicable only for ARRAU dataset
        self.is_hou_like = False
        self.use_additional_data = False
        self.additional_data = None
        self.additional_data_jasonlines = None

        self.wordnet_vec_alg = PATH2VEC

        self.bashi_data = BridgingVecDataReader(json_objects_path=bashi_data_path, vec_path=bashi_vec_path,
                                 is_read_head_vecs=self.is_read_head_vecs, is_sem_head=self.is_sem_head,
                                 is_all_words=self.is_all_words, path2vec_sim=self.path2vec_sim,
                                 is_hou_like=self.is_hou_like,wn_vec_alg=self.wordnet_vec_alg)

        self.is_notes_data = BridgingVecDataReader(json_objects_path=is_notes_data_path, vec_path=is_notes_vec_path,
                                 is_read_head_vecs=self.is_read_head_vecs, is_sem_head=self.is_sem_head,
                                 is_all_words=self.is_all_words, path2vec_sim=self.path2vec_sim,
                                 is_hou_like=self.is_hou_like,wn_vec_alg=self.wordnet_vec_alg)

        self.arrau_data = BridgingVecDataReader(json_objects_path=arrau_data_path, vec_path=arrau_vec_path,
                                 is_read_head_vecs=self.is_read_head_vecs, is_sem_head=self.is_sem_head,
                                 is_all_words=self.is_all_words, path2vec_sim=self.path2vec_sim,
                                 is_hou_like=self.is_hou_like,wn_vec_alg=self.wordnet_vec_alg)


        if self.dataset_name == BASHI:
            self.jsonlines = bashi_jsonlines
            self.data = self.bashi_data

            self.additional_data = self.is_notes_data
            self.additional_data_jasonlines = is_notes_jsonlines
        elif self.dataset_name == ISNotes:
            self.jsonlines = is_notes_jsonlines
            self.data = self.is_notes_data

            self.additional_data = self.bashi_data
            self.additional_data_jasonlines = bashi_jsonlines

        elif self.dataset_name == ARRAU:
            self.jsonlines = arrau_jsonlines
            self.data = self.arrau_data

            self.additional_data = self.bashi_data
            self.additional_data_jasonlines = bashi_jsonlines

        else:
            raise NotImplementedError
        self.save_model_dir = tf_trained_models_path
        self.ae_save_model_dir= auto_encoder_trained_models_path
        self.is_fourier_features = False
        self.is_normalize = False
        self.is_encoder = False
        self.is_classification = False
        self.is_ranking = not self.is_classification
        self.is_scoring = False

        self.is_latent_svm_expt = False


    def set_data_params(self):
        self.is_consider_arti_good_pairs = False
        self.is_consider_arti_bad_pairs = False
        self.is_consider_arti_ugly_pairs = False

        self.is_soon = False
        self.is_positive_surr = False
        self.is_balanced = False
        self.surr_window = 2

        self.train_sen_window_size = 5
        self.test_sen_window_size = 5

        self.is_soft_attn=True
        self.is_consider_saleint = True

    def set_embs(self,is_emb_pp, is_wsd, is_avg, is_w2vec, is_glove, is_bert, is_path2vec,w2vec_dim, glove_dim):
        self.is_wsd = is_wsd
        self.is_avg = is_avg
        self.is_w2vec = is_w2vec
        self.is_glove = is_glove
        self.is_bert = is_bert
        self.is_path2vec = is_path2vec
        self.is_emb_pp = is_emb_pp
        self.w2vec_dim = w2vec_dim
        self.glove_dim = glove_dim

    def set_nn_expt(self):
        self.is_classification = False
        self.is_ranking = True
        self.is_scoring = False

        # assert self.is_classification or self.is_scoring
        # assert not (self.is_classification and self.is_scoring)
        assert xor(xor(self.is_classification,self.is_ranking),self.is_scoring)


    def set_nn_params(self, max_num_sense, num_ff_layer, epochs, batch_size, learning_rate,dropout,opt):
        self.max_num_sense = max_num_sense
        self.num_ff_layer = num_ff_layer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.dropout = dropout
        self.opt = opt

        self.is_subtract = False
        self.is_multiply = False
        self.is_add = False

        self.is_distance_measure = False
        self.distance_measure = "cosine"
        self.is_share_weights = True

        self.is_gru_head_finder = False

        if self.is_ranking or self.is_scoring:
            self.is_coref_ranking_loss = True
            self.is_min_max_ranking_loss = False
            self.is_tf_ranking_loss = False

            assert xor(xor(self.is_coref_ranking_loss, self.is_min_max_ranking_loss), self.is_tf_ranking_loss)

            if self.is_tf_ranking_loss:
                self.is_paiwise_hinge_loss = True
                self.is_pairwise_logistic_loss = False
                self.is_pairwise_soft_zero_one_loss = False

                assert xor(xor(self.is_paiwise_hinge_loss, self.is_pairwise_logistic_loss), self.is_pairwise_soft_zero_one_loss)

    def _set_latent_svm_training_params(self, max_num_sense, degree, c, gamma,kernel,lr,opt):
        self.max_num_sense = max_num_sense
        self.max_ante = max_ante
        self.degree = degree
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.lr = lr
        self.opt = opt
        self.alpha = 1

        self.is_subtract = False
        self.is_multiply = True
        self.is_add = False
        self.is_concat = False

        self.is_consider_ext_emb_sim = False

    def set_params(self,is_emb_pp, is_wsd, is_avg, is_w2vec, is_glove, is_bert, is_path2vec,
                   max_num_sense,w2vec_dim,glove_dim, num_ff_layer, epochs, batch_size, learning_rate,dropout,opt):
        self.set_data_params()
        self.set_embs(is_emb_pp,is_wsd, is_avg, is_w2vec, is_glove, is_bert, is_path2vec, w2vec_dim, glove_dim)
        self.set_nn_params(max_num_sense, num_ff_layer, epochs, batch_size, learning_rate,dropout,opt)
        self.data.set_params(self)
        self.additional_data.set_params(self)

    def set_latent_svm_params(self,is_emb_pp, is_wsd, is_avg, is_w2vec, is_glove, is_bert, is_path2vec,
                   max_num_sense,w2vec_dim,glove_dim,degree, c, gamma,kernel,lr,opt):
        self.set_data_params()
        self.set_embs(is_emb_pp,is_wsd, is_avg, is_w2vec, is_glove, is_bert, is_path2vec, w2vec_dim, glove_dim)
        self._set_latent_svm_training_params(max_num_sense, degree, c, gamma,kernel,lr,opt)
        self.data.set_params(self)
        self.additional_data.set_params(self)

    def set_men_rep(self):
        self.men_rep_dim = 0
        if self.is_bert:
            self.men_rep_dim += bert_dim
        if self.is_emb_pp:
            self.men_rep_dim += embeddings_pp_dim
        if self.is_w2vec:
            self.men_rep_dim += self.w2vec_dim
        if self.is_glove:
            self.men_rep_dim += self.glove_dim
        if self.is_path2vec and not self.is_wsd:
            self.men_rep_dim += path2vec_dim


    def set_men_rep_and_neurons(self):
        self.concat_ffnn_neurons = None
        self.non_concat_ffnn_neurons = None
        self.set_men_rep()

        if self.men_rep_dim == 50: # glove / word2vec
            # so after concatenating it will become 100
            self.concat_ffnn_neurons = [[],[50],[50,25]]
            # without concate it remains 50
            self.non_concat_ffnn_neurons = [[], [25]]

        if self.men_rep_dim == 150 : # glove / word2vec + emb_pp
            # so after concatenating it will become 300
            self.concat_ffnn_neurons = [[],[150],[150,75],[150,75,40]]
            # without concate it remains 150
            self.non_concat_ffnn_neurons = [[], [75],[75,40]]

        if self.men_rep_dim == 200 : # glove / word2vec + emb_pp
            # so after concatenating it will become 400
            self.concat_ffnn_neurons = [[], [200], [200,100],[200,100,50]]
            # without concate it remains 200
            self.non_concat_ffnn_neurons = [[],[100],[100,50],[100,50,25]]

        if self.men_rep_dim == 350: # glove / word2vec + avg / lesk
            # so after concatenating it will become 700
            self.concat_ffnn_neurons = [[], [350],[350,175], [350,175, 80],[350,175, 80,40]]
            # without concate it remains 350
            self.non_concat_ffnn_neurons = [[], [175], [175, 80],[175, 80,40]]

        if self.men_rep_dim == 100 or (self.men_rep_dim == 50 and self.is_wsd): # glove / word2vec /emb_pp
            # so after concatenating it will become 200
            self.concat_ffnn_neurons = [[],[100],[100,50],[100,50,25]]
            # without concate it remains 100
            self.non_concat_ffnn_neurons = [[], [50],[50,25]]

        if self.men_rep_dim == 400: # glove / word2vec /emb_pp + avg / lesk
            # so after concatenating it will become 800
            self.concat_ffnn_neurons = [[], [400],[400,200], [400,200,100],[400,200,100,50]]
            # without concate it remains 400
            self.non_concat_ffnn_neurons = [[], [200], [200,100],[200,100,50]]

        if self.men_rep_dim == 450: # glove / word2vec + emb_pp + avg / lesk
            # so after concatenating it will become 900
            self.concat_ffnn_neurons = [[], [450],[450,200], [450,200,100],[450,200,100,50]]
            # without concate it remains 450
            self.non_concat_ffnn_neurons = [[], [200], [200,100],[200,100,50]]

        if self.men_rep_dim == 500: # glove / word2vec + emb_pp + avg / lesk
            # so after concatenating it will become 1000
            self.concat_ffnn_neurons = [[],[500] ,[500,250], [500, 250,125], [500, 250, 125,50]]
            # without concate it remains 500
            self.non_concat_ffnn_neurons = [[],[250], [250,125], [250, 125,60]]

        if self.men_rep_dim == 100 and self.is_wsd:
            # so after concatenating it will become 400
            self.concat_ffnn_neurons = [[], [200], [200,100],[200,100,50]]
            # without concate it remains 200
            self.non_concat_ffnn_neurons = [[],[100],[100,50],[100,50,25]]

        if self.men_rep_dim == 300: # glove / word2vec
            # so after concatenating it will become 600
            self.concat_ffnn_neurons = [[],[300],[300,150],[300,150,75],[300,150,75,40]]
            # without concate it remains 300
            self.non_concat_ffnn_neurons = [[],[150],[150,75],[150,75,40]]

        if self.men_rep_dim == 600 or (self.men_rep_dim == 300 and self.is_wsd): # glove / word2vec + avg / lesk
            # so after concatenating it will become 1200
            self.concat_ffnn_neurons = [[],[600],[600,300],[600,300,150],[600,300,150,75]]
            # without concate it remains 600
            self.non_concat_ffnn_neurons = [[],[300],[300,150],[300,150,75],[300,150,75,40]]

        if self.men_rep_dim == 768 or self.men_rep_dim == 700 or self.men_rep_dim == 868:  # bert
            # so after concatenating it will become 1536
            self.concat_ffnn_neurons = [[], [750],[750,300], [750,300, 150], [750,300, 150, 75]]
            # without concate it remains 768
            self.non_concat_ffnn_neurons = [[], [375],[375,180], [375,180,90],[375,180,90,40]]

        if self.men_rep_dim == 1068 or (self.men_rep_dim == 768 and self.is_wsd) or self.men_rep_dim == 1168:  # bert + avg / lesk
            # so after concatenating it will become 2136
            self.concat_ffnn_neurons = [[], [1000],[1000,500], [1000,500, 250], [1000,500, 250, 125]]
            # without concate it remains 1068
            self.non_concat_ffnn_neurons = [[],[600] ,[500,250], [500, 250,125], [500, 250, 125,50]]

        assert self.concat_ffnn_neurons is not None and self.non_concat_ffnn_neurons is not None
        for i,(c,n) in enumerate(zip(self.concat_ffnn_neurons,self.non_concat_ffnn_neurons)):
            assert len(c) == len(n) == i

        if self.is_add or self.is_subtract or self.is_multiply or self.is_distance_measure or self.is_encoder:
            self.num_nerons = self.non_concat_ffnn_neurons[self.num_ff_layer]
        else:
            self.num_nerons = self.concat_ffnn_neurons[self.num_ff_layer]

    def get_expt_name(self, expt_with=None, fold=None):
        if expt_with is None:
            if self.is_latent_svm_expt:
                expt_with = "latent_svm_"
            else:
                expt_with = "nn_"
            if self.is_classification:
                expt_with = expt_with + "class_model"
            elif self.is_ranking:
                expt_with = expt_with + "ranking_model"
            elif self.is_scoring:
                expt_with = expt_with + "scoring_model"
            elif self.is_encoder:
                expt_with = "auto_encoder_model"
            else:
                raise NotImplementedError

        if fold is not None:
            expt_with = "{}_{}".format(fold, expt_with)
        # if self.is_bashi:
        #     expt_with = expt_with + "_bashi_data"
        # else:
        #     expt_with = expt_with + "_isnotes_data"
        _data_name= "_{}_{}_data".format(self.dataset_name,self.sub_dataset_name) if self.dataset_name == ARRAU else "_{}_data".format(self.dataset_name)
        expt_with = expt_with + _data_name
        if self.use_additional_data:
            expt_with = expt_with + "_additional_training_data"
        if self.is_fourier_features:
            expt_with = expt_with + "_rff"
        if self.is_normalize:
            expt_with = expt_with + "_norm"
        if self.is_all_words:
            expt_with = expt_with + "_all_words"
        if self.is_read_head_vecs:
            if self.is_sem_head:
                expt_with = expt_with + "_sem_heads"
            else:
                expt_with = expt_with + "_synt_heads"
        if self.is_bert:
            expt_with = expt_with + "_bert"
        if self.is_emb_pp:
            expt_with = expt_with + "_emb_pp"
        if self.is_glove:
            expt_with = expt_with + "_glove_{}d".format(self.glove_dim)
        if self.is_w2vec:
            expt_with = expt_with + "_w2vec_{}d".format(self.w2vec_dim)
        if self.is_hou_like and (self.is_emb_pp or self.is_glove or self.is_w2vec):
            expt_with = expt_with + "_hou_like"
        if self.is_path2vec:
            if self.wordnet_vec_alg == PATH2VEC or self.wordnet_vec_alg == BRWN2VEC:
                expt_with = expt_with + path2vec_sims_file_extns[self.path2vec_sim]
                _actual_vec = "path2vec" if self.wordnet_vec_alg == PATH2VEC else "br_cust_vec"
                if self.is_wsd and not self.is_latent_svm_expt:
                    expt_with = expt_with + "_wsd_"+_actual_vec
                    if self.is_soft_attn:
                        expt_with = expt_with + "_soft_attn"
                    else:
                        expt_with = expt_with + "_hard_attn"
                elif self.is_avg:
                    expt_with = expt_with + "_avg_"+_actual_vec
                else:
                    expt_with = expt_with + "_lesk_"+_actual_vec
            elif self.wordnet_vec_alg == WN2VEC:
                expt_with = expt_with + "_wn2vec"
            elif self.wordnet_vec_alg == RWWN2VEC:
                expt_with = expt_with + "_rw_wn_vec"
            else:
                raise NotImplementedError
        if not self.is_encoder:
            expt_with = expt_with + "_train_{}_test_{}_sen_wind_size".format(self.train_sen_window_size,self.test_sen_window_size)
        if "nn" in expt_with:
            if self.is_distance_measure:
                wt = "wt_share" if self.is_share_weights else "wt_sep"
                expt_with = expt_with + "_ana_ante_{}_dist_{}".format(self.distance_measure,wt)
            else:
                if self.is_subtract:
                    expt_with = expt_with + "_ana_ante_subtr"
                elif self.is_multiply:
                    expt_with = expt_with + "_ana_ante_multi"
                elif self.is_add:
                    expt_with = expt_with + "_ana_ante_add"
                else:
                    expt_with = expt_with + "_ana_ante_concat"

            expt_with = expt_with + "_num_ff_layers_{}".format(self.num_ff_layer)
            expt_with = expt_with + "_lr_{}".format(self.lr)
            expt_with = expt_with + "_batch_{}".format(self.batch_size)
            expt_with = expt_with + "_drop_{}".format(self.dropout)
            expt_with = expt_with + "_opt_{}".format(self.opt)
        elif "latent" in expt_with:
            expt_with = expt_with + "_C_{}".format(self.c)
            expt_with = expt_with + "_gamma_{}".format(self.gamma)
            expt_with = expt_with + "_kernel_{}".format(self.kernel)
            expt_with = expt_with + "_degree_{}".format(self.degree)
            expt_with = expt_with + "_lr_{}".format(self.lr)
            expt_with = expt_with + "_opt_{}".format(self.opt)
        elif "svm" in expt_with:
            expt_with = expt_with + "_C_{}".format(self.epochs)
            expt_with = expt_with + "_gamma_{}".format(self.batch_size)
            expt_with = expt_with + "_kernel_{}".format(self.lr)
        else:
            pass

        if self.is_all_words:
            expt_with = expt_with + "_all_words"
        if self.data.is_consider_arti_good_pairs and self.data.is_consider_arti_bad_pairs and self.data.is_consider_arti_ugly_pairs:
            expt_with = expt_with + "_all_art_tlinks"
        if self.is_positive_surr:
            expt_with = expt_with + "_{}_surrounding_positives".format(self.surr_window)
        if self.is_soon:
            expt_with = expt_with + "_soon"
        if self.is_balanced and "class" in expt_with:
            expt_with = expt_with + "_balanced_data"
        return expt_with

    def save_eval(self, expt_name, scores, is_acc=False):
        if is_acc:
            result_file_path = os.path.join(result_path, expt_name + ".acc.txt")
            result_text = ["{0} : [{1:2.2f}]".format(e + 1, ac) for e, ac in enumerate(scores)]
        else:
            result_file_path = os.path.join(result_path, expt_name + ".result.txt")
            result_text = ["{0} : [{1:2.2f} & {2:2.2f} & {3:2.2f}]".format(e + 1, p * 100, r * 100, f1 * 100) for
                           e, (p, r, f1) in
                           enumerate(scores)]
        write_text_file(result_file_path, result_text)

    def get_data_param_comb(self,is_svm):
        is_wsds = [False, True]
        is_w2vecs = [True, False]
        is_gloves = [True, False]
        is_berts = [True, False]
        is_path2vecs = [False, True]
        is_avgs = [False, True]
        is_emb_pps = [False, True]

        is_wsds = [True]
        # is_w2vecs = [False]
        # is_gloves = [False]
        # is_berts = [True]
        is_path2vecs = [True]
        is_avgs = [False]
        # is_emb_pps = [False]

        # w2vec_dims = [50, 100, 300]
        # glove_dims = [50, 100, 300]

        w2vec_dims = [300]
        glove_dims = [300]

        params = []
        for ( is_w2vec,w2vec_dim, is_glove,glove_dim, is_bert, is_path2vec,  is_emb_pp,is_wsd_, is_avg_) in itertools.product(
             is_w2vecs,w2vec_dims, is_gloves,glove_dims, is_berts, is_path2vecs,  is_emb_pps,is_wsds, is_avgs):
            if (not (
                    is_bert or is_w2vec or is_glove or is_path2vec or is_emb_pp)): continue  # skip False, False condition and train,train or test,test
            if (is_glove and is_w2vec) or (is_w2vec and is_bert) or (is_glove and is_bert): continue
            if (not is_path2vec and is_avg_) or (is_avg_ and is_wsd_): continue
            #TODO : following line is present only for two new wordnet embeddings; should be commented afterwards
            # if is_emb_pp and (is_bert or is_w2vec or is_glove):continue
            if is_wsd_:
                if not is_path2vec: continue
                if not (is_w2vec or is_glove or is_bert or is_emb_pp): continue
                if self.wordnet_vec_alg != PATH2VEC: continue
            # if is_path2vec:
                # if self.wordnet_vec_alg != PATH2VEC and not is_avg_:
                #     continue
            # if is_emb_pp and (is_w2vec or is_glove):continue
            if is_w2vec and (glove_dim == 100 or glove_dim == 50): continue
            if is_glove and (w2vec_dim == 100 or w2vec_dim == 50): continue
            if is_bert and (glove_dim == 100 or glove_dim == 50 or w2vec_dim == 100 or w2vec_dim == 50): continue
            if (is_path2vec and is_avg_) and (not (is_bert or is_w2vec or is_glove)) and (
                    glove_dim == 50 or glove_dim == 100 or w2vec_dim == 100 or w2vec_dim == 50): continue
            if (is_path2vec and not is_avg_ and not is_wsd_) and (not (is_bert or is_w2vec or is_glove)) and (
                    glove_dim == 50 or glove_dim == 100 or w2vec_dim == 100 or w2vec_dim == 50): continue
            if (is_emb_pp and not is_path2vec) and (not (is_bert or is_w2vec or is_glove)) and (
                    glove_dim == 100 or glove_dim == 50 or w2vec_dim == 100 or w2vec_dim == 50): continue
            if (is_emb_pp and is_wsd_) and (not (is_bert or is_w2vec or is_glove)) and (
                    glove_dim == 50 or glove_dim == 100 or w2vec_dim == 100 or w2vec_dim == 50): continue
            if is_svm and is_wsd_ : continue
            _param = [is_emb_pp,is_wsd_, is_avg_, is_w2vec, is_glove, is_bert, is_path2vec,max_num_sense,w2vec_dim, glove_dim]
            params.append(_param)
        # params = params[0:2]
        print("TOTAL NUMBER OF EXPTS {}".format(len(params)))
        # params.reverse()
        # self.set_data_params()
        # for p in params:
        #     is_emb_pp, is_wsd, is_avg, is_w2vec, is_glove, is_bert, is_path2vec, _max_num_sense, w2vec_dim, glove_dim = p
        #     self.set_embs(is_emb_pp, is_wsd, is_avg, is_w2vec, is_glove, is_bert, is_path2vec, w2vec_dim, glove_dim)
        #     # print(p)
        #     print(self.get_expt_name("dummy_"))
        # sys.exit()
        return params

    def get_nn_hyper_param_comb(self):
        num_ff_layers = [1]
        # dropouts = [0.4,0.3,0.2,0]
        # optimizers = ['adam','sgd','rmsprop','adagrad','adadelta']
        optimizers = ['adam']
        dropouts = [0]
        self.ffnn_activation = None
        self.score_activation = None
        if self.is_classification:
            batch_sizes = [3000]
            epochs = [300]
            learning_rates = [0.0005]
        if self.is_scoring or self.is_ranking:
            epochs = [500]
            batch_sizes = [600]
            learning_rates = [0.1]
        params = []
        for (lr, num_ff_layer,batch_size, epoch,dropout,opt) in itertools.product(
            learning_rates, num_ff_layers, batch_sizes,epochs,dropouts,optimizers):
            _param = [num_ff_layer, epoch, batch_size, lr,dropout,opt]
            params.append(_param)
        print("TOTAL NUMBER OF HYPER PARAM {}".format(len(params)))
        for p in params:
            print(p)
        return params

    def get_auto_encoder_nn_params(self):
        num_ff_layers = [3]
        optimizers = ['adam']
        dropouts = [0]
        epochs = [30]
        batch_sizes = [10000]
        learning_rates = [0.1]
        params = []
        for (lr, num_ff_layer,batch_size, epoch,dropout,opt) in itertools.product(
            learning_rates, num_ff_layers, batch_sizes,epochs,dropouts,optimizers):
            _param = [num_ff_layer, epoch, batch_size, lr,dropout,opt]
            params.append(_param)
        return params

    def get_latent_svm_hyper_param_comb(self):

        self.epochs = 200
        self.batch_size = 1000

        degrees = [""]
        cs = [1]  # C
        gammas = [""]  # gamma
        kernels = [''] # kernel
        optimizers = ['adam']
        learning_rates = [0.001]

        params = []
        for (degree,c, gamma,kernel,lr,opt) in itertools.product(
            degrees,cs, gammas,kernels,learning_rates,optimizers):
            _param = [degree, c, gamma,kernel,lr,opt]
            params.append(_param)
        print("TOTAL NUMBER OF HYPER PARAM {}".format(len(params)))
        for p in params:
            print(p)
        return params


    def get_svm_hyper_param_comb(self):
        self.is_fourier_features = False
        self.is_normalize = False

        # epochs = [1, 10]  # C
        # batch_sizes = [0.001, 0.0001]  # gamma
        # learning_rates = ['rbf','linear','poly'] # kernel
        # learning_rates = ['linear']  # kernel


        # cs = [1, 10]  # C
        # gammas = [0.001, 0.0001]  # gamma
        # kernels = ['rbf','linear'] # kernel

        # degrees = [2,3,4]
        # cs = [0.001,0.01,0.1,1, 10,100]  # C
        # gammas = [0.0001,0.001,0.01,0.1,1, 10,100 ]  # gamma
        # kernels = ['rbf','linear','poly'] # kernel

        degrees = [2]
        cs = ["",0.001,0.01,0.1,1, 10,100,500,1000]  # C
        # cs = ["",0.01, 0.1, 10,1000]  # C
        gammas = [""]  # gamma
        kernels = ["",'linear'] # kernel

        # degrees = [2,3,4]
        # cs = [0.001,0.01,0.1,1, 10,100]  # C
        # gammas = [0.0001,0.001,0.01,0.1,1, 10,100 ]  # gamma
        # kernels = ['rbf','poly'] # kernel


        # degrees = [2]
        # cs = [""]  # C
        # gammas = [""]  # gamma
        # kernels = [''] # kernel

        params = []
        for (degree,c, gamma,kernel) in itertools.product(
            degrees,cs, gammas,kernels):
            if kernel != 'poly' and degree>2: continue
            if kernel == "linear" and gamma != gammas[0]:continue
            if kernel == "linear" and c == "" : continue
            if kernel == "" and (c != "" or gamma != ""):continue
            # if kernel == "" and gamma != "": continue
            _param = [degree, c, gamma,kernel,"",""]
            params.append(_param)
        print("TOTAL NUMBER OF HYPER PARAM {}".format(len(params)))
        for p in params:
            print(p)
        return params


    def get_log_reg_hyper_param_comb(self):
        params = [["","","","",""]]
        return params

if __name__ == '__main__':
    ep = ExptParams()
    ep.set_nn_expt()
    ps = ep.get_data_param_comb(False)
    hyper_params = ep.get_nn_hyper_param_comb()
    for p in ps:
        best_parm = p + hyper_params[0]
        ep.set_params(*best_parm)
        print(ep.get_expt_name())
    # ep.get_nn_hyper_param_comb(False)
    # ep.get_svm_hyper_param_comb()