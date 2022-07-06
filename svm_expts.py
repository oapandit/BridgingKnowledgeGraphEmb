from common_code import *
import tensorflow as tf
from tf_base import *
from tf_base import TFSummary
from coref_bridging_models import get_classification_model
from coref_bridging_models import get_ranking_loss_model
import numpy as np
import os
from evaluate_models import EvaluateModels

np.random.seed(1313)
# tf.set_random_seed(1331)
from bridging_json_extract import BridgingJsonDocInformationExtractor
from coref_bridging_models import get_ranking_loss_model,get_scoring_model
from neural_expts import BridgingNeuralExpts
from bridging_utils import *
from evaluation_measures import *
from expt_params import ExptParams
from word_embeddings import EmbeddingsPP
from itertools import repeat
import itertools
from sklearn.svm import SVC
import sklearn
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
from random import randrange
num_workers = mp.cpu_count()
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# is_fit_features = True
print("Num CPUs Available: ", num_workers)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tf_ranking_expt import *

class SVMExpts(ExptParams):
    def __init__(self,is_classification,is_concat_vecs,outer_k,inner_k):
        self.is_classification = is_classification
        self.is_concat_vecs = is_concat_vecs
        self.outer_k = outer_k
        self.inner_k = inner_k
        self.additional_training_data = None
        self.is_fit_features = True
        self.set_model_name()
        super().__init__()
        self.true_total_anaphors = bashi_anaphors if self.dataset_name == BASHI else is_notes_anaphors

    def set_model_name(self):
        model_name = "svm"
        model_name += "_class" if self.is_classification else "_ranking"
        model_name += "_concat" if self.is_concat_vecs else "_multi"
        self.model_name = model_name

    def set_evaluation_params(self,train_object,dev_object,test_object,svm_ranking_data_path,evaluate_svm_model):
        # if evaluate_svm_model is None:
        #     evaluate_svm_model = EvaluateModels()
        evaluate_svm_model.svm_ranking_data_path = svm_ranking_data_path
        evaluate_svm_model.set_svm_params(self,train_object,dev_object,test_object)
        return evaluate_svm_model

    def _set_additional_data(self,data_param):
        if self.use_additional_data:
            param = data_param + self.hyper_params[0]
            self.set_params(*param)
            self.additional_training_data = self.additional_data.generate_svm_data(self.additional_data_jasonlines,self.is_classification, True, None,
                                                                              sen_window_size=self.train_sen_window_size,
                                                                              is_consider_saleint=self.is_consider_saleint,
                                                                              is_concat=self.is_concat_vecs, is_return_vecs=True, is_fourier_features=self.is_fourier_features, is_fit_features=self.is_fit_features, is_normalize=self.is_normalize)
            self.is_fit_features = False

    def train_svm_model(self, train_object, dev_object=None, test_object=None,inner_loop=None,is_evaluate_train=False,is_evaluate_all_windows=False,evaluate_svm_model=None,svm_ranking_data_path=None,is_add_pred_pairs=False):
        """
        train simple classification model
        :param kwrgs:
        :return:
        """

        class_acc,train_acc,dev_acc, test_acc = 0,0,None,None


        expt_name = self.get_expt_name(self.model_name,inner_loop)
        print(10 * "---")
        print("expt : {}".format(expt_name))
        print(10 * "---")

        logger.critical(10 * "---")
        logger.critical("expt : {}".format(expt_name))
        logger.critical(10 * "---")

        model_dat_file = "{}_model.dat".format(randrange(1000000))
        svm_train_model = os.path.join(svm_ranking_data_path,model_dat_file)

        train_dat_file = "train.dat"
        if inner_loop is not None:
            train_dat_file = str(inner_loop)+"_"+train_dat_file
        train_dat = os.path.join(svm_ranking_data_path,train_dat_file)

        self.set_evaluation_params(train_object,dev_object,test_object,svm_ranking_data_path,evaluate_svm_model)
        c = self.epochs
        gamma = self.batch_size
        kernel = self.lr
        degree = self.num_ff_layer
        # print("c : {} g : {} k: {}".format(c,gamma,kernel))

        if (not self.is_classification and not os.path.exists(train_dat)) or self.is_classification:
            # assert not (is_svm and not self.is_classification) or inner_loop is None
            X, bridging_labels_vec_train = self.data.generate_svm_data(train_object, self.is_classification, True, train_dat,
                                        sen_window_size=self.train_sen_window_size,
                                        is_consider_saleint=self.is_consider_saleint,
                                        is_concat=self.is_concat_vecs, additional_data=self.additional_training_data,
                                        is_fourier_features=self.is_fourier_features,
                                        is_fit_features=self.is_fit_features, is_normalize=self.is_normalize)

        if self.is_classification:
            svm_train_model = SVC(C=c,kernel=kernel,gamma=gamma,degree=degree,probability=True,class_weight='balanced')

        start_time = time.time()
        if self.is_classification:
            svm_train_model.fit(X, bridging_labels_vec_train)
        else:
            command = get_svm_rank_command(c=c, kernel=kernel, gamma=gamma, degree=degree, model_input_dat=train_dat, pred_dat=None,
                                 svm_model=svm_train_model, is_train=True)
            logger.debug("executing command {}".format(command))
            os.system(command)

        end_time = time.time()
        print("It took {} to train with {}".format(hms_string(end_time - start_time),self.model_name))



        if self.is_classification:
            class_acc = svm_train_model.score(X, bridging_labels_vec_train)
            print("simple pairwise classification accuracy {}".format(class_acc))

        train_acc,dev_acc,test_acc,total_pred,total_correct = evaluate_svm_model.evaluate_svm_for_current_model(is_evaluate_train, svm_train_model, self.is_classification,is_evaluate_all_windows,is_add_pred_pairs=is_add_pred_pairs)

        return class_acc,train_acc,dev_acc,test_acc,total_pred,total_correct

    def _train_each_inner_fold(self, td_objs):
        _t_object, _d_object, inner_loop, _train_accs, _dev_accs,evaluate_svm_model,svm_ranking_data_path = td_objs
        # assert len(_t_object) == 36
        # assert len(_d_object) == 9
        _, _train_acc, _dev_acc, _, _, _ = self.train_svm_model(_t_object, dev_object=_d_object, test_object=None, inner_loop=inner_loop, is_evaluate_train=False,
                        is_evaluate_all_windows=False,
                        evaluate_svm_model=evaluate_svm_model, svm_ranking_data_path=svm_ranking_data_path)
        _train_accs.append(_train_acc)
        _dev_accs.append(_dev_acc)

    def _train_each_hyper_param(self,train_objects, dev_objects, hc, hyper_param, param_to_dev_score_dict, fold,data_param,evaluate_svm_model,svm_ranking_data_path):
        param = data_param + hyper_param
        train_accs = []
        dev_accs = []

        z_train_accs = self.inner_k * [train_accs]
        z_dev_accs = self.inner_k * [dev_accs]

        logger.critical(10 * "!!!!")
        logger.critical("HYPER PARAMS : {}/{}".format(hc, len(self.hyper_params)))
        logger.critical(10 * "!!!!")
        self.set_params(*param)
        # serial processing
        for inner_loop, (train_object, dev_object) in enumerate(
                zip(train_objects, dev_objects)):
            data_set_num = "{}_{}".format(fold, inner_loop)
            self._train_each_inner_fold([train_object, dev_object, data_set_num, train_accs, dev_accs,evaluate_svm_model,svm_ranking_data_path])
        # concurrancy with threading
        # data_set_nums = ["{}_{}".format(_fold,inner_loop) for inner_loop in range(inner_k)]
        # with ThreadPoolExecutor(inner_k) as ex:
        #     ex.map(_train_each_inner_fold, zip(_train_objects, _dev_objects,data_set_nums,_z_train_accs,_z_dev_accs))
        print("dev accs {}".format(dev_accs))
        print("train accs {}".format(train_accs))
        assert len(dev_accs) == len(train_accs) == self.inner_k
        avg_dev_score = float(sum(dev_accs)) / len(dev_accs)
        avg_train_score = float(sum(train_accs)) / len(train_accs)
        param_to_dev_score_dict[tuple(param)] = [avg_train_score, avg_dev_score]
        logger.critical(
            "train : {} , dev : {} accuracy for current hyperparams".format(avg_train_score, avg_dev_score))
        return (tuple(param), [avg_train_score, avg_dev_score])

    def svm_get_best_hyper_param_for_fold(self,train_object,data_param,fold,evaluate_svm_model,svm_ranking_data_path):
        is_train_hyper_serially = False
        best_parm = None
        if len(self.hyper_params) > 1:
            _train_objects, _dev_objects = self.data.get_k_fold_json_objects_split(train_object, k=self.inner_k)
            if not self.is_classification:
                param = data_param + self.hyper_params[0]
                self.set_params(*param)
                for inner_loop, tobj in enumerate(_train_objects):
                    train_dat_file = "{}_{}_train.dat".format(fold, inner_loop)
                    train_dat = os.path.join(svm_ranking_data_path, train_dat_file)
                    self.data.generate_svm_data(tobj, self.is_classification, True, train_dat,
                                                sen_window_size=self.train_sen_window_size,
                                                is_consider_saleint=self.is_consider_saleint,
                                                is_concat=self.is_concat_vecs, additional_data=self.additional_training_data,
                                                is_fourier_features=self.is_fourier_features,
                                                is_fit_features=self.is_fit_features, is_normalize=self.is_normalize)
                    self.is_fit_features = False
            print("len of hyper params {} and hyper params {}".format(len(self.hyper_params), self.hyper_params))

            param_to_dev_score_dict = create_data_obj(mp,dict,is_train_hyper_serially)
            func_params = []
            for hc, hyper_param in enumerate(self.hyper_params):
                func_param = _train_objects, _dev_objects, hc, hyper_param, param_to_dev_score_dict,fold,data_param,evaluate_svm_model,svm_ranking_data_path
                func_params.append(func_param)
            execute_func(self._train_each_hyper_param,func_params,mp,is_train_hyper_serially)

            best_dev_acc = -1
            print(param_to_dev_score_dict)
            assert len(param_to_dev_score_dict) == len(self.hyper_params),"score dict {} and len hyper param {}".format(param_to_dev_score_dict,len(self.hyper_params))
            for p in param_to_dev_score_dict.keys():
                train_acc, dev_acc = param_to_dev_score_dict[p]
                if dev_acc > best_dev_acc:
                    best_parm = p
                    best_dev_acc = dev_acc
        else:
            best_parm = data_param + self.hyper_params[0]
        return best_parm

    def svm_get_score_for_fold(self, train_object, test_object, fold, train_accs, dev_accs, test_accs, total_preds,
                               total_corrects, data_param,evaluate_svm_model,svm_ranking_data_path):

        # assert len(train_object) == 45
        # assert len(test_object) == 5

        best_parm = self.svm_get_best_hyper_param_for_fold(train_object,data_param,fold,evaluate_svm_model,svm_ranking_data_path)
        logger.critical("best params {} {} {}".format(best_parm[-4], best_parm[-3], best_parm[-2]))
        self.set_params(*best_parm)
        logger.critical(10 * "==")
        logger.critical("---train with best params---")
        class_acc, train_acc, dev_acc, test_acc, total_pred, total_correct = self.train_svm_model(train_object,
                                                                                                  dev_object=None,
                                                                                                  test_object=test_object,
                                                                                                  inner_loop=None,
                                                                                                  is_evaluate_train=True,
                                                                                                  is_evaluate_all_windows=False,
                                                                                                  evaluate_svm_model=evaluate_svm_model,
                                                                                                  svm_ranking_data_path=svm_ranking_data_path,
                                                                                                  is_add_pred_pairs=True)

        _is_fit_features = False
        train_accs.append(train_acc)
        dev_accs.append(dev_acc)
        test_accs.append(test_acc)
        total_preds.append(total_pred)
        total_corrects.append(total_correct)
        logger.critical("dev accuracy {}".format(dev_acc))

    def svm_get_best_score_for_data_param_with_nested_cross_validation(self,data_param,results):
        """
        for a given data setting SVM models will be trained and the best score.
        :param data_param:
        :return:
        """
        # print("experiments with {} data params".format(len(data_param)))

        evaluate_svm_model = EvaluateModels()
        train_objects, test_objects = self.data.get_k_fold_json_objects_split(self.jsonlines, k=self.outer_k)
        self.hyper_params = self.get_svm_hyper_param_comb()

        svm_ranking_data_path = os.path.join(is_notes_svm_ranking_data_path, str(randrange(1000000)))
        create_dir(svm_ranking_data_path)
        self._set_additional_data(data_param)

        train_accs, dev_accs, test_accs, total_preds, total_corrects = [],[],[],[],[]

        start_time = time.time()
        for fold, (train_object, test_object) in enumerate(
                zip(train_objects, test_objects)):
            logger.critical(10 * "^^^^")
            logger.critical("FOLD : {}/{}".format(fold, self.outer_k))
            logger.critical(10 * "^^^^")
            self.svm_get_score_for_fold(train_object, test_object, fold, train_accs, dev_accs, test_accs, total_preds, total_corrects, data_param,evaluate_svm_model,svm_ranking_data_path)
            # sys.exit()
        assert len(test_accs)>0
        assert len(total_preds) == len(total_corrects) == len(test_accs) == len(train_accs) == len(dev_accs) == self.outer_k
        total_pred = sum(total_preds)
        if self.dataset_name == BASHI:
            assert total_pred == bashi_anaphors,"total pred {} ".format(total_pred)
        else:
            assert total_pred == is_notes_anaphors
        total_corrects_np = np.array(total_corrects)
        assert total_corrects_np.shape[0]==10
        assert total_corrects_np.shape[1] == 4 or total_corrects_np.shape[1] == 1
        cum_accs = []
        for i in range(total_corrects_np.shape[1]):
            cum_accs.append(sum(total_corrects_np[:,i])*1.0/total_pred)
        avg_train_score = float(sum(train_accs)) / len(train_accs)
        if len(dev_accs) == 0:
            avg_dev_score = float(sum(dev_accs)) / len(dev_accs)
        else:
            avg_dev_score = 0
        avg_test_score = float(sum(test_accs)) / len(test_accs)
        print("test accs {}".format(test_accs))
        print("average test acc : {}".format(avg_test_score))
        test_windows = [2,3,4,-1]
        _test_res = ""
        for i in range(total_corrects_np.shape[1]):
            t = test_windows[i]
            print("test accuracy micro with {} window is {}".format(t,cum_accs[i]))
            _test_res += " test micro - {} with window {}".format(cum_accs[i],t)
        expt_name = self.get_expt_name(self.model_name)
        logger.critical("train : {} , dev : {} , test averaged: {}, {}".format(avg_train_score,avg_dev_score,avg_test_score,_test_res))
        curr_expt_res = "{} : train - {} , dev - {} , test macro - {}, {}".format(expt_name,avg_train_score,avg_dev_score,avg_test_score,_test_res)
        results.append(curr_expt_res)
        print("------")
        print(curr_expt_res)
        print("------")
        result_file = os.path.join(result_path, timestr + '_final.result.txt')
        write_text_file(result_file,results)
        end_time = time.time()
        logger.critical("It took {} to get result for this data settings".format(hms_string(end_time - start_time)))
        num_pred_pairs = 0
        for single_json_object in evaluate_svm_model.pred_json_objects:
            if single_json_object.get('predicted_pairs', None) is not None:
                num_pred_pairs += len(single_json_object.get('predicted_pairs'))
        assert num_pred_pairs == self.true_total_anaphors, "number of pairs {}".format(num_pred_pairs)
        json_lines_file_path = os.path.join(pred_jsons_path, expt_name + "_" + timestr + '_pred.pairs.jsonl')
        write_jsonlines_file(evaluate_svm_model.pred_json_objects, json_lines_file_path)

    def svm_get_best_score_for_data_param_with_simple_routine(self,data_param,results):
        evaluate_svm_model = EvaluateModels()
        train_objects,dev_objects,test_objects = self.data.get_arrau_json_objects_split(self.jsonlines,self.sub_dataset_name)
        self.hyper_params = self.get_svm_hyper_param_comb()

        svm_ranking_data_path = os.path.join(is_notes_svm_ranking_data_path, str(randrange(1000000)))
        create_dir(svm_ranking_data_path)

        train_accs,dev_accs,test_accs = [],[],[]
        for hc,hyper_param in enumerate(self.hyper_params):
            param = data_param + hyper_param

            logger.critical(10 * "!!!!")
            logger.critical("HYPER PARAMS : {}/{}".format(hc, len(self.hyper_params)))
            logger.critical(10 * "!!!!")

            self.set_params(*param)

            _, _train_acc, _dev_acc, _test_acc, _, _ = self.train_svm_model(train_objects, dev_object=dev_objects, test_object=test_objects,
                                                                    inner_loop=0, is_evaluate_train=True,
                                                                    is_evaluate_all_windows=False,
                                                                    evaluate_svm_model=evaluate_svm_model,
                                                                    svm_ranking_data_path=svm_ranking_data_path)
            train_accs.append(_train_acc)
            dev_accs.append(_dev_acc)
            test_accs.append(_test_acc)

        max_index = dev_accs.index(max(dev_accs))
        print("maximum validation score is obtained with {} and scores are -".format(self.hyper_params[max_index],dev_accs[max_index]))
        print("train : {}, dev : {} and test : {}".format(train_accs[max_index], dev_accs[max_index],test_accs[max_index]))

        expt_name = self.get_expt_name(self.model_name)

        curr_expt_res = "{} : train - {} , dev - {} , test macro - {}".format(expt_name,train_accs, dev_accs, test_accs)
        results.append(curr_expt_res)
        print("------")
        print(curr_expt_res)
        print("------")
        result_file = os.path.join(result_path, timestr + '_final.result.txt')
        write_text_file(result_file,results)

    def svm_get_best_score_for_data_param_with_cross_validation(self,data_param, results):
        evaluate_svm_model = EvaluateModels()
        train_objects,dev_objects,test_object = self.data.get_arrau_json_objects_split(self.jsonlines,self.sub_dataset_name)
        train_object = train_objects+dev_objects
        self.hyper_params = self.get_svm_hyper_param_comb()

        svm_ranking_data_path = os.path.join(is_notes_svm_ranking_data_path, str(randrange(1000000)))
        create_dir(svm_ranking_data_path)

        train_accs,dev_accs,test_accs,total_preds,total_corrects = [],[],[],[],[]
        self.svm_get_score_for_fold(train_object, test_object, 0, train_accs, dev_accs, test_accs, total_preds,
                                    total_corrects, data_param, evaluate_svm_model, svm_ranking_data_path)

        expt_name = self.get_expt_name(self.model_name)

        curr_expt_res = "{} : train - {} , dev - {} , test macro - {}, {}".format(expt_name,train_accs[0], dev_accs[0], test_accs[0],total_corrects[0][0]/total_preds[0])
        results.append(curr_expt_res)
        print("------")
        print(curr_expt_res)
        print("------")
        result_file = os.path.join(result_path, timestr + '_final.result.txt')
        write_text_file(result_file,results)


    def svm_get_best_score_for_data_param(self,data_param,results):
        if self.dataset_name == BASHI or self.dataset_name == ISNotes:
            self.svm_get_best_score_for_data_param_with_nested_cross_validation(data_param,results)
        elif self.dataset_name == ARRAU:
            if self.sub_dataset_name == RST :
                self.svm_get_best_score_for_data_param_with_simple_routine(data_param,results)
            elif self.sub_dataset_name == PEAR or self.sub_dataset_name == TRAINS:
                self.svm_get_best_score_for_data_param_with_cross_validation(data_param, results)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


    def svm_expts(self):
        is_serially = True
        data_params = self.get_data_param_comb(is_svm=True)
        print("experiments with {} data params".format(len(data_params)))
        results = create_data_obj(mp,list,is_serially)
        func_params = []
        for dc,data_param in enumerate(data_params):
            # logger.critical(10 * "####")
            # logger.critical("DATA PARAM : {}/{}".format(dc,len(data_params)))
            # logger.critical(10 * "####")
            func_param = data_param,results
            func_params.append(func_param)
        execute_func(self.svm_get_best_score_for_data_param,func_params,mp,is_serially)


if __name__ == '__main__':
    bm = SVMExpts(is_classification=False,is_concat_vecs=False,outer_k=10,inner_k=5)
    bm.svm_expts()
    # bm.svm_expts(False)
