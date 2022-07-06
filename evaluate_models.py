from bridging_utils import *
from bridging_json_extract import BridgingJsonDocInformationExtractor
import numpy as np
from evaluation_measures import *
from numpy.linalg import norm
from random import randrange
from tf_base import *

class EvaluateModels:
    def __init__(self):
        self.svm_ranking_data_path = None
        self.pred_json_objects = []

    def set_svm_params(self,expt_params,train_object,dev_object,test_object):
        self.data = expt_params.data
        self.is_fourier_features = expt_params.is_fourier_features
        self.is_normalize = expt_params.is_normalize
        self.is_consider_saleint = expt_params.is_consider_saleint
        self.is_concat_vecs = expt_params.is_concat_vecs

        self.train_object = train_object
        self.dev_object = dev_object
        self.test_object = test_object

    def set_nn_params(self,expt_params):
        self.data = expt_params.data
        self.is_wsd = expt_params.is_wsd

        self.test_sen_window_size = expt_params.test_sen_window_size
        self.save_model_dir = expt_params.save_model_dir

        self.test_eval_vecs = expt_params.test_eval_vecs
        self.dev_eval_vecs = expt_params.dev_eval_vecs
        self.train_eval_vecs = expt_params.train_eval_vecs

        self.is_classification = expt_params.is_classification
        self.is_ranking = expt_params.is_ranking
        self.is_scoring = expt_params.is_scoring


    def set_latent_svm_params(self,expt_params,sess,iterator,predictions,loss,expt_name,saver,train_jsonlines,dev_jsonlines,test_jsonlines,kwrgs):
        self.training_stats = expt_params.ts
        self.data = expt_params.data
        self.is_wsd = expt_params.is_wsd

        self.test_sen_window_size = expt_params.test_sen_window_size
        self.save_model_dir = expt_params.save_model_dir

        self.test_eval_vecs = expt_params.test_eval_vecs
        self.dev_eval_vecs = expt_params.dev_eval_vecs
        self.train_eval_vecs = expt_params.train_eval_vecs

        self.is_classification = expt_params.is_classification
        self.is_ranking = expt_params.is_ranking

        self.sess = sess
        self.expt_name = expt_name
        self.saver = saver
        self.loss = loss

        self.train_jsonlines,self.dev_jsonlines,self.test_jsonlines = train_jsonlines, dev_jsonlines, test_jsonlines
        self.iterator = iterator

        self.anaphors_context = kwrgs[0]
        self.anaphors_ext = kwrgs[1]
        self.antecedents_context = kwrgs[2]
        self.antecedents_ext = kwrgs[3]
        self.pair_labels = kwrgs[4]

        self.predictions = predictions



    def evaluate_svm_for_current_model(self,is_evaluate_train,_model,is_classification,is_evaluate_all_windows,is_add_pred_pairs=False):
        train_acc,dev_acc, test_acc = 0,None,None
        total_pred,total_correct = None,None
        if is_evaluate_train:
            train_acc,_,_ = self.evaluate_svm(_model, self.train_object, is_training=True,is_classification=is_classification)
            print("train accuracy {}".format(train_acc))
            logger.critical("train accuracy {}".format(train_acc))

        if self.dev_object is not None:
            start_time = time.time()
            dev_acc,_,_ = self.evaluate_svm(_model,self.dev_object,is_classification=is_classification)
            print("dev accuracy {}".format(dev_acc))
            logger.critical("dev accuracy {}".format(dev_acc))
            end_time = time.time()
            print("It took {} to evaluate with dev data ".format(hms_string(end_time - start_time)))

        if self.test_object is not None:
            test_acc,total_pred,total_correct = self.evaluate_svm(_model,self.test_object,is_classification=is_classification,is_all_window=is_evaluate_all_windows,is_add_pred_pairs=is_add_pred_pairs)
            print("test accuracy {}".format(test_acc))
            logger.critical("test accuracy {}".format(test_acc))

        return train_acc,dev_acc,test_acc,total_pred,total_correct

    def evaluate_svm(self, svm_model, eval_jsonlines, is_training=False,is_classification=True,is_all_window=False,is_add_pred_pairs=False):
        if not is_classification:
            return self.evaluate_svm_ranking(svm_model, eval_jsonlines, is_training,is_all_window=is_all_window,is_add_pred_pairs=is_add_pred_pairs)

        json_objects = self.data.get_json_objects(eval_jsonlines)

        acc_preds = []
        # test_window_sizes = [5, None, None, None]
        if is_all_window:
            #test with all window sizes
            test_window_sizes = [2,3,4,-1]
        else:
            test_window_sizes = [2,None,None,None]

        for test_window_size in test_window_sizes:
            if test_window_size is not None:
                cum_acc_pred, cum_total_pred, cum_total_gold = 0, 0, 0
                for single_json_object in json_objects:
                    mention_ops = BridgingJsonDocInformationExtractor(single_json_object, logger)
                    if mention_ops.is_file_empty:
                        continue
                    cand_ante_per_anaphor, context_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec = self.data.get_cand_ante_per_anaphor_and_vecs(
                        mention_ops=mention_ops,sen_window_size=test_window_size,
                        is_consider_saleint=self.is_consider_saleint,
                        is_training=is_training)
                    predicted_pairs = []
                    assert len(cand_ante_per_anaphor) == len(mention_ops.anaphors)
                    for i, cand_ante in enumerate(cand_ante_per_anaphor):
                        curr_ana = mention_ops.anaphors[i]
                        logger.debug(3 * "----")
                        logger.debug("anaphor {} candidate antecedents {} ".format(curr_ana, cand_ante))
                        assert len(cand_ante) != 0
                        curr_labels = len(cand_ante) * [0]
                        is_cont_vec = context_vecs is not None
                        m1_vec, m2_vec,_, _, _, _, _ = self.data.generate_vecs_for_pairs(is_cont_vec,context_vecs, path2vec_vecs,emb_pp_ana_vec,emb_pp_cand_ant_vec,
                                                                              [cand_ante],
                                                                              [curr_labels],
                                                                              mention_ops, [curr_ana])
                        if svm_model is None:
                            logger.debug("m1 shape {} m2 shape {}".format(m1_vec.shape, m2_vec.shape))
                            cos_sim = np.sum(np.multiply(m1_vec, m2_vec), axis=1) / (
                                    norm(m1_vec) * norm(m2_vec))  # + 0.0001)
                            # cos_sim = sklearn.metrics.pairwise.cosine_similarity(m1_vec, m2_vec)
                            logger.debug("cosine simi shape {}".format(cos_sim.shape))
                            pred = cos_sim

                        else:
                            # if not is_classification:
                            #     arr = m1_vec.reshape(m1_vec.shape[0], 1, m1_vec.shape[1])
                            #     m1_vec = np.repeat(arr, max_ante, axis=1)
                            if self.is_concat_vecs:
                                x = np.concatenate([m1_vec, m2_vec], axis=1)
                            else:
                                x = np.multiply(m1_vec, m2_vec)
                            logger.debug("x shape {}".format(x.shape))
                            pred = svm_model.predict_proba(x)
                            logger.debug("pred shape {}".format(pred.shape))
                            logger.debug("pred probs {}".format(pred))
                            pred = pred[:, 1]

                        logger.debug("pred shape {}".format(pred.shape))
                        logger.debug("predicted probs {}".format(np.array(pred).flatten()))
                        max_score_ind = np.argmax(np.array(pred).flatten())
                        logger.debug("max score at {}".format(max_score_ind))
                        pred_ante = cand_ante[max_score_ind]

                        pair = (pred_ante, curr_ana)
                        predicted_pairs.append(pair)
                        logger.debug("pair added {}".format(pair))

                    assert len(predicted_pairs) == len(mention_ops.bridging_clusters)

                    accurate_pred, total_pred, total_gold, pred_pair_labels = get_accurate_pred_gold_numbers(mention_ops,
                                                                                                             predicted_pairs)
                    error_analysis(mention_ops, predicted_pairs, pred_pair_labels)
                    cum_acc_pred += accurate_pred
                    cum_total_gold += total_gold
                    cum_total_pred += total_pred
                acc = get_accuracy(cum_acc_pred, cum_total_pred, cum_total_gold)
                print("total anaphors {}, correctly linked {}, accuracy: {}, with window size {}".format(cum_total_pred,cum_acc_pred,acc,test_window_size))
            acc_preds.append(cum_acc_pred)
        acc = get_accuracy(acc_preds[0], cum_total_pred, cum_total_gold)
        return acc,cum_total_pred,acc_preds

    def get_acc_from_pred_pairwise_scores(self, json_objects, pred_for_all_docs, test_window_size, is_training,is_consider_saleint,is_classification=False,is_add_pred_pairs=False):
        pred_counter = 0
        num_anaphors = 0
        start_current_ana = 0
        end_current_ana = 0
        cum_acc_pred, cum_total_pred, cum_total_gold = 0, 0, 0
        pairs_to_be_processed = pred_for_all_docs.shape[0]

        for file_counter,single_json_object in enumerate(json_objects):
            mention_ops = BridgingJsonDocInformationExtractor(single_json_object, logger)
            if mention_ops.is_file_empty:
                continue
            cand_ante_per_anaphor = self.data.generate_cand_ante_by_sentence_window(mention_ops, sen_window_size=test_window_size,
                                                                               is_consider_saleint=is_consider_saleint, is_training=is_training,
                                                                               ana_to_antecedent_map=None,
                                                                               anaphors=None)

            predicted_pairs = []
            assert len(cand_ante_per_anaphor) == len(mention_ops.anaphors)
            num_anaphors += len(mention_ops.anaphors)
            for ana_ind, cand_ante in enumerate(cand_ante_per_anaphor):
                # print("cand ante from vec {} and from here {}".format(len(cand_ante_per_anaphor_from_vec[ana_ind]),len(cand_ante)))
                curr_ana = mention_ops.anaphors[ana_ind]
                logger.debug(3 * "----")
                assert len(cand_ante) != 0
                if is_classification:
                    # print("file {}/{} - anaphor {}/{}".format(file_counter,len(json_objects)-1,ana_ind,len(mention_ops.anaphors)-1))
                    end_current_ana = start_current_ana + len(cand_ante)
                    # print("START :{}, eND {}".format(start_current_ana, end_current_ana))
                    curr_pred_scores = pred_for_all_docs[start_current_ana:end_current_ana].flatten()
                    start_current_ana = end_current_ana

                else:
                    pred = pred_for_all_docs[pred_counter]
                    logger.debug("all pred scores : {}".format(pred))
                    curr_pred_scores = pred[0:len(cand_ante)]  # select scores from legit candidates not from dummy
                logger.debug("curr anaphor {} candidates {} and scores {}".format(curr_ana, cand_ante,
                                                                                  curr_pred_scores))

                max_score_ind = np.argmax(curr_pred_scores)
                logger.debug("max scores at {}".format(max_score_ind))
                sel_ant = cand_ante[max_score_ind]
                logger.debug("anaphor :[{}]".format(mention_ops.get_words_for_start_end_indices(curr_ana)))
                logger.debug("candidate antecedents :{}".format(mention_ops.get_span_words(cand_ante)))
                logger.debug("selected antecedent : {}".format(mention_ops.get_words_for_start_end_indices(sel_ant)))
                pair = (sel_ant, curr_ana)
                predicted_pairs.append(pair)
                logger.debug("pair added {}".format(pair))

                pred_counter += 1

            assert len(predicted_pairs) == len(mention_ops.bridging_clusters)

            accurate_pred, total_pred, total_gold, pred_pair_labels = get_accurate_pred_gold_numbers(mention_ops,
                                                                                                     predicted_pairs)
            error_analysis(mention_ops, predicted_pairs, pred_pair_labels)
            cum_acc_pred += accurate_pred
            cum_total_gold += total_gold
            cum_total_pred += total_pred
            if is_add_pred_pairs:
                single_json_object['predicted_pairs'] = predicted_pairs
                self.pred_json_objects.append(single_json_object)
        if is_classification:
            assert pairs_to_be_processed == end_current_ana,"pairs to be processed {} and end {}".format(pairs_to_be_processed,end_current_ana)
        else:
            assert pred_for_all_docs.shape[0] == pred_counter == num_anaphors
        acc = get_accuracy(cum_acc_pred, cum_total_pred, cum_total_gold)
        return acc,cum_acc_pred, cum_total_pred, cum_total_gold,self.pred_json_objects

    def evaluate_svm_ranking(self, svm_model, eval_jsonlines, is_training=False,is_all_window=False,is_add_pred_pairs=False):
        json_objects = self.data.get_json_objects(eval_jsonlines)

        test_dat_file = "{}_test.dat".format(randrange(1000000))
        test_dat = os.path.join(self.svm_ranking_data_path,test_dat_file)

        pred_dat_file = "{}_pred.dat".format(randrange(1000000))
        pred_dat = os.path.join(self.svm_ranking_data_path,pred_dat_file)

        acc_preds = []
        test_window_sizes = [5]
        max_antes = [30]
        #test with all window sizes
        # if is_all_window and not is_training:
        #     test_window_sizes = [2,3,4,-1]
        #     max_antes = [30,50,70,130]
        # else:
        #     test_window_sizes = [2]
        #     max_antes = [30]

        # pred_objects = []
        for counter,test_window_size in enumerate(test_window_sizes):

            self.data.max_ante = max_antes[counter]
            self.data.generate_svm_data(eval_jsonlines,False,is_training, test_dat,
                                        sen_window_size=test_window_size, is_consider_saleint=self.is_consider_saleint, is_concat=self.is_concat_vecs, is_ignore_dummy=False, is_fourier_features=self.is_fourier_features, is_normalize=self.is_normalize)
            command = get_svm_rank_command(model_input_dat=test_dat, pred_dat=pred_dat,svm_model=svm_model, is_train=False)
            logger.debug("executing command {}".format(command))
            os.system(command)
            pred_for_all_docs = np.loadtxt(pred_dat)
            logger.debug("pred shape {}".format(pred_for_all_docs))
            pred_for_all_docs = pred_for_all_docs.reshape(-1,max_antes[counter])
            logger.debug("pred shape after reshape {}".format(pred_for_all_docs))
            acc,cum_acc_pred, cum_total_pred, cum_total_gold,_ = self.get_acc_from_pred_pairwise_scores(json_objects, pred_for_all_docs, test_window_size, is_training,self.is_consider_saleint,is_add_pred_pairs=is_add_pred_pairs)
            acc_preds.append(cum_acc_pred)
            # pred_objects.append(pred_json_objects)
            print("total anaphors {}, correctly linked {}, accuracy: {}, with window size {}".format(cum_total_pred,
                                                                                                     cum_acc_pred, acc,
                                                                                                     test_window_size))
        acc = get_accuracy(acc_preds[0], cum_total_pred, cum_total_gold)
        self.data.max_ante = max_ante
        return acc,cum_total_pred,acc_preds


    def nn_eval_and_report(self, sess, predictions, loss, train_jsonlines, dev_jsonlines, epoch, epoch_loss, batch_acc, max_acc, saver,
                           expt_name, loss_list, kwrgs, scores, is_evaluate_train):
        train_acc = 0
        eval_kwrgs = {"sess": sess,"loss":loss, "predictions": predictions, "eval_jsonlines": dev_jsonlines}
        eval_kwrgs.update(kwrgs)
        eval_kwrgs['eval_vecs'] = self.dev_eval_vecs
        dev_acc,_,_,dev_loss,_ = self.eval_nn_model_accuracy(**eval_kwrgs)
        scores.append(dev_acc)

        if is_evaluate_train:
            eval_kwrgs["eval_jsonlines"] = train_jsonlines
            eval_kwrgs["is_training"] = True
            eval_kwrgs['eval_vecs'] = self.train_eval_vecs
            train_acc,_,_,train_loss,_ = self.eval_nn_model_accuracy(**eval_kwrgs)

        print(
            "{0}:[loss={1:0.4f},train accuracy ={4:0.4f},dev loss ={3:0.4f}, dev accuracy={2:0.4f}]".format(epoch, epoch_loss,
                                                                                     dev_acc,dev_loss,train_acc))
        logger.critical(
            "{0}:[loss={1:0.4f},train accuracy ={4:0.4f},dev loss ={3:0.4f}, dev accuracy={2:0.4f}]".format(epoch, epoch_loss,
                                                                                     dev_acc,dev_loss,train_acc))
        logger.critical(3 * "*****")
        save_model(saver, sess, epoch, self.save_model_dir, curr_score=dev_acc, max_score=max_acc,
                   model_name=timestr+"_"+expt_name)
        if dev_acc > max_acc:
            max_acc = dev_acc
        loss_list.append(epoch_loss)
        if len(loss_list) > 10 and len(set(loss_list[-5:])) == 1:
            print("for last 50 iterations loss has been constant, training halted.")
            return True,train_acc, max_acc
        return False,train_acc, max_acc

    def nn_eval_final(self, sess, predictions, loss, test_josnlines, expt_name, saver, kwrgs, is_add_pred_pairs, is_load_model=True):
        if is_load_model:
            load_model(saver, sess, self.save_model_dir, model_name=timestr+"_"+expt_name)
        eval_kwrgs = {"sess": sess,"loss":loss, "predictions": predictions, "eval_jsonlines": test_josnlines}
        eval_kwrgs.update(kwrgs)
        eval_kwrgs["is_debug"] = True
        eval_kwrgs["is_add_pred_pairs"] = is_add_pred_pairs
        eval_kwrgs['eval_vecs'] = self.test_eval_vecs
        test_acc,cum_acc_pred, cum_total_pred,_,pred_json_objects = self.eval_nn_model_accuracy(**eval_kwrgs)
        print("{0}:[accuracy={1:0.4f}]".format("TEST EVALUATION", test_acc))
        logger.critical("{0}:[accuracy={1:0.4f}]".format("TEST EVALUATION", test_acc))
        # sys.exit()
        return test_acc,cum_acc_pred, cum_total_pred,pred_json_objects

    def eval_nn_model_accuracy(self, **kwrgs):
        """
        :return:
        """
        sess = kwrgs.get("sess")
        predictions = kwrgs.get("predictions")
        loss = kwrgs.get("loss")
        eval_jsonlines = kwrgs.get("eval_jsonlines")
        anaphors = kwrgs.get("anaphors")
        antecedents = kwrgs.get("antecedents")
        anaphors_context = kwrgs.get("anaphors_context")
        anaphors_ext = kwrgs.get("anaphors_ext")
        antecedents_context = kwrgs.get("antecedents_context")
        antecedents_ext = kwrgs.get("antecedents_ext")
        is_training = kwrgs.get("is_training", False)
        is_debug = kwrgs.get("is_debug", False)
        is_add_pred_pairs = kwrgs.get("is_add_pred_pairs", False)
        eval_vecs = kwrgs.get("eval_vecs")


        json_objects = self.data.get_json_objects(eval_jsonlines)
        l = 0
        m1_vec, m2_vec, m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, _ = eval_vecs

        if self.is_wsd:
            pred = sess.run(predictions, feed_dict={anaphors_context: m1_cont_vec,
                                                    anaphors_ext: m1_ext_vec,
                                                    antecedents_context: m2_cont_vec,
                                                    antecedents_ext: m2_ext_vec})
        else:
            pred = sess.run(predictions, feed_dict={anaphors: m1_vec, antecedents: m2_vec})

        acc,cum_acc_pred, cum_total_pred, cum_total_gold,pred_json_objects = self.get_acc_from_pred_pairwise_scores(json_objects,pred,self.test_sen_window_size,is_training,is_consider_saleint=True,is_add_pred_pairs=is_add_pred_pairs,is_classification=self.is_classification)
        return acc,cum_acc_pred, cum_total_pred,l,pred_json_objects



    def latent_svm_eval_and_report(self, epoch, epoch_loss, max_acc, loss_list, scores, is_evaluate_train):
        train_acc = 0
        test_loss, test_acc = None,None
        eval_kwrgs = {}
        eval_kwrgs["eval_jsonlines"] = self.dev_jsonlines
        eval_kwrgs["is_training"] = False
        eval_kwrgs['eval_vecs'] = self.dev_eval_vecs

        dev_acc,_,_,dev_loss,_ = self.eval_latent_svm_model_accuracy(**eval_kwrgs)
        scores.append(dev_acc)

        eval_kwrgs["eval_jsonlines"] = self.test_jsonlines
        eval_kwrgs['eval_vecs'] = self.test_eval_vecs
        test_acc, _, _, test_loss, _ = self.eval_latent_svm_model_accuracy(**eval_kwrgs)

        if is_evaluate_train:
            eval_kwrgs["eval_jsonlines"] = self.train_jsonlines
            eval_kwrgs["is_training"] = True
            eval_kwrgs['eval_vecs'] = self.train_eval_vecs
            train_acc,_,_,train_loss,_ = self.eval_latent_svm_model_accuracy(**eval_kwrgs)
        # print("train acc : {}".format(train_acc))
        # print("dev acc : {}".format(dev_acc))
        # print("epoch : {}".format(epoch))
        # print("epoch_loss : {}".format(epoch_loss))
        # print("dev loss : {}".format(dev_loss))
        self.training_stats.append_current_loss_acc_stats(epoch_loss,train_acc,dev_loss,dev_acc,test_loss,test_acc)
        print(
            "{0}:[loss={1:0.4f},train accuracy ={4:0.4f},dev loss ={3:0.4f}, dev accuracy={2:0.4f}]".format(epoch, epoch_loss,
                                                                                     dev_acc,dev_loss,train_acc))
        logger.critical(
            "{0}:[loss={1:0.4f},train accuracy ={4:0.4f},dev loss ={3:0.4f}, dev accuracy={2:0.4f}]".format(epoch, epoch_loss,
                                                                                     dev_acc,dev_loss,train_acc))
        logger.critical(3 * "*****")
        save_model(self.saver, self.sess, epoch, self.save_model_dir, curr_score=dev_acc, max_score=max_acc,
                   model_name=timestr+"_"+self.expt_name)
        if dev_acc > max_acc:
            max_acc = dev_acc
        loss_list.append(epoch_loss)
        if len(loss_list) > 10 and len(set(loss_list[-5:])) == 1:
            print("for last 50 iterations loss has been constant, training halted.")
            return True,train_acc, max_acc
        return False,train_acc, max_acc

    def latent_svm_eval_final(self,is_add_pred_pairs, is_load_model=True):
        if is_load_model:
            load_model(self.saver, self.sess, self.save_model_dir, model_name=timestr+"_"+self.expt_name)
        eval_kwrgs = {}
        eval_kwrgs["eval_jsonlines"] = self.test_jsonlines
        eval_kwrgs["is_debug"] = True
        eval_kwrgs["is_add_pred_pairs"] = is_add_pred_pairs
        eval_kwrgs['eval_vecs'] = self.test_eval_vecs
        test_acc,cum_acc_pred, cum_total_pred,_,pred_json_objects = self.eval_latent_svm_model_accuracy(**eval_kwrgs)
        print("{0}:[accuracy={1:0.4f}]".format("TEST EVALUATION", test_acc))
        logger.critical("{0}:[accuracy={1:0.4f}]".format("TEST EVALUATION", test_acc))
        # sys.exit()
        return test_acc,cum_acc_pred, cum_total_pred,pred_json_objects



    def eval_latent_svm_model_accuracy(self, **kwrgs):
        """
        :return:
        """
        eval_jsonlines = kwrgs.get("eval_jsonlines")
        is_training = kwrgs.get("is_training", False)
        is_add_pred_pairs = kwrgs.get("is_add_pred_pairs", False)
        eval_vecs = kwrgs.get("eval_vecs")

        json_objects = self.data.get_json_objects(eval_jsonlines)
        l = 0
        m1_vec, m2_vec, m1_cont_vec, m1_ext_vec, m2_cont_vec, m2_ext_vec, true_labels = eval_vecs

        dummy_labels = np.zeros_like(true_labels)
        feed_dict = {self.anaphors_context: m1_cont_vec,self.anaphors_ext: m1_ext_vec,
                     self.antecedents_context: m2_cont_vec,self.antecedents_ext: m2_ext_vec,
                     self.pair_labels:true_labels}


        self.sess.run(self.iterator.initializer, feed_dict=feed_dict)
        # pred = self.sess.run(self.predictions, feed_dict=feed_dict)
        ss = []
        num_batches = 0
        try:
            while True:
                pred,_l = self.sess.run([self.predictions,self.loss])
                # print(pred.shape)
                ss.append(pred)
                l+=_l
                num_batches +=1
        except tf.errors.OutOfRangeError:
            pred = np.concatenate(ss)
            l = l/num_batches
            # print(pred.shape)
        acc,cum_acc_pred, cum_total_pred, cum_total_gold,pred_json_objects = self.get_acc_from_pred_pairwise_scores(json_objects,pred,self.test_sen_window_size,is_training,is_consider_saleint=True,is_add_pred_pairs=is_add_pred_pairs,is_classification=self.is_classification)
        # sys.exit()
        return acc,cum_acc_pred, cum_total_pred,l,pred_json_objects