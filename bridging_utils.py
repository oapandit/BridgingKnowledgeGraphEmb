import common_code
from common_utils import *

import logging
from logging.handlers import RotatingFileHandler
import time
from bridging_constants import *
import fnmatch
import numpy as np
import socket
import sys
from sklearn.utils import shuffle as shuffle_arrays


hostname = socket.gethostname()

create_dir(data_path)
create_dir(result_path)
create_dir(log_files_path)
create_dir(is_notes_vec_path)

# Logger settings -------------------------------------------------

logger = logging.getLogger()
timestr = time.strftime("%d%m%Y-%H%M%S")
logger.setLevel(logging.DEBUG)
create_dir(log_files_path)
create_dir(result_path)

log_format = ('[%(levelname)-8s %(filename)s:%(lineno)s] %(message)s')
# output debug logs to this file
debug_file = os.path.join(log_files_path, timestr + '.debug.log')
# fh = logging.FileHandler(debug_file)
fh = RotatingFileHandler(debug_file, maxBytes=10000000, backupCount=2)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter(log_format)
fh.setFormatter(formatter)
logger.addHandler(fh)

# output only info logs to this file
log_format = ('%(message)s')
info_file = os.path.join(log_files_path, timestr + '.info.log')
# fh = logging.FileHandler(info_file)
fh = RotatingFileHandler(info_file, maxBytes=10000000, backupCount=2)
fh.setLevel(logging.INFO)
formatter = logging.Formatter(log_format)
fh.setFormatter(formatter)
logger.addHandler(fh)

# output only info logs to this file
log_format = ('%(message)s')
criti_file = os.path.join(result_path, timestr + '.result.log')
# fh = logging.FileHandler(info_file)
fh = RotatingFileHandler(criti_file)
fh.setLevel(logging.CRITICAL)
formatter = logging.Formatter(log_format)
fh.setFormatter(formatter)
logger.addHandler(fh)

def search_and_get_file_path(path,pattern):
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                file_path = os.path.join(root, basename)
                return file_path
    print("ERROR : No file with {0} pattern in directory {1}".format(pattern,path))
    sys.exit()

def get_id_number(id):
    id_cont_list = id.split('_')
    assert len(id_cont_list)==2
    return int(id_cont_list[1])

def remove_markers(sentence):
    words_list = sentence.split()
    words_list = [w for w in words_list if not (w==span_start_indi or w==span_end_indi)]
    return " ".join(words_list)

def down_sample(label, *args):
    pos_ind = np.equal(label, 1).reshape(-1)
    neg_ind = np.equal(label, 0).reshape(-1)
    num_pos_samples = label[pos_ind].shape[0]
    logger.debug("number of positive samples {} and negative examples {}".format(num_pos_samples,label[neg_ind].shape[0]))

    balanced_samples = []
    args = list(args)
    args.append(label)
    for a in args:
        logger.debug(5*"---")
        logger.debug("original shape {}".format(a.shape))
        logger.debug("positve shape {}".format(a[pos_ind].shape))
        logger.debug("negative shape {}".format(a[neg_ind][0:num_pos_samples].shape))
        a_balanced = np.concatenate([a[pos_ind], a[neg_ind][0:num_pos_samples]])
        logger.debug("after sampling shape {}".format(a_balanced.shape))
        balanced_samples.append(a_balanced)
    balanced_samples = shuffle_arrays(*balanced_samples)
    return balanced_samples

def log_start(txt):
    logger.debug("********* {} **********".format(txt))

def log_finish():
    logger.debug("*********")

def print_dims(*args):
    print("****** training data dimensions ******")
    for a in args:
        if isinstance(a,list): continue
        print(a.shape)
        if a.shape[-1]==1:
            print("positive samples : {}".format(np.count_nonzero(a == 1)))
    print("***************")
    # print_dims("total samples {} positive {}".format(args[-1].shape,np.count_nonzero(args[-1] == 1)))

def normalize(v):
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    v = (v - v_min) / (v_max - v_min)
    return v


def execute_func(func,func_params,mp,is_serial_exec=True,parallel_execute_jobs=20):
    jobs = []
    for i,func_param in enumerate(func_params):
        if is_serial_exec:
            func(*func_param)
        else:
            process = mp.Process(target=func,args=(func_param))
            jobs.append(process)
            if i % parallel_execute_jobs == 0 or i == len(func_params) - 1:
                for j in jobs:
                    j.start()
                # Ensure all of the processes have finished
                for j in jobs:
                    j.join()
                jobs = []

def create_data_obj(mp,obj,is_serially):
    manager = mp.Manager()
    if obj==dict:
        return {} if is_serially else manager.dict()
    elif obj == list:
        return [] if is_serially else manager.list()

def get_svm_rank_command(c=None, kernel=None, gamma=None, degree=None, model_input_dat=None,pred_dat=None,svm_model=None, is_train=None):
    command = ""
    if is_train:
        k = -1
        if kernel in kernels:
            k = kernels.index(kernel)
        if kernel == 'linear': assert k == 0
        if k == 0:
            command = "cd {}/Code/svm;./svm_rank_learn -v 0 -c {} -t {} {} {}".format(base_dir, c, k, model_input_dat,
                                                                                      svm_model)
        elif k == 1:
            command = "cd {}/Code/svm;./svm_rank_learn -v 3 -y 3 -c {} -t {} -d {} {} {}".format(base_dir, c, k, degree,
                                                                                                 model_input_dat, svm_model)
        elif k == 2:
            command = "cd {}/Code/svm;./svm_rank_learn -v 3 -y 3 -c {} -t {} -g {} {} {}".format(base_dir, c, k, gamma,
                                                                                                 model_input_dat, svm_model)
        else:
            command = "cd {}/Code/svm;./svm_rank_learn -v 0 -c 3 {} {}".format(base_dir, model_input_dat, svm_model)
    else:
        command = "cd {}/Code/svm/;./svm_rank_classify -v 0 {} {} {}".format(base_dir, model_input_dat, svm_model, pred_dat)
    print(command)
    return command


def average(l):
    return sum(l)/len(l)