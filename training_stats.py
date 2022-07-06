import os
import matplotlib.pyplot as plt
from bridging_constants import *
import numpy as np
def simple_line_plot(y_values, x_lable,y_lable, legends, fig_name):
    x = list(range(len(y_values[0])))
    color_styles = ['-r','-b',"-g"]
    plt.style.use('seaborn-whitegrid')
    # assert len(y_values) == len(legends)
    for i,(y_value,legend) in enumerate(zip(y_values, legends)):
        if len(y_value)==len(x):
            plt.plot(x, y_value,color_styles[i],label=legend)
    # plt.axis('equal')
    plt.legend();

    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.savefig(fig_name)
    plt.clf()

class TrainingStats:
    def __init__(self):
        self.plts_dir = train_plots_path
        self.train_loss = []
        self.dev_loss = []
        self.train_acc = []
        self.dev_acc = []
        self.test_loss = []
        self.test_acc = []
        self.scoring_weights = []
        self.gradients = []

    def append_current_loss_acc_stats(self,t_loss,t_acc,d_loss,d_acc,test_loss=None,test_acc=None):
        self.train_loss.append(t_loss)
        self.dev_loss.append(d_loss)
        self.train_acc.append(t_acc)
        self.dev_acc.append(d_acc)
        if test_acc is not None:
            self.test_acc.append(test_acc)
        if test_loss is not None:
            self.test_loss.append(test_loss)


    def append_current_wts_grads(self,wt,grad):
        self.scoring_weights.append(wt)
        self.gradients.append(grad)


    def get_training_stats(self):
        assert len(self.train_loss) == len(self.dev_loss)
        fig_name = os.path.join(self.plts_dir,"train_dev_loss_plot.png")
        simple_line_plot([self.train_loss,self.dev_loss,self.test_loss], "Epochs", "Loss", ["train","dev","test"], fig_name)

        assert len(self.train_acc) == len(self.dev_acc)
        fig_name = os.path.join(self.plts_dir,"train_dev_acc_plot.png")
        simple_line_plot([self.train_acc,self.dev_acc,self.test_acc], "Epochs", "Accuracy", ["train","dev","test"], fig_name)

        # assert len(self.scoring_weights) == len(self.gradients)
        #
        # norm = lambda x: [np.linalg.norm(np.array(_x)) for _x in x]
        # wts = norm(self.scoring_weights)
        # grads = norm(self.gradients)
        #
        # fig_name = os.path.join(self.plts_dir, "scoring_wts_plot.png")
        # simple_line_plot([wts], "Epochs", "Weights", ["weights norm"], fig_name)
        #
        # fig_name = os.path.join(self.plts_dir, "grad_plot.png")
        # simple_line_plot([grads], "Epochs", "gradients", ["gradient"], fig_name)