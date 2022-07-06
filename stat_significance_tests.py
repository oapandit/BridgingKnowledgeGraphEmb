import sys
import numpy as np
from scipy import stats

## McNemar test
def calculateContingency(data_A, data_B, n):
    ABrr = 0
    ABrw = 0
    ABwr = 0
    ABww = 0
    for i in range(0,n):
        if(data_A[i]==1 and data_B[i]==1):
            ABrr = ABrr+1
        if (data_A[i] == 1 and data_B[i] == 0):
            ABrw = ABrw + 1
        if (data_A[i] == 0 and data_B[i] == 1):
            ABwr = ABwr + 1
        else:
            ABww = ABww + 1
    return np.array([[ABrr, ABrw], [ABwr, ABww]])

def mcNemar(table):
    statistic = float(np.abs(table[0][1]-table[1][0]))**2/(table[1][0]+table[0][1])
    pval = 1-stats.chi2.cdf(statistic,1)
    return pval

def mcNemar_test(data_A, data_B,alpha=0.01):
    # print(
    #     "\nThis test requires the results to be binary : A[1, 0, 0, 1, ...], B[1, 0, 1, 1, ...] for success or failure on the i-th example.")
    f_obs = calculateContingency(data_A, data_B, len(data_A))
    mcnemar_results = mcNemar(f_obs)
    if (float(mcnemar_results) <= float(alpha)):
        print("\nTest result is significant with p-value: {}".format(mcnemar_results))
        return
    else:
        print("\nTest result is not significant with p-value: {}".format(mcnemar_results))
        return

def wilcoxon_test(data_A, data_B,alpha=0.01):
    wilcoxon_results = stats.wilcoxon(data_A, data_B)
    if (float(wilcoxon_results[1]) <= float(alpha)):
        print("\n Wilcoxon significance test result is significant with p-value: {}".format(wilcoxon_results[1]))
        return
    else:
        print("\n Wilcoxon significance test result is not significant with p-value: {}".format(wilcoxon_results[1]))
        return