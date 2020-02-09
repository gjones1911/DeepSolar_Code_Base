import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dis_mod

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import sys
from _products.utility_fnc import *
from _products.visualization_tools import *
from sklearn import metrics
pd.options.mode.use_inf_as_na = True
viz = Visualizer()


# #####################################################
# #####################################################
# #########   TODO: Regression Performance     ########
# #####################################################
# #####################################################
def Rvar(ytrue, ypred):
    ymean = ypred.mean(axis=0)
    ssreg = SSREG(ytrue, ymean=ymean)
    ssres = SSRES(ytrue=ytrue, ypred=ypred)
    return (SSREG(ypred, ymean) / len(ypred)) / (SSTOT(ytrue) / len(ypred))

def SSE( ytrue, ypred):
    return sum([(yt - yp) ** 2 for yp, yt in zip(ytrue, ypred)])

def MSE(ytrue, ypred):
    n = len(ytrue)
    return SSE(ytrue, ypred) / n

def RMSE(ytrue, ypred):
    return math.sqrt(MSE(ytrue, ypred))

def MAD(ytrue, ypred):
    n = len(ytrue)
    return sum([abs(yt - yp) for yp, yt in zip(ytrue, ypred)]) / n

def MAE(ytrue, ypred):
    n = len(ytrue)
    return sum([abs(yt - yp) for yp, yt in zip(ytrue, ypred)]) / n


def SSREG(ypred, ymean):
    return sum([(yp - ymean) ** 2 for yp in ypred])

def SSRES(ytrue, ypred):
    return sum([(yt - yp) ** 2 for yp, yt in zip(ytrue, ypred)])

def COD(ytrue, ypred):
    return 1 - (SSRES(ytrue, ypred)/SSTOT(ytrue))

def SSTOT(ytrue):
    ymean = ytrue.mean(axis=0)
    return sum([(yt - ymean) ** 2 for yt in ytrue])  # scatter total (sum of squares)

def calculate_log_like(attribs, params):
    #attribs.append('const')
    l = []
    for attrib in attribs:
        l.append(params[attrib])
    return np.exp(l).tolist()

def calculate_vif (x):
    return pd.Series([VIF(x.values, i)
                      for i in range(x.shape[1])],
                      index=x.columns)


# #############################################################
# #############################################################
# ###########    TODO: Classification metrics    ##############
#                      source: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
# #############################################################
# #############################################################
def class_count(ytrue, val):
    return len(ytrue[ytrue == val])

def class_prob(ytrue, val):
    return class_count(ytrue, val)/len(ytrue)

def c_p_handler2(method, param_dict):
    method(param_dict[0], param_dict[1])

def c_p_handler3(method, param_dict):
    method(param_dict[0], param_dict[1], param_dict[2])

def c_p_handler4(method, param_dict):
    method(param_dict[0], param_dict[1], param_dict[2], param_dict[3])

def correct(ytrue, ypredict):
    if not type_check(ytrue, 'numpy'):
        ytrue = np.array(ytrue)
    if not type_check(ypredict, 'numpy'):
        ypredict = np.array(ypredict)
    ytrue = ytrue.reshape(len(ytrue), 1)
    ypredict = ypredict.reshape(len(ypredict), 1)
    # count the predictions that are correct
    return sum(yt == yp for yt, yp in zip(ytrue, ypredict))

def shape_check(yc, yd):
    if yc.shape != yd.shape:
        yc =  yc.reshape(yd.shape[0], yd.shape[1])
    return yc

def cnt_false(y):
    return sum([e[0] == 0 for e in y])

def cnt_true(y):
    return sum([e == 1 for e in y])

def cnt_val(y, val):
    return sum([e == val for e in y])

def correct_label(ytrue, ypredict, label):
    ypredict = shape_check(ypredict, ytrue)
    # count the predictions for labeling of label that are correct
    # grab every thing that agrees with the truth
    # and count the number that were correctly labeled label
    #return ytrue[ytrue == ypredict and ytrue == label].tolist().count(label)
    return len(ytrue[ytrue == ypredict and ytrue == label])

def true_positives(ytrue, ypredict, label=1):
    ypredict = shape_check(ypredict, ytrue)
    # count the predictions for a 1 that are correct
    # grab every thing that agrees with the truth
    # and count the number that were correctly labeled label
    tp = sum([yt == yp and yt == 1 for yt, yp in zip(ytrue, ypredict)])
    return tp[0]

def true_negatives(ytrue, ypredict, label=0):
    ypredict = shape_check(ypredict, ytrue)
    # count the predictions for label that are correct
    # grab every thing that agrees with the truth
    # and count the number that were miss labeled label
    if type(ytrue) != type(np.array([0])):
        ytrue = np.array(ytrue)
    if type(ypredict) != type(np.array([0])):
        ypredict = np.array(ypredict)
    """
    print('--------------------------------------')
    print('after the conversion to numpy arrays')
    print('ytrue, size: {}'.format(len(ytrue)))
    print('ypredict, size: {}'.format(len(ypredict)))
    print('ytrue, shape: {}'.format(ytrue.shape))
    print('ypredict, shape: {}'.format(ypredict.shape))
    print('--------------------------------------')
    print('--------------------------------------')
    print('ypredict')
    print(ypredict)
    print('ytrue')
    print(ytrue)
    print()
    """
    ytrue = ytrue.reshape(len(ytrue),1)
    ypredict = ypredict.reshape(len(ypredict),1)
    """
    print('--------------------------------------')
    print('after the conversion to numpy arrays')
    print('ytrue, size: {}'.format(len(ytrue)))
    print('ypredict, size: {}'.format(len(ypredict)))
    print('ytrue, shape: {}'.format(ytrue.shape))
    print('ypredict, shape: {}'.format(ypredict.shape))
    print('--------------------------------------')
    print('--------------------------------------')
    print('ypredict')
    print(ypredict)
    print('ytrue')
    print(ytrue)
    print()
    """
    quck = sum([yt==yp for yt, yp in zip(ytrue, ypredict)])
    #print('correct total: {}'.format(quck))
    c_z = sum([(yp == yt and yp) == 0 for yp, yt in zip(ypredict, ytrue)])
    #print('correct by iter', c_z)
    #print('label',label)
    #c_z_f = ytrue[ytrue == ypredict and ytrue == label].tolist().count(label)
    #rint('correct by fancy', c_z_f)
    return c_z[0]

def incorrect(ytrue, ypredict, timed=False):
    ypredict = shape_check(ypredict, ytrue)
    # count the predictions for a label that are incorrect
    # grab every thing that agrees with the truth
    # and count the number that were correctly labeled label
    return sum(yt == yp for yt, yp in zip(ytrue, ypredict))[0]

def incorrect_label(ytrue, ypredict, label, timed=False):
    ypredict = shape_check(ypredict, ytrue)
    # count the predictions for a label that are incorrect
    # grab every thing that disagrees with the truth
    # and count the number that were incorrectly labeled label
    return sum(yt != yp and yp == label for yt, yp in zip(ytrue, ypredict))[0]

def false_positives(ytrue, ypredict, label=1):
    ypredict = shape_check(ypredict, ytrue)
    # count the predictions for a 1 that are incorrect
    # grab every thing that disagrees with the truth
    # and count the number of label that were miss labeled
    return sum(yt != yp and yp == 1 for yt, yp in zip(ytrue, ypredict))[0]

def false_negatives(ytrue, ypredict, label=0):
    ypredict = shape_check(ypredict, ytrue)
    # count the predictions for a 0 that are incorrect
    # grab every thing that disagrees with the truth
    # and count the number of label that were miss labeled
    return sum(yt != yp and yp == 0 for yt, yp in zip(ytrue, ypredict))[0]

def false_others(ytrue, ypredict, label):
    ypredict = shape_check(ypredict, ytrue)
    false_others_cnt_dic = {}
    # get the others
    # from predicted list
    other_pred = ypredict[ypredict != label]
    # from true list
    other_true = ytrue[ytrue != label]
    # grab the unique values for the other labels
    others = set(ypredict[ypredict != label].tolist())
    for other in others:
        # how many
        false_others_cnt_dic[other] = len(other_pred[other_pred == label and other_true == other ])
    return false_others_cnt_dic

def accuracy(ytrue, ypredict):
    ypredict = shape_check(ypredict, ytrue)
    return correct(ytrue, ypredict)/len(ytrue)

def sensitivity(ytrue, ypredict, label=1):
    ypredict = shape_check(ypredict, ytrue)
    """ accuracy in predicting positives
    :param ytrue: the ground truth outcomes
    :param ypredict: the predicted outcomes
    :return: #float true positive predictions / the total number of oucomes
    """
    tp = true_positives(ytrue, ypredict, label=label)
    N = cnt_true(ytrue)
    return tp/max(N, 1e-15)

def specificity(ytrue, ypredict):
    """ accuracy in predicting negatives
        :param ytrue: the ground truth outcomes
        :param ypredict: the predicted outcomes
        :return: #float true negative predictions / the total number of outcomes
    """
    ytrue = np.array(ytrue).reshape(len(ytrue), 1)
    ypredict = np.array(ypredict).reshape(len(ypredict), 1)
    tn = true_negatives(ytrue, ypredict, 0)
    N = cnt_false(ytrue)
    return max(tn, 1e-15) / max(N, 1e-15)



def precision(ytrue, ypredict, label=1):
    """   quality/ability to predict ones correctly (how well it does not call negatives positive
            :param ytrue: the ground truth outcomes
            :param ypredict: the predicted outcomes
            :return: #float true negative predictions / the total number of outcomes
    """
    tp = true_positives(ytrue, ypredict, label)
    fp = false_positives(ytrue, ypredict, label)
    return tp / max(tp+fp, 1e-15)

def NPV(ytrue, ypredict):
    """  negative predictive value (NPV) a measure of how well it does not call positivs negative
            :param ytrue: the ground truth outcomes
            :param ypredict: the predicted outcomes
            :return: #float true negative predictions / the total number of outcomes
    """
    tn = true_negatives(ytrue, ypredict)
    fn = false_negatives(ytrue, ypredict)
    return tn / max(tn+fn, 1e-15)


def Gconfusion_matrix(ytrue, ypredict):
    """ Generates a binary confusion matrix
    :param ytrue:
    :param ypredict:
    :return:
    """
    tp = true_positives(ytrue, ypredict)
    tn = true_negatives(ytrue, ypredict)
    fp = false_positives(ytrue, ypredict)
    fn = false_negatives(ytrue, ypredict)
    cm = [[tn, fp],
          [fn, tp]]
    return cm

class ClassificationPerformance:
    def class_count(self, ytrue, val):
        return len(ytrue[ytrue == val])

    def class_prob(self, ytrue, val):
        return class_count(ytrue, val) / len(ytrue)

    def c_p_handler2(self, method, param_dict):
        method(param_dict[0], param_dict[1])

    def c_p_handler3(self, method, param_dict):
        method(param_dict[0], param_dict[1], param_dict[2])

    def c_p_handler4(self, method, param_dict):
        method(param_dict[0], param_dict[1], param_dict[2], param_dict[3])

    def correct(self, ytrue, ypredict):
        # count the predictions that are correct
        return len(ytrue[ytrue == ypredict])

    def correct_label(self, ytrue, ypredict, label):
        # count the predictions for labeling of label that are correct
        # grab every thing that agrees with the truth
        # and count the number that were correctly labeled label
        # return ytrue[ytrue == ypredict and ytrue == label].tolist().count(label)
        return len(ytrue[ytrue == ypredict and ytrue == label])

    def true_positives(self, ytrue, ypredict, label=1):
        # count the predictions for a 1 that are correct
        # grab every thing that agrees with the truth
        # and count the number that were correctly labeled label
        return ytrue[ytrue == ypredict].tolist().count(label)

    def true_negatives(self, ytrue, ypredict, label=0):
        # count the predictions for label that are correct
        # grab every thing that agrees with the truth
        # and count the number that were miss labeled label
        return ytrue[ytrue == ypredict].tolist().count(label)

    def incorrect(self, ytrue, ypredict, timed=False):
        # count the predictions for a label that are incorrect
        # grab every thing that agrees with the truth
        # and count the number that were correctly labeled label
        return len(ytrue[ytrue != ypredict])

    def incorrect_label(self, ytrue, ypredict, label, timed=False):
        # count the predictions for a label that are incorrect
        # grab every thing that disagrees with the truth
        # and count the number that were incorrectly labeled label
        return ypredict[ypredict != ytrue].tolist().count(label)

    def false_positives(self, ytrue, ypredict, label=1):
        # count the predictions for a 1 that are incorrect
        # grab every thing that disagrees with the truth
        # and count the number of label that were miss labeled
        return ypredict[ypredict != ytrue].tolist().count(label)

    def false_negatives(self, ytrue, ypredict, label=0):
        # count the predictions for a 0 that are incorrect
        # grab every thing that disagrees with the truth
        # and count the number of label that were miss labeled
        return ypredict[ypredict != ytrue].tolist().count(label)

    def false_others(self, ytrue, ypredict, label):
        false_others_cnt_dic = {}
        # get the others
        # from predicted list
        other_pred = ypredict[ypredict != label]
        # from true list
        other_true = ytrue[ytrue != label]
        # grab the unique values for the other labels
        others = set(ypredict[ypredict != label].tolist())
        for other in others:
            # how many
            p_o =ypredict[ypredict != ytrue and ypredict == other]
            t_o = other_pred[other_true == other]
            false_others_cnt_dic[other] = len()
        return false_others_cnt_dic

    def accuracy(self, ytrue, ypredict):
        return correct(ytrue, ypredict) / len(ytrue)

    def sensitivity(self, ytrue, ypredict, label=1):
        """ accuracy in predicting positives
        :param ytrue: the ground truth outcomes
        :param ypredict: the predicted outcomes
        :return: #float true positive predictions / the total number of oucomes
        """
        tp = true_positives(ytrue, ypredict, label=label)
        N = len(ypredict)
        return tp / max(N, 1e-15)

    def specificity(self, ytrue, ypredict, label=0):
        """ accuracy in predicting negatives
            :param ytrue: the ground truth outcomes
            :param ypredict: the predicted outcomes
            :return: #float true negative predictions / the total number of outcomes
        """
        tn = true_negatives(ytrue, ypredict, label)
        N = len(ypredict)
        return tn / max(N, 1e-15)

    def precision(self, ytrue, ypredict, label=1):
        """   quality/ability to predict ones correctly (how well it does not call negatives positive
                :param ytrue: the ground truth outcomes
                :param ypredict: the predicted outcomes
                :return: #float true negative predictions / the total number of outcomes
        """
        tp = true_positives(ytrue, ypredict, label)
        fp = false_positives(ytrue, ypredict, label)
        return tp / max(tp + fp, 1e-15)

    def NPV(self, ytrue, ypredict):
        """  negative predictive value (NPV) a measure of how well it does not call positivs negative
                :param ytrue: the ground truth outcomes
                :param ypredict: the predicted outcomes
                :return: #float true negative predictions / the total number of outcomes
        """
        tn = true_negatives(ytrue, ypredict)
        fn = false_negatives(ytrue, ypredict)
        return tn / max(tn + fn, 1e-15)

    def __init(self, yt=None, yp=None):
        self.yt = yt
        self.yp = yp



def find_significant(x,pvals):
    cnt = -1
    for e in pvals:
        if cnt > -1:
            print(x[cnt], ":", np.around(e,4))
        cnt += 1


def h_regression(dataset, ysets, xsets):

    blocks = list()
    #dataset = fix_dataset(dset[ysets+xsets[0]])
    for y in ysets:
        print('##############################################################################')
        print('\t\t\t\t\t\t',y)
        print('##############################################################################')
        cnt = 0
        for x in xsets:
            blocks += x
            # my method up above to take care of missing or unusable values
            dmodel  = fix_dataset(dataset[[y]+blocks])
            Y = dmodel[y]
            print()
            print()
            print('################################################################################')
            print('#####################################     Block {:d}'.format(cnt+1))
            print('################################################################################')
            print('\t\tX', x)
            print('################################################################################')
            print('################################################################################')
            print('################################################################################')
            print()
            X = dmodel[blocks]
            #print(X['per_capita_income'])
            #X.loc[:, 'per_capita_income'] = (dmodel['per_capita_income'].values - dmodel['per_capita_income'].mean())/dmodel['per_capita_income'].std()
            #print(X['per_capita_income'])
            #X = dataset.loc[:, x]
            X2 = sm.add_constant(X)
            est = sm.OLS(Y, X2)
            est2 = est.fit()
            print(est2.summary())
            cnt += 1
            print()
            print()
    return


# performs some for of regression
# either linear or logistic
def analyze_data(ysets, xsets, ytest, xtest, type='LinR', normalize=False):
    #dataset = fix_dataset(dset[ysets+xsets[0]])
    regre_type = ''
    if type == 'LinR':
        regre_type = 'Linear Regression'
    elif type == 'LogR':
        regre_type = 'Logistic Regression'
    else:
        print('Error Unknown regression method {:s}'.format(type))
        quit()
    old_rsqr = 0
    old_fstat = 10e20
    del_rsqr = 0
    del_fstat = 0
    num_sig = 0
    for y, yt in zip(ysets, ytest):
        print('##############################################################################')
        #print('\t\t\t\t\t\t',y)
        print('##############################################################################')
        cnt = 0
        for x, xt in zip(xsets, xtest):

            Y = y
            Yt = yt
            print(len(Y), len(x))
            print()
            print('################################################################################')
            print('#####################################    Testing x set {:d}'.format(cnt+1))
            #print('#####################################    Using {:s} on dependent variable {:s}'.format(regre_type, y))
            print('################################################################################')
            print('\t\tX or dependent variables:\n', x.columns.values.tolist())
            print('################################################################################')
            print('################################################################################')
            print('################################################################################')
            print()
            X = x
            Xt = xt
            #print('+++++++++++++++++++++++++++++++++++++++++Before: ', X[0,0])

            #print('+++++++++++++++++++++++++++++++++++++++++After: ', X.iloc[0,0])
            #print(X['per_capita_income'])
            #X.loc[:, 'per_capita_income'] = (dmodel['per_capita_income'].values - dmodel['per_capita_income'].mean())/dmodel['per_capita_income'].std()
            #print(X['per_capita_income'])
            #X = dataset.loc[:, x]
            X2 = sm.add_constant(X)
            Xt2 = sm.add_constant(Xt)
            if type == 'LinR':
                est = sm.OLS(Y, X2)
                print('\n\nThe basic dirs are\n', dir(est))
                est2 = est.fit()
                print('\n\nThe fitted dirs are\n', dir(est2))
                rsqr = est2.rsquared
                if rsqr > old_rsqr:
                    old_rsqr = rsqr
                pvals = est2.pvalues
                fval = est2.fvalue
                ftest = est2.f_test
                print('R-squared:',rsqr)
                print('P-values:\n', pvals)
                find_significant(x, pvals)
                print('Fvalue\n',fval)
                print(est2.summary())
                print('\n\nThe summary dirs are:\n',dir(est2.summary()))
                vif = calculate_vif(X2)
                print('VIF:\n', vif)
            elif type == 'LogR':
                #clf = LogisticRegression(solver='lbfgs',max_iter=1000).fit(X2, Y)
                #params = clf.coef_
                #log_like = np.log(np.abs(params))
                #print(params)
                #print(log_like)
                #print('the y and x')
                #print(Y.values, X2.values)
                n = len(X2)
                print('n',n)
                model = dis_mod.Logit(Y.values, X2)
                model2 = model.fit()
                loglikly= calculate_log_like(x, model2.params)
                print(dir(model))
                print(model.df_model)
                print(model2.summary())
                llfv = model2.llf
                llnullv = model2.llnull
                print('llf: ', llfv)
                print('llf: ', llnullv)
                print('McFadden’s pseudo-R-squared: ', 1 - (llfv/llnullv)) # https://statisticalhorizons.com/r2logistic
                cxsn = G_Cox_Snell_R2(llnullv, llfv, n)
                print('Cox\'s Snell: {}'.format(cxsn) )
                print('model 2',dir(model2))
                print('R squared:', model2.prsquared) # McFadden’s pseudo-R-squared.
                print(dir(model2.summary().tables))
                print('The log likelyhoods are:')
                show_labeled_list(loglikly, x)
                print('pvalue for {:s}: {:f}'.format(X2.columns.values.tolist()[0], model2.pvalues.loc[x.columns.values.tolist()[0]]))
                y_pred = model2.predict(Xt2, linear=True)
                #print(y_pred)
                yp = list()
                for e in y_pred:
                    if e > 0:
                        yp.append(1)
                    else:
                        yp.append(0)
                #print(model.loglikeobs(x))
                #df_confusion = pd.crosstab(Y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
                viz.plot_confusion_matrix(Yt, yp, classes=['NA', 'A'],
                                      title='Confusion matrix, without normalization')
                #plot_confusion_matrix(df_confusion)
                #vif = pd.Series([VIF(X2.values, i)
                #           for i in range(X2.shape[1])],
                #          index=X2.columns)
                vif = calculate_vif(X2)
                print('VIF:\n',vif)
                plt.show()
            cnt += 1
            print()
            print()
    return

def G_Cox_Snell_R2(llnull, llmodel, n):
    v = 2/n
    print('v',v)
    va = np.exp(llnull)
    vb = np.exp(llmodel)
    print('va, vb', va, vb)
    return 1 - (va/vb)**v

# #####################################################
# #####################################################
# #########   TODO: Regression Performance     ########
# #####################################################
# #####################################################
def SM_Logit(Training, Testing, verbose=False):
    X = Training[0]
    Y = Training[1]
    print(X)
    Xt = Testing[0]
    Yt = Testing[1]

    # add the constant to the model
    X2 = sm.add_constant(X)
    Xt2 = sm.add_constant(Xt)

    # grab the size of the data
    n = len(X2)
    if verbose:
        print('n', n)
    # create and fit the model
    model = dis_mod.Logit(Y.values, X2)
    model2 = model.fit()
    # calculate the loglikely hood
    loglikly = calculate_log_like(X, model2.params)

    if verbose:
        print(dir(model))
        print(model.df_model)
        print(model2.summary())
    # grab the log likely hood for the model and just the intercept for later calculations
    llfv = model2.llf
    llnullv = model2.llnull
    print('llf: ', llfv)
    print('llf: ', llnullv)
    print('McFadden’s pseudo-R-squared: ', 1 - (llfv / llnullv))  # https://statisticalhorizons.com/r2logistic
    cxsn = G_Cox_Snell_R2(llnullv, llfv, n)
    print('Cox\'s Snell: {}'.format(cxsn))
    print('model 2', dir(model2))
    print('R squared:', model2.prsquared)  # McFadden’s pseudo-R-squared.
    # print(dir(model2.summary().tables))
    print('The log likelyhoods are:')
    show_labeled_list(loglikly, X)
    print('pvalue for {:s}: {:f}'.format(X2.columns.values.tolist()[0], model2.pvalues.loc[X.columns.values.tolist()[0]]))
    y_pred = model2.predict(Xt2, linear=True)
    # print(y_pred)
    yp = list()
    predicted_prob = list()
    for e in y_pred:
        # print('e: {}, ln(e): {}, e^(e): {}'.format(e, np.log(e), np.exp(e)))
        predicted_prob.append(np.exp(e))
        if e > 0:
            yp.append(1)
        else:
            yp.append(0)
    # print(model.loglikeobs(x))
    # df_confusion = pd.crosstab(Y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    rd = viz.plot_confusion_matrix(Yt, yp, classes=['NA', 'A'],
                                   title='Confusion matrix, without normalization')
    vif = calculate_vif(X2)
    print('VIF:\n',vif)
    plt.show()

    rdict = {'Accuracy':rd['Accuracy'], 'Sensitivity':rd['Sensitivity'],
             'Precision':rd['Precision'], 'Specificity':rd['Specificity'], 'MacFadden_Rsquare':[model2.prsquared]}
    return rdict

class SM_Logit_model():
    def __init__(self):
        self.X=None
        self.y=None
        self.Xt=None
        self.yt=None
        self.model1=None
        self.fitted_model=None
        self.predicted_prob = list()

    def fit(self, X, Y):
        self.X=sm.add_constant(X)
        self.Y=Y
        self.model = dis_mod.Logit(self.Y.values, self.X)
        try:
            self.fitted_model = self.model.fit()
            return
        except np.linalg.LinAlgError:
            print('uh oh !!! some linear algebra broke ignore this set and move on')
            return -1

    def predict(self, X):
        print('X')
        X2 = sm.add_constant(X)
        print(X2)
        y_pred = self.fitted_model.predict(X2, linear=True)
        yp = list()
        for e in y_pred:
            # print('e: {}, ln(e): {}, e^(e): {}'.format(e, np.log(e), np.exp(e)))
            self.predicted_prob.append(np.exp(e))
            if e > 0:
                yp.append(1)
            else:
                yp.append(0)
        return yp

    def score(self, X, Y, metric='Accuracy'):
        yp = self.predict(X)
        cm = confusion_matrix(Y, yp)
        rd = process_cm(cm)
        return rd[metric]

    def get_Macfadden(self):
        return self.fitted_model.prsquared




def clustering_performance(clstr_clf, X, y, X2, y2, verbose=False, comp_kn=None):
    if comp_kn is None:
        clstr_clf.fit(X,y)
        yp = clstr_clf.predict(X)
    else:
        yp = clstr_clf.fit_predict(X)
    hmo1 = metrics.homogeneity_score(y, yp)
    acc1 = metrics.accuracy_score(y, yp)
    sens1 = metrics.recall_score(y, yp)
    spec1 = metrics.precision_score(y, yp)
    if verbose:
        print('-----------------------------------')
        print('-----------------------------------')
        print('----------- Training Set -----------')
        print('Homogeniety: {:.3f}'.format(hmo1))
        print('Accuracy: {:.3f}'.format(acc1))
        print('Recall: {:.3f}'.format(sens1))
        print('Precision: {:.3f}'.format(spec1))
    yp2 = clstr_clf.predict(X2)
    hmo2 = metrics.homogeneity_score(y2, yp2)
    acc2 = metrics.accuracy_score(y2, yp2)
    sens2 = metrics.recall_score(y2, yp2)
    spec2 = metrics.precision_score(y2, yp2)
    if verbose:
        print('-----------------------------------')
        print('----------- Testing Set -----------')
        print('Homogeniety: {:.3f}'.format(hmo2))
        print('Accuracy: {:.3f}'.format(acc2))
        print('Recall: {:.3f}'.format(sens2))
        print('Precision: {:.3f}'.format(spec2))
        print('-----------------------------------')
        print('-----------------------------------')

    train_res = {'Homogeniety':hmo1, 'Accuracy':acc1, 'Recall':sens1, 'Precision':spec1}
    test_res = {'Homogeniety': hmo2, 'Accuracy': acc2, 'Recall': sens2, 'Precision': spec2}
    return train_res, test_res


def process_cm(cm, verbose=False):
    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    sensitivity = cm[1][1] / (cm[1][0] + cm[1][1])
    overall_acc = (cm[1][1] + cm[0][0]) / (cm[1][0] + cm[1][1] + cm[0][0] + cm[0][1])
    precision = (cm[0][0] / (cm[0][0] + cm[1][0]))
    print('Accuracy: {:.3f}'.format(overall_acc))
    print('Recall: {:.3f}'.format(sensitivity))
    print('Specificity: {:.3f}'.format(specificity))
    print('Precision: {:.3f}'.format(precision))
    title = 'Accuracy: {:.3f}\nrecall: {:.3f}\nprecision: {:.3f}\nspecificity: {:.3f}'.format(overall_acc,
                                                                                              sensitivity,
                                                                                              precision,
                                                                                              specificity)
    return dict({'Accuracy': overall_acc, 'Sensitivity': sensitivity,'Precision': precision, 'Specificity': specificity, 'CM': cm})




# ######################################################################################
# ######################################################################################
# ##################       TODO: Timing tools               ############################
# ######################################################################################
# ######################################################################################

def method_timer(method, param_dict):
    st = time.time()
    method(param_dict)
    exe_time = time.time() - st
    return exe_time


