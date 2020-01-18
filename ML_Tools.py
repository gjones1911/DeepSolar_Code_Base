import numpy as np
import pandas as pd
from math import *
import sys
import matplotlib.pyplot as plt
from _products.visualization_tools import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from _products.utility_fnc import *
from sklearn import metrics
from _products.performance_metrics import MSE as gmse

viz = Visualizer()

# =========================================================================
# =========================================================================
#                  TODO:Feature Selection  and Preprocessing tools
# =========================================================================
# =========================================================================

def LCN_transform(df_base, target='Adoption', mtc=0, mxcc=1, corrs=('kendall', 'pearson'), inplace=False, verbose=False):
    """ This will peform a LCN search and either return a reduced data frame
        or reduce the given
    :param df: data frame
    :param target: the target('s) of the analysis
    :param mtc:    minimum target correlation
    :param mxcc:   maximum cross correlation between independent variables
    :param corrs:  types of correlation matrices to create
    :param inplace: if true will modify original, otherwise returns a reduced version
    :return:
    """
    corr_dfk = df_base.corr(method=corrs[0])
    corr_df = df_base.corr(method=corrs[1])
    corr_df[target] = corr_dfk[target].values.flatten()
    corr_df.loc[target, :] = corr_dfk.loc[target,:]

    dfattribs = list(df_base.columns.values.tolist()).copy()
    del dfattribs[dfattribs.index(target)]
    print(dfattribs)

    lcn_d = LCN(corr_df, target=target)

    rl = HCTLCCL(lcn_d, [], target=target, options=dfattribs, target_corr_lim=mtc,
                 cross_corr_lim=mxcc)

    if verbose:
        print()
        print('return features:')
        print(rl)
        print()

    return df_base.loc[:, rl], df_base.loc[:, [target]]



# sorts a correlation and
def LCN(corr_M, threshold=100, target='Adoption'):
    """Takes a correlation matrix some threshold value and the name of the target column.
        This method will use the given correlation matrix to create a dictionary for all features
        in the matrix where the keys are the features and the values are a dictionary of all other
         features as keys and vals are the key features correlations to those variables....
            dict = {'feat1': {'feat2: correl_feat1_feat2}}
        result is a dictionary keyed on the features, with values of a sorted dictionary keyed on other
         features sorted on correlation
    :param corr_M: correlation matrix
    :param threshold: TODO: I don't remeber exactly what this does
    :param target:
    :return:
    """
    # go through Data frame of correlations grabbing the list and sorting from lowest to
    # grab the attribs
    attribs = corr_M.columns.values.tolist()
    lv1_d = {}
    for ata in attribs:
        lv1_d[ata] = dict()
        for atb in attribs:
            if ata != atb and atb != target:
                if corr_M.loc[ata, atb] < threshold:
                    lv1_d[ata][atb] = abs(corr_M.loc[ata, atb])
        if ata == target:
            lv1_d[ata] = sort_dict(lv1_d[ata], reverse=True)
        else:
            lv1_d[ata] = sort_dict(lv1_d[ata], reverse=False)

    return lv1_d
def HCTLCCL(corr_dic, start_vars, target, options, target_corr_lim = .09, cross_corr_lim=.55):
    rl = list(start_vars)
    for p_var in  corr_dic[target]:
        if corr_dic[target][p_var] > target_corr_lim and p_var not in rl and p_var in options:
            rl = check_correlation(p_var, corr_dic, rl, '', cross_corr_lim)
        if corr_dic[target][p_var] < target_corr_lim:
            return rl
    return rl
def check_correlation(check_var, corr_dic, cl, used, cross_corr_lim = .55):
    # go through current list
    for variable in cl:
        # checking cross correlation between the possible
        # variable to be added and the current one from
        # the current list if it surpasses the threshold
        # return the current list as is
        if corr_dic[check_var][variable] > cross_corr_lim:
            return cl
    # if correlations with all current variables
    # are within limits return the current list
    # updated with the new value
    return cl + [check_var]
def forward_selector_test(x, y, x2, y2):
    pass
# =========================================================================
# =========================================================================
#                  TODO: Statistics and Preprocessing
# =========================================================================
# =========================================================================
class NORML():
    """a data normalizer. has two normalization methods
        1) min max normalization:
            * equation : (X-Min_val)/(Max_val - Min_val)
            * rescales the data to [0,1] values
            * centers pdf on the mean
        2) z standardization (X-Min_val)/(Max_val - Min_val)
            * equation : (X-Mean)/(Standard_Deviation)
            * rescales the data to [-1,1] values
            * centers pdf on 0 with std = 1
        Select the type by setting the input argument
        nrmlz_type to:
            * : minmax for option 1
            * : zstd for option 2
    """
    def __init__(self, nrmlz_type='minmax'):
        self.mu=None
        self.std=None
        self.cov=None
        self.corr=None
        self.cov_inv=None
        self.cov_det=None
        self.min = None
        self.max = None
        self.normlz_type=nrmlz_type

    def set_type(self, n_type):
        self.normlz_type = n_type

    def fit(self, df=None, X=None):
        if type(df) != type(np.array([0])):
            df = pd.DataFrame(df)
        self.process_df(df)

    def process_df(self, df):
        self.mu = df.values.mean(axis=0)
        self.std = df.values.std(axis=0)
        self.min = df.min()
        self.max = df.max()
        self.cov = df.cov()
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

    def transform(self,df, headers=None):
        if type(df) !=type(np.array([0])):
            df = pd.DataFrame(df)
        if self.normlz_type == 'zstd':
            if headers is not None:
                return pd.DataFrame((df - self.mu) / self.std, columns=headers)
            return pd.DataFrame((df - self.mu) / self.std)
        elif self.normlz_type == 'minmax':
            if headers is not None:
                return pd.DataFrame((df - self.min) / (self.max - self.min), columns=headers)
            return pd.DataFrame((df - self.min) / (self.max - self.min))

def standardize_data(X, X2, scaler_ty='minmax'):
    #scaler_ty = 'std'
    if scaler_ty == 'minmax':
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(X)
        Xtrn = mm_scaler.transform(X)
        Xtsn = mm_scaler.transform(X2)
    elif scaler_ty == 'std':
        std_scaler = StandardScaler()
        std_scaler.fit(X)
        Xtrn = std_scaler.transform(X)
        Xtsn = std_scaler.transform(X2)
    return Xtrn, Xtsn

def cross_val_splitter(X, y, tr=.5, ts=.5, vl=0, seed=None, verbose=False, target=None):
    train_idx, val_idx, test_idx = split_data(X, y, p_train=tr, p_test=vl, p_val=ts, verbose=verbose,seed=seed)

def split_data(X, y, p_train=.70, p_test=.30, p_val=.0, priors = None, verbose=False, seed=False, lr=True):
    """Returns a randomized set of indices into an array for the purposes of splitting data"""
    dXY = None
    if type(X) != type(pd.DataFrame([0])):
        nx = pd.DataFrame(X)
        nx[X.shape[1]] = y.values.tolist()
        dXY = nx.values
        np.random.shuffle(dXY)
    N = len(X)

    train = int(np.around(N * p_train, 0))
    if p_val == 0:
        test = int(np.around(N * p_test, 0, ))
        val = 0
    else:
        test = N - train
        val = N - train - test

    tr = dXY[0:train]
    ts = dXY[train:train+test]
    if p_val != 0:
        vl = dXY[train+test:]

    tr_X, tr_y = tr[:][0:len(dXY[0])], tr[:][len(dXY[0])]
    ts_X, ts_y = ts[:][0:len(dXY[0])], ts[:][len(dXY[0])]
    vl_X, vl_y = list(), list()
    if p_val != 0:
        vl_X, vl_y = tr[:][0:len(dXY[0])], tr[:][len(dXY[0])]
    '''
    if priors is not None:
        print('need to set up the distribution of the weights')

    if verbose:
        print('train set size: ', train)
        print('test set size: ', test)
        print('val set size: ', val)
    np.random.shuffle(X)
    tr_idx = rc

    for i in range(0, train):
        trn_idx.append(r_c[i])

    for i in range(train, train+test):
        tst_idx.append(r_c[i])

    for i in range(train+test, data_size):
        val_idx.append(r_c[i])

    if val == 0:
        return trn_idx, tst_idx
    else:
        return trn_idx, tst_idx, val_idx
    '''
    if p_val != 0:
        return (tr_X, tr_y), (ts_X, ts_y), (vl_X, vl_y)
    return (tr_X, tr_y), (ts_X, ts_y), (vl_X, vl_y)

def gstandardize_data(X, X2, scaler_ty='minmax'):
    if scaler_ty == 'minmax':
        nrml = NORML()
        nrml.fit(X)
        Xr = nrml.transform(X)
        xrts = nrml.transform(X2)
    if scaler_ty == 'zstd':
        nrml = NORML(scaler_ty=scaler_ty)
        nrml.fit(X)
        Xr = nrml.transform(X)
        xrts = nrml.transform(X2)
    return Xr, xrts

def normalize(X, mu, std, min, max, type='z', copy=False):
    if type == 'z':
        return z_normalize(X, mu, std)
    elif type == 'mm':
        return min_max_normalize(X, min, max)

def z_normalize(X, mu, std):
    return pd.DataFrame((X - mu)/std, columns=X.columns)

def min_max_normalize(X, min, max):
    return pd.DataFrame((X - min)/(max - min), columns=X.columns)


# =========================================================================
# =========================================================================
#                            TODO: Modeling tools
# =========================================================================
# =========================================================================
#Classification Model
class CMODEL():
    """a representation of a model for machine learning can in take in multiple data sets and perform
       a column wise merge
    """
    def __init__(self, file_list, exclude_list, target, df=None, usecols=None, usecol_list=None, verbose=False, lcn=False,
                 labeled=True, joins=('fips', 'fips', 'fips'), impute='drop', nas=(-999, ), drop_joins=False,st_vars=[],
                 mtc=.0, mxcc=1, dim_red=None, split_type='tt', tr_ts_vl = (.6, .4, 0), normal =None, complx=False):
        self.normlz = normal
        self.target = target                # the current models classification objective
        self.classes = list()               # the class values for this model
        self.model_mean = None              # the attribute mean values
        self.model_std = None               # the attribute std values
        self.model_cov = None
        self.model_cov_det = None
        self.model_cov_inv = None
        self.class_splits = dict()          # the data set split by class
        self.class_counts = dict()          # a count for each class in the data set
        self.class_priors = dict()          # the prior probaility of each class initialized by data set
        self.class_means = dict()           # the attribute means for each class
        self.class_std = dict()             # the attribute std for each class
        self.class_cov = dict()             # the covariance matrix for each class
        self.class_cor = dict()
        self.class_cov_inv = dict()         # the covariance matrix inverse for each class
        self.class_cov_det = dict()         # the class covariance matrix determinant
        self.attribs = None                 # the names of the attributes by column
        self.excluded=None                  # the excluded variables, can be added back as onehot encoded versions
        self.data_set=None                  # holds the desired data set
        self.og_dataset= None               # the merged set before any preprossing
        self.data_corr = None
        self.X = None                       # data or independent variables
        self.y = None                       # the target values or dependent variable
        self.Xtr_n=None
        self.Xts_n=None
        self.corr = None                    # the correlation matrix for the data
        self.Xfld = None                    # an fld transformed version of the data
        self.Xpca = None                    # a pca transformed version of the data
        self.dim_red = DimensionReducer()   # the models dimension reducer
        self.Xts=None
        self.yts=None
        self.complx=complx
        self.process_files(file_list, exclude_list, target, usecols, usecol_list, verbose, labeled, joins,
                           impute, nas, drop_joins=drop_joins, lcn=lcn, mtc=mtc, mxcc=mxcc, tr_ts_vl=tr_ts_vl,
                           df=df, st_vars=st_vars)

    def process_files(self, file_list, exclude_list, target, usecols, usecol_list, verbose, labeled,
                      joins=('fips', 'fips', 'fips'), impute='drop', nas=(-999,), drop_joins=False,
                      lcn=False,  mtc=.1, mxcc=.4, tr_ts_vl=(.6, .4, 0), df=None, to_encode=None, drops=None, st_vars=[]):
        df_list = list([])
        # go through and create and clean up data frames
        # dropping those in the exclude list
        doit=True
        if drops is None:
            tormv = list()
        else:
            tormv = drops
        #  If there was a data frame passed
        if df is not None:
            self.og_dataset = df
            self.excluded = tormv
            if to_encode is not None:
                hold_over = df.low[:, to_encode]
        else:
            # loop to set up input do data merger
            for df, ex in zip(file_list, exclude_list):
                print('Data file:', df)
                print('adding to be excluding', ex)
                df_list.append(pd.read_excel(df))
                tormv += ex
                #if doit and len(ex) > 0:
            self.excluded = tormv
            self.og_dataset = data_merger(df_list, joins=joins, verbose=verbose, drop_joins=True, target=target)

        #print(self.og_dataset)
        merged = self.og_dataset.drop(columns=tormv, inplace=False)
        self.data_corr = merged.sort_values(by=target, axis='index', ascending=False).corr(method='kendall')
        if usecols is not None:
            merged = merged.loc[:, usecols]
            self.data_corr = merged.corr(method='kendall')
        if lcn:
            self.data_corr = merged.corr(method='kendall')
            lcn_d = LCN(self.data_corr, target=target)
            rl = HCTLCCL(lcn_d, st_vars, target=target, options=merged.columns.values.tolist(), target_corr_lim=mtc,
                         cross_corr_lim=mxcc)
            merged = merged.loc[:, rl + [target]]
            print('columns used:')
            print(merged.columns)
        if impute == 'drop':
            for n in nas:
                merged.replace(n, np.nan)  # this value is used by the SVI data set to represent missing data
            merged = merged.dropna()
        self.data_set = merged
        self.attribs = merged.columns.values.tolist()
        print(self.attribs)
        del self.attribs[self.attribs.index(self.target)]
        self.X = pd.DataFrame(merged.loc[:, self.attribs], columns=self.attribs)
        self.y = pd.DataFrame(merged.loc[:, self.target], columns=[self.target])
        # TODO: NOW SPLIT THE DATA INTO DESIRED NUMBER OF FOLDS
        targets0 = self.y[target]
        ts = tr_ts_vl[1] + tr_ts_vl[2]
        print('ts size', ts)
        tr = 1 - ts
        # Create training and testing sets for the data
        X_train0, X_test0, y_train0, y_test0 = train_test_split(self.X, targets0, stratify=targets0, test_size=ts,
                                                                train_size=tr, )
        self.train_counts = y_train0.value_counts(normalize=True)
        self.test_counts = y_test0.value_counts(normalize=True)
        self.X, self.y = pd.DataFrame(X_train0, columns=self.attribs), pd.DataFrame(y_train0, columns=[self.target])
        self.Xts, self.yts = pd.DataFrame(X_test0, columns=self.attribs), pd.DataFrame(y_test0, columns=[self.target])
        self.N = self.X.shape[0]
        self.Nts = self.Xts.shape[0]
        self.d = self.X.shape[1]
        self.corr = self.X.corr()

        if self.normlz is not None:
            print('Normalize', self.normlz)
            self.X, self.Xts = standardize_data(self.X, self.Xts, scaler_ty=self.normlz)
            self.X = pd.DataFrame(self.X, columns=self.attribs)
            self.Xts = pd.DataFrame(self.Xts, columns=self.attribs)
            self.y.index = self.X.index
            self.yts.index = self.Xts.index
            print('y len', len(self.y.values))
            print('X len', len(self.X.values))
        self.grab_model_stats()
        # TODO: need to a some time move LCN stuff here
        # grab class specific stats
        self.calculate_class_stats()
        self.model_data = list((self.X, self.y))                    # store data and labels together in lit



    def grab_model_stats(self):
        print(self.X.values)
        print(self.X)
        self.model_mean = self.X.values.mean(axis=0)
        self.model_std = self.X.values.std(axis=0).mean()
        self.model_cov  = self.X.cov()
        if self.complx:
            self.model_cov_inv = np.linalg.inv(self.model_cov)
            self.model_cov_det = np.linalg.det(self.model_cov)
    def calculate_class_stats(self):
        self.classes = list(set(self.y[self.target]))
        print('classes', self.classes)
        for c in self.classes:
            self.splits_priors(c)
            self.class_means_std(c)
            if self.complx:
                self.class_cov_inv_det(c)
                self.class_cor[c] = self.class_splits[c].corr()
    def splits_priors(self,c):
        self.class_splits[c] = self.X.loc[self.y[self.target] == c, :]
        self.class_counts[c] = self.class_splits[c].shape[0]
        self.class_priors[c] = self.class_splits[c].shape[0] / self.N
    def class_means_std(self,c):
        self.class_means[c] = self.class_splits[c].values.mean(axis=0)
        self.class_std[c] = self.class_splits[c].values.std(axis=0)
    def class_cov_inv_det(self, c):
        self.class_cov[c] = self.class_splits[c].cov()
        if self.complx:
            self.class_cov_inv[c] = np.linalg.inv(self.class_cov[c].values)
            self.class_cov_det[c] = np.linalg.det(self.class_cov[c].values)
    def show_data_report(self):
        print('=======================================================================================')
        print('Train data size:', self.X.shape[0])
        print('y_train class distribution 0')
        print(self.train_counts)
        print('Test data size:', self.Xts.shape[0])
        print('y_test class distribution 0')
        print(self.test_counts)
        print('=======================================================================================')
        print('Features in Model:')
        print(self.X.columns.values)
        print('Predicting for {}'.format(self.target))
    def Reduce_Dimension(self, dr_type='FLD', pov=None, pc=None):
        if dr_type == 'FLD':
            self.perform_FLD()
    def perform_FLD(self):
        self.dim_red.fld_fit(self)
        self.Xfld = self.dim_red.FLD(self.X)
        self.tsXfld = self.dim_red.FLD(self.X)

class RMODEL():
    def __init__(self, X=None, Y=None, columns=None, impute=None, verbose=False, trtsspl=(.7, ),
                 n_type=None):
        self.X = np.array(X)
        self.Y = np.array(Y)
        if columns is None:
            self.dataframe = pd.DataFrame(self.X)
        else:
            self.dataframe = pd.DataFrame(self.X, columns=columns)
        self.dataframe['target'] = Y
        self.Xtr = None
        self.ytr = None
        self.Xts = None
        self.yts = None
        self.train_test_split = trtsspl
        self.cross_val_split()
        self.columns = columns
        self.impute=impute
        self.verbose=verbose
        self.normalizer = NORML()
        self.n_type=n_type
        if n_type is not None:
            self.normalize()

    def cross_val_split(self):
        if self.train_test_split[0] == 0:
            self.Xtr = self.X
            self.Xts = self.X
            self.ytr = self.Y
            self.yts = self.Y
            return
        else:
            np.random.shuffle(np.array(self.X))


    def normalize(self, n_type='minmax'):
        self.normalizer.set_type(n_type=n_type)
        self.normalizer.fit(self.Xts)
        self.Xtr = self.normalizer.transform(self.X)
        self.Xts = self.normalizer.transform(self.Xts)




# =========================================================================
# =========================================================================
#                   TODO: Dimension Reduction tools
# =========================================================================
# =========================================================================

class DimensionReducer():
    def __init__(self):
        self.type=None
        self.class_splits=None
        self.classes=None
        self.class_means=None
        self.data_means=None
        self.data_std = None
        self.class_cov=None
        self.class_cov_inv=None
        self.class_cov_det=None
        self.class_priors=None
        self.class_counts=None
        self.eig_vec = None
        self.eig_vals = None
        self.pval = None
        self.W = None
        self.WT = None
        self.WT2 = None
        self.k=None
        self.k_90 = None
        self.s = None
        self.vh = None
        self.i_l = list()
        self.dr_type = None
        self.y2 = None
        self.x2 = None
        self.x1 = None
        self.y1 = None
        self.N = None
        self.z, self.one = list(), list()

    def FLDA(self, df, dftr, y, classes=(0,1), class_label='type'):
        y = pd.DataFrame(y, columns=[class_label])
        c1 = dftr.loc[y[class_label] == classes[0], :]
        n1 = len(c1)
        c2 = dftr.loc[y[class_label] == classes[1], :]
        n2 = len(c2)
        #print('There are {0} negative and {1} positive samples'.format(n1, n2))
        Sw_inv = np.linalg.inv((n1-1)*c1.cov() + (n2-1)*c2.cov())
        #print(Sw_inv.shape)
        #print(c1.mean().shape)
        #w = np.dot(Sw_inv, np.dot((c1.mean() - c2.mean()), (c1.mean()-c2.mean()).transpose()))
        w = np.dot(Sw_inv, (c1.values.mean(axis=0) - c2.values.mean(axis=0)))
        #print('w', w.shape)
        #print('df', df.shape)
        return np.dot(df,w)
    def fld_fit(self, cmodel):
        self.dr_type = 'fld'
        self.attribs = cmodel.attribs
        self.class_splits = cmodel.class_splits
        self.class_counts = cmodel.class_counts
        self.classes = cmodel.classes
        self.class_means = cmodel.class_means
        self.data_means = cmodel.model_mean
        self.data_std = cmodel.model_std
        self.class_cov = cmodel.class_cov
        self.class_cov_inv = cmodel.class_cov_inv
        self.class_cov_det = cmodel.class_cov_det
        self.class_priors = cmodel.class_priors
        self.N=None
        self.kmm=None
        self.Calculate_W_FLD()
    def pca_fit(self, X):
        self.eig_vals, self.eig_vec = np.linalg.eig(X.cov())
        #print('eigvec', self.eig_vec)
        self.N = len(X)
        print('eigvals', self.eig_vals)
    def svd_w_np(self, X):
        u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
        self.s = s
        self.vh = vh
        self.N = len(X)
        self.d = X.shape[1]
        return
    def svd_pov(self, s, accuracy=.90, verbose=False, pov_plot=False, show_now=False):
        sum_s = sum(s.tolist())
        ss = s**2
        sum_ss = sum(ss.tolist())
        self.prop_list = list()
        found = False
        k = 0
        x1, y1, x2, y2, = 0, 0, 0, 0
        p_l, i_l = 0, 0
        found = False
        self.prop_list.append(0)
        self.i_l.append(0)
        for i in range(1, len(ss)+1):
            perct = sum(ss[0:i]) / sum_ss
            # perct = sum(s[0:i]) / sum_s
            if np.around(perct, 2) >= accuracy and not found:
                self.x1 = i
                self.y1 = perct
                found = True
            self.prop_list.append(perct)
            self.i_l.append(i)
        self.single_vals = np.arange(1, self.N + 1)
        if pov_plot:
            plt.figure()
            plt.plot(self.i_l, self.prop_list)
            plt.scatter(self.x1, self.y1, c='r', marker='o', label='Point at {:.1f}% accuracy'.format(self.y1*100))
            plt.title('Proportion of Variance vs. Number of Eigen Values\n{:d} required for {:.2f}'.format(self.x1, self.y1*100))
            plt.legend()
            plt.xlabel('Number of Eigen values')
            plt.ylabel('Proportion of Variance')
            if show_now:
                plt.show()
        return self.x1 + 1
    def svd_fit(self,X, vh=None, k=None, get_pov=True, pov_thresh=.90, verbose=False, plot=False, usek=False,
                gen_plot=True, y=None):
        if vh is None:
            self.svd_w_np(X)
            u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
            if get_pov:
                print('getting pov')
                self.kmm =  self.svd_pov(s, accuracy=pov_thresh, verbose=verbose, pov_plot=plot, show_now=True)
            if usek:
                k = self.kmm
        self.N = len(X)
        self.data_means = X.mean(axis=0).values.flatten()
        #print('data means', self.data_means)
        print('vector shape', vh.shape)
        #vt = np.transpose(self.vh)
        vt = np.transpose(self.vh)
        # grab the first k principle components
        if k is not None:
            self.W = vt[:, 0:k]
            self.k = k
        else:
            self.W = vt[:, :]
            self.k = len(X)
        self.WT = np.transpose(self.W)

        # grab the first two principle components
        W2 = vt[:, 0:2]
        W3 = vt[:, 0:3]
        self.WT2 = np.transpose(W2)
        self.WT3 = np.transpose(W3)
        if gen_plot:
            # get 0's and 1's
            for row, adp in zip(self.WT2, y):
                if adp == 0:
                    self.z.append(row)
                else:
                    self.one.append(row)
    def svd_transform(self, X, treD=False):
        z_array = list()
        z2_array = list()
        z3_array = list()
        for row in X.values:
            #print('row',row)
            #print('data means')
            #print(self.data_means)
            c_x = row - self.data_means
            z_array.append(np.dot(self.WT, c_x))
            z2_array.append(np.dot(self.WT2, c_x))
            if treD:
                z3_array.append(np.dot(self.WT3, c_x))
        Z = np.array(z_array, dtype=np.float)
        Z2 = np.array(z2_array, dtype=np.float)
        if treD:
            Z3 = np.array(z3_array, dtype=np.float)
            return Z, Z2, Z3
        return Z, Z2
    def Calculate_W_FLD(self, ):
        c1 = self.class_splits[0]
        n1 = self.class_counts[0]
        c2 = self.class_splits[1]
        n2 = self.class_counts[1]
        # TODO: evalute below to see if needed or not
        if False and type(c1) != type(pd.DataFrame({0:0})):
            c1 = pd.DataFrame(c1).values
            c2 = pd.DataFrame(c2).values
        Sw_inv = np.linalg.inv((n1 - 1) * c1.cov() + (n2 - 1) * c2.cov())
        self.W_fld = np.dot(Sw_inv, (c1.values.mean(axis=0) - c2.values.mean(axis=0)))

    def pca_transform(self, X, p=None):
        if p is None:
            return self.convert_basis(X, self.eig_vec)
        else:
            return self.convert_basis(X, self.eig_vec[0:p])

    def FLD(self, X):
        return pd.DataFrame(np.dot(X,self.W_fld))

    def PCA(self, df, pov, eig_vec=None, m=None, verbose=False, ret_level=0,
            pov_plot=False, show_now=False):
        # if not given eigen values calculate them based on the
        # desired proportion of variance (pov) covered
        if eig_vec is None:
            #print(df.cov())
            eig_vals, eig_vec = np.linalg.eig(df.cov())
            if m is None:
                pov_l, m = self.calculate_p(eig_vals=eig_vals, pov=pov, verbose=verbose, show_now=show_now,
                                            pov_plot=pov_plot)
                print('The Number of eigenvectors to cover {:.2f} of the variance is {:d}'.format(100*pov, m))
            else:
                pov_l, cm = self.calculate_p(eig_vals=eig_vals, pov=pov, verbose=verbose, show_now=show_now,
                                             pov_plot=pov_plot)
                print('The number of eigenvectors is set to {:d}'.format(m))
            #print(eig_vec)
            eig_vec = eig_vec[0:m]
            # now perform transform
            y = self.convert_basis(df, pd.DataFrame(eig_vec))
            if ret_level == 0:
                return y
            elif ret_level == 2:
                return pov_l, m, eig_vec, pd.DataFrame(y)
            elif ret_level == 1:
                return eig_vec, y

    def calculate_p(self, pov, eig_vals=None,  verbose=False, pov_plot=False, show_now=False):
        #print('calculating k')
        # calculate total sum
        if eig_vals is None:
            eig_vals = self.eig_vals
        s_m = sum(eig_vals)
        if verbose:
            print('The sum of the eigen values is {0}'.format(s_m))
            print('The length of the eigen values vector is {0}'.format(len(eig_vals)))
            print('here it is')
            print(eig_vals)
        # now go through to find your required k for the
        # desired Proportion of Variance (pov)
        pov_l, cpov, csum, k, found = [], 0, 0, 0, False
        pov_found = 0
        for v in range(len(eig_vals)):
            csum += np.around(eig_vals[v], 2)
            pov_l.append(np.around(csum/s_m, 2))
            if verbose:
                print('The sum at {:d} is {:.2f}, pov {:.2f}'.format(v, csum, pov_l[v]))
            if pov_l[-1] >= pov and not found:
                k = v+1
                pov_found = pov_l[-1]
                print(k, pov_found)
                found = True
        if pov_plot:
            plt.figure()
            plt.plot(list(range(1, len(eig_vals)+1)), pov_l)
            plt.scatter(k, pov_found, c='r', marker='o', label='Point at {:.1f}% accuracy'.format(pov_found*100))
            plt.title('Proportion of Variance vs. Number of Eigen Values')
            plt.legend()
            plt.xlabel('Number of Eigen values')
            plt.ylabel('Proportion of Variance')
            if show_now:
                plt.show()

        self.pov_l = pov_l
        self.k = k
        return pov_l, k

    def convert_basis(self, df, new_basis):
        return np.dot(df, new_basis.transpose())

    def PCA_Eig(self, X, class_means):
        return 1

    def PCA_SVD(self, X):
        return 1

# =========================================================================
# =========================================================================
#                   TODO: Learners
# =========================================================================
# =========================================================================

def epsilon(emax, emin, k, kmax):
    return emax * ((emin/emax)**(min(k, kmax)/kmax))

class Learner(ABC):
    """Template class for a learning machine"""
    def __init__(self,):
        pass
    def finish_init(self):
        pass
    def fit(self, cmodel):
        pass
    def predict(self, X):
        pass
    def score(self, X, Y):
        pass

class bayes_classifiers(Learner):
    def __init__(self, cmodel=None, df_list=None, case=1):
        super().__init__()
        self.fit(cmodel=cmodel)
        self.case = case
        if self.case == 1:
            self.func = 'euclid'
        elif self.case == 2:
            self.func = 'mahala'
        if self.case == 1:
            self.func = 'euclid'
        elif self.case == 3:
            self.func = 'quadratic'
    def fit(self, cmodel):
        self.cmodel = cmodel

    def bayes_classifier_model_finder(self, dfx, dfy, case=1, func='euclid', scale= .1, verbose=False,
                                      priors=()):
        #n, p = list(), list()
        #f, b = .10, .90

        #while f <= b:
        #    n.append(b)
        #    p.append(f)
        #    f += scale
        #    b -= scale
        #back = n[0:]
        #rback = n[0:]
        #rback.reverse()
        #ford = p[0:-1]
        #rford = ford[0:]
        #rford.reverse()
        #pos = ford + rback
        #neg = back + rford

        if len(priors) == 0:
            pos, neg = self.generate_priors(scale=scale)
        else:
            if scale is not None:
                pos, neg = self.generate_priors(scale=scale, prior1=priors[0], prior2=priors[1])
            else:
                pos, neg = list([priors[0]]), list([priors[1]])
        #print(pos)
        #print(neg)
        best_acc, best_scr, best_posnegs, scr = 0,  None, None, 0
        pr_l1, pr_l2, accuracy , sens, spec= list(), list(), list(), list(), list()
        best_pr = [0, 0]
        for ps, ng in zip(pos, neg):
            pr = [ps, ng]
            #print(pr[0] + pr[1])
            acc, scr, posnegs = self.bayes_classifier_predict_and_score(dfx, dfy, case=case, priors=pr, func=func, verbose=verbose)
            accuracy.append(acc)
            pr_l1.append(pr[0])
            pr_l2.append(pr[1])
            sens.append(posnegs['sen'])
            spec.append(posnegs['spe'])
            if acc > best_acc:
                best_acc = acc
                best_scr = scr
                best_posnegs = posnegs
                best_pr[0] = pr[0]
                best_pr[1] = pr[1]

        result_dic = {'best_accuracy':best_acc,
                      'best_scores':best_scr,
                      'best_postnegs':best_posnegs,
                      'pr_l1':pr_l1,
                      'pr_l2':pr_l2,
                      'sens':sens,
                      'spec':spec,
                      'accuracy_list':accuracy,
                      'best_priors': best_pr}

        #return best_acc, best_scr, best_posnegs, pr_l1, pr_l2, sens, spec, accuracy, best_pr
        return result_dic

    def bayes_classifier_model_finderB(self, dfx, dfy, case=1, func='euclid', scale= .1, verbose=False,
                                      priors=()):
        #n, p = list(), list()
        #f, b = .10, .90

        #while f <= b:
        #    n.append(b)
        #    p.append(f)
        #    f += scale
        #    b -= scale
        #back = n[0:]
        #rback = n[0:]
        #rback.reverse()
        #ford = p[0:-1]
        #rford = ford[0:]
        #rford.reverse()
        #pos = ford + rback
        #neg = back + rford

        if len(priors) == 0:
            pos, neg = self.generate_priors(scale=scale)
        else:
            if scale is not None:
                pos, neg = self.generate_priors(scale=scale, prior1=priors[0], prior2=priors[1])
            else:
                pos, neg = list([priors[0]]), list([priors[1]])
        print(pos)
        print(neg)
        best_acc, best_scr, best_posnegs, scr = 0,  None, None, 0
        pr_l1, pr_l2, accuracy , sens, spec= list(), list(), list(), list(), list()
        best_pr = [0, 0]
        for ps, ng in zip(pos, neg):
            pr = [ps, ng]
            #print(pr[0] + pr[1])
            acc, scr, posnegs = self.bayes_classifier_predict_and_scoreB(dfx, dfy, case=case, priors=pr, func=func, verbose=verbose)
            accuracy.append(acc)
            pr_l1.append(pr[0])
            pr_l2.append(pr[1])
            sens.append(posnegs['sen'])
            spec.append(posnegs['spe'])
            if acc > best_acc:
                best_acc = acc
                best_scr = scr
                best_posnegs = posnegs
                best_pr[0] = pr[0]
                best_pr[1] = pr[1]

        result_dic = {'best_accuracy': best_acc,
                      'best_scores': best_scr,
                      'best_postnegs': best_posnegs,
                      'pr_l1': pr_l1,
                      'pr_l2': pr_l2,
                      'sens': sens,
                      'spec': spec,
                      'accuracy_list': accuracy,
                      'best_priors': best_pr}

        # return best_acc, best_scr, best_posnegs, pr_l1, pr_l2, sens, spec, accuracy, best_pr
        return result_dic

    def bayes_classifier_predict(self, dfx, case=1, verbose=False, priors = [1,1], func='euclid'):
        ypred = list()
        case = self.case
        func = self.func
        # figure out the priors situation
        if priors is None:
            prior1 = self.cmodel.class_priors[0]
            prior2 = self.cmodel.class_priors[1]
        else:
            prior1 = priors[0]
            prior2 = priors[1]
        #print('====================================>   Priors: ', prior1, prior2)

        # figure out what discriminant function to use
        if case == 1:
            func = 'euclid'
            #cov = self.gauss_params['std'] ** 2
            cov = self.cmodel.model_std  ** 2
            print('cov', cov)
        elif case == 2:
            func = 'mahala'
            cov = self.cmodel.model_cov
        elif case == 3:
            func = 'quadratic'
            cov = [self.cmodel.class_cov[0], self.cmodel.class_cov[1]]

        #mu1 = self.gauss_params['mu_c1']
        #mu1 = self.gauss_params['mu_c1']
        #mu1 = self.Cmu_array[0]
        #mu2 = self.Cmu_array[1]
        mu1 = self.cmodel.class_means[0]
        mu2 = self.cmodel.class_means[1]
        #print(func)
        # make some predictions
        for xi in dfx.values:
            if case != 3:
                pc1 = self.discriminate_function(xi, mu1, cov, prior1, func=func)
                pc2 = self.discriminate_function(xi, mu2, cov, prior2, func=func)
            else:
                #print(func)
                pc1 = self.discriminate_function(xi, mu1, cov[0], prior1, func=func)
                pc2 = self.discriminate_function(xi, mu2, cov[1], prior2, func=func)
            if verbose:
                print('pc1',pc1)
                print('pc2',pc2)
            if pc1 > pc2:
                ypred.append(0)
            else:
                ypred.append(1)
        return ypred

    def bayes_classifier_predict_and_scoreB(self, dfx, dfy, case=1, priors=[1,1], func='euclid', verbose=False):
        case = self.case
        func = self.func
        ypred = self.bayes_classifier_predict(dfx, case=case, priors=priors, func=func, verbose=verbose)
        return self.bayes_classifier_score(dfy, ypred)

    def bayes_classifier_predict_and_score(self, dfx, dfy, case=1, priors=[1,1], func='euclid', verbose=False):
        case = self.case
        func = self.func
        ypred = self.bayes_classifier_predict(dfx, case=case, priors=priors, func=func, verbose=verbose)
        return self.bayes_classifier_score(dfy, ypred)

    def bayes_classifier_score(self, yactual, ypred, vals = [0,1]):
        return bi_score(ypred, yactual, vals, classes=vals)

    def generate_priors(self, scale, prior1=None, prior2=None):
        if prior1 is not None and prior2 is not None:
            # set up prior1 side
            if prior1 < prior2:
                l1 = list([1])
                l2 = list([.0001])
                while l1[-1] - scale >= prior1:
                    l1.append(l1[-1] - scale)
                    l2.append(1 - l1[-1])
                if l1[-1] - prior1 < 0:
                    l1[-1] = prior1
                    l2[-1] = 1 - prior1
            else:
                l1 = list([.0001])
                l2 = list([1])
                while l2[-1] - scale >= prior2:
                    l2.append(l2[-1] - scale)
                    l1.append(1 - l2[-1])
                if l2[-1] - prior2 < 0:
                    l2[-1] = prior2
                    l1[-1] = 1 - prior2
            return l1, l2

        l1 = list([0.001])
        l2 = list([1])
        while np.around(l1[-1] + scale, 3) < 1:
            l1.append(np.around(l1[-1] + scale, 3))
            l2.append(np.around(1 - l1[-1], 3))
        return l1, l2

    def predict(self, X, case=1, num_cls = 2, priors=(1,1)):
        return self.bayes_classifier_predict(X, case=1, verbose=False, priors=priors)
        #if num_cls == 2:
        #    return self.generate_predictions_bi(X, case=case)

    def score(self, Ya, Yp):
        pass

    def dim_reduce(self, type='', attribs=()):
        pass

    def euclidian_disc(self, mu, x, cov, prior):
        return (-np.dot(x.transpose(), np.dot(mu, x))/cov**2) + np.log(prior)

    def mahalanobis_disc(self, mu, x, cov_inv, prior):
        return -np.dot(x.transpose(), np.dot(mu, x)) + np.log(prior)

    def quadratic_disc(self, mu, x, cov_inv, cov_det, prior):
        return -np.dot(x.transpose(), np.dot(mu, x)) + np.log(prior)

    def min_euclid(self, mean_ib, xvec, sig1, prior):
        return (np.dot(mean_ib.T, xvec) / sig1) - (np.dot(mean_ib.T, mean_ib) / (sig1 * 2)) + np.log(prior)
        #return -(np.sqrt((np.linalg.norm(xvec-mean_ib))))/(2*sig1) + np.log(prior)

    def min_mahalanobis(self,mu,x,siginv,prior):
        return np.dot(mu.T,np.dot(siginv.T, x)) - (.5 * np.dot(mu.T, np.dot(siginv, mu))) + np.log(prior)
    def quadratic_machine(self, x, mu, siginv, detsig, prior):
        return (-.5 * np.dot(x.T, np.dot(siginv, x))) + np.dot(mu.T, np.dot(siginv.T, x)) - (.5*np.dot(mu.T, np.dot(siginv, mu))) - (.5*np.log(detsig))+np.log(prior)
        #return (-.5 * np.dot(x.T, np.dot(siginv, x))) + np.dot(np.dot(siginv, mu).T, x) - (.5*np.dot(mu.T, np.dot(siginv, mu))) - (.5*np.log(detsig))+np.log(prior)

    def generate_predictions_bi(self,X, case=1):
        y = list()
        for x in X:
            # get the posterior probability of
            # and set the class as the MPP
            c1 = self.case_picker(X, case, 0)
            c2 = self.case_picker(X, case, 1)
            if c1 > c2:
                y.append(0)
            else:
                y.append(1)
        return y

    def case_picker(self, X, case, class_val):
        case = self.case
        if case == 1:
            return self.min_euclid(self.cmodel.class_means[class_val], X, self.cmodel.model_std**2,
                                   self.cmodel.class_priors[class_val])
            #return self.euclidian_disc(self.cmodel.class_means[class_val], X, self.cmodel.,
            #                           self.cmodel.class_priors[class_val])
        elif case == 2:
            return self.mahalanobis_disc(self.cmodel.class_means[class_val], X, self.cmodel.model_cov_inv,
                                       self.cmodel.class_priors[class_val])
        elif case == 3:
            return self.quadratic_disc(self.cmodel.class_means[class_val], X, self.cmodel.class_cov_inv[class_val],
                                         self.cmodel.class_cov_det[class_val], self.cmodel.class_priors[class_val])

    def discriminate_function(self, df, mu, cov, prior, func='euclid', verbose=False):
        func = self.func
        if func.lower() == 'euclid':
            #print('euclid')
            if verbose:
                print('X:\n', df)
                print('mu:\n', mu)
                print('std:', cov)
            return self.min_euclid(mu, df, cov, prior)
        elif func.lower() == 'mahala':
            return self.min_mahalanobis(mu, df, np.linalg.inv(cov), prior)
            #print('mahala')
        elif func.lower() == 'quadratic':
            #print('quad')
            return self.quadratic_machine(df, mu, np.linalg.inv(cov), np.linalg.det(cov), prior)

class clusters():
    """Represents a group of clusters"""
    def __init__(self, k, method='kmeans', init='random', df=None, distance_metric='dmin',
                 distance_calc='euclid', verbose=True, distance_type='city_block'):
        self.methods = ['kmu', 'wta', 'kohonen']
        self.init_types = ['random', 'random_sample', 'normal']
        self.k = k                                                                  # desired number of clusters
        self.df = df                                                                # the data frame I was given
        self.del_thrsh = .09
        self.dist_type = distance_type
        self.distance_metric = distance_metric                                      # what type of distance metric used
        self.distance_calc = distance_calc                                          # how to calculate the distance
        #self.data = self.df.values                                                 # the numpy array of my data
        self.size = df.shape[0]                                                     # the number of samples in the data
        self.dimen = df.shape[1]                                                    # the number of features
        self.method = method                                                        # the clustering method
        print('method:', method)
        self.top_grid = list()
        self.epochs = 0
        self.emax = 1
        self.emin = .0001
        self.kmax =100
        self.time_taken = 0
        self.std = int(np.around(df.values.std(axis=0).mean(),0))
        self.mu = int(np.around(df.values.mean(axis=0).mean(),0))
        print('std ', self.std)
        print('mu ', self.mu)
        self.init = init                                                            # the method of initializing clusters
        self.dist_LUT = None
        #if distance_metric in ['dmin', 'dmax']:
        #    print('dminmax')
        #    self.dist_LUT = self.calculate_distance_LUT(df.values)                               # generate look up table of distances
        self.my_clusters = None
        self.my_clusters = self.check_method(df.values)                             # the list of clusters initialized
        if verbose:
            print('There are {:d} clusters to start'.format(self.check_size()))

    def set_threshold(self, ):
        if self.distance_metric in ['dmax', 'dmean']:
            return -9999
        elif self.distance_metric == 'dmin':
            return 9999

    def perform_dist_test(self, threshold):
        if self.distance_metric == 'dmin':
            pass
    def calculate_needed_dist(self, Apt, Bpts):
        na_row = self.dist_LUT[Apt]
        # if we are looking for dmin (minimum distanct between clusters)
        if self.distance_metric == 'dmin':
            bpt, distance = get_select_min_idx(na_row, Bpts)
            return distance
        # if we are looking for dmin (minimum distanct between clusters)
        elif self.distance_metric == 'dmax':
            bpt, distance = get_select_max_idx(na_row, Bpts)
            return distance
    def update_means(self, cls, data):
        for n in cls:
            if len(cls[n][1]) > 0:
                cls[n][0] = np.array(np.around(data[cls[n][1]].mean(axis=0), 0), dtype=np.int)
        return cls
    def perform_epoch(self):
        threshold = self.set_threshold()
        clusterA, clusterB = 0, 0
        # for each cluster look up the  distance between it's inhabitants
        # and all other inhabitants
        for c1 in range(len(self.my_clusters)-1):
            for c2 in range(c1 + 1, len(self.my_clusters)):
                # grab the two cluster list of point the cover
                c1_pts = self.my_clusters[c1].inhabitants
                c2_pts = self.my_clusters[c2].inhabitants

                #for p1 in self.my_clusters[c1].inhabitants:
                # go through cluster 1's points and look at the distance
                # between each of those, and each of the ones in the
                # current other cluster
                for p1 in c1_pts:
                    better, distance, = self.calculate_needed_dist(threshold=threshold, Apt=p1, Bpts = c2_pts)


                    # grab the 2nd clusters points
                    #for p2 in self.my_clusters[c2].inhabitants:
                    #    # compare to current threshold and if it is better
                    #    if self.dist_LUT[p1][p2] > threshold:
                    #        pass

        # and every other clusters inhabitants
        # based on whether we are looking at d max
        # dmin or dmean keep track of the shortest one
        # and whitch two clusters this involves
        # once done merge the two with min distance and repeat
        # until desired number of clusters is found
    def check_size(self):
        return len(self.my_clusters)
    def merge(self, c1i, c2i, verbose=True):
        """Hopefully will merge the two clusters"""
        c1 = self.my_clusters[c1i]
        c2 = self.my_clusters[c2i]
        # get the average for the new cluster
        self.my_clusters[c1i].value = np.stack(c1.value, c2.value).mean(axis=0)
        c1.inhabitants += c2.inhabitants
        if verbose:
            print('the merged inhabitants are ')
            print(c1.inhabitants)
            quit(-745)
        return
    def calculate_distance_LUT(self, data):
        """Will Create a look up table for the distance
           from each point to every other thing
        """
        tstart = time.time()
        adj = list(([[0]*self.size]*self.size))
        for row in range(self.size):
            for col in range(self.size):
                if row == col:
                    if self.distance_metric == 'dmin':
                        adj[row][col] = 9000
                    elif self.distance_metric == 'dmax':
                        adj[row][col] = -9000
                else:
                    if self.dist_type == 'city_block':
                        #print('city block')
                        adj[row][col] = np.linalg.norm(data[row]-data[col])
                     # dc[i2] = np.linalg.norm(self.data[i1] - self.data[i2])
            # rd[i1] = sort_dict(dc)
        #pd.DataFrame(adj[0:len(self.size)/4], dtype=np.int).to_excel('The_LUT.xlsx')
        print('Making the LUT took {}'.format(time.time()-tstart))
        return np.array(adj, dtype=int)
    def calculate_cluster_diffs(self, rl):
        for cls in range(len(rl)-1):
            for cls2 in range(cls+1, len(rl)):
                dis = np.linalg.norm((rl[cls].value - rl[cls2]))
                rl[cls].cluster_dist[cls2] = dis
                rl[cls2].cluster_dist[cls] = dis
        # sort the dictionary of distances by value
        for cls in range(len(rl)):
            rl[cls].cluster_dict = sort_dict(rl[cls].cluster_dict)
        return rl
    def dmin_merge(self):
        pass
    def dmax_merge(self):
        pass
    def dmean_merge(self):
        pass
    def merge_clusters(self):
        if self.distance_metric == 'dmin':
            self.dmin_merge()
        elif self.distance_metric == 'dmax':
            self.dmax_merge()
        elif self.distance_metric == 'dmean':
            self.dmean_merge()
    def epsilon(self, emax, emin, k, kmax):
        return emax * ((emin/emax)**(k/kmax))

    def wta_update_cls(self, cmean, X, verbose=False, epsln=.001):
        return cmean + epsln*(X - cmean)
    def wta_init_run(self, data, change_threshold=0.09):
        gaussrndm = get_truncated_normal(sd = self.std, mean=self.mu)
        rl = list()
        cls = dict()
        change_threshold = self.del_thrsh
        epsln = .1
        # initialize the clusters randomly with a
        # gaussian distribution of random numbers
        for l in range(self.k):
            #print(l)
            cls[l] = []
            cls[l].append(get_rounded_int_array(gaussrndm.rvs(3)))
            cls[l].append(list())
        #cls = self.update_means(cls, data)
        tot = 0
        #for c in cls:
        #    print(cls[c])
        #    print(c)
        #    tot += len(cls[c][1])

        change = True
        tstart = time.time()
        # for each point calculate the distance and as you go keep track of the min
        # at end of loop add self to one with min distance
        # then adjust means and repeat until there or no more changes
        while change:
            change = False
            change_cnt = 0
            # for each sample pixel
            # find its nearest mean and put in its
            # cluster and adjust that cluster
            # mean toward the new point
            for sample in range(len(data)):
                dis = 999999
                best = None
                cnt = 0
                # go through all clusters
                for i in cls:
                    # if using dmin or max
                    if self.distance_metric in ['dmin', 'dmax']:
                        if len(cls[i][1]) == 0:
                            cdis = np.linalg.norm(cls[i][0] - data[sample])
                        elif self.distance_metric == 'dmin':
                            if cnt == 0:
                                print('dmin')
                            tmpd = 99999
                            for pt in cls[i][1]:
                                if pt != sample:
                                    dp = np.linalg.norm(data[sample]-data[pt])
                                    if dp < tmpd and dp != 9999:
                                        #print(dp, tmpd)
                                        tmpd = dp
                                        best = i
                            cdis = tmpd
                            #if cnt == 0:
                            #    print('the min dis is {}'.format(cdis))
                            cnt += 1
                        elif self.distance_metric == 'dmax':
                            tmpd = -99999
                            for pt in cls[i][1]:
                                dp = np.linalg.norm(data[sample]-data[pt])
                                if dp > tmpd:
                                    tmpd = dp
                            cdis = tmpd
                    elif self.dist_type == 'city_block':
                        cdis = np.linalg.norm((cls[i][0]- data[sample]))
                    else:
                        if len(cls[i][1]) <= 1:
                            cdis = np.linalg.norm((cls[i][0]- data[sample]))
                        elif np.linalg.cond(cls[i][1]) < 1 / sys.float_info.epsilon:
                            cov = np.linalg.inv(cls[i][1])
                        else:
                            cov = pd.DataFrame(data[cls[i][1]]).std(axis=0).mean().values
                            cov = cov**2
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov, is_std=True)
                            #print(cov)
                            if self.epochs < 5:
                                print('cov')
                                print(cov)
                                print(cov.shape)
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov)
                        #cdis = mahalanobis_distance(data[sample], cls[i][0])
                    cnt += 1
                    #cdis = np.linalg.norm((cls[i][0]- data[sample]))
                    #print('evaluating {}'.format(cdis))
                    if cdis < dis:
                        dis = cdis
                        best = i
                # if I am already in this cluster
                # keep going
                if sample in cls[best][1]:
                    continue
                else:
                    change = True
                    change_cnt += 1
                    # find where the sample was and remove it
                    for n in cls:
                        if sample in cls[n][1]:
                            del cls[n][1][cls[n][1].index(sample)]
                            break
                    cls[best][1].append(sample)
                    # now update the center
                    cls[best][0] = self.wta_update_cls(cls[best][0], data[sample], epsln=epsln)
            # once we are done with this run adjust means
            # cls = self.update_means(cls, data)
            # at end of loop see what % of points changed
            # if less than threshold stop
            if self.epochs > 0 and self.epochs%1 == 0:
                #epsln = epsln *.1
                epsln = self.epsilon(emax=.1, emin=.0001, k=self.epochs, kmax=30)
                print('-----------------------------epsilon', epsln)
            if (change_cnt/self.size) < change_threshold:
                change = False
                print('the threshold was hit {}'.format(change_cnt/self.size))
            elif self.epochs%50 == 0:
                print('{0} points changed or {1}%, {2}'.format(change_cnt, (change_cnt/self.size), epsln))

            self.epochs += 1
            print('Epoch {:d}, changed {:d}'.format(self.epochs, change_cnt))
        self.time_taken = time.time() - tstart
        return self.rescale_ppm(data, cls)

    def kmean_init_run(self, data, change_threshold=.001):
        gaussrndm = get_truncated_normal(sd = self.std, mean=self.mu)
        rl = list()
        cls = dict()
        #change_threshold = self.del_thrsh
        rdch = np.random.choice(range(self.size), self.size, replace=False)
        start = 0
        end = int(self.size/self.k)
        step = int(self.size/self.k)
        #print('step 886', step)
        for l in range(self.k):
            #print(l)
            cls[l] = []
            cls[l].append(get_rounded_int_array(gaussrndm.rvs(3)))
            cls[l].append(list())
            for i in range(start, end):
                cls[l][1].append(rdch[i])
            start = end
            end = min(end + step, self.size)
        # initialize the means based on whats in the
        cls = self.update_means(cls, data)
        #tot = 0
        #for c in cls:
        #    #print(cls[c])
        #    #print(c)
        #    tot += len(cls[c][1])
        #print('total', tot)
        change = True
        tstart = time.time()
        # for each point calculate the distance and as you go keep track of the min
        # at end of loop add self to one with min distance
        # then adjust means and repeat until there or no more changes
        self.epochs = 0
        print('Starting the while loop')
        epsln = .1
        while change:
            change = False
            change_cnt = 0
            # perform the epoch
            #  for every sample
            for sample in range(len(data)):
                dis = 999999
                best = None
                change_cnt = 0
                # go through current means
                # and find the closest
                for i in cls:
                    if self.distance_metric in ['dmin', 'dmax']:
                        #print('dmin or max')
                        if len(cls[i][1]) == 0:
                            cdis = np.linalg.norm(cls[i][0] - data[sample])
                        elif self.distance_metric == 'dmin':
                            tmpd = 99999
                            for pt in cls[i][1]:
                                if pt != sample:
                                    dp = np.linalg.norm(data[sample] - data[pt])
                                    if dp < tmpd:
                                        tmpd = dp
                            cdis = tmpd
                        elif self.distance_metric == 'dmax':
                            tmpd = -99999
                            for pt in cls[i][1]:
                                if pt != sample:
                                    dp = np.linalg.norm(data[sample] - data[pt])
                                    if dp > tmpd:
                                        tmpd = dp
                            cdis = tmpd
                    elif self.dist_type == 'city_block':
                        cdis = np.linalg.norm((cls[i][0]- data[sample]))
                    else:
                        if len(cls[i][1]) <= 1 and np.linalg.cond(cls[i][1]) < 1 / sys.float_info.epsilon:
                            cov = np.linalg.inv(cls[i][1])
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov)
                        elif len(cls[i][1]) <= 1:
                            cdis = np.linalg.norm((cls[i][0]- data[sample]))
                        else:
                            cov = pd.DataFrame(data[cls[i][1]]).std(axis=0).mean()
                            cov = cov**2
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov, is_std=True)
                            #print(cov)
                            if self.epochs < 5:
                                print('cov')
                                print(cov)
                                print(cov.shape)
                            #cdis = mahalanobis_distance(data[sample], cls[i][0], cov)


                        if self.epochs < 5:
                            print('my mahala in kmean')
                        if len(cls[i][1]) <= 1:
                            cdis = np.linalg.norm((cls[i][0]- data[sample]))
                        else:
                            cov = pd.DataFrame(data[cls[i][1]]).cov().values
                            print(cov)
                            if self.epochs < 5:
                                print('cov')
                                print(cov)
                                print(cov.shape)
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov)
                    if cdis < dis:
                        dis = cdis
                        best = i
                # if I am already in the closest cluster
                # stay there and go to next sample
                if sample in cls[best][1]:
                    continue
                # otherwise put the sample in it's
                # closest cluster after removing
                # it from its current one then
                # set that a change occurred
                # and keep track of how many
                else:
                    change = True
                    change_cnt += 1
                    # find where the sample was and remove it
                    for n in cls:
                        if sample in cls[n][1]:
                            del cls[n][1][cls[n][1].index(sample)]
                    cls[best][1].append(sample)
                    cls = self.update_means(cls, data)

            if self.epochs > 0 and self.epochs % 1 == 0:
                #epsln = epsln * .1
                epsln = self.epsilon(.1, .0001, self.epochs, 40)
                print('-----------------------------epsilon', epsln)
            if (change_cnt / self.size) < change_threshold:
                change = False
                print('the threshold was hit {}'.format(change_cnt / self.size))
            elif self.epochs % 50 == 0:
                print('{0} points changed or {1}%, {2}'.format(change_cnt, (change_cnt / self.size), epsln))
            # once we are done with this run adjust means
            self.epochs += 1
            print('Epoch {:d}, changed {:d}'.format(self.epochs, change_cnt))
        print('it took {} epochs'.format(self.epochs))
        self.time_taken = time.time() - tstart
        return self.rescale_ppm(data, cls)

    def phi(self, coord1, coordwinner, sig=1):
        if coord1 in [0,self.k-1] and coordwinner in [0, self.k-1] and coord1 != coordwinner:
            coord1, coordwinner = 1, 0
        return np.exp(-1*((((coord1 - coordwinner)**2)/(2*sig**2))))
    def kohonen_update_cls(self, cmeans, X, winner, verbose=False, epsln=.01, alpha=.0001):
        for i in range(len(cmeans)):
            cmeans[i][0] = cmeans[i][0] + epsln*self.phi(i, winner)*(X - cmeans[i][0])
        return cmeans
    def kohonen_init_run(self, data, change_threshold=.01):
        gaussrndm = get_truncated_normal(sd = self.std, mean=self.mu)
        rl = list()
        cls = dict()
        #change_threshold = self.del_thrsh
        epsln = .1
        tmp_dict = dict()
        vecs, dists = list(), list()
        # initialize the clusters randomly with a
        # gaussian distribution of random numbers
        for l in range(self.k):
            vecs.append(get_rounded_int_array(gaussrndm.rvs(3)))
            dists.append(np.linalg.norm(vecs[-1]))

        dists_sort = sorted(dists)
        vecs2 = list()
        for i in range(len(dists_sort)):
            idx = dists.index(dists_sort[i])
            dists[idx] = -99
            vecs2.append(vecs[idx])


        print(self.k)
        print(len(vecs2))
        #tmp_dict = sort_dict(tmp_dict, sort_by='keys')
        #print(tmp_dict)
        #mus = list(tmp_dict.values())
        #print(len(mus))
        #quit(-1104)

        # initialize the clusters randomly with a
        # gaussian distribution of random numbers
        for l in range(self.k):
            #print(l)
            cls[l] = []
            cls[l].append(vecs2[l])
            cls[l].append(list())

        #cls = self.update_means(cls, data)
        tot = 0
        #for c in cls:
        #    print(cls[c])
        #    print(c)
        #    tot += len(cls[c][1])

        change = True
        tstart = time.time()
        # for each point calculate the distance and as you go keep track of the min
        # at end of loop add self to one with min distance
        # then adjust means and repeat until there or no more changes
        while change:
            change = False
            change_cnt = 0
            # for each sample pixel
            # find its nearest mean and put in its
            # cluster and adjust that cluster
            # mean toward the new point
            for sample in range(len(data)):
                dis = 999999
                best = None
                for i in cls:
                    if self.dist_type == 'city_block':
                        cdis = np.linalg.norm((cls[i][0]- data[sample]))
                    else:
                        if self.epochs < 5:
                            print('')
                        if len(cls[i][1]) <= 1:
                            cdis = np.linalg.norm((cls[i][0]- data[sample]))
                        elif np.linalg.cond(cls[i][1]) < 1 / sys.float_info.epsilon:
                            cov = np.linalg.inv(cls[i][1])
                        else:
                            cov = pd.DataFrame(data[cls[i][1]]).std(axis=0).mean().values
                            cov = cov**2
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov, is_std=True)
                            #print(cov)
                            if self.epochs < 5:
                                print('cov')
                                print(cov)
                                print(cov.shape)
                            cdis = mahalanobis_distance(data[sample], cls[i][0], cov)
                        #cdis = mahalanobis_distance(data[sample], cls[i][0])

                    if cdis < dis:
                        dis = cdis
                        best = i
                # if I am already in this cluster
                # keep going
                if sample in cls[best][1]:
                    continue
                else:
                    change = True
                    change_cnt += 1
                    # find where the sample was and remove it
                    for n in cls:
                        if sample in cls[n][1]:
                            del cls[n][1][cls[n][1].index(sample)]
                            break
                    cls[best][1].append(sample)
                    # now update the center
                    cls = self.kohonen_update_cls(cls, data[sample], best, epsln=.001)
            # once we are done with this run adjust means
            # cls = self.update_means(cls, data)
            # at end of loop see what % of points changed
            # if less than threshold stop
            if self.epochs > 0 and self.epochs%10 == 0:
                #epsln = epsln *.1
                epsln = self.epsilon(.001, .0001, self.epochs, 40)
                print('-----------------------------epsilon', epsln)
            if (change_cnt/self.size) < change_threshold:
                change = False
                print('the threshold was hit {}'.format(change_cnt/self.size))
            elif self.epochs%50 == 0:
                print('{0} points changed or {1}%, {2}'.format(change_cnt, (change_cnt/self.size), epsln))
            self.epochs += 1
            print('Epoch {:d}, changed {:d}'.format(self.epochs, change_cnt))
        self.time_taken = time.time() - tstart
        return self.rescale_ppm(data, cls)

    def init_random_sample(self, ):
        print('rndm samp')
        print(self.df)
        cp = self.df.copy().values
        np.random.shuffle(cp)
        print(cp)
        new_clusters = cp[0:self.k]
        rl = list()
        # create the list of cluster objects
        for c in range(len(new_clusters)):
            rl.append(cluster(self.k, rc=c, value=new_clusters[c]))
        # calculate the cluster distances
        rl = self.calculate_cluster_diffs(rl)
        return rl
    def normal(self, ):
        pass
    def check_init(self, data):
        if self.init is 'random' and self.method == 'kmeans':
            # generate k random 1X3 list
            # that are from 0 -256
            print('kmeans')
            return self.kmean_init_run(data)
        elif self.init is 'random' and self.method == 'wta':
            print('wta')
            # generate k random 1X3 list
            # that are from 0 -256
            return self.wta_init_run(data)
        elif self.method == 'kohonen':
            print('kohonen')
            return self.kohonen_init_run(data)
        elif self.init is 'normal':
            pass

    def algo_init(self, data, verbose=True):
        """Will initialize the clusters to just
          start as the different samples
        """
        #rl = list([[0]*self.dimen]*self.size)
        rl = list()
        # create a cluster for every row
        # to start
        for row in range(len(data)):
            rl.append(cluster(k=self.k, rc=row, value=data[row]))
            rl[-1].inhabitants.append(data[row])
        return rl

    def adjust_pic(self, df, cls):
        for cl in cls:
            df[cl] = df[cl].mean(axis=0)
        return df

    def find_my_cluster(self, pt, cls):
        for cl in range(len(cls)):
            if pt in cls[cl]:
                return cl
        return None

    def algo_init2(self, data, verbose=True):
        rl = list()
        print('initializing for algorithmic cluster')
        for row1 in range(len(data)-1):
            #if row1 > 20:
            #    break
            for row2 in range(row1+1, len(data)):
                if row1 != row2:
                    rl.append([ np.linalg.norm(data[row1] - data[row2]), int(row1), int(row2)])

        #dummy = np.array(rl[0:10])
        #if verbose:
        #    print('the dummy is ')
        #    print(dummy)
        #dummy2 = col_sort(dummy, 0)
        #if verbose:
        #    print('the dummy2 is ')
        #    print(dummy2)

        rl = col_sort(np.array(rl))
        print('it')
        sound_alert_file(r'')
        print(rl[0:5])
        cls = list([[]]*self.size)

        for idx in range(self.size):
            cls[idx].append(idx)
        for edge in rl:
            cl1 = self.find_my_cluster(edge[1], cls)
            cl2 = self.find_my_cluster(edge[2], cls)
            if cl1 == cl2:
                #already in the same group
                continue
            else:
                cls.append(cls[cl1] + cls[cl2])
                del cls[cl1]
                del cls[cl1]
            if len(cls) == self.k:
                break

        return self.adjust_pic(data, cls)

    def check_method(self, data, verbose=False):
        if self.method is 'algo':
            if verbose:
                print('checked, algo')
            return self.algo_init2(data)
        else:
            return self.check_init(data)
    def rescale_ppm(self, df, cls):
        new_image = None
        # for all of my clusters
        # go through the pixels that belong to it
        # and change thier color values to the clusters
        # color values
        for c in cls:
            val = cls[c][0]
            to_fix = cls[c][1]
            df[to_fix] = val
        #for clstr in self.my_clusters:
        #    for pix in clstr.inhabitants:
        #        self.df[pix] = clstr.value
        return df
class cluster():
    """Represents an individual cluster"""
    def __init__(self, k, rc, value):
        self.k = k                      # number of sibling cluster
        self.value = value              # the current value of the mean I hold
        self.row=rc                     # the row in the toplogical grid I'm in.
        self.cluster_dist = dict()      # distances to the other clusters
        self.inhabitants = list()       # the row number of samples in this cluster
    def get_size(self):
        return len(self.inhabitants)
    def k_mean_calculate_mean(self, df):
        self.value = df[self.inhabitants, ].mean(axis=0)
    def wta_calculate_mean(self, df, eta):
        pass
    def kohonen_calculate_mean(self, df, eta, phi, pho_std):
        pass

class cluster_algos():
    """My collection of clustering algorithms"""
    def __init__(self, df, method='algo', init='random', k=None, distance_type='city_block', distance_metric='dmean'):
        self.df = df                                    # the data we will be working with
        self.my_clusters = clusters(k=k, df=self.df, method=method, init=init, distance_type=distance_type, distance_metric=distance_metric)      # my collection of clusters
        self.k = k                                      # desired number of clusters
    def algorithmic_cluster(self, ):
        """Algorithmic clustering"""

        # while the number of clusters is < k
        cnt = 0
        # do my algorithmic thing yo !!!
        # i.e. run throu some number of epochs or until the desired
        # number of k's is reached
        while self.my_clusters.check_size() > self.k:
            if cnt%(1000) == 0: # shows every thousandth cluster
                print('There are {} clusters'.format(self.my_clusters.check_size()))
            cnt += 1
            # tell the clusters to perform and epoch
            # this will conjoin the closest groups
            # two at a time
            self.my_clusters.perform_epoch()
            # TODO: create a conversion method to
            #  convert old image into rescaled one
            self.my_clusters.rescale_ppm(self.df)

    def finish_init(self):
        pass
    def fit(self, cmodel):
        pass
    def predict(self, X):
        pass
    def score(self, X, Y):
        pass

        # takes the known or Training set and the testing set
        def calculate_distances(self, cluster_means, samples, dist_dic):
            distances_dict = {}

            # iterate through samples of test set
            # calculating the distances between each
            # sample and all other samples in the training set
            for sample1 in range(len(samples)):
                # print(df.iloc[sample1, :])
                # print('==================================')
                # print('==================================')
                # print('==================================')

                # create a dictionary for this sample this will store the distances
                distances_dict[sample1] = {}
                for sample2 in range(len(cluster_means)):
                    # calculate the distance and store it in the dictionary for this entry
                    # print(df.iloc[sample2, :])
                    # print(np.linalg.norm(df_te.iloc[sample1, :].values - df_tr.iloc[sample2, :].values))
                    # print(self.euclidian_dist(df_tr.iloc[sample2, :].values, df_te.iloc[sample1, :].values))
                    # distances_dict[sample1][sample2] = self.euclidian_dist(df_tr.iloc[sample2, :].values, df_te.iloc[sample1, :].values)
                    distances_dict[sample1][sample2] = np.linalg.norm(
                        samples.iloc[sample1, :].values - cluster_means.iloc[sample2, :].values)

                # distances_dict[sample1] = sorted(distances_dict[sample1].items(), key=lambda kv: kv[1])
                distances_dict[sample1] = dict(sorted(distances_dict[sample1].items(), key=operator.itemgetter(1)))
                # distances_dict[sample1] = sorted(distances_dict[sample1].items(), key=operator.itemgetter(1))
                # print(distances_dict[sample1])
            return distances_dict

class Gknn():
    def __init__(self, k=10, dist_metric='euclidean'):
        self.k=k
        self.X=None
        self.y=None
        self.cov=None
        self.inv_cov=None
        self.dist_metric=dist_metric
    def fit(self, X, y):
        self.y = y
        self.X = X
        self.cov = X.cov()
        self.inv_cov = np.linalg.inv(self.cov)

    def predict(self, X):
        dist_dic = self.calculate_distances(X)
        real = self.y.values.flatten().tolist()
        candidates = list(set(real))
        final_tallys = list()
        projected = list()
        yp = list()
        for zone in dist_dic:
            nn = list(dist_dic[zone].keys())
            votes = self.y.values[nn, :].flatten().tolist()
            ballot = {}
            for c in candidates:
                ballot[c] = votes.count(c)
            yp.append(sort_dict(ballot) )


    def calculate_distances(self, X):
        cov=None
        if type_check(X, against='dataframe'):
            cov = self.X.cov()
        else:
            cov = pd.DataFrame(X).cov()

        ret_dict = {}
        for i in range(len(X)):
            cdl = {}
            for j in range(len(self.X)):
                #cdl[j] = mahalanobis_distance(X[i], self.X.values[j], )
                #cdl[j] = np.linalg.norm(X[i]-self.X.values[j])
                cdl[j] = self.get_distance(X[i], j)
                cdl = sort_dict(cdl)
                cnt = 0
                ndl = {}
                for ky in cdl:
                    ndl[ky] = cdl[ky]
                    cnt += 0
                if cnt == self.k:
                    break
            ret_dict[i] = ndl
        return ret_dict

    def get_distance(self, x, j):
        if self.dist_metric == 'city_block':
            return np.linalg.norm(x - self.X.values[j])
        elif self.dist_metric == 'mahalanobis':
            return mahalanobis_distance(x, self.X.values[j], self.inv_cov)
        else:
            return euclidean_distance(x, self.X.values[j], np.mean(self.X.values.std(axis=0)))

class Neuron():
    # source for activation functions
    # https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
    """
        Class object representing a neuron
    """
    def __init__(self, input_size, eta=.01, w=None, activation=0, error='mse'):
        self.input_size = input_size        # number of synaptic connections on dendrites
        self.eta=eta                        # learning rate
        self.activation = activation        # the type of activation function to use
        self.b = np.array([-1], dtype=np.float)                       # bias or threshold to overcome
        self.error=error
        # if no initial weights given
        # then set them to random values of either
        # -.1 or .1, with input_size elements
        if w is None:
            self.w = np.array([np.random.choice([-.1, .1]) for i in range(input_size)])
        else:
            self.w = w

    def processInput(self, x, w, b):
        """
                Performs summation operation for the neuron and adds bias
        :param x: inputs signals to sum
        :param w: weights of the various inputs
        :param b: bias to overcome
        :return:   the value of the summation operation on the weighted inputs
                   with the biase added (xn*wn **** + b)
        """
        return np.dot(x, w) + b

    def activation_func(self, x, w, b):
        """  Performs the activation function calculation selected when neuron
             was instantiated
        :param x:  input value numpy array from inputing neurons
        :param w:  weights on inputs
        :return:
        """
        if self.activation == 0:
            return self.sigmoid(self.processInput(x,w.transpose(),b))
        elif self.activation == 1:
            return self.linear(self.processInput(x, w.transpose(), b))
        elif self.activation == 2:
            self.relu(self.processInput(x,w.transpose(),b))
        elif self.activation == 3:
            self.tanH(self.processInput(x,w.transpose(),b))
        elif self.activation == 4:
            self.relu(self.processInput(x,w.transpose(),b))
        elif self.activation == 5:
            self.softplus(self.processInput(x,w.transpose(),b))
        elif self.activation == 6:
            self.garctan(self.processInput(x,w.transpose(),b))
        elif self.activation == 7:
            self.perceptron(self.processInput(x,w.transpose(),b))

    def activation_funcPrime(self, x, w, b):
        """  Performs the activation function calculation selected when neuron
             was instantiated
        :param x:  input value numpy array from inputing neurons
        :param w:  weights on inputs
        :return:
        """
        if self.activation == 0:
            return self.sigmoid_prime(self.processInput(x,w.transpose(),b))
        elif self.activation == 1:
            self.linear_prime(self.processInput(x,w.transpose(),b))
        elif self.activation == 2:
            self.relu_prim(self.processInput(x, w.transpose(), b))
        elif self.activation == 3:
            self.tanH_prime(self.processInput(x,w.transpose(),b))
        elif self.activation == 4:
            self.relu_prim(self.processInput(x,w.transpose(),b))
        elif self.activation == 5:
            self.softplus_prime(self.processInput(x,w.transpose(),b))
        elif self.activation == 6:
            self.arctan_prime(self.processInput(x,w.transpose(),b))
        elif self.activation == 7:
            self.perceptron_prime(self.processInput(x,w.transpose(),b))

    def calculate(self, x, w=None, b=None ):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        return self.activation_func(x, w, b)

    def calculate_error(self,yt, yp, error=None):
        if error is None:
            error = self.error
        if error == 'mse':
            return gmse(yt, yp)
        elif error == 'entropy':
            return 1

    def error_func(self, ):
        pass
    def process_output1(self, val):
        if val > 0:
            return 1
        return 0

    def adjust_weights(self, delta, lr=None):
        if lr is None:
            lr = self.eta
        return self.w - lr*delta

    def adjust_bias(self, delta, lr=None):
        if lr is None:
            lr = self.eta
        return self.b - lr*delta

    def backpropagate(self, x, y, act_func=None):
        # set up the arrays for updateing
        # the weights and biases
        if act_func is None:
            act_func = self.activation
        else:
            self.activation = act_func



        pass

    def linear(self, z):
        return z

    def linear_prime(self, z):
        return 1

    ### Miscellaneous functions
    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanH(self, z):
        """the Tanh activation function"""
        return (2.0 / (1.0 + np.exp(-2 * z))) - 1

    def tanH_prime(self, z):
        return 1 - self.tanH(z) ** 2

    def softplus(self, z):
        return np.log(1 + np.exp(z))

    def softplus_prime(self, z):
        return 1 / (1 + np.exp(-z))

    def garctan(self, z):
        return np.arctan(z)

    def arctan_prime(self, z):
        return 1 / (z ** 2 + 1)

    def perceptron(self, z):
        """
            perceptron thresholding function, returns 1 iff
            z is non negative, otherwise returns 0
        :param z: input to threshold
        :return:
        """
        if z >= 0:
            return 1
        else:
            return 0

    def perceptron_prime(self, z):
        if z != 0:
            return 0

    def relu(self, z):
        if z < 0:
            return 0
        else:
            return z

    def relu_prim(self, z):
        if z < 0:
            return 0
        else:
            return 1

class G_NN():
    def __init(self):
        self.data=None
        self.X = None
        self.y = None
        self.Trainset= None
        self.Testset=None


class naive_bayes_classifier():
    def __init__(self, classifiers=(), verbose=True):
        self.cms = None
        self.LUT = None
        self.clsf = [clf for clf in classifiers]
        self.accuracies = list()
    # will fit all classifiers with given training
    # set and generate confusion matrices for each
    def fit(self, X, y, verbose=False):
        for clf in range(len(self.clsf)):
            self.clsf[clf].fit(X,y)
            yp = self.clsf[clf].predict(X)
            self.accuracies.append(accuracy_score(y, yp))
        self.cms = self.generate_cms(X, y)
        self.LUT = self.naive_bayes_cm_fnc(self.cms)

    def generate_cms(self, X, y, verbose=False):
        """
            will generate the confusion matrices for the baysian
            classifier
        :param X: predictor
        :param y: target
        :param verbose:
        :return: list of confusion matricies, entry  i is classifier i's
        """
        return [confusion_matrix(y, clsf.predict(X)) for clsf in self.clsf]

    def set_cms(self, cms):
        for clf in range(len(self.clsf)):
            self.clsf[clf].cm = cms[clf]

    def predict(self, x):
        """ Will go through classifiers, getting prediction lists
        :param x:
        :return:
        """
        # will contain a list of lists where list i is for
        # observation i, and the contents of list i are
        # the predictions for observation i from the classifiers
        predictions = list()

        if type(x) == type(pd.DataFrame([])):
            print('fix the object')
            x = x.values

        for obs in range(len(x)):
            # create list i
            predictions.append(list())
            # move through classifiers generating predictions for
            # observation i
            for clsf in self.clsf:
                predictions[obs].append(clsf.predict([x[obs]]))
                #print(predictions[obs])
        tl = []
        # now go through the predictions for each observation using the
        # look up table to make the final prediction
        yp = list()
        for obs in range(len(predictions)):
            #print(self.LUT[predictions[obs][0], predictions[obs][1]].tolist())
            yp.append( self.LUT[predictions[obs][0], predictions[obs][1]].tolist()[0].index(max(self.LUT[predictions[obs][0], predictions[obs][1]].tolist()[0])))

        #print(yp)
        return yp

    def naive_bayes_cm_fnc(self, conmats, verbose=True):
        # print(cm1.transpose().reshape((9,)))
        # print(cm1[0:, 0])
        # print(cm2[0:, 0])
        # print(cm1[0:, 0]*cm2[0:,0])

        nbcml = []                      # naive bayes confusion matrix list
        for cms in conmats:
            nbcml.append(self.nb_cm(cms))
        nb1 = self.nb_cm(conmats[0])           # confusion matrix object 1
        nb2 = self.nb_cm(conmats[1])           # confusion matrix object 2

        if False:
            print('cm1')
            print(nb1.cm)
            print('cm2')
            print(nb2.cm)
            print('prob table 1')
            print(nb1.prob_table)
            print('prob table 2')
            print(nb2.prob_table)
            print('cm1:')
            print(cm1)
            print('cm2:')
            print(cm2)
            print('product')
            print(cm1 * cm2)

#        shp = cm1.shape[0]
#        shp = nbcml[0].cm.shape[1]
        shp = conmats[0].shape[0]
        # create empty look up table
        tupparam = tuple([shp for i in range(shp+1)])
        look_up = np.empty(tupparam)
        print(look_up)

        # print(nb_mat)
        # nb_mat[0,0] = cm1[:,0]*cm2[:,0]
        # print(nb_mat[0,0,:])
        # print(nb_mat[0,1,:])

        for col1 in range(shp):
            for colb in range(shp):
                look_up[col1, colb] = nb1.prob_table[:, col1] * nb2.prob_table[:, colb]
        return look_up

    class nb_cm():
        """represents a confusion matrix from a classifier"""
        def __init__(self, cm):
            self.cm = cm                                    # the confusion matrix stored
            self.row_sums = self.calc_row_sums()            # the row sums (class counts) for each class
            self.prob_table = self.create_prob_table()      # probability table used to make look up table
        # counts the number of each class in the confusion matrix
        def calc_row_sums(self):
            return [sum(r) for r in self.cm]
        # probability table created from confusion matrix
        def create_prob_table(self):
            return self.cm / self.row_sums

class GLIN_Regressor(Learner):
    def __init__(self, X, Y, Xts, Yts, w=None, c = 0, intercept=True, eta=.0005, etamin=.000001,
                 etamax=.0001, kmax=200, eta_dec=True, epochs=90000, cost_func='mse', wgt='zero', tol=1e-3, lm=4):
        super().__init__()
        self.data = None
        self.intercept = intercept
        self.w = w
        self.wgt = wgt
        self.c = c
        self.X = X
        self.Y = Y
        self.Xts = Xts
        self.Yts = Yts
        self.tol = tol
        self.Ymean = Y.mean(axis=0)
        self.test_epochs = None
        self.test_mae = None
        self.test_cod = None
        self.test_rmse = None
        #print(self.Ymean)
        self.Ymeants = Yts.mean(axis=0)
        #print(self.Ymeants)
        self.best_MAE = 999999
        self.N = len(X)
        print('N:', self.N)
        self.d = X.shape[1]
        self.eta = eta
        self.epochs = epochs
        self.etamax=etamax
        self.etamin=etamin
        self.kmax=kmax
        self.eta_dec=eta_dec
        self.cost_fnc = cost_func
        self.p_scores = None
        self.Vif_scores = None
        self.Rsqr = None
        self.wald_chi = None
        self.best_w = None
        self.best_b = None
        self.best_MSE = 200000000000
        self.best_RMSE = 200000000000
        self.best_COD = 200000000000
        self.best_MAE = 200000000000
        self.best_Rsqr = 20000000000
        self.epoch_stop = -99
        self.lm = lm
        self.finish_init()

    # performs multiple linear regression on the x and y data
    # and returns the generated parameter vector W
    def multi_linear_regressor(self, x_data, y_data):
        x = np.array(x_data, dtype=np.float)
        y = np.array(y_data, dtype=np.float)
        x_transpose = np.transpose(x)
        xtx = np.dot(x_transpose, x)
        xtx_inv = np.linalg.inv(xtx)
        xtx_inv_xt = np.dot(xtx_inv, x_transpose)
        w = np.dot(xtx_inv_xt, y)
        return w

    def wgt_predict(self, wgt, X):
        return np.dot(X.transpose(), wgt)

    def sigmoid(self, X):
        return 1/(1-np.e**(-X))

    def finish_init(self):
        """
            Sets up the weights and b
        :return:
        """
        if self.w is None:
            if self.wgt == 'random':
                self.w = get_truncated_normal(mean=0, sd=1, low=-1, upp=1).rvs(self.d)
            elif self.wgt== 'zero':
                self.w = np.array([0] * self.d)
            elif self.wgt == 'ols':
                self.w = np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose())
                self.w = np.dot(self.w, self.Y)
            """
            if self.intercept:
                if self.wgt == 'random':
                    # get a normally distributed randomized weight vector
                    self.w = get_truncated_normal(mean=0, sd=1, low=-1, upp=1).rvs(self.d)
                    self.w = np.array(self.w + [1])
                else:
                    #b = list([[1]]*(self.N))                    # add intercept
                    #self.X = np.hstack((self.X, b))
                    #self.d = self.d + 1
                    #b = list([[1]]*(len(self.Xts)))                    # add intercept
                    #self.Xts = np.hstack((self.Xts, b))
            else:
                self.w = get_truncated_normal(mean=0, sd=1, low=-1, upp=1).rvs(self.d)
                self.w = np.array(self.w)
        self.wd = np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose())
        self.wd = np.dot(self.wd, self.Y)
        print('wd shape', self.wd.shape)
        print(self.wd)
        """

    def predicted_weights(self, yp):
        wd_a = np.dot(np.linalg.inv(np.dot(self.X.transpose(), self.X)), self.X.transpose())
        return np.dot(wd_a, yp)

    def predict_score(self,X, ytr, ypr, verbose=False):
        pass

    def report_scores(self, ytr, ypr, verbose=False, Ymean=None):
        if Ymean is None:
            Ymean = self.Ymean
        n = len(ytr)
        mmse = self.MSE(ytr, ypr, n)
        print('n',n)
        print('ypr', len(ypr))
        print('          ----------------RMSE:',np.sqrt(mmse))
        print('          -----------------MSE: ', mmse)
        print('          ---------sklearn MSE:', metrics.mean_squared_error(ytr, ypr))
        print('          -------var explained:', metrics.explained_variance_score(ytr, ypr))
        print('          -----------------MAE:', self.MAE(ytr, ypr, n))
        print('          -----------------CD:', self.R2(ytr, ypr, Ymean))
        print('          ---------sklearn r2:', metrics.r2_score(ytr, ypr))
        print('          -----------------R^2', self.Rvar(ytr, ypr, Ymean))
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------\n')

    def cost_derivative(self, X, ytruth, ypred, cost_fnc='mae'):
        if cost_fnc == 'mae':
            maePrime_w = -1/len(ytruth) * np.dot((ytruth-ypred)/(abs(ytruth - ypred)),X)
            maePrime_b = -1 / len(ytruth) * sum([(yt-yp)/abs(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            return [maePrime_w, maePrime_b]
        if cost_fnc == 'mse':
            print('mse')
            msePrime_w = -2/len(ytruth) * np.dot((ytruth-ypred), X)
            msePrime_b = -2 / len(ytruth) * sum([(yt - yp) for yt, yp, in zip(ytruth, ypred)])
            return [msePrime_w, msePrime_b]


    def fit(self,solver='mae'):
        #est = .0001
        #est = .00992
        est = self.eta
        etamax = self.eta
        etamin = self.etamin
        epochs = self.epochs
        kmax = self.kmax
        self.test_cod, self.test_epochs, self.test_mae, self.test_rmse = list(), list(), list(), list()
        #self.w = np.array([0]*self.d)
        #self.w = self.wd
        #print(self.w)
        n = self.N
        print('-----------------------N', self.N)
        print(self.cost_fnc)
        d = self.d
        mmse_old = 0
        mmae_old = 0
        old_thresh = list()
        for i in range(epochs):
            #np.random.shuffle(self.X.values)
            # get a prediction
            #print('y means', self.Ymean)
            #print('ytest means', self.Ymeants)
            #print('w\n', self.w)
            yp = np.dot(self.X, self.w) + self.c
            #print('yp\n',yp)
            #print('gf\n', self.Y)
            mmse =self.MSE(self.Y, yp, n)
            mmae =self.MAE(self.Y, yp, n)
            rmse = self.RMSE(self.Y, yp)
            self.test_epochs.append(i)
            self.test_rmse.append(rmse)
            self.test_cod.append(self.R2(self.Y, yp))
            self.test_mae.append(mmae)
            if self.cost_fnc == 'mse':
                old_thresh.append(mmse)
            elif self.cost_fnc == 'mae':
                old_thresh.append(mmae)
            if self.cost_fnc == 'mse' and mmse < self.best_MSE:
                print('--------------------------------------------------------------    New Best MSE:', mmse)
                self.best_MSE = mmse
                self.best_MAE = mmae
                self.best_RMSE = rmse
                self.best_COD = self.R2(self.Y, ypred=yp, ymean=self.Ymean)
                self.best_b = self.c
                self.best_w = self.w
                self.best_score = mmse
                self.best_epoch = i
            if self.cost_fnc == 'mae' and mmae < self.best_MAE:
                print('--------------------------------------------------------------    New Best MAE:', mmae)
                self.best_MAE = mmae
                self.best_MSE = mmse
                self.best_RMSE = rmse
                self.best_COD = self.R2(self.Y, ypred=yp, ymean=self.Ymean)
                self.best_b = self.c
                self.best_w = self.w
                self.best_score = mmae
                self.best_epoch = i


            #yp = self.X*self.w + self.c

            #print('pred',yp)
            #D_m = (-2/n) * sum(np.dot(self.X.transpose(), (self.Y - yp)))
            #D_m = (-2/n) * sum(self.X.values.reshape(self.N, 1) * (self.Y - yp))
            #D_m = (-2/n) * sum((self.Y - yp).values.reshape(self.N, 1)*self.X)
            #D_m = (-2/n) * sum((self.X.reshape(1, self.N))*(self.Y - yp))
            #D_m = (-2/n) * sum((self.X.reshape(1, self.N))*(self.Y - yp))
            #D_m = (-2/n) * sum((self.X.transpose())*(self.Y - yp))
            #print('-----------------------------------------')
            #print(self.X.shape)
            #print(self.Y-yp)

            w_b = self.cost_derivative(self.X, self.Y, yp, self.cost_fnc)

            D_m = w_b[0]
            D_c = w_b[1]
            #D_m = (-1/n) * (np.dot(1/abs(self.Y - yp), self.X))
            #D_c = (-1/n)* (1/sum(abs(self.Y-yp)))
            self.w = self.w - self.eta*D_m
            self.c = self.c - self.eta*D_c
            print('eta: {} -----------------RMSE:'.format(self.eta), np.sqrt(mmse))
            print('Epoch: {} -----------------MSE:'.format(i+1), mmse)
            print('          ---------sklearn MSE:'.format(i+1), metrics.mean_squared_error(self.Y, yp))
            print('          ---------var explained:'.format(i+1), metrics.explained_variance_score(self.Y, yp))
            print('          -----------------MAE', self.MAE(self.Y, yp, n))
            print('          -----------------CD:', self.R2(self.Y, yp, self.Ymean))
            print('          ---------sklearn r2:'.format(i+1), metrics.r2_score(self.Y, yp))
            print('          -----------------R^2', self.Rvar(self.Y, yp, self.Ymean))
            print('-------------------------------------------------------------')
            print('-------------------------------------------------------------\n')
            #if abs(mmse - mmse_old) < .00000000001:
            lm = self.lm
            if len(old_thresh) >= lm and   abs(sum(old_thresh[-lm:])/lm - old_thresh[-1]) < self.tol:
                print('-- -- -- -- -- -- -- ****** thresh met {} ******'.format(abs(sum(old_thresh[-lm:])/lm - old_thresh[-1])))
                break
            if self.cost_fnc == 'mae' and abs(mmae - mmae_old) < self.tol *.00001:
                    #print('thresh met {}'.format(abs(mmse - mmse_old)))
                    print(' ****** thresh met {} ******'.format(abs(mmae - mmae_old)))
                    break
            if self.cost_fnc == 'mse' and abs(mmse - mmse_old) < self.tol * .00001:
                    print(' ****** thresh met {} ******'.format(abs(mmse - mmse_old)))
                    break
            mmse_old = mmse
            mmae_old = mmae
            self.eta = epsilon(emax=etamax, emin=etamin, k=i, kmax=kmax)

    def fit2(self, cmodel):
        cnt = 0
        est = 1/100000
        self.eta = est
        etamax = est
        etamin = est*.01
        kmax = 10000000
        threshold = .1
        dif = 1000000
        #ymean = self.Y.values.mean(axis=0)
        ymean = self.Ymean
        self.w = self.wd
        w = self.w
        while .0001 < dif:
            pred = []
            """
            # go through making predictions correcting the error as you go
            #for x, y, w in zip(self.X, self.Y, self.w.transpose()):
            for x, y in zip(self.X, self.Y):
                # make prediction
                    #print('shape of x')
                    #print(self.wd.transpose().shape[0])
                    #print('wd')
                #print(self.w)
                    #print(self.wd.shape)
                g = np.dot(x, self.w.transpose())
                pred.append(g)
                # get the error of the derirvative

                    #wd = np.dot(np.linalg.inv(np.dot(x.transpose(), x)),x.transpose())
                    #print('w')
                    #print(self.w)
                    #cw = np.dot(wd, self.Y)
                #div = -2*np.linalg.norm(self.w - self.wd)
                #print('g')
                    #print(g)
                    #print('                  y')
                    #print(y)
                    #print('error', div)
                #self.w = self.w - eta*div
                #self.wd = self.wd - eta*div
                #print(self.w)
            # once predict done calculate error and if
            if cnt > 10:
                k = 0
            """

            """
            #score
            sum = 0
            rss = 0
            tss = 0
            for g, y in zip(pred, self.Y):
                sum += (g - y)**2
                rss +=  (g - ymean)**2
                tss += (y-ymean)**2
            print('sum', sum)
            scr = (sum/self.N)
            rsqu = (rss/tss)
            print('MSE')
            print(scr)
            print('rsqur')
            print(1- rsqu)
            """
            yp =   self.predict(self.Xts)
            #print(yp[0:5])
            #print(self.Y[0:5])
            mae, mse, rsqu, rvar, mse_prime =   self.score(ypred=yp)
            #mae, mse, rsqu, rvar, mse_prime =   self.score(ypred=pred)
            print('Epoch: {} eta:{}, mae: {}, mse: {}, R2: {}, Rvar: {}, dif {}'.format(cnt, np.around(self.eta,3), mae, mse, rsqu, rvar, dif))
            old = self.wd
            #print('old')
            #print(old)
            #print('w')
            #print(self.w)
            print('prime')
            print(mse_prime[0][0:self.d])
            print('prime')
            print(mse_prime[1])
            self.wd[0:self.d] = self.wd[0:self.d] - self.eta * mse_prime[0]
            self.wd[self.d] = self.wd[self.d] - self.eta * mse_prime[1]

            #print('old')
            #print(old)
            #print('w')
            #print(self.w)
            dif = abs(np.dot(self.w, old))

            print('')
            self.eta = epsilon(emax=etamax, emin=etamin, k=cnt, kmax=kmax)

            cnt += 1

    def g_OLS(self, x, y):
        pass

    def predict(self, X):
        return np.dot(X, self.best_w) + self.best_b

    def SSE(self, ytrue, ypred):
        return sum([(yt - yp) ** 2 for yp, yt in zip(ytrue, ypred)])

    def MSE(self, yt, yp, n=None):
        if n is None:
            n = len(yt)
        return self.SSE(yt, yp)/n
    def RMSE(self, yt, yp, n=None):
        if n is None:
            n = len(yt)
        return sqrt(self.SSE(yt, yp)/n)
    def MAE(self, ytrue, ypred, n=None):
        if n is None:
            n = len(ytrue)
        return sum([abs(yt - yp) for yp, yt in zip(ytrue, ypred)]) / n

    def SSREG(self, ypred, ymean):
        return sum([(yp - ymean) ** 2 for yp in ypred])

    def SSRES(self, ytrue, ypred):
        return sum([(yt - yp) ** 2 for yp, yt in zip(ytrue, ypred)])

    def R2(self, ytrue, ypred, ymean=None):
        if ymean is None:
            ymean = self.Ymean
        return 1 - (self.SSRES(ytrue, ypred)/self.SSTOT(ytrue, ymean))

    def Rvar(self, ytrue, ypred, ymean):
        ssreg = self.SSREG(ytrue, ymean=ymean)
        ssres = self.SSRES(ytrue=ytrue, ypred=ypred)
        return (self.SSREG(ypred, ymean)/self.N)/(self.SSTOT(ytrue, ymean)/self.N)
        # return self.SSREG(ypred, ymean)/ (ssres + ssreg)

    def SSTOT(self, ytrue, ymean):
        return sum([(yt - ymean) ** 2 for yt in ytrue])  # scatter total (sum of squares)


    def score(self, ypred, ytrue=None, ymean=None, verbose=False):
        """returns a number of scoring metrics for the predictions from a linear regression
        :param ypred: predicted values from a learner
        :param ytrue: the ground truth values
        :param ymean: the average value for the target variable
        :param verbose: how talkative you want the scoring to be
        :return: mae (mean absolute error), mse(mean square error), R2 (coefficient of determination), R2var (proportion of variance explained)
        """
        if ytrue is None:
            ytrue = self.Yts
        if ymean is None:
            ymean = self.Ymeants

        ssres = sum([(yt-yp)**2 for yp, yt in zip(ytrue, ypred)])   # residual sum of squares (error)
        mae = sum([abs(yt-yp) for yp, yt in zip(ytrue, ypred)])/len(self.Xts) # mean absolute error
        sstot = sum([(yt-ymean)**2 for yt in ytrue])           # scatter total (sum of squares)
        ssreg = sum([(yp-ymean)**2 for yp in ypred])           # sum of sqaures(variance from mean of predictions)
        mse_prime = []
        mse = ssres/len(self.Xts)
        mse_prime.append(-2*sum([np.dot(x,(yt-yp)) for yp, yt, x in zip(ytrue, ypred, self.Xts)])/len(self.Xts))
        mse_prime.append(-2*sum([(yt-yp) for yt, yp in zip(ytrue, ypred)])/len(self.Xts))
        R2 = 1 - (ssres)
        R2var = ssreg/max(.01, (ssres + ssreg))

        return mae, mse, R2, R2var, mse_prime

def random_forest_tester(X_tr, y_tr, X_ts, y_ts, verbose=False, param_grid=None, s=0, cv=5, save_feats=False):
    if param_grid is None:
        param_grid = {
            # 'n_estimators': [1500, 1800, 2000],  # how many trees in forest
            'n_estimators': [1000, 2200],  # how many trees in forest
            # 'max_features': [None, 'sqrt', 'log2'],       # maximum number of features to test for split
            'max_features': [None],  # maximum number of features to test for split
            # 'max_features': ['sqrt'],
            # 'criterion': ['gini'],
            'criterion': ['entropy'],  # how best split is decided
            # 'max_depth': [None, 10, 100, 1000, 10000],            #
            # 'max_depth': [None, 10, 100],                      # how large trees can grow
            # 'max_depth': [None, 10, 20],  # how large trees can grow
            'max_depth': [50, None],  # how large trees can grow
            'oob_score': [True],  #
            # 'min_samples_leaf': [1, 3, 5],                             # The minimum number of samples required to be at a leaf node
            'min_samples_leaf': [1],  # The minimum number of samples required to be at a leaf node
            # 'max_leaf_nodes': [None, 2, 10],
            'max_leaf_nodes': [None],
            'min_weight_fraction_leaf': [0],  #
            # 'min_samples_split': [2, .75],
            'min_samples_split': [2],
            'min_impurity_decrease': [0, .01],
            'random_state': [None],
            # 'class_weight': [None,]
            'class_weight': ['balanced_subsample', 'balanced', None, {0: .4, 1: .6}]
        }

    RF_clf0 = RandomForestClassifier()
    scorers0 = {
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),  # (TP + TN) / (TP+FP+TN+FN), overall accuracy of model
        'precision_score': make_scorer(precision_score),
        'confusion_matrix': make_scorer(confusion_matrix)
    }
    scorersSens = {
        'recall_score': make_scorer(recall_score)
    }
    scorersAcc = {
        'accuracy_score': make_scorer(accuracy_score)  # (TP + TN) / (TP+FP+TN+FN), overall accuracy of model
    }
    scorersPrec = {
        'precision_score': make_scorer(precision_score),
        # TP/(TP+FP), a metric of models ability to not miss label a positive
    }
    scorers = [scorersAcc, scorersSens, scorersPrec]
    scr = ['accuracy_score', 'recall_score', 'precision_score']
    GSCV_clf0 = GridSearchCV(estimator=RF_clf0, param_grid=param_grid, cv=cv, scoring=scorers[s], refit=scr[s])
    GSCV_clf0.fit(X_tr, y_tr)
    print('Scoring for {:s}'.format(scr[s]))
    print('ZBest Params:')
    print(GSCV_clf0.best_params_)
    print('best score: ',GSCV_clf0.best_score_)
    RF_clfstd = GSCV_clf0.best_estimator_
    feature_impz = RF_clfstd.feature_importances_
    ypz = RF_clfstd.predict(X_ts)
    feates = viz.display_significance(feature_impz, X_tr.columns.values.tolist(), verbose=True)
    if save_feats:
        pd.DataFrame({'variables': list(feates.keys()), 'Sig': list(feates.values())}).to_excel(
            'RandomForest_Feature_significance_{}_.xlsx'.format(today_is()))
    accuracy, scores, posneg, = bi_score(ypz, y_ts, vals=[0, 1], classes='')
    print('Sensitivity:', posneg['sen'])
    viz.show_performance(scores=scores, verbose=True)
    print('=================================================================================================')
    print('=================================================================================================')

def logistic_tester(X_tr, y_tr, X_ts, y_ts, verbose=False, param_grid=None, pg=1, s=0, cv=5, save_feats=False):
    if param_grid is None:
        # set up parameter grid for grid search testing
        param_gridB = {'penalty': ['elasticnet'],
                           'dual': [False],
                           'tol': [1e-4, 1e-6],
                           'Cs': [10],
                           'fit_intercept': [True],
                           'class_weight': ['balanced', {0: .6, 1: .4}, {0: .4, 1: .6}],
                           'solver': ['saga'],
                           'max_iter': [5000, 100000],
                              }
        param_gridA = {'penalty': ['l2'],
                           'dual': [False],
                           'tol': [1e-1, 1e-3],
                           'Cs': [10, 1, 5],
                           'cv': [3, 5],
                           'fit_intercept': [True],
                           'class_weight': ['balanced', {0: .5, 1: .5}, {0: .55, 1: .45}],
                           'solver': ['newton-cg', 'lbfgs', 'sag'],
                           'max_iter': [1000, 5000, 10000],
                              }
        param_gridl = {'penalty': ['l1'],
                           'dual': [False],
                           'tol': [1e-2, 1e-3],
                           'Cs': [10],
                           'cv': [3, 5],
                           'fit_intercept': [True],
                           # 'class_weight': [{0: .5, 1: .5}, {0: .6, 1: .4}],
                           'class_weight': ['balanced', {0: .5, 1: .5}, {0: .6, 1: .4}],
                           'solver': ['liblinear', 'saga'],
                           # 'max_iter': [1000, 2000, 5000],
                           'max_iter': [900, 2000, 5000],
                              }
        param_grid = [param_gridB,param_gridA, param_gridl]
        param_grid = param_grid[pg]

    # create the classifier
    log_clf0 = LogisticRegressionCV()
    RF_clf0 = log_clf0
    scorers0 = {
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),  # (TP + TN) / (TP+FP+TN+FN), overall accuracy of model
        'precision_score': make_scorer(precision_score),
        'confusion_matrix': make_scorer(confusion_matrix)
    }
    scorersSens = {
        'recall_score': make_scorer(recall_score)
    }
    scorersAcc = {
        'accuracy_score': make_scorer(accuracy_score)  # (TP + TN) / (TP+FP+TN+FN), overall accuracy of model
    }
    scorersPrec = {
        'precision_score': make_scorer(precision_score),
        # TP/(TP+FP), a metric of models ability to not miss label a positive
    }
    scorers = [scorersAcc, scorersSens, scorersPrec]
    scr = ['accuracy_score', 'recall_score', 'precision_score']
    s = 0
    cv = 5
    # perform the grid search cross validation
    GSCV_clf0 = GridSearchCV(estimator=RF_clf0, param_grid=param_grid, cv=cv, scoring=scorers[s], refit=scr[s])
    GSCV_clf0.fit(X_tr, y_tr)
    print('Scoring for {:s}'.format(scr[s]))
    print('ZBest Params for set 1:')
    print(GSCV_clf0.best_params_)
    print('best score: ', GSCV_clf0.best_score_)

    RF_clfstd = GSCV_clf0.best_estimator_
    ypz = RF_clfstd.predict(X_ts)
    # fit the
    # RF_clfstd.fit(X_trz, y_train0)

    feature_impz = RF_clfstd.coef_[0]
    # from D_Space import get_current_date
    feates = viz.display_significance(feature_impz, X_tr.columns.values.tolist(), verbose=True)
    pd.DataFrame({'variables': list(feates.keys()), 'Sig': list(feates.values())}).to_excel(
        'Logistic_correlations.xlsx')
    # generate_excel(dic=feates, name='NREL_FEAT_{}_.xlsx'.format(get_current_date()))
    accuracy, scores, posneg, = bi_score(ypz, y_ts, vals=[0,1], classes='')
    print('Sensitivity:', posneg['sen'])
    viz.show_performance(scores=scores, verbose=True)
# =========================================================================
# =========================================================================
#                            numpy tools
# =========================================================================
# =========================================================================
def get_int_mean(na):
    return np.array((np.around(na.mean(axis=0), 0)), dtype=int)

def col_sort(df, col=0):
    return df[df[:, col].argsort()]

def get_select_max_idx(na_row, selection):
    """ find the maximum distance in the given row
            based on the columns (other points) in the selection list
            the idea is that it will find the minimum distance in the
            sample rows row in the distance look up table, to points
            in some other cluster

        :param na_row:  a samples row in the distance look up table, the columns
                        represent the other points in the sample population
        :param selection:  the points you want to look at the distances too,
                           represent points in some other cluster
        :return:        returns the maximum distance and the point that relates to it
        """
    maxi = na_row[selection].max()
    ret = np.where(na_row == maxi)
    ret = ret[0][0]
    return ret, maxi

def get_select_min_idx(na_row, selection):
    """ find the minimum distance in the given row
        based on the columns (other points) in the selection list
        the idea is that it will find the minimum distance in the
        sample rows row in the distance look up table, to points
        in some other cluster

    :param na_row:  a samples row in the distance look up table, the columns
                    represent the other points in the sample population
    :param selection:  the points you want to look at the distances too,
                       represent points in some other cluster
    :return:        returns the minimum distance and the point that relates to it
    """
    mini = na_row[selection].min()
    ret = np.where(na_row == mini)
    ret = ret[0][0]
    return ret, mini
# =========================================================================
# =========================================================================
#                            Usefull math tools
# =========================================================================
# =========================================================================
def mahalanobis_distance(X, mu, cov, is_std=False):
    #print('covariance')
    #print(cov
    xminmu = X - mu
    #print('x - mu')
    #print(xminmu)
    if is_std:
        return np.sqrt((np.dot(xminmu.transpose(), xminmu))/cov)
    return np.sqrt(np.dot(np.dot(xminmu.transpose(), np.linalg.inv(cov)), xminmu))


def euclidean_distance(X, mu, std, is_std=False):
    #print('covariance')
    #print(cov
    xminmu = X - mu
    #print('x - mu')
    #print(xminmu)
    return np.sqrt((np.dot(xminmu.transpose(), xminmu))/std)


def ppm_MSE(ppm1, ppm2):
    # get their covariance matricies
    #print('originals')
    #print(ppm1)
    #print(ppm2)
    cov_1 = pd.DataFrame(ppm1).cov()
    cov_2 = pd.DataFrame(ppm2).cov()
    #print('covariances')
    #print(cov_1.head())
    #print(cov_2.head())
    cov_dif = cov_1 - cov_2
    sum_e = 0
    #for row1, row2 in zip()
    #print('covariance and its square')
    #print(cov_dif.head())
    cov_dif = (cov_dif ** 2)
    #print(cov_dif.head())
    sum = 0
    for row in cov_dif.values:
        sum += row.sum()
    return sum/len(ppm1)


def ppm_AAE(ppm1, ppm2):
    # get their covariance matricies
    #print('originals')
    #print(ppm1)
    #print(ppm2)
    cov_1 = pd.DataFrame(ppm1).cov()
    cov_2 = pd.DataFrame(ppm2).cov()
    #print('covariances')
    #print(cov_1.head())
    #print(cov_2.head())
    cov_dif = cov_1 - cov_2
    sum_e = 0
    #for row1, row2 in zip()

    #print(cov_dif.head())
    cov_dif = np.abs(cov_dif)
    #print(cov_dif.head())
    sum = 0
    for row in cov_dif.values:
        sum += row.sum()
    return sum/len(ppm1)


# =========================================================================
# =========================================================================
#                            Scoring tools
# =========================================================================
# =========================================================================
def bi_score(g, y, vals, classes='', method='accuracy', verbose=False, train=False, retpre=False):
    scores = {'tp':0,
              'fp':0,
              'fn':0,
              'tn':0,}
    posneg = {'bestn':0,
              'bestp':0,
              'predictions':list(g)}

    # go through the guesses and the actual y values scoring
    # * true positives: tp
    # * false positives: fp
    # * true negatives: tn
    # * false negatives: fn
    for gs, ay in zip(g,y.values):
            # check for negative
            if int(gs) == int(vals[0]):
                if int(ay) == int(gs):
                    scores['tn'] += 1
                else:
                    scores['fn'] += 1
            elif int(gs) == int(vals[1]):
                if int(ay) == int(gs):
                    scores['tp'] += 1
                else:
                    scores['fp'] += 1
            else:
                print('Uh Oh!!!!!: {0}'.format(gs))
                print('line number 2461')
                quit(-463)

    posneg['bestn'] = scores['tn']/(scores['fp']+scores['tn'])
    posneg['bestp'] = scores['tp']/(scores['tp']+scores['fn'])

    # calculate and return the overall accuracy
    if method == 'accuracy':
        if retpre:
            accuracy, sum, sensitivity, specificity, precision = viz.show_performance(scores=scores,
                                                                                      verbose=verbose, retpre=retpre)
            posneg['Sensitivity'] = sensitivity
            posneg['Specificity'] = specificity
            posneg['Precision'] = precision
        else:
            accuracy, sum, sensitivity, specificity = viz.show_performance(scores=scores, verbose=verbose,)
            posneg['sen'] = sensitivity
            posneg['spe'] = specificity
        if train:
            return accuracy, scores, posneg
        return accuracy, scores, posneg


# =========================================================================
# =========================================================================
#                Result Recording analysis and documentation
# =========================================================================
# =========================================================================
# *** *** *** *** *** ***     can be used to keep records of tests
class ResultsLog():
    """
        This class can store different results of ML testing
    """
    def __init__(self, result_dict, df_old_log=None, infile_name_old_log=None, sheet_name=None, sort_bys=None,
                 outfile_name_updated_log=None, usecols=None, verbose=False):
        self.result_dict = result_dict
        self.record_name = list(result_dict.keys())           # the names of the attributes
        self.records = list(result_dict.values())             # the values to be added to the log
        self.df_old_log = df_old_log                          # the data frame that contains the logged data if needed, can be left None and will be loaded based on the old log file
        self.infile_name_old_log = infile_name_old_log        # the name of the file to be added to TODO: need to add checker method for file/diretory existence and remove this
        if outfile_name_updated_log is not None:
            self.outfile_name_updated_log=outfile_name_updated_log # TODO: currently neccessary soon to be optional name of new log file if desired
        else:
            self.outfile_name_updated_log=infile_name_old_log # TODO: currently neccessary soon to be optional name of new log file if desired
        self.sheet_name=sheet_name                            # TODO: modify the saving portion to use an excel writer so I can get at specific sheets w/o overwriting the old file
        self.sort_bys=sort_bys                                # optional if you want the log file sorted in a specific way
        self.usecols=usecols                                  #  optional: can select specific columns of the log file to log
        self.verbose=verbose
        if df_old_log is None and (infile_name_old_log is not None):
            self.process_file_name()
        elif df_old_log is not None:
            self.process_df()

    def process_file_name(self,):
        # if the file exists
        if os.path.isfile(self.infile_name_old_log):
            if self.verbose:
                print('The file {} loading...'.format(self.infile_name_old_log))
            if self.usecols is None:
                df_old = pd.read_excel(self.infile_name_old_log)
            else:
                df_old = pd.read_excel(self.infile_name_old_log, usecols=self.usecols)
            df_old = concat_columns(df_old, self.record_name, self.records)
            if self.sort_bys is None:
                df_old.to_excel(self.outfile_name_updated_log, index=False)
            else:
                df_old.sort_values(by=self.sort_bys, inplace=False, ascending=False).to_excel(
                    self.outfile_name_updated_log, index=False)
        # if the file does not exist
        else:
            df_old = pd.DataFrame()
            for p, v in zip(self.record_name, self.records):
                df_old[p] = list([v])
            if self.sort_bys is None:
                df_old.to_excel(self.outfile_name_updated_log, index=False)
            else:
                df_old.sort_values(by=self.sort_bys, inplace=False, ascending=False).to_excel(
                    self.outfile_name_updated_log, index=False)
            if self.verbose:
                print('The file {} created...'.format(self.infile_name_old_log))

        """ 
        if self.sheet_name is None:
            if self.sort_bys is None:
                dumdf = concat_columns(df_old, self.record_name, self.records)
                dumdf.to_excel(self.outfile_name_updated_log, index=False)
            else:
                dumdf = concat_columns(df_old, self.record_name, self.records)
                dumdf.sort_values(by=self.sort_bys, inplace=False, ascending=False).to_excel(self.outfile_name_updated_log, index=False)
        else:
            if self.sort_bys is None:
                # dumdf = concat_columns(df_old, self.record_name, self.records)
                df_old.to_excel(self.outfile_name_updated_log, index=False, sheet_name=self.sheet_name)
            else:
                df_old.sort_values(by=self.sort_bys, inplace=True, ascending=False).to_excel(self.outfile_name_updated_log, index=False, sheet_name=self.sheet_name)
        """

    def process_df(self,):
        if self.sheet_name is None:
            if self.sort_bys is None:
                concat_columns(self.df_old_log, self.record_name, self.records).to_excel(self.outfile_name_updated_log, index=False)
            else:
                concat_columns(self.df_old_log, self.record_name, self.records).sort_values(by=self.sort_bys, inplace=True, ascending=True).to_excel(self.outfile_name_updated_log, index=False)
        else:
            if self.sort_bys is None:
                concat_columns(self.df_old_log, self.record_name, self.records).to_excel(self.outfile_name_updated_log, index=False, sheet_name=self.sheet_name)
            else:
                concat_columns(self.df_old_log, self.record_name, self.records).sort_values(by=self.sort_bys, inplace=True).to_excel(self.outfile_name_updated_log, index=False, sheet_name=self.sheet_name)

# =========================================================================
# =========================================================================
#               TODO: Grid searches
# =========================================================================
# =========================================================================

class GGridSearcher():
    def __init__(self, cmodel, Xtr=None, ytr=None, Xts=None, yts=None, clf=None, param_dict=None, verbose=False,
                 m_type='classifier', make_reports=True, non_prediction=False, attribs=None, model_vars=None,
                 current_model=None, newfile_Per=None, newfile_FI=None, newfile_Re=None, new_tree_png=None):
        if cmodel is not None:
            self.cmodel = cmodel
            self.Xtr = cmodel.X
            self.ytr = cmodel.y
            self.Xts = cmodel.Xts
            self.yts = cmodel.yts
        else:
            self.cmodel = cmodel
            self.Xtr = Xtr
            self.ytr = ytr
            self.Xts = Xts
            self.yts = yts
        self.clf=clf
        self.verbose=verbose
        self.m_type = m_type
        self.param_dict=param_dict
        self.make_reports = make_reports
        self.SkSVCparam_dict = None
        self.GLinRegparam_dict = None
        self.GClstrparam_dict = None
        self.SkKmuparam_dict = None
        self.SkMbKmuparam_dict = None
        self.SKRFparam_dict=None
        self.non_prediction = non_prediction
        self.attribs = attribs
        self.current_model=current_model
        self.model_vars=model_vars
        self.newfile_Per=newfile_Per
        self.newfile_FI=newfile_FI
        self.newfile_Re=newfile_Re
        self.new_tree_png=new_tree_png

    def set_clf(self,clf):
        self.clf=clf
    def set_param_grid(self,param_dict):
        self.param_dict=param_dict
    def set_verbose(self,verbose):
        self.verbose=verbose
    def get_clf(self, ):
        return self.clf
    def get_param_grid(self, ):
        return self.param_dict
    def get_verbose(self, verbose):
        return self.verbose

    def GO(self, report_dict, file=None, sortbys = None, sheet_name=None, usecols=None, ):
        """
            Can be used to run a series of grid search runs for various algorithms.
            The different types are controled by the parameter self.clf
            currently the options are:
                                      * skleranSVC
                                      * sklearnKmu
                                      * sklearnRandomForest

        :param report_dict: a dictionary containing the performance results you want logged
        :param file:  the file you would like to store the report log into
        :param sortbys: the columns you would like to sort the result log by if any
        :param sheet_name: the sheet name of the log file if any TODO: need to create an excel writer method so I can manipulate the sheets
        :param usecols: the columns of the model tested
        :return:
        """
        file = self.newfile_Re
        if self.clf == 'sklearnSVC':
            from sklearn.svm import SVC
            self.SkSVCparam_dict = {'C': [1],
                                    'kernel': ['rbf'],      # kernel type used for algorithm
                                    'degree': [3],          # degree used for polynomial kernel, ignored by all others
                                    'gamma': ['scale'],     # scale (sigma) used in kernel
                                    'coef0': [0],           # bias for poly and sigmoid kernels
                                    'shrk':[True],          # whether to use shrinking huristic
                                    'dsf':['ovr'],          # use one-vs-rest or one vs one (ovo)
                                    'cw':['balanced'],      # weights of different classes (priors)
                                    'prob':[False],         # Whether to enable probability estimates.
                                    'tol':[1e-3],           # Tolerance for stopping criterion.
                                    'max_it':[-1]}         # max allowable iterations
            # change defaults to passed settings
            for pu in self.param_dict:
                if pu in self.SkSVCparam_dict:
                    self.SkSVCparam_dict[pu] = self.param_dict[pu]
            # now do grid search
            C = self.SkSVCparam_dict['C']
            Krnl= self.SkSVCparam_dict['kernel']
            dgr = self.SkSVCparam_dict['degree']
            gma = self.SkSVCparam_dict['gamma']
            co = self.SkSVCparam_dict['coef0']
            shk = self.SkSVCparam_dict['shrk']
            dsf = self.SkSVCparam_dict['dsf']
            clsw = self.SkSVCparam_dict['cw']
            prob = self.SkSVCparam_dict['prob']
            tol = self.SkSVCparam_dict['tol']
            mxit = self.SkSVCparam_dict['max_it']
            for c in C:
                for k in Krnl:
                    for mx in mxit:
                        for s in shk:
                            for df in dsf:
                                for p in prob:
                                    for t in tol:
                                        for cw in clsw:
                                            if k in ['rbf', 'poly', 'sigmoid']:
                                                for g in gma:
                                                    if k in ['poly', 'sigmoid']:
                                                        for coef in co:
                                                            if k is 'poly':
                                                                   for d in dgr:
                                                                       svc_clf = SVC(C=c, kernel=k, degree=d, gamma=g,
                                                                                   coef0=coef, shrinking=s, probability=p,
                                                                                   tol=t, class_weight=cw, max_iter=mx,
                                                                                   decision_function_shape=df)
                                                                       strtm = time.time()
                                                                       svc_clf.fit(self.Xtr, self.ytr.values.flatten())
                                                                       trpast = time_past(strtm)
                                                                       yp = svc_clf.predict(self.Xts)
                                                                       acc, scr, posneg = bi_score(yp, self.yts, vals=[0,1], retpre=True)
                                                                       if self.make_reports:
                                                                           for r in posneg:
                                                                               if r in report_dict:
                                                                                   report_dict[r] = posneg[r]
                                                                           report_dict['C'] = c
                                                                           report_dict['kernel'] = k
                                                                           report_dict['degree'] = d
                                                                           report_dict['gamma'] = g
                                                                           report_dict['coef0'] = coef
                                                                           report_dict['shrk'] = s
                                                                           report_dict['dsf'] = df
                                                                           report_dict['cw'] = cw
                                                                           report_dict['prob'] = p
                                                                           report_dict['tol'] = t
                                                                           report_dict['max_it'] = mx

                                                                           ResultsLog(report_dict,
                                                                                       infile_name_old_log=file,
                                                                                       outfile_name_updated_log=file,
                                                                                       sheet_name=sheet_name,
                                                                                       usecols=usecols,
                                                                                       sort_bys=sortbys)
                                                            else:
                                                                svc_clf = SVC(C=c, kernel=k, gamma=g,
                                                                              coef0=coef, shrinking=s, probability=p,
                                                                              tol=t, class_weight=cw, max_iter=mx,
                                                                              decision_function_shape=df)
                                                                strtm = time.time()
                                                                svc_clf.fit(self.Xtr, self.ytr.values.flatten())
                                                                trpast = time_past(strtm)
                                                                yp = svc_clf.predict(self.Xts)
                                                                acc, scr, posneg = bi_score(yp, self.yts, vals=[0, 1], retpre=True)
                                                                if self.make_reports:
                                                                    for r in posneg:
                                                                        if r in report_dict:
                                                                            report_dict[r] = posneg[r]
                                                                    report_dict['C'] = c
                                                                    report_dict['kernel'] = k
                                                                    report_dict['degree'] = -1
                                                                    report_dict['gamma'] = g
                                                                    report_dict['coef0'] = coef
                                                                    report_dict['shrk'] = s
                                                                    report_dict['dsf'] = df
                                                                    report_dict['cw'] = cw
                                                                    report_dict['prob'] = p
                                                                    report_dict['tol'] = t
                                                                    report_dict['max_it'] = mx
                                                                    report_dict['time'] = trpast

                                                                    ResultsLog(report_dict,
                                                                               infile_name_old_log=file,
                                                                               outfile_name_updated_log=file,
                                                                               sheet_name=sheet_name,
                                                                               usecols=usecols,
                                                                               sort_bys=sortbys)
                                                    else: # when rbf
                                                        for g in gma:
                                                            svc_clf = SVC(C=c, kernel=k, gamma=g,
                                                                          shrinking=s, probability=p,
                                                                          tol=t, class_weight=cw, max_iter=mx,
                                                                          decision_function_shape=df)
                                                            strtm = time.time()
                                                            svc_clf.fit(self.Xtr, self.ytr.values.flatten())
                                                            trpast = time_past(strtm)
                                                            yp = svc_clf.predict(self.Xts)
                                                            acc, scr, posneg = bi_score(yp, self.yts, vals=[0, 1], retpre=True)
                                                            if self.make_reports:
                                                                for r in posneg:
                                                                    if r in report_dict:
                                                                        report_dict[r] = posneg[r]
                                                                report_dict['Accuracy'] = acc
                                                                report_dict['C'] = c
                                                                report_dict['kernel'] = k
                                                                report_dict['degree'] = -1
                                                                report_dict['gamma'] = g
                                                                report_dict['coef0'] = -1
                                                                report_dict['shrk'] = s
                                                                report_dict['dsf'] = df
                                                                report_dict['cw'] = cw
                                                                report_dict['prob'] = p
                                                                report_dict['tol'] = t
                                                                report_dict['max_it'] = mx
                                                                report_dict['time'] = trpast

                                                                ResultsLog(report_dict,
                                                                           infile_name_old_log=file,
                                                                           outfile_name_updated_log=file,
                                                                           sheet_name=sheet_name,
                                                                           usecols=usecols,
                                                                           sort_bys=sortbys)
                                            else:   # if linear
                                                svc = SVC(C=c, kernel=k, )
                                                strtm = time.time()
                                                svc_clf = SVC(C=c, kernel=k, shrinking=s, probability=p,
                                                              tol=t, class_weight=cw, max_iter=mx,
                                                              decision_function_shape=df)
                                                svc_clf.fit(self.Xtr, self.ytr.values.flatten())
                                                trpast = time_past(strtm)
                                                yp = svc_clf.predict(self.Xts)
                                                acc, scr, posneg = bi_score(yp, self.yts, vals=[0, 1], retpre=True)
                                                if self.make_reports:
                                                    for r in posneg:
                                                        if r in report_dict:
                                                            report_dict[r] = posneg[r]
                                                    report_dict['C'] = c
                                                    report_dict['kernel'] = k
                                                    report_dict['degree'] = -1
                                                    report_dict['gamma'] = -1
                                                    report_dict['coef0'] = -1
                                                    report_dict['shrk'] = s
                                                    report_dict['dsf'] = df
                                                    report_dict['cw'] = cw
                                                    report_dict['prob'] = p
                                                    report_dict['tol'] = t
                                                    report_dict['max_it'] = mx
                                                    report_dict['time'] = trpast

                                                    ResultsLog(report_dict,
                                                               infile_name_old_log=file,
                                                               outfile_name_updated_log=file,
                                                               sheet_name=sheet_name,
                                                               usecols=usecols,
                                                               sort_bys=sortbys)
        elif self.clf == 'sklearnKmu':
            from sklearn.cluster import KMeans as kmu
            self.SkKmuparam_dict = {'n_clusters':[2],
                                    'init':['k-means++'],
                                    'n_init':[10],
                                    'max_iter':[300],
                                    'tol':[1e-4],
                                    'algorithm':['auto',]}
            # add user chosen test sets
            for pu in self.param_dict:
                if pu in self.SkKmuparam_dict:
                    self.SkKmuparam_dict[pu] = self.param_dict[pu]
                # now do grid search
            n_clusters = self.SkKmuparam_dict['n_clusters']
            ini = self.SkKmuparam_dict['init']
            nini = self.SkKmuparam_dict['n_init']
            algo = self.SkKmuparam_dict['algorithm']
            tol = self.SkKmuparam_dict['tol']
            mxit = self.SkKmuparam_dict['max_iter']

            for ncl in n_clusters:
                for i in ini:
                    for n in nini:
                        for al in algo:
                            for tl in tol:
                                for mx in mxit:
                                    KM = kmu(n_clusters=ncl, init=i, algorithm=al, tol=tl, max_iter=mx, n_init=n)
                                    strtm = time.time()
                                    KM.fit(self.Xtr, self.ytr)
                                    trpast = time_past(strtm)
                                    yp = KM.predict(self.Xts)
                                    if ncl == 2:
                                        acc, scr, posneg = bi_score(yp, self.yts, vals=[0, 1], retpre=True)
                                    else:
                                        acc = metrics.accuracy_score(self.yts, yp)
                                        posneg = {}
                                        posneg['Sensitivity'] = -999
                                        posneg['Specificity'] = -999
                                        posneg['Precision'] = -999
                                    if self.make_reports:
                                        for r in posneg:
                                            if r in report_dict:
                                                report_dict[r] = posneg[r]
                                        report_dict['Accuracy'] = acc
                                        report_dict['Homogeneity'] = metrics.homogeneity_score(self.yts.values.flatten(), yp)
                                        report_dict['n_clusters'] = ncl
                                        report_dict['init'] = i
                                        report_dict['n_init'] = n
                                        report_dict['algorithm'] = al
                                        report_dict['tol'] = tl
                                        report_dict['max_iter'] = mx
                                        report_dict['time'] = trpast

                                        ResultsLog(report_dict,
                                                   infile_name_old_log=file,
                                                   outfile_name_updated_log=file,
                                                   sheet_name=sheet_name,
                                                   usecols=usecols,
                                                   sort_bys=sortbys)
        elif self.clf == 'sklearnRandomForest':
            nruns = 1
            model_vars = self.model_vars
            current_model = self.current_model
            rl = self.attribs
            self.SKRFparam_dict = {
                                    'n_estimators': [2200],  # how many trees in forest
                                    'max_features': [None],  # maximum number of features to test for split
                                    'criterion': ['entropy'],  # how best split is decided
                                    'max_depth': [None],  # how large trees can grow
                                    'oob_score': [True],  #
                                    'warm_start': [True],
                                    'min_samples_leaf': [1],  # The minimum number of samples required to be at a leaf node
                                    'max_leaf_nodes': [None],
                                    'min_weight_fraction_leaf': [0],  #
                                    'min_samples_split': [2],
                                    'min_impurity_decrease': [0],
                                    'random_state': [None],
                                    'class_weight': [None],
                                    'number of warm runs':1
                                }
            for pu in self.param_dict:
                if pu in self.SKRFparam_dict:
                    self.SKRFparam_dict[pu] = self.param_dict[pu]
            warm_start = self.SKRFparam_dict['warm_start']
            if warm_start:
                self.SKRFparam_dict['n_estimators'] = sorted(self.SKRFparam_dict['n_estimators'])
            nruns = self.SKRFparam_dict['number of warm runs']
            for ne in self.SKRFparam_dict['n_estimators']:
                for crit in self.SKRFparam_dict['criterion']:
                    for mxd in self.SKRFparam_dict['max_depth']:
                        for mln in self.SKRFparam_dict['max_leaf_nodes']:
                            RF_clfstd = RandomForestClassifier(n_estimators=ne, criterion=crit, max_depth=mxd,
                                                               warm_start=True, max_leaf_nodes=mln)
                            best_estimator_fit_stime = time.time()
                            for i in range(nruns):
                                RF_clfstd.fit(self.Xtr, self.ytr)
                            best_estimator_fit_etime = time.time() - best_estimator_fit_stime
                            if self.verbose:
                                print("Fitting the best one took {}".format(best_estimator_fit_etime))
                            feature_impz = RF_clfstd.feature_importances_
                            testing_stime = time.time()
                            ypz = RF_clfstd.predict(self.Xts)
                            testing_etime = time.time() - testing_stime
                            feates = display_significance(feature_impz, rl, verbose=True)
                            scores0 = cross_val_score(RF_clfstd, self.Xts, self.yts, cv=2)
                            avg_scr = scores0.mean()
                            print('The Average score set {0}: {0}'.format(0, avg_scr))
                            # score the models performance and show a confusion matrix for it
                            accuracy, scores, posneg, = bi_score(ypz, self.yts, vals=[0, 1], classes='', retpre=True)
                            nwim = self.new_tree_png
                            tmpim = r'C:\Users\gjone\DeepSolar_Code_Base\tree.dot'
                            if nwim is not None:
                                print('creating')
                                print(nwim)
                                viz.display_DT(RF_clfstd.estimators_[0], rl, ['0','1'], newimg=nwim, tmpimg=tmpim,
                                               precision=2)

                            # pd.DataFrame({'variables':list(feates.keys()), 'Sig':list(feates.values())}).to_excel('RandomForest_Feature_significance_18_{}_.xlsx'.format(get_current_date()))
                            # TODO: below line store in generic time and date stamped file
                            # generate_excel(dic=feates, name='RandomForest_Feature_significance_{}_.xlsx'.format(get_current_date()))

                            if self.verbose:
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                                print('Accuracy: {:.2f}'.format(accuracy))
                                print('Cross val score: {:.3f}'.format(avg_scr))
                                print('Sensitivity:', posneg['Sensitivity'])
                                print('Specificity:', posneg['Specificity'])
                                print('Precision:', posneg['Precision'])
                                viz.show_performance(scores=scores, verbose=True)
                                # print('Training/Testing Split {0}/{1}'.format(tr, ts))
                                print('Training time {}'.format(best_estimator_fit_etime))
                                print('Testing time {}'.format(testing_etime))
                                print('Total time {}'.format(testing_etime + best_estimator_fit_etime))
                                print('Model file ', current_model)
                                print(
                                    '=================================================================================================')
                                print(
                                    '=================================================================================================')
                            # now save the results dummy
                            params_re = {'Accuracy': accuracy,
                                         'Cross_V2': avg_scr,
                                         'Sensitivity': posneg['Sensitivity'],
                                         'Precision': posneg['Precision'],
                                         'Specificity': posneg['Specificity'], 'runs': 0,
                                         'time': testing_etime + best_estimator_fit_etime}
                            # TODO: fix this file and below as well
                            #new_file = '__Data/__Mixed_models/policy/RF_Perf_{}_.xlsx'.format(
                            #    'DeepSolar_Model_2019-12-30_mega')

                            # store the log files if needed
                            if self.make_reports:
                                for r in posneg:
                                    if r in report_dict:
                                        report_dict[r] = posneg[r]

                                report_dict['Accuracy'] = np.around(accuracy, 3)
                                for r in  self.SKRFparam_dict:
                                    report_dict[r] = self.SKRFparam_dict[r]

                                ResultsLog(report_dict,
                                           infile_name_old_log=file,
                                           outfile_name_updated_log=file,
                                           sheet_name=sheet_name,
                                           usecols=usecols,
                                           sort_bys=sortbys)
                                if self.newfile_Per is not None:
                                    pandas_excel_maker(self.newfile_Per, params_re, mode='performance')
                                # RF_FI = 'RF_FI_{}_.xlsx'.format('DeepSolar_Model_2019-12-30_mega'+'_tc{}xc{}tr{}ts{}')
                                # pandas_excel_maker('__Data/__Mixed_models/policy/RF_FI_{}_.xlsx'.format(model_vars),
                                if self.newfile_FI is not None:
                                    pandas_excel_maker(self.newfile_FI,
                                                       params=feates)

def display_significance(feature_sig, features, verbose=False):
    rd = {}
    for s, f in zip(feature_sig, features):
        rd[f] = s

    sorted_rd = dict(sorted(rd.items(), key=operator.itemgetter(1), reverse=True))
    if verbose:
        display_dic(sorted_rd)
    return sorted_rd

def GJ_sklearn_train_test(df, target, trsz=.50, cv=2, rl=None, verbose=True):

    if rl is None:
        rl = rmv_list(df.columns.values.tolist(), target)

    # targets0 = df[target].values.flatten()
    targets0 = df.loc[:, target].values.tolist()
    targets0 = [x[0] for x in targets0]
    print(targets0)
    df = df.loc[:, rl]
    if verbose:
        print(df.describe())
        print()
    ts = .50
    tr = 1 - ts
    # Create training and testing sets for the data
    X_train0, X_test0, y_train0, y_test0 = train_test_split(df, targets0, stratify=targets0, test_size=ts,
                                                            train_size=tr)
    return (X_train0, y_train0), (X_test0, y_test0)

def get_suggested_eta(N, denom=12):
    return N/denom

def get_suggested_perp(N, pct=.01):
    return N * pct

def performance_logger(performance_dict, log_file, verbose=False):
    """
        will store the performance results of some form of testing
    :param performance_dict: dictionary where keys are the metric/parameter, and vals are results
    :param log_file: the file name you want to use to store the results
    :param verbose: how much of the process you want displayed to std out
    :return: None
    """
    # check for file and if not found make it

def process_grid_input():
    lcnq = input("perform lcn?: y/n")
    if lcnq.lower() == 'y':
        lcn_reduce = True  # want to reduce it by correlation filtering?
        gmtc = int(input('minimum target correlation? (-1) for none: '))
        gmxcc = int(input('maximmum predictor cross correlation? (2) for none: '))
    else:
        lcn_reduce = False  # want to reduce it by correlation filtering?
    use_full = input('Use the full model (y) or a select predictor set (n)? (y/n): ')
    if use_full.lower() == 'n':
        use_full = False
        usecols = input('Give me the name of the attrib file: ')
        usecols = pd.read_excel(usecols)['variable'].values.tolist()
    else:
        use_full = True  # do you want to use the full model or select features
        usecols = None     # if allowed to be none will use the drops list
    tssp = float(input('validation set percentage (ex. .50): '))
    current_model = input('Give me the name or path to the model file: ')  # the model file to load
    scaler_ty = 'None'
    #s = 0
    #cv = 3
    #n_est = 2200
    #crit = 'entropy'
    #mx_dth = 20
    #print_tree = True

def load_tree_trunc_features(df=None, dffile=None, limit=.00, verbose=False):
    if df is None:
        df = pd.read_excel(dffile, usecols=['Variable', 'Imp_trunc'])

    df = df.loc[df['Imp_trunc'] >= limit, 'Variable']
    print(list(df))
    return list(df)


def forward_sub2(Train_data, Test_data, feats, clf, verbose=True):
    """performs forward substitution dimension reduction
    :param Train_data: list for X,y of training data
    :param Test_data: list for X,y of testing data
    :param feats: features to test
    :param clf: the classifiery to test, must have a fit method
    :param verbose:
    :return: the list of all vars that lead to increase in performance
    """
    # set up vars
    # need a used up list
    from _products.performance_metrics import calculate_vif, calculate_log_like
    best_scr, BRsqr = 0, 0
    used, good, goodR2, current = list(), list(), list(), list()
    best_R2, BRacc = 0, 0,
    tvar = list(feats[:])
    Rtvar = list(feats[:])
    cadd = None
    better_score = True
    # go through checking each variable one by one
    # subing in values
    while better_score:
        better_score = False
        cadd = None
        # go through each of the remaining vars
        # looking for best result, and adding the one that leads to this
        for var in tvar:
            if var not in good:
                current = good + [var]
                v = clf.fit(Train_data[0].loc[:,current ], Train_data[1])

                if v is not None:
                    print('NEED TO HANDLE THE ISSUE')
                    continue
                # tr_scr = cross_val_score(clf, Train_data[0].loc[:, current], Train_data[1], cv=2).mean()
                ts_scr = clf.score(Test_data[0].loc[:,current], Test_data[1])
                Rsqr = clf.get_Macfadden()
                if verbose:
                    pass
                    #print('current:')
                    #print(current)
                    #print()
                    #print('p-value of {}'.format(var))
                    #print(clf.fitted_model.pvalues[var])
                    #if len(current) > 1:
                    #    vif = calculate_vif(Train_data[0].loc[:,current ])
                    #    print('VIF:\n', vif)
                    #print('# ################################################3')
                    #print('# ################################################3')
                    #print('Anova: ')
                    #print(clf.fitted_model.summary())
                    #print('# ################################################3')
                    #print('# ################################################3')
                if ts_scr > best_scr:
                    if clf.fitted_model.pvalues[var] < .055:
                        better_score=True
                        print(' ******************   p value {:.3f}'.format(clf.fitted_model.pvalues[var]))
                        print(' ******************   New best from {} of {}'.format(var, ts_scr))
                        print(' ******************   Rsquare of {}'.format(Rsqr))
                        best_scr = ts_scr
                        best_R2 = Rsqr
                        cadd = [var]
        if cadd is None:
            print('Good Accuracy list, score: {:.3f}'.format(best_scr))
            print(good)
            #sound_alert_file('sounds/this_town_needs.wav')
            break
        good += cadd
        print('Good is now: score: {}'.format(best_R2))
        #print(good)
        tvar = rmv_list(tvar, cadd[0])

    current = list()
    better_score = True
    while better_score:
        better_score = False
        radd = None
        for var in Rtvar:
            if var not in goodR2:
                current = goodR2 + [var]
                if verbose:
                    pass
                    #print('current list to test:')
                    #print(current)
                    #print()
                clf.fit(Train_data[0].loc[:, current], Train_data[1])
                ts_scr = clf.score(Test_data[0].loc[:, current], Test_data[1])
                # tr_scr = cross_val_score(clf, Train_data[0].loc[:, current], Train_data[1], cv=2).mean()
                if verbose:
                    pass
                    #if len(current):
                    #    vif = calculate_vif(Train_data[0].loc[:,current ])
                    #    print('VIF:\n', vif)
                    #print('# ################################################3')
                    #print('# ################################################3')
                    #print('Anova: ')
                    #print(clf.fitted_model.summary())
                    #print('p-values')
                    #print(clf.fitted_model.pvalues[var])
                    #print('# ################################################3')
                    #print('# ################################################3')
                Rsqr = clf.get_Macfadden()
                if Rsqr > BRsqr and clf.fitted_model.pvalues[var] < .055:
                    # check for significance of model
                    print(clf.fitted_model.pvalues)
                    better_score=True
                    print(' ******************   New best Rsqr {} of {}'.format(var, Rsqr))
                    print(' ******************   Accuracy of {}'.format(ts_scr))
                    print(' ******************   pvalue {:.3f}'.format(clf.fitted_model.pvalues[var]))
                    BRsqr = Rsqr
                    BRacc = ts_scr
                    radd = [var]

        if radd is None:
            better_score=False
            print('Best list for R squared')
            print(goodR2)
            print('Anova: ')
            print(clf.fitted_model.summary())
            sound_alert_file('sounds/this_town_needs.wav')
            break
        goodR2 += radd
        print('GoodR2 is now:')
        print(goodR2)
        Rtvar = rmv_list(Rtvar, radd[0])
    return good, goodR2, [best_scr, best_R2], [BRsqr, BRacc]


def forward_sub(Train_data, feats, clf, cv=2, verbose=False):
    """performs forward substitution dimension reduction
    :param Train_data: list for X,y of training data
    :param Test_data: list for X,y of testing data
    :param feats: features to test
    :param clf: the classifiery to test, must have a fit method
    :param verbose:
    :return: the list of all vars that lead to increase in performance
    """
    # set up vars
    # need a used up list
    best_scr = 0
    used, good, current, acc_l = list(), list(), list(), list()
    acc_inc, best_l = list(), list()
    tvar = list(feats[:])
    cadd = None
    better_score = True
    # go through checking each variable one by one
    # subing in values
    while better_score:
        better_score = False
        cadd = None
        better_score = True
        #best_scr = 0
        # go through each of the remaining vars
        # looking for best result, and adding the one that leads to this
        for var in tvar:
            if var not in good:
                current = good + [var]
                if verbose:
                    print('current:')
                    print(current)
                    print()
                # clf.fit(Train_data[0].loc[:,current ], Train_data[1])
                tr_scr = cross_val_score(clf, Train_data[0].loc[:, current], Train_data[1], cv=cv).mean()
                if tr_scr > best_scr:
                    better_score=True
                    print(' ******************   New best test from {} of {}'.format(var, tr_scr))
                    best_scr = tr_scr
                    cadd = [var]

        if cadd is None:
            print('returning list')
            print(good)
            sound_alert_file('sounds/this_town_needs.wav')
            return good, best_scr, acc_l, acc_inc
        acc_l.append(best_scr)
        if len(good) == 0:
            acc_inc.append(best_scr)
        else:
            acc_inc.append(best_scr - acc_inc[-1])
        good += cadd
        print('Score: {}, Good is now:'.format(best_scr))
        print(good)
        tvar = rmv_list(tvar, cadd[0])
        # print(tvar)
    return good, best_scr, acc_l, acc_inc


