from abc import ABC, abstractmethod
import sys
import os
import gzip
import shutil
import struct
import pandas as pd
import numpy as np
from datetime import datetime
import array
import time
from math import *
import operator
from _products.DeepSolarModels import *
from scipy.stats import truncnorm
pd.options.mode.use_inf_as_na = True
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold,cross_val_score, train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

# =========================================================
# =========================================================
#          TODO:          Generic Methods
# =========================================================
# =========================================================
def type_check(tocheck, against='dataframe'):
    """
        will return true or false based of if the given object is of the same type as the against string argument
    :param tocheck:  object to check
    :param against: type of object you want to see if it matches options are:
                    * dataframe
                    * string
                    * float
                    * int
                    * numpy for numpy array
                    * list
                    * dict
    :return: boolean
    """
    if against == 'dataframe':
        return type(tocheck) == type(pd.DataFrame([0]))
    elif against == 'string':
        return type(tocheck) == type(str('s'))
    elif against == 'float':
        return type(tocheck) == type(float(0))
    elif against == 'dict':
        return type(tocheck) == type(dict())
    elif against == 'numpy':
        return type(tocheck) == type(np.array([0]))
    elif against == 'int':
        return type(tocheck) == type(int(0))
    elif against == 'list':
        return type(tocheck) == type(list())

# =========================================================
# =========================================================
#         TODO:           Dictionary Methods
# =========================================================
# =========================================================

def sort_dict(dic, sort_by='vals', reverse=False):
    """
        Returns a sorted version of the given dictionary
    :param dic: dictionary to sort
    :param sort_by: 'vals' to sort by values, 'keys', to sort by keys
    :param reverse: set to to True to get largest to smallest
    :return:
    """
    if sort_by == 'vals':
        return dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=reverse))
    elif sort_by == 'keys':
        return dict(sorted(dic.items(), key=operator.itemgetter(0), reverse=reverse))

def display_dic2lvl(dic):
    for k in dic:
        print('-------   {0}   ------:'.format(k))
        display_dic(dic[k])
        print()

def display_dic(dic):
    for k in dic:
        print('{0}:'.format(k))
        print('--- {0}'.format(dic[k]))
# =========================================================
# =========================================================
#         TODO:           List Methods
# =========================================================
def rmv_list(l, r):
    del l[l.index(r)]
    return l

def rmv_list_list(l, rl):
    for r in rl:
        l = rmv_list(l, r)
    return l

def show_list(x):
    cnt = 1
    for l in x:
        print('{}: {}'.format(cnt, l))
        cnt += 1
    return

def show_labeled_list(x, labels):
    for l,label in zip(x, labels):
        print('{:s}: {}'.format(label, l))
    return

def make_repeated_list(n, v):
    return list([v]*n)

# =========================================================
# =========================================================
# =========================================================
#        TODO:            pandas methods
# =========================================================
# =========================================================
def fix_dataset(dataset, option=1):
    if option == 1:
        dataset.replace(-999, np.NaN)
        return dataset.dropna(axis=0)

def select_model(filename, attrib_file=None, attrib_list=None):
    '''Allows user to select from a data set only those variables they are interested in
    :param filename: data file that holds the larger set
    :param attrib_file: if desired can use an excel file to load the desired variables
                        Save a list of variables into an xlsx file with a column header
                        of Variables. Then pass the name of this file. The list will be
                        used to build the model. Will throuw an error if the data file
                        does not have column headers for these variables. If only passed
                        the name of works just like a regular pandas read from excel or csv
                        Looks for an xlsx or csv file.
    :param attrib_list: A list of the variables desired in model
    :return: a data frame of the data with only the desired variables
    '''
    if filename[-4:] == '.csv':
        if attrib_file is None and attrib_list is None:
            return pd.read_csv(filename)
        else:
            df = process_csv(filename=filename, attrib_file=attrib_file, attrib_list=attrib_list)
    elif filename[-5:] == '.xlsx':
        if attrib_file is None and attrib_list is None:
            return pd.read_excel(filename)
        else:
            return process_xlsx(filename=filename, attrib_file=attrib_file, attrib_list=attrib_list)

def process_csv(filename, attrib_file, attrib_list=None):
    if attrib_file is not None:
        attribs = pd.read_excel(attrib_file).loc['Variables'].values.tolist()
        return pd.read_csv(filename, usecols=attribs)
    elif attrib_list is not None:
        return pd.read_csv(filename, usecols=attrib_list)

def process_xlsx(filename, attrib_file, attrib_list=None):
    if attrib_file is not None:
        attribs = pd.read_excel(attrib_file).loc[:,'Variables'].values.tolist()
        print(attribs)
        return pd.read_excel(filename, usecols=attribs)
    elif attrib_list is not None:
        return pd.read_excel(filename, usecols=attrib_list)

def create_variable_file(df, new_name):
    """
        This can be used to store the attributes of a selected model in an excel file
        Then the file can be used to select the those variables again for a training model
    :param df: dataframe of current model
    :param new_name: name of variable file
    :return:
    """
    attribs = df.columns.values.tolist()
    pd.DataFrame({'Variables':attribs}).to_excel(new_name)
    return

def data_merger2(data_sets, joins=('fips', 'FIPS', 'geoid'), target=None, verbose=False, drop_joins=False,):
    """This method can be used to merge a set of data frames using a shared
       data column. the first argument is a list of the dataframes to merge
       and the second argument is a list of the column labels used to perform the merge
       TODO: some work needs to be done for error checking
       TODO: add more flexibility in how the merge is perfomed
       TODO: make sure the copy rows are removed
    :param data_sets: a list of data frames of the data sets that are to be joined
    :param joins: a list of the column labels used to merge, the labels should be in the s
                  same order as the data frames for the method to work. Right now this works
                  best if the label used is the same for all. This makes sured the duplicate
                   columns are not created.
    :param verbose: at this point does nothing but can be used to inform user of what
                    has occured
    :return: a reference to the new merged dataframe
    """

    cnt = 0
    for df in range(1,len(data_sets)):
       data_sets[0] =  data_sets[0].merge(data_sets[df], left_on=joins[0], right_on=joins[df], how='left')
       if verbose:
        print(data_sets[0].columns)

       if drop_joins and ((joins[0] + '_x') in data_sets[0].columns.values.tolist() or (
               (joins[0] + '_y') in data_sets[0].columns.values.tolist()) or ((joins[1] + '_y') in data_sets[0].columns.values.tolist())):
           pass
           #data_sets[0].drop(columns=[(joins[0]+'_y'), (joins[0]+'_y')], inplace=True)
       if  (target is not None and ((target + '_x') in data_sets[0].columns.values.tolist() or (
               (target + '_y') in data_sets[0].columns.values.tolist()))):
           data_sets[0][target] = data_sets[0].loc[:, target + '_x']
           data_sets[0].drop(columns=[(target + '_x'), (target + '_y')], inplace=True)
    if drop_joins:
        data_sets[0].drop(columns=list(joins), inplace=True)
    return data_sets[0]



def percentage_generator(df, part, total):
    """    will calculate the percentage
           of the total value part is in a list
    :param df:
    :param part:
    :param total:
    :return:
    """
    return list(df[part]/df[total])


def check_cols(col1, col2):
    """
        Takes two dictionaries and based on the columns of some data frame and
        checks to see if the column headers are the same
    :param col1:
    :param col2:
    :return:
    """
    col1, col2 = col1.keys(), col2.keys()
    if len(col1) != len(col2):
        print('the length of the two lists do not match 1: {}, 2: {}'.format(len(col1), len(col2)))
        return 420
    cl = list()
    cnt = 0
    for f1, f2 in zip(col1, col2):
        if f1 != f2:
            cl.append((cnt,f1,f2))
        cnt += 1
    if len(cl) == 0:

        print('The two lists match.' )
        return 0
    print('The two lists do not match!!')
    print("Below is a list of the number of the non matching columns\nwith the index and values")
    print(cl, '\nwas not found in both lists')
    return 1, cl

def report_var_stats(df, name, saveit=True, sort_type=None, sort_list=[], ascending=True, axis=0,
                     re_nan=(-999,), verbose=False):
    """Creates an excel file containing:
                    * missing counts for each variable
                    * the range for each variable
                    * mean for each variable
                    * standard deviation for each variable
                    * the range for each variable
                    * TODO: need to add skew to table
    :param df:   The data frame containing the data to add to report
    :param name: The name of the new file
    :param saveit: if true report will be saved under given name
    :param sort_type: options are:
                      * 'index' for row sorting
                      * 'columns' for column sorting
                      * None (default) for no sorting
    :param sort_list: the list of columns or indices to sort by, empty will just do lex sort
    :return: returns the newly created data frame used
    """
    # add given list of nan representations
    for re in re_nan:
        df.replace(re, np.nan)
    # grab stat statistices
    descr = df.describe()
    # grab total number of entries
    if verbose:
        print('-------         There are {:d} entries in the set         -------')
    N = len(df)
    # set the indices to that of the given data frame dummy
    dfskew = df.skew()
    print('skew index\n',dfskew.index)
    print('given df index\n',df.index)
    rdic = {'Missing':[], 'Range':[], 'Mean':[], 'std':[], 'Skew':[]}
    #rdic = {'Missing':[], 'Range':[], 'Mean':[], 'std':[]}
    for var in descr.columns.values.tolist():
        rdic['Missing'].append(N-descr.loc['count',var])
        rdic['Range'].append([np.around(descr.loc['min', var], 4), np.around(descr.loc['max', var],4)])
        rdic['Mean'].append(descr.loc['mean',var])
        rdic['std'].append(descr.loc['std',var])
        rdic['Skew'].append(dfskew.loc[var])
    # create data from from created dictionary
    rdf = pd.DataFrame(rdic, index=descr.columns.values.tolist())
    if sort_type is not None:
        if sort_type == 'value':
            rdf.sort_values(by=sort_list, axis=axis, inplace=True, ascending=ascending)
        elif sort_type == 'index':
            rdf.sort_index(axis=axis, inplace=True, ascending=ascending)
    if saveit:
        rdf.to_excel(name)
    return rdf

def concat_columns(df, cols, datas, verbose=False):
    rdf = {}
    for col, data in zip(cols, datas):
        if verbose:
            print('col',col)
            print('data',data)
        rdf[col] = df[col].values.tolist()
        rdf[col].append(data)
    rdfdf = pd.DataFrame(rdf)
    if verbose:
        print('return df', rdf)
    return rdfdf

def concat_col(df, col, data, verbose=False):
    ldf = df[col].values.tolist()
    dl = [data]
    if verbose:
        print('data frame \n', ldf,'\ndata\n', dl)
    return ldf + dl

def data_merger(data_sets, joins=('fips', 'FIPS', 'geoid'), target=None, verbose=False, drop_joins=False,):
    """This method can be used to merge a set of data frames using a shared
       data column. the first argument is a list of the dataframes to merge
       and the second argument is a list of the column labels used to perform the merge
       TODO: some work needs to be done for error checking
       TODO: add more flexibility in how the merge is perfomed
       TODO: make sure the copy rows are removed
    :param data_sets: a list of data frames of the data sets that are to be joined
    :param joins: a list of the column labels used to merge, the labels should be in the s
                  same order as the data frames for the method to work. Right now this works
                  best if the label used is the same for all. This makes sured the duplicate
                   columns are not created.
    :param verbose: at this point does nothing but can be used to inform user of what
                    has occured
    :return: a reference to the new merged dataframe
    """

    cnt = 0
    if len(data_sets) == 1:
        return data_sets[0]
    for df in range(1,len(data_sets)):
       data_sets[0] =  data_sets[0].merge(data_sets[df], left_on=joins[0], right_on=joins[df], how='left')
       if verbose:
        print(data_sets[0].columns)

       if (joins[0] + '_x') in data_sets[0].columns.values.tolist() or (
               (joins[0] + '_y') in data_sets[0].columns.values.tolist()):
           data_sets[0].drop(columns=[(joins[0]+'_x'), (joins[1]+'_y')], inplace=True)
       if target is not None and ((target + '_x') in data_sets[0].columns.values.tolist() or (
               (target + '_y') in data_sets[0].columns.values.tolist())):
           data_sets[0][target] = data_sets[0].loc[:, target + '_x']
           data_sets[0].drop(columns=[(target + '_x'), (target + '_y')], inplace=True)
    if drop_joins:
        data_sets[0].drop(columns=list(joins), inplace=True)
    return data_sets[0]

def generate_excel_descending_list_dic(dic, headers):
    h1l, h2l = list(), list()
    for h1 in dic:
        h1l.append(h1)
        h2l.append(dic[h1])
    return {headers[0]:h1l,
            headers[1]:h2l}

def generate_excel(dic=None, df=None, name='df_excel2.xlsx', index=False):
    """
    The method will generate an excel file of the given name from either a given dictionary
    or a given data fram
    :param dic: a dictionary that will be converted to an data
                frame and then writen to an excel file
    :param df: data frame to write to file
    :param name:
    :return:
    """
    if dic is None:
        df.to_excel(name, index=index)
    else:
        df =  pd.DataFrame(dic)
        df.to_excel(name, index=index)
    return

def create_combo_var_sum(df, list_to_sum):
    return df.loc[:, list_to_sum].sum(axis=1).values.tolist()

def add_renewable_gen(df, val, dictl):
    df['Ren'] = list([0]*len(df))
    for st in dictl:
        df.loc[df[val] == st, 'Ren'] = dictl[st]
    return df

def store_var_ranges(df, vars):
    """
        takes a data frame and the variables you want to find the ranges of and returns a new data frame
        with one column of the variables and the other the corresponding varialbes range
    :param df:
    :param vars:
    :return:
    """
    var_stats = df.describe()
    var = list()
    ranges = list()
    for v in vars:
        var.append(v)
        ranges.append('[{0}, {1}]'.format(var_stats.loc['min', v], var_stats.loc['max', v]))
    return pd.DataFrame({'Variable':var, 'Original Range':ranges})


def recode_var_sub(sought, check, keyd):
    """
        will create a list of recoded variables based on a list of substrings(sought) that will be
        searched for in the check list, useing the recode map keyd
    :param sought:
    :param check:
    :param keyd:
    :return:
    """
    rl = list()
    for c in check:
        for substr in sought:
            print(substr)
            print(c)
            if pd.isna(c):
                print('bad c!',c)
                rl.append(np.nan)
                break
            elif substr in c:
                print(c)
                print(substr)
                rl.append(keyd[substr])
                break
    return rl

def load_model_attribs(filename, colname='Variables'):
    """
        Loads a set of features from a given excel file
    :param filename:
    :param colname:
    :return:
    """
    return pd.read_excel(filename).loc[:,colname].values.tolist()


def thresh_binary_recode(df, var, valthresh=0):
    bin_re = list([0]*df.shape[0])
    print('new list is of size {}'.format(len(bin_re)))
    df[var + '_bin'] = bin_re
    df.loc[df[var] > valthresh, var + '_bin'] = 1

def generate_mixed(df, vars, mix_name):
    df[mix_name] = df[vars[0]].values.tolist()
    for v in range(1, len(vars)):
        df[mix_name] = (df[mix_name].values * df[vars[v]].values).tolist()

def shuffle_deck(deck):
    np.random.shuffle(deck.values)


# =========================================================
# =========================================================
#       TODO:             list methods
# =========================================================
# =========================================================


# =========================================================
# =========================================================
#      TODO:              I/O methods
# =========================================================
# =========================================================
def read_line_binary(f, b=1, ignore=None, stop=b'\n'):
    ch = ''
    line = ''
    while ch != stop:
        ch = f.read(b).decode('utf-8')
        line += ch
    return line

def process_ppm(file, verbose=False):
    f = open(file, 'rb')
    #magic_number = f.read(1)
    magic_number = read_line_binary(f, b=1, stop='\n').strip().split()
    second_line = read_line_binary(f, b=1, stop='\n').strip().split()
    width = int(second_line[0])
    height = int(second_line[1])
    third_line = read_line_binary(f, b=1, stop='\n').strip().split()
    max_val = int(third_line[0])
    data_samples = list()
    if verbose:
        print('magic number:', magic_number)
        print('width:', width)
        print('height:', height)
        print('Max value:',max_val)
    original_header = {'magic_number': magic_number[0],
                       'width':width,
                       'height':height,
                       'Max_value':max_val}
    for h in range(height*int(width)):
        sample = list([])
        for i in range(3):
            next = struct.unpack('B', f.read(1))
            if next in (b' ', b'\t'):
                continue
            sample.append(next[0])
        data_samples.append(sample)

    return pd.DataFrame(data_samples, columns=['r', 'g', 'b'], dtype=np.int), original_header

def write_ppm(file, data, header_dict, verbose=False):
    """This will write a ppm to the given file name/destination
       using the header_dict, to save the magic_number, width,
       height, max size and then the data (pandas data frame or numpy
       array) to save the pixels (red, green, blue) of the ppm
    :param file: the name or destination\name that you want to save the new ppm as.
                 must end in .ppm to work as one
    :param data: a pandas data frame or numpy array that stores the pixel rgb values
                 in each row.
    :param header_dict: dictionary storing the following ppm header information
                        header_dict['magic_number'] = P3 for ascii file and P6 for binary
                        header_dict['width'] = number of columns of the ppm
                        header_dict['height] = number of rows of the ppm
                        header_dict['Max_size] = the maximum pixel value
                                                 this is used to determing if there are 1
                                                 byte r/g/b values (max <=256) or if 2 byte red, and green,
                                                 and blue values(max > 256).
    :param verbose:     used for debugging
    :return:
    """
    mn = header_dict['magic_number']
    w = header_dict['width']
    h = header_dict['height']
    mx = header_dict['Max_value']
    # make a string for the header information
    ppm_header = f'{mn}\n{w} {h}\n{mx}\n'

    f = open(file, 'wb')
    # write the header to file
    f.write(bytearray(ppm_header, 'utf-8'))

    cnt = 0
    print('length of values', len(data.values))
    for v in data.values:
        pckr = struct.pack('B', v[0])
        pckg = struct.pack('B', v[1])
        pckb = struct.pack('B', v[2])
        f.write(pckr)
        f.write(pckg)
        f.write(pckb)
        cnt += 1
    print('there were {} pixels written'.format(int(cnt)))
    f.close()

def write_ppm2(file, data, header_dict, verbose=False):
    import codecs
    mn = header_dict['magic_number']
    w = header_dict['width']
    h = header_dict['height']
    mx = header_dict['Max_value']
    # magic number
    ppm_header = f'P6\n{w} {h}\n{mx}\n'
    sender = []
    for v in data.values.tolist():
        sender += v
    print(len(sender))
    image = array.array('B', sender)
    #f.write(bytearray(ppm_header, 'ascii'))
    #with codecs.open(file, 'w', 'utf-8-sig') as f:
    #    f.write(ppm_header)
    #    image.tofile(f)
    #quit(-171)
    #f = codecs.open(file, 'w', 'utf-8')
    f = open(file, 'wb')

    f.write(bytearray(ppm_header, 'ascii'))
    #f.write(bytes(str(header_dict['Max_value'])+'\n', encoding='ascii'))
    cnt = 0
    print('length of values', len(data.values))
    for v in data.values:
        #f.write(bytes(str(' '), encoding='ascii'))
        #for p in v:
        pckr = struct.pack('B', v[0])
        pckg = struct.pack('B', v[1])
        pckb = struct.pack('B', v[2])

        #f.write(bytes(str(v[0]), encoding='utf-8'))
        #f.write(bytes(str(' '), encoding='ascii'))
        #f.write(bytes(str(v[1]), encoding='utf-8'))
        #f.write(bytes(str(' '), encoding='ascii'))
        #f.write(bytes(str(v[2]), encoding='utf-8'))
        f.write(pckr)
        #f.write(bytearray(str(v[0]), encoding='utf-8'))
        #f.write(bytes(str(' '), encoding='ascii'))
        f.write(pckg)
        #f.write(bytearray(str(v[1]), encoding='utf-8'))
        # f.write(bytes(str(' '), encoding='ascii'))
        f.write(pckb)
        #f.write(bytearray(str(v[2]), encoding='utf-8'))
        #f.write(bytes(str(' '), encoding='ascii'))
        #if cnt < len(data)-1:
        #    f.write(bytes(str('\n'), encoding='ascii'))
        #f.write(bytes(str(' '), encoding='ascii'))
        cnt += 1
        #if cnt%(int(header_dict['width'])) == 0:
        #    f.write(bytes(str('\n'), encoding='ascii'))
        #else:
        #    pass
            #f.write(bytes(str(' '), encoding='ascii'))
    #f.write(bytes(str('\n'), encoding='ascii'))
    print(cnt)
    f.close()



# =========================================================
# =========================================================
#   TODO: file manipulation and scripting methods
# =========================================================
# =========================================================
def test_runs(exe_file, numruns=1, r1=None, r2=None):
    if numruns is not None:
        for i in range(numruns):
            os.system('python {}'.format(exe_file))
    else:
        print('range')
        for i in range(r1,r2):
            print('python {}'.format(exe_file + ' ' + str(i)))
            os.system('python {}'.format(exe_file + ' '+ str(i)))

def find_dir_parent(path, run=0, filename=None, ret_val=1):
    """
        Can be used to find the file name and parent directory of a given path
    :param path:
    :param run:
    :param filename:
    :param ret_val:
    :return:
    """
    for i in range(-1, -len(path), -1):
        if path[i] == '/':
            if run != ret_val:
                return find_dir_parent(path[:i], run=run+1, filename=path[i+1:], ret_val=ret_val)
            else:
                return (filename, path[i+1:])
    return (path)


def dir_maker(dir_name):
    """
        Can be used to check for and if needed create a directory
    :param dir_name:
    :return:
    """
    import os
    if os.path.isdir(dir_name):
        print("File exist")
        return
    else:
        print("File not exist")
        os.system('mkdir {}'.format(dir_name))
        return

def pandas_excel_maker(new_file, params, mode='feature', init=False):
    """
        Can be used check for and create and excel file for data frames
    :param new_file:
    :param params:
    :param mode:
    :return:
    """
    import os
    if os.path.isfile(new_file):
        print("File exist")
        update_mode_handler(mode, new_file, params)
        return 0
    else:
        print("File not exist")
        init_mode_handler(mode, new_file, list(params.keys()))
        update_mode_handler(mode, new_file, params)
        return 1


def init_mode_handler(mode, new_file, params):
    """
        Will handle the initialization of a file
    :param mode:
    :param new_file:
    :param params:
    :return:
    """
    if mode == 'feature':
        print('feature init')
        init_feat_importance(new_file, params)
    elif mode == 'performance':
        print('performance init')
        init_performance_log(new_file, params)

def update_mode_handler(mode, new_file, params):
    """
            Will handle the updating of log files
        :param mode:
        :param new_file:
        :param params:
        :return:
        """
    if mode == 'feature':
        print('feats')
        update_feat_importance(new_file, params)
    elif mode == 'performance':
        print('performance')
        update_performance_log(new_file, params)
    return


def init_feat_importance(filename, params):
    """
        Used to initialize a feature importance average file
        hopefully will be used to get an average of the placement of a feature
        from sum number of runs and get a better feel for where the importance truly lies
    :param filename: new file to initilize
    :param params: list of the variables in the feature list
    :return:
    """
    n_dict = init_params(params, ori='v')
    ps = list(n_dict.keys())
    vals = list(n_dict.values())
    run_count = make_repeated_list(len(params), 0)
    pd.DataFrame({'Variable':ps, 'Avg_Importance':vals, 'runs':run_count}).to_excel(filename, index=False)
    print("made the file {}?".format(filename))
    return
def init_performance_log(filename, params):
    """
        Used to initialize a performance log average file
        hopefully will be used to get an average of a set of test runs
    :param filename: new file to initilize
    :param params: list of metrics and parameters to store
    :return:
    """
    n_dict = init_params(params, ori='h')
    print(n_dict)
    #ps = list(n_dict.keys())
    #vals = list(n_dict.values())
    #ps += 'runs'
    #vals += [0]
    pd.DataFrame(n_dict).to_excel(filename, index=False)
    return

def init_params(params, val=0, ori='h'):
    """ Used to initialize a dictionary with the params as keys and zeros as values"""
    n_dict = {}
    for p in params:
        if ori == 'h':
            n_dict[p] = [val]
        if ori == 'v':
            n_dict[p] = val
    return n_dict

def update_feat_importance(filename, params_updates):
    feature_file = pd.read_excel(filename, index_col='Variable')
    N = feature_file['runs'].values[0] +1
    replc_runs = list([N]* len(feature_file))
    feature_file['runs'] = list([N]* len(feature_file))
    if N > 1:
        N = 2
    print('Dividing by {}'.format(N))
    for p in params_updates:
        if p != 'runs':
            feature_file.loc[p, 'Avg_Importance'] = np.around((feature_file.loc[p, 'Avg_Importance'] + params_updates[p])/ N, 3)
    feature_file.sort_values(by=['Avg_Importance'], ascending=False).to_excel(filename, index=True)


def update_performance_log(filename, params_updates):
    feature_file = pd.read_excel(filename)
    N = feature_file.loc[0, 'runs'] + 1
    feature_file['runs'] = list([N] * len(feature_file))
    if N > 1:
        N = 2
    print('Dividing by {}'.format(N))
    for p in params_updates:
        print('params {}, val  {}'.format(p, params_updates[p]))
        if p != 'runs':
            feature_file[p] = list((feature_file[p] + params_updates[p]) / N)
        #feature_file[p] = list((feature_file[p].values[0] + params_updates[p]) / N)
    feature_file.to_excel(filename, index=False)


def gen_RF_param_file(file, params=None, index=False):
    """
        Generates a random forest parameter file that can be used to load a set of parameters
        into a random forest analysis
    :param file: name/name and path you want to save the file to
    :param params: parameter dictionary with the names and values (as a list) that you want to use
    :param index: determines if the index of the data frame will be saved
    :return: None
    """
    param_grid0 = {
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
        'class_weight': [None]
    }
    if params is not None:
        for p in params:
            param_grid0[p] = params[p]
    pd.DataFrame(param_grid0).to_excel(file, index=index)
    return
# =======================================================
# =========================================================
#        TODO: Image work
# =========================================================
# =========================================================







# =========================================================
# =========================================================
#        TODO:   timing and date methods
# =========================================================
# =========================================================
def today_is():
    """this returns the current data and time"""
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
def get_current_date(time_only=True, type='s'):
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if time_only:
        if type == 's':
            return str(current_time).replace(':','_',5)
        elif type == 'n':
            return current_time
    return 'Current Time ={}'.format(current_time).replace(':',':',5)
def get_seconds_past(start):
    return time.time() - start
def get_minutes_past(start):
    return (time.time() - start)/60
def get_hours_past(start):
    return (time.time() - start)/(60*60)
def time_past(start):
    delta = time.time() - start
    if delta <= 60:
        return delta, 'seconds'
    elif delta > 60 and delta <= 60**2:
        return get_minutes_past(start), 'mins'
    else:
        return get_hours_past(start), 'hours'
def how_long(start, rtn=False):
    if rtn:
        return time_past(start)
    tm, mes = time_past
    return tm

# =========================================================
# =========================================================
#        TODO:     audio cue methods
# =========================================================
# =========================================================
def sound_alert_beep(frq, dur):
    import winsound
    frequency = frq*1000  # Set Frequency To 2500 Hertz
    duration = dur*1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def sound_alert_file(filename):
    import winsound
    winsound.PlaySound(filename, winsound.SND_FILENAME)

def sound_player_playsound(file):
    from playsound import playsound
    playsound(file)

def blocking_sound_player(filename):
    import sounddevice as sd
    import soundfile as sf
    # Extract data and sampling rate from file
    data, fs = sf.read(filename, dtype='float32')
    sd.play(data, fs)
    status = sd.wait()  # Wait until file is done playing


# =========================================================
# =========================================================
#          TODO:       function methods
# =========================================================
# =========================================================

def gsummation(zipped, func):
    return sum([func(a,b) for a,b in zipped])

def Nx_gaussian(x, mu, sig, prior=1, verbose=False):
    """Returns the result of a gaussian operation
    :param x: input sample
    :param mu: mean
    :param sig: std
    :param prior: prior probability
    :param verbose: nothing yet just a habit for debugging
    :return: the values of the gausian pdf operation
    """
    return ((1 / (sig * sqrt(2 * np.pi)))) * np.exp((-(x - mu) ** 2) / (2 * sig ** 2))*prior

def generate_gaussian(xarray, mu, sig, prior=None, verbose=False):
    return [Nx_gaussian(x,mu,sig, prior, verbose=verbose) for x in xarray]

# converts give aray to a rounded integer version
def get_rounded_int_array(dta):
    return np.array(np.around(dta, 0), dtype=np.int)

# generates a gausian random number from the given statistics
def get_truncated_normal(mean=128, sd=1, low=0, upp=255):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)