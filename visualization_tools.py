import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.use_inf_as_na = True
from _products.utility_fnc import *

def gaussian_plot(xarrays, mus, stds, priors=[1,1], verbose=False):
    for xarray, mu, std, prior in zip(xarrays, mus, stds, priors):
        Visualizer().basic_plot(xarray, generate_gaussian(xarray, mu, std, prior, verbose), xlabel='x', ylabel='prob',
               title='test gaussian', show=False, fig_num=1, m_label=[['a'], ['b']], legend=True)
    plt.show()

class Visualizer:
    """ a lot of visualization methods
        There are:
                 ploting methods:
                    * dict_bar_plotter(): uses a dict to make a bar plot
                    *
                 stdout put methods
                    * print_test_params: takes a dictionary of paramter names and values and prints them to stdout
    """
    def print_test_params(self, param_d):
        print('Test Parameters:')
        for p in param_d:
            print('                 * {0}{1}'.format(p, param_d[p]))
        return

    def dict_bar_plotter(self, bar_dict, xlabel='Number of Hidden Neurons', ylabel='Time to train seconds',
                         title='Time to Complete for different Hidden neurons', save_fig=False, fig_name=''):
        y_pos = np.arange(len(bar_dict))
        bar_dict = sort_dict(bar_dict)
        performance = bar_dict.values()
        lables = list(bar_dict.keys())

        plt.barh(y_pos, performance, align='center', alpha=0.5)
        plt.yticks(y_pos, lables)
        plt.xlabel(ylabel)
        plt.ylabel('Number of hidden Neurons')
        plt.title(title)
        if save_fig:
            plt.savefig(fig_name)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
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
        rd = {'Accuracy':overall_acc, 'Sensitivity':sensitivity,
              'Precision':precision, 'Specificity':specificity, 'CM':cm}
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        rd['ax'] = ax
        return rd

    def basic_plot(self, x, y, xlabel='xlabel', ylabel='ylabel', title='K value vs accuracy',
                   marker='x', show=False, fig_num=None, m_label=[''], legend=False):
        # artis for this plot
        art = None
        if fig_num is None:
            plt.figure()
        elif fig_num == 'ignore':
            pass
        else:
            plt.figure(fig_num)
        art = plt.plot(x,y,marker)
        #plt.scatter(x,y,color=color, marker=marker,label=m_label)
        if legend:
            plt.legend([m_label])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if show:
            plt.show()
        return art[0]

    def basic_plot_scatter(self, x, y, color='r', xlabel='xlabel', ylabel='ylabel', title='K value vs accuracy',
                   marker='x', show=False, fig_num=None, m_label=''):
        if fig_num is None:
            plt.figure()
        elif fig_num == 'ignore':
            pass
        else:
            plt.figure(fig_num)
        #plt.plot(x,y,color=color, marker=marker,label=m_label)
        plt.scatter(x,y,color=color, marker=marker,label=m_label)
        lgd = plt.legend(loc='best')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if show:
            plt.show()

    def sub_plotter(self, xarray, yarray, xlabels, ylabels, titles, markers, sharex='none', sharey='none', show=False,
                    fig_num=None, orientation='v'):
        # set up the subplot arrays using the
        # length of xarray
        num_plots = len(xarray)
        if orientation == 'v':
            fig, axs = plt.subplots(nrows=num_plots, ncols=1, sharex=sharex, sharey=sharey)
        else:
            fig, axs = plt.subplots(nrows=1, ncols=num_plots, sharex=sharex, sharey=sharey)


        for i in range(num_plots):
            axs[i].plot(xarray[i], yarray[i])
            axs[i].set_xlabel(xlabel=xlabels[i])
            axs[i].set_ylabel(ylabel=ylabels[i])
            axs[i].set_title(titles[i])
        if show:
            plt.show()

    def multi_plot(self, xarray, yarray, xlabel='x label', ylabel='y label',
                            title='MULTIPLOT TITLE', fig_num=None, legend_array=['me','you'], marker_array=['x', 'x'], show=False,
                            show_last=False, save=False, fig_name='Fig'):
        found = False
        l = len(xarray)
        cnt = 0
        arts = list()
        for x, y, m, la in zip(xarray, yarray, marker_array, legend_array):
            if fig_num is None and not found:
                fig_num = plt.figure().number
            #print('Fig num',fig_num)
            if show_last:
                if cnt < l-1:
                    a = self.basic_plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                        m_label=[la], marker=m, show=False)
                    arts.append(a)
                else:
                    a = self.basic_plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                    m_label=legend_array, marker=m, show=True, legend=True)
                    arts.append(a)
                cnt += 1
            else:
                a = self.basic_plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                m_label=[la], marker=m, show=False)
                arts.append(a)
        lgd = plt.legend(arts, legend_array, loc='best')
        if save:
            plt.savefig(fig_name)
        plt.show()
        return fig_num


    def multi_plot_scatter(self, xarray, yarray, color_array=['r', 'b'], xlabel='x label', ylabel='y label',
                            title='MULTIPLOT TITLE', fig_num=None, legend_array=['me','you'], marker_array=['x', 'x'], show=False,
                            show_last=False):
        found = False
        l = len(xarray)
        cnt = 0
        for x, y, c, la, m in zip(xarray, yarray, color_array, legend_array, marker_array):
            if fig_num is None and not found:
                fig_num = plt.figure().number
            #print('Fig num',fig_num)
            if show_last:
                if cnt < l-1:
                    self.basic_plot_scatter(x=x, y=y, color=c, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                    m_label=la, marker=m, show=False)
                else:
                    self.basic_plot_scatter(x=x, y=y, color=c, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                    m_label=la, marker=m, show=True)
                cnt += 1
            else:
                self.basic_plot_scatter(x=x, y=y, color=c, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                m_label=la, marker=m, show=show)
        return fig_num


    def bi_class_colored_scatter(self, x, y, class_dict, fig_num=None, legend=['class 0', 'class 1'], annotate=False, show=True,
                                 xl='x', yl='y', title='title'):
        for X, Y in zip(x,y):
            plt.scatter(X[0], X[1],  c=class_dict[Y])
        plt.title(title)
        plt.xlabel(xl)
        plt.ylabel(yl)
        leg = plt.legend(legend, loc='best', borderpad=0.3, shadow=False, markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        if show:
            plt.show()


    def bi_class_scatter3D(self, x, y, class_dict, fig_num=None, legend=['class 0', 'class 1'], annotate=False, show=True, treD=False,
                           xl = 'x', yl='y', zl='z', cols=(0, 1, 2), title='3D Class Scatter'):

        a = cols[0]
        b = cols[1]
        c = cols[2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for X, Y in zip(x, y):
            ax.scatter(X[a], X[b], X[c], c=class_dict[Y])
        plt.legend(['non adopters', 'adopters'])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_zlabel(zl)
        plt.title(title)
        plt.show()

    def fancy_scatter_plot(self, x, y, styl, title, c, xlabel, ylabel, labels, legend,
                           annotate=True, s=.5, show=False):

        for z1, z2, label in zip(x, y, labels):
            plt.scatter(z1, z2, s=s, c=c)
            if annotate:
                plt.annotate(label, (z1, z2))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        leg = plt.legend([legend], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)

        if show:
            plt.show()

    def make_prop_o_var_plot(self, s, num_obs, threshold=.95, show_it=True, last_plot=True):

        sum_s = sum(s.tolist())

        ss = s ** 2

        sum_ss = sum(ss.tolist())

        prop_list = list()

        found = False

        k = 0

        x1, y1, x2, y2, = 0, 0, 0, 0
        p_l, i_l = 0, 0
        found = False

        for i in range(1, num_obs + 1):
            perct = sum(ss[0:i]) / sum_ss
            # perct = sum(s[0:i]) / sum_s

            if np.around((perct * 100), 0) >= threshold*100 and not found:
                y2 = perct
                x2 = i
                x1 = i_l
                y1 = p_l
                found = True
            prop_list.append(perct)
            i_l = i
            p_l = perct

        if np.around(y2, 2) == .90:
            k_val = x2
        else:
            print('it is over 90%', x2)
            #vk_val = line_calc_x(x1, y1, x2, np.around(y2, 2), .9)

        single_vals = np.arange(1, num_obs + 1)

        if show_it:
            fig = plt.figure(figsize=(8, 5))
            plt.plot(single_vals, prop_list, 'ro-', linewidth=2)
            plt.title('Proportion of Variance, K should be {:d}'.format(x2))
            plt.xlabel('Eigenvectors')
            plt.ylabel('Prop. of var.')

            p90 = prop_list.index(y2)

            # plt.plot(k_val, prop_list[p90], 'bo')
            plt.plot(x2, prop_list[p90], 'bo')

            leg = plt.legend(['Eigenvectors vs. Prop. of Var.', '90% >=  variance'],
                             loc='best', borderpad=0.3,shadow=False, markerscale=0.4)
            leg.get_frame().set_alpha(0.4)
            #leg.draggable(state=True)

            if last_plot:
                plt.show()

        return x2


    def Groc(self, tpr, tnr):
        self.basic_plot(1-tnr, tpr)

    def gaussian_plot(self, xarrays, mus, stds, priors=[1, 1], verbose=False):
        for xarray, mu, std, prior in zip(xarrays, mus, stds, priors):
            Visualizer().basic_plot(xarray, generate_gaussian(xarray, mu, std, prior, verbose), xlabel='x',
                                    ylabel='prob',
                                    title='test gaussian', show=False, fig_num=1, m_label=[['a'], ['b']], legend=True)
        plt.show()
    # ================================================================================
    # ================================================================================
    # ======                           std out methods                  ==============
    # ================================================================================
    # ================================================================================
    def string_padder(self,str='What Up Yo!', pstr=' ', addstr='Just Added', padl=20, right=True):
        if right:
            return str + '{:{}>{}s}'.format(addstr, pstr, padl)
        return str + '{:{}<{}s}'.format(addstr, pstr, padl)

    def border_maker(self, item, bsize=35):
        rs = ''
        for i in range(bsize):
            rs += item
        return rs

    def border_printer(self, border, padl=2):
        for i in range(padl):
            print(border)

    def create_label_string(self, label, border, lpad=4, lpstr=' ', b_size=35):
        # calculate border left over
        rpd = self.border_maker(lpstr, lpad)
        label = rpd + label + rpd
        b_left_over = b_size - len(label)
        if b_left_over%2 == 0:
            bleft = int(b_left_over/2)
            bright = int(b_left_over/2)
        else:
            bleft = int(np.around((b_left_over/2), 0))-1
            bright = int(np.around(b_left_over/2, 0))

        #return self.string_padder(str=border[0:bleft-(len(label))], pstr=lpstr, addstr=label, padl=lpad,
        return border[0:bleft] + label + border[0:bright]

    def block_label(self, label, lpad=4, lpstr=' ', border_marker=None, border_size=35, bpadl=2):
        if border_marker is not None:
            border =self.border_maker(border_marker, bsize=border_size)
            self.border_printer(border, padl=bpadl)
        else:
            border = self.border_maker('=', bsize=border_size)
            self.border_printer(border, padl=bpadl)

        print(self.create_label_string(label, border, lpad=lpad, lpstr=lpstr, b_size=border_size))

        if border_marker is not None:
            self.border_printer(self.border_maker(border_marker, bsize=border_size), padl=bpadl)
        else:
            self.border_printer(self.border_maker('=', bsize=border_size), padl=bpadl)

    def display_significance(self, feature_sig, features, verbose=False):
        """Takes """
        rd = {}
        for s, f in zip(feature_sig, features):
            rd[f] = s

        sorted_rd = dict(sorted(rd.items(), key=operator.itemgetter(1), reverse=True))
        if verbose:
            display_dic(sorted_rd)
        return sorted_rd

    def show_performance(self, scores, verbose=False, retpre=False):
        """displays a confusion matrix on std out"""
        true_sum = scores['tp'] + scores['tn']
        false_sum = scores['fp'] + scores['fn']
        sum = true_sum + false_sum

        # do this so we don't divde by zero
        tpfp = max(scores['tp']+scores['fp'], .00000001)
        tpfn = max(scores['tp']+scores['fn'], .00000001)
        precision = scores['tp']/tpfp
        recall = scores['tp']/tpfn
        accuracy = true_sum / sum
        #                 probability ot a true positive
        sensitivity = scores['tp'] / (scores['tp'] + scores['fn'])
        #                 probability ot a true negative
        specificity = scores['tn'] / (scores['tn'] + scores['fp'])
        if verbose:
            print('=====================================================')
            print('=====================================================')
            print('             |  predicted pos   |   predicted neg   |')
            print('----------------------------------------------------')
            print(' actual pos  |   {:d}            |   {: 3d}            |'.format(scores['tp'], scores['fn']))
            print('----------------------------------------------------')
            print(' actual neg  |   {:d}            |   {:d}            |'.format(scores['fp'], scores['tn']))
            print('-------------------------------------------------------------------')
            print('                                        Correct  |   {:d}'.format(true_sum))
            print('                                          Total  | % {:d}'.format(sum))
            print('                                                 | ------------------------')
            print('                                       Accuracy  | {:.2f}'.format(accuracy))
            print('                                      Precision  | {:.2f}'.format(precision))
            #print('                                         recall  | {:.2f}'.format(recall))
            print('                                    Sensitivity  | {:.2f}'.format(sensitivity))
            print('                                    Specificity  | {:.2f}'.format(specificity))
            print('=======================================================================================')
        if retpre:
            return accuracy, sum, sensitivity, specificity, precision

        return accuracy, sum, sensitivity, specificity


    def show_image(self, filename):
        """
            Can be used to display images to the screen
        :param filename:
        :return:
        """
        img = mpimg.imread(filename)
        plt.imshow(img)
        plt.show()

    def display_DT(self, estimator, features, classes, newimg='tree.png', tmpimg='tree.dot', precision=2):
        from sklearn.tree import export_graphviz
        import io
        import pydotplus
        #graph = Source(export_graphviz(estimator, out_file=None
        #                                    , feature_names=features, class_names=['0', '1']
        #                                    , filled=True))
        #display(SVG(graph.pipe(format='svg')))
        # plot_tree(estimator, filled=True)
        # plt.show()
        # return

        # Export as dot file
        export_graphviz(estimator, out_file=tmpimg,
                                   feature_names=features,
                                   class_names=classes,
                                   rounded=True, proportion=False,
                                   precision=3, filled=True)
        #from subprocess import call
        #call(['dot', '-Tpng', tmpimg, '-o', newimg, '-Gdpi=600'])
        # os.system('dot -Tpng {} -o {}, -Gdpi=600'.format(tmpimg, newimg))
        # Display in python
        #import matplotlib.pyplot as plt

        # Draw graph
        #graph = graphviz.Source(dot_data)
        #dotfile = io.StringIO()
        graph = pydotplus.graph_from_dot_file(tmpimg)
        graph.write_png(newimg)
        print(graph)

        # Convert to png using system command (requires Graphviz)

        # plt.figure(figsize=(14, 18))
        # plt.imshow(plt.imread(newimg))
        # plt.axis('off')
        # plt.show()

        #from subprocess import call
        #os.system('dot -Tpng tmpimg -o newimg, -Gdpi=600')
        #self.show_image(newimg)




