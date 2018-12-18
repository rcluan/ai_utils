import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan
from ai_utils.metrics import fac2

name = 'graphics'


def plot_f2_single_graphic(y_pred, y_true, figname=None, figsize=(12, 9), print_fot=False,
                           x_label='', y_label='', title='', length=2000, s=1):
    '''
    @param y_pred: numpy array with predict data.
    @param y_true: numpy array with real data.
    @param figname: figure name to save.
    @param figsize: tuple that represents size of figure.
    @param print_fot: boolean to indicate if factor of two metric should be printed or not.
    @param x_label: label to put in x axis.
    @param y_label: label to put in y axis.
    @param title: title of the figure.
    @param length: line limits of factor of two metric.
    @param s: size of points.
    '''

    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    co_angle = 1
    end_y = length * sin(atan(co_angle))
    end_x = length * cos(atan(co_angle))
    plt.plot([0, end_x], [0, end_y], 'k')

    co_angle = 2
    end_y = length * sin(atan(co_angle))
    end_x = length * cos(atan(co_angle))
    plt.plot([0, end_x], [0, end_y], '--k')

    co_angle = 0.5
    end_y = length * sin(atan(co_angle))
    end_x = length * cos(atan(co_angle))
    plt.plot([0, end_x], [0, end_y], '--k')

    plt.scatter(y_pred, y_true, s=s)

    max_x = y_pred.max()
    max_y = y_true.max()
    if max_x < max_y:
        max_x = max_y

    plt.axis([0, max_x, 0, max_x])

    if print_fot:
        f2 = fac2(y_true, y_pred, to_numpy=True)
        plt.text(0, max_x, ('Fator de 2: %.2f%%' % (f2 * 100)).replace(".",","), fontsize=8)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname, bbox_inches='tight')


def plot_f2_multiple_graphic(to_figure, figname=None, figsize=(12, 9), print_fot=False,
                             x_label='', y_label='', title='', length=2000, s=1):
    '''
    @param to_figure: matrix containing data pairs to be compared.
    @param figname: figure name to save.
    @param figsize: tuple that represents size of figure.
    @param print_fot: boolean to indicate if factor of two metric should be printed or not.
    @param x_label: label to put in x axis.
    @param y_label: label to put in y axis.
    @param title: title of the figure.
    @param length: line limits of factor of two metric.
    @param s: size of points.
    '''
    to_figure = np.array(to_figure)

    plt.figure(figsize=figsize)
    nrows, ncols, qtd_dados, qtd_elementos = to_figure.shape
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)

    for i in range(nrows):
        for j in range(ncols):
            y_one = to_figure[i][j][0]
            y_two = to_figure[i][j][1]
            axes[i, j].set_title(title)
            axes[i, j].scatter(y_one, y_two, s=s)
            axes[i, j].set(xlabel=x_label, ylabel=y_label)

            co_angle = 1
            end_y = length * sin(atan(co_angle))
            end_x = length * cos(atan(co_angle))
            axes[i, j].plot([0, end_x], [0, end_y], 'k')

            co_angle = 2
            end_y = length * sin(atan(co_angle))
            end_x = length * cos(atan(co_angle))
            axes[i, j].plot([0, end_x], [0, end_y], '--k')

            co_angle = 0.5
            end_y = length * sin(atan(co_angle))
            end_x = length * cos(atan(co_angle))
            axes[i, j].plot([0, end_x], [0, end_y], '--k')

            max_x = y_pred.max()
            max_y = y_true.max()
            if max_x < max_y:
                max_x = max_y

            axes[i, j].axis([0, max_x, 0, max_x])

            if print_fot:
                f2 = fac2(y_true, y_pred, to_numpy=True)
                axes[i, j].text(0, max_x, ('Fator de 2: %.2f%%' % (f2 * 100)).replace(".",","), fontsize=8)

            if figname is None:
                plt.show()
            else:
                plt.savefig(figname, bbox_inches='tight')
