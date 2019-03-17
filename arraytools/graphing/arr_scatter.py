# -*- coding: UTF-8 -*-
"""
===========
arr_scatter
===========

Script : arr_scatter.py

Author : Dan_Patterson@carleton.ca

Modified : 2019-03-07

Purpose:
--------
    Sample scatterplot plotting, in 2D and 3D

Notes
-----
    >>> print(plt.style.available)
    >>> import matplotlib.pyplot.figure as fig
    # figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None,
    #        frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>,
    #        clear=False, **kwargs)
    # matplotlib.pyplot.subplots
    # subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
    #          subplot_kw=None, gridspec_kw=None, **fig_kw)

References
----------

`<https://matplotlib.org/users/customizing.html>`_.

`<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html>`_.

`<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html>`_.

"""
# ---- imports, formats, constants ----

import sys
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

__all__ = ['plot_2d', 'plot_3d']

# ---- functions ----
#
def plot_2d(pnts, title='Title', r_c=False, ax_lbls=None, pnt_labels=True):
    """Plot points for Nx2 array representing x,y or row,col data.
    
    Parameters
    ----------
    see _params() to specify special parameters

    pnts : array-like
        2D array of point-like objects ie a row/column array
    r_c : boolean
        If True, the y-axis is inverted to represent row-column formatting
        rather than x,y formatting.
    ax_lbls : list
        A list, like ['X', 'Y'] is needed, if left to `None` then that will
        be the default.

    Returns
    -------
    A scatterplot representing the data.  It is easier to modify the
    script below than to provide a whole load of input parameters.

    """
    def scatter_params(plt, fig, ax, title="Title", ax_lbls=None):
        """Default parameters for plots
        :Notes:
        :  ticklabel_format(useoffset), turns off scientific formatting
        """
        fig.set_figheight = 8
        fig.set_figwidth = 8
        fig.dpi = 200
        if ax_lbls is None:
            ax_lbls = ['X', 'Y']
        x_label, y_label = ax_lbls
        font1 = {'family': 'sans-serif', 'color': 'black',
                 'weight': 'bold', 'size': 14}  # set size to other values
        ax.set_aspect('equal', adjustable='box')
        ax.ticklabel_format(style='sci', axis='both', useOffset=False)
        ax.xaxis.label_position = 'bottom'
        ax.yaxis.label_position = 'left'
        plt.xlabel(x_label, fontdict=font1, labelpad=12, size=14)
        plt.ylabel(y_label, fontdict=font1, labelpad=12, size=14)
        plt.title(title + "\n", loc='center', fontdict=font1, size=20)
        plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
        plt.grid(True)
        return
    #
    def label_pnts(pnts, plt):
        """as it says"""       
        lbl = np.arange(len(pnts))
        for label, xpt, ypt in zip(lbl, pnts[:, 0], pnts[:, 1]):
            plt.annotate(label, xy=(xpt, ypt), xytext=(2, 2), size=12,
                        textcoords='offset points', ha='left', va='bottom')
        return
    # ---- main plotting routine
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    markers = MarkerStyle.filled_markers
    # ---- set basic parameters ----
    scatter_params(plt, fig, ax, title, ax_lbls)
    if isinstance(pnts, (list, tuple)):
        mn = np.min([i.min(axis=0) for i in pnts], axis=0) - [0.5, 0.5]
        mx = np.max([i.max(axis=0) for i in pnts], axis=0 )+ [0.5, 0.5]
        x_min, y_min = np.floor(mn)
        x_max, y_max = np.ceil(mx)
    else:
        x_min, y_min = np.floor(pnts.min(axis=0) - [0.5, 0.5])
        x_max, y_max = np.ceil(pnts.max(axis=0) + [0.5, 0.5])
    # 
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if r_c:
        plt.ylim(y_max, y_min)
    # ---- enable multiple point files ----
    if isinstance(pnts, (list, tuple)):
        for i, p in enumerate(pnts):  # plot x, y using marker i.
            plt.scatter(p[:, 0], p[:, 1], marker=markers[i])
            if pnt_labels:
                label_pnts(p, plt)
    else:
        plt.scatter(pnts[:, 0], pnts[:, 1])  # , marker=markers[0])
        if pnt_labels:
            label_pnts(pnts, plt)
#    plt.ion()
    plt.show()


def plot_3d(a):
    """Plot an xyz sequence in 3d

    Parameters
    ----------
    a : array-like
        A 3D array of point objects representing X,Y and Z values

    References
    ----------
    `<https://matplotlib.org/tutorials/toolkits/mplot3d.html#sphx-glr-
    tutorials-toolkits-mplot3d-py>`_.

    Example
    -------
    >>> x = np.arange(10)
    >>> y = np.arange(10)
    >>> z = np.array([5,4,3,2,1,1,2,3,4,5])
    >>> xyz = np.c_[(x,y,z)]
    >>> plot_3d(xyz)
    """
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    #
    mpl.rcParams['legend.fontsize'] = 10
    #
    fig = plt.figure()
    fig.set_figheight = 8
    fig.set_figwidth = 8
    fig.dpi = 200
    ax = Axes3D(fig)  # old  #ax = fig.gca(projection='3d')
    #
    x = a[:, 0]
    y = a[:, 1]
    z = a[:, 2]
    ax.plot(x, y, z, label='xyz')
    ax.legend()
    #plt.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
    plt.show()

# ----------------------------------------------------------------------------
# ---- running script or testing code section ----
def _demo():
    """Plot 20 points which have a minimum 1 unit point spacing
    :
    """
    a = np.array([[0.4, 0.5], [1.2, 9.1], [1.2, 3.6], [1.9, 4.6],
                  [2.9, 5.9], [4.2, 5.5], [4.3, 3.0], [5.1, 8.2],
                  [5.3, 9.5], [5.5, 5.7], [6.1, 4.0], [6.5, 6.8],
                  [7.1, 7.6], [7.3, 2.0], [7.4, 1.0], [7.7, 9.6],
                  [8.5, 6.5], [9.0, 4.7], [9.6, 1.6], [9.7, 9.6]])
    plot_2d(a, title='Points no closer than... test',
               r_c=False, ax_lbls=None, pnt_labels=True)
    return a

# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    a, plt, ax = _demo()
