import numpy as np
import matplotlib.pyplot as plt

font_ticks = {'size': 17, 'family': 'Arial'}
font_label = {'size': 20, 'family': 'Arial'}


def plot_RSI2013(ax=None):
    """
    de Loubens C, Lentle RG, Love RJ, Hulls C, Janssen PWM. 2013 Fluid mechanical consequences of pendular activity,
    segmentation and pyloric outflow in the proximal duodenum of the rat and the guinea pig.
    J R Soc Interface 10: 20130027.
    http://dx.doi.org/10.1098/rsif.2013.0027
    """
    if ax is None:
        plt.rc('font', **font_ticks)
        ax = plt.axes()
        external_ax = False
    else:
        external_ax = True
    a0 = 1
    a1 = 0.9
    t = np.linspace(0, 6, 100)
    strain = -a0 * t * np.exp(-t / a1)
    ax.plot(t, strain)
    if not external_ax:
        ax.set_xlabel('Time (s)', fontdict=font_label)
        ax.set_ylabel('Strain', fontdict=font_label)
        plt.tight_layout()
        plt.show()


def plot_JP2004(ax=None):
    """
    Gwynne RM, Thomas EA, Goh SM, Sjövall H, Bornstein JC.
    Segmentation induced by intraluminal fatty acid in isolated guinea-pig duodenum and jejunum.
    J Physiol. 2004 Apr 15;556(Pt 2):557-69. doi: 10.1113/jphysiol.2003.057182.
    """
    import pandas as pd
    if ax is None:
        plt.rc('font', **font_ticks)
        ax = plt.axes()
        external_ax = False
    else:
        external_ax = True
    data = pd.read_csv('2004-JP-Fig2B.csv', names=['t', 'D']).sort_values('t')
    ax.plot('t', 'D', '', data=data)
    if not external_ax:
        ax.set_xlabel('Time (s)', fontdict=font_label)
        ax.set_ylabel('Diameter (mm)', fontdict=font_label)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # plot_RSI2013()
    plot_JP2004()
