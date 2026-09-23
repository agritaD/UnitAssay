"""The manuscript's figure style, read out of the author's own Illustrator file.

None of this is guessed. `Figure3.ai` is PDF-1.6 inside, so the page size came
from its MediaBox, the type sizes from the `Tf` operators in its content
stream, and the colours were sampled from the artwork. Build at PAGE_W_PT wide
and one matplotlib point is one point on the page, so type lands at the size
the figure was designed in.

This is the single definition; the analysis repo imports it rather than keeping
a copy.
"""
PAGE_W_PT = 643.465                     # 8.937 in
W_IN = PAGE_W_PT / 72.0
LMARG = 6.1 / PAGE_W_PT                 # the .ai's own left margin

SZ = dict(title=26.0, panel=16.1, sub=11.1, lab=9.05, axis=8.2, tick=7.4,
          cap=7.4)

FONT_STACK = ['Arial', 'Liberation Sans', 'Nimbus Sans', 'Helvetica',
              'DejaVu Sans']

# sampled from the artwork
C = dict(red='#BF1E2D', blue='#3953A4', green='#0AAA4B', pink='#E386B7',
         orange='#F58430', purple='#A1238F',
         ink='#14213D', grey='#58595B', faint='#A7A9AC', wash='#EEF2F9',
         rule='#D8DCE6', teal='#12695A', warm='#C1440E', neutral='#9AA3AE')


def apply():
    """rcParams for a figure built at the page's own scale."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',      # text stays text
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.major.size': 2.4,
        'ytick.major.size': 2.4, 'xtick.labelsize': SZ['tick'],
        'ytick.labelsize': SZ['tick'], 'axes.labelsize': SZ['axis'],
        'axes.titlesize': SZ['sub'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.fontsize': SZ['tick'], 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
