from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.set_size_inches(1000./fig.dpi,600./fig.dpi)
        fig.savefig(pp, format='pdf')
    pp.close()