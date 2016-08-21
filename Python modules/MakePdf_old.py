# test
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        #fig.set_size_inches(1000./fig.dpi,600./fig.dpi)
        fig.set_size_inches(1200./fig.dpi,900./fig.dpi)
        fig.savefig(pp, format='pdf')
    pp.close()
    
    #plt.figure(figsize=(8, 6), dpi=80)
    
def multipage_longer(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        #fig.set_size_inches(1000./fig.dpi,600./fig.dpi)
        fig.set_size_inches(1600./fig.dpi,1200./fig.dpi)
        fig.savefig(pp, format='pdf')
    pp.close()
