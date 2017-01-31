import numpy as np
import pandas
import tables

#filename = 'boe.hdf5'
#
#
#x = pandas.HDFStore(filename, 'r')
#print x
#for df in x.select('data', chunksize = 10):
#    print df
#

x = tables.openFile('2017-01-05-1452_ImageSequence__250.000kX_10.000kV_30mu_10.hdf5', 'r')
 
d = x.getNode('/data/Counter channel 1 : PMT red/PMT red time-resolved TRANSIENT/data')

#85.254857142857148

res = 0
for k in range(1400):
    print k
    res += np.mean(d[:, k, :, :])
    print res

print res

res2 = np.mean(d)

print res - res2

