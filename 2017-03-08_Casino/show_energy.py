import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data

d = get_data('results_2nm.dat')
#d = get_data('test.dat')

# calculate distance from center line (x, y, z) -> (r, z)

no_of_trajectories = 10000
dist = np.array([])
#for n in range(len(d)):
for n in range(no_of_trajectories):
    x = d[n][:, 0]
    y = d[n][:, 1]
    z = d[n][:, 2]
    en = d[n][:, 6]
    r = np.sqrt( x**2 + y**2 )

    try:
        den = np.gradient(en)
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        dl = np.sqrt(dx**2+dy**2+dz**2)

        en_grad = den/dl
        
    except:
        print("couldn't calculate gradient, replacing with zero")
        en_grad = 0 * en

    hlp = np.transpose(np.array([z, r, en, en_grad]))
    
    dist = np.vstack((dist, hlp)) if dist.size else hlp

all_z = dist[:, 0]
all_r = dist[:, 1]
all_en = dist[:, 2]
all_diff_en = dist[:, 3]




import pandas as pd

nz = 250
nr = 500

hall_z, bin_z = pd.cut(all_z, nz, retbins = True)
hall_r, bin_r = pd.cut(all_r, nr, retbins = True)

hall_en = pd.Series(all_diff_en)
g = hall_en.groupby([hall_z.codes, hall_r.codes])

result = np.zeros([nz, nr])
for k, gp in g:
             #print 'key=' + str(k)
             #print gp

             #result[k] = np.mean(np.array(g.get_group(k)))
             result[k] = np.sum(np.array(g.get_group(k)))/no_of_trajectories

#plt.pcolor(bin_r, bin_z, result)
#plt.xlabel('radius')
#plt.ylabel('z')
#plt.colorbar()
#
##plt.xlim(0, 800)
##plt.ylim(0, 1000)
#
#plt.figure()
#plt.plot(all_r, all_z)
#
#plt.figure()
#plt.plot(all_z, all_en)

plt.show()

np.savez('data_2nm', bin_r, bin_z, result, all_r, all_z)


