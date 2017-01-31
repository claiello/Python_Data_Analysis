import numpy as np
import matplotlib.pyplot as plt
import my_fits

#x = np.array([25.7, 27.8, 30.5, 44.4, 50.5, 74.6, 94.9])
#x = np.array([0.000567346,0.000685339,0.00154156, 0.00222674])   #np.average in V
x1 = np.array([0.000553752, 0.000683432, 0.0015373, 0.00219519]) #AVG of FIRST FRAME OF Jan 05 data
x2 = np.array([0.00296101,0.00257478,]) # AVG of FIRST FRAME OF Jan 13 data


x = np.concatenate([x1,x2])
#delta = np.array([-0.046e-3, 0.238e-3, 0.156e-3, 0.225e-3, 0.359e-3, 0.505e-3, 0.660e-3])
delta = np.array([0.225e-3, 0.359e-3, 0.505e-3, 0.660e-3])
delta2 = np.array([+0.240e-3,+0.229e-3])

(a, b) = my_fits.linear_fit(x, delta)

plt.plot(x, delta, 'o-')
plt.plot(x, a*x+b, '-')
plt.show()

print(a)
print(b)

