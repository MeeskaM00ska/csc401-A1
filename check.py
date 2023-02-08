import numpy as np

sample_data = np.load('sample_feats.npz')
result_data = np.load('result.npz')

x = sample_data['arr_0']
y = result_data['arr_0']
z = (x[4] == y[4])

for i in range(len(z)):
    print(f'{i}  {z[i]}')



