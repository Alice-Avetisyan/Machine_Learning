import matplotlib.pyplot as plt
import numpy as np

dot_dict = {'a': np.arange(50),
            'c': np.random.randint(0, 50, 50),
            'd': np.random.randn(50)}

dot_dict['b'] = dot_dict['a'] + 10 * np.random.randn(50)
dot_dict['d'] = np.abs(dot_dict['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=dot_dict)
plt.xlabel('Entry a')
plt.ylabel('Entry b')
#plt.show()

plt.savefig('fig.png', bbox_inches='tight')