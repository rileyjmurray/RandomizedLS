import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
sns.set_theme()
uniform_data = np.random.rand(10, 12)
sns.heatmap(uniform_data)
plt.savefig('2D Plot/Matrix Size/Iteration Number/test')
plt.show()
