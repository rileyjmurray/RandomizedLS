import numpy as np
import matplotlib.pyplot as plt

# create data
x = [1, 2, 3, 4, 5]
y = [3, 3, 3, 3, 3]


# # plot lines
# plt.plot(x, y, label="line 1", linestyle="-")
# plt.plot(y, x, label="line 2", linestyle="--")
# plt.plot(x, np.sin(x), label="curve 1", linestyle="-.")
# plt.plot(x, np.cos(x), label="curve 2", linestyle=":")
# plt.legend()
# plt.show()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

for i in np.arange(2):
    for j in np.arange(2):
        axes[i, j].set_xlabel('m/1000')
        axes[i, j].set_ylabel('log10(condition number)')
        axes[i, j].set_title('log10(Iteration number) of ')
        axes[i, j].plot(x+i, y+j, label="line 1", linestyle="-")

plt.legend()
plt.show()
