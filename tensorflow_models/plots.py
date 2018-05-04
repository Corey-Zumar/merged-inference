import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dense_2_64_vanilla = [0.214895009995]
dense_2_64_comb = [0.210766077042]

dense_2_128_vanilla = [0.139623165131]
dense_2_128_comb = [0.132287025452]

dense_2_512_vanilla = [0.0599920749664]
dense_2_512_comb = [0.0619909763336]

dense_4_64_vanilla = [0.321605920792, 0.319534063339]
dense_4_64_comb = [0.301759004593, 0.299952983856]

dense_8_1_comb = [18.4110701084]
dense_8_1_vanilla = [17.2301671505]

dense_8_2_comb = [14.1395981312]
dense_8_2_vanilla = [17.4567348957]

dense_8_4_vanilla = [8.73798513412]
dense_8_4_comb = [7.07296395302]

dense_8_8_vanilla = [3.928180933]
dense_8_8_comb = [3.8620929718]

dense_8_16_vanilla = [1.96825885773]
dense_8_16_comb = [1.93749785423]

dense_8_32_vanilla = [0.999138832092]
dense_8_32_comb = [0.984143018723]

dense_8_64_comb = [0.504207134247]
dense_8_64_vanilla = [0.520090103149]

dense_8_128_vanilla = [0.334774017334]
dense_8_128_comb = [0.305157184601]

dense_8_256_comb = [0.211450099945]
dense_8_256_vanilla = [0.229965925217]

dense_8_512_vanilla = [0.164115905762]
dense_8_512_comb = [0.159904003143]

dense_16_64_comb = [0.903111934662]
dense_16_64_vanilla = [0.94295501709]

dense_16_512_comb = [0.292978048325]
dense_16_512_vanilla = [0.306090831757]

x1 = (2, 4, 8, 16)
vanilla_dense_64 = []
comb_dense_64 = []
for i in x1:
    l_comb = eval('dense_' + str(i) + '_64_comb')
    comb_dense_64.append(sum(l_comb) / float(len(l_comb)))
    l_vanilla = eval('dense_' + str(i) + '_64_vanilla')
    vanilla_dense_64.append(sum(l_vanilla) / float(len(l_vanilla)))

x2 = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
vanilla_dense_8 = []
comb_dense_8 = []
for i in x2:
    l_comb = eval('dense_8_' + str(i) + '_comb')
    vanilla_dense_8.append(sum(l_comb) / float(len(l_comb)))
    l_vanilla = eval('dense_8_' + str(i) + '_vanilla')
    comb_dense_8.append(sum(l_vanilla) / float(len(l_vanilla)))

# fig, ax = plt.subplots()
# plt.title('MNIST SingleInput,MultipleLayers, Batch Size = 64')
# plt.xlabel('Number of Ensembled Models')
# plt.ylabel('Seconds')
# line1, = ax.plot(x1, comb_dense_64, '-', marker='o', markersize=2, linewidth=1, c='b')
# line2, = ax.plot(x1, vanilla_dense_64, '-', marker='o', markersize=2, linewidth=1, c='r')
# plt.show()
build_hue = ['combined' for _ in range(len(comb_dense_8[1:]))]
x_axis = x2[1:]
points = (np.array(comb_dense_8[1:]) / np.array(vanilla_dense_8[1:])) - np.ones(len(comb_dense_8[1:]))
df = pd.DataFrame(dict(bp_elems=x_axis, attribute=points, model_type=build_hue))
ax = sns.pointplot(x="bp_elems", y="attribute", hue="model_type", data=df)
ax.set(xlabel="Batch Size", ylabel="Latency Improvement (ratio)")
ax.legend_.remove()
# fig, ax = plt.subplots()
# plt.title('MNIST SingleInput,MultipleLayers, N_Ensembles = 8')
# plt.xlabel('Batch Size')
# plt.ylabel('Seconds/Batch')
# line1, = ax.plot(x2[1:4], comb_dense_8[1:4], '-', marker='.', markersize=2, linewidth=1, c='b')
# line2, = ax.plot(x2[1:4], vanilla_dense_8[1:4], '-', marker='.', markersize=2, linewidth=1, c='r')
plt.show()

# fig, ax = plt.subplots()
# plt.title('MNIST SingleInput,MultipleLayers, N_Ensembles = 8')
# plt.xlabel('Batch Size')
# plt.ylabel('Seconds')
# line1, = ax.plot(x2[4:], comb_dense_8[4:], '-', marker='.', markersize=2, linewidth=1, c='b')
# line2, = ax.plot(x2[4:], vanilla_dense_8[4:], '-', marker='.', markersize=2, linewidth=1, c='r')
# plt.show()