import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window):
    weights = np.repeat(1, window)/window
    return np.convolve(data, weights, mode='valid')
''''''
''''''
sav = np.genfromtxt('sav_default.csv', delimiter=',')
sav = np.reshape(sav, [21, 11, 2])
sav_partial1 = sav[:, :, 0]
sav_partial2 = sav[:, :, 1]
fig1 = plt.figure(num=3)
ax10 = fig1.add_subplot(1, 2, 1)
ax11 = fig1.add_subplot(1, 2, 2)
im10 = ax10.imshow(sav_partial1)
im11 = ax11.imshow(sav_partial2)

'''
fe_exp = np.genfromtxt('feature_expectations.csv', delimiter=',')
fe_exp_expert = fe_exp[0, :-1]
fe_exp_expert = fe_exp_expert.reshape([21, 11, 2])
fig00 = plt.figure(num=4, figsize=(5, 5))
ax000 = fig00.add_subplot(1, 2, 1)
ax001 = fig00.add_subplot(1, 2, 2)
im000 = ax000.imshow(fe_exp_expert[:, :, 0])
im001 = ax001.imshow(fe_exp_expert[:, :, 1])
'''
reward = np.genfromtxt('reward_default.csv', delimiter=',')
reward = np.reshape(reward, [21, 11, 2])

reward_partial1 = reward[:, :, 0]
reward_partial2 = reward[:, :, 1]
fig0 = plt.figure(num=2,figsize=(5, 5))
ax00 = fig0.add_subplot(1, 2, 1)
ax = fig0.add_subplot(1, 2, 2)
im00 = ax00.imshow(reward_partial1)
imm = ax.imshow(reward_partial2)

''''''
#success_scatter0 = moving_average(np.genfromtxt('sc_scatter_defaultval0.csv', delimiter=','), 300)
success_scatter1  = moving_average(np.genfromtxt('sc_scatter_defaultval0.csv', delimiter=',')[:, 1], 10)
#success_scatter2 = moving_average(np.genfromtxt('sc_scatter_ALval0.csv', delimiter=',')[:, 0], 300)
#success_scatter3 = moving_average(np.genfromtxt('AL_results/state and action/iter1/sc_scatter_ALval0.csv', delimiter=','), 300)
#success_scatter4 = moving_average(np.genfromtxt('AL_results/state and action/iter2/sc_scatter_ALval0.csv', delimiter=','), 300)
#success_scatter5 = moving_average(np.genfromtxt('AL_results/state and action/iter3/sc_scatter_ALval0.csv', delimiter=','), 300)

fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)

plt.title('Success Rate Comparison')
plt.xlabel('iteration')
plt.ylabel('success rate')
#ax.scatter(np.arange(0, len(success_scatter0)), np.asarray(success_scatter0),s=3, label='reward 3')
ax.plot(np.arange(0, len(success_scatter1)), np.asarray(success_scatter1), label='default reward')
#ax.plot(np.arange(0, len(success_scatter2)), np.asarray(success_scatter2), label='iter0')
#ax.plot(np.arange(0, len(success_scatter3)), np.asarray(success_scatter3), label='iter1')
#ax.plot(np.arange(0, len(success_scatter4)), np.asarray(success_scatter4), label='iter2')
#ax.plot(np.arange(0, len(success_scatter5)), np.asarray(success_scatter5), label='iter3')


ax.legend()
ax.grid()

plt.show()


