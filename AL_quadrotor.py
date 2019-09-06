from sklearn.svm import SVC
import numpy as np
from rl_3D import RL_Multi
from rewardconstruct import FeatureExpectation, Rewardconstruct
from os.path import isfile
from os import remove
from demo_discrete import Discretize


def reward_reconstruct(filename):
    FEA_EXP = FeatureExpectation()
    R = Rewardconstruct()

    demo_traj = np.genfromtxt('demodata.csv', delimiter=',')
    dis = Discretize(data=demo_traj)
    demo_traj = dis.discretize_data()

    demo_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=demo_traj)
    if isfile(filename):
        agent_traj = np.genfromtxt(filename, delimiter=',')
        dis.data = agent_traj
        agent_traj = dis.discretize_data()
        agent_feature_exp, _ = FEA_EXP.featureexpectations(trajectories=agent_traj)
    else:
        # randomly initialise the agent feature expectations
        np.random.seed(2)
        agent_feature_exp = list(np.random.random([462, ]))

    if isfile('feature_expectations.csv'):
        f_e = list(np.genfromtxt('feature_expectations.csv', delimiter=','))
        new_entry = np.concatenate((agent_feature_exp, np.array([-1])))
        f_e.append(np.array(new_entry))
        f_e[0] = np.concatenate((demo_feature_exp, np.array([1])))
    else:
        f_e = []
        new_entry = np.concatenate((demo_feature_exp, np.array([1])))
        f_e.append(np.array(new_entry))
        new_entry = np.concatenate((agent_feature_exp, np.array([-1])))
        f_e.append(np.array(new_entry))

    train = np.array(f_e)
    linclf = SVC(kernel='linear')
    linclf.fit(train[:, :-1], train[:, -1])
    support_index = linclf.support_
    print(support_index)
    omega = np.squeeze(linclf.coef_)
    omega = omega / np.linalg.norm(omega)
    dif = np.transpose(np.array(f_e) - np.array(f_e[0]))

    dif = dif[0:-1]
    t = np.dot(omega, dif)
    print('distance to the expert expectation:', t)
    np.savetxt('pred_error.csv', t, delimiter=',')
    reward_scheme = R.reward_scheme(omega, scale=1)
    reward_scheme = reward_scheme.reshape([462, ])
    # relocate the maximum reward to 0, and the minimum to -1
    reward_scheme = (reward_scheme - max(reward_scheme))/(max(reward_scheme) - min(reward_scheme))
    try:
        remove('reward_AL.csv')
    except Exception:
        pass
    np.savetxt('reward_AL.csv', reward_scheme, delimiter=',')
    input('please check the reward function and then proceed')
    np.savetxt('feature_expectations.csv', f_e, delimiter=',')


if __name__ == "__main__":
    filename = 'agentdata_AL.csv'
    reward_reconstruct(filename)
    if isfile('reward_AL.csv'):
        print('reward saved')
    else:
        input('reward not found')
    RL_Multi(reward_flag='AL', validation=0, learningrate=0.005, render=False, episode=10000)

    RL_Multi(reward_flag='AL', validation=1, render=False, episode=2000)
    if isfile('agentdata_AL.csv'):
        print('agent traj saved')
    else:
        input('agent traj not found')
