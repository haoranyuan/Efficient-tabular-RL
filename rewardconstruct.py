import numpy as np


class feature:
    def __init__(self):
        self.state_action_num = [21, 11, 2]
        self.target = [10, 5]

        span = []
        for idx in range(len(self.state_action_num)):
            i = 1
            if idx + 1 < len(self.state_action_num):
                for j in self.state_action_num[idx + 1:]:
                    i = i * j
            span.append(i)
        self.span = span

    def get_feature(self, dis_state_action):
        i = 1
        for j in self.state_action_num:
            i = i * j
        feature = np.zeros(i)
        feature_code = int(np.dot(list(dis_state_action), self.span))
        feature[feature_code] = 1
        return feature


class FeatureExpectation:
    def __init__(self):
        self.state_action_num = [21, 11, 2]
        self.target = [10, 5]
        self.feature = feature()

    def featureexpectations(self, trajectories, ep_len=200):
        self.ep_len = ep_len
        traj_count = 0
        DISCOUNT = 0.9999
        discount = 1
        i = 1
        for j in self.state_action_num:
            i = i * j
        EXPECTATION = np.zeros(i)
        exp = np.zeros_like(EXPECTATION)
        for i, realization in enumerate(trajectories):
            feature = self.feature.get_feature(realization)
            exp = exp + discount * np.asarray(feature)
            discount = discount * DISCOUNT
            if (i+1)%self.ep_len == 0: #  or\
                    #realization[0] == 0 or realization[0] == self.state_action_num[0] - 1:
                # or\ #list(realization[:len(self.target)]) == self.target or\
                traj_count += 1
                discount = 1
                EXPECTATION += exp
        # feature occupancy
        EXPECTATION = EXPECTATION / traj_count / self.ep_len
        self.expectation = EXPECTATION
        print(traj_count)
        return EXPECTATION, traj_count


class Rewardconstruct:
    def __init__(self):
        self.state_action_num = [21, 11, 2]
        self.feature = feature()
        self.rewardscheme = np.zeros(self.state_action_num,dtype=float)

    def reward_scheme(self,omega,scale):
        self.omega = omega
        i = 1
        for j in self.state_action_num:
            i = i * j
        for index in range(i):
            dis_state_action = tuple(np.unravel_index(index, self.state_action_num))
            fea = self.feature.get_feature(dis_state_action)
            r = np.dot(fea, self.omega)
            self.rewardscheme[dis_state_action] = r
        self.rewardscheme = self.rewardscheme * scale
        return self.rewardscheme


class CustomizeReward:
    def __init__(self):
        self.state_action_num = [21, 11, 2]
        self.target = tuple([10, 5])
        self.rewardscheme = np.zeros(self.state_action_num)

    def get_reward_scheme(self, plan_id):
        for y in range(self.state_action_num[0]):
            for y_d in range(self.state_action_num[1]):
                for p1 in range(self.state_action_num[2]):
                        idx = (y, y_d, p1)
                        self.rewardscheme[idx] = self.rewardplan(idx,plan_id)
        self.rewardscheme = self.rewardscheme.reshape([462, ])
        self.rewardscheme = self.rewardscheme / abs(min(self.rewardscheme))
        np.savetxt('reward_'+plan_id+'.csv',self.rewardscheme,delimiter=',')

    def rewardplan(self,idx,plan_id):
        if plan_id == 'default':
            dist = np.array(idx[0:len(self.target)]) - np.array(self.target)
            r = -(1 * (abs(dist[0])**2) + 1 * (abs(dist[1])**1))
        if plan_id == 'A':
            if idx[0:len(self.target)] == self.target:
                r = 0
            else:
                r = -1
        #if plan_id == 'B'
        return r



if __name__ == '__main__':
    R = CustomizeReward()
    plan = 'default'
    R.get_reward_scheme(plan_id=plan)
    #R.get_reward_scheme(plan_id='A')
    r = np.genfromtxt('reward_'+plan+'.csv', delimiter=',')
    #r = np.genfromtxt('reward_A.csv', delimiter=',')
    print(np.unravel_index(np.argmax(r), [21, 11, 2]))