import numpy as np
from os.path import isfile


class errorlog():
    def __init__(self, filename):
        self.filename = filename
        if isfile(self.filename):
            print('loading error log')
            e_rate = np.genfromtxt(self.filename, delimiter=',')
            self.e = e_rate[0,:]
            self.success_rate = e_rate[1,:]
        else:
            self.e = []
            self.success_rate = []

    def collect(self, e_entries, success_rate):
        self.e = np.append(self.e, e_entries)
        self.e = list(self.e)
        self.success_rate = np.append(self.success_rate, success_rate)
        self.success_rate = list(self.success_rate)

    def save_e(self):
        e_r = np.asarray([self.e,self.success_rate])
        np.savetxt(self.filename, e_r, delimiter=',')


class elligibility_trace():

    def __init__(self, set_sav, get_sav, params,
                 lr=0.02, epsilon=0.7, gamma=0.8000, eli_decay=0.8000):

        self.thread_object = None
        self.log = np.zeros([1,8])
        self.E = epsilon
        self.set_sav = set_sav
        self.get_sav = get_sav
        self.optimal = 0
        self.target_state = params['q1']['target_state']
        self.sav_error = 0
        self.state_action_num = params['q1']['sa_num']
        self.terminal_states = params['q1']['terminal']
        self.ep_s = []
        self.ep_r = []
        self.ep_a = []
        self.ep_s_ = []
        self.statecont_log = []
        self.actioncont_log = []
        self.lr = lr
        self.gamma = gamma
        self.tem = 0.99
        self.elligibility_trace = np.zeros_like(self.get_sav())
        self.eli_decay = eli_decay

    def learn(self, a_):
        s = self.ep_s[-1]
        s_ = self.ep_s_[-1]
        a = self.ep_a[-1]
        r = self.ep_r[-1]
        sav = self.get_sav()
        sa = tuple(np.concatenate((s, a)))
        q_predict = sav[sa]
        s_a_ = tuple(np.concatenate((s_, a_)))
        update_target = r + self.gamma * sav[s_a_]
        self.elligibility_trace[tuple(s)] *= 0
        self.elligibility_trace[sa] = 1
        sav += self.lr * (update_target - q_predict) * self.elligibility_trace
        self.set_sav(sav_new=sav)
        self.elligibility_trace *= self.eli_decay

    def store_trans(self, s, a, r, s_, scont_=None, acont=None):
        # s, a, s_ are numpy arrays
        self.ep_s.append(s)
        self.ep_a.append(a)
        self.ep_r.append(r)
        self.ep_s_.append(s_)
        self.statecont_log.append(scont_)
        self.actioncont_log.append(acont)

    def policy(self, state):
        state = tuple(state)
        sav = self.get_sav()
        sav_action = sav[state]
        # An epsilon-greedy policy
        where_max = np.where(sav_action == np.amax(sav_action))
        prob = (1-self.E)/sav_action.size*np.ones_like(sav_action)
        prob[where_max] += self.E/len(where_max[0])
        # alternative policy
        #soft_max = [np.e**(i/self.tem) for i in sav_action]
        #prob = [i/np.sum(soft_max) for i in soft_max]

        return np.random.choice(len(sav_action), size=1, p=prob)


