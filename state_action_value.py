import os.path
import numpy as np


class StateActionValue:
    def __init__(self,state_action_num, reward_flag):
        self.state_action_num = state_action_num

        self.filename1 = 'sav_'+reward_flag+'.csv'
        if os.path.isfile(self.filename1):
            print('loading state action value function')
            sav = np.genfromtxt(self.filename1, delimiter=',')
            sav = sav.reshape(state_action_num)
        else:
            sav = np.zeros(shape=state_action_num, dtype=np.float64)
            #sav = -0.001 * np.random.random(state_action_num)
        self.filename2 = 'sav_count.csv'
        if os.path.isfile(self.filename2):
            sav_count = np.genfromtxt(self.filename2, delimiter=',')
            sav_count = sav_count.reshape(state_action_num)
        else:
            sav_count = np.zeros(shape=state_action_num, dtype=np.int32)
            # sav = -0.001 * np.random.random(state_action_num)
        self.sav = sav
        self.sav_count = sav_count

    def change_sav(self, index, value):
        self.sav[tuple(index)] = value
        self.sav_count[tuple(index)] += 1
        #print(index,self.sym_idx(index))


    def get_sav(self,):
        return self.sav

    def save_file(self):
        SAV = self.get_sav()
        dim = 1
        for d in self.state_action_num:
            dim = dim * d
        SAV = SAV.reshape([dim, 1])
        SAV_COUNT = self.sav_count.reshape([dim, 1])
        np.savetxt(self.filename1, SAV, delimiter=',')
        #np.savetxt(self.filename2, SAV_COUNT, delimiter=',')
