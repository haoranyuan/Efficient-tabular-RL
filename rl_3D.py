import quad_env,gui,rl_3d_agent,state_action_value
from tqdm import tqdm
import numpy as np
from numpy.random import uniform
from os import remove
from os.path import isfile


def RL_Multi(QUAD_DYNAMICS_UPDATE=0.05,
             reward_flag='default',
             learningrate=0.005,
             validation=0,
             episode=5000,
             render=True,
             verbose=False,
             cont_state=False):

    Range = {
        'linear_z': [-3, 3],
        'linear_zrate': [-0.5, 0.5],
    }
    action_num = np.int32(2)
    state_num = np.array([21, 11], dtype=np.int32)
    state_action_num = np.hstack((state_num, action_num))
    target_state = np.array([10, 5], dtype=np.int32)

    terminal_state = []
    for s1 in range(state_action_num[0]):
        for s2 in range(state_action_num[1]):
            for a in range(state_action_num[2]):
                if s1 == 0 or s1 == state_action_num[0]-1 or \
                   s1 == target_state[0] and s2 == target_state[1]:
                    terminal_state.append([s1, s2])

    QUADCOPTER = {
        'q1': {'position': [0, 0, uniform(Range['linear_z'][0], Range['linear_z'][1])], 'orientation': [0, 0, 0], 'L': 0.3,
               'r': 0.1, 'prop_size': [10, 4.5], 'weight': 1.2, 'range': Range, 'sa_num': state_action_num,
                'terminal': terminal_state, 'target_state': target_state}}

    CONTROLLER_PARAMETERS = {'Motor_limits': [5200, 5500]}
    sav = state_action_value.StateActionValue(state_action_num=state_action_num,
                                              reward_flag=reward_flag)
    quad = quad_env.Quadcopter(motorlimits=CONTROLLER_PARAMETERS['Motor_limits'],
                               quads=QUADCOPTER,
                               rewardflag=reward_flag)
    if render:
        gui_object = gui.GUI(quads=QUADCOPTER)
    if validation:
        E = 1
    else:
        E = 0.7
    ctrl1 = rl_3d_agent.RL_3D(change_sav=sav.change_sav,
                              get_sav=sav.get_sav,
                              lr=learningrate,
                              params=QUADCOPTER,
                              epsilon=E)
    #errorlog1 = rl_3d_agent.errorlog('e_'+reward_flag+'.csv')
    print('initialization finished, begin episodes.')

     # episodes
    success_rate = []
    reward_log = []
    if isfile('sc_scatter_'+reward_flag+'val'+str(validation)+'.csv'):
        success_rate = list(np.genfromtxt('sc_scatter_'+reward_flag+'val'+str(validation)+'.csv', delimiter=','))
        print('load success rate data')

    sav_old = np.copy(sav.get_sav())
    #MAX_EP_STEP = 500
    MAX_EP_STEP = 200
    for i in tqdm(range(episode)):
        state = quad.reset_quads()
        success = 0
        done = 0
        reward = 0
        for j in range(MAX_EP_STEP):
            # If not success but cross the boarder then reset
            if done and not success:
                state = quad.reset_quads()
            action = ctrl1.policy(state)
            state_, r, done, sc, statecont_, actioncont_ = quad.one_step(action=action,
                                            dt=QUAD_DYNAMICS_UPDATE)
            ctrl1.store_trans(state, action, r, state_, statecont_, actioncont_)
            reward += r
            if not validation:
                ctrl1.learn()
            if render:
                gui_object.quads['q1']['position'] = quad.get_position('q1')
                gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
                gui_object.update()



            # Keep the previous state at the boarder if reset
            state = state_
        if sum(np.prod((ctrl1.ep_s_[-10:] == target_state), axis=1)) > 3:
            success = 1
        success_rate.append(np.array([success, reward]))
        if verbose:
            if i % 10 == 0:
                print('episode={0:5d}, r_total={1:8.2f}, r_average={2:8.2f}, success rate: {3:1.3f}, '
                      'norm(old_sav-new_sav)={4:2.4f}'
                      .format(i, np.sum(ctrl1.ep_r), np.sum(ctrl1.ep_r) / len(ctrl1.ep_r),
                       success_rate[-1], np.linalg.norm(sav_old - sav.get_sav())))
                sav_old = np.copy(sav.get_sav())
        if validation:
            try:
                agentdata = np.concatenate((agentdata, np.hstack((ctrl1.statecont_log, ctrl1.actioncont_log))), axis=0)
            except Exception:
                agentdata = np.hstack((ctrl1.statecont_log, ctrl1.actioncont_log))
        ctrl1.ep_r = []
        ctrl1.ep_a = []
        ctrl1.ep_s_ = []
        ctrl1.ep_s = []
        ctrl1.statecont_log = []
        ctrl1.actioncont_log = []
    if not validation:
        sav.save_file()
    # Save continuous state log

    # Save success log
    np.savetxt('sc_scatter_'+reward_flag+'val'+str(validation)+'.csv', success_rate, delimiter=',')
    # Save trajectories
    if validation:
        try:
            remove('agentdata_'+reward_flag+'csv')
        except Exception:
            pass
        np.savetxt('agentdata_'+reward_flag+'.csv', agentdata, delimiter=',')


if __name__ == "__main__":
    RL_Multi(QUAD_DYNAMICS_UPDATE=0.2, render=False, reward_flag='default', verbose=False, validation=0, episode=10000)
