# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        #if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
        if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rad/s]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label=' base velx measured')
        if log["base_x"]: a.plot(time, log["base_x"], label=' base x measured')
        # if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        # a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        # a.legend()
        # # plot base vel y
        # a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity')
        a.legend()
        # plot base vel yaw
        a = axs[0, 1]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot base vel z
        # a = axs[1, 2]
        a = axs[0, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='z vel')
        if log["base_z"]: a.plot(time, log["base_z"], label='z pos')
        if log["base_vel_x_inertial"]: a.plot(time, log["base_vel_x_inertial"], label='x vel inertia')

        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base z')
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()


        if log["veloVec"]!=[]:
            a.plot(time,log["veloVec"])
            a.set(xlabel='time [s]',title='All Amplitudes')
            a.legend() 

        fig6 = plt.figure()
        plt.subplot(2, 2, 1)

        mil=plt.plot(time,log["veloVec1"])
        plt.legend(iter(mil),('Little-Dog', 'Spot-Micro', 'Solo' ,'Mini-Cheetah'))
        ax = plt.gca()
        ax.set_ylim(0,1.72)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Base Velocity (m/s)')

        plt.subplot(2, 2, 2)
        mil=plt.plot(time,log["veloVec2"])
        plt.legend(iter(mil),( 'A1', 'Go1', 'Aliengo' ,'Laikago'))
        ax = plt.gca()
        ax.set_ylim(0,1.72)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Base Velocity (m/s)')

        plt.subplot(2, 2, 3)
        mil=plt.plot(time,log["veloVec3"])
        plt.legend(iter(mil),('Anymal-B', 'Anymal-C', 'Spot' ,'B1'))
        ax = plt.gca()
        ax.set_ylim(0,1.72)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Base Velocity (m/s)')

        plt.subplot(2, 2, 4)
        mil=plt.plot(time,log["veloVec4"])
        plt.legend(iter(mil),( 'HYQ', 'Dog 1', 'Dog 2' ,'Dog 3'))
        ax = plt.gca()
        ax.set_ylim(0,1.72)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Base Velocity (m/s)')
        plt.show()





    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()