# much inspiration from sentdex, especially
# https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/?completed=/q-learning-analysis-reinforcement-learning-python-tutorial/


import pygame
import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt

from resources.mobs import Predator, Prey, Food


pygame.init()

# resources
RES = 'resources'
LOG = open(os.path.join(RES, 'game.log'), 'w')

# pygame setup
WIDTH = 400  # 1080
HEIGHT = 400  # 800
gameDisplay = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
FPS = 30

# Q learning variables
EPISODES = 1000  # 22500  # with epsilon decay rate at 0.9998, this corresponds to <1% random moves
SHOW = 1000  # how often to visualize
FRAMES = 300  # per episode
EPSILON = 0.9  # random action threshhold
DECAY_RATE = 0.999#8  # espilon *= DECAY_RATE

# load/save Q tables
TABLES = 'q_tables'
PREY_TABLE = False  # os.path.join(RES, TABLES, 'Prey-7965313.Q')
PRED_TABLE = False  # os.path.join(RES, TABLES, 'Predator-8637585.Q')
SAVE_Q = True

# plotting
PLOTS = 'plots'

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def init_mobs(food=0, prey=(0, False), pred=(0, False)):
    mobs = {'Food': [],
            'Prey': [],
            'Predator': [],
           }
    dims = (WIDTH, HEIGHT)
    
    for f in range(food):
        mobs['Food'].append(Food(x=0, y=0, dims=dims))
    
    for p in range(prey[0]):
        mobs['Prey'].append(Prey(x=0, y=0, dims=dims, load=prey[1]))
    
    for p in range(pred[0]):
        mobs['Predator'].append(Predator(x=0, y=0, dims=dims, load=pred[1]))
    
    print('\n' + '='*60 + '\n')
    return mobs


def display_stats(episode, frame, mobs):
    font = pygame.font.SysFont(None, 32)
    
    text = font.render('episode/frame: {}/{}'.format(episode, frame), True, WHITE)
    gameDisplay.blit(text,(0, 0))
    
    for num, key in enumerate(mobs.keys()):
        tally = 0
        total = len(mobs[key])
        for mob in mobs[key]:
            tally = tally + 1 if mob.alive else tally  # FIXME add rewards (avg?)
        message = '{}: {}/{}'.format(key, tally, total)
        text = font.render(message, True, WHITE)
        gameDisplay.blit(text,(0, (num+1)*40))


def episode_cleanup(episode, mobs, rewards):
    print('Episode {}/{} completed at {}'.format(episode+1, EPISODES, time.asctime()))
    for mob_type, mob_list in mobs.items():
        if mob_type != 'Food':
            for mob in mob_list:
                print('{} {:>8}: {:<9} ({})'.format(mob.__class__, mob.serial, round(rewards[mob][episode], 3), mob.alive))
    print('\n' + '='*60 + '\n')


def plot_rewards(mobs=None, rewards=None):
    for mob_type, mob_list in mobs.items():
        if mob_type != 'Food':
            for mob in mob_list:
                plt.plot(rewards[mob], label='{}:{}'.format(mob_type, mob.serial))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    fig_name = os.path.join(RES, PLOTS, '{}.png'.format(int(time.time())))
    print('Saving episode rewards plot as {}'.format(fig_name))
    plt.savefig(fig_name)
    plt.close()


def exit_sim():
    fade_out = 1
    pygame.mixer.music.fadeout(fade_out * 1000)
    LOG.write('\nExiting normally!\n')
    #LOG.close()
    time.sleep(fade_out)
    pygame.quit()


def run():
    
    mobs = init_mobs(food=100, prey=(1, PREY_TABLE), pred=(0, PRED_TABLE))

    epsilon = EPSILON

    rewards = {}
    for mob_type, mob_list in mobs.items():
        for mob in mob_list:
            rewards[mob] = [ 0 ] * EPISODES
    
    for episode in range(EPISODES):
        show_this = True if episode % SHOW == 0 else False
        
        # reset all the mobs for this episode
        for mob_type, mob_list in mobs.items():
            for mob in mob_list:
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)
                mob.reset(x=x, y=y)
        
        # run the episode
        for k in range(FRAMES):
            gameDisplay.fill(BLACK)
            
            # update all mobs
            for mob_type, mob_list in mobs.items():
                for mob in mob_list:
                    if mob.alive:
                        q_key = mob.observe(mobs=mobs)  # find the closest food/prey/predator
                        mx, my, choice = mob.action(epsilon=epsilon, q_key=q_key)  # take an action
                        reward, _ = mob.check(mobs=mobs, mx=mx, my=my)  # check to see what has happened
                        mob.update_q(mobs=mobs, q_key=q_key, choice=choice, reward=reward)  # learn from what mob did

                        rewards[mob][episode] += (reward[0] + reward[1])  # tally for episode rewards

            # display all mobs
            if show_this:
                for mob_type, mob_list in mobs.items():
                    for mob in mob_list:
                        if mob.alive:
                            mob.display(gameDisplay)
            
            # complete the render and wait to cycle
            display_stats(episode, k+1, mobs)
            pygame.display.update()
            if show_this:
                clock.tick(FPS)  # no need to wait if we aren't visualizing
            else:
                clock.tick(10**10)

        # clean up the episode
        episode_cleanup(episode, mobs, rewards)
        epsilon *= DECAY_RATE

    if SAVE_Q:
        for mob_type, mob_list in mobs.items():
            if mob_type != 'Food':
                for mob in mob_list:
                    mob.q_table.save(os.path.join(RES, TABLES), mob_type, mob.serial)
    else:
        print('Q table saving disabled')

    return mobs, rewards


def main():
    mobs, rewards = run()
    exit_sim()
    plot_rewards(mobs=mobs, rewards=rewards)


if __name__ == '__main__':
    import traceback
    try:
        main()  # FIXME just run this
    except Exception as e:
        LOG.write('{}\n'.format(e))
        tb = traceback.format_exc()
    else:
        tb = 'no error'
    finally:
        LOG.write('{}\n'.format(tb))
        LOG.write('End!')
        LOG.close()
