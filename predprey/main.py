# much inspiration from sentdex, sspecially
# https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/?completed=/q-learning-analysis-reinforcement-learning-python-tutorial/


import pygame
import time
import random
import os
import numpy as np

import resources.colors as colors
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
EPISODES = 20000
SHOW = 1  # how often to visualize
epsilon = 0.9  # random action threshhold
DECAY_RATE = 0.9998  # espilon *= DECAY_RATE

# existing Q tables
PREY_TABLE = False
PRED_TABLE = False


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
    
    return mobs


def display_stats(episode, mobs):
    font = pygame.font.SysFont(None, 32)
    
    text = font.render('episode: {}'.format(episode), True, colors.white)
    gameDisplay.blit(text,(0, 0))
    
    for num, key in enumerate(mobs.keys()):
        tally = 0
        total = len(mobs[key])
        for mob in mobs[key]:
            tally = tally + 1 if mob.alive else tally  # FIXME add rewards (avg?)
        message = '{}: {}/{}'.format(key, tally, total)
        text = font.render(message, True, colors.white)
        gameDisplay.blit(text,(0, (num+1)*40))


def exit_sim():
    fade_out = 1
    pygame.mixer.music.fadeout(fade_out * 1000)
    LOG.write('\nExiting normally!\n')
    #LOG.close()
    time.sleep(fade_out)
    pygame.quit()
    #quit()


def run():
    
    mobs = init_mobs(food=40, prey=(5, PREY_TABLE), pred=(1, PRED_TABLE))

    # FIXME
    # tally up all episode rewards so we can graph them for each mob 'brain'
    rewards = {}
    for mob_type, mob_list in mobs.items():
        for mob in mob_list:
            rewards[(mob.__class__, mob.serial)] = []

    SEC = 7
    
    for episode in range(EPISODES):
        show_this = True if episode % SHOW == 0 else False
        
        # reset all the mobs for this episode
        for mob_type, mob_list in mobs.items():
            for mob in mob_list:
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)
                mob.reset(x=x, y=y)
        
        # run the episode
        for k in range(SEC * FPS):
            gameDisplay.fill(colors.black)
            
            # update and display all mobs
            for mob_type, mob_list in mobs.items():
                for mob in mob_list:
                    if mob.alive:

                        observation = mob.observe(mobs=mobs)  # find the closest food/prey/predator
                        mob.action(epsilon=epsilon, observation=observation)  # take an action
                        mob.check(mobs)  # check to see what has happened
                        mob.update_q()  # learn from what mob did

                        if show_this:
                            mob.display(gameDisplay)
                    else:
                        pass  # poppable?
            
            # complete the render and wait to cycle
            display_stats(episode, mobs)
            pygame.display.update()
            if show_this:
                clock.tick(FPS)  # no need to wait if we aren't visualizing

        # clean up the episode


def main():
    run()
    exit_sim()

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
        LOG.close()
