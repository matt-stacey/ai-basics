# much inspiration from sentdex, sspecially
# https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/?completed=/q-learning-analysis-reinforcement-learning-python-tutorial/


import pygame
import time
import random
import os

import resources.colors as clr
from resources.mobs import Predator, Prey, Food

pygame.init()

RES = 'resources'
LOG = open(os.path.join(RES, 'game.log'), 'w')

WIDTH = 1080
HEIGHT = 800

gameDisplay = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()
FPS = 30


def init_mobs(food=0, prey=0, pred=0):
    mobs = {'food': [],
            'prey': [],
            'predator': [],
           }
    dims = (WIDTH, HEIGHT)
    
    for f in range(food):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        mobs['food'].append(Food(x=x, y=y, dims=dims))
    
    for p in range(prey):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        mobs['prey'].append(Prey(x=x, y=y, dims=dims))
    
    for p in range(pred):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        mobs['predator'].append(Predator(x=x, y=y, dims=dims))
    
    return mobs


def display_stats():
    font = pygame.font.SysFont(None, 32)
    message = 'statistics'
    text = font.render(message, True, clr.white)
    gameDisplay.blit(text,(0, 0))


def exit_sim():
    fade_out = 1
    pygame.mixer.music.fadeout(fade_out * 1000)
    LOG.write('\nExiting normally!\n')
    LOG.close()
    time.sleep(fade_out)
    pygame.quit()
    quit()


def run():
    
    food = 100
    prey = 5
    pred = 1
    
    mobs = init_mobs(food=food, prey=prey, pred=pred)
    
    SEC = 10
    
    for k in range(SEC * FPS):
        # update all mobs
        for mob_type, list in mobs.items():
            for mob in list:
                mob.action(random.randint(0, 16))
        
        # display all mobs
        gameDisplay.fill(clr.black)
        for mob_type, list in mobs.items():
            for mob in list:
                mob.display(gameDisplay)
        
        # complete the render and wait to cycle
        display_stats()
        pygame.display.update()
        clock.tick(FPS)


def main():
    run()
    exit_sim()

if __name__ == '__main__':
    main()