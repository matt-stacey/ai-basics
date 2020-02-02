# much inspiration from sentdex, sspecially
# https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/?completed=/q-learning-analysis-reinforcement-learning-python-tutorial/


import pygame
import time
import random
import os

import resources.colors as colors
from resources.mobs import Predator, Prey, Food

pygame.init()

RES = 'resources'
LOG = open(os.path.join(RES, 'game.log'), 'w')

WIDTH = 400
HEIGHT = 400

gameDisplay = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()
FPS = 30


def init_mobs(food=0, prey=(0, False), pred=(0, False)):
    mobs = {'food': [],
            'prey': [],
            'predator': [],
           }
    dims = (WIDTH, HEIGHT)
    
    for f in range(food):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        mobs['food'].append(Food(x=x, y=y, dims=dims))
    
    for p in range(prey[0]):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        mobs['prey'].append(Prey(x=x, y=y, dims=dims, load=prey[1]))
    
    for p in range(pred[0]):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        mobs['predator'].append(Predator(x=x, y=y, dims=dims, load=pred[1]))
    
    return mobs


def display_stats():
    font = pygame.font.SysFont(None, 32)
    message = 'statistics'
    text = font.render(message, True, colors.white)
    gameDisplay.blit(text,(0, 0))


def exit_sim():
    fade_out = 1
    pygame.mixer.music.fadeout(fade_out * 1000)
    LOG.write('\nExiting normally!\n')
    #LOG.close()
    time.sleep(fade_out)
    pygame.quit()
    #quit()


def run():
    
    mobs = init_mobs(food=40, prey=(5, False), pred=(1, False))
    
    SEC = 10
    
    for k in range(SEC * FPS):
        gameDisplay.fill(colors.black)
        
        # update and display all mobs
        for mob_type, list in mobs.items():
            for mob in list:
                if mob.alive:
                    mob.action(random.randint(0, 16))
                    mob.display(gameDisplay)
                else:
                    pass  # poppable?
        
        # complete the render and wait to cycle
        display_stats()
        pygame.display.update()
        clock.tick(FPS)


def main():
    run()
    exit_sim()

if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        LOG.write('{}\n'.format(e))
        tb = traceback.format_exc()
    else:
        tb = 'no error'
    finally:
        LOG.write('{}\n'.format(tb))
        LOG.close()
