# much inspiration from sentdex, especially
# https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/?completed=/q-learning-analysis-reinforcement-learning-python-tutorial/


import pygame
import time
import random
import os
import argparse
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

# Q learning variables [DEFAULTS]
EPISODES = 100  # 22500  # with epsilon decay rate at 0.9998, this corresponds to <1% random moves
SHOW = 100  # how often to visualize
FRAMES = 100  # per episode
EPSILON = 0.9  # random action threshhold
DECAY_RATE = 0.9998  # espilon *= DECAY_RATE

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


def sim_init(food=0, prey=0, pred=0):

    mobs = init_mobs(food=food, prey=(prey, PREY_TABLE), pred=(pred, PRED_TABLE))

    epsilon = EPSILON

    rewards = {}
    for mob_type, mob_list in mobs.items():
        for mob in mob_list:
            rewards[mob] = [ 0 ] * EPISODES

    return mobs, epsilon, rewards


def init_mobs(food=0, prey=(0, False), pred=(0, False)):
    mobs = {'Food': [],
            'Prey': [],
            'Predator': [],
           }
    
    for f in range(food):
        mobs['Food'].append(Food(x=0, y=0))
    
    for p in range(prey[0]):
        mobs['Prey'].append(Prey(x=0, y=0, load=prey[1]))
    
    for p in range(pred[0]):
        mobs['Predator'].append(Predator(x=0, y=0, load=pred[1]))
    
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
    #LOG.close()   # FIXME when you remove traceback
    time.sleep(fade_out)
    pygame.display.quit()
    pygame.quit()


def train(food=0, prey=(0, False), pred=0):

    allow_prey_movement = prey[1]
    valued_customer = 'Prey' if allow_prey_movement else 'Predator'
    
    # set a sceen wize to just larger than the mob can see
    WIDTH = int(Prey.sight * 1.5) if pred == 0 else int(Predator.sight * 1.5)
    HEIGHT = WIDTH
    pygame.display.set_mode((WIDTH, HEIGHT))
    mobs, epsilon, rewards = sim_init(food=food, prey=prey[0], pred=pred)

    for episode in range(EPISODES):
        show_this = True if episode % SHOW == 0 else False
        end_ep = False
        
        # reset all the mobs for this episode
        for mob_type, mob_list in mobs.items():
            for mob in mob_list:
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)
                if mob_type == valued_customer:
                    x = WIDTH / 2  # centered
                    y = HEIGHT / 2
                mob.reset(x=x, y=y)
        
        # run the episode
        for k in range(FRAMES):
            gameDisplay.fill(BLACK)
            
            # update mobs
            for mob_type, mob_list in mobs.items():
                for mob in mob_list:
                    update_this = True if (allow_prey_movement and mob_type == 'Prey') or mob_type == 'Predator' else False
                    if mob.alive and update_this:
                        q_key = mob.observe(mobs=mobs)  # find the closest food/prey/predator
                        mx, my, choice = mob.action(epsilon=epsilon, q_key=q_key, max_dims=(WIDTH, HEIGHT))  # take an action
                        reward, _ = mob.check(mobs=mobs, mx=mx, my=my)  # check to see what has happened
                        mob.update_q(mobs=mobs, q_key=q_key, choice=choice, reward=reward)  # learn from what mob did
        
                        rewards[mob][episode] += (reward[0] + reward[1])  # tally for episode rewards
                    elif not mob.alive:
                        end_ep = True  # die when one of the mobs does

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
                clock.tick(FPS)
            else:
                clock.tick(10**10)

            if end_ep:
                if show_this:
                    time.sleep(1)  # pause at the end state
                break

        # clean up the episode
        episode_cleanup(episode, mobs, rewards)
        epsilon *= DECAY_RATE

    if SAVE_Q:
        for mob in mobs[valued_customer]:
            mob.q_table.save(os.path.join(RES, TABLES), valued_customer, mob.serial)
    else:
        print('Q table saving disabled')

    return mobs, rewards


def run(food=0, prey=0, pred=0):
    mobs, epsilon, rewards = sim_init(food=food, prey=prey, pred=pred)
    
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
                        mx, my, choice = mob.action(epsilon=epsilon, q_key=q_key, max_dims=(WIDTH, HEIGHT))  # take an action
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
                clock.tick(FPS)
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
    parser = argparse.ArgumentParser(description='''Predator/Prey AI Trainer and Visualizer''')

    parser.add_argument('-m', '--mode', help='training/execution mode for AI', default='prey')

    # mob selection
    parser.add_argument('--pred', help='number of predator mobs', default=0)
    parser.add_argument('--prey', help='number of prey mobs', default=1)
    parser.add_argument('--food', help='number of food mobs', default=100)

    # load/save mob q_tables
    parser.add_argument('--q_pred', help='pre-generated predator Q table', default=False)
    parser.add_argument('--q_prey', help='pre-generated prey Q table', default=False)
    parser.add_argument('--save_q', help='save final Q tables', dest='save_q', action='store_true')
    parser.set_defaults(save_q=False)
    parser.add_argument('--no-plot', help='don\'t plot episode rewards', dest='plot_rew', action='store_false')
    parser.set_defaults(plot_rew=True)

    # training variables
    parser.add_argument('--episodes', help='number of training episodes', default=EPISODES)
    parser.add_argument('--show', help='regularity to visualize environment', default=SHOW)
    parser.add_argument('--frames', help='steps per training episode', default=FRAMES)
    parser.add_argument('--epsilon', help='random decision threshold', default=EPSILON)
    parser.add_argument('--decay', help='random decision threshold decay rate', default=DECAY_RATE)

    args = parser.parse_args()
    mobs = None
    rewards = None

    globals()['PREY_TABLE'] = False if not args.q_prey else os.path.join(RES, TABLES, args.q_prey)
    globals()['PRED_TABLE'] = False if not args.q_pred else os.path.join(RES, TABLES, args.q_pred)
    globals()['SAVE_Q'] = args.save_q

    globals()['EPISODES'] = int(args.episodes)
    globals()['SHOW'] = int(args.show)
    globals()['FRAMES'] = int(args.frames)
    globals()['EPSILON'] = float(args.epsilon)
    globals()['DECAY_RATE'] = float(args.decay)

    if args.mode == 'pred':
        mobs, rewards = train(food=0, prey=(1, False), pred=1)  # train the predator Q table
    elif args.mode == 'prey':
        mobs, rewards = train(food=1, prey=(1, True), pred=0)  # train the prey Q table (target)
    elif args.mode == 'evade':
        mobs, rewards = train(food=0, prey=(1, True), pred=1)  # train the prey Q table (flee)
    else:
        mobs, rewards = run(food=int(args.food), prey=int(args.prey), pred=int(args.pred))

    exit_sim()
    for mob_type in ('Prey', 'Predator'):  # 'Food' has no q_table
        for mob in mobs[mob_type]:
            mob.q_table.plot_q(os.path.join(RES, PLOTS, '{}_Q.png'.format(mob.serial)))  # moving out of game loop removed seg fault

    if args.plot_rew:
        plot_rewards(mobs=mobs, rewards=rewards)


if __name__ == '__main__':
    import traceback
    tb = 'no error'
    try:
        main()  # FIXME just run this
    except Exception as e:
        LOG.write('{}\n'.format(e))
        tb = traceback.format_exc()
    finally:
        LOG.write('{}\n'.format(tb))
        LOG.write('End!')
        LOG.close()
    print('Exiting main() with {}'.format(tb))
