import random
import math
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

try:
    import pygame_sdl2
    pygame_sdl2.import_as_pygame()
except ImportError:
    pass
import pygame


WHITE = (255, 255, 255)


def angle(coords=(0,0)):
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    
    x, y = coords
    
    r2d = 180 / np.pi
    ang = np.arctan2(x, -1*y) * r2d
    
    return ang


def distance(coords=(0,0)):
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    
    return ((coords[0]**2)+(coords[1]**2))**0.5


class Q_table():
    
    def __init__(self, r=0, bands=4, slices=8, actions=18, load=False):
        slices = int(slices) if slices >= 8 else 8  # at least 1 per move direction
        bands = int(bands) if bands >= 4 else 4  # range discrimination
        theta = 360 / slices
        half = theta / 2
        self.actions = actions
        
        self.quads = [(i * theta - half, i * theta + half) for i in range(slices)]
        self.ranges = [(r * i / bands, r * (i+1) / bands) for i in range(bands)]
        
        self.angle_bounds = (self.quads[0][0], self.quads[-1][-1])
        self.range_bounds = (0, r)
        
        if load:
            print('Loading Q table from {}'.format(load))
            with open(load, 'rb') as f:
                self.table = pickle.load(f)
        else:
            self.table = self.q_table_setup(actions=actions)
        
    def q_table_setup(self, actions=1):
        table = {}
        
        # keys will be tuples of "quadrants" and range bands, which are stored as [L, R) angles and [min, max) distance respectively
        angle_keys = [None] + list(range(len(self.quads)))
        range_keys = [None] + list(range(len(self.ranges)))
        
        table = {(a, r): [np.random.uniform(-actions, 0) for i in range(actions-1)] + [0] for a in angle_keys for r in range_keys}  # random action starts at 0
        
        return table
    
    def get_quad(self, theta):
        if theta == None:
            return None
        
        while theta < self.angle_bounds[0]:
            theta += 360
        while theta > self.angle_bounds[1]:
            theta -= 360
        
        for num, key in enumerate(self.quads):
            low, high = key
            if low <= theta < high:
                return num
        
        return None
    
    def get_range(self, r):
        if r == None:
            return None
        
        for num, key in enumerate(self.ranges):
            low, high = key
            if low <= r < high:
                return num
        
        return None

    def plot_q(self, filename, tgt=True):
        # plot: rows for range, cols for quad (where tgt is) plus 1 for None
        # subplot: x/choice, y/q value
        # lines: based on flee location, 9 total (None, short/long for fwd/back/L/R)
        print('Plotting Q table {}'.format(filename))
        rc = len(self.quads)
        for i in range(1,20):
            if i**2 >= len(self.quads):
                rc = i
                break
        
        line_minmax = [0, 0]
        odds = [o for o in range(self.actions) if o % 2 == 1]
        fig, axes = plt.subplots(nrows= rc, ncols=rc, sharex=True, sharey=True, figsize=[3*rc, 3*rc])
        
        q = len(self.quads)
        for r in range(rc):
            for c in range(rc):
                q = c + r * rc
                if q >= len(self.quads):
                    break
                styles = ('D', 's', 'o', '+', 'x', '*', '.')
                for l in range(len(self.ranges)):
                    key = ((q, l), (None, None)) if tgt else ((None, None), (q, l))
                    f = l
                    while f > len(styles) - 1:
                        f -= len(styles)  # there is probably a better way to do this...intertools.cycle?
                    fmt = '--{}'.format(styles[f])
                    axes[r, c].plot(self.table[key], fmt, label=self.ranges[l])
                    line_minmax[0] = line_minmax[0] if line_minmax[0] <= min(self.table[key]) else min(self.table[key])
                    line_minmax[1] = line_minmax[1] if line_minmax[1] >= max(self.table[key]) else max(self.table[key])
                axes[r,c].set_title(self.quads[q])
                for o in odds:
                    axes[r, c].plot([o, o], line_minmax, 'r--', alpha=0.3)
                line_minmax = [0, 0]
        
        axes[0, rc-1].legend()
        plt.xlabel('Action')
        plt.ylabel('q_value')
        
        plt.savefig(filename)  # needs to include directory structure
        plt.close()
    
    def save(self, directory, mob_type, serial, key):
        filename = '{}/{}-{}-{}.Q'.format(directory, mob_type, serial, key)
        print('Saving Q table as {}'.format(filename))
        with open(filename, 'wb') as f:  # microseconds
            pickle.dump(self.table, f)


class Mob():
    sight = 0
    
    def __init__(self, x=None, y=None, dims=None):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Mob not passed position in init!')
        self.x = x
        self.y = y
        self.health_init = 0
        self.health = self.health_init
        
        self.alive = True
        self.serial = int(time.time() * 10**6) % (10**7)
        
        self.target = [None, None]
        self.flee = [None, None]
        
        #self.sight = 0
        self.slices = 0
        self.speed = (0, 0)
        self.moves = [(self.x, self.y)]
        
        self.color = (0, 0, 0)
        self.show_moves = False  # False or number (True for all)
        
        self.learning_rate = 0  # 0: no learning, 1: no memory
        self.discount = 0  # 0: no time preference, 1: infinite time preference
        
        self.q_table = {'target': None, 'flee': None}
        
    def __str__(self):
        return '{} at ({}, {})'.format(self.__class__, self.x, self.y)
    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return (self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    @property
    def r(self):
        return self.health ** 0.5
        
    def reset(self, x=None, y=None):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Mob not passed position in init!')
        self.x = x
        self.y = y
        
        self.health = self.health_init
        self.alive = True
        self.target[1] = None
        self.flee[1] = None
        self.moves = []
        
    def observe(self, mobs=None):
        # prevent mobs from switching targets too much
        delta = self.speed[1] + self.speed[0]  # how much closer to switch targets/flee
        
        for mob_type, mob_list in mobs.items():
            if mob_type == self.target[0]:
                for mob in mob_list:
                    ds = distance(mob - self)
                    if mob.alive and (self.target[1] == None or ds < distance(self.target[1] - self) - delta):
                        self.target[1] = mob
            
            elif mob_type == self.flee[0]:
                for mob in mob_list:
                    ds = distance(mob - self)
                    if mob.alive and (self.flee[1] == None or ds < distance(self.flee[1] - self) - delta):
                        self.flee[1] = mob
        
        if self.target[1] != None and distance(self.target[1] - self) > self.sight:
            self.target[1] = None
        if self.flee[1] != None and distance(self.flee[1] - self) > self.sight:
            self.flee[1] = None
        
        q_key = []
        for mob in (self.target[1], self.flee[1]):
            theta = None if mob == None else angle(mob - self)
            #t.append(theta)
            quad = self.q_table.get_quad(theta)
            
            rng = None if mob == None else distance(mob - self)
            #r.append(rng)
            band = self.q_table.get_range(rng)
            q_key.append((quad, band))
        
        q_key = tuple(q_key)

        return q_key

    def action(self, epsilon=0, q_key=None, max_dims=(0, 0)):
        #t = []
        #r = []
        
        if random.random() > epsilon:
            choice = np.argmax(self.q_table.table[q_key])
        else:
            choice = random.randint(0,16)

        # 18 possible actions: move in 8 inter/cardinal directions, at wander/run pace
        # 17 is random
        # 0 is hold
        # 1 is north/wander, clockwise to 8
        # 9 is north/run, clockwise to 16

        if choice == 17:
            choice = random.randint(0,16)

        # east/west component
        if choice in (2, 3, 4, 10, 11, 12):
            dx = 1
        elif choice in (6, 7, 8, 14, 15, 16):
            dx = -1
        else:
            dx = 0
        
        # north/south component
        if choice in (8, 1, 2, 16, 9, 10):
            dy = -1  # flipped for pixels!
        elif choice in (4, 5, 6, 12, 13, 14):
            dy = 1
        else:
            dy = 0
        
        if choice > 8:
            run = True
            dx *= self.speed[1]
            dy *= self.speed[1]
        else:
            run = False
            dx *= self.speed[0]
            dy *= self.speed[0]
            
        mx, my = self.move(dx=dx, dy=dy, run=run, max_dims=max_dims)
        return mx, my, choice  #, t, r
        
    def move(self, dx=None, dy=None, run=False, max_dims=(0, 0)):
        dx = dx if isinstance(dx, (int, float)) else random.randint(0, self.speed[1])
        dy = dy if isinstance(dy, (int, float)) else random.randint(0, self.speed[1])
        
        ds = (dx ** 2 + dy ** 2) ** 0.5
        if ds != 0:
            scale = self.speed[1] / ds if run else self.speed[0] / ds
        else:
            scale = 0
        mx = round(dx * scale, 2)
        my = round(dy * scale, 2)

        mx = 0 - self.x if self.x + mx < 0 else mx  # left bound
        mx = max_dims[0] - self.x if max_dims[0] < self.x + mx else mx  # right bound
        self.x += mx
        
        my = 0 - self.y if self.y + my < 0 else my  # upper bound
        my = max_dims[1] - self.y if max_dims[1] < self.y + my else my  # lower bound
        self.y += my
        
        self.moves.append((self.x, self.y))
        return mx, my
        
    def check(self, mobs=None, mx=0, my=0):
        move_reward = -1  # turn penalty
        move_reward -= (mx**2 + my**2) ** 0.5  # move penalty
        act_reward = 0

        eaten_mobs = []
        
        for mob_type, mob_list in mobs.items():
            if mob_type == self.target[0]:
                for mob in mob_list:
                    ds = distance(mob - self)
                    if mob.alive and ds < (self.r + mob.r):
                        # eat the prey/food, be rewarded
                        self.health += mob.health
                        act_reward += mob.health_init
                        self.target[1] = None
                        mob.health = 0
                        mob.alive = False
                        eaten_mobs.append(mob.serial)
            elif mob_type == self.flee[0]:
                for mob in mob_list:
                    ds = distance(mob - self)
                    if mob.alive and ds < (self.r + mob.r):
                        # pentalty for being eaten
                        act_reward -= self.health_init
        
        reward = (move_reward, act_reward)
        return reward, eaten_mobs

    def update_q(self, mobs=None, q_key=((None, None), (None, None)), choice=-1, reward=0):
        # make recursive based on subsequent repeats of same action, or best action?
        current_q = self.q_table.table[q_key][choice]

        new_q_key = self.observe(mobs=mobs)  # sentdex for advice
        max_future_q = np.max(self.q_table.table[new_q_key])

        move_reward, act_reward = reward
        reward_sum = move_reward + act_reward
        if act_reward > 0:  # eating
            new_q = act_reward
        else:
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward_sum + self.discount * max_future_q)

        self.q_table.table[q_key][choice] = new_q
        
    def display(self, gameDisplay=None):
        if gameDisplay:
            
            # show move history
            if self.show_moves and len(self.moves) > 0:
                if self.show_moves == True:
                    start = 0
                elif len(self.moves) < self.show_moves:
                    start = 0
                else:
                    start = len(self.moves) - self.show_moves
                for pt in range(start, len(self.moves) - 1):
                    pygame.draw.line(gameDisplay, self.color, self.moves[pt], self.moves[pt+1], 1)
                pygame.draw.line(gameDisplay, self.color, self.moves[-1], (self.x, self.y), 1)
            
            # show current location
            try:
                pygame.draw.ellipse(gameDisplay, self.color, (self.x - self.r, self.y - self.r, self.r * 2, self.r * 2))  # physical, pygame
            except:
                pygame.draw.circle(gameDisplay, self.color, (self.x, self.y), self.r)  # pygame_sdl2
            
            # show target/flee
            if self.target[1]:
                pygame.draw.line(gameDisplay, WHITE, (self.target[1].x, self.target[1].y), (self.x, self.y), 1)
            #elif self.target[0] != None:
                #pygame.draw.line(gameDisplay, WHITE, (0, 0), (self.x, self.y), 1)
            if self.flee[1]:
                pygame.draw.line(gameDisplay, WHITE, (self.flee[1].x, self.flee[1].y), (self.x, self.y), 3)
            #elif self.flee[0] != None:
                #pygame.draw.line(gameDisplay, WHITE, (self.max_x, self.max_y), (self.x, self.y), 1)
            
            # show sight ring
            if self.sight >= 1:
                try:
                    pygame.draw.ellipse(gameDisplay, self.color, (self.x - self.sight, self.y - self.sight, self.sight * 2, self.sight * 2), 1)  # sight range, pygame
                except:
                    pygame.draw.circle(gameDisplay, self.color, (self.x, self.y), self.sight, width=1)  # pygame_sdl2
        
        else:
            raise RuntimeError('Error drawing {}'.format(self.__class__))


class Food(Mob):
    def __init__(self, x=None, y=None):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Food not passed position in init!')
        super().__init__(x=x, y=y)

        self.health_init = 9
        self.health = self.health_init
        self.color = (0, 255, 0)
    
    def observe(self, *args, **kwargs):
        return None, None  # return a tuple of no target/flee
    def action(self, *args, **kwargs):
        self.health += 0.2  # grow!
        return 0, 0, 0
    def check(self, *args, **kwargs):
        return (0,0), []
    def update_q(self, *args, **kwargs):
        pass


class Prey(Mob):
    sight = 60
    
    def __init__(self, x=None, y=None, load=False):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Prey not passed position in init!')
        super().__init__(x=x, y=y)
        
        self.health_init = 25
        self.health = self.health_init
        
        self.target = ['Food', None]  # type, mob
        self.flee = ['Predator', None]
        
        #self.sight = 60
        self.bands = 4
        self.slices = 8
        self.speed = (self.health_init ** 0.5, self.sight / 4)  # wander, run
        
        self.color = (0, 0, 255)
        self.show_moves = 100
        
        self.learning_rate = 0.08
        self.discount = 0.67
        
        if not load:
            for key in self.q_table:
                self.q_table[key] = Q_table(r=self.sight, bands=self.bands, slices=self.slices)
        else:
            for key in self.q_table:
                self.q_table[key] = Q_table(r=self.sight, bands=self.bands, slices=self.slices, load=load)


class Predator(Mob):
    sight = 100
    
    def __init__(self, x=None, y=None, load=False):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Predator not passed position in init!')
        super().__init__(x=x, y=y)
        
        self.health_init = 50
        self.health = self.health_init
        
        self.target = ['Prey', None]
        
        #self.sight = 200
        self.bands = 8
        self.slices = 16
        self.speed = (self.health_init ** 0.5, self.sight / 4)
        
        self.color = (255, 0, 0)
        self.show_moves = True
        
        self.learning_rate = 0.12  # faster learner
        self.discount = 0.9  # with better time pref than prey
        
        if not load:
            self.q_table['target'] = Q_table(r=self.sight, bands=self.bands, slices=self.slices)
        else:
            self.q_table['target'] = Q_table(r=self.sight, bands=self.bands, slices=self.slices, load=load)
        
        self.log = False  # open('resources/pred.log', 'w')
        if self.log:
            self.log.write('{}, {}\n'.format(self.__class__, self.serial))
            #self.log.write('{}\n'.format(self.q_table['target'].table.keys()))
            self.log.write('{}\n{}\n'.format(self.q_table['target'].quads, self.q_table['target'].ranges))
            #self.log.write('{}\n'.format(self.q_table['target'].table))

    
    def action(self, epsilon=0, q_key=None, max_dims=(0, 0)):
        # movement logging for debugging
        mx, my, choice = super().action(epsilon=epsilon, q_key=q_key, max_dims=max_dims)  #t, r debug
        
        if self.log:
            self.log.write('choice: {:<2}  |  move: ({:>6}, {:>6})\n'.format(choice, round(mx, 2), round(my, 2)))
            #self.log.write('angles: {}  |  ranges: {}\n'.format(t, r))
            
        return mx, my, choice

    def check(self, mobs=None, mx=0, my=0):
        reward, eaten_mobs = super().check(mobs=mobs, mx=mx, my=my)
        
        if len(eaten_mobs) > 0 and self.log:
            self.log.write('** ATE: {}\n'.format(eaten_mobs))
        
        return reward, eaten_mobs
