import random
import math
import numpy as np
import pickle
import time
import resources.colors as colors

try:
    import pygame_sdl2
    pygame_sdl2.import_as_pygame()
except ImportError:
    pass
import pygame


def angle(coords=(0,0)):
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    
    x, y = coords
    
    r2d = 180 / np.pi
    ang = np.arctan2(x, 0-y) * r2d
    
    return ang


def distance(coords=(0,0)):
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    
    return ((coords[0]**2)+(coords[1]**2)**0.5)


class Q_table():
    
    def __init__(self, r=0, slices=0, actions=17, load=False):
        slices = int(slices) if slices >= 8 else 8  # at least 1 per move direction
        theta = 360 / slices
        half = theta / 2
        
        self.quads = [(i * theta - half, i * theta + half) for i in range(slices)]
        self.ranges = [(r * i / 4, r * (i+1) / 4) for i in range(4)]
        
        self.angle_bounds = (self.quads[0][0], self.quads[-1][-1])
        
        if load:
            self.table = self.q_table_setup(actions=actions)  # FIXME pickle in
            #with open(load, 'rb') as f:
                #self.table = pickle.load(f)
        else:
            self.table = self.q_table_setup(actions=actions)
        
    def q_table_setup(self, actions=1):
        table = {}
        
        # keys will be tuples of "quadrants" and range bands, which are stored as [L, R) angles and [min, max) distance respectively, for both a target and whatever is being fleed
        angle_keys = [None] + list(range(len(self.quads)))
        range_keys = [None] + list(range(len(self.ranges)))
        
        table = {((ta, tr), (fa, fr)): [np.random.uniform(-actions, 0) for i in range(actions)] for ta in angle_keys for tr in range_keys for fa in angle_keys for fr in range_keys}
        
        return table
    
    def get_quad(self, addrac):
        theta = angle(addrac)
        if theta == None:
            return None
        
        if theta < self.angle_bounds[0]:
            theta += 360
        if theta >= self.angle_bounds[1]:
            theta -= 360
        
        
    
    def get_range(self, addrac):
        r = distance(addrac)
        if r == None:
            return None
        
        
    
    def save(self, serial):
        pass  # FIXME pickle out
        #with open('{}-{}.Q'.format(self.__class__, serial), 'wb') as f:  # microseconds
            #pickle.dump(self.table, f)


class Mob():
    def __init__(self, x=None, y=None, dims=None):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Mob not passed position in init!')
        self.x = x
        self.y = y
        self.health = 0
        
        if dims and isinstance(dims, (list, tuple)):
            self.max_x = dims[0]
            self.max_y = dims[1]
        else:
            raise ValueError('No app dimensions passed to mob!')
        
        self.alive = True
        self.serial = int(time.time() * 10**6) % (10**7)
        
        self.target = [None, None, None]
        self.flee = [None, None, None]
        self.reward = 0
        
        self.sight = 0
        self.slices = 0
        self.speed = (0, 0)
        self.moves = [(self.x, self.y)]
        
        self.color = (0, 0, 0)
        self.show_moves = False  # False or number (True for all)
        
        self.learning_rate = 0  # 0: no learning, 1: no memory
        self.discount = 0  # 0: no time preference, 1: infinite time preference
        
        self.q_table = None
        
    def __str__(self):
        return '{} at ({}, {})'.format(self.__class__, self.x, self.y)
    def __repr__(self):
        return str(self)

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
        
        self.alive = True
        self.target[1] = None
        self.flee[1] = None
        self.moves = []
        
    def observe(self, mobs=None):
        # most of this os to prevent mobs from switching targets mid-chase
        
        closest_target = [None, (10**5, 10**5), (2*10**5)**.5]  # to ensure it is reset
        current_target = [self.target[1], self.target[2], distance(self.target[2])]  # serial, coords, distance
        
        closest_flee = [None, (10**5, 10**5), (2*10**5)**.5]
        current_flee = [self.flee[1], self.flee[2], distance(self.flee[2])]
        
        delta = self.speed[1] - self.speed[0]  # how much closer to switch targets/flee
        
        for mob_type, mob_list in mobs.items():
            if mob_type == self.target[0]:
                for mob in mob_list:
                    ds = distance(mob - self)
                    if mob.serial == current_target[0]:
                        current_target[1] = mob - self
                        current_target[2] = distance(mob - self)
                    if ds < closest_target[2]:
                        closest_target = [mob.serial, mob - self, distance(mob - self)]
            elif mob_type == self.flee[0]:
                for mob in mob_list:
                    ds = distance(mob - self)
                    if mob.serial == current_flee[0]:
                        current_flee[1] = mob - self
                        current_flee[2] = distance(mob - self)
                    if ds < closest_flee[2]:
                        closest_flee = [mob.serial, mob - self, distance(mob - self)]
        
        if (current_target[2] and (closest_target[2] <= current_target[2] - delta)) or (closest_target[2] <= self.speed[1]) or (self.target[0] and current_target[0] == None):
            current_target = closest_target
            self.target = [self.target[0], current_target[0], current_target[1]]
        if (current_flee[2] and (closest_flee[2] <= current_flee[2] - delta)) or (closest_flee[2] <= self.speed[1]) or (self.flee[0] and current_flee[0] == None):
            current_flee = closest_flee
            self.flee = [self.flee[0], current_flee[0], current_flee[1]]
            
        return (current_target[1], current_flee[1])  # mob - self (x, y)

    def action(self, epsilon=0, observation=None):
        if random.random() > epsilon:
            choice = random.randint(0,16)  # np.argmax
            q_key = []
            for addrac in observation:
                quad = self.q_table.get_quad(addrac)
                rng = self.q_table.get_range(addrac)
                q_key.append((quad, rng))
            q_key = tuple(q_key)
        else:
            choice = random.randint(0,16)

        # 17 possible actions: move in 8 inter/cardinal directions, at wander/run pace
        # 0 is hold
        # 1 is north/wander, clockwise to 8
        # 9 is north/run, clockwise to 16

        # north/south component
        if choice in (8, 1, 2, 16, 9, 10):
            dx = -1  # flipped for pixels!
        elif choice in (4, 5, 6, 12, 13, 14):
            dx = 1
        else:
            dx = 0
        
        # east/west component
        if choice in (2, 3, 4, 10, 11, 12):
            dy = 1
        elif choice in (6, 7, 8, 14, 15, 16):
            dy = -1
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
            
        mx, my = self.move(dx=dx, dy=dy, run=run)
        return (choice, mx, my)
        
    def move(self, dx=None, dy=None, run=False):
        dx = dx if isinstance(dx, int) else random.randint(0, self.speed[1])
        dy = dy if isinstance(dy, int) else random.randint(0, self.speed[1])
        
        ds = (dx ** 2 + dy ** 2) ** 0.5
        if ds != 0:
            scale = self.speed[1] / ds if run else self.speed[0] / ds
        else:
            scale = 0
        
        mx = round(dx * scale, 2)
        mx = 0 - self.x if self.x + mx < 0 else mx  # left bound
        mx = self.max_x - self.x if self.max_x < self.x + mx else mx  # right bound
        self.x += mx
        
        my = round(dy * scale, 2)
        my = 0 - self.y if self.y + my < 0 else my  # upper bound
        my = self.max_y - self.y if self.max_y < self.y + my else my  # lower bound
        self.y += my
        
        self.moves.append((self.x, self.y))
        self.reward -= (mx**2 + my**2)  # 1/2mv**2: moving further costs exponentially
        return (mx, my)
        
    def check(self, mobs=None):
        pass

    def update_q(self):
        pass
        
    def display(self, gameDisplay=None):
        if gameDisplay:
            
            # show move history
            if self.show_moves:
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
                pygame.draw.line(gameDisplay, colors.white, self.target[2], (self.x, self.y), 1)
            if self.flee[1]:
                pygame.draw.line(gameDisplay, colors.white, self.flee[2], (self.x, self.y), 1)
            
            # show sight ring
            if self.sight >= 1:
                try:
                    pygame.draw.ellipse(gameDisplay, self.color, (self.x - self.sight, self.y - self.sight, self.sight * 2, self.sight * 2), 1)  # sight range, pygame
                except:
                    pygame.draw.circle(gameDisplay, self.color, (self.x, self.y), self.sight, width=1)  # pygame_sdl2
        
        else:
            raise RuntimeError('Error drawing {}'.format(self.__class__))


class Food(Mob):
    def __init__(self, x=None, y=None, dims=None):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Food not passed position in init!')
        if dims and isinstance(dims, (list, tuple)):
            super().__init__(x=x, y=y, dims=dims)
        else:
            raise ValueError('No app dimensions passed to Food!')
        
        self.health = 4
        self.color = (0, 255, 0)
        
    def action(self, epsilon=0, observation=None):
        self.reward += 1  # grow!


class Prey(Mob):
    def __init__(self, x=None, y=None, dims=None, load=False):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Prey not passed position in init!')
        if dims and isinstance(dims, (list, tuple)):
            super().__init__(x=x, y=y, dims=dims)
        else:
            raise ValueError('No app dimensions passed to Prey!')
        
        self.health = 20
        
        self.target = ['Food', None, None]  # type, serial, coords
        self.flee = ['Predator', None, None]
        
        self.sight = 50
        self.slices = 8
        self.speed = (2, 6)  # wander, run
        
        self.color = (0, 0, 255)
        self.show_moves = 100
        
        self.learning_rate = 0.08
        self.discount = 0.67
        
        if not load:
            self.q_table = Q_table(r=self.sight, slices=self.slices)
        else:
            self.q_table = Q_table(load=load)


class Predator(Mob):
    def __init__(self, x=None, y=None, dims=None, load=False):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Predator not passed position in init!')
        if dims and isinstance(dims, (list, tuple)):
            super().__init__(x=x, y=y, dims=dims)
        else:
            raise ValueError('No app dimensions passed to Predator!')
        
        self.health = 50
        
        self.target = ['Prey', None, None]
        
        self.sight = 100
        self.slices = 16
        self.speed = (4, 10)
        
        self.color = (255, 0, 0)
        self.show_moves = True
        
        self.learning_rate = 0.12  # faster learner
        self.discount = 0.95  # with better time pref than prey
        
        if not load:
            self.q_table = Q_table(r=self.sight, slices=self.slices)
        else:
            self.q_table = Q_table(load=load)
        
        self.log = open('resources/pred.log', 'w')
        self.log.write('{}, {}'.format(self.__class__, self.serial))
        self.log.write('{}\n'.format(self.q_table.table))

    def observe(self, mobs=None):
        target, flee = super().observe(mobs=mobs)
        self.log.write('target: {:<10}  | flee: {}'.format(self.target, self.flee))
        return (target, flee)
    
    def action(self, epsilon=0, observation=None):
        # movement logging for debugging
        choice, mx, my = super().action(epsilon=epsilon, observation=observation)
        if self.log:
            self.log.write('choice: {:<2}  |  move: ({:>6}, {:>6})  |  ds = {}\n'.format(choice, round(mx, 2), round(my, 2), round((mx**2 +my**2)**.5, 2)))
        return (choice, mx, my)
