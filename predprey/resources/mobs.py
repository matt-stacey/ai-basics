import random
import math
import numpy as np
'''
try:
    import pygame_sdl2
    pygame_sdl2.import_as_pygame()
except ImportError:
    pass
'''
import pygame


def angle(x=None, y=None):
    if not x or not y:
        return None
    
    r2d = 180 / np.pi
    
    ang = np.arctan2(x, 0-y) * r2d
    
    while ang < 0:
        ang += 360
    while ang > 360:
        ang -= 360
    
    return ang


class Q_table():
    
    def __init__(self, r=0, slices=0, load=False):
        slices = int(slices) if slices >= 8 else 8  # at least 1 per move direction
        self.theta = 360 / slices
        self.ranges = (0, r /4, r / 2, r * 3 / 4, r)
        
        if load:
            self.table = self.q_table_setup()  # FIXME pickle in
        else:
            self.table = self.q_table_setup()
        
    def q_table_setup(self):
        table = {}
        
        # keys will be tuples of [L, R) angles and [min, max) distance
        half = self.theta / 2
        
        start_angle = -half
        L = start_angle
        while L < start_angle + 359.9:  # slop for non
            R = L + self.theta
            for r in range(len(self.ranges) - 1):
                angle_bounds = (L, R)
                range_bounds = (self.ranges[r], self.ranges[r+1])
                table[(angle_bounds, range_bounds)] = [np.random.uniform(-5, 0) for a in range(17)]  # for each action
            
            L = R
        
        return table

    def save(self):
        pass  # FIXME pickle out


class Mob():
    def __init__(self, x=None, y=None, dims=None):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Mob not passed position in init!')
        self.x = x
        self.y = y
        self.r = 0
        
        if dims and isinstance(dims, (list, tuple)):
            self.max_x = dims[0]
            self.max_y = dims[1]
        else:
            raise ValueError('No app dimensions passed to mob!')
        
        self.alive = True
        
        self.target = None
        self.flee = None
        self.reward = 0
        
        self.sight = 0
        self.speed = (0, 0)
        self.moves = [(self.x, self.y)]
        
        self.color = (0, 0, 0)
        self.show_moves = False  # False or number (True for all)
        
        self.learning_rate = 0  # 0: no learning, 1: no memory
        self.discount = 0  # 0: no time preference, 1: infinite time preference
        
        self.q_table = None
        
    def __str__(self):
        return '{} at ({}, {})'.format(self.__class__, self.x, self.y)
    
    def action(self, choice=0):
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
        return choice, mx, my
    
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
        return (mx, my)
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    def display(self, gameDisplay=None):
        if gameDisplay:
            # show current location
            try:
                pygame.draw.ellipse(gameDisplay, self.color, (self.x - self.r, self.y - self.r, self.r * 2, self.r * 2))  # physical, pygame
            except:
                pygame.draw.circle(gameDisplay, self.color, (self.x, self.y), self.r)  # pygame_sdl2
            if self.sight >= 1:
                try:
                    pygame.draw.ellipse(gameDisplay, self.color, (self.x - self.sight, self.y - self.sight, self.sight * 2, self.sight * 2), 1)  # sight range, pygame
                except:
                    pygame.draw.circle(gameDisplay, self.color, (self.x, self.y), self.sight, width=1)  # pygame_sdl2
            
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
        
        self.r = 2
        self.reward = 20
        self.color = (0, 255, 0)


class Prey(Mob):
    def __init__(self, x=None, y=None, dims=None, load=False):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError('Prey not passed position in init!')
        if dims and isinstance(dims, (list, tuple)):
            super().__init__(x=x, y=y, dims=dims)
        else:
            raise ValueError('No app dimensions passed to Prey!')
        
        self.r = 5
        
        self.target = Food
        self.flee = Predator
        self.reward = 100
        
        self.sight = 50
        self.speed = (2, 6)  # wander, run
        
        self.color = (0, 0, 255)
        self.show_moves = 100
        
        self.learning_rate = 0.08
        self.discount = 0.67
        
        if not load:
            self.q_table = Q_table(r=self.sight, slices=8)
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
        
        self.r = 10
        
        self.target = Prey
        self.flee = None
        
        self.sight = 100
        self.speed = (4, 10)
        
        self.color = (255, 0, 0)
        self.show_moves = True
        
        self.learning_rate = 0.12  # faster learner
        self.discount = 0.95  # with better time pref than prey
        
        if not load:
            self.q_table = Q_table(r=self.sight, slices=16)
        else:
            pass  # FIXME pickle in
        
        self.log = open('resources/pred.log', 'w')
        self.log.write('{}\n'.format(self.q_table.table))

    def action(self, choice=0):
        # movement logging for debugging
        choice, mx, my = super().action(choice=choice)
        if self.log:
            self.log.write('choice: {:<2}  |  move: ({:>6}, {:>6})  |  ds = {}\n'.format(choice, round(mx, 2), round(my, 2), round((mx**2 +my**2)**.5, 2)))
