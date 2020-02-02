import random

try:
    import pygame_sdl2
    pygame_sdl2.import_as_pygame()
except ImportError:
    pass
import pygame

class Mob():
    def __init__(self, x=None, y=None, dims=None):
        if not x or not y:
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
        self.color = (0, 0, 0)
        
        self.sight = 0
        self.speed = (0, 0)
        
        self.q_table = None
        self.learning_rate = 0  # 0: no learning, 1: no memory
        self.discount = 0  # 0: no time preference, 1: infinite time preference
        
    def __str__(self):
        return '{} at ({}, {})'.format(self.__class__, self.x, self.y)
    
    def action(self, choice):
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
        else:
            run = False
            
        self.move(dx=dx, dy=dy, run=run)
    
    def move(self, dx=None, dy=None, run=False):
        dx = dx if dx else random.randint(0, self.speed[1])
        dy = dy if dy else random.randint(0, self.speed[1])
        
        ds = (dx ** 2 + dy ** 2) ** 0.5
        if ds != 0:
            scale = self.speed[1] / ds if run else self.speed[0] / ds
        else:
            scale = 0
        
        mx = round(dx * scale, 1)
        mx = 0 - self.x if self.x + mx < 0 else mx
        mx = self.max_x - self.x if self.max_x < self.x + mx else mx
        self.x += mx
        
        my = round(dy * scale,1)
        my = 0 - self.y if self.y + my < 0 else my
        my = self.max_y - self.y if self.max_y < self.y + my else my
        self.y += my
        
        return (mx, my)
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    def display(self, gameDisplay=None):
        if gameDisplay:
            pygame.draw.circle(gameDisplay, self.color, (self.x, self.y), self.r)
        else:
            raise RuntimeError('Error drawing {}'.format(self.__class__))


class Food(Mob):
    def __init__(self, x=None, y=None, dims=None):
        if not x or not y:
            raise ValueError('Food not passed position in init!')
        if dims and isinstance(dims, (list, tuple)):
            super().__init__(x=x, y=y, dims=dims)
        else:
            raise ValueError('No app dimensions passed to Food!')
        
        self.r = 5
        self.flee = Prey  # even though we can't move
        self.color = (0, 255, 0)


class Prey(Mob):
    def __init__(self, x=None, y=None, dims=None):
        if not x or not y:
            raise ValueError('Prey not passed position in init!')
        if dims and isinstance(dims, (list, tuple)):
            super().__init__(x=x, y=y, dims=dims)
        else:
            raise ValueError('No app dimensions passed to Prey!')
        
        self.r = 5
        
        self.target = Food
        self.flee = Predator
        self.color = (0, 0, 255)
        
        self.sight = 50
        self.speed = (2, 6)  # wander, run
        
        self.learning_rate = 0.08
        self.discount = 0.67


class Predator(Mob):
    def __init__(self, x=None, y=None, dims=None):
        if not x or not y:
            raise ValueError('Predator not passed position in init!')
        if dims and isinstance(dims, (list, tuple)):
            super().__init__(x=x, y=y, dims=dims)
        else:
            raise ValueError('No app dimensions passed to Predator!')
        
        self.r = 10
        
        self.target = Prey
        self.flee = None
        self.color = (255, 0, 0)
        
        self.sight = 100
        self.speed = (4, 10)
        
        self.learning_rate = 0.12  # faster learner
        self.discount = 0.95  # with better time pref than prey
