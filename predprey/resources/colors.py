import random as rnd

black = (0, 0, 0)
white = (255, 255, 255)


def random_range(red, green, blue):
    r = rnd.randint(red[0], red[1])
    g = rnd.randint(green[0], green[1])
    b = rnd.randint(blue[0], blue[1])
    
    return (r, g, b)

def random():
    return random_range((0, 255), (0, 255), (0, 255))