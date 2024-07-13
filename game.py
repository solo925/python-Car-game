import pygame
import random
import numpy as np
import pickle
from time import sleep

# Define actions
ACTIONS = ["LEFT", "RIGHT", "STAY"]

# Define Q-learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate

# Discretize the state space
STATE_SPACE_X = np.linspace(310, 460, num=5)  # Discretized car x position
STATE_SPACE_Y = np.linspace(-600, 600, num=10)  # Discretized enemy car y position

def get_discrete_state(car_x, enemy_y):
    car_x_idx = (np.abs(STATE_SPACE_X - car_x)).argmin()
    enemy_y_idx = (np.abs(STATE_SPACE_Y - enemy_y)).argmin()
    return car_x_idx, enemy_y_idx

class CarRacing:
    def __init__(self):
        pygame.init()
        self.display_width = 800
        self.display_height = 600
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.clock = pygame.time.Clock()
        self.gameDisplay = None
        self.initialize()

        # Initialize Q-table
        self.q_table = np.zeros((len(STATE_SPACE_X), len(STATE_SPACE_Y), len(ACTIONS)))

    def initialize(self):
        self.crashed = False
        self.carImg = pygame.image.load('/home/davinci/Desktop/python/Car Game/img/car.png')
        self.car_x_coordinate = (self.display_width * 0.45)
        self.car_y_coordinate = (self.display_height * 0.8)
        self.car_width = 49
        self.enemy_car = pygame.image.load('/home/davinci/Desktop/python/Car Game/img/enemy_car_1.png')
        self.enemy_car_startx = random.randrange(310, 450)
        self.enemy_car_starty = -600
        self.enemy_car_speed = 5
        self.enemy_car_width = 49
        self.enemy_car_height = 100
        self.bgImg = pygame.image.load("/home/davinci/Desktop/python/Car Game/img/back_ground.jpg")
        self.bg_x1 = (self.display_width / 2) - (360 / 2)
        self.bg_x2 = (self.display_width / 2) - (360 / 2)
        self.bg_y1 = 0
        self.bg_y2 = -600
        self.bg_speed = 3
        self.count = 0

    def car(self, car_x_coordinate, car_y_coordinate):
        self.gameDisplay.blit(self.carImg, (car_x_coordinate, car_y_coordinate))

    def racing_window(self):
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Car Game')
        self.run_car()

    def run_car(self):
        while not self.crashed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.crashed = True

            # Agent logic
            self.agent_move()

            self.gameDisplay.fill(self.black)
            self.back_ground_raod()

            self.run_enemy_car(self.enemy_car_startx, self.enemy_car_starty)
            self.enemy_car_starty += self.enemy_car_speed

            if self.enemy_car_starty > self.display_height:
                self.enemy_car_starty = 0 - self.enemy_car_height
                self.enemy_car_startx = random.randrange(310, 450)

            self.car(self.car_x_coordinate, self.car_y_coordinate)
            self.highscore(self.count)
            self.count += 1
            if self.count % 100 == 0:
                self.enemy_car_speed += 1
                self.bg_speed += 1

            if self.car_y_coordinate < self.enemy_car_starty + self.enemy_car_height:
                if self.car_x_coordinate > self.enemy_car_startx and self.car_x_coordinate < self.enemy_car_startx + self.enemy_car_width or self.car_x_coordinate + self.car_width > self.enemy_car_startx and self.car_x_coordinate + self.car_width < self.enemy_car_startx + self.enemy_car_width:
                    self.crashed = True
                    self.display_message("Game Over !!!")

            if self.car_x_coordinate < 310 or self.car_x_coordinate > 460:
                self.crashed = True
                self.display_message("Game Over !!!")

            pygame.display.update()
            self.clock.tick(60)

    def agent_move(self):
        # Get the current state
        car_x_idx, enemy_y_idx = get_discrete_state(self.car_x_coordinate, self.enemy_car_starty)

        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < EPSILON:
            action_idx = random.choice(range(len(ACTIONS)))  # Explore
        else:
            action_idx = np.argmax(self.q_table[car_x_idx, enemy_y_idx])  # Exploit

        action = ACTIONS[action_idx]

        # Take action
        if action == "LEFT" and self.car_x_coordinate > 310:
            self.car_x_coordinate -= 50
        elif action == "RIGHT" and self.car_x_coordinate < 460:
            self.car_x_coordinate += 50

        # Get new state
        new_car_x_idx, new_enemy_y_idx = get_discrete_state(self.car_x_coordinate, self.enemy_car_starty)

        # Get reward
        reward = 1  # Default reward
        if self.car_y_coordinate < self.enemy_car_starty + self.enemy_car_height:
            if self.car_x_coordinate > self.enemy_car_startx and self.car_x_coordinate < self.enemy_car_startx + self.enemy_car_width or self.car_x_coordinate + self.car_width > self.enemy_car_startx and self.car_x_coordinate + self.car_width < self.enemy_car_startx + self.enemy_car_width:
                reward = -100
        if self.car_x_coordinate < 310 or self.car_x_coordinate > 460:
            reward = -100

        # Update Q-table
        old_value = self.q_table[car_x_idx, enemy_y_idx, action_idx]
        next_max = np.max(self.q_table[new_car_x_idx, new_enemy_y_idx])
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        self.q_table[car_x_idx, enemy_y_idx, action_idx] = new_value

        if reward == -100:
            self.crashed = True
            self.display_message("Game Over !!!")

    def display_message(self, msg):
        font = pygame.font.SysFont("comicsansms", 72, True)
        text = font.render(msg, True, (255, 255, 255))
        self.gameDisplay.blit(text, (400 - text.get_width() // 2, 240 - text.get_height() // 2))
        pygame.display.update()
        self.clock.tick(60)
        sleep(1)
        self.initialize()
        self.racing_window()

    def back_ground_raod(self):
        self.gameDisplay.blit(self.bgImg, (self.bg_x1, self.bg_y1))
        self.gameDisplay.blit(self.bgImg, (self.bg_x2, self.bg_y2))

        self.bg_y1 += self.bg_speed
        self.bg_y2 += self.bg_speed

        if self.bg_y1 >= self.display_height:
            self.bg_y1 = -600

        if self.bg_y2 >= self.display_height:
            self.bg_y2 = -600

    def run_enemy_car(self, thingx, thingy):
        self.gameDisplay.blit(self.enemy_car, (thingx, thingy))

    def highscore(self, count):
        font = pygame.font.SysFont("lucidaconsole", 20)
        text = font.render("Score : " + str(count), True, self.white)
        self.gameDisplay.blit(text, (0, 0))


if __name__ == '__main__':
    car_racing = CarRacing()
    car_racing.racing_window()
