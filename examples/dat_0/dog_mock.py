import pygame
import sys
import cv2
import pygame.camera
import os

class Dog:
    def __init__(self, screen):
        self.screen = screen
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.image = pygame.image.load('dog.png')
        self.rect = self.image.get_rect()
        self.rect.center = (screen.get_width() // 2, screen.get_height() // 2)
        self.speed = 5

    def move(self, keys):
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < self.screen.get_width():
            self.rect.x += self.speed
        if keys[pygame.K_UP] and self.rect.top > 0:
            self.rect.y -= self.speed
        if keys[pygame.K_DOWN] and self.rect.bottom < self.screen.get_height():
            self.rect.y += self.speed

    def draw(self):
        self.screen.blit(self.image, self.rect)

class Camera:
    def __init__(self):
        pygame.camera.init()
        self.cam = cv2.VideoCapture(0)

    def start(self, screen):
        while True:
            ret, frame = self.cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converts BGR to RGB for Pygame
            frame = cv2.resize(frame, (screen.get_width(), screen.get_height()))
            frame = cv2.flip(frame, 1)
            frame = pygame.surfarray.make_surface(frame)
            screen.blit(frame, (0, 0))
            pygame.display.update() ## Update the display with the drawn frame

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    return  # Stop camera on 's' key press

class Game:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Dog Control")
        self.clock = pygame.time.Clock()
        self.dog = Dog(self.screen)
        self.camera = Camera()

    def run(self):
        while True:
            self.screen.fill((255, 255, 255))
            keys = pygame.key.get_pressed()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        self.camera.start(self.screen)  # Open camera on 'c' key press

            self.dog.move(keys)
            self.dog.draw()

            pygame.display.update()
            self.clock.tick(30)

if __name__ == "__main__":
    game = Game()
    game.run()
