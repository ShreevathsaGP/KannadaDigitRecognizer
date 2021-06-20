# imports
import pygame
import random
import feedforward
from feedforward import *
import numpy as np
import cv2

# constants
model_path = "model/final_checkpoint.pth.tar"
image_dims = (28, 28)
input_size = image_dims[0] * image_dims[1]
no_classes = 10
window_width = 700
window_height = 730
x_buffer = 15
y_buffer = 15
y_bottom = x_buffer
y_top = window_height - (window_width - (2 * x_buffer))

# pygame
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Kannada Digit Recognizer")
font_size = 45
font = pygame.font.SysFont('timesnewroman.ttf', font_size)
text_colour = (235, 223, 199)
fps_clock = pygame.time.Clock()
fps = 60

# model
checkpoint = torch.load(model_path)
model = feedforward.NN(input_size, no_classes)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# map list index
def map_to_index(x1, y1, x2, y2):
    return (int(x2 / x1), int(y2 / y1))

# digit canvas
class DigitCanvas:
    # boundaries
    x_bounds = [x_buffer, (window_width - x_buffer)]
    y_bounds = [y_top, (window_height - y_bottom)]
    grid_dims = image_dims
    
    # measurements
    height = y_bounds[1] - y_bounds[0]
    width = x_bounds[1] - x_bounds[0]
    x, y = x_bounds[0], y_bounds[0]
    pos = (x, y)
    line_width = 2

    # box dims
    x_increment = width / grid_dims[0]
    y_increment = height / grid_dims[1]
    box_dims = (x_increment, y_increment)

    def __init__(self):
        self.surface = pygame.Surface((self.width, self.height))
        self.surface.fill((0,0,0))
        self.image = [[0 for _ in range(self.grid_dims[1])] for _ in range(self.grid_dims[0])]

    def render(self):
        colour = (170, 120, 20)
        
        # boxes
        for c, _ in enumerate(self.image):
            for r, __ in enumerate(self.image[c]):
                if self.image[c][r] == 1:
                    pygame.draw.rect(self.surface, colour, (c * self.x_increment + self.line_width, r * self.y_increment + self.line_width, *self.box_dims))
                else:
                    pygame.draw.rect(self.surface, (0,0,0), (c * self.x_increment + self.line_width, r * self.y_increment + self.line_width, self.x_increment - self.line_width, self.y_increment - self.line_width))

        # vertical lines 
        pygame.draw.line(self.surface, colour, (0, 0), (0, self.height), self.line_width)
        for i in range(self.grid_dims[0] - 1):
            x = self.x_increment * (i + 1)
            pygame.draw.line(self.surface, colour, (x, 0), (x, self.height), self.line_width)
        pygame.draw.line(self.surface, colour, (self.width - self.line_width, 0), (self.width - self.line_width, self.height), self.line_width)

        # horizontal lines
        pygame.draw.line(self.surface, colour, (0,0), (self.width, 0), self.line_width)
        for i in range(self.grid_dims[1] - 1):
            y = self.y_increment * (i + 1)
            pygame.draw.line(self.surface, colour, (0, y), (self.width, y), self.line_width)
        pygame.draw.line(self.surface, colour, (0, self.height - self.line_width), (self.width, self.height - self.line_width), self.line_width)

    def click(self, point) :
        # check for in bounds
        if point[0] <= self.x_bounds[0] or point[0] >= self.x_bounds[1] or point[1] <= self.y_bounds[0] or point[1] >= self.y_bounds[1]:
            return

        point = (point[0] - self.x_bounds[0], point[1] - self.y_bounds[0])
        index = map_to_index(*self.box_dims, *point)

        self.image[index[0]][index[1]] = 1

    def clear(self):
        self.image = [[0 for _ in range(self.grid_dims[0])] for _ in range(self.grid_dims[1])]

canvas = DigitCanvas()
prediction = None
confidence = None

# main loop
interaction = True
dragging = False
while interaction:
    # iterate over events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            interaction = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # left click
            if event.button == 1:

                dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            # left click
            if event.button == 1:
                dragging = False
        
        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                canvas.click(event.pos)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                canvas.clear()

            elif event.key == pygame.K_RETURN:
                with torch.no_grad():
                    x = torch.Tensor(canvas.image).to(torch.float32) * 255
                    x = x.transpose(0, 1)
                    # image = x
                    x = x.flatten()
                    x = x.reshape(1, x.shape[0]) 
                    y = model(x)

                    # # show digit image
                    # image = image.numpy()
                    # scale = 2000
                    # width = int(image.shape[0] * (scale / 100))
                    # height = int(image.shape[1] * (scale / 100))
                    # image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
                    # cv2.imshow("digit", image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    prediction = int(y.max(dim = 1)[1])
                    confidence = round(float(y.max(dim = 1)[0] / torch.sum(y, dim = 1)), 1)
    
    # canvas render
    canvas.render()
    window.fill((0,0,0))
    window.blit(canvas.surface, canvas.pos)
    
    # model render
    prediction = prediction if prediction != None else "NaN"
    confidence = confidence if confidence != None else "NaN"

    text = font.render("Confidence: {}".format(confidence), True, text_colour)
    rect = text.get_rect()
    rect.x += x_buffer + 45
    rect.y += y_buffer
    window.blit(text, rect)

    text = font.render("Prediction: {}".format(prediction), True, text_colour)
    rect = text.get_rect()
    rect.x = (window_width / 2) + 45
    rect.y += y_buffer
    window.blit(text, rect)

    # pygame
    pygame.display.update()
    fps_clock.tick(fps)

