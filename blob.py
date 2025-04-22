import copy
import math
import random
from brain import Brain
from constants import *

class ManualBrain:
    def __init__(self, blob):
        self.blob = blob

    def forward(self):
        angle = self.blob.get_closest_food_angle()
        
        if abs(angle) < 15:
            return next(o for o in self.blob.outputs if o.name == "increase_speed")
        elif angle < 0:
            return next(o for o in self.blob.outputs if o.name == "rotate_left")
        else:
            return next(o for o in self.blob.outputs if o.name == "rotate_right")
        
class Input:
    def __init__(self, name: str, get_function: callable, min_value: float = 0.0, max_value: float = 1.0):
        self.name = name
        self.get_function = get_function
        self.min_value = min_value
        self.max_value = max_value
        self.value = 0.0
    
    def get(self):
        return self.get_function()
    
class Output:
    def __init__(self, name: str, action: callable):
        self.name = name
        self.action = action

    def perform(self):
        self.action()

class Blob:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0 # degrees
        self.speed = 0.0
        self.max_speed = 0.0
        self.min_speed = 0.0
        self.turn_speed = 0 # degrees per step
        self.life = 0
        self.max_life = MAX_LIFE
        self.detection_radius = 0.0

        self.closest_food_distance = 0.0
        self.closest_food_angle = 0.0

        self.generation = 0

        self.food_map = []
        self.food_eaten = 0

        self.id = f"{random.randint(1000000, 9000000)}_{self.generation}"

        self.generate_blob(manual=False)

    def generate_blob(self, manual: bool = False):
        self.x = random.uniform(MIN_X, MAX_X)
        self.y = random.uniform(MIN_Y, MAX_Y)
        self.orientation = random.uniform(0, 360)
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        self.turn_speed = random.uniform(MIN_TURN_SPEED, MAX_TURN_SPEED)
        self.speed = random.uniform(self.min_speed, self.max_speed)
        self.life = self.max_life
        # self.detection_radius = random.uniform(50, 100)
        self.detection_radius = 200

        self.inputs = [
            # Input("x", self.get_x, MIN_X, MAX_X),
            # Input("y", self.get_y, MIN_Y, MAX_Y),
            Input("orientation", self.get_orientation, 0, 360),
            # Input("speed", self.get_speed, self.min_speed, self.max_speed),
            Input("life", self.get_life, 0, self.max_life),
            Input("closest_food_distance", self.get_closest_food_distance, 0, self.detection_radius),
            Input("closest_food_angle", self.get_closest_food_angle, -180, 180),
        ]

        self.outputs = [
            Output("increase_speed", self.increase_speed),
            Output("decrease_speed", self.decrease_speed),
            Output("rotate_left", self.rotate_left),
            Output("rotate_right", self.rotate_right),
        ]
        self.outputs = [
            Output("rotate_left", self.rotate_left),
            Output("rotate_right", self.rotate_right),
            Output("nothing", self.nothing),
        ]

        if manual == True:
            self.brain = ManualBrain(self)
        else:
            self.brain = Brain()
            self.brain.inputs = self.inputs
            self.brain.outputs = self.outputs
            self.brain.generate_brain()

    def step(self, foods):
        if self.life > 0:
            # Update inputs for the brain
            self.food_map = foods

            action: Output = self.brain.forward()
            # Perform the action
            action.perform()

            # Move the blob
            self.move()

            # Decrease life
            self.life -= 1
            if self.life <= 0:
                self.die()

    # Getters for inputs
    def get_x(self):
        return self.x
    
    def get_y(self):    
        return self.y
    
    def get_orientation(self):
        return self.orientation

    def get_closest_food_info(self):
        closest_distance = self.detection_radius
        closest_angle = 0.0

        for food in self.food_map:
            dx = food[0] - self.x
            dy = food[1] - self.y
            distance = math.hypot(dx, dy)  # Más eficiente y legible

            if distance < closest_distance:
                closest_distance = distance
                absolute_angle = math.degrees(math.atan2(dy, dx))
                relative_angle = absolute_angle - self.orientation
                # Normalizar ángulo a [-180, 180]
                closest_angle = (relative_angle + 180) % 360 - 180

        return closest_distance, closest_angle

    def get_closest_food_distance(self):
        return self.get_closest_food_info()[0]

    def get_closest_food_angle(self):
        return self.get_closest_food_info()[1]
    
    def get_speed(self):
        return self.speed
    
    def get_life(self):
        return self.life

    def die(self):
        # print(f"Blob at ({self.x}, {self.y}) has died.")
        pass

    def set_position(self, x: float, y: float):
        self.x = x
        self.y = y

    def move(self):
        rad = math.radians(self.orientation)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

    # Possible Actions
    def increase_speed(self):
        self.speed += 0.01
        if self.speed > self.max_speed:
            self.speed = self.max_speed

    def decrease_speed(self):
        self.speed -= 0.01
        if self.speed < self.min_speed:
            self.speed = self.min_speed

    def rotate(self, angle: float):
        self.orientation += angle
        if self.orientation >= 360:
            self.orientation -= 360
        elif self.orientation < 0:
            self.orientation += 360

    def rotate_left(self):
        self.rotate(-self.turn_speed)

    def rotate_right(self):
        self.rotate(self.turn_speed)

    def nothing(self):
        pass

    def reproduce(self):
        child = Blob()  # Nuevo blob completamente nuevo
        child.x = self.x + random.uniform(-5, 5)  # Nacimiento cerca del padre
        child.y = self.y + random.uniform(-5, 5)

        # Heredar parámetros con pequeñas mutaciones
        child.max_speed = self.max_speed + random.uniform(-0.1, 0.1)
        child.min_speed = self.min_speed + random.uniform(-0.05, 0.05)
        child.turn_speed = self.turn_speed + random.uniform(-0.05, 0.05)
        # child.detection_radius = self.detection_radius + random.uniform(-2, 2)

        # Mutar el cerebro
        child.brain.copy_and_mutate(self.brain)
        # child.brain = copy.deepcopy(self.brain)

        child.generation = self.generation + 1  # Incrementar generación
        child.id = f"{self.id[:7]}_{child.generation}"

        return child

    def eat(self):
        self.life += EATING_LIFE_GAIN
        self.food_eaten += 1
        if self.life > self.max_life:
            self.life = self.max_life
        # print(f"Blob at ({self.x}, {self.y}) has eaten food. Life: {self.life}")
        if self.food_eaten > MIN_FOOD_TO_REPRODUCE:
            return self.reproduce()
        return None