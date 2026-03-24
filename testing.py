import numpy as np
import cv2
import random
import time
import tensorflow as tf

# Load Models
dust_model = tf.keras.models.load_model('dust_detection_cnn.h5')
stair_model = tf.keras.models.load_model('stair_detection_cnn.h5')

# Parameters
MAP_SIZE = (10, 10)
CELL_SIZE = 30
SPEED = 80

# Preprocessing
def preprocess_patch(patch):
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = cv2.resize(patch, (64, 64))
    patch = patch / 255.0
    patch = np.expand_dims(patch, axis=0)
    return patch

def predict_dust_patch(patch):
    img = preprocess_patch(patch)
    pred = dust_model.predict(img, verbose=0)
    return "Dusty" if np.argmax(pred) == 0 else "Clean"

def predict_stair_patch(patch):
    img = preprocess_patch(patch)
    pred = stair_model.predict(img, verbose=0)
    classes = ["Obstacle", "Plain Floor", "Stair", "Unknown"]
    return classes[np.argmax(pred)]

# Create Floor
def create_virtual_floor_realistic():
    floor = np.zeros((MAP_SIZE[0], MAP_SIZE[1], 3), dtype=np.uint8)
    color = (255, 220, 200) if random.random() < 0.5 else (230, 230, 230)
    for i in range(MAP_SIZE[0]):
        for j in range(MAP_SIZE[1]):
            floor[i, j] = color
    floor[MAP_SIZE[0]-2:MAP_SIZE[0], MAP_SIZE[1]-2:MAP_SIZE[1]] = (0, 255, 255)  # Yellow = Stairs
    return floor

# Add Dust
def add_dust(floor_map):
    for i in range(MAP_SIZE[0]):
        for j in range(MAP_SIZE[1]):
            if random.random() < 0.15:
                floor_map[i, j] = (0, 0, 255)
    return floor_map

# Draw Map
def draw_whole_map(floor, vacuum_pos):
    img = np.zeros((MAP_SIZE[0]*CELL_SIZE, MAP_SIZE[1]*CELL_SIZE, 3), dtype=np.uint8)
    for i in range(MAP_SIZE[0]):
        for j in range(MAP_SIZE[1]):
            img[i*CELL_SIZE:(i+1)*CELL_SIZE, j*CELL_SIZE:(j+1)*CELL_SIZE] = floor[i, j]
    x, y = vacuum_pos
    cv2.circle(img, (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3, (0, 255, 0), -1)
    return img

# Smart Movement with Memory
def move_vacuum_smart(floor, pos, memory):
    x, y = pos
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    candidates = []

    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < MAP_SIZE[0] and 0 <= new_y < MAP_SIZE[1]:
            candidates.append((new_x, new_y))

    random.shuffle(candidates)
    for candidate in candidates:
        if candidate not in memory:
            patch = floor[candidate]
            if predict_dust_patch(patch) == "Dusty":
                return candidate

    for candidate in candidates:
        if candidate not in memory:
            return candidate

    return random.choice(candidates) if candidates else pos

# Simulate Stair Climb
def simulate_stair_climb():
    for step in range(10):
        stair_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(stair_img, f"Climbing Step {step+1}...", (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("AI Vacuum Simulation", stair_img)
        cv2.waitKey(300)
    cv2.destroyWindow("AI Vacuum Simulation")

# Main Simulation
def run_simulation():
    floor_number = 1
    while True:
        print(f"\n🏠 Starting Floor {floor_number}")
        floor_map = create_virtual_floor_realistic()
        floor_map = add_dust(floor_map)
        vacuum_pos = (0, 0)
        step = 0
        cleaned_tiles = 0
        visited_memory = set()

        cv2.namedWindow("AI Vacuum Simulation", cv2.WINDOW_NORMAL)

        while True:
            img = draw_whole_map(floor_map, vacuum_pos)
            cv2.imshow("AI Vacuum Simulation", img)
            key = cv2.waitKey(SPEED)
            if key == 27:
                print("🚪 Exiting Simulation")
                cv2.destroyAllWindows()
                return

            x, y = vacuum_pos
            patch = floor_map[x, y]
            dust_status = predict_dust_patch(patch)

            if (patch == np.array([0, 0, 255])).all():
                print(f"[Step {step}] 🧹 Cleaning dust at {vacuum_pos}")
                floor_map[x, y] = (255, 255, 255)
                cleaned_tiles += 1
            else:
                print(f"[Step {step}] - Dust Status: {dust_status} at {vacuum_pos}")

            visited_memory.add(vacuum_pos)

            # Check if all dust is cleaned
            dust_remaining = any((floor_map[i, j] == np.array([0, 0, 255])).all()
                                 for i in range(MAP_SIZE[0])
                                 for j in range(MAP_SIZE[1]))

            if not dust_remaining and x >= MAP_SIZE[0]-2 and y >= MAP_SIZE[1]-2:
                if (patch == np.array([0, 255, 255])).all():
                    stair_status = predict_stair_patch(patch)
                    if stair_status == "Stair":
                        print("🛗 Detected stairs! Climbing...")
                        simulate_stair_climb()
                        floor_number += 1
                        break

            vacuum_pos = move_vacuum_smart(floor_map, vacuum_pos, visited_memory)
            step += 1

# Run everything
run_simulation()
