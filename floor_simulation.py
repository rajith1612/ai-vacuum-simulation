import numpy as np
import cv2
import time

# Map settings
MAP_SIZE = (10, 10)
CELL_SIZE = 50  # Slightly larger for better visualization on small map

def simulate_stair_climb(vacuum_pos):
    x, y = vacuum_pos
    for step in range(5):
        stair_frame = np.zeros((MAP_SIZE[0]*CELL_SIZE, MAP_SIZE[1]*CELL_SIZE, 3), dtype=np.uint8)
        center_x = y * CELL_SIZE + CELL_SIZE // 2
        center_y = x * CELL_SIZE + CELL_SIZE // 2 - step * 10
        cv2.circle(stair_frame, (center_x, center_y), CELL_SIZE // 3, (0, 255, 0), -1)
        cv2.putText(stair_frame, f"Climbing Step {step + 1}...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Stair Climb", stair_frame)
        cv2.waitKey(300)
    cv2.destroyWindow("Stair Climb")

# Initialize floor map
def create_floor_map():
    floor = np.zeros(MAP_SIZE, dtype=np.uint8)
    dust_positions = np.random.choice([0, 1], size=MAP_SIZE, p=[0.7, 0.3]).astype(np.uint8)
    floor += dust_positions

    # Common obstacle layout for all floors
    floor[2:3, 2:5] = 2  # obstacle 1
    floor[6:8, 4:6] = 2  # obstacle 2

    # Common stairs at bottom-right corner
    floor[8:10, 8:10] = 9

    return floor

def draw_map(map_data, vacuum_pos):
    display = np.zeros((MAP_SIZE[0]*CELL_SIZE, MAP_SIZE[1]*CELL_SIZE, 3), dtype=np.uint8)
    for i in range(MAP_SIZE[0]):
        for j in range(MAP_SIZE[1]):
            val = map_data[i, j]
            color = (200, 200, 200)  # default empty
            if val == 1:
                color = (0, 0, 255)    # dust
            elif val == 2:
                color = (100, 100, 100)  # obstacle
            elif val == 3:
                color = (255, 255, 255)  # cleaned
            elif val == 9:
                color = (0, 255, 255)    # stairs
            cv2.rectangle(display, (j*CELL_SIZE, i*CELL_SIZE),
                          ((j+1)*CELL_SIZE, (i+1)*CELL_SIZE), color, -1)
    x, y = vacuum_pos
    cv2.circle(display, (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2),
               CELL_SIZE//3, (0, 255, 0), -1)
    return display

def move_vacuum(map_data, pos, allow_stairs=False):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    np.random.shuffle(directions)
    for dx, dy in directions:
        new_x, new_y = pos[0] + dx, pos[1] + dy
        if 0 <= new_x < MAP_SIZE[0] and 0 <= new_y < MAP_SIZE[1]:
            target = map_data[new_x, new_y]
            if target == 2:
                continue  # Obstacle
            if not allow_stairs and target == 9:
                continue  # Don't allow stair entry during cleaning
            return (new_x, new_y)
    return pos

def clean_cell(map_data, pos):
    x, y = pos
    if map_data[x, y] == 1:
        map_data[x, y] = 3

def run_simulation():
    map_data = create_floor_map()
    vacuum_pos = (0, 0)

    cv2.namedWindow("Floor 1", cv2.WINDOW_NORMAL)
    step = 0

    # --- Cleaning Phase ---
    while np.any(map_data == 1):
        clean_cell(map_data, vacuum_pos)
        img = draw_map(map_data, vacuum_pos)
        cv2.imshow("Floor 1", img)
        key = cv2.waitKey(100)
        if key == 27:
            break
        vacuum_pos = move_vacuum(map_data, vacuum_pos, allow_stairs=False)
        step += 1

    print("Cleaning complete. Searching for stairs...")

    # --- Post-cleaning: Move to stairs ---
    stair_positions = list(zip(*np.where(map_data == 9)))
    if not stair_positions:
        print("No stairs found!")
        return

    while vacuum_pos not in stair_positions:
        img = draw_map(map_data, vacuum_pos)
        cv2.imshow("Floor 1", img)
        key = cv2.waitKey(100)
        if key == 27:
            return
        vacuum_pos = move_vacuum(map_data, vacuum_pos, allow_stairs=True)

    # --- Stair Climbing ---
    print("Stairs detected! Climbing to Floor 2...")
    simulate_stair_climb(vacuum_pos)

    for step in range(5):
        print(f"Climbing step {step + 1}...")
        time.sleep(0.5)

    print("Reached Floor 2!")
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    # Load Floor 2 (same layout, new dust)
    map_data = create_floor_map()
    vacuum_pos = (0, 0)
    step = 0

    cv2.namedWindow("Floor 2", cv2.WINDOW_NORMAL)
    print("Starting cleaning on Floor 2...")

    while np.any(map_data == 1):
        clean_cell(map_data, vacuum_pos)
        img = draw_map(map_data, vacuum_pos)
        cv2.imshow("Floor 2", img)
        key = cv2.waitKey(100)
        if key == 27:
            break
        vacuum_pos = move_vacuum(map_data, vacuum_pos, allow_stairs=False)
        step += 1

    print(f"Floor 2 cleaning complete in {step} steps.")
    cv2.waitKey(1000)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    run_simulation()