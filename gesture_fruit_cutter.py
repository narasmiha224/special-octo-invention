import cv2
import mediapipe as mp
import numpy as np
import random
import time
import sys

# Sound playback
try:
    import pygame
    pygame.mixer.init()
    SOUND_ENABLED = True
except ImportError:
    print("pygame not installed, sound disabled.")
    SOUND_ENABLED = False

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FRUIT_RADIUS = 40

FRUIT_COLORS = [
    (255, 0, 0),    # Apple
    (0, 255, 0),    # Lime
    (0, 165, 255),  # Orange
    (255, 0, 255),  # Berry
    (0, 255, 255)   # Exotic
]
FRUIT_TYPES = ["Apple", "Lime", "Orange", "Berry", "Exotic"]

FINGER_TIP_ID = 8
SLICE_DISTANCE_THRESHOLD = 50
MAX_FRUITS = 7
FRUIT_SPAWN_INTERVAL = 1.0
GRAVITY = 0.2

class Fruit:
    def __init__(self, fruit_type, start_pos, velocity):
        self.fruit_type = fruit_type
        self.color = FRUIT_COLORS[FRUIT_TYPES.index(fruit_type)]
        self.position = np.array(start_pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = FRUIT_RADIUS
        self.sliced = False
        self.slice_time = None
        self.left_pos = None
        self.right_pos = None
        self.left_vel = None
        self.right_vel = None

    def update(self):
        if not self.sliced:
            self.velocity[1] += GRAVITY
            self.position += self.velocity
        else:
            self.left_pos += self.left_vel
            self.right_pos += self.right_vel
            self.left_vel[1] += GRAVITY
            self.right_vel[1] += GRAVITY

    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (int(self.position[0]), int(self.position[1])), self.radius, self.color, -1)
            stem_start = (int(self.position[0]), int(self.position[1] - self.radius))
            stem_end = (int(self.position[0]), int(self.position[1] - self.radius - 15))
            cv2.line(frame, stem_start, stem_end, (34, 139, 34), 3)
        else:
            if self.left_pos[1] < WINDOW_HEIGHT + self.radius:
                cv2.ellipse(frame,
                            (int(self.left_pos[0]), int(self.left_pos[1])),
                            (self.radius, self.radius),
                            0,
                            90, 270,
                            self.color,
                            -1)
            if self.right_pos[1] < WINDOW_HEIGHT + self.radius:
                cv2.ellipse(frame,
                            (int(self.right_pos[0]), int(self.right_pos[1])),
                            (self.radius, self.radius),
                            0,
                            270, 90,
                            self.color,
                            -1)

    def slice(self):
        if not self.sliced:
            self.sliced = True
            self.slice_time = time.time()
            self.left_pos = self.position.copy()
            self.right_pos = self.position.copy()
            self.left_vel = np.array([-5, 2], dtype=float)
            self.right_vel = np.array([5, 1], dtype=float)

    def is_off_screen(self):
        if not self.sliced:
            return self.position[1] - self.radius > WINDOW_HEIGHT
        else:
            return (self.left_pos[1] - self.radius > WINDOW_HEIGHT) and (self.right_pos[1] - self.radius > WINDOW_HEIGHT)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    fruits = []
    last_spawn_time = 0
    score = 0
    combo_counter = 0
    last_slice_time = 0
    combo_timeout = 2

    if SOUND_ENABLED:
        try:
            slice_sound = pygame.mixer.Sound('slice.wav')
        except pygame.error:
            print("slice.wav not found. Sound disabled.")
            slice_sound = None
    else:
        slice_sound = None

    font = cv2.FONT_HERSHEY_SIMPLEX
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        fingertip_positions = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, c = frame.shape
                landmark = hand_landmarks.landmark[FINGER_TIP_ID]
                x, y = int(landmark.x * w), int(landmark.y * h)
                fingertip_positions.append(np.array([x, y]))
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (x, y), 12, (0, 255, 255), 3)

        current_time = time.time()
        if current_time - last_spawn_time > FRUIT_SPAWN_INTERVAL and len(fruits) < MAX_FRUITS:
            start_x = random.randint(FRUIT_RADIUS, WINDOW_WIDTH - FRUIT_RADIUS)
            start_y = -FRUIT_RADIUS - 10
            vel_x = random.uniform(-4, 4)
            vel_y = random.uniform(5, 10)
            fruit_type = random.choice(FRUIT_TYPES)
            fruit = Fruit(fruit_type, (start_x, start_y), (vel_x, vel_y))
            fruits.append(fruit)
            last_spawn_time = current_time

        for fruit in fruits[:]:
            fruit.update()
            fruit.draw(frame)
            if fruit.is_off_screen():
                fruits.remove(fruit)

        for fingertip_pos in fingertip_positions:
            for fruit in fruits:
                if not fruit.sliced:
                    dist = np.linalg.norm(fruit.position - fingertip_pos)
                    if dist < SLICE_DISTANCE_THRESHOLD:
                        fruit.slice()

                        base_score = 1
                        if fruit.fruit_type == "Berry":
                            base_score = 3
                        elif fruit.fruit_type == "Exotic":
                            base_score = 5
                        if random.random() < 0.1:
                            base_score += 2

                        score += base_score

                        now = time.time()
                        if now - last_slice_time <= combo_timeout:
                            combo_counter += 1
                        else:
                            combo_counter = 1
                        last_slice_time = now

                        if combo_counter == 3:
                            score += 5
                            cv2.putText(frame, "3 Combo! +5", (600, 100), font, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                        elif combo_counter == 5:
                            score += 15
                            cv2.putText(frame, "5 Combo! +15", (600, 150), font, 1.5, (0, 200, 255), 4, cv2.LINE_AA)
                        elif combo_counter == 10:
                            score += 30
                            cv2.putText(frame, "10 Combo! +30", (600, 200), font, 2, (0, 128, 255), 6, cv2.LINE_AA)

                        cv2.putText(frame, f"+{base_score}", (int(fruit.position[0]), int(fruit.position[1])),
                                    font, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

                        if slice_sound:
                            slice_sound.play()

        cv2.putText(frame, f"Score: {score}", (30, 50), font, 1.5, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, f"Score: {score}", (30, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "Slice fruits using index fingers from both hands!", 
                    (30, WINDOW_HEIGHT - 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time + 1e-5)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (WINDOW_WIDTH - 140, 50), font, 1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (WINDOW_WIDTH - 140, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Gesture-Controlled Fruit Cutter Game", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    if SOUND_ENABLED:
        pygame.mixer.quit()

if __name__ == "__main__":
    print("Starting Gesture-Controlled Fruit Cutter Game")
    print("Press ESC or 'q' to quit.")
    main()
