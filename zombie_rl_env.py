# zombie_rl_env.py
import numpy as np
import pyautogui
import gym
from gym import spaces
import time
import keyboard
import cv2
import tensorflow as tf
import threading

# Центры линий
LANE_Y = (324, 539, 765, 1009, 1254)
# Центры клеток по X
CELL_X = (509, 699, 886, 1108, 1279, 1469, 1662, 1848, 2068)
# Кнопки выбора растений
PLANT_MENU_POS = ((613, 102), (730, 102))
# Цвет иконы солнца
PLANT_RECHARGE_Y = 23
PLANT_RECHARGE_X = (613, 730)
PLANT_UNREADY_COLOR = (58, 57, 55)

# Глобальные флаги
loss_triggered = False
win_triggered = False
pause_flag = False

# Слушатель Ctrl → поражение (будет активироваться каждый раз)
def listen_for_ctrl():
    global loss_triggered
    while True:
        keyboard.wait("ctrl")
        loss_triggered = True

# Слушатель Shift → победа (будет активироваться каждый раз)
def listen_for_shift():
    global win_triggered
    while True:
        keyboard.wait("shift")
        win_triggered = True

# Слушатель Space → пауза (как было)
def listen_for_space():
    global pause_flag
    while True:
        keyboard.wait("z")
        pause_flag = not pause_flag
        print(f"[INFO] {'PAUSED' if pause_flag else 'RESUMED'}")

# Запуск потоков
threading.Thread(target=listen_for_ctrl,  daemon=True).start()
threading.Thread(target=listen_for_shift, daemon=True).start()
threading.Thread(target=listen_for_space, daemon=True).start()


def listen_for_shift():
    global win_triggered
    keyboard.wait("shift")
    win_triggered = True
threading.Thread(target=listen_for_shift, daemon=True).start()


class ZombieEnv(gym.Env):
    def __init__(self):
        super(ZombieEnv, self).__init__()
        self.observation_space = spaces.MultiBinary(5)
        self.action_space = spaces.MultiDiscrete([3, 5, 9])
        self.model = tf.keras.models.load_model("zombie_detector.h5")
        self.sun = 50
        self.plant_cost = (100, 50)
        self.occupied = [[False]*9 for _ in range(5)]
        # для отсчёта времени
        self.last_time = time.time()

    def _get_line_image(self, y):
        shot = pyautogui.screenshot(region=(300, y-50, 1600, 100))
        img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (400,100)).astype(np.float32) / 255.0
        return img

    def _get_observation(self):
        imgs = [self._get_line_image(y) for y in LANE_Y]
        return np.stack(imgs, axis=0)

    def _predict_zombies(self, obs):
        return np.array([
            self.model.predict(img[np.newaxis], verbose=0)[0][0] > 0.5
            for img in obs
        ])

    def _check_loss(self):
        return loss_triggered

    def _check_win(self):
        return win_triggered

    def _choose_plant(self, idx):
        x,y = PLANT_MENU_POS[idx]
        pyautogui.click(x,y)
        time.sleep(0.1)

    def _plant_is_ready(self, idx):
        rgb = pyautogui.screenshot().getpixel((PLANT_RECHARGE_X[idx], PLANT_RECHARGE_Y))
        return rgb != PLANT_UNREADY_COLOR

    def _update_sun_count(self):
        shot = pyautogui.screenshot(region=(400,225,1750,1125))
        img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        mask = cv2.inRange(img, np.array([0,240,240]), np.array([20,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w*h < 500: continue
            cx, cy = x+w//2+400, y+h//2+225
            pyautogui.click(cx, cy)
            self.sun += 25

    def _update_occupied(self):
        shot = pyautogui.screenshot()
        for r in range(5):
            for c in range(9):
                if self.occupied[r][c]:
                    x0, y0 = CELL_X[c], LANE_Y[r]
                    found=False
                    for dx in range(-25,26,5):
                        for dy in range(-25,26,5):
                            try:
                                if sum(shot.getpixel((x0+dx, y0+dy)))<100:
                                    found=True; break
                            except: pass
                        if found: break
                    if not found:
                        self.occupied[r][c] = False

    def collect_suns(self):
        self._update_sun_count()

    def reset(self):
        global loss_triggered, win_triggered
        print("[INFO] Сброс среды...")
        self.sun = 50
        self.occupied = [[False]*9 for _ in range(5)]
        loss_triggered = False
        win_triggered = False
        # сброс таймера
        self.last_time = time.time()
        time.sleep(1)
        return self._get_observation()

    def step(self, action):
        # пауза
        while pause_flag:
            time.sleep(0.1)

        now = time.time()
        # добавляем 0.5 награды за каждую прошедшую секунду
        reward = 0.5 * (now - self.last_time)
        self.last_time = now

        # проверка победы “Shift”
        if self._check_win():
            print("[GAME] Победа!")
            win_triggered  = False
            return self._get_observation(), reward + 50, True, {}

        self.collect_suns()
        self._update_occupied()

        plant_idx, lane_idx, cell_idx = action
        obs = self._get_observation()
        zombies = self._predict_zombies(obs)

        if plant_idx == 2:
            pass
        else:
            if self.occupied[lane_idx][cell_idx]:
                reward -= 0.2
            elif self._plant_is_ready(plant_idx) and self.sun >= self.plant_cost[plant_idx]:
                self._choose_plant(plant_idx)
                pyautogui.click(CELL_X[cell_idx], LANE_Y[lane_idx])
                reward += 0.5 + (0.5 if zombies[lane_idx] else 0)
                self.sun -= self.plant_cost[plant_idx]
                self.occupied[lane_idx][cell_idx] = True
            else:
                reward -= 0.01

        # проверка поражения “Ctrl”
        if self._check_loss():
            print("[GAME] Поражение.")
            loss_triggered = False
            return self._get_observation(), reward - 50, True, {}

        return self._get_observation(), reward, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass
