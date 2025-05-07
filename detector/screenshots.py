# save_screenshot.py
import pyautogui
import cv2
import numpy as np
import time

LANE_Y = [250, 400, 550, 700, 850]
j = 0

def capture_line_images():
    for i, y in enumerate(LANE_Y):
        img = pyautogui.screenshot(region=(300, y - 50, 1600, 100))  # координаты прямоугольника
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"line_{j}-{i}.png", img)

if __name__ == "__main__":
    print("Наведи на игру, нажми Enter...")
    input()
    while True:
        time.sleep(10)
        j += 1
        capture_line_images()
        print("Скриншоты сохранены — раскидай их по папкам (zombie / no_zombie)")
