# train_agent.py
import time
import numpy as np
import gymnasium as gym
from zombie_rl_env import ZombieEnv
from dqn_agent import DQNAgent
import csv
import os
import threading
import keyboard  # Для экстренного завершения в любом окне

# Параметры обучения
EPISODES = 100
BATCH_SIZE = 32
SAVE_EVERY = 50
TARGET_UPDATE_EVERY = 5
LOG_FILE = "training_log.csv"

# Флаг экстренного завершения
terminate_flag = False

def listen_for_esc():
    global terminate_flag
    keyboard.wait('x')
    terminate_flag = True

# Запускаем поток для отслеживания клавиши Esc
esc_listener = threading.Thread(target=listen_for_esc, daemon=True)
esc_listener.start()

# Создаём среду и агента
env = ZombieEnv()
agent = DQNAgent((5, 100, 400, 3), env.action_space)

# Создаём лог-файл, если он ещё не существует
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward", "Steps", "Epsilon", "Memory"])

try:
    for e in range(EPISODES):
        if terminate_flag:
            print("[INFO] Экстренное завершение обучения")
            break

        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        total_reward = 0
        done = False
        steps = 0

        while not done:
            if terminate_flag:
                print("[INFO] Экстренное завершение обучения") 
                raise KeyboardInterrupt
            
            action = agent.act(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, done, _, _ = step_result
            else:
                next_state, reward, done, _ = step_result

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.replay(BATCH_SIZE)

        if (e + 1) % TARGET_UPDATE_EVERY == 0:
            agent.update_target_model()
            print("[INFO] Обновлена целевая модель")

        print(f"[EPISODE {e+1}/{EPISODES}] Reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {agent.epsilon:.4f} | Memory: {len(agent.memory)}")

        # Сохраняем лог в файл
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([e + 1, total_reward, steps, round(agent.epsilon, 4), len(agent.memory)])

        if (e + 1) % SAVE_EVERY == 0:
            agent.save()

except KeyboardInterrupt:
    print("[INFO] Прерывание обучения пользователем. Сохраняем модель и выходим...")

finally:
    agent.save()
    print("[INFO] Модель сохранена. Завершение.")
