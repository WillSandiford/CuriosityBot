import pyautogui
import keyboard
import time
import numpy as np
import cv2
import threading
import torch

# Manages screen and state relation
class State:
    def __init__(self, size):
        self.size = size

    def capture_screen(self):
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        return screenshot
    
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.size, self.size))
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def state(self):
        screenshot = self.capture_screen()
        state = self.preprocess_image(screenshot)
        state = torch.tensor(state).unsqueeze(0)
        return state
    

# Manages screen control and action interaction
class Actions:
    def __init__(self, move_delay=0.02, mouse_delay=0.02, mouse_delta=500):
        self.move_delay = move_delay
        self.mouse_delay = mouse_delay
        self.mouse_delta = mouse_delta
        self.move_actions = np.array([self.move_forward, self.sprint, self.sprint_jump])
        self.look_actions = np.array([self.look_left, self.look_right, self.look_up, self.look_down])
        self.actions = np.array([self.move_forward, self.sprint, self.sprint_jump,
                        self.look_left, self.look_right, self.look_up, self.look_down])

    def move_forward(self):
        pyautogui.keyDown('w')
        time.sleep(self.move_delay)
        pyautogui.keyUp('w')

    def sprint(self):
        pyautogui.keyDown('w')
        pyautogui.keyDown('ctrl')
        time.sleep(self.move_delay)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('w')

    def sprint_jump(self):
        pyautogui.keyDown('w')
        pyautogui.keyDown('ctrl')
        pyautogui.keyDown('space')
        time.sleep(self.move_delay)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('w')
        pyautogui.keyUp('space')

    def look_left(self):
        pyautogui.moveRel(-self.mouse_delta, 0, duration=self.mouse_delay)

    def look_right(self):
        pyautogui.moveRel(self.mouse_delta, 0, duration=self.mouse_delay)

    def look_up(self):
        pyautogui.moveRel(0, -self.mouse_delta, duration=self.mouse_delay)

    def look_down(self):
        pyautogui.moveRel(0, self.mouse_delta, duration=self.mouse_delay)

    def reset(self):
        pyautogui.press('/')
        pyautogui.press('c')
        pyautogui.press('p')
        pyautogui.press('enter')

    def act(self, action_index, move=True):
        action = self.actions[action_index]
        action()

    def random_action_index(self, move=True):
        return np.random.randint(len(self.move_actions) * int(not move), len(self.actions))


# Bridges the signals from the REINFORCE algorithm to the minecraft interface
# Multithreading allows rewards to be given while the agent is training by pressing 'x' or 'c'
class MCEnvironment():
    def __init__(self, state_size, move_delay=0.02, mouse_delay=0.02, mouse_delta=50):
        self.state_interface = State(state_size)
        self.actions_interface = Actions(move_delay, mouse_delay, mouse_delta)

        self.state_size = state_size
        self.action_size = len(self.actions_interface.actions)             
    
    def check_reward(self):
        while self.is_running:
            if keyboard.is_pressed('x'):
                self.reward = 1
            elif keyboard.is_pressed('c'):
                self.reward = -1
            else:
                self.reward = 0
            time.sleep(0.1)

    def check_terminal(self):
        while self.is_running:
            self.terminal =  keyboard.is_pressed('z')
            time.sleep(0.05)

    def reset(self):
        self.actions_interface.reset()
        return self.state_interface.state()
    
    def random_action_index(self, move=True):
        return self.actions_interface.random_action_index(move)

    def step(self, action, move=True):
        self.actions_interface.act(action, move)
        state = self.state_interface.state()
        if (self.reward != 0):
            print('reward:', self.reward)
        if (self.terminal):
            print('terminal')
        return state, self.reward, self.terminal
    
    def start(self):
        self.is_running = True
        self.reward = 0
        self.terminal = False

        self.reward_thread = threading.Thread(target=self.check_reward)
        self.terminal_thread = threading.Thread(target=self.check_terminal)
        self.reward_thread.start()
        self.terminal_thread.start() 

    def close(self):
        self.is_running = False
        self.reward_thread.join()
        self.terminal_thread.join()
