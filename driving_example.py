import glob
import os
import sys
import cv2
import numpy as np
import math
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from collections import deque
from threading import Thread
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random
import time
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
address = "localhost"
port = 2000
actor_list = []

SECONDS_PER_EPISODE = 15
IM_WIDTH = 224
IM_HEIGHT = 112
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 8
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"
SHOW_PREVIEW = False

MEMORY_FRACTION = 0.9

# RL params
MIN_REWARD = -200
DISCOUNT = 0.99

EPISODES = 100
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10
from keras.callbacks import TensorBoard
...
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
    
    def _write_logs(self, logs, step):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, step)
        self.writer.flush()


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STREER_AMT = 1.0
    front_camera=None
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self):
        self.client = carla.Client(address,port)
        self.client.set_timeout(2)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
    
    def process_img(self,image):
        i = np.array(image.raw_data) # 1D array
        i2 = i.reshape((self.im_height,self.im_width,4)) #RGBA image
        i3 = i2[:,:,:3] # RGB image
        if self.SHOW_CAM:
            cv2.imshow("",i)
            cv2.waitKey(1)
        self.front_camera = i3
    
    def collision_data(self, event):
        self.collision_hist.append(event)

    def reset(self):
        self.collision_list = []
        self.actor_list = []

        self.respawn = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3,self.respawn)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x",f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y",f"{self.im_height}")
        self.rgb_cam.set_attribute("fov",f"110")

        transform = carla.Transform(carla.Location(x=2.5,z=0.7)) # Place sensor at position (2.5,0.7) relative to the actor's center
        self.sensor = self.world.spawn_actor(self.rgb_cam,transform,attach_to=self.vehicle) # Attach sensor
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0)) # 

        time.sleep(4) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        col_sensor = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_sensor,transform,attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.1)
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))
        return self.front_camera
    
    def step(self,action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=-1*self.STREER_AMT)) # Left
        elif action == -1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=0)) # Straight
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer=1*self.STREER_AMT)) # Right
        
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        if len(self.collision_list) != 0:
            done = True # terminal condition
            reward = -200
        
        elif kmh < 50: # Avoid going around the circle -> low velocity
            done = False
            reward = -1
        
        else:
            done = False
            reward = 1
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        return self.front_camera, reward, done, None # obs, reward, done, extra_info

class DQNAgent:
    def __init__(self):
        self.target_model = self.create_model() # Actor - select action 
        self.model = self.create_model() # Critic - evaluate action 
        
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False

        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predicted_score  = Dense(3,activation="linear")(x) # predict score for each state-action pair
        model = Model(inputs = base_model.input, output = predicted_score) # Policy        
        model.compile(loss="mse",optimizer = Adam(lr=0.001), metrics= ["accuracy"])
        return model

    def update_replay_memory(self,transition): # Keep track of the memory for replaying 
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Start training when replay memory is sufficient
        minibatch = random.sample(self.replay_memory,MINIBATCH_SIZE)

        # For each reply -> start with initial state s
        current_states = np.array([transition[0] for transition in minibatch])/255 # Get RGB image
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE) # Q(s,a)

        # Look for the next state s'
        new_current_states = np.array([transition[3] for transition in minibatch])/255 # Get RGB image
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)  

        X = []
        y = []
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q # reward for this current action -> plus the approx future reward
            if done:
                new_q = reward

            current_qs = current_qs_list[index]

            # action is an index [-1,0,1] 
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs) # Try to match prediction of target network

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step=True
            self.last_logged_episode = self.tensorboard.step

        # Log if log_this_step is true
        with self.graph.as_default():
            self.model.fit(np.array(X/255,np.array(y),batch_size = TRAINING_BATCH_SIZE, verbose = 0, shuffle = False, callbacks=[self.tensorboard] if log_this_step else None))
        
        if log_this_step:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
    def get_qs(self,state):
        #state = np.array(state).reshape(-1,*state.shape)
        state = np.array(state)[np.newaxis,...]
        return self.model.predict(state/255)[0]
    
    # Train and predict in different thread
    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

if __name__ == '__main__':
    FPS=6
    ep_rewards = [-200]

    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION))
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    backend.set_session(sess)

    if not os.path.isdir("models"):
        os.makedirs("models")

    agent = DQNAgent()
    env = CarEnv()

    trainer_thread = Thread(target=agent.train_in_loop,daemon=True) # Create a daemon thread running in background


    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01) # Waiting until agent has been initalized
    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # ascii for Windows
    for episode in tqdm(range(1,EPISODES+1),ascii=True,unit="episodes"):
        # Initialize episode
        env.collision_hist = []
        agent.tensorboard.step = episode 
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        # Start training
        while True:
            # Predict
            if np.random.random() > epsilon:
                # Exploit
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0,3)
                # This takes no time, so we add a delay matching the FPS (prediction above takes longer)
                time.sleep(1/FPS)
            
            # Execute action
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Train model
            agent.update_replay_memory((new_state, action, reward, new_state, done))

            step += 1
            if done:
                break
        
        for actor in env.actor_list:
            actor.destroy()
        
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    
    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
