import glob
import os

import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import gym
from gym import spaces
from carla import ColorConverter as cc
import random
import time
import numpy as np
import cv2
import pygame
import weakref
from datetime import datetime

address = "localhost"
port = 2000
IM_WIDTH = 480
IM_HEIGHT = 480
FPS = 30
SECONDS_PER_EPISODE = 60

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

global outdir

class CameraManager(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.recording = False
        self.front_camera = None
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.2, z=1.3))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB','rgb'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)','depth'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)','depth_gray'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)','depth_log'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)','seg'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)','seg'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)','lidar']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(IM_WIDTH))
                bp.set_attribute('image_size_y', str(IM_HEIGHT))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def toggle_recording(self):
        self.recording = not self.recording

    def set_sensor(self, index, show_cam=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image, show_cam))
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)


    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, show_cam):
        self = weak_self()
        if not self:
            return
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if show_cam:
            cv2.imshow("",array)
            cv2.waitKey(1)
        self.front_camera = array
        if self.recording and image.frame_number % 100 == 0:
            global outdir
            image.save_to_disk(f'{outdir}/{self.sensors[self.index][3]}_{image.frame_number}')

class CarEnv(gym.Env):
    front_camera = None

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and obs space
        self.action_space = spaces.Box(low=-90, high=90,shape=(1),dtype=np.float32)
        self.observation_space = spaces.Box(low=0,high=1,shape=(IM_HEIGHT,IM_WIDTH,3),dtype=np.float32)


        self.client = carla.Client(address,port)
        self.client.set_timeout(2)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = random.choice(self.blueprint_library.filter("walker.*"))
        self.SHOW_CAM = True
        
    
    def process_img(self,image):
        i = np.array(image.raw_data) # 1D array
        i = i.reshape((IM_HEIGHT,IM_WIDTH,4)) #RGBA image
        i = i[:,:,:3] # RGB image
        if self.SHOW_CAM:
            cv2.imshow("",i)
            cv2.waitKey(1)
        self.front_camera = i
    
    def detect_collision(self,event):
        self.collision_hist.append(event)

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.sensors = []
        spawn_point = carla.Transform()
        loc = self.world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
        #spawn_point = random.choice(self.world.get_map().get_spawn_points())

        self.pedestrian = self.world.spawn_actor(self.bp,spawn_point)
        self.actor_list.append(self.pedestrian)
        self._rotation = self.pedestrian.get_transform().rotation # current heading

        # Set RGB
        self.rgb_cam = CameraManager(self.pedestrian)
        self.rgb_cam.set_sensor(0)
        self.sensors.append(self.rgb_cam)
        self.actor_list.append(self.rgb_cam.sensor)

        # Set semantic
        self.semantic_cam = CameraManager(self.pedestrian)
        self.semantic_cam.set_sensor(5,self.SHOW_CAM)
        self.sensors.append(self.semantic_cam)
        self.actor_list.append(self.semantic_cam.sensor)

        # Set depth
        self.depth_cam = CameraManager(self.pedestrian)
        self.depth_cam.set_sensor(3)
        self.sensors.append(self.depth_cam)
        self.actor_list.append(self.depth_cam.sensor)

        #Set up main camera
        self.main_cam = self.rgb_cam

        # Set RGB camera
        # self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        # self.rgb_cam.set_attribute("image_size_x",f"{IM_WIDTH}")
        # self.rgb_cam.set_attribute("image_size_y",f"{IM_HEIGHT}")
        # self.rgb_cam.set_attribute("fov",f"100")

        # transform = carla.Transform(carla.Location(x=0.2,z=0.8)) # Place sensor at position relative to the actor's center
        # self.sensor = self.world.spawn_actor(self.rgb_cam,transform,attach_to=self.pedestrian) # Attach sensor
        # self.sensor.listen(lambda data: self.process_img(data)) # Callback function is triggered everytime the data is retrieved

        #self.actor_list.append(self.sensor)

        self.pedestrian.apply_control(carla.WalkerControl())

        time.sleep(4) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        col_sensor = self.blueprint_library.find('sensor.other.collision')
        transform = carla.Transform(carla.Location(x=1.2,z=1.3)) # Place sensor at position relative to the actor's center
        self.col_sensor = self.world.spawn_actor(col_sensor,transform,attach_to=self.pedestrian)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.detect_collision(event)) # Call collision_data() everytime collision is detected

        # Wait until the RGB image has been collected
        while self.main_cam.front_camera is None:
            time.sleep(0.1)
        for sensor in self.sensors:
            sensor.recording = True
        # Let's rock n roll
        self.episode_start = time.time()
        return self.main_cam.front_camera
    
    def step(self,action):
    
        constant_speed = 0.8
        # action = [-90,90]
        self._rotation.yaw += action
        next_orientation = self._rotation.get_forward_vector()
        control = carla.WalkerControl(direction=next_orientation,speed=constant_speed)
        self.pedestrian.apply_control(control)

        done = False
        if len(self.collision_hist) != 0:
            done = True # terminal condition
            reward = -200
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        reward = -1
        return self.main_cam.front_camera,reward,done,None # obs, reward, done, extra_info
    
    def close(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('done.')

if __name__ == '__main__':
    pes_env = PesEnv()
    outer_clock = time.time()
    now = datetime.now()
    now = now.strftime("%d%m%Y_%H%M%S")
    global outdir
    outdir = f'{now}_out'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    pes_env.reset()
    print("Start")
    # Run for 2 minutes
    while time.time() < outer_clock + 90:
        repeat_action_seconds = 3
        action = random.choice(range(91)) # Try some action
        done = False
        start_time = time.time()
        # Repeat that action for that seconds
        while time.time() <= start_time + repeat_action_seconds and not done:
            img,_,done,_ = pes_env.step(action)
            action = 0 # Rotate then keep that direction
            if done:
                break
    pes_env.close()


