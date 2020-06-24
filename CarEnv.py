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
import copy
from stable_baselines.common.evaluation import evaluate_policy
import carla
import gym
from gym import spaces
from carla import ColorConverter as cc
import random
import collections
import time
import numpy as np
import cv2
import pygame
import weakref
import re
from datetime import datetime
import settings

_HOST = settings._HOST
_PORT = settings._PORT
IM_WIDTH = settings.IM_WIDTH
IM_HEIGHT = settings.IM_HEIGHT
SECONDS_PER_EPISODE = settings.SECONDS_PER_EPISODE

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

global outdir
now = datetime.now()
now = now.strftime("%d%m%Y_%H%M%S")
outdir = '{}_out'.format(now)

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

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
        if self.recording and image.frame_number % 10 == 0:
            image.save_to_disk('{}/{}_{}'.format(outdir,self.sensors[self.index][3],image.frame_number,))

class CarEnv(gym.Env):
    front_camera = None

    def __init__(self, out_dir,spawn_point=None,save_video=False,cam_idx_list=(3,4),skip_secs=0.5,n_stacks=5,a_space_type="discrete"):
        super(CarEnv, self).__init__()
        # Define action and obs space
        self.skip_secs = skip_secs
        self.cam_idx_list = cam_idx_list
        self.out_dir = out_dir
        self.save_video = save_video
        self._weather_index = 0
        #self.action_space = spaces.Discrete(20)
        self.a_space_type = a_space_type
        if self.a_space_type == 'discrete':
            self.action_space = spaces.Discrete(28)
        else:
            self.action_space = spaces.Box(low=np.array([0,-1.0]),high=np.array([1.0,1.0]),dtype=np.float32)
        self.n_stacks = n_stacks
        self.observation_space = spaces.Box(low=0,high=1,shape=(IM_HEIGHT,IM_WIDTH,n_stacks*2),dtype=np.float32)
        self.actor_list = []
        self.client = carla.Client(_HOST,_PORT)
        self.client.set_timeout(2)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.bp = random.choice(self.blueprint_library.filter("walker.*"))
        self._weather_presets = find_weather_presets()
        self.SHOW_CAM = True
        self.spawn_point = spawn_point
        self.video = None
        self.cnt = 0
    
    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])
        
    
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
        self.close()
        self.collision_hist = []
        self.actor_list = []
        self.cnt += 1
        init_obs = np.zeros((IM_HEIGHT,IM_WIDTH))
        self.stacks = collections.deque(self.n_stacks * [init_obs],self.n_stacks)
        if self.video:
            self.video.release()
        if self.save_video:
            self.video = cv2.VideoWriter('{}/navigation_video_{}.avi'.format(self.out_dir,self.cnt),
            cv2.VideoWriter_fourcc('M','J','P','G'), 10,
            (IM_WIDTH,IM_HEIGHT)
        )
        
        # Spawn many points and select to avoid collision
        spawn_points = []
        for i in range(5000):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        random.shuffle(spawn_points)

        if self.spawn_point:
            spawn_points.insert(0,self.spawn_point)
    
        # if self.spawn_point is None:
        #     spawn_point = carla.Transform()
        #     loc = self.world.get_random_location_from_navigation()
        #     if (loc != None):
        #         spawn_point.location = loc
        #     self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        i = -1
        while i < len(spawn_points):
            try:
                i+=1 
                self.pedestrian = self.world.spawn_actor(self.bp, spawn_points[i])
            except:
                print('Collision, try next point')
            else:
                break
            
        self.actor_list.append(self.pedestrian)
        self.prev_speed = 0
        self.prev_dist = 0
        self.cam_list = []
        for cam_idx in self.cam_idx_list:
            cam = CameraManager(self.pedestrian)
            cam.set_sensor(cam_idx)
            if cam_idx  == 3:
                self.depth_cam = cam
            elif cam_idx == 4:
                self.seg_cam = cam
            self.cam_list.append(cam)
            self.actor_list.append(cam.sensor)
        start_time = time.time()
        crnt_time = start_time
        self._control = carla.WalkerControl()
        self._rotation = self.pedestrian.get_transform().rotation # current heading
        self._control.direction = self._rotation.get_forward_vector()
        self._control.speed = 3
        while crnt_time - start_time < 1.5:
            self.pedestrian.apply_control(self._control)
            crnt_time = time.time()
        self.start_location = self.pedestrian.get_location()

        time.sleep(4) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        col_sensor = self.blueprint_library.find('sensor.other.collision')
        transform = carla.Transform(carla.Location(x=1.2,z=1.3)) # Place sensor at position relative to the actor's center
        self.col_sensor = self.world.spawn_actor(col_sensor,transform,attach_to=self.pedestrian)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.detect_collision(event)) # Call collision_data() everytime collision is detected

        # Wait until the depth image has been collected
        while self.depth_cam.front_camera is None:
            time.sleep(0.1)
        # for sensor in self.sensors:
        #     sensor.recording = True
        # Let's rock n roll
        self.episode_start = time.time()
        depth_obs = self.depth_cam.front_camera[:,:,0] / 255.0
        seg_obs = self.seg_cam.front_camera[:,:,0]
        new_obs = np.stack((depth_obs,seg_obs)*self.n_stacks,axis=-1)
        #self.stacks.append(obs)
        #new_obs = np.stack(self.stacks,axis=-1)
        self.last_time = self.episode_start
        #print("===================Reset================")
        #print("Start  ",self.start_location)
        return new_obs
    
    def step(self,action):
        delta_time = time.time() - self.last_time
        #print('current ',distance)
        # 5 x 4 = 20 action space  
        if self.a_space_type == 'discrete':
            angle = action//4
            speed = action%4
            max_speed = 1
            min_speed = 0
            #angle = angle * 22.5 - 45 # [-90 , 90]
            angle = angle * 15 - 45
            speed = speed * (max_speed/3) + min_speed # [0 , 0.9]
        else:
            speed = action[0].item()
            angle = action[1].item() * 45

        #constant_speed = 0.8
        # action = [-90,90]
        self._control.speed = speed
        self._rotation.yaw += angle
        self._control.direction = self._rotation.get_forward_vector()

        start_time = time.time()
        crnt_time = start_time
        obs_buffer = []
        while crnt_time - start_time < self.skip_secs:
            self.pedestrian.apply_control(self._control)
            depth = self.depth_cam.front_camera[:,:,0]/255.0
            seg = self.seg_cam.front_camera[:,:,0]
            obs = np.stack((depth,seg),axis=-1)
            obs_buffer.append(obs)
            crnt_time = time.time()
        # crnt_obs = self.main_cam.front_camera[:,:,0]
        # obs = np.hstack((last_obs,crnt_obs))
        # cv2.imshow("",obs)
        # cv2.waitKey(1)
        next_location = self.pedestrian.get_location()
        #print(self.start_location, "  ",next_location)
        distance = self.start_location.distance(next_location)
        #print("Start ",self.start_location)
        delta_dist = distance - self.prev_dist
        #print('distance ',distance,' travel ', delta_dist, " speed ",speed, "time ",delta_time)
        #print('next ',distance)

        diff_speed = abs(self.prev_speed - speed)
        
        done = False
        reward = 0
        if len(self.collision_hist) != 0:
            done = True # terminal condition
            reward -= 200
        
        if not done:
            reward += speed
            reward -= diff_speed
            reward += 50*(delta_dist) # fps = 60

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        
        # Weights rewards (not for terminal state)
        # reward *= (time.time() - self.episode_start) / SECONDS_PER_EPISODE
        
        #obs = self.main_cam.front_camera[:,:,0] / 255.0
        self.prev_speed = speed
        self.prev_dist = distance
        #self.stacks.append(obs)
        #new_obs = obs[...,np.newaxis]
        #new_obs = np.stack(self.stacks,axis=-1)
        gap = len(obs_buffer) // self.n_stacks
        new_obs = np.dstack([ob for i,ob in enumerate(obs_buffer) if (i+1) % gap == 0])
        del obs_buffer
        if new_obs.shape[-1] > self.n_stacks*2:
            new_obs = new_obs[:,:,:-2]
        self.last_time = time.time()
        return new_obs,reward,done,{"distance":distance} # obs, reward, done, extra_info
    
    def render(self,mode='video'):
        if mode == 'video' and self.video:
            self.video.write(self.cam_list[0].front_camera)
        elif mode == 'display':
            cv2.imshow("",self.cam_list[0].front_camera)
            cv2.waitKey(1)
        #cv2.imwrite('{}/{}.png'.format(self.out_dir,self.cnt),self.main_cam.front_camera)
        
    def close(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('close')

if __name__ == '__main__':
    env = CarEnv('tmp',cam_idx_list=[0,3,4])
    obs = env.reset()
    cv2.imshow("",obs[:,:,-1])
    cv2.waitKey(1)
    cmd = input("Enter command")
    while cmd != 'q':
        if cmd == 'w': # angle = 0
            action = 15
        elif cmd == 'a': # angle = -45
            action = 3
        elif cmd == 'd': # angle = 45
            action = 31
        elif cmd == 's': # speed = 0
            action = 0
        obs,_,_,_ = env.step(action)
        cv2.imshow("",obs[:,:,-1])
        cv2.waitKey(1)
        #env.render(mode='display')
        cmd = input("Enter command")
        
    env.close()