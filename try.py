import sys
import numpy as np
import glob
import os
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except Exception:
    print("Cannot import carla")
    pass
import carla
import settings
client=carla.Client(settings._HOST,settings._PORT)
client.set_timeout(10)
world = client.get_world()
print(world)