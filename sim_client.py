import asyncio
from threading import Event, Thread
import base64
from io import BytesIO
import multiprocessing as mp

from aiohttp import web
import socketio
from PIL import Image

from server import start_server

# this is hacky, but as long as the interface is clean we can 
# focus on making it not hacky if we need to later

class Telemetry:
    def __init__(self, data):
        self.steering = data['steering_angle']
        self.throttle = data['throttle']
        self.speed = data['speed']
        self.front_camera_image = Image.open(BytesIO(base64.b64decode(data['front_image'])))
        self.overhead_camera_image = Image.open(BytesIO(base64.b64decode(data['overhead_image'])))
        self.delta_time = data['delta_time']
        self.x, self.y, self.z = data['x'], data['y'], data['z']
        self.rot_x = data['rot_x']
        self.rot_y = data['rot_y']
        self.rot_z = data['rot_z']

    def __str__(self):
        return (
            'Steering: {0}\n'
            'Throttle: {1}\n'
            'Speed: {2}\n'
            'Delta Time: {3}\n'
            'Pos: {4}\n'
            'Rot: {5}\n'
        ).format(
            self.steering, self.throttle, self.speed, self.delta_time,
            (self.x, self.y, self.z,), 
            (self.rot_x, self.rot_y, self.rot_z,))


class SimClient:
    def start(self):
        self.tel_queue = mp.Queue()
        self.instr_queue = mp.Queue()
        self.server_process = mp.Process(
            target=start_server,
            args=(self.tel_queue, self.instr_queue,))
        self.server_process.start()

    def get_telemetry(self):
        '''
        Gets Telemetry from car. Don't call twice in one step 
        or you'll be blocked forever.
        '''
        return Telemetry(self.tel_queue.get())
        
    def send_instructions(self, steering, throttle):
        '''
        Call this or reset instruction, never both.
        Sends instructions to the car for it to run with during next timestep.
        '''
        self.instr_queue.put({
            'steering_angle': str(float(steering)),
            'throttle': str(float(throttle)),
            'reset': 'no'
        })

    def reset_instruction(self):
        '''
        Call this or send instruction, never both.
        Resets car to original state for restarting training.
        '''
        self.instr_queue.put({'reset': 'yes'})

    def stop(self):
        self.server_process.terminate()
