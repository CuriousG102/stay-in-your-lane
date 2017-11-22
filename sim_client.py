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

    def __str__(self):
        return ('Steering: {0}\n'
                'Throttle: {1}\n'
                'Speed: {2}\n'
                'Delta Time: {3}\n').format(self.steering, self.throttle,
                                            self.speed, self.delta_time)


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
        Sends instructions to the car for it to run with during next timestep.
        '''
        self.instr_queue.put({
            'steering_angle': str(float(steering)),
            'throttle': str(float(throttle))
        })
    def show_Image(self):
        t = self.get_telemetry()
        t.front_camera_image
        return t

    def follow_Line(self):


    def stop(self):
        self.server_process.terminate()


