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
        self.front_camera_image = data['front_image']
        self.overhead_camera_image = data['overhead_image']

    def __str__(self):
        return ('Steering: {0}\n'
                'Throttle: {1}\n'
                'Speed: {2}\n').format(self.steering, self.throttle, self.speed)


class SimClient:
    def start(self):
        # self.loop = asyncio.get_event_loop()
        # handler = app.make_handler()
        # f = self.loop.create_server(handler, '127.0.0.1', 4567)
        # self.srv = asyncio.ensure_future(f)
        # asyncio.get_event_loop().run_until_complete(self.srv)
        self.tel_queue = mp.Queue()
        self.instr_queue = mp.Queue()
        self.server_process = mp.Process(
            target=start_server,
            args=(self.tel_queue, self.instr_queue,))
        self.server_process.start()

    def get_telemetry(self):
        '''
        Gets Telemetry from car. Blocks on new telemetry after simulator
        timestep being available. If called more than once after the same
        timestep, raises an exception, so save the returned telemetry somewhere.
        '''
        return Telemetry(self.tel_queue.get())
        
    def send_instructions(self, steering, throttle):
        '''
        Sends instructions to the car for it to run with during next timestep.
        '''
        pass

    def stop(self):
        self.server_process.terminate()
