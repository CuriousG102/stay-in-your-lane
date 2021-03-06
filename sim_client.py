import asyncio
from threading import Event, Thread
import base64
from io import BytesIO
import multiprocessing as mp
import threading

from aiohttp import web
import socketio
from PIL import Image

from server import start_server

# this is hacky, but as long as the interface is clean we can 
# focus on making it not hacky if we need to later

class Telemetry:
    def __init__(self, data):
        self.steering = float(data['steering_angle'])
        self.throttle = float(data['throttle'])
        self.speed = float(data['speed'])
        self.front_camera_image = Image.open(BytesIO(base64.b64decode(data['front_image'])))
        self.overhead_camera_image = Image.open(BytesIO(base64.b64decode(data['overhead_image'])))
        self.cheater_camera_image = Image.open(BytesIO(base64.b64decode(data['cheater_image'])))
        self.delta_time = float(data['delta_time'])
        self.x, self.y, self.z = float(data['x']), float(data['y']), float(data['z'])
        self.rot_x = float(data['rot_x'])
        self.rot_y = float(data['rot_y'])
        self.rot_z = float(data['rot_z'])
        self.colliding = data['is_colliding'] == 'True'
        self.finished = data['is_finished'] == 'True'

    def __str__(self):
        return (
            'Steering: {0}\n'
            'Throttle: {1}\n'
            'Speed: {2}\n'
            'Delta Time: {3}\n'
            'Pos: {4}\n'
            'Rot: {5}\n'
            'Colliding: {6}\n'
            'Finished: {7}'
        ).format(
            self.steering, self.throttle, self.speed, self.delta_time,
            (self.x, self.y, self.z,), 
            (self.rot_x, self.rot_y, self.rot_z,),
            self.colliding, self.finished)

class OrderingException(Exception):
    pass

class SimObserverClient:
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

    def stop(self):
        self.server_process.terminate()

latest_tel = None

def continuously_pipe_telemetry(tel_in_queue, new_data_event, stop_event):
    global latest_tel
    while True:
        latest_tel = tel_in_queue.get()
        new_data_event.set()
        if (stop_event.is_set()):
            break

class SimObserverSamplingClient(SimObserverClient):
    '''
    Just like SimObserverClient, but get_telemetry call 
    gets latest telemetry from the car rather than the least
    recent unread telemetry. In other words, moves us from a streaming model
    to a model where you can sample as quickly as you can process data.
    '''
    def start(self):
        super().start()
        self.stop_event = threading.Event()
        self.new_data_event = threading.Event()
        self.telemetry_consumer_loop = threading.Thread(
            target=continuously_pipe_telemetry,
            args=(self.tel_queue, self.new_data_event, self.stop_event))
        self.telemetry_consumer_loop.start()

    def get_telemetry(self):
        '''
        Gets latest telemetry.
        '''
        self.new_data_event.wait()
        self.new_data_event.clear()
        return Telemetry(latest_tel)

    def stop(self):
        super().stop()
        self.stop_event.set()
        self.telemetry_consumer_loop.join(5)


class SimClient(SimObserverClient):
    def start(self):
        super().start()
        self.got_telemetry = False

    def get_telemetry(self):
        '''
        Gets Telemetry from car. Don't call twice in one step 
        or you'll be blocked forever.
        '''
        self.got_telemetry = True
        return super().get_telemetry()
        
    def send_instructions(self, steering, throttle):
        '''
        Call this or reset instruction, never both.
        Sends instructions to the car for it to run with during next timestep.
        '''
        if not self.got_telemetry:
            self.stop()
            raise OrderingException('Must always get telemetry '
                                    'before send_instructions '
                                    'Server has been shut down.')
        self.got_telemetry = False
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
        if not self.got_telemetry:
            self.stop()
            raise OrderingException('Must always get telemetry '
                                    'before reset_instructions. '
                                    'Server has been shut down.')
        self.got_telemetry = False
        self.instr_queue.put({'reset': 'yes'})

        # Sometimes Unity takes a few steps to reset, so we loop here to mask this
        # from the client
        for _ in range(20):
            self.get_telemetry()
            self.send_instructions(0, 0)
