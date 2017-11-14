import asyncio
from asyncio import Event
import base64
from io import BytesIO

from aiohttp import web
import socketio
from PIL import Image

# this is hacky, but as long as the interface is clean we can 
# focus on making it not hacky if we need to later

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

@sio.on('connect')
def connect(sid, environ):
    print('Simulation Connected')

@sio.on('disconnect')
def disconnect(sid):
    print('Simulation Disconnected')

car_data = None
car_data_available = Event()

@sio.on('telemetry')
def telemetry(sid, data):
    # print('Telemetry')
    # print('Steering: ', data['steering_angle'])
    # print('Throttle: ', data['throttle'])
    # print('   Speed: ', data['speed'])
    car_data = data
    car_data_available.set()

class Telemetry:
    def __init__(self, data):
        self.steering = data['steering_angle']
        self.throttle = data['throttle']
        self.speed = data['speed']

class SimClient:
    def start(self):
        self.loop = asyncio.get_event_loop()
        handler = app.make_handler()
        f = self.loop.create_server(handler, '127.0.0.1', 4567)
        self.srv = self.loop.run_until_complete(f)

    def get_telemetry(self):
        '''
        Gets Telemetry from car. Blocks on new telemetry after simulator
        timestep being available. If called more than once after the same
        timestep, raises an exception, so save the returned telemetry somewhere.
        '''
        _ = yield from car_data_available.wait()
        print(car_data)
        
    def send_instructions(self, steering, throttle):
        '''
        Sends instructions to the car for it to run with during next timestep.
        '''
        pass

    def stop(self):
        self.srv.close()
        self.loop.run_until_complete(self.srv.wait_closed())
        self.loop.run_until_complete(app.shutdown())
        self.loop.run_until_complete(handler.shutdown(60.0))
        self.loop.run_until_complete(app.cleanup())
