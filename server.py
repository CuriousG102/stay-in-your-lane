import asyncio
import base64
from io import BytesIO
import multiprocessing as mp

from aiohttp import web
import socketio
from PIL import Image

# this is hacky, but as long as the interface is clean we can 
# focus on making it not hacky if we need to later

def start_server(tel_queue, instr_queue):
    sio = socketio.AsyncServer()
    app = web.Application()
    sio.attach(app)

    @sio.on('connect')
    def connect(sid, environ):
        print('Simulation Connected')

    @sio.on('disconnect')
    def disconnect(sid):
        print('Simulation Disconnected')

    @sio.on('telemetry')
    def telemetry(sid, data):
        # print('Telemetry')
        # print('Steering: ', data['steering_angle'])
        # print('Throttle: ', data['throttle'])
        # print('   Speed: ', data['speed'])
        # car_data = data
        # car_data_available.set()
        tel_queue.put(data)
        # pipe.send(data)

    @sio.on('instruction')
    def instruction(sid, data):
        # client_instructions = pipe.rcv()
        instructions = instr_queue.get()
        sio.emit('instructions', instructions)

    web.run_app(app, host='127.0.0.1', port=4567)
