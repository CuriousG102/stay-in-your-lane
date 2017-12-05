import Perspective as P
import sim_client

def do_it(sim):
    while True:
        tel = sim.get_telemetry()
        img=  P.convert_front_to_bird(tel)
        steering,throttle = get_params_from_img(img)
        sim.send_instructions(steering,throttle)
def get_params_from_img(img):
    return 1,1