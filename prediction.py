import math
import numpy as np
import track_floor_utils

CAR_AXLE_FRONT = track_floor_utils.CAR_SCALE * 1.27
CAR_AXLE_BACK = track_floor_utils.CAR_SCALE * - 1.6

def telemetry_after_delta_time(telemetry, pos, rot_y, delta_time):
    '''
    Returns ((x, z), rot_y) predicted on track floor after delta time.

    Takes:
    t: Telemetry
    x: Known or guessed position of car
    z: Known or guessed position of car
    rot_y: Known or guessed rotation of car on y axis
    delta_time: Amount of time that passes while producing our estimate.
    '''
    t = telemetry
    x, z = pos
    rot_y = math.radians(rot_y)
    s_angle = math.radians(t.steering)
    print('s_angle: ' ,s_angle)
    speed = t.speed
    d = -CAR_AXLE_BACK+ CAR_AXLE_FRONT
    print('d: ',d)
    initial_front = np.array([
        x + math.sin(rot_y) * CAR_AXLE_FRONT, 
        z + math.cos(rot_y) * CAR_AXLE_FRONT])
    print('intial_front: ',initial_front)
    initial_back = np.array([
        x + math.sin(rot_y) * CAR_AXLE_BACK, 
        z + math.cos(rot_y) * CAR_AXLE_BACK])
    print('intial_back: ', initial_back)
    v_b_vector = (np.array([math.sin(rot_y), 
                            math.cos(rot_y)]) 
                  * speed)
    print('v_b_vector: ',v_b_vector)
    v_b = np.sqrt(np.sum([elem**2 for elem in v_b_vector]))
    print('v_b: ', v_b)
    final_back = initial_back+v_b_vector*delta_time
    print('final_back: ', final_back)
    print('v_b: ', v_b)
    print('s_angle: ', s_angle)
    print('delta_time: ',delta_time)
    print('math.tan(s_angle): ', math.tan(s_angle))


    final_front_y = np.cos(s_angle)**2 *(-d+delta_time*v_b+np.sqrt(d**2 +delta_time*v_b*(2*d-delta_time*v_b)*math.tan(s_angle)**2))
    print('change in front_y: ',final_front_y)
    final_front_x = final_front_y*math.tan(s_angle)
    print('change in front_x: ', final_front_x)
    # final_front_x_rotated = final_front_x
    # final_front_y_rotated = final_front_y
    final_front_y_rotated = (final_front_y*math.cos(rot_y)-final_front_x*math.sin(rot_y))
    final_front_x_rotated = final_front_x * math.cos(rot_y) + final_front_y * math.sin(rot_y)
    print('final_front_x_rotated: ', final_front_x_rotated)
    print('final_front_y_rotated: ', final_front_y_rotated)
    final_front = [final_front_x_rotated,final_front_y_rotated]
    print('final front: ',final_front+initial_front)
    final_front = final_front+initial_front

    print('delta front: ', final_front-final_back)
    #final_front = (2*v_b*delta_time/math.tan(s_angle)+np.sqrt())
    #print(c)
    print('initial_front-initial_back: ',initial_front-initial_back)
    print('final_front-final_back: ',final_front-final_back)
    #intial_direction = initial_front-initial_back
    final_direction = final_front-final_back
    change_in_angle = math.atan(final_direction[0]/final_direction[1])-rot_y

    fin_rot_y = rot_y+change_in_angle

    direction = (final_front-final_back)/np.sqrt(sum([x**2 for x in (final_front-final_back)]))

    ##Make sure this below is ok
    from_back = abs(CAR_AXLE_BACK)*direction
    ##This above

    car_pos = final_back+from_back
    # final_front = (
    #     np.array([math.cos(rot_y + s_angle),
    #               math.sin(rot_y + s_angle)])
    #     * np.sqrt((v_b_vector**2).sum())
    #     / math.cos(s_angle) + initial_front)
     #print('back:', initial_back, final_back)
     #print('front: ', initial_front, final_front)
    # fbx, fbz = final_back
    # ffx, ffz = final_front
    # final_rot_y = math.atan((ffz - fbz) / (ffx - fbx))
    print('d at beginning: ', np.sqrt(sum([x**2 for x in initial_front-initial_back])))
    print('d at end: ', np.sqrt(sum([x ** 2 for x in final_front - final_back])))

    return (car_pos,
            np.rad2deg(fin_rot_y))

def break_down_into_times(telemetry, pos, rot_y, delta_time,desired_break_down):
    '''
    Just in case the resolution isn't good enough you can do an interpolation.
    :param telemetry:
    :param pos:
    :param rot_y:
    :param delta_time:
    :param desired_break_down:
    :return:
    '''
    time = delta_time/desired_break_down
    for _ in range(desired_break_down):

        pos,rot_y = telemetry_after_delta_time(telemetry, pos, rot_y, time)
        #Produce new telemetry
    return pos,rot_y


if __name__ == '__main__':
    import sim_client as sim
    t = sim.Telemetry()
    print('CAR_AXLE_FRONT: ' + str(CAR_AXLE_FRONT))
    print('CAR_AXLE_BACK: ' + str(CAR_AXLE_BACK))
    print('d = '+ str(CAR_AXLE_FRONT-CAR_AXLE_BACK))
    s_angle = math.radians(t.steering)
    print('phi = '+ str(s_angle))
