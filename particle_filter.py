import itertools

import numpy as np

import prediction
import track_floor_utils

RADIUS_SWEEP_DISTANCE = .1
NUM_RADIUS_SWEEPS = 4
SWEEP_SLICES = 6
ROT_Y_CHANGE_INCREMENT = 5
ROT_Y_CHANGE_BOUND = 20
PARTICLE_NUM_CAP = 50
# TODO(hutson): don't just guess at your random match value. Gather data.
RANDOM_MATCH = .3


# defined as a separate function rather than a member function of Particle
# because by doing this we amortize the cost of resizing
# the cam img in telemetry when we call get_img_equality_fraction
def get_particle_probabilities(tel_at_judgement, particles):
    location_candidates = []
    for particle in particles:
        location_candidates.append((particle.pos, particle.rot_y))
    equality_fractions = list(
        track_floor_utils.get_img_equality_fraction(
            tel_at_judgement, 'cheater_camera_image', 
            location_candidates))
    for particle, fraction in zip(particles, equality_fractions):
        if track_floor_utils.pos_in_track(particle.pos):
            probability = max(0, (fraction[1] - RANDOM_MATCH) / (1 - RANDOM_MATCH))
        else:
            probability = 0
        yield (particle, probability)

class Particle:
    def __init__(self, tel_at_generation, est_pos, est_rot_y):
        self.telemetry = tel_at_generation
        self.pos = est_pos
        self.rot_y = est_rot_y

    def generate_particles(self, tel_at_generation, delta_time):
        particles = []
        def add_particle(pos, rot_y):
            particles.append(Particle(tel_at_generation, pos, rot_y))
        base_prediction = prediction.telemetry_after_delta_time(
            self.telemetry, self.pos, self.rot_y, delta_time)
        
        radius_mult_generator = range(NUM_RADIUS_SWEEPS)
        rotation_generator = range(
                -ROT_Y_CHANGE_BOUND, ROT_Y_CHANGE_BOUND + 1, 
                ROT_Y_CHANGE_INCREMENT)
        sweep_slice_mult_generator = range(SWEEP_SLICES)
        product_generator = itertools.product(
            radius_mult_generator, rotation_generator, 
            sweep_slice_mult_generator)

        for r_multiplier, rot_y_additive, s_slice_multiplier in product_generator:
            (base_x, base_z), base_rot_y = base_prediction
            particle_r = RADIUS_SWEEP_DISTANCE * r_multiplier
            particle_radian = s_slice_multiplier * np.pi / (SWEEP_SLICES / 2)
            particle_rot_y = base_rot_y + rot_y_additive
            particle_pos = (particle_r * np.cos(particle_radian) + base_x, 
                            particle_r * np.sin(particle_radian) + base_z)
            add_particle(particle_pos, particle_rot_y)
        return particles

class ParticleFilter:
    def __init__(self, starting_tel, starting_pos, starting_rot_y):
        self.__init__(
            self, 
            [(Particle(starting_tel, starting_pos, starting_rot_y), 1)])

    def __init__(self, initial_particles, initial_probabilities):
        self.particles = initial_particles

    def get_mle(self):
        numerator_x = 0
        numerator_z = 0
        numerator_rot_y = 0
        denominator = 0
        for particle, probability in self.particles:
            x, z = particle.pos
            rot_y = particle.rot_y
            numerator_x += x * probability
            numerator_z += z * probability
            numerator_rot_y += rot_y * probability
            denominator += probability

        return (
            (numerator_x / denominator, numerator_z / denominator),
            numerator_rot_y / denominator)

    def update_filter(self, tel, delta_time):
        new_particles = []
        for particle, probability in self.particles:
            new_particles.extend(particle.generate_particles(
                tel, delta_time))
        new_particles = list(get_particle_probabilities(tel, new_particles))
        new_particles.sort(key=lambda p: p[1], reverse=True)
        new_particles = new_particles[:PARTICLE_NUM_CAP]
        self.particles = new_particles
