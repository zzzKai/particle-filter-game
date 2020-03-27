import numpy as np
import cv2
import math

# need to do: pip install opencv-python-headless
# map = cv2.imread("./MarioMap.png");
# cv2.imshow('image',map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
map_path = "./BayMap.png"
map_init = cv2.imread(map_path)
width, length = map_init.shape[0] // 50 * 50, map_init.shape[1] // 50 * 50
map = cv2.resize(map_init, (length, width))
actual_pos = np.array([np.random.randint(1, width // 50) * 50 - 1, np.random.randint(1, length // 50) * 50 - 1, ])
num_particles = length * width // 1000
noise = 5
sub_img = 24


def update(pos, control):
    pos = pos + np.random.normal(0, noise, size=(2,)).astype(np.int)
    pos[pos < 0] = 0
    if pos[0] >= width:
        pos[0] = width - 1
    if pos[1] >= length - 1:
        pos[1] = length - 1
    if control == 2:
        if pos[1] >= 50:
            pos[1] -= 50
    elif control == 3:
        if pos[1] <= length - 50:
            pos[1] += 50
    elif control == 0:
        if pos[0] >= 50:
            pos[0] -= 50
    elif control == 1:
        if pos[0] <= width - 50:
            pos[0] += 50
    return pos


def update_particles(pos, control):
    for i in range(pos.shape[0]):
        pos[i] = update(pos[i], control)
    return pos


def redraw(true_pos, pos, weight):
    map_copy = map.copy()
    weight = (weight * 100).astype(np.int) + 1
    num_in_circle = 0
    num_all = 0
    for (x, y), w in zip(pos, weight):
        cv2.circle(map_copy, (y, x), radius=w, color=(0, 0, 255), thickness=w)
        num_all += 1
        if math.sqrt((y - true_pos[1]) ** 2 + (x - true_pos[0]) ** 2) < 30 * w:
            num_in_circle += 1
    cv2.circle(map_copy, (true_pos[1], true_pos[0]), radius=20 * w, color=(0, 255, 0), thickness=5 * w)
    metric = num_in_circle / num_all
    print(metric)
    return map_copy


def start_particle():
    particle_x = np.random.uniform(0, width, (num_particles, 1)).astype(np.int)
    particle_y = np.random.uniform(0, length, (num_particles, 1)).astype(np.int)
    position = np.concatenate([particle_x, particle_y], axis=1)
    return position, np.ones(position.shape[0]) / 100


def cal_weight(true_pos, pos):
    image_pieces = []
    sub_map = np.pad(map, ((sub_img, sub_img), (sub_img, sub_img), (0, 0)), mode='constant')
    for x, y in pos:
        left = x
        right = x + sub_img * 2
        up = y
        down = y + sub_img * 2
        image_pieces.append(sub_map[left: right, up: down, :][np.newaxis, ...])
    target_image = sub_map[true_pos[0]: true_pos[0] + sub_img * 2, true_pos[1]: true_pos[1] + sub_img * 2, :]
    image_pieces = np.concatenate(image_pieces, axis=0)
    weight = np.mean(np.power(image_pieces - target_image, 2).astype(np.float), axis=(1, 2, 3))
    weight = np.max(weight) - weight
    weight = weight.astype(np.float) / np.sum(weight)
    return weight


def sample_with_weight(pos, weight):
    distribution = np.zeros(weight.shape[0] + 1)
    distribution[1:] = np.cumsum(weight)
    particles = np.random.uniform(0, 1, num_particles)
    hist, bin_edges = np.histogram(particles, bins=distribution)
    position = []
    for i, count in enumerate(hist):
        for bin_edges in range(count):
            position.append(pos[i])
    position = np.stack(position, axis=0)
    return position, np.ones(position.shape[0]) / 100


pos_cur, weight_cur = start_particle()
true_pos = actual_pos
map_cur = redraw(true_pos, pos_cur, weight_cur)
while True:
    cv2.imshow("simulation", map_cur)
    weight_cur = cal_weight(true_pos, pos_cur)
    map_cur = redraw(true_pos, pos_cur, weight_cur)
    cv2.imshow("simulation", map_cur)
    cv2.waitKey(500)
    pos_cur, weight_cur = sample_with_weight(pos_cur, weight_cur)
    control = cv2.waitKey(0)
    true_pos = update(true_pos, control)
    pos_cur = update_particles(pos_cur, control)

