""" Script to create a video showing the system as it fits to data """

import argparse
from collections import namedtuple
import csv

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser("Fit Video")
    parser.add_argument("fit_path", help="Path to the fit CSV file")
    parser.add_argument("data_path", help="Path to the data CSV file (created from -dump_data)")
    parser.add_argument("--mp4_path", help="Path to the output MP4 file")
    parser.add_argument("--landscape", action="store_true",
                        help="Whether to create a landscape video")
    return parser.parse_args()


Point = namedtuple("Point", ["x", "y"])
class Ellipse(namedtuple("Ellipse", ["h", "k", "a", "b"])):
    def to_patch(self):
        xy = np.array([self.h, self.k])
        return patches.Ellipse(xy, 2*self.a, 2*self.b, fill=False)


def _main():
    args = _parse_args()

    data = []
    with open(args.data_path) as file:
        reader = csv.DictReader(file)
        for line in reader:
            data.append(Point(float(line["x"]), float(line["y"])))

    data = np.array(data)

    cost = []
    ellipses = []
    with open(args.fit_path) as file:
        reader = csv.DictReader(file)
        for line in reader:
            cost.append(float(line["Cost"]))
            ellipses.append(Ellipse(float(line["h"]),
                                    float(line["k"]),
                                    float(line["a"]),
                                    float(line["b"])))

    steps = np.arange(len(cost))
    if args.landscape:
        fig = plt.figure(figsize=(8, 4))
        cost_ax = plt.subplot(121)
        fit_ax = plt.subplot(122)
    else:
        fig = plt.figure(figsize=(4, 8))
        cost_ax = plt.subplot(211)
        fit_ax = plt.subplot(212)

    cost_ax.set_xlabel("Cost")
    cost_ax.set_ylabel("Step")
    line, = cost_ax.plot(steps, cost, 'r-')
    fit_ax.set_xlim(-3, 5)
    fit_ax.set_ylim(-3, 5)
    fit_ax.scatter(data[::4, 0], data[::4, 1], marker='.')
    ellipse = ellipses[0].to_patch()
    fit_ax.add_patch(ellipse)

    def init():
        line.set_data([], [])
        return line, ellipse

    def animate(i):
        line.set_data(steps[:i + 1], cost[:i+1])
        ellipse.set_center(np.array([ellipses[i].h, ellipses[i].k]))
        ellipse.set_width(2 * ellipses[i].a)
        ellipse.set_height(2 * ellipses[i].b)
        return line, ellipse

    anim = FuncAnimation(fig, animate, init_func=init,
                              frames=len(cost), interval=100, blit=True)

    if args.mp4_path:
        anim.save(args.mp4_path, fps=12, extra_args=['-vcodec', 'libx264'])

    plt.show()


if __name__ == "__main__":
    _main()
