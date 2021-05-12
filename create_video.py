import argparse
from collections import namedtuple
import csv

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patches
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser("Fit Video")
    parser.add_argument("fit_path", help="Path to the fit CSV file")
    parser.add_argument("data_path", help="Path to the data CSV file")
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
    fig = plt.figure()
    ax = plt.subplot(211)
    ax.plot(steps, cost)
    ax = plt.subplot(212)
    ax.scatter(data[:, 0], data[:, 1])
    ax.add_patch(ellipses[0].to_patch())
    plt.show()


if __name__ == "__main__":
    _main()
