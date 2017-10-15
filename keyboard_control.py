"""
keyboard_control.py:
    Control a Nao robot that is currently running the Naoqi software with your
    keyboard! Key bindings are defined in controllers.txt.

    Requires naomanager.py & an installation of pynaoqi
"""

import argparse
import sys
import time
import math
import getch
import json
from threading import Thread

from naomanager import NaoManager, DEFAULT_PORT
# from naoqi import ALProxy

__author__ = "Michiel van der Meer, Caitlin Lagrand"
__copyright__ = "Copyright 2017, Dutch Nao Team"
__version__ = "0.2.0"

getch = getch._Getch()


def load_motion(name):
    print("Performing motion {}".format(name))
    with open("animations/{}.json".format(name), "r") as f:
        fdict = json.load(f)
        fdict["names"] = [str(x) for x in fdict["names"]]
    return fdict


# Main program to perform actions
def main(args):
    # Create list of nao objects
    naos = NaoManager()
    for nao in args.nao:
        try:
            ip, port = nao.split(':')
            # naos.addnao(ip, int(port))
        except ValueError:
            ip = nao
            port = DEFAULT_PORT
            naos.addnao(ip, int(port))
    if len(naos) == 1:
        print("Connected {} Nao".format(len(naos)))
    else:
        print("Connected {} Naos".format(len(naos)))
    x = 0.0
    y = 0.0
    theta = 0.0
    frequency = 0.3
    CommandFreq = 0.5
    starttime = time.time()
    print("Ready for liftoff at: {}".format(starttime))

    while True:
        key_press = getch()
        key_press = key_press.decode('ascii')
        if (key_press == 'z'):
            print("Closing Connection")
            for nao in naos:
                nao.motion.rest()
                time.sleep(CommandFreq)
                nao.motion.killAll()
                nao.stop()
                exit()
        elif (key_press == 'w'):
            print("Moving forward")
            for nao in naos:
                x = 0.5
                nao.motion.setWalkTargetVelocity(x, 0, 0, frequency)
                time.sleep(CommandFreq)
        elif (key_press == 's'):
            print("Moving backward")
            for nao in naos:
                x = -0.5
                nao.motion.setWalkTargetVelocity(x, 0, 0, frequency)
                time.sleep(CommandFreq)
        elif (key_press == 'x'):
            print("Stopping")
            for nao in naos:
                nao.motion.setWalkTargetVelocity(0, 0, 0, frequency)
                time.sleep(CommandFreq)
        elif (key_press == 'a'):
            print("Turning left")
            for nao in naos:
                theta = 0.5
                nao.motion.setWalkTargetVelocity(x, y, theta, frequency)
                time.sleep(CommandFreq)
        elif (key_press == 'd'):
            print("Turning right")
            for nao in naos:
                theta = -0.5
                nao.motion.setWalkTargetVelocity(x, y, theta, frequency)
                time.sleep(CommandFreq)
        elif (key_press == 'q'):
            print("Going to rest")
            naos.motion.post.rest()
            print("Resting (zzz)")
        elif (key_press == 'e'):
            print("Standup")
            naos.posture.post.goToPosture("StandInit", 0.5)
            print("Done standing up")
        elif (key_press == 'y'):
            fdict = load_motion("robot")
            try:
                naos.motion.post.angleInterpolationBezier(fdict["names"], fdict["times"], fdict["keys"])
            except BaseException, err:
                print err

        elif (key_press == 'u'):
            print("Kick")
            names = ['LShoulderRoll', 'LShoulderPitch', 'RShoulderRoll', 'RShoulderPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch',
                     'LAnkleRoll', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
            angles = [[0.3], [0.4], [-0.5], [1.0], [0.0], [-0.4, -0.2], [0.95,
                                                                         1.5], [-0.55, -1], [-0.2], [0.0], [-0.4], [0.95], [-0.55], [-0.2]]
            times = [[0.5], [0.5], [0.5], [0.5], [0.5], [0.4, 0.8], [0.4, 0.8],
                     [0.4, 0.8], [0.4], [0.5], [0.4], [0.4], [0.4], [0.4]]
            naos.motion.post.angleInterpolation(names, angles, times, True)
            naos.motion.post.angleInterpolation(['LShoulderPitch', 'LHipPitch', 'LKneePitch', 'LAnklePitch'],
                                                [1.0, -0.7, 1.05, -0.5], [[0.1], [0.1], [0.1], [0.1]], True)
            naos.motion.post.angleInterpolation(['LHipPitch', 'LKneePitch', 'LAnklePitch'],
                                                [-0.5, 1.1, -0.65], [[0.25], [0.25], [0.25]], True)
            naos.posture.post.goToPosture("StandInit", 0.5)

        elif (key_press == 'h'):
            fdict = load_motion("wave")
            try:
                naos.motion.post.angleInterpolationBezier(fdict["names"], fdict["times"], fdict["keys"])
            except BaseException, err:
                print err

        elif (key_press == 'j'):
            fdict = load_motion("macarena")
            try:
                naos.motion.post.angleInterpolationBezier(fdict["names"], fdict["times"], fdict["keys"])
            except BaseException, err:
                print err
        else:
            print("Doing nothing..")
            for nao in naos:
                x = 0
                theta = 0
                nao.motion.setWalkTargetVelocity(x, y, theta, frequency)
            time.sleep(CommandFreq)  # sleep after every command


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nao', action='append')
    args = parser.parse_args()

    if not args.nao:
        print('No naos to test specified!')
        sys.exit(0)

    main(args)
