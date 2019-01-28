#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import os
from face_detection import face_detector
from face_landmark import landmark_detector
import cv2
import time
import argparse

from wheelheaps import draw

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default='/barn1/yuan/saved_models/3d_fan_light_128_2019-01-04/frozen_graph_epoch_0.pb')
    parser.add_argument('--mode', dest='mode', type=str, default='video')
    parser.add_argument('--source', dest='source', type=str, default='/barn1/yuan/test/phoning_testset/00000002.mp4')
    return parser.parse_args()


def main(args):

    fd = face_detector.FaceDetector()
    ld = landmark_detector.LandmarkDetector(model_path=args.model)
    
    if args.mode == 'video':
        cap = cv2.VideoCapture(args.source)
    elif args.mode == 'camera':
        cap = cv2.VideoCapture(int(args.source))
    while True:
        _, frame = cap.read()


        if frame is None:
            print('exiting: all frames in specified video are iterated over')
            exit(0)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pts = ld.infer(frame_rgb, fd)

        frame = draw.landmark(frame, pts)

        try:
            cv2.imshow('frame', frame)
        except:
            print('inferred 1 image')
        key = cv2.waitKey(delay=1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('script closed')
    
    return






if __name__ == '__main__':
    args = parse_arguments()
    main(args)