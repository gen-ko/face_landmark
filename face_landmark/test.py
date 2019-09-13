#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import os
from face_detection import face_detector
from face_landmark import landmark_detector
import cv2
import time
import argparse

from lambdaflow.visualization import image_overlay


def parse_arguments():
    #BARN1_PATH = os.environ['BARN1_PATH']
    #BARN2_PATH = os.environ['BARN2_PATH']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, required=True)
    parser.add_argument('--mode', dest='mode', type=str, default='video')
    parser.add_argument('--source', dest='source', type=str, required=True)
    parser.add_argument('--dump_result', dest='dump_result', type=str, default=None)
    return parser.parse_args()


def main(args):

    fd = face_detector.FaceDetector()
    ld = landmark_detector.LandmarkDetector(model_path=args.model)
    
    if args.mode == 'video':
        cap = cv2.VideoCapture(args.source)
    elif args.mode == 'camera':
        cap = cv2.VideoCapture(int(args.source))
    
    if cap.isOpened(): 
        # get vcap property 
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        raise ValueError('video cannot be opened')
    
    if args.dump_result is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(args.dump_result,fourcc, fps, (int(w),int(h)))

    while True:
        _, frame = cap.read()


        if frame is None:
            print('exiting: all frames in specified video are iterated over')
            exit(0)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pts = ld.infer(frame_rgb, fd)

        frame = image_overlay.landmark(frame, pts)

        try:
            cv2.imshow('frame', frame)
        except:
            print('inferred 1 image')
        
        if args.dump_result is not None:
            out.write(frame)

        key = cv2.waitKey(delay=1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    if args.dump_result is not None:
        out.release()
    cv2.destroyAllWindows()
    print('script closed')
    
    return






if __name__ == '__main__':
    args = parse_arguments()
    main(args)
