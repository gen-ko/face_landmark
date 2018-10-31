#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import os
from face_detection import face_detector
from face_landmark import preprocess
from face_landmark import inputs
from face_landmark import train
import cv2
import time



def main():
    fan = train.FAN(checkpoint_path='/barn2/yuan/saved_model/face_landmark/v2_80_8891_0162', 
                    infer_graph=True,
                    train_path=None,
                    eval_path=None)
    #fan.freeze('/barn2/yuan/saved_model/fan_new')
    
    cap = cv2.VideoCapture('/barn2/zilong/Dataset/Internal/eye_closure/gen_eye_closure_dataset_11_20/gen_dataset_new_bbox_with_glass_11_20/WIN_20171129_09_50_24_Pro_05_1.avi')
    while True:
        _, frame = cap.read()
        if frame is None:
            print('warning: frame is none')
            exit(0)


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts, _ = fan.infer(frame_rgb)

        h, w, _ = frame.shape
        pts = (pts * np.array([w, h])).astype(int)

        for i in range(pts.shape[0]):
            cv2.circle(frame, tuple(pts[i]), 1, (0, 255, 0), 1)
            #cv2.putText(frame, 'score:' + str(score), org=(60, 40), fontFace=cv2.FONT_HERSHEY_DUPLEX,
            #            fontScale=0.4, color=(141, 25, 255))

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
    main()