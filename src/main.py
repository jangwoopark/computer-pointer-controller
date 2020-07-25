import logging as log
import cv2
import time
from input_feeder import InputFeeder
from argparse import ArgumentParser
from mouse_controller import MouseController
from face_detection import face_detection
from facial_landmarks_detection import facial_landmarks_detection
from head_pose_estimation import head_pose_estimation
from gaze_estimation import gaze_estimation
import os
import numpy as np
import math


def read_argument():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-face_m", "--face_detection_model", required=True, type=str,
                        help="Path to an face_detection_model xml file with a trained model.")
    parser.add_argument("-head_pose_m", "--head_pose_estimation", required=True, type=str,
                        help="Path to an head_pose_estimation xml file with a trained model.")
    parser.add_argument("-facial_m", "--facial_landmarks_detection", required=True, type=str,
                        help="Path to an facial_landmarks_detection xml file with a trained model.")
    parser.add_argument("-gaze_m", "--gaze_estimation", required=True, type=str,
                        help="Path to an gaze_estimation xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags for face_detect, face_landmark_detect, head_pose, gaze_est" 
                             "For ex --flags face_detect face_landmark_detect head_pose gaze_est"
                             "to see the visualization of different model outputs of each frame")

    return parser

def main():

    args = read_argument().parse_args()

    logger_obj = log.getLogger()
    
    if args.input == 'CAM':
        input_feeder = InputFeeder('cam')
    elif args.input.endswith('jpg') or args.input.endswith('bmp'):
        input_feeder = InputFeeder('image', args.input)
    elif args.input.endswith('mp4'):
        input_feeder = InputFeeder('video', args.input)
    else:
        logger_obj.error("Unsupported input, valid inputs are image(jpg and bmp), video file(mp4) or webcam/video stream.")

    face_detection_model = face_detection(args.face_detection_model, args.device, args.prob_threshold, args.cpu_extension)
    facial_landmarks_detection_model = facial_landmarks_detection(args.facial_landmarks_detection, args.device, args.cpu_extension) 
    head_pose_estimation_model = head_pose_estimation(args.head_pose_estimation, args.device, args.cpu_extension)
    gaze_estimation_model = gaze_estimation(args.gaze_estimation, args.device, args.cpu_extension)

    mouse_controller = MouseController('medium', 'fast')

    start_time = time.time()
    face_detection_model.load_model()
    face_detection_model_load_time = time.time()
    logger_obj.error("Face detection load time in seconds: {:.2f} ms".format((time.time() - start_time) * 1000))

    facial_landmarks_detection_model.load_model()
    facial_landmarks_detection_load_time = time.time()
    logger_obj.error("Facial Landmark detection load time in seconds: {:.2f} ms".format((time.time() - start_time) * 1000))


    head_pose_estimation_model.load_model()
    head_pose_estimation_load_time = time.time()
    logger_obj.error("Head pose detection load time in seconds: {:.2f} ms".format((time.time() - start_time) * 1000))


    gaze_estimation_model.load_model()
    gaze_estimation_load_time = time.time()
    logger_obj.error("Gaze estimation load time in seconds: {:.2f} ms".format((time.time() - start_time) * 1000))

    input_feeder.load_data()

    total_model_load_time = time.time() - start_time

    counter = 0
    inference_start_time = time.time()

    for flag, frame in input_feeder.next_batch():
           
        if not flag: 
            break

        pressed_key = cv2.waitKey(60)
        counter = counter + 1
        
        face_detection_output, coords = face_detection_model.predict(frame)

        head_pose_estimation_output = head_pose_estimation_model.predict(face_detection_output)

        left_eye_image, right_eye_image, eye_coord = facial_landmarks_detection_model.predict(face_detection_output)

        mouse_controller_coordinate, gaze_estimation_vector = gaze_estimation_model.predict(left_eye_image, right_eye_image,
                                                                             head_pose_estimation_output)


        preview_flag = args.previewFlags

        if len(preview_flag) != 0:
            
            preview_window = frame.copy()
            
            if 'face_detect' in preview_flag:

                cv2.rectangle(preview_window, (coords[0], coords[1]),
                                  (coords[2], coords[3]), (0, 0, 255), 3)

            if 'face_landmark_detect' in preview_flag:
                
                if 'face_detect' in preview_flag:
                    preview_window = face_detection_output

                cv2.rectangle(preview_window, (eye_coord[0][0], eye_coord[0][1]), (eye_coord[0][2], eye_coord[0][3]),
                              (255, 0, 255))
                cv2.rectangle(preview_window, (eye_coord[1][0], eye_coord[1][1]), (eye_coord[1][2], eye_coord[1][3]),
                              (255, 0, 255))

            if 'head_pose' in preview_flag:

                cv2.putText(preview_window,
                            "yaw:{:.1f} | pitch:{:.1f} | roll:{:.1f}".format(head_pose_estimation_output[0],
                                                                             head_pose_estimation_output[1],
                                                                             head_pose_estimation_output[2]),
                            (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1)
                

            if 'gaze_est' in preview_flag:

                yaw = head_pose_estimation_output[0]
                pitch = head_pose_estimation_output[1]
                roll = head_pose_estimation_output[2]

                focal_length = 950
                scale = 50

                center_of_face = (face_detection_output.shape[1] / 2, face_detection_output.shape[0] / 2, 0)
        
        if len(preview_flag) != 0:

            image = cv2.resize(preview_window, (500, 500))

        else:
            image = cv2.resize(frame, (500, 500))

        cv2.imshow('Visualization', image)

        mouse_controller.move(mouse_controller_coordinate[0], mouse_controller_coordinate[1])

        if pressed_key == 27:
            logger_obj.error("exit key is pressed..")
            break

    inference_time = round(time.time() - inference_start_time, 2)
    fps = int(counter) / inference_time


    logger_obj.error("counter {} seconds".format(counter))
    logger_obj.error("Total model load time in seconds: {:.2f} s".format(total_model_load_time))
    logger_obj.error("Total inference time in seconds: {:.2f} s".format(inference_time))
    logger_obj.error("fps {}".format(fps))
    
    input_feeder.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
