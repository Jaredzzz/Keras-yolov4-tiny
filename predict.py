#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
from core.activation import Mish, Mish6
import tensorflow as tf
import numpy as np
from skimage import io
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 512, 512 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = config['predict']['obj_thresh'], config['predict']['nms_thresh']

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    infer_model = load_model(config['train']['saved_weights_name'],
                             compile=False,
                             custom_objects={'Mish': Mish,
                                             'Mish6': Mish6})

    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'webcam' in input_path:  # do detection on the first webcam
        video_reader = cv2.VideoCapture(int(input_path[-1]))
        # video_reader = cv2.VideoCapture("http://172.22.192.43:8080/video?dummy=param.mjpg")

        # the main loop
        batch_size  = 1
        images      = []
        time_count=0
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True: images += [image]

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):
                    image,allobj=draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)
                    if(len(batch_boxes)>0):
                        cv2.imwrite(output_path + str(time_count)+'.png', np.uint8(image))
                        f = open(output_path + str(time_count)+'.txt', 'a+')
                        for j in allobj:
                            f.write(str(j[0])+','+str(j[1])+','+str(j[2])+','+str(j[3])+','+str(j[4]) +"\n")
                        f.close()
                    cv2.imshow('video with bboxes', images[i])
                images = []
            time_count=time_count+1
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()        
    elif input_path[-4:] == '.mp4':  # do detection on a video
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                    for i in range(len(images)):
                        # draw bounding boxes on the image using labels
                        draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)   

                        # show the video with detection bounding boxes          
                        if show_window: cv2.imshow('video with bboxes', images[i])  

                        # write result to the output video
                        video_writer.write(images[i]) 
                    images = []
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        if show_window:
            cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       
    else:  # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        time_list = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes: xmin, ymin, xmax, ymax, severity, conf, classes
            batch_boxes, total_time = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)
            # print(boxes)
            print('total time: {0:.6f}s'.format(total_time))
            time_list.append(total_time)
            # draw bounding boxes on the image using labels
            boxes = batch_boxes[0]
            intensity_image, allobj = draw_boxes(image, boxes, config['model']['labels'], obj_thresh)
     
            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))
            if(len(boxes) > 0):
                f = open(output_path + image_path.split('/')[-1].split('.')[0]+'.txt', 'w+')
                for i in allobj:
                    # xmin, ymin, xmax, ymax, classes, conf
                    f.write(str(i[0])+','+str(i[1])+','+str(i[2])+','+str(i[3])+','+str(i[4])+"\n")
                f.close()
        average_time = sum(time_list[1:])/(len(time_list)-1)
        print('[INFO] average time: {0:.6f}s'.format(average_time))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
