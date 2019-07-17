# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: zhangzhipeng2017@ia.ac.cn
# Detail: test siamese on a specific video (provide init bbox and video file)
# ------------------------------------------------------------------------------

import _init_paths
import os
import cv2
import random
import argparse
import numpy as np

import models.models as models

from os.path import exists, join
from torch.autograd import Variable
from tracker.siamfc import SiamFC
from tracker.siamrpn import SiamRPN
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou


def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    parser.add_argument('--arch', default='SiamRPNRes22', type=str, help='backbone architecture')
    parser.add_argument('--resume', default='./snapshot/CIResNet22RPN.model', type=str, help='pretrained model')
    parser.add_argument('--video', default='./data/%04d.png', type=str, help='video file path')
    parser.add_argument('--init_bbox', default=None, help='bbox in the first frame None or [lx, ly, w, h]')
    parser.add_argument('--output', default='./output', help='path of the output')
    args = parser.parse_args()

    return args


def track_video(tracker, model, video_path, init_box=None):

    #assert os.path.isfile(video_path), "please provide a valid video file"

    cap = cv2.VideoCapture(video_path)
    display_name = 'Video: {}'.format(video_path.split('/')[-1])
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 960, 720)
    success, frame = cap.read()
    cv2.imshow(display_name, frame)

    if success is not True:
        print("Read failed.")
        exit(-1)

    # init
    if init_box is not None:
        lx, ly, w, h = init_box
        target_pos = np.array([lx + w/2, ly + h/2])
        target_sz = np.array([w, h])
        state = tracker.init(frame, target_pos, target_sz, model)  # init tracker

    else:
        while True:

            frame_disp = frame.copy()

            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       1, (0, 0, 255), 1)

            lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])
            state = tracker.init(frame_disp, target_pos, target_sz, model)  # init tracker

            break
    i = 0
    restarts = 0
    #assert os.path.isfile('./bbox'), "Please create a directory called bbox to store the values of the bounding boxes"
    f = open('./bbox/SiamRPN.txt','+a')
    box_color = (0,255,0)
    while True:
        ret, frame = cap.read()

        if frame is None:
            return

        frame_disp = frame.copy()

        timer = cv2.getTickCount()
        # Draw box
        state = tracker.track(state, frame_disp)  # track

        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(location[1] + location[3])
        font_color = (0, 0, 255)
        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), box_color, 5)
        cv2.putText(frame_disp, 'Restarts: '+str(restarts), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   font_color, 2)
        cv2.putText(frame_disp, 'Frame Number: '+'{:04d}'.format(i), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   font_color, 2)
        cv2.putText(frame_disp, 'FPS: '+str(fps), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   font_color, 2)
        frame_name = '{:04d}'.format(i)
        #assert os.path.isfile(args.output), "Please create a directory to store the output files"
        cv2.imwrite("./output/"+str(frame_name)+".png",frame_disp)

        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        bbox = [x,y,w,h]
        box = [str(i) for i in bbox]
        f.write(box[0]+"\t"+box[1]+"\t"+box[2]+"\t"+box[3]+"\n")

        # Display the resulting frame
        cv2.imshow(display_name, frame_disp)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            restarts = restarts + 1
            if restarts%2 == 1:
                box_color = (255,0,0)
            else:
                box_color = (0,255,0)

            #ret, frame = cap.read()
            #frame_disp = frame.copy()

            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       1.5,
                       (0, 0, 0), 1)

            cv2.imshow(display_name, frame_disp)
            lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])
            state = tracker.init(frame_disp, target_pos, target_sz, model)
        i = i + 1
    # When everything done, release the capture
    os.rename('./bbox/SiamRPN.txt','./bbox/SiamRPN_'+str(restarts)+'.txt')
    f.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # prepare model (SiamRPN or SiamFC)

    # prepare tracker
    info = edict()
    info.arch = args.arch
    info.dataset = args.video
    info.epoch_test = True
    info.cls_type = 'thinner'

    if 'FC' in args.arch:
        net = models.__dict__[args.arch]()
        tracker = SiamFC(info)
    else:
        net = models.__dict__[args.arch](anchors_nums=5, cls_type='thinner')
        tracker = SiamRPN(info)

    print('[*] ======= Track video with {} ======='.format(args.arch))

    net = load_pretrain(net, args.resume)
    net.eval()
    net = net.cuda()

    # check init box is list or not
    if not isinstance(args.init_bbox, list) and args.init_bbox is not None:
        args.init_bbox = list(eval(args.init_bbox))
    else:
        pass

    track_video(tracker, net, args.video, init_box=args.init_bbox)

if __name__ == '__main__':
    main()
