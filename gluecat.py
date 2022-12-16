from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torch.multiprocessing
from tqdm import tqdm

import cv2
from scipy.spatial.distance import cdist

from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified, frame2tensor)

from models.matching import Matching
from models.matchingsuperglue import Matching_ori
from sjlee.loss import loss_superglue

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--eval', action='store_true',
    help='Perform the evaluation'
            ' (requires ground truth pose and intrinsics)')

parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1023,
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
            'resize to the exact dimensions, if one number, resize the max '
            'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--cache', action='store_true',
    help='Skip the pair if output .npz files are already found')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization based on OpenCV instead of Matplotlib')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')

parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
parser.add_argument(
    '--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
    help='Path to the list of image pairs for evaluation')
parser.add_argument(
    '--shuffle', action='store_true',
    help='Shuffle ordering of pairs before processing')
parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')

parser.add_argument(
    '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--eval_output_dir', type=str, default='test_matches',
    help='Path to the directory in which the .npz results and optional,'
            'visualizations are written')
parser.add_argument(
    '--learning_rate', type=float, default=0.0001,  #0.0001
    help='Learning rate')

parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')
parser.add_argument(
    '--train_path', type=str, default='/home/cvlab09/projects/seungjun_an/dataset/train2014/', 
    help='Path to the directory of training imgs.')
parser.add_argument(
    '--epoch', type=int, default=1,
    help='Number of epoches')




if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    # make sure the flags are properly used
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    numOftrainSet = 20

    # store viz results
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization images to',
        'directory \"{}\"'.format(eval_output_dir))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to('cuda')
    matching.load_state_dict(torch.load('/home/cvlab09/projects/seungjun_an/superglue_test/model_state_dict_epoch4.pth'))

    matching_ori = Matching_ori(config).eval().to('cuda')

    matching2 = Matching(config).eval().to('cuda')
    sum_loss = 0.


    device = 'cuda'
    
    for epoch in range(1, opt.epoch+1):
        epoch_loss = 0
        
        ##superglue.double().train()  ##########################################
        for i in range(numOftrainSet):
            file_name =opt.train_path+str(i+1)+'.jpg' 
            image0, inp0, scales0 = read_image(
                    file_name, opt.resize, 0, opt.resize_float)
            
            if str(type(image0)) != '<class \'numpy.ndarray\'>' : 
                continue

            width, height = image0.shape[:2]

            corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)

            warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)


            

            # get the corresponding warped image
            M = cv2.getPerspectiveTransform(corners, corners + warp)
            warped = cv2.warpPerspective(src=image0, M=M, dsize=(image0.shape[1], image0.shape[0]))

            inp1 = frame2tensor(warped)

            #print(i)

            scores, data, pred = matching({'image0': inp0, 'image1': inp1})
            

            #################################################################################################

            if data['skip_train'] : continue


            key1, key2, des1, des2 = pred['keypoints0'], pred['keypoints1'], pred['descriptors0'], pred['descriptors0']
            #all match 만들자
            ###################################################################
            
            kp1 = key1.squeeze()
            kp2 = key2.squeeze()
            kp1_np = np.array(key1.cpu()).squeeze()
            kp2_np = np.array(key2.cpu()).squeeze()
            descs1 = des1.cpu().detach().numpy().squeeze().transpose(0, 1)
            descs2 = des2.cpu().detach().numpy().squeeze().transpose(0, 1)
            


            # obtain the matching matrix of the image pair
            kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :] 
            
            if len(kp1_projected) == 1 or len(kp2_np) == 1 : continue
            
            dists = cdist(kp1_projected, kp2_np)

            min1 = np.argmin(dists, axis=0)
            min2 = np.argmin(dists, axis=1)

            min1v = np.min(dists, axis=1)
            min1f = min2[min1v < 3]

            xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
            matches = np.intersect1d(min1f, xx)

            missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
            missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

            MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
            MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
            MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
            all_matches = np.concatenate([MN, MN2, MN3], axis=1)
            all_matches = torch.tensor(all_matches).unsqueeze(1)
            

            #######################################################################

            if data['skip_train'] == True: # image has no keypoint
                continue

            #print(Loss)

            
            # for every 50 images, print progress and visualize the matches
        
         

            ### eval ###
            # Visualize the matches.
            print('model->eval') ############################################################################
            #matching.eval()
            image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
            
            image0, image1 = image0[0], image1[0]
            
            kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
            matches, conf = data['matches0'].cpu().detach().numpy(), data['matching_scores0'].cpu().detach().numpy()
            
            kpts0, kpts1 = kpts0[0], kpts1[0]
            
            image0 = read_image_modified(image0, opt.resize, opt.resize_float)
            image1 = read_image_modified(image1, opt.resize, opt.resize_float)
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            viz_path = eval_output_dir / '{}_trainedcatmatches.{}'.format(str(i), opt.viz_extension)
            color = cm.jet(mconf)
            stem = file_name
            text = []

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, stem, stem, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches')

            print('################################################################')################################
        



            scores, data, pred = matching_ori({'image0': inp0, 'image1': inp1})
            

            #################################################################################################

            if data['skip_train'] : continue


            key1, key2, des1, des2 = pred['keypoints0'], pred['keypoints1'], pred['descriptors0'], pred['descriptors0']
            #all match 만들자
            ###################################################################
            
            kp1 = key1.squeeze()
            kp2 = key2.squeeze()
            kp1_np = np.array(key1.cpu()).squeeze()
            kp2_np = np.array(key2.cpu()).squeeze()
            descs1 = des1.cpu().detach().numpy().squeeze().transpose(0, 1)
            descs2 = des2.cpu().detach().numpy().squeeze().transpose(0, 1)
            


            # obtain the matching matrix of the image pair
            kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :] 
            
            if len(kp1_projected) == 1 or len(kp2_np) == 1 : continue
            
            dists = cdist(kp1_projected, kp2_np)

            min1 = np.argmin(dists, axis=0)
            min2 = np.argmin(dists, axis=1)

            min1v = np.min(dists, axis=1)
            min1f = min2[min1v < 3]

            xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
            matches = np.intersect1d(min1f, xx)

            missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
            missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

            MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
            MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
            MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
            all_matches = np.concatenate([MN, MN2, MN3], axis=1)
            all_matches = torch.tensor(all_matches).unsqueeze(1)
            

            #######################################################################

            if data['skip_train'] == True: # image has no keypoint
                continue

            #print(Loss)

            
            # for every 50 images, print progress and visualize the matches
        
         

            ### eval ###
            # Visualize the matches.
            print('model->eval') ############################################################################
            #matching.eval()
            image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
            
            image0, image1 = image0[0], image1[0]
            
            kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
            matches, conf = data['matches0'].cpu().detach().numpy(), data['matching_scores0'].cpu().detach().numpy()
            
            kpts0, kpts1 = kpts0[0], kpts1[0]
            
            image0 = read_image_modified(image0, opt.resize, opt.resize_float)
            image1 = read_image_modified(image1, opt.resize, opt.resize_float)
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            viz_path = eval_output_dir / '{}_originmatches.{}'.format(str(i), opt.viz_extension)
            color = cm.jet(mconf)
            stem = file_name
            text = []

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, stem, stem, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches')



            print('################################################################')################################
        



            scores, data, pred = matching2({'image0': inp0, 'image1': inp1})
            

            #################################################################################################

            if data['skip_train'] : continue


            key1, key2, des1, des2 = pred['keypoints0'], pred['keypoints1'], pred['descriptors0'], pred['descriptors0']
            #all match 만들자
            ###################################################################
            
            kp1 = key1.squeeze()
            kp2 = key2.squeeze()
            kp1_np = np.array(key1.cpu()).squeeze()
            kp2_np = np.array(key2.cpu()).squeeze()
            descs1 = des1.cpu().detach().numpy().squeeze().transpose(0, 1)
            descs2 = des2.cpu().detach().numpy().squeeze().transpose(0, 1)
            


            # obtain the matching matrix of the image pair
            kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :] 
            
            if len(kp1_projected) == 1 or len(kp2_np) == 1 : continue
            
            dists = cdist(kp1_projected, kp2_np)

            min1 = np.argmin(dists, axis=0)
            min2 = np.argmin(dists, axis=1)

            min1v = np.min(dists, axis=1)
            min1f = min2[min1v < 3]

            xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
            matches = np.intersect1d(min1f, xx)

            missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
            missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

            MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
            MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
            MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
            all_matches = np.concatenate([MN, MN2, MN3], axis=1)
            all_matches = torch.tensor(all_matches).unsqueeze(1)
            

            #######################################################################

            if data['skip_train'] == True: # image has no keypoint
                continue

            #print(Loss)

            
            # for every 50 images, print progress and visualize the matches
        
         

            ### eval ###
            # Visualize the matches.
            print('model->eval') ############################################################################
            #matching.eval()
            image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
            
            image0, image1 = image0[0], image1[0]
            
            kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
            matches, conf = data['matches0'].cpu().detach().numpy(), data['matching_scores0'].cpu().detach().numpy()
            
            kpts0, kpts1 = kpts0[0], kpts1[0]
            
            image0 = read_image_modified(image0, opt.resize, opt.resize_float)
            image1 = read_image_modified(image1, opt.resize, opt.resize_float)
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]
            viz_path = eval_output_dir / '{}_notraincatmatches.{}'.format(str(i), opt.viz_extension)
            color = cm.jet(mconf)
            stem = file_name
            text = []

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, stem, stem, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches')

            
