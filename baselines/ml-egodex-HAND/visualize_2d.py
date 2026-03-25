'''
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Script for reprojecting 3D skeletal annotations into the 2D video. 
Note that there may be some perspective error in the reprojection.
'''
import argparse
from simple_dataset import SimpleDataset
from utils.skeleton_tfs import RIGHT_FINGERS, RIGHT_INDEX, RIGHT_THUMB, RIGHT_RING, RIGHT_MIDDLE, RIGHT_LITTLE
from utils.skeleton_tfs import LEFT_FINGERS, LEFT_INDEX, LEFT_THUMB, LEFT_RING, LEFT_MIDDLE, LEFT_LITTLE
from utils.draw_utils import draw_line_sequence, imgs_to_mp4, map_fingers_to_colors
from utils.data_utils import convert_to_camera_frame

QUERY_TFS = RIGHT_FINGERS + ['rightHand', 'rightForearm'] + LEFT_FINGERS + ['leftHand', 'leftForearm']

def main(args):
    # use simple pytorch dataset to load the data
    dataset = SimpleDataset(args.data_dir, query_tfs=QUERY_TFS)
    tf2idx = {k: i for i, k in enumerate(QUERY_TFS)}
    num_transitions = dataset.cumulative_len[args.num_episodes - 1]

    def get_finger_pts(finger_tf_names, tfs_in_cam, right=True):
        hand_name = 'rightHand'
        if not right:
            hand_name = 'leftHand'
        
        finger_points = [tfs_in_cam[tf2idx[hand_name], :3, -1]] # grab 3D position from SE(3) pose
        for tfname in finger_tf_names:
            finger_points.append(tfs_in_cam[tf2idx[tfname], :3, -1])

        return finger_points
    
    right_dict = {'index': RIGHT_INDEX, 'thumb': RIGHT_THUMB, 'middle': RIGHT_MIDDLE, 'ring': RIGHT_RING, 'little': RIGHT_LITTLE}
    left_dict = {'index': LEFT_INDEX, 'thumb': LEFT_THUMB, 'middle': LEFT_MIDDLE, 'ring': LEFT_RING, 'little': LEFT_LITTLE}

    def draw_hand(hand_dict, tfs_in_cam, cam_img, cam_int, right=True):
        # draw fingers
        for finger in ['little', 'ring', 'middle', 'index', 'thumb']: # roughly stack lines so closer fingers are drawn on top
            points = get_finger_pts(hand_dict[finger], tfs_in_cam, right)
            draw_line_sequence(points, cam_img, cam_int,
                           color=map_fingers_to_colors([finger])[0].tolist())
        
        # draw forearm
        if right:
            forearm_points = [tfs_in_cam[tf2idx['rightForearm'], :3, -1]]
            forearm_points.append(tfs_in_cam[tf2idx['rightHand'], :3, -1])
        else:
            forearm_points = [tfs_in_cam[tf2idx['leftForearm'], :3, -1]]
            forearm_points.append(tfs_in_cam[tf2idx['leftHand'], :3, -1])
        draw_line_sequence(forearm_points, cam_img, cam_int,
                           color=map_fingers_to_colors(['middle'])[0].tolist())
        
    out_imgs = []
    for i in range(num_transitions):
        tfs, cam_ext, cam_int, cam_img, _, _ = dataset[i]
        cam_img = cam_img.detach().cpu().numpy().transpose(1, 2, 0) # C x H x W -> H x W x C

        # transform poses to camera frame
        tfs_in_cam = convert_to_camera_frame(tfs, cam_ext)

        # draw hands
        draw_hand(right_dict, tfs_in_cam, cam_img, cam_int, right=True)
        draw_hand(left_dict, tfs_in_cam, cam_img, cam_int, right=False)

        out_imgs.append(cam_img)

    # write to video
    imgs_to_mp4(out_imgs, args.output_mp4)
    print('Done. Video saved to: {}'.format(args.output_mp4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path to data directory')
    parser.add_argument('--num_episodes', help='number of episodes to visualize', default=1)
    parser.add_argument('--output_mp4', help='where to save the output video', default='output.mp4')
    args = parser.parse_args()

    try:
        main(args)
    except ValueError as exp:
        print("Error:", exp)