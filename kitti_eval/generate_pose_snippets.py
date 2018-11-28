from __future__ import division
import inspect
import os
import sys
# hack to import files from different directory
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import numpy as np
import argparse
from pose_evaluation_utils import mat2euler, quat2mat, dump_pose_seq_TUM
from geonet_test_pose import load_test_frames, load_times

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",       type=str, default="kitti", help="Dataset name (kitti, tum)")
parser.add_argument("--dataset_dir",   type=str,                  help="Path to dataset files")
parser.add_argument("--output_dir",    type=str,                  help="Path to output pose snippets")
parser.add_argument("--pose_test_seq", type=str, default="9",     help="Sequence name to generate groundtruth pose snippets")
parser.add_argument("--seq_length",    type=int, default=5,       help="Sequence length of pose snippets")

args = parser.parse_args()


def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False


def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Load test frames
    N, test_frames = load_test_frames(args)
    # Load timestamps
    times = load_times(args)

    # Read groung-truth poses
    if args.dataset == 'kitti':
        pose_gt_dir = args.dataset_dir + 'poses/'

        with open(pose_gt_dir + '%.2d.txt' % int(args.pose_test_seq), 'r') as f:
            poses = f.readlines()
        poses_gt = []
        for pose in poses:
            pose = np.array([float(s) for s in pose[:-1].split(' ')]).reshape((3, 4))
            rot = np.linalg.inv(pose[:, :3])
            tran = -np.dot(rot, pose[:, 3].transpose())
            rz, ry, rx = mat2euler(rot)
            poses_gt.append(tran.tolist() + [rx, ry, rz])
        poses_gt = np.array(poses_gt)

    if args.dataset == 'tum':
        pose_gt_dir = args.dataset_dir

        with open(pose_gt_dir + '%s/groundtruth.txt' % args.pose_test_seq, 'r') as f:
            poses = f.readlines()
        # Filter out comment lines and timestamps then convert to float
        poses = [pose.split()[1:] for pose in poses if not pose.startswith('#')]
        poses = np.array(poses).astype(np.float)

        # Translate poses from [tx ty tz qx qy qz qw] to [tx ty tz rx ry rz]
        poses_gt = []
        for pose in poses:
            tran = pose[0:3]
            # Get quaternion as qw qx qy qz (original format is qx qy qz qw)
            quat = pose[np.array([6, 3, 4, 5])]
            rot = quat2mat(quat)
            rz, ry, rx = mat2euler(rot)
            poses_gt.append(tran.tolist() + [rx, ry, rz])

    # Store sequences of GT poses
    max_src_offset = (args.seq_length - 1)//2
    for tgt_idx in range(N):
        if not is_valid_sample(test_frames, tgt_idx, args.seq_length):
            continue
        if tgt_idx % 100 == 0:
            print('Progress: %d/%d' % (tgt_idx, N))
        pred_poses = poses_gt[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        out_file = args.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
        dump_pose_seq_TUM(out_file, pred_poses, curr_times)


main()
