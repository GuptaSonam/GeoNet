from __future__ import division
import inspect
import os
import sys
# hack to import files from different directory
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import numpy as np
import tensorflow as tf
from pose_evaluation_utils import mat2euler, dump_pose_seq_TUM
from geonet_test_pose import load_test_frames, load_times

flags = tf.app.flags
flags.DEFINE_string("dataset",                  "kitti",   "Dataset name (kitti, tum)")
#parser.add_argument("--dataset",     type=str, help="name of dataset (kitti, tum)")
flags.DEFINE_string("dataset_dir",                  "",    "Path to dataset files")
#parser.add_argument("--dataset_dir", type=str, help="path to dataset files")
flags.DEFINE_string("output_dir",                 None,    "Path to output pose snippets")
#parser.add_argument("--output_dir",  type=str, help="path to output pose snippets")
flags.DEFINE_string("pose_test_seq",                "9",   "Sequence name to generate groundtruth pose snippets")
#parser.add_argument("--seq_id",      type=str, default="9", help="sequence name to generate groundtruth pose snippets")
flags.DEFINE_integer("seq_length",                   5,    "Sequence length of pose snippets")
#arser.add_argument("--seq_length",  type=int, default=5, help="sequence length of pose snippets")

opt = flags.FLAGS


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
    pose_gt_dir = opt.dataset_dir + 'poses/'
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)
    # Load test frames
    N, test_frames = load_test_frames(opt)
    # Load time file
    times = load_times(opt)

    with open(pose_gt_dir + '%.2d.txt' % int(opt.pose_test_seq), 'r') as f:
        poses = f.readlines()
    poses_gt = []
    for pose in poses:
        pose = np.array([float(s) for s in pose[:-1].split(' ')]).reshape((3, 4))
        rot = np.linalg.inv(pose[:, :3])
        tran = -np.dot(rot, pose[:, 3].transpose())
        rz, ry, rx = mat2euler(rot)
        poses_gt.append(tran.tolist() + [rx, ry, rz])
    poses_gt = np.array(poses_gt)

    max_src_offset = (opt.seq_length - 1)//2
    for tgt_idx in range(N):
        if not is_valid_sample(test_frames, tgt_idx, opt.seq_length):
            continue
        if tgt_idx % 100 == 0:
            print('Progress: %d/%d' % (tgt_idx, N))
        pred_poses = poses_gt[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        curr_times = times[tgt_idx - max_src_offset:tgt_idx + max_src_offset + 1]
        out_file = opt.output_dir + '%.6d.txt' % (tgt_idx - max_src_offset)
        dump_pose_seq_TUM(out_file, pred_poses, curr_times)

main()

