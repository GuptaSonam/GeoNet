import argparse
import glob
import os
import numpy as np
from trajectory_utils import merge_sequences_poses
from kitti_eval.pose_evaluation_utils import read_file_list, associate
import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--gtruth_dir", type=str, help="Path to the directory with ground-truth snippets")
parser.add_argument("--pred_dir", type=str, help="Path to the directory with predicted trajectories")
args = parser.parse_args()

def load_gt(gt_dir):
    gt_poses = []

    with open(gt_dir + 'groundtruth.txt', 'r') as f:
        gt_poses = f.readlines()

    gt_poses = [pose.split()[1:] for pose in gt_poses if not pose.startswith('#')]
    gt_poses = np.array(gt_poses).astype(np.float)

    return gt_poses

def load_sequences(sequence_dir):
    seq_arr = []

    gt_seqs_files = sorted(glob.glob(sequence_dir + "*.txt"))
    # filter out files so only last frame overlaps
    # gt_seqs_files = [gt_seqs_files[i] for i in range(len(gt_seqs_files)) if i % 4 == 0]
    for file in gt_seqs_files:
        with open(file, 'r') as f:
            lines = f.readlines()

        # Remove timestamps
        lines = [line.split()[1:] for line in lines]
        seq = np.array(lines).astype(np.float)
        seq_arr.append(seq)

    return seq_arr


def load_scaled_sequences(gt_dir, pred_dir):
    gt_seqs = []
    pred_seqs_scaled = []

    pred_files = sorted(glob.glob(pred_dir + '/*.txt'))
    for i in range(len(pred_files)):
        gt_file = gt_dir + os.path.basename(pred_files[i])

        # Find matches between predictions and GT poses
        gt_list = read_file_list(gt_file)
        pred_list = read_file_list(pred_files[i])
        matches = associate(gt_list, pred_list, 0, 0.01)
        if len(matches) < 2:
            return False

        # Convert to float
        gt_seq = np.array([[float(val) for val in gt_list[a][:]] for a,b in matches])
        pred_seq = np.array([[float(val) for val in pred_list[a][:]] for a,b in matches])

        # Align first matched frames
        offset = gt_seq[0,0:3] - pred_seq[0,0:3]
        pred_seq[:,0:3] += offset[None,:]

        # Get scaling factor (sx, sy, sz)
        sx = np.sum(gt_seq[:,0] * pred_seq[:,0]) / np.sum(pred_seq[:,0] ** 2)
        sy = np.sum(gt_seq[:,1] * pred_seq[:,1]) / np.sum(pred_seq[:,1] ** 2)
        sz = np.sum(gt_seq[:,2] * pred_seq[:,2]) / np.sum(pred_seq[:,2] ** 2)

        # Scale predictions to GT scale
        scaled_xs = pred_seq[:,0] * sx
        scaled_ys = pred_seq[:,1] * sy
        scaled_zs = pred_seq[:,2] * sz
        pred_seq_scaled = pred_seq
        pred_seq_scaled[:,0:3] = np.array([scaled_xs, scaled_ys, scaled_zs]).transpose()

        # Add to list of scaled predictions
        gt_seqs.append(gt_seq)
        pred_seqs_scaled.append(pred_seq)

    return gt_seqs, pred_seqs_scaled



def plot_trajectories(gt_points, pred_points):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.gca()

    #ax.plot(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], label='GT map')
    ax.plot(gt_points[:, 0], gt_points[:, 1], label='GT map')
    #ax.plot(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], label='Predicted map')
    ax.plot(pred_points[:, 0], pred_points[:, 1], label='Predicted map')
    ax.legend()
    plt.show()


def main():
    # # load sequences
    # poses_gt = load_sequences(args.gtruth_dir)
    # #poses_gt = load_gt(args.gtruth_dir)
    # poses_pred = load_sequences(args.pred_dir)
    #
    # # merge poses
    # gt_x, gt_y, gt_z = merge_sequences_poses(poses_gt)
    # merged_gt = np.array([gt_x, gt_y, gt_z]).transpose()
    # #merged_gt = poses_gt[:,0:3]
    # pred_x, pred_y, pred_z = merge_sequences_poses(poses_pred)
    # merged_pred = np.array([pred_x, pred_y, pred_z]).transpose()

    # Load and scale 5 frame sequences
    poses_gt, poses_pred = load_scaled_sequences(args.gtruth_dir, args.pred_dir)

    # Merge into a single trajectory by pose composition
    gt_x, gt_y, gt_z = merge_sequences_poses(poses_gt)
    merged_gt = np.array([gt_x, gt_y, gt_z]).transpose()
    pred_x, pred_y, pred_z = merge_sequences_poses(poses_pred)
    merged_pred = np.array([pred_x, pred_y, pred_z]).transpose()

    # Draw trajectories
    plot_trajectories(merged_gt, merged_pred)


main()
