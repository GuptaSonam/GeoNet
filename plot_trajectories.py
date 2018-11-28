import argparse
import glob
import numpy as np
from trajectory_utils import merge_sequences_poses
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

def load_sequence(sequence_dir):
    seq_arr = []

    gt_seqs_files = sorted(glob.glob(sequence_dir + "*.txt"))
    for file in gt_seqs_files:
        with open(file, 'r') as f:
            lines = f.readlines()

        # Remove timestamps
        lines = [line.split()[1:] for line in lines]
        seq = np.array(lines).astype(np.float)
        seq_arr.append(seq)

    return seq_arr


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
    # load sequences
    #poses_gt = load_sequence(args.gtruth_dir)
    poses_gt = load_gt(args.gtruth_dir)
    poses_pred = load_sequence(args.pred_dir)

    # merge poses
    #gt_x, gt_y, gt_z = merge_sequences_poses(poses_gt)
    #merged_gt = np.array([gt_x, gt_y, gt_z]).transpose()
    merged_gt = poses_gt[:,0:3]
    pred_x, pred_y, pred_z = merge_sequences_poses(poses_pred)
    merged_pred = np.array([pred_x, pred_y, pred_z]).transpose()

    plot_trajectories(merged_gt, merged_pred)


main()
