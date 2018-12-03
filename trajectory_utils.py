import numpy as np
from kitti_eval.pose_evaluation_utils import pose_vec_to_mat


def convert_sequence_to_mat(poses):
    poses_mat = []

    for pose_vec in poses:
        pose_mat = pose_vec_to_mat(pose_vec)
        poses_mat.append(pose_mat)

    return poses_mat


def convert_and_change_coordinate_system(poses, new_coord_index=0):
    coord_pose = pose_vec_to_mat(poses[new_coord_index])

    out = []
    for pose_vec in poses:
        pose = pose_vec_to_mat(pose_vec)
        pose = np.dot(coord_pose, np.linalg.inv(pose))
        out.append(pose)

    return out


# ps - sequence of N poses (predicted pose vector from network output, e.g. N=5)
# ps_arr - array of poses sequences
def merge_sequences_poses(ps_arr):
    #ps_arr = [convert_and_change_coordinate_system(ps) for ps in ps_arr]
    ps_arr = [convert_sequence_to_mat(ps) for ps in ps_arr]

    poses_global = []
    first_seq = True
    idx_global = 0
    for ps in ps_arr:
        if first_seq:
            # First sequence is already in global coordinates, take p0 and p1
            ps_ = ps[0:2]
            first_seq = False
        else:
            ps_ = []
            p = ps[1]
            p_ = np.dot(poses_global[idx_global], p)
            ps_.append(p_)

        idx_global += 1

        # Add transformed pose to global poses
        for pose_global in ps_:
            poses_global.append(pose_global)

    # convert to sequence
    print('Tam secuencia: %d ' % len(poses_global))
    poses_global = np.stack(poses_global)
    # get interesting values
    txs = poses_global[:, 0, 3]
    tys = poses_global[:, 1, 3]
    tzs = poses_global[:, 2, 3]

    return txs, tys, tzs
