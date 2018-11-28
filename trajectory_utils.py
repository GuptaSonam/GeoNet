import numpy as np
from kitti_eval.pose_evaluation_utils import pose_vec_to_mat


def convert_and_change_coordinate_system(poses, new_coord_index=0):
    coord_pose = pose_vec_to_mat(poses[new_coord_index])

    out = []
    for pose_vec in poses:
        pose = pose_vec_to_mat(pose_vec)
        pose = np.dot(coord_pose, np.linalg.inv(pose))
        out.append(pose)

    return out


# ps - sequence of N poses (predicted pose vector from network output, e.g. N=5)
# ps_arr - array of poses sequences with a single overlapping pose (last element)
def merge_sequences_poses(ps_arr):
    ps_arr = [convert_and_change_coordinate_system(ps) for ps in ps_arr]

    poses_global = []
    first_seq = True
    id_global = 1

    for ps in ps_arr:
        if first_seq:
            ps_ = ps
            first_seq = False
        else:
            # TODO: code
            p = ps[-1]
            p_ = np.dot(poses_global[id_global], p)
            ps_.append(p_)

        id_global = id_global+1

        # Add to global poses
        for pose_global in ps_:
            poses_global.append(pose_global)

    # get interesting values
    print(poses_global)
    txs = poses_global[:, 0, 3]
    tys = poses_global[:, 1, 3]
    tzs = poses_global[:, 2, 3]

    return txs, tys, tzs


    # poses_global = []
    # ps_prev_last = None
    # for ps in ps_arr:
    #     if ps_prev_last is None: # first group - do nothing
    #         ps_ = ps
    #     else: # use overlapping pose to translate current ps to global coordinate system
    #         ps_ = []
    #         for p in ps:
    #             p_ = np.dot(ps_prev_last, p)
    #             ps_.append(p_)
    #
    #     ps_prev_last = ps_[-1]
    #
    #     # skip the last overlapping pose
    #     for pose_global in ps_[:-1]:
    #         poses_global.append(pose_global)
    #
    # # get interesting values
    # poses_stacked = np.stack(poses_global)
    # txs = poses_stacked[:, 0, 3]
    # tys = poses_stacked[:, 1, 3]
    # tzs = poses_stacked[:, 2, 3]
    #
    # return txs, tys, tzs # example - outputing just the position (x,y,z)
