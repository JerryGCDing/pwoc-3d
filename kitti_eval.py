import os.path
import sys
from tensorflow.python.keras import backend
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import imageio

import datasets
import network
import metrics
import utils


def eval(checkpoint, data):
    # construct model
    with backend.get_graph().as_default():
        net = network.Network()

    # load weights
    net.load_weights(checkpoint)

    # make predictions
    for idx, (images, gt) in enumerate(data):
        print("Evaluating sequence %d..." % idx)
        # predict scene flow
        res = net(inputs=images)[0].numpy()

        file_id = f'{datasets.KITTI_VALIDATION_IDXS[idx]:06d}'
        imageio.imwrite(f"./kitti_val/flow/{file_id}_10.png", utils.colored_flow(res[:, :, :2]))
        imageio.imwrite(f"./kitti_val/disp_1/{file_id}_10.png", utils.colored_disparity(res[:, :, 2]))
        imageio.imwrite(f"./kitti_val/disp_2/{file_id}_10.png",
                        utils.colored_disparity(res[:, :, 3], maxdisp=np.max(res[:, :, 2])))


if __name__ == "__main__":
    if not os.path.exists('./kitti_val'):
        os.mkdir('./kitti_val')
        os.mkdir('./kitti_val/disp_1')
        os.mkdir('./kitti_val/disp_2')
        os.mkdir('./kitti_val/flow')

    data = datasets.get_kitti_dataset(datasets.KITTI_VALIDATION_IDXS, batch_size=1)
    eval('./data/pwoc3d-kitti', data)
