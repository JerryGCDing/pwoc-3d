from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import backend
from keras_flops import get_flops

import network
import datasets

with backend.get_graph().as_default():
    net = network.Network()

data = datasets.get_kitti_dataset(datasets.KITTI_VALIDATION_IDXS, batch_size=1)
images, gt = next(data)
inp = Input(images)
out = net(inputs=inp)
model = Model(inp, out)

flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03}G")
