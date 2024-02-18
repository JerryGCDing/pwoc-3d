from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import backend
from keras_flops import get_flops

import network

with backend.get_graph().as_default():
    net = network.Network()

inp = Input((375, 1242, 3))
out = net(inputs=inp)
model = Model(inp, out)

flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03}G")
