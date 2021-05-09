from tensorflow.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.models import Sequential


conv_sizes = [16, 32, 64, 128]
dense_sizes = [512, 256, 128, 64]

activations = ['relu']
optimizers = ['Rmsprop']

epochs = 50


for conv in conv_sizes:
	for dense in dense_sizes:
		NAME = '{0}-first_conv-{1}-nodes-{2}-denseS-{4}-conv_layers-{3}x{3}--{5}mode'.format(
                                        layer, layer_size, dense_size, sizeIM, conv_layer,
                                        color_mode,)
