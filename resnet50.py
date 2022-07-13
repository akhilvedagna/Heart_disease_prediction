import keras
from keras.layers import *
import keras.models



def con1d_layer(X, fils, kel, strds = 1):
    X = Conv1D(filters = fils, kernel_size = kel, strides = strds)(X)
    X = BatchNormalization()(X)
    return X

def convolutional_layer_in_resnet(X, fils, kel, strds = 1):
    X = con1d_layer(X, fils, kel, strds)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    return X

def resnet_block_1(X, filts, kel = 1, strds = 3):
    f1, f2, f3 = filts
    X_skip = X
    
    X = convolutional_layer_in_resnet(X, f1, kel)
    X = convolutional_layer_in_resnet(X, f2, kel)    
    
    X = con1d_layer(X, f3, kel, strds=1)
    
    X_skip = Conv1D(filters = f3, kernel_size = kel, strides = strds)(X)
    X_skip = BatchNormalization()(X)
    
    X = Add()([X, X_skip])
    
    X = ReLU()(X)
    
    return X

def resnet_block_2(X, filts, kel = 1):
    f1, f2, f3 = filts
    X_skip = X

    X = convolutional_layer_in_resnet(X, f1, kel)
    X = convolutional_layer_in_resnet(X, f2, kel)    
    
    X = con1d_layer(X, f3, kel)
    
    X = Add()([X, X_skip])
    
    X = ReLU()(X)
    
    return X
    
    
def Resnet50(Input_shape):
    kel = 1
    
    X_Input = Input(Input_shape)
    X = con1d_layer(X_Input, 32, kel)
    X = ReLU()(X)
    X = MaxPooling1D()(X)
    
    X = resnet_block_1(X, [64, 64, 256])
    X = resnet_block_2(X, [64, 64, 256])
    X = resnet_block_2(X, [64, 64, 256])
    
    X = resnet_block_1(X, [128,128,512])
    X = resnet_block_2(X,[128,128,512])
    X = resnet_block_2(X,[128,128,512])
    X = resnet_block_2(X,[128,128,512])
    
    X = resnet_block_1(X, [256, 256, 1024])
    X = resnet_block_2(X, [256, 256, 1024])
    X = resnet_block_2(X, [256, 256, 1024])
    X = resnet_block_2(X, [256, 256, 1024])
    X = resnet_block_2(X, [256, 256, 1024])
    X = resnet_block_2(X, [256, 256, 1024])
    
    X = resnet_block_1(X, [512, 512, 2048])
    X = resnet_block_2(X, [512, 512, 2048])
    X = resnet_block_2(X, [512, 512, 2048])
    
    X = AveragePooling1D()(X)
#     X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(1, activation = "sigmoid")(X)
    
    model = keras.models.Model(inputs = X_Input , outputs = X, name='ResNet50')
    return model