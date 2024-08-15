import math
from pathlib import Path

import numpy as np
from keras.optimizers import Nadam
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.metrics import AUC
from keras.layers import ELU
from keras.models import load_model
from keras.losses import binary_crossentropy
from sklearn.utils import class_weight

from utils import PlotLosses

def step_decay(epoch):
    initial_lrate = 0.00085 #0.001
    drop = 1 #0.95
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    # print(lrate)
    # return lrate
    return 0.000001

def train_model(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, sample_weights, log_dir, retrain_model=False):
    if not retrain_model:
        model.compile(loss=binary_crossentropy,optimizer=Nadam(),metrics=[AUC(name='auc')]) 
    lrate = LearningRateScheduler(step_decay)
    
    # train_run = "Model-{}".format(int(time.time())) 
    # tensorboard = TensorBoard(log_dir='/home/ahmedelkaffas/logs/{}'.format(train_run))#time()#NAME
    # tensorboard = TensorBoard(log_dir=log_dir / Path(train_run))
    
    y_integers = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_integers), y=y_integers) #None
    weights = dict(enumerate(class_weights))
    print('Weights')
    print(weights)
    
    if not retrain_model:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    else:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    mc = ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
    history = model.fit(x_train, y_train,
              validation_data=(x_valid,y_valid),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[lrate, es, mc, PlotLosses()],
              #callbacks=[callbacks_list],
            #   callbacks=[PlotLosses()],
              #callbacks=[ta.utils.live()],
              verbose=1, 
              class_weight = weights) #callbacks=[tensorboard], sample_weight = smpWeights, class_weight = weights, 
    
    if not retrain_model:
        model = load_model("best_model.keras", custom_objects = {"ELU": ELU}) #TODO
    
    return model, history