from src import utils,mlp_model,mnist_data
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
def train(model,X,Y,batch_size,epoches,lr_schedule,weight_decay):
    train_loss, train_acc = [], []
    for epoch in range(epoches):
        loss, acc = model.one_epoch(X, Y, batch_size, epoch,lr_schedule,weight_decay,train = True)
        if epoch%10 == 0:
            print("Training step: Epoch {}/{}: Loss={}, Accuracy={}".format(epoch, epoches, loss, acc))
        train_loss.append(loss)
        train_acc.append(acc)
    return train_loss, train_acc, model.W


def parameter_sweep_hidden_layer():
    #parameter
    lr_sche=[200,200,0.1,0.01]
    batch_size = 256
    epoches = 200
    weight_decay=0
    np.random.seed(4)
    hidden_layer_list= np.linspace(10, 780, 10)
    v_acc_list=[]
    for i in hidden_layer_list:
        print("Hidden layer dim={}".format(int(i)))
        model=mlp_model.MLP_model(int(i))
        trainX, trainY, valX, valY, testX, testY = mnist_data.load_dataset()
        trainX, valX, testX = trainX/255, valX/255, testX/255
        trainY = utils.to_onehot(trainY)
        valY = utils.to_onehot(valY)
        testY = utils.to_onehot(testY)
        train_loss, train_acc, W= train(model, trainX, trainY, batch_size, epoches,lr_sche,weight_decay)

        v_loss, v_acc = model.one_epoch(valX, valY, batch_size, 1,lr_sche,weight_decay,train = False)
        v_acc_list.append(v_acc)
    fig = plt.figure(1)
    plt.xlabel("hidden layer dim")
    plt.ylabel("val_acc after 200 epochs")
    plt.plot(hidden_layer_list,v_acc_list)
    plt.savefig('./plot/sweep_hidden_layer.png')
    plt.show()
    return hidden_layer_list[v_acc_list.index(max(v_acc_list))]

def parameter_sweep_lr(hidden_layer):
    lr_max=[0.01,0.1,0.5,1,5]
    v_acc_list=[]
    for i in lr_max:
        print("lr={}".format(i))
        lr_sche=[100,200,i,i/10]
        batch_size = 256
        epoches = 200
        weight_decay=0
        np.random.seed(4)
        model=mlp_model.MLP_model(hidden_layer)
        trainX, trainY, valX, valY, testX, testY = mnist_data.load_dataset()
        trainX, valX, testX = trainX/255, valX/255, testX/255
        trainY = utils.to_onehot(trainY)
        valY = utils.to_onehot(valY)
        testY = utils.to_onehot(testY)
        train_loss, train_acc, W= train(model, trainX, trainY, batch_size, epoches,lr_sche,weight_decay)
        v_loss, v_acc = model.one_epoch(valX, valY, batch_size, 1,lr_sche,weight_decay,train = False)
        v_acc_list.append(v_acc)
    x = range(len(lr_max))
    fig = plt.figure(2)
    plt.xlabel("learning rate")
    plt.ylabel("val_acc after 200 epochs")
    plt.plot(x,v_acc_list,marker='o')
    plt.xticks(x,lr_max)
    plt.savefig('./plot/sweep_lr.png')
    plt.show()
    return lr_max[v_acc_list.index(max(v_acc_list))]

def parameter_sweep_weight_decay(hidden_layer,lr):
    lr_sche=[100,200,lr,lr/10]
    batch_size = 256
    epoches = 200
    weight_decay_list=[0.00001,0.0001,0.001,0.01]
    v_acc_list=[]
    np.random.seed(4)
    model=mlp_model.MLP_model(hidden_layer)
    for i in weight_decay_list:
        print("weight decay parameter lambda={}".format(i))
        weight_decay=i
        trainX, trainY, valX, valY, testX, testY = mnist_data.load_dataset()
        trainX, valX, testX = trainX/255, valX/255, testX/255
        trainY = utils.to_onehot(trainY)
        valY = utils.to_onehot(valY)
        testY = utils.to_onehot(testY)
        train_loss, train_acc,W = train(model, trainX, trainY, batch_size, epoches,lr_sche,weight_decay)
        v_loss, v_acc = model.one_epoch(valX, valY, batch_size, 1,lr_sche,weight_decay,train = False)
        v_acc_list.append(v_acc)
    x = range(len(weight_decay_list))
    fig = plt.figure(3)
    plt.xlabel("weight decay parameter lambda")
    plt.ylabel("val_acc after 200 epochs")
    plt.plot(x,v_acc_list,marker='o')
    plt.xticks(x,weight_decay_list)
    plt.savefig('./plot/sweep_weight_decay.png')
    plt.show()
    return weight_decay_list[v_acc_list.index(max(v_acc_list))]
    
    # testing procedure
    #test_loss, test_acc = model.one_epoch(testX, testY, batch_size, train = False)
    #print("Test: Loss={}, Accuracy={}".format(test_loss,test_acc))

def plot_model(hidden_layer,lr,weight_decay):
    lr_sche=[100,200,lr,lr/10]
    batch_size = 256
    epoches = 200
    np.random.seed(4)
    model=mlp_model.MLP_model(hidden_layer)
    trainX, trainY, valX, valY, testX, testY = mnist_data.load_dataset()
    trainX, valX, testX = trainX/255, valX/255, testX/255
    trainY = utils.to_onehot(trainY)
    valY = utils.to_onehot(valY)
    testY = utils.to_onehot(testY)
    train_loss, train_acc,W = train(model, trainX, trainY, batch_size, epoches,lr_sche,weight_decay)
    v_loss, v_acc = model.one_epoch(valX, valY, batch_size, 1,lr_sche,weight_decay,train = False)
    print("Val: Loss={}, Accuracy={}".format(v_loss,v_acc))
    epoch_list=np.linspace(1, epoches, epoches)
    fig = plt.figure(4)
    plt.xlabel("epoch")
    plt.ylabel("training accuracy")
    plt.plot(epoch_list,train_acc)
    plt.savefig('./plot/train_acc.png')
    plt.show()
    fig = plt.figure(5)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.plot(epoch_list,train_loss)
    plt.savefig('./plot/train_loss.png')
    plt.show()
    fig = plt.figure(6)
    plt.axis('off')
    plt.imshow(W[0], cmap='RdBu')
    plt.savefig('./plot/weight_visualization_W1.png')
    plt.show()
    fig = plt.figure(7)
    plt.axis('off')
    plt.imshow(W[1], cmap='RdBu')
    plt.savefig('./plot/weight_visualization_W2.png')
    plt.show()
    test_loss, test_acc = model.one_epoch(testX, testY, batch_size, 1,lr_sche,weight_decay,train = False)
    print("Test: Loss={}, Accuracy={}".format(test_loss,test_acc))
    return model
    
    
def main():
    #hidden_layer=parameter_sweep_hidden_layer()
    hidden_layer=100
    #lr=parameter_sweep_lr(hidden_layer)
    lr=0.5
    #weight_decay=parameter_sweep_weight_decay(hidden_layer,lr)
    weight_decay=0.0001
    m=plot_model(hidden_layer,lr,weight_decay)
    pickle.dump(m,open("./Model_save/trained_model.dat","wb"))
    #loaded_model = pickle.load(open("trained_model.dat","rb"))
    #loaded_model.predict(xtest)
    
    
if __name__ == "__main__":
    main()
