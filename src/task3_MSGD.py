import argparse

import numpy as np

import matplotlib.pyplot as plt
from common.get_data import get_feature, get_train, shuffle_split
from common.metric import avg_acc
from common.model import GNN
from common.optimizer import MomentumSGD


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--D', '-d', type=int, default=8, help='Dimension of feature vector')
    parser.add_argument('--T', '-t', type=int, default=2, help='Max step of aggregation')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of training dataset')
    parser.add_argument('--batch', '-b', type=int, default=256, help='batch size')

    args = parser.parse_args()

    train_H, train_y, train_node_size = get_train()

    seed = 1996

    train_H, train_y, val_H, val_y, train_node_size, val_node_size = shuffle_split(train_H, train_y, train_node_size, split_size=0.7, seed=seed)

    # feature dimension
    D = args.D

    # step size
    T = args.T

    # learning rate and momentum
    alpha = 0.0015
    momentum = 0.9

    # epoch size
    max_epoch = args.epoch

    # batch size
    batch_size = args.batch

    # get step per epoch
    train_size = len(train_H)
    iter_per_epoch = train_size//batch_size if (train_size%batch_size) == 0 else (train_size//batch_size)+1
    
    # make feature vector(train)
    train_x = get_feature(D, train_H, train_node_size)

    # make feature vector(validation)
    val_x = get_feature(D, val_H, val_node_size)

    model = GNN(D, T)
    optimizer = MomentumSGD(alpha=alpha, momentum=momentum)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
                
    for epoch in range(max_epoch):
        np.random.seed(int(epoch*1234))
        shuffle_idx = np.random.permutation(train_H.shape[0])
        train_H = train_H[shuffle_idx]
        train_x = train_x[shuffle_idx]
        train_y = train_y[shuffle_idx]
        for num in range(iter_per_epoch):
            if train_size > (num+1)*batch_size:
                batch_H = train_H[num*batch_size:(num+1)*batch_size]
                batch_x = train_x[num*batch_size:(num+1)*batch_size]
                batch_y = train_y[num*batch_size:(num+1)*batch_size]
            else:
                batch_H = train_H[num*(batch_size):]
                batch_x = train_x[num*(batch_size):]
                batch_y = train_y[num*(batch_size):]
        
            # get batch gradient and update parameters
            batch_grads = None
            for idx in range(len(batch_H)):
                grad = model.get_gradient(batch_x[idx], batch_H[idx], batch_y[idx])
                if batch_grads == None:
                    batch_grads = {}
                    for key, val in grad.items():
                        batch_grads[key] = np.zeros_like(val)
                for key in grad.keys():
                    batch_grads[key] += (grad[key] / len(batch_H))
            optimizer.update(model.params, batch_grads)
        
        # train loss and average accuracy
        loss = 0
        train_pred = np.zeros((len(train_y), 1))
        for idx in range(len(train_H)):
            loss += model.loss(train_x[idx], train_H[idx], train_y[idx]) / len(train_H)
            predict = 0 if model.predict(train_x[idx], train_H[idx]) < 1/2 else 1
            train_pred[idx] = predict
        train_score = avg_acc(train_y, train_pred)
        
        # validation loss and average accuracy
        val_loss = 0
        val_pred = np.zeros((len(val_y), 1))
        for idx in range(len(val_H)):
            val_loss += model.loss(val_x[idx], val_H[idx], val_y[idx]) / len(val_H)
            predict = 0 if model.predict(val_x[idx], val_H[idx]) < 1/2 else 1
            val_pred[idx] = predict
        val_score = avg_acc(val_y, val_pred)

        print('epoch:{} loss:{:.5f} val_loss:{:.5f} avg_acc:{:.5f} val_avg_acc:{:.5f}'.format(epoch+1, loss, val_loss, train_score, val_score))
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_score)
        val_acc_list.append(val_score)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
    x = np.arange(len(train_loss_list))
    ax1.plot(x, train_loss_list, label='train')
    x = np.arange(len(val_loss_list))
    ax1.plot(x, val_loss_list, label='validation')
    ax1.legend()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    x = np.arange(len(train_acc_list))
    ax2.plot(x, train_acc_list, label='train')
    x = np.arange(len(val_acc_list))
    ax2.plot(x, val_acc_list, label='validation')
    ax2.legend()
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('average accuracy')

    fig.savefig('src/graph/GNN_Momentum.png')
    plt.close()

if __name__ == "__main__":
    main()
