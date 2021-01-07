import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################

# YOUR CODE HERE
# DEFINE THE LOSS FUNCTIONS
criterion = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()

###################################################

def train(args, model, optimizer, dataloaders):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0

    # training
    print('Network training starts ...')
    for epoch in range(args.epochs):
        model.train()

        loss_train_epoch = 0.0
        correct_train_epoch = 0
        num_samples = 0
        batch_time = time.time(); iter_time = time.time()

        for i, data in enumerate(trainloader):

            img_L = data['img_L']; img_S = data['img_S']; labels = data['label']
            img_L, img_S, labels = img_L.to(device), img_S.to(device), labels.to(device)

            if args.mode == 'CAM':
                ###################################################
                output, *_ = model(img_L)

                loss = criterion(output, labels)
                loss_train_batch = loss.item()
                loss_train_epoch += loss_train_batch

                labels = labels.detach().cpu().numpy()
                labels_pred = torch.argmax(torch.softmax(output, dim=1), dim=1).detach().cpu().numpy()

                correct_train_batch = np.sum(labels_pred == labels)
                correct_train_epoch += correct_train_batch

                current_batch_size = img_L.shape[0]
                num_samples += current_batch_size
                accuracy_batch = (correct_train_batch / current_batch_size) * 100

                # YOUR CODE HERE
                # INPUT TO CAM MODEL AND COMPUTE THE LOSS

                ###################################################

            elif args.mode == 'SEG':

                ###################################################
                output_L, output_S, f_L, f_S, w_L, w_S = model(img_L, img_S)

                loss_L_cls = criterion(output_L, labels)
                loss_S_cls = criterion(output_S, labels)

                # downscale f_L to be the same size as f_S
                # batch_size x 128 x 14 x 14
                f_L_downsampled = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)(f_L)

                # apply softmax
                # batch_size x 128
                w_L_normalized = torch.softmax(w_L, dim=1)
                w_S_normalized = torch.softmax(w_S, dim=1)

                current_batch_size = img_L.shape[0]
                num_filters = w_L_normalized.shape[-1]

                # extract the shape of the feature maps
                _, _, feature_map_height, feature_map_width = f_L_downsampled.shape
                CAM_L = torch.zeros((current_batch_size, feature_map_height, feature_map_width), dtype=torch.float32).to(device)
                CAM_S = torch.zeros((current_batch_size, feature_map_height, feature_map_width), dtype=torch.float32).to(device)

                # multiply the feature maps with the normalized weights
                for image_index in range(current_batch_size):
                    CAM_L[image_index] = torch.sum((w_L_normalized[image_index] * f_L_downsampled[image_index].reshape(num_filters, -1).T).T.reshape(num_filters, feature_map_height, feature_map_width), dim=0)
                    CAM_S[image_index] = torch.sum((w_S_normalized[image_index] * f_S[image_index].reshape(num_filters, -1).T).T.reshape(num_filters, feature_map_height, feature_map_width), dim=0)


                loss_seg = criterion_mse(CAM_L, CAM_S)

                # aggregate the losses
                loss = ((loss_L_cls + loss_S_cls) / 2) + loss_seg

                # calculate accuracy
                labels = labels.detach().cpu().numpy()
                labels_pred = torch.argmax(torch.softmax(output_L, dim=1), dim=1).detach().cpu().numpy()

                correct_train_batch = np.sum(labels_pred == labels)
                correct_train_epoch += correct_train_batch

                current_batch_size = img_L.shape[0]
                num_samples += current_batch_size
                accuracy_batch = (correct_train_batch / current_batch_size) * 100

                # YOUR CODE HERE
                # INPUT TO SEG MODEL, DEFINE THE SCALE EQUIVARIANT LOSS
                # AND COMPUTE THE TOTAL LOSS

                ###################################################

            else:
                output_L, output_S, f_L, f_S, w_L, w_S = model(img_L, img_S)

                loss_L_cls = criterion(output_L, labels)
                loss_S_cls = criterion(output_S, labels)

                # apply softmax
                # batch_size x 128
                w_L_normalized = torch.softmax(w_L, dim=1)
                w_S_normalized = torch.softmax(w_S, dim=1)

                current_batch_size = img_L.shape[0]
                num_filters = w_L_normalized.shape[-1]

                # extract the shape of the feature maps
                _, _, feature_map_height, feature_map_width = f_L.shape
                CAM_L = torch.zeros((current_batch_size, feature_map_height, feature_map_width), dtype=torch.float32).to(device)
                CAM_S = torch.zeros((current_batch_size, feature_map_height, feature_map_width), dtype=torch.float32).to(device)

                # multiply the feature maps with the normalized weights
                for image_index in range(current_batch_size):
                    CAM_L[image_index] = torch.sum((w_L_normalized[image_index] * f_L[image_index].reshape(num_filters, -1).T).T.reshape(num_filters, feature_map_height, feature_map_width), dim=0)
                    CAM_S[image_index] = torch.sum((w_S_normalized[image_index] * f_S[image_index].reshape(num_filters, -1).T).T.reshape(num_filters, feature_map_height, feature_map_width), dim=0)

                loss_seg = criterion_mse(CAM_L, CAM_S)

                # aggregate the losses
                loss = ((loss_L_cls + loss_S_cls) / 2) + loss_seg

                # calculate accuracy
                labels = labels.detach().cpu().numpy()
                labels_pred = torch.argmax(torch.softmax(output_L, dim=1), dim=1).detach().cpu().numpy()

                correct_train_batch = np.sum(labels_pred == labels)
                correct_train_epoch += correct_train_batch

                current_batch_size = img_L.shape[0]
                num_samples += current_batch_size
                accuracy_batch = (correct_train_batch / current_batch_size) * 100

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logging
            if i % 100 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                                                                           time.time()-iter_time, loss.item()))
                print(f'Accuracy Batch: {accuracy_batch:.3f} %') ########################
                iter_time = time.time()
            # break

        # calculate train loss of epoch
        num_batches = len(trainloader)
        loss_train_epoch /= num_batches

        # calculate train accuracy of epoch
        accuracy_epoch = (correct_train_epoch / num_samples) * 100

        batch_time = time.time() - batch_time
        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))
        print(f'Accuracy Epoch: {accuracy_epoch:.3f} %') #############################
        # evaluation
        if epoch % 1 == 0:
            testing_accuracy = evaluate(args, model, testloader)
            print('testing accuracy: {:.3f}'.format(testing_accuracy))
            print('-------------------------------------------------')

            if testing_accuracy > best_testing_accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, './checkpoints/{}_checkpoint.pth'.format(args.exp_id))
                best_testing_accuracy = testing_accuracy
                print('new best model saved at epoch: {}'.format(epoch))
                print('-------------------------------------------------')
    print('-------------------------------------------------')
    print('best testing accuracy achieved: {:.3f}'.format(best_testing_accuracy))


def evaluate(args, model, testloader):
    total_count = torch.tensor([0.0]); correct_count = torch.tensor([0.0])

    for i, data in enumerate(testloader):
        img_L = data['img_L']; labels = data['label']
        img_L, labels = img_L.to(device), labels.to(device)
        total_count += labels.size(0)

        with torch.no_grad():
            cls_L_scores, *_ = model(img_L, img_S=None)
            predict_L = torch.argmax(cls_L_scores, dim=1)
            correct_count += (predict_L.cpu().detach() == labels.cpu().detach()).sum()
    testing_accuracy = correct_count / total_count

    return testing_accuracy.item()


def resume(args, model, optimizer):
    checkpoint_path = './checkpoints/{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    checkpoint_saved = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint_saved['model_state_dict'])
    optimizer.load_state_dict(checkpoint_saved['optimizer_state_dict'])

    print('Resume completed for the model\n')

    return model, optimizer
