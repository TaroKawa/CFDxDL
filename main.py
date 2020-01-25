# -*- coding: utf-8 -*-
"""
main code for Shcok Net
==============

**Author**: Taro Kawasaki

"""
############ TODO ####################
#
# * 初期値をどうするか？ コンスタントにすべき？
# * optimizerはどれがいいのか？
# * segmentationの指標を導入する
# *
#
# ----------------------------------------
# * decoderのoutputの活性化関数について、今のところleaky reluを使っているがそれでもよいのか？
# -> convolution層に変更した
#
########################################
from __future__ import print_function

import csv
import datetime
import os
import random

# import shape_to_shock.loader as loader
# from shape_to_shock.loss import loss_function
# from shape_to_shock.shock_net import ShockNet
import loader as loader
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from loss import loss_function
from shock_net import ShockNet
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from tqdm import tqdm


def save_cmx(y_true, y_pred, threshold):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    df_cmx.to_csv("treshold_" + str(threshold) + ".csv")


def write_csv(file, save_dict):
    save_row = {}

    with open(file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=save_dict.keys(), delimiter=",", quotechar='"')
        writer.writeheader()

        k1 = list(save_dict.keys())[0]
        length = len(save_dict[k1])

        for i in range(length):
            for k, vs in save_dict.items():
                save_row[k] = vs[i]

            writer.writerow(save_row)



"""
def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.5)
        # nn.init.kaiming_normal_(m.weight.data,mode="fan_in",nonlinearity="leaky_relu")
        #nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Conv2d:
        # nn.init.normal_(m.weight.data, 0.0, 0.5)
        nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # nn.init.normal_(m.weight.data, 0.0, 0.5)
        # nn.init.constant(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.ConvTranspose2d:
        # nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="leaky_relu")
        # nn.init.constant_(m.bias.data, 0)
        # nn.init.normal_(m.weight.data, 0.0, 0.5)
        nn.init.constant_(m.bias.data, 0)

"""

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.5)
        # nn.init.kaiming_normal_(m.weight.data,mode="fan_in",nonlinearity="leaky_relu")
        nn.init.normal_(m.bias.data, 0.0, 0.5)
    elif type(m) == nn.Conv2d:
        # nn.init.normal_(m.weight.data, 0.0, 0.5)
        nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="leaky_relu")
        #nn.init.constant_(m.bias.data, 0)
        nn.init.normal_(m.bias.data, 0.0, 0.1)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0.0, 0.1)
    elif type(m) == nn.ConvTranspose2d:
        # nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="leaky_relu")
        # nn.init.constant_(m.bias.data, 0)
        # nn.init.normal_(m.weight.data, 0.0, 0.5)
        nn.init.normal_(m.bias.data, 0.0, 0.5)

def main():
    gpu_devices = ','.join([str(id) for id in [0, 1]])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    # from shape_to_shock.parameter import parameters
    from parameter import parameters
    """# HyperParameters"""
    temp = parameters()

    parameters = temp.__dict__

    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = torch.cuda.device_count()
    print("Let's use", ngpu, "GPUs!")

    ######################################################################
    # Data
    # Create the dataloader

    train_dataloader = loader.load(phase="train",
                                   batch_size=parameters["batch_size"],
                                   input_size=parameters["input_size"])

    print("train dataloader  is done")
    val_dataloader = loader.load(phase="val",
                                 batch_size=parameters["batch_size"],
                                 input_size=parameters["input_size"])

    print("val dataloader  is done")
    ######################################################################
    # GPU or CPU ?
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # device = torch.device("cpu")

    ######################################################################
    # Define network
    print('==> Making model..')
    shock_net = ShockNet().to(device)
    num_params = sum(p.numel() for p in shock_net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        shock_net = nn.DataParallel(shock_net, device_ids=list(range(ngpu)))
        torch.backends.cudnn.benchmark = True
        print("multi-gpu")

    # Apply the weights_init fun to randomly initialize all weights

    shock_net.apply(weights_init)

    ######################################################################
    # Define Optimizer
    optimizerS = optim.Adam(shock_net.parameters(),
                            lr=parameters["lr"],
                            )
    # betas=(parameters["beta1"], 0.8))
    # betas = (parameters["beta1"], 0.999))

    # optimizerS = optim.SGD(shock_net.parameters(),
    #                        lr=3e-1)
    # optimizerS = optim.Adadelta(shock_net.parameters(),
    #                       lr=1e-1)
    ###################################################################### # Training

    epoch_list = []
    epoch_test_list = []
    S_losses = []

    epoch_test_loss_list = []

    ce_losses = []
    continuity_losses = []
    momentum_x_losses = []
    momentum_y_losses = []
    energy_losses = []

    pred_losses = []
    rh_losses = []

    train_score = []
    test_score_03 = []
    test_score_04 = []
    test_score_05 = []
    test_score_06 = []
    test_score_07 = []
    test_score_08 = []
    test_score_09 = []

    print("Starting Train Loop...")
    shock_net.train()

    dt_now = datetime.date.today()
    folder_name = ("./result/" + str(dt_now))
    save_model_path = folder_name + "/save_model/"
    save_fig_path = folder_name + "/save_figure/"
    model_name = "_shock_net.pth"

    try:
        os.mkdir(folder_name)
        os.mkdir(save_model_path)
        os.mkdir(save_fig_path)

    except:
        pass

    with open(folder_name + '/parameters.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(["name", "value"])
        w.writerow("\n")
        for t in parameters.items():
            w.writerow(list(t))
            w.writerow("\n")
            w.writerow("\n")
    epoch_check = 1
    min_score = 1e+5
    for epoch in tqdm(range(parameters["num_epoch"])):
        print("\n")
        print("epoch : ", epoch)

        epoch_test_loss = 0.0
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_continuity_loss = 0.0
        epoch_momentum_x_loss = 0.0
        epoch_momentum_y_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_rh_loss = 0.0

        epoch_list.append(epoch)

        ###########################
        # train

        # For each batch in the dataloader
        train_temp_score_list = []
        #for i, data in enumerate(train_dataloader):
        for i, data in enumerate(tqdm(train_dataloader)):
            #        if i % 3 == 0:
            #            print(i)
            # boundary, distance, input_labels, fluid, shock_labels,attention_maps = data
            # print(i)
            # start = time.time()
            boundary, distance, input_labels, fluid, shock_labels = data
            bou = boundary
            # print(boundary.shape)
            # print(distance.shape)
            # print(input_labels.shape)
            # print(fluid.shape)
            # print(shock_labels.shape)
            # print(boundary[0,:,:])
            # plt.imshow(np.reshape(boundary[0,0:,:],(200,200)))
            # plt.show()
            ###########################
            # conver cpu to gpu
            boundary = boundary.to(device)
            distance = distance.to(device)
            input_labels = input_labels.to(device)
            fluid = fluid.to(device)
            shock_labels = shock_labels.to(device)
            # attention_maps=attention_maps.to(device)

            # convert to Float
            boundary = boundary.type(torch.cuda.FloatTensor)
            distance = distance.type(torch.cuda.FloatTensor)
            input_labels = input_labels.type(torch.cuda.FloatTensor)
            fluid = fluid.type(torch.cuda.FloatTensor)
            shock_labels = shock_labels.type(torch.cuda.FloatTensor)
            #        attention_maps=attention_maps.type(torch.cuda.FloatTensor)

            ##########################
            # predict fluid and shock labels
            fluid_prediction, shock_prediction = shock_net(distance, input_labels)

            # elapse_time = time.time() - start
            # print("forward time {:.3f}".format(elapse_time))

            # start = time.time()
            # fluid_prediction = shock_net(distance, input_labels)

            # calculate loss on all batch
            # loss, each_loss = loss_function(fluid_prediction, shock_prediction, fluid, shock_labels, boundary,attention_maps)
            loss, each_loss = loss_function(fluid_prediction, shock_prediction, fluid, shock_labels, boundary, "train")
            # loss = loss_function(fluid_prediction, fluid, shock_labels, boundary)
            # elapse_time = time.time() - start
            # print("loss time {:.3f}".format(elapse_time))

            # start = time.time()
            if epoch % epoch_check == 0:
                shock_prediction = shock_prediction.detach().cpu().numpy()
                label_shock = shock_labels.detach().cpu().numpy()

                shock_prediction = shock_prediction.flatten()
                label_shock = label_shock.flatten()
                shock_prediction_train = np.where((shock_prediction >= 0.5) & (shock_prediction <= 1), 1, 0)

                #train_temp_score_list.append(jaccard_score(label_shock, shock_prediction_train))

            # add epoch loss
            epoch_loss = epoch_loss + loss.item()
            #        print(loss.item())

            epoch_ce_loss = epoch_ce_loss + each_loss[0].item()
            #     epoch_con_loss = epoch_con_loss + each_loss[1].item()

            epoch_continuity_loss = epoch_continuity_loss + each_loss[1].item()
            epoch_momentum_x_loss = epoch_momentum_x_loss + each_loss[2].item()
            epoch_momentum_y_loss = epoch_momentum_y_loss + each_loss[3].item()
            epoch_energy_loss = epoch_energy_loss + each_loss[4].item()

            epoch_pred_loss = epoch_pred_loss + each_loss[5].item()
            epoch_rh_loss = epoch_rh_loss + each_loss[6].item()

            ## Train with all-real batch
            shock_net.zero_grad()
            optimizerS.zero_grad()

            # calculate the gradients for this batch
            loss.backward()

            # update optimizer
            optimizerS.step()
            # elapse_time = time.time() - start
            # print("backward time {:.3f}".format(elapse_time))
#        if epoch % epoch_check == 0:
#            epoch_test_list.append(epoch)
#            train_score.append(100 * sum(train_temp_score_list) / len(train_temp_score_list))
        ###########################
        # post process
        epoch_loss = epoch_loss / len(train_dataloader.dataset)

        epoch_ce_loss = epoch_ce_loss / len(train_dataloader.dataset)
        epoch_continuity_loss = epoch_continuity_loss / len(train_dataloader.dataset)
        epoch_momentum_x_loss = epoch_momentum_x_loss / len(train_dataloader.dataset)
        epoch_momentum_y_loss = epoch_momentum_y_loss / len(train_dataloader.dataset)
        epoch_energy_loss = epoch_energy_loss / len(train_dataloader.dataset)

        # epoch_con_loss = epoch_con_loss / len(train_dataloader.dataset)

        epoch_pred_loss = epoch_pred_loss / len(train_dataloader.dataset)
        epoch_rh_loss = epoch_rh_loss / len(train_dataloader.dataset)

        S_losses.append(epoch_loss)
        ce_losses.append(epoch_ce_loss)
        # con_losses.append(epoch_con_loss)
        continuity_losses.append(epoch_continuity_loss)
        momentum_x_losses.append(epoch_momentum_x_loss)
        momentum_y_losses.append(epoch_momentum_y_loss)
        energy_losses.append(epoch_energy_loss)

        pred_losses.append(epoch_pred_loss)
        rh_losses.append(epoch_rh_loss)

        ###########################

        test_fluid, test_shock = shock_net(distance, input_labels)
        if epoch % epoch_check == 0:
            example = test_fluid[0][3]
            example = example.cpu()
            example2 = example.detach().clone().numpy()
            # print(example2.shape)
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            bou = bou[0].reshape([parameters["input_size"], parameters["input_size"]])
            bou = bou.numpy()
            example2 = example2 * bou
            fluid_example = example2
            plt.title("pressure")
            plt.imshow(example2, vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_fig_path + str(epoch) + "_fluid.png")
            plt.close()

            example = fluid[0][3]
            example = example.cpu()
            example2 = example.detach().clone().numpy()
            # print(example2.shape)
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("pressure")
            plt.imshow(example2, vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_fig_path + str(epoch) + "_gt_fluid.png")
            plt.close()

            example = test_shock[0]
            example = example.cpu()
            example2 = example.detach().clone().numpy()
            # print(example2.shape)
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("shock")
            plt.imshow(example2, vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_fig_path + str(epoch) + "_shock.png")
            plt.close()

            example = shock_labels[0]
            example = example.cpu()
            example2 = example.detach().clone().numpy()
            # print(example2.shape)
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("shock")
            plt.imshow(example2, vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_fig_path + str(epoch) + "_shock_gt.png")
            plt.close()

            example = fluid[0][3]
            example = example.cpu()
            example2 = example.detach().clone().numpy()
            # print(example2.shape)
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("difference")
            plt.imshow(abs(fluid_example - example2), cmap=cm.jet, vmax=0.5, vmin=0.0)
            plt.colorbar()
            plt.savefig(save_fig_path + str(epoch) + "_difference.png")
            plt.close()

            # temp_score_list = []
            temp_score_03_list = []
            temp_score_04_list = []
            temp_score_05_list = []
            temp_score_06_list = []
            temp_score_07_list = []
            temp_score_08_list = []
            temp_score_09_list = []
            for i, data in enumerate(val_dataloader):
            #for i, data in enumerate(tqdm(val_dataloader)):
                # boundary, distance, input_labels, fluid, shock_labels,_ = data
                boundary, distance, input_labels, fluid, shock_labels = data

                # conver cpu to gpu
                distance = distance.to(device)
                input_labels = input_labels.to(device)

                # convert to Float
                distance = distance.type(torch.cuda.FloatTensor)
                input_labels = input_labels.type(torch.cuda.FloatTensor)

                _, shock_prediction = shock_net(distance, input_labels)
                loss = loss_function(fluid_prediction, shock_prediction, fluid, shock_labels, boundary, "val")

                epoch_test_loss = epoch_test_loss + loss.item()

                shock_prediction = shock_prediction.detach().cpu().numpy()

                shock_prediction_03 = np.where((shock_prediction >= 0.3) & (shock_prediction <= 1), 1, 0)
                shock_prediction_04 = np.where((shock_prediction >= 0.4) & (shock_prediction <= 1), 1, 0)
                shock_prediction_05 = np.where((shock_prediction >= 0.5) & (shock_prediction <= 1), 1, 0)
                shock_prediction_06 = np.where((shock_prediction >= 0.6) & (shock_prediction <= 1), 1, 0)
                shock_prediction_07 = np.where((shock_prediction >= 0.7) & (shock_prediction <= 1), 1, 0)
                shock_prediction_08 = np.where((shock_prediction >= 0.8) & (shock_prediction <= 1), 1, 0)
                shock_prediction_09 = np.where((shock_prediction >= 0.9) & (shock_prediction <= 1), 1, 0)

                shock_prediction_03 = shock_prediction_03.flatten()
                shock_prediction_04 = shock_prediction_04.flatten()
                shock_prediction_05 = shock_prediction_05.flatten()
                shock_prediction_06 = shock_prediction_06.flatten()
                shock_prediction_07 = shock_prediction_07.flatten()
                shock_prediction_08 = shock_prediction_08.flatten()
                shock_prediction_09 = shock_prediction_09.flatten()

                shock_labels = shock_labels.flatten()

                #temp_score_03_list.append(jaccard_score(shock_labels, shock_prediction_03))
                #temp_score_04_list.append(jaccard_score(shock_labels, shock_prediction_04))
                #temp_score_05_list.append(jaccard_score(shock_labels, shock_prediction_05))
                #temp_score_06_list.append(jaccard_score(shock_labels, shock_prediction_06))
                #temp_score_07_list.append(jaccard_score(shock_labels, shock_prediction_07))
                #temp_score_08_list.append(jaccard_score(shock_labels, shock_prediction_08))
                #temp_score_09_list.append(jaccard_score(shock_labels, shock_prediction_09))

                # print(temp_score_list)
            epoch_test_loss = epoch_test_loss / len(val_dataloader.dataset)
            epoch_test_loss_list.append(epoch_test_loss)

            #test_score_03.append(100 * sum(temp_score_03_list) / len(temp_score_03_list))
            #test_score_04.append(100 * sum(temp_score_04_list) / len(temp_score_04_list))
            #test_score_05.append(100 * sum(temp_score_05_list) / len(temp_score_05_list))
            #test_score_06.append(100 * sum(temp_score_06_list) / len(temp_score_06_list))
            #test_score_07.append(100 * sum(temp_score_07_list) / len(temp_score_07_list))
            #test_score_08.append(100 * sum(temp_score_08_list) / len(temp_score_08_list))
            #test_score_09.append(100 * sum(temp_score_09_list) / len(temp_score_09_list))

        # display loss
        data = []
        data.append(epoch_list)
        data.append(ce_losses)
        data.append(epoch_test_loss_list)
        with open(folder_name + "/train_loss_vs_val_loss.csv", "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerows(data)

        plt.plot(epoch_list, epoch_test_loss_list, label="validatioin")
        plt.plot(epoch_list, ce_losses, label="train")
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(right=0.8)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Loss vs Epoch", fontsize=14)
        plt.yscale("log")
        plt.grid(which='major', color='black', linestyle='-')
        plt.grid(which='minor', color='black', linestyle='-')
        plt.savefig(folder_name + "/train_loss_vs_val_loss.png")
        plt.close()

        print("\n")
        print('epoch: [%d], loss: %.3f' % (epoch + 1, epoch_loss))

        data = []
        data.append(epoch_list)
        data.append(S_losses)
        data.append(ce_losses)
        data.append(continuity_losses)
        data.append(momentum_x_losses)
        data.append(momentum_y_losses)
        data.append(energy_losses)
        data.append(pred_losses)
        data.append(rh_losses)
        with open(folder_name + "/epoch_vs_loss.csv", "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerows(data)
        # plt.plot(epoch_list,epoch_test_loss_list, label="val")
        plt.plot(epoch_list, S_losses, label="train")
        plt.plot(epoch_list, ce_losses, label="CE")
        # plt.plot(epoch_list, con_losses, label="con loss")
        plt.plot(epoch_list, continuity_losses, label="continuity")
        plt.plot(epoch_list, momentum_x_losses, label="moment_x")
        plt.plot(epoch_list, momentum_y_losses, label="moment_y")
        plt.plot(epoch_list, energy_losses, label="energy")
        plt.plot(epoch_list, pred_losses, label="pred")
        plt.plot(epoch_list, rh_losses, label="rh")
        # loc='lower right'で、右下に凡例を表示
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        # 右側の余白を調整
        plt.subplots_adjust(right=0.8)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Each Loss vs Epoch", fontsize=14)
        plt.yscale("log")
        plt.grid(which='major', color='black', linestyle='-')
        plt.grid(which='minor', color='black', linestyle='-')
        plt.savefig(folder_name + "/loss_vs_epoch.png")
        plt.close()

        #data = []
        #data.append(epoch_test_list)
        #data.append(train_score)
        #data.append(test_score_03)
        #data.append(test_score_04)
        #data.append(test_score_05)
        #with open(folder_name + "/score.csv", "w") as file:
        #    writer = csv.writer(file, lineterminator="\n")
        #    writer.writerows(data)

        #plt.plot(epoch_test_list, train_score, label="train")
        #plt.plot(epoch_test_list, test_score_03, label="test_03")
        #plt.plot(epoch_test_list, test_score_04, label="test_04")
        #plt.plot(epoch_test_list, test_score_05, label="test_05")
        ## plt.plot(epoch_test_list, test_score_06, label="test_06")
        ## plt.plot(epoch_test_list, test_score_07, label="test_07")
        ## plt.plot(epoch_test_list, test_score_08, label="test_08")
        ## plt.plot(epoch_test_list, test_score_09, label="test_09")
        ## plt.legend(bbox_to_anchor=(0, -0.1), loc="upper left", borderaxespad=0, fontsize=8)
        ## loc='lower right'で、右下に凡例を表示
        #plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        ## 右側の余白を調整
        #plt.subplots_adjust(right=0.8)
        #plt.xlabel("Epoch", fontsize=14)
        #plt.ylabel("Score (%)", fontsize=14)
        #plt.title("Score vs Epoch", fontsize=14)
        #plt.grid(which='major', color='black', linestyle='-')
        #plt.grid(which='minor', color='black', linestyle='-')
        #plt.savefig(folder_name + "/score.png")
        #plt.close()

        if epoch > epoch_check and min_score > epoch_test_loss_list[-1]:
            # continue
            # display loss
            torch.save(shock_net, save_model_path + model_name)
            min_score = epoch_test_loss_list[-1]
            print("min_score is {}".format(min_score))
        if epoch % 100 == 0:
            torch.save(shock_net, save_model_path + str(epoch) + model_name)

        epoch_loss = 0.0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
