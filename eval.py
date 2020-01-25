# -*- coding: utf-8 -*-
"""
main code for Shcok Net
==============

**Author**: Taro Kawasaki

"""
############ TODO ####################
#
# * decoderのoutputの活性化関数について、今のところleaky reluを使っているがそれでもよいのか？
# * 初期値をどうするか？
#
# ----------------------------------------
# * parameters だけのclassを作成する
# * FVSの値がおかしい問題を何とかする
########################################
from __future__ import print_function

import random
import os

import matplotlib.pyplot as plt
import torch.utils.data
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import matplotlib.cm as cm

import shape_to_shock.loader as loader
import numpy as np
from shape_to_shock.parameter import parameters
import stat


def execute(target_directory):
    """処理を実行します。"""
    for root, dirs, files in os.walk(target_directory):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            if not os.access(full_path, os.W_OK):
                os.chmod(full_path, stat.S_IWRITE)
def main():
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
    test_dataloader = loader.load(phase="test",
                                   batch_size=parameters["batch_size"],
                                   input_size=parameters["input_size"])

    ######################################################################
    # GPU or CPU ?
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    ######################################################################
    # Define network
    #DAY="sub_2020-01-08"
    DAY="2020-01-06"
    save_path = "./result/"+DAY+"/save_model/"
    execute(save_path)
    model_name = "100_shock_net.pth"
    #model_name = "_shock_net.pth"

    shock_net = torch.load(save_path + model_name)
    #shock_net = torch.load("./"+model_name)
    shock_net = shock_net.to(device)

    # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #    print("multi-gpu")
    #    shock_net = nn.DataParallel(shock_net, device_ids=list(range(ngpu)))
    #    torch.backends.cudnn.benchmark = True

    ######################################################################

    epoch_list = []
    S_losses = []

    ce_losses = []
    con_losses = []
    pred_losses = []
    rh_losses = []

    print("Starting evaluation...")

    save_path = "./result/"+DAY+"/evaluation/"

    try:
        os.mkdir(save_path)
    except:
        pass

    shock_net.eval()
    for epoch in tqdm(range(1)):
        print("\n")
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_con_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_rh_loss = 0.0

        epoch_list.append(epoch)
        test_score = []

        ###########################
        # train

        # For each batch in the dataloader
        for i, data in enumerate(tqdm(test_dataloader)):
            #        if i % 3 == 0:
            #            print(i)
            boundary, distance, input_labels, fluid, shock_labels,folder_path = data
            shape_path=folder_path[0][12:].split("\\")[0]
            condition_path=folder_path[0][12:].split("\\")[1]
            try:
                os.mkdir("./result/"+DAY+"/evaluation/"+shape_path)
            except:
                pass


            try:
                #print("./result/"+DAY+"/evaluation/"+shape_path+"/"+condition_path)
                os.mkdir("./result/"+DAY+"/evaluation/"+shape_path+"/"+condition_path)
            except:
                pass
            save_path = "./result/" + DAY + "/evaluation/" + shape_path + "/" + condition_path + "/"
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

            # convert to Float
            boundary = boundary.type(torch.cuda.FloatTensor)
            distance = distance.type(torch.cuda.FloatTensor)
            input_labels = input_labels.type(torch.cuda.FloatTensor)
            fluid = fluid.type(torch.cuda.FloatTensor)
            shock_labels = shock_labels.type(torch.cuda.FloatTensor)

            ##########################
            # predict fluid and shock labels
            fluid_prediction, shock_prediction = shock_net(distance, input_labels)

            # fluid_prediction = shock_net(distance, input_labels)

            test_shock = shock_prediction.detach().cpu().numpy()
            label_shock = shock_labels.detach().cpu().numpy()

            test_shock = test_shock.flatten()
            test_shock=np.where(test_shock>0.1,np.ones(test_shock.shape),np.zeros(test_shock.shape))
            label_shock = label_shock.flatten()
            test_score.append(jaccard_score(label_shock, test_shock.round()))
            #print(jaccard_score(label_shock, test_shock.round()))
            #print("score:",test_score)

            #print(input_labels[0])
            example = fluid_prediction[0][3]
            example2 = example.detach().cpu().numpy()
            # print(example2.shape)
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            bou = bou[0].reshape([parameters["input_size"], parameters["input_size"]])
            bou = bou.numpy()
            example2 = example2 * bou
            fluid_example = example2
            plt.title("pressure")
            plt.imshow(example2, vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_path+ str(input_labels[0].detach().cpu()) + "_fluid.png")
            plt.close()

            ###########################
            # post process

            ###########################

            example = fluid[0][3]
            example2 = example.detach().cpu().numpy()
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])

            example_pre = fluid_prediction[0][3]
            example_pre2 = example_pre.detach().cpu().numpy()
            example_pre2 = example_pre2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("pressure")
            plt.imshow(abs(example2-example_pre2), vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_path+ str(input_labels[0].detach().cpu()) + "_fluid_diff.png")
            plt.close()

            example = fluid[0][3]
            example2 = example.detach().cpu().numpy()
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("pressure")
            plt.imshow(example2, vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_path+ str(input_labels[0].detach().cpu()) + "_fluid_gt.png")
            plt.close()

            test_shock = shock_prediction.detach().cpu().numpy()
           # label_shock = shock_labels.detach().cpu().numpy()

            example =shock_prediction[0]
            example2 = example.detach().cpu().numpy()
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("shock")
            plt.imshow(example2, vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_path+ str(input_labels[0].detach().cpu()) + "_shock.png")
            plt.close()

            example= shock_labels[0]
            example2 = example.detach().cpu().numpy()
            #print(example2.shape)
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("shock")
            plt.imshow(example2, vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_path+ str(input_labels[0].detach().cpu()) + "_shock_gt.png")
            plt.close()

            example = shock_labels[0]
            example2 = example.detach().cpu().numpy()
            # print(example2.shape)
            example2 = example2.reshape([parameters["input_size"], parameters["input_size"]])
            example_pre =shock_prediction[0]

            example_pre2 = example_pre.detach().cpu().numpy()
            example_pre2 = example_pre2.reshape([parameters["input_size"], parameters["input_size"]])
            plt.title("shock")
            plt.imshow(abs(example2-example_pre2), vmin=0.0, vmax=1.0, cmap=cm.jet)
            plt.colorbar()
            plt.savefig(save_path + str(input_labels[0].detach().cpu()) + "_shock_diff.png")
            plt.close()
        #print("score:",sum(test_score)/len(test_score))

if __name__ == "__main__":
    main()

