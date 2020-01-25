# -*- coding: utf-8 -*-
"""shock_detection

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12LzQcQBPCav26l7JJqEO-w9rwYQMk-0w
"""

import os
#path="C:/Users/User/PycharmProjects/kawasaki/shape_to_shock/data/train/naca0012_10_10/paraview"
path="C:/Users/User/Desktop/naca0012_18_10/paraview"
print(path)
directories = os.listdir(path)
#print(directories)
print(len(directories))

import os
import os.path
import shutil
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

import random

Mach_num = 0.5

thresh1 = 0.1
thresh2 = 0.01


for index in tqdm(range(0, len(directories))):
    directory = directories[index]
    try:
        U = np.loadtxt(path+"/" + directory + "/u.csv", delimiter=",")
        U = U.transpose()

        V = np.loadtxt(path+"/" + directory + "/v.csv", delimiter=",")
        V = V.transpose()

        P = np.loadtxt(path+"/" + directory + "/pressure.csv", delimiter=",")
        P = P.transpose()

        Mach = np.loadtxt(path+"/" + directory + "/mach.csv", delimiter=",")
        Mach = Mach.transpose()
        print(U.shape)
        print(V.shape)
        print(P.shape)
        print(Mach.shape)

        if index == 0:
            mask = np.where(Mach > 0.1 + 1e-4, 1, 0)
            mask_modified = np.ones((256, 256))
            for i in range(0, 255):
                for j in range(0, 255):
                    if mask[i, j] == 0:
                        for x in range(-1, 2):
                            for y in range(-1, 2):
                                mask_modified[i + x, j + y] = 0.0

        delta_P = np.gradient(P)
        delta_P_norm = np.sqrt(delta_P[0] ** 2 + delta_P[1] ** 2)
        max_delta_P = np.max(delta_P_norm)
        # print(max_delta_P)

        sound_speed = np.where(mask_modified > 0.0 + 1e-2, np.sqrt(U ** 2 + V ** 2) / Mach, 0)
        Ma_n = np.where(mask_modified > 0.0 + 1e-2, (U * delta_P[1] + V * delta_P[0]) / (sound_speed * delta_P_norm), 0)
        Ma_n = np.where((abs(1 - Ma_n) <= thresh1) & (abs(delta_P_norm) >= thresh2 * max_delta_P), 1, 0)
        np.savetxt(path+"/" + directory + "/result.csv", Ma_n, delimiter=",")
        print("\n")
    except:
        shutil.rmtree(path+"/"+directory)
        print(directory)
        print("")

# if random.randrange(10)==0:
#    plt.imshow(Mach)
#    plt.colorbar()
#    plt.show()
#    plt.close()
#
#    plt.imshow(P)
#    plt.colorbar()
#    plt.show()
#    plt.close()
#
#    plt.imshow(Ma_n)
#    plt.colorbar()
#    plt.show()
#    plt.close()
#    flag=False

cd / content / drive / My
Drive / shape_to_shock / data / naca0012_14_10 / paraview

folder_name = "Mach1.2_Angle-2.1"
result = np.loadtxt("./" + folder_name + "/result.csv", delimiter=",")
plt.imshow(result)
plt.colorbar()
plt.show()

Mach_num = 0.6
# 1.1
thresh1 = 0.4
thresh2 = 0.004
for i in range(0, 1, 1):
    Mach_num + i * 0.1
    Angle = -4.0
    while Angle <= -3.5:
        Mach_num = round(Mach_num, 2)
        Angle = round(Angle, 2)
        print("Mach {}".format(Mach_num))
        print("Angle {}".format(Angle))
        folder_name = "./Mach" + str(Mach_num) + "_Angle" + str(Angle)

        if os.path.exists(folder_name):
            print("Start!")

            U = np.loadtxt("./" + folder_name + "/u.csv", delimiter=",")
            U = U.T

            V = np.loadtxt("./" + folder_name + "/v.csv", delimiter=",")
            V = V.transpose()

            P = np.loadtxt("./" + folder_name + "/pressure.csv", delimiter=",")
            P = P.transpose()

            Mach = np.loadtxt("./" + folder_name + "/mach.csv", delimiter=",")
            Mach = Mach.transpose()

            mask_shape = np.ones((256, 256))

            for i in range(0, 256):
                for j in range(0, 256):
                    if P[i, j] == 0:
                        mask_shape[i, j] = 0

            mask = np.array([[2, 3, 3, 2, 2],
                             [3, 4, 4, 4, 3],
                             [3, 4, 5, 4, 3],
                             [3, 4, 4, 4, 3],
                             [2, 3, 3, 3, 2]])

            for i in range(0, 6):
                for i in range(0, 256):
                    for j in range(0, 256):
                        if i > 2 and i < 254 and j > 2 and j < 254:
                            P[i, j] = np.sum(mask * P[i - 2:i + 3, j - 2:j + 3] / np.sum(mask))

            P = P * mask_shape
            deltaP = np.zeros((256, 256, 2), float)
            deltaP[:, :, 0] = np.gradient(P)[1]
            deltaP[:, :, 1] = np.gradient(P)[0]
            # for i in range(7, 249):
            #    for j in range(7, 249):
            #        deltaP[i, j, 1] = np.sum(P[i:i + 3, j:j + 7] * mask_u)
            #        deltaP[i, j, 0] = np.sum(P[i:i + 7, j:j + 3] * mask_v)
            #        #print(deltaP[i,j,0])
            DeltaP = deltaP
            deltaP = normalize(deltaP)
            # np.where(abs(deltaP)<1e-6,1e-6,deltaP)

            velocity = np.ones((256, 256, 2), float)
            velocity[:, :, 0] = U
            velocity[:, :, 1] = V
            velocity = normalize(velocity)

            X = np.arange(0, 256, 1)
            Y = np.arange(0, 256, 1)

            # X1, X2 = np.meshgrid(X, Y)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            ## surf = ax.plot_surface(X1, X2,P, cmap='bwr', linewidth=0)
            # surf = ax.plot_surface(X1, X2, deltaP[:,:,1], cmap='bwr', linewidth=0)
            # fig.colorbar(surf)
            # ax.set_title("Surface Plot")
            # fig.show()

            # fig, ax = plt.subplots()
            # q = ax.quiver(X, Y, deltaP[:, :, 1], deltaP[:, :, 0],
            #               color="red", angles="xy",
            #               scale_units="xy")

            # plt.show()
            # # plt.imshow(U,cmap="jet")
            # plt.imshow(P, cmap="jet")
            # plt.colorbar()
            # plt.xlabel("X")
            # plt.ylabel("Y")
            # plt.show()

            shock_detection = np.zeros((256, 256), float)
            DeltaP_max = (np.max(np.linalg.norm(DeltaP, ord=1, axis=-1)))

            # 0.9
            # thresh1 = 0.2
            # thresh2 = 0.01
            # count=2

            # 1.0
            # thresh1 = 0.2
            # thresh2 = 0.006
            # count=2

            # 1.7 1.8,
            # thresh1 = 0.2
            # thresh2 = 0.0001
            # 1.7 1.8,1.9,2.0,1.6,1.5
            # thresh1 = 0.35
            # thresh2 = 0.002
            # count=4

            # # 1.2
            # thresh1 = 0.15
            # thresh2 = 0.0009

            for i in range(2, 254):
                for j in range(2, 254):
                    mach_n = Mach[i, j] * np.dot(velocity[i, j], deltaP[i, j])
                    # shock_detection[i,j]=mach_n
                    if abs(mach_n - 1.0) < thresh1:
                        # if mach_n < 1.0 + thresh1 and mach_n > 1.0 - thresh1:
                        if np.linalg.norm(DeltaP[i, j]) > DeltaP_max * thresh2:
                            shock_detection[i, j] = 1.0
                            # print("Thresh 2")

            for i in range(4, 252):
                for j in range(4, 252):
                    count = 0
                    if shock_detection[i, j] == 1.0:
                        shock_detection[i, j] = 0.0
                        for l in range(0, 6):
                            for m in range(0, 6):
                                if shock_detection[i + l - 2, j + m - 2] == 1.0:
                                    count = count + 1
                        if count >= 5:
                            shock_detection[i, j] = 1.0
                            # P[i, j] = -1
                    else:
                        shock_detection[i, j] = 0.0

            for i in range(0, 256):
                for j in range(0, 256):
                    if shock_detection[i, j] == 1:
                        P[i, j] = -1

                # plt.show()

                # plt.imshow(shock_detection, cmap="jet", vmin=0.97, vmax=1.03)
                # plt.imshow(P, cmap="jet")
            plt.imshow(shock_detection, cmap="jet")
            plt.colorbar()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            plt.close()

            plt.imshow(P, cmap="jet")
            plt.colorbar()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            # plt.savefig(folder_name + "/result.jpg")
            # np.savetxt( folder_name+ "/result.csv", shock_detection, delimiter=",")
            plt.imshow(Mach, cmap="jet")
            plt.colorbar()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            # plt.savefig(folder_name+ "/result.jpg")
            plt.close()
            np.savetxt(folder_name + "/result.csv", shock_detection)
        Angle = Angle + 0.1