import glob
import math
import os.path as osp

import numpy as np
import torch.utils.data as data

"""# Data Loader"""


def make_data_path_list(phase="train"):
    """

    Parameters
    ----------
    phase : 'train' or 'val'

    Returns
    -------
    path_list : list
    """

    rootpath = "./data/"
    target_path = osp.join(rootpath + phase + '/**/*/')
    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list


class Dataset(data.Dataset):
    """

    Attributes
    ----------
    file_list : list
    transform : object
    phase : 'train' or 'test'
    """

    def __init__(self, file_list, phase='train', input_size=200):
        self.file_list = file_list  # file path
        # self.transform = transform  #
        self.phase = phase  # train or val
        self.size = input_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        read_list = ["rho", "u", "v", "pressure"]

        ##########
        # Input
        input_labels = []
        attention_maps = []

        ##########
        # Output
        fluid = []
        shock_labels = []

        data_path = self.file_list[index]

        # delta_x =0
        # delta_y =0

        # delta_x = random.randrange(256 - self.size)
        # delta_y = random.randrange(256 - self.size)
        delta_x = int((256 - self.size) / 2)
        delta_y = int((256 - self.size) / 2)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # distance
        ### 転置が掛かってるかもだけど、なんかおかしい。向きに注意
        ### 保存するときに転置かけてからほぞんしている
        data_path_distance = "/".join(data_path.split("\\")[:2])
        data_path_distance = data_path_distance + "/distance.csv"
        # data_path_distance = data_path_distance + "/distance_diff.csv"

        data_distance = np.loadtxt(data_path_distance, dtype="float", delimiter=",")
        data_distance = data_distance / (math.sqrt(2) * self.size)
        # print(data_distance)
        # data_distance=data_distance.T

        data_distance = data_distance[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
        # import matplotlib.pyplot as plt
        # plt.imshow(data_distance)
        # plt.colorbar()
        # plt.show()
        data_distance = np.reshape(data_distance, (1, self.size, self.size))

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # boundary
        data_path_boundary = "/".join(data_path.split("\\")[:2])
        data_path_boundary = data_path_boundary + "/modified_boundary.csv"
        boundary = np.loadtxt(data_path_boundary, dtype="float", delimiter=",")
        boundary = boundary[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
        # plt.imshow(boundary)
        # plt.colorbar()
        # plt.show()
        boundary = np.reshape(boundary, (1, self.size, self.size))

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # input_labels
        temp_label = data_path.split("\\")[2].split("_")
        input_labels.append(float(temp_label[0][4:]))
        input_labels.append(float(temp_label[1][5:]))

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # fluid
        for item in read_list:
            # path = osp.join(data_path + "/" + item + ".csv")
            path = osp.join(data_path + item + ".csv")
            data = np.loadtxt(path, delimiter=",")
            data = data.T
            data = data[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
            data = np.reshape(data, (self.size, self.size))
            # plt.imshow(data)
            # plt.colorbar()
            # plt.show()
            fluid.append(data)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # shock_labels
        shock_path = osp.join(data_path + "/" + "result" + ".csv")

        shock = np.loadtxt(shock_path, delimiter=",")
        # shock = shock.T

        shock = shock[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
        shock = np.reshape(shock, (self.size, self.size))
        # plt.imshow(shock)
        # plt.colorbar()
        # plt.show()
        shock_labels.append(shock)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        ## attention map
        # attention_path= osp.join(data_path + "/" + "attention" + ".csv")

        # attention= np.loadtxt(attention_path, delimiter=",")

        #        attention=attention[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
        #        attention= np.reshape(attention, (self.size, self.size))
        #        # plt.imshow(shock)
        #        # plt.colorbar()
        #        # plt.show()
        #        attention_maps.append(attention)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # summary

        boundary = np.array(boundary, dtype="float64")
        distance = np.array(data_distance, dtype="float64")
        input_labels = np.array(input_labels, dtype="float64")
        fluid = np.array(fluid, dtype="float64")
        shock_labels = np.array(shock_labels, dtype="float64")
        attention_maps = np.array(attention_maps, dtype="float64")

        # return boundary, distance, input_labels, fluid, shock_labels,attention_maps
        return boundary, distance, input_labels, fluid, shock_labels


class Dataset_Test(data.Dataset):
    """

    Attributes
    ----------
    file_list : list
    transform : object
    phase : 'train' or 'test'
    """

    def __init__(self, file_list, phase='train', input_size=200):
        self.file_list = file_list  # file path
        # self.transform = transform  #
        self.phase = phase  # train or val
        self.size = input_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        read_list = ["rho", "u", "v", "pressure"]

        ##########
        # Input
        input_labels = []
        attention_maps = []

        ##########
        # Output
        fluid = []
        shock_labels = []

        data_path = self.file_list[index]

        # delta_x =0
        # delta_y =0

        # delta_x = random.randrange(256 - self.size)
        # delta_y = random.randrange(256 - self.size)
        delta_x = int((256 - self.size) / 2)
        delta_y = int((256 - self.size) / 2)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # distance
        ### 転置が掛かってるかもだけど、なんかおかしい。向きに注意
        ### 保存するときに転置かけてからほぞんしている
        data_path_distance = "/".join(data_path.split("\\")[:2])
        data_path_distance = data_path_distance + "/distance.csv"
        # data_path_distance = data_path_distance + "/distance_diff.csv"

        data_distance = np.loadtxt(data_path_distance, dtype="float", delimiter=",")
        data_distance = data_distance / (math.sqrt(2) * self.size)
        # print(data_distance)
        # data_distance=data_distance.T

        data_distance = data_distance[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
        # import matplotlib.pyplot as plt
        # plt.imshow(data_distance)
        # plt.colorbar()
        # plt.show()
        data_distance = np.reshape(data_distance, (1, self.size, self.size))

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # boundary
        data_path_boundary = "/".join(data_path.split("\\")[:2])
        data_path_boundary = data_path_boundary + "/modified_boundary.csv"
        boundary = np.loadtxt(data_path_boundary, dtype="float", delimiter=",")
        boundary = boundary[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
        # plt.imshow(boundary)
        # plt.colorbar()
        # plt.show()
        boundary = np.reshape(boundary, (1, self.size, self.size))

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # input_labels
        temp_label = data_path.split("\\")[2].split("_")
        input_labels.append(float(temp_label[0][4:]))
        input_labels.append(float(temp_label[1][5:]))

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # fluid
        for item in read_list:
            # path = osp.join(data_path + "/" + item + ".csv")
            path = osp.join(data_path + item + ".csv")
            data = np.loadtxt(path, delimiter=",")
            data = data.T
            data = data[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
            data = np.reshape(data, (self.size, self.size))
            # plt.imshow(data)
            # plt.colorbar()
            # plt.show()
            fluid.append(data)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # shock_labels
        shock_path = osp.join(data_path + "/" + "result" + ".csv")

        shock = np.loadtxt(shock_path, delimiter=",")
        # shock = shock.T

        shock = shock[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
        shock = np.reshape(shock, (self.size, self.size))
        # plt.imshow(shock)
        # plt.colorbar()
        # plt.show()
        shock_labels.append(shock)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        ## attention map
        # attention_path= osp.join(data_path + "/" + "attention" + ".csv")

        # attention= np.loadtxt(attention_path, delimiter=",")

        #        attention=attention[delta_y:self.size + delta_y, delta_x:self.size + delta_x]
        #        attention= np.reshape(attention, (self.size, self.size))
        #        # plt.imshow(shock)
        #        # plt.colorbar()
        #        # plt.show()
        #        attention_maps.append(attention)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # summary

        boundary = np.array(boundary, dtype="float64")
        distance = np.array(data_distance, dtype="float64")
        input_labels = np.array(input_labels, dtype="float64")
        fluid = np.array(fluid, dtype="float64")
        shock_labels = np.array(shock_labels, dtype="float64")
        attention_maps = np.array(attention_maps, dtype="float64")

        # return boundary, distance, input_labels, fluid, shock_labels,attention_maps
        return boundary, distance, input_labels, fluid, shock_labels, data_path


def load(phase, batch_size, input_size):
    if phase == "train" or "val":
        dataset = Dataset(file_list=make_data_path_list(phase),
                          phase=phase,
                          input_size=input_size)
        dataloader = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=6
        )
    if phase == "test":
        dataset = Dataset_Test(file_list=make_data_path_list(phase),
                               phase=phase,
                               input_size=input_size)
        dataloader = data.DataLoader(
            dataset, batch_size=10, shuffle=False, num_workers=8
        )
    return dataloader

######################################################################
