class parameters():
    def __init__(self):
        # fluid parameters
        self.gamma = 1.4

        # NN parameters
        self.batch_size = 10
        # self.batch_size =10
        self.input_size = 256
        self.num_epoch = 300
        self.lr = 0.005
        self.beta1 = 0.9

        self.input_channel = 1
        self.conv_channel1 = 4 * 2 * 2
        self.conv_channel2 = 8 * 2 * 2
        self.conv_channel3 = 16 * 2 * 2
        self.conv_channel4 = 32 * 2 * 2
        self.conv_channel5 = 64 * 2 * 2

        self.conv_kernel_1 = 2
        self.conv_kernel_2 = 2
        self.conv_kernel_3 = 2
        self.conv_kernel_4 = 2
        self.conv_kernel_5 = 2

        self.conv_stride_1 = 2
        self.conv_stride_2 = 2
        self.conv_stride_3 = 2
        self.conv_stride_4 = 2
        self.conv_stride_5 = 2

        self.linear_size = int((((((((((
                                               self.input_size - self.conv_kernel_1) / self.conv_stride_1 + 1) - self.conv_kernel_2) / self.conv_stride_2 + 1) - self.conv_kernel_3) / self.conv_stride_3 + 1) - self.conv_kernel_4) / self.conv_stride_4 + 1) - self.conv_kernel_5) / self.conv_stride_5 + 1)
        self.n_hidden = 10

        self.unet_seed_channel = 16

        self.output_channel = 4

        # loss
        self.lambda_ce = 3.0
        # self.lambda_prediction = 1.5
        self.lambda_prediction = 1.0
        # self.lambda_conservation = 0.0
        # self.lambda_conservation = 1.0
        self.lambda_continuity = 1.0
        self.lambda_momentum_x = 1.0
        self.lambda_momentum_y = 1.0
        self.lambda_energy = 1.0
        self.lambda_rh = 1.0e-6

        self.balance_term = 0.1
        # self.lambda_rh = 0.1
