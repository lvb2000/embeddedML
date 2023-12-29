import torch
import torch.nn as nn
from hparam import hparam as hp

architecture_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: [ Tuples: (kernel_size, num_filters, stride, padding), repetitions]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # TODO: How is a batchnorm implemented on the MAXIM board? Is it in general possible to use predefined pytorch functions?
        self.batchnorm = nn.BatchNorm2d(out_channels)
        # TODO: same with leaky relu?
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers()
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1)) # Flatten with start=dim=1 to keeps batch dimension (e.g. pictures stay separated but everything else is flattened)

    def _create_conv_layers(self):
        layers = []

        for x in self.architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        self.in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                self.in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            self.in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    self.in_channels = conv2[1]

        return nn.Sequential(*layers) # *layers unpacks the list of layers into the sequential function

    def _create_fcs(self):
        S, B, C = hp['S'], hp['B'], hp['C']
        return nn.Sequential(
            nn.Flatten(),
            # the linear layer is fully connected, which means it has many nodes to computed which is why i reduce the number of nodes with the linear layer from 4096 to 496
            # This can be further tuned maybe if the model is too large for the MAXIM board
            # 1024 is the output dimension of the last CNN layer * this is multiplied by S*S where S is the grid size the image has been split into
            # This is done because the CNN looks at every patch of the grid individually
            nn.Linear(1024 * S * S, 496),
            # TODO: set later to 0.5
            nn.Dropout(hp['dropout']),
            nn.LeakyReLU(0.1),
            # S*S*(C+B*5): for every patch of the grid there are C different classes and B different boxes with 5 parameters
            # The 5 parameters are: x, y, w, h, confidence
            nn.Linear(496, S * S * (C + B * 5))
        )

def test():
    model = YOLOv1()
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

# test()