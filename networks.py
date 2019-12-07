import torch.nn as nn

class discriminator(nn.Module):
    
    def __init__(self):
        super(discriminator,self).__init__()
        # self.net     = nn.Sequential(nn.Linear(784,512), nn.Dropout(), nn.ReLU(True), 
        #                              nn.Linear(512,256), nn.Dropout(), nn.ReLU(True),
        #                              nn.Linear(256,1), nn.Sigmoid())
        ################################################################################
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.net  = nn.Sequential(*self.conv_layer(1,128,3,2), *self.conv_layer(128,128,3,1),
                                *self.conv_layer(128,256,3,2), *self.conv_layer(256,256,3,1),
                                *self.conv_layer(256,512,3,2))  
        self.linear = nn.Sequential(nn.Linear(512*8*8, 1))
        
        _initialize_weights(self)

    def conv_layer(self, in_channels, out_channels, kernelsize, stride):
            net = [nn.Conv2d(in_channels, out_channels, kernel_size=kernelsize, stride=stride,
                                            padding=int((kernelsize-1)/2), bias=False), 
                #    nn.InstanceNorm2d(out_channels, affine=True), nn.LeakyReLU(0.2, False)]
                   nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, False)]
            return net

    def forward(self, input):
        out = self.net(input)
        out = out.view(out.size(0), -1)
        output = self.linear(out)
        return output

class generator(nn.Module):

    def __init__(self):
        super(generator,self).__init__()
        
        ##### For MNIST #####
        # self.net     = nn.Sequential(nn.Linear(100,256), nn.BatchNorm1d(256), nn.ReLU(True), 
        #                              nn.Linear(256,512), nn.BatchNorm1d(512), nn.ReLU(True),
        #                              nn.Linear(512,784), nn.Sigmoid()) 

        ##### For generate noise #####        
        self.linear = nn.Sequential(nn.Linear(100, 8*8*512), nn.BatchNorm1d(8*8*512), nn.ReLU(True))
        # self.net    = nn.Sequential(*self.deconv_layer(512,256,2,2), *self.deconv_layer(256,256,3,1,padding=1), 
        #                             *self.deconv_layer(256,128,2,2), *self.deconv_layer(128,128,3,1,padding=1),
        #                             nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, bias=True),
        #                             nn.Sigmoid()) 
        self.net    = nn.Sequential(*self.deconv_layer(512,256,2,2), *self.deconv_layer(256,256,3,1,padding=1), 
                                    *self.deconv_layer(256,256,3,1,padding=1), *self.deconv_layer(256,256,3,1,padding=1), 
                                    *self.deconv_layer(256,128,2,2), *self.deconv_layer(128,128,3,1,padding=1),
                                    *self.deconv_layer(128,128,3,1,padding=1), *self.deconv_layer(128,128,3,1,padding=1),
                                    nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, bias=True),
                                    nn.Sigmoid()) 

        _initialize_weights(self)

    def deconv_layer(self, in_channels, out_channels, kernelsize, stride, padding=0):
            net = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernelsize, stride=stride, padding=padding, bias=False), 
                   nn.BatchNorm2d(out_channels), nn.ReLU(True)]
            return net
            
    def forward(self, input):
        out = self.linear(input)
        out = out.view(out.size(0),512,8,8)
        output = self.net(out)
        return output


def _initialize_weights(network):
    for m in network.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            #nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)