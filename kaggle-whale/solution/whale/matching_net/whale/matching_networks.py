import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def convLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob),
    )
    return cnn_seq


def subblock(in_channels, filters, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, filters, 1, 1),
        nn.BatchNorm2d(filters),
        nn.Conv2d(filters, filters,3,1,1),
        nn.BatchNorm2d(filters),
        nn.Conv2d(filters,in_channels,1,1)
    )
    return cnn_seq


class Classifier(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28):
        super(Classifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:F
        """
        self.Block1 = nn.Sequential(
            nn.Conv2d(1,64,9,2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.Block1_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,1),
            nn.ReLU(True)
        )
        self.Block1_2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64, 3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,128, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(128)
        )

        self.Block2 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,1),
            nn.ReLU(True),
            nn.BatchNorm2d(256)
        )

        self.Block3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 384, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(384)
        )

        self.Block4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 512, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512)
        )

        self.subblock1 = subblock(128, 64)
        self.subblock2 = subblock(256, 64)
        self.subblock3 = subblock(384, 96)
        self.subblock4 = subblock(512, 128)

        self.outSize = 512

        
    
    def forward(self, x):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.Block1(x)
        x = self.Block1_1(x)
        x = self.Block1_2(x)

        for _ in range(4):
            #print(x.shape)
            y = x
            x = self.subblock1(x)
            x += y

        x = self.Block2(x)
        for _ in range(4):
            y = x
            x = self.subblock2(x)
            x += y

        x = self.Block3(x)
        for _ in range(4):
            y = x
            x = self.subblock3(x)
            x += y

        x = self.Block4(x)
        for _ in range(4):
            y = x
            x = self.subblock4(x)
            x += y

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size()[0], -1)

        return x

class Classifier_o(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28):
        super(Classifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.model = [convLayer(num_channels, layer_size, keep_prob)]
        for i in range(3):
            self.model.append(convLayer(layer_size, layer_size, keep_prob))   
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

        
    
    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        for F in self.model:
            x = F(x)
        x = x.view(x.size()[0], -1)
        return x


class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Products pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
        :param support_set_y:[batch_size,sequence_length,classes_num]
        :return: Softmax pdf shape[batch_size,classes_num]
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds


class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        forward implement
        :param support_set:the embeddings of the support set images.shape[classes_per_set,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """

        # eps = 1e-10
        # similarities = []
        # for support_image in support_set:
        #     sum_support = torch.sum(torch.pow(support_image, 2), 1)
        #     support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
        #     dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
        #     cosine_similarity = dot_product * support_manitude
        #     similarities.append(cosine_similarity)
        # similarities = torch.stack(similarities)
        eps = 1e-10
        #cossimilarities = []
        #edsimilarity = []
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
            #sum = torch.sum(torch.pow(torch.abs(support_image - input_image),2),1)
            #edsimilarity =
        similarities = torch.stack(similarities)
        return similarities.t()


class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim,use_cuda):
        super(BidirectionalLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size: 
        :param vector_dim: 
        """
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = self.init_hidden(self.use_cuda)

    def init_hidden(self,use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda(),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False).cuda())
        else:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),requires_grad=False))

    # def repackage_hidden(self,h):
    #     """Wraps hidden states in new Variables, to detach them from their history."""
    #     if type(h) == Variable:
    #         return Variable(h.data)
    #     else:
    #         return tuple(self.repackage_hidden(v) for v in h)

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            if isinstance(h, tuple) or isinstance(h, list):
                return tuple(self.repackage_hidden(v) for v in h)
            else:
                return h.detach()

    def forward(self, inputs):
        # self.hidden = self.init_hidden(self.use_cuda)
        self.hidden = self.repackage_hidden(self.hidden)
        print("use the lstm")
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output


class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob, batch_size=32, num_channels=1, learning_rate=1e-3, fce=False, num_classes_per_set=20, \
                 num_samples_per_class=1, image_size=28, use_cuda=True):
        """
        This is our main network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.fce = fce
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.g = Classifier(layer_size=64, num_channels=num_channels, keep_prob=keep_prob, image_size=image_size)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        if self.fce:
            self.lstm = BidirectionalLSTM(layer_size=[32], batch_size=self.batch_size, vector_dim=self.g.outSize,use_cuda=use_cuda)

    def forward(self, support_set_images, support_set_y_one_hot, target_image, target_y):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:
        :return:
        """
        # produce embeddings for support set images
        encoded_images = []
        for i in np.arange(support_set_images.size(1)):
            gen_encode = self.g(support_set_images[:, i, :, :])
            encoded_images.append(gen_encode)

        # produce embeddings for target images
        gen_encode = self.g(target_image)
        encoded_images.append(gen_encode)
        output = torch.stack(encoded_images)

        # use fce?
        if self.fce:
            outputs = self.lstm(output)

        # get similarities between support set embeddings and target
        similarites = self.dn(support_set=output[:-1], input_image=output[-1])

        # produce predictions for target probabilities
        preds = self.classify(similarites, support_set_y=support_set_y_one_hot)

        #print(preds)
        # calculate the accuracy
        crossentropy_loss = F.cross_entropy(preds, target_y.long())

        preds2 = preds

        indice = np.argpartition(preds2.detach().numpy(), (-5, -4, -3, -2, -1), axis=1)[:,-5:]

        sum = float(0)
        for i in range(indice.shape[0]):
            for j in range(indice.shape[1]):
                if indice[i][j] == target_y[i]:
                    print(indice[i][j], target_y[i])
                    sum += 1 / (5 - j)
                    break
        mAP = sum / preds.shape[0]

        return mAP, crossentropy_loss
