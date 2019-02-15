import numpy as np
import os
import collections
from scipy.misc import imread


class WhaleNShotDataset():
    def __init__(self, batch_size, classes_per_set=20, samples_per_class=1, seed=2017, shuffle=True, use_cache=False):
        """
        onstruct N-shot dataset
        :param batch_size:  Experiment batch_size 
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        :param seed: seed for random function
        :param shuffle: if shuffle the dataset
        :param use_cache: if true,cache dataset to memory.It can speedup the train but require larger memory
        """
        # np.random.seed(seed)
        # self.x = np.load('data/data.npy')
        # self.x = np.reshape(self.x, newshape=(self.x.shape[0], self.x.shape[1], 28, 28, 1))

        #############################
        ID_label_dict = collections.defaultdict(int)
        image_array = []

        fstream = open('../train.csv')
        count = 0
        next(fstream) # drop the first line
        index_c = 0
        for line in fstream:
            if count >= 50:
                break
            line = line.strip().split(',')
            if line[1] not in ID_label_dict:
                ID_label_dict[line[1]] = index_c
                image_array.append([np.transpose([imread('../train/'+line[0],mode = 'L')],(1,2,0))])
                index_c += 1
            else:
                image_array[ID_label_dict[line[1]]].append(np.transpose([imread('../train/'+line[0],mode = 'L')],(1,2,0)))
            count += 1
        fstream.close()
        self.num_channel = image_array[0][0].shape[2]
        self.image_size =  image_array[0][0].shape[0]

        self.x_train = np.array(image_array)

        image_test_list =  os.listdir('../test')
        image_array =[]
        for name in image_test_list:
            image_array.append(np.transpose([imread('../train/'+name,mode = 'L')],(1,2,0)))
        self.x_test = np.array(image_array)

        self.count = 0
        # import test dataset
        # group by whale type
        # whale_dict = collections.defaultdict(list)
        # for k,v in file_dict.items():
        #     whale_dict[v].append(k)
        #
        # # convert to whale array
        # whale_list = whale_dict.values()
        # whale_train = np.zeros((len(whale_list),128,128,1),dtype=float32)
        #     img = imread(file_name)
        # self.file_dict = file_dict
        # self.whale_dict = whale_dict
        # self.x_train = whale_list

        # if shuffle:
        #     np.random.shuffle(self.x)
        # self.x_train, self.x_val, self.x_test = self.x[:1200], self.x[1200:1411], self.x[1411:]
        # self.mean = np.mean(list(self.x_train) + list(self.x_val))
        # self.x_train = self.processes_batch(self.x_train, np.mean(self.x_train), np.std(self.x_train))
        # self.x_test = self.processes_batch(self.x_test, np.mean(self.x_test), np.std(self.x_test))
        # self.x_val = self.processes_batch(self.x_val, np.mean(self.x_val), np.std(self.x_val))
        # self.std = np.std(list(self.x_train) + list(self.x_val))
        self.batch_size = batch_size
        #self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        #self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "test": self.x_test}
        self.use_cache = use_cache
        # if self.use_cache:
        #     self.cached_datatset = {"train": self.load_data_cache(self.x_train),
        #                             "val": self.load_data_cache(self.x_val),
        #                             "test": self.load_data_cache(self.x_test)}

    def processes_batch(self, x_batch, mean, std):
        """
        Normalizes a batch images
        :param x_batch: a batch images
        :return: normalized images
        """
        return (x_batch - mean) / std

    def _sample_new_batch(self, data_pack):
        """
        Collect 1000 batches data for N-shot learning
        :param data_pack: one of(train,test,val) dataset shape[classes_num,20,28,28,1]
        :return: A list with [support_set_x,support_set_y,target_x,target_y] ready to be fed to our networks
        """
        classes_per_set = self.x_train.shape[0]
        if data_pack == "train":
            classes_per_set = self.classes_per_set

        image_size = self.image_size
        image_channel = self.num_channel

        support_set_x = np.zeros((self.batch_size, classes_per_set, self.samples_per_class,
                                  image_size, image_size, image_channel))

        support_set_y = np.zeros((self.batch_size, classes_per_set, self.samples_per_class),np.int32)
        target_x = np.zeros((self.batch_size, image_size, image_size, image_channel))
        target_y = np.zeros((self.batch_size, 1),np.int32)

        # for i in range(self.batch_size):
        #     classes_idx = np.arange(data_pack.shape[0])
        #     samples_idx = np.arange(data_pack.shape[1])
        #     choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
        #     choose_label = np.random.choice(self.classes_per_set, size=1)
        #     choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)
        #
        #     x_temp = data_pack[choose_classes]
        #     x_temp = x_temp[:, choose_samples]
        #     y_temp = np.arange(self.classes_per_set)
        #     support_set_x[i] = x_temp[:, :-1]
        #     support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
        #     target_x[i] = x_temp[choose_label, -1]
        #     target_y[i] = y_temp[choose_label]

        #################################

        for i in range(self.batch_size):
            # choose support data
            #  take every whale type
            classes_idx = np.arange(self.datatset['train'].shape[0])
            label_list = np.random.choice(classes_idx, size = classes_per_set, replace=False)
            for label in range(len(label_list)):
                # choose k samples in one type
                for k in range(self.samples_per_class):
                    choice = np.random.choice(len(self.datatset['train'][label]))
                    support_set_x[i][label][k] = self.datatset['train'][label][choice]
                    support_set_y[i][label][k] = label
                    # choose target data
                    if data_pack == "train":
                        target_y[i] = np.random.choice(np.arange(len(label_list)))
                        choice = np.random.choice(len(self.datatset['train'][label_list[target_y[i][0]]]))
                        target_x[i] = self.datatset['train'][label_list[target_y[i][0]]][choice]

            if data_pack == "test":
                target_x[i] = self.datatset["test"][self.count]
                self.count += 1

        ##########

        return support_set_x, support_set_y, target_x, target_y

    def _rotate_data(self, image, k):
        """
        Rotates one image by self.k * 90 degrees counter-clockwise
        :param image: Image to rotate
        :return: Rotated Image
        """
        return np.rot90(image, k)

    def _rotate_batch(self, batch_images, k):
        """
        Rotates a whole image batch
        :param batch_images: A batch of images
        :param k: integer degree of rotation counter-clockwise
        :return: The rotated batch of images
        """
        batch_size = batch_images.shape[0]
        for i in np.arange(batch_size):
            batch_images[i] = self._rotate_data(batch_images[i], k)
        return batch_images

    def _get_batch(self, dataset_name, augment=False):
        """
        Get next batch from the dataset with name.
        :param dataset_name: The name of dataset(one of "train","val","test")
        :param augment: if rotate the images
        :return: a batch images
        """
        if self.use_cache:
            support_set_x, support_set_y, target_x, target_y = self._get_batch_from_cache(dataset_name)
        else:
            support_set_x, support_set_y, target_x, target_y = self._sample_new_batch(dataset_name)
        if augment:
            k = np.random.randint(0, 4, size=(self.batch_size, self.classes_per_set))
            a_support_set_x = []
            a_target_x = []
            for b in range(self.batch_size):
                temp_class_set = []
                for c in range(self.classes_per_set):
                    temp_class_set_x = self._rotate_batch(support_set_x[b, c], k=k[b, c])
                    if target_y[b] == support_set_y[b, c, 0]:
                        temp_target_x = self._rotate_data(target_x[b], k=k[b, c])
                    temp_class_set.append(temp_class_set_x)
                a_support_set_x.append(temp_class_set)
                a_target_x.append(temp_target_x)
            support_set_x = np.array(a_support_set_x)
            target_x = np.array(a_target_x)
        support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                               support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
        support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
        return support_set_x, support_set_y, target_x, target_y

    def get_train_batch(self, augment=False):
        return self._get_batch("train", augment)

    def get_val_batch(self, augment=False):
        return self._get_batch("val", augment)

    def get_test_batch(self, augment=False):
        return self._get_batch("test", augment)

    def load_data_cache(self, data_pack, argument=True):
        """
        cache the dataset in memory
        :param data_pack: shape[classes_num,20,28,28,1]
        :return:
        """
        cached_dataset = []
        classes_idx = np.arange(data_pack.shape[0])
        samples_idx = np.arange(data_pack.shape[1])
        for _ in range(1000):
            support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                      data_pack.shape[3], data_pack.shape[4]), np.float32)

            support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
            target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                                np.float32)
            target_y = np.zeros((self.batch_size, 1), np.int32)
            for i in range(self.batch_size):
                choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
                choose_label = np.random.choice(self.classes_per_set, size=1)
                choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

                x_temp = data_pack[choose_classes]
                x_temp = x_temp[:, choose_samples]
                y_temp = np.arange(self.classes_per_set)
                support_set_x[i] = x_temp[:, :-1]
                support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
                target_x[i] = x_temp[choose_label, -1]
                target_y[i] = y_temp[choose_label]
            cached_dataset.append([support_set_x, support_set_y, target_x, target_y])
        return cached_dataset

    def _get_batch_from_cache(self, dataset_name):
        """

        :param dataset_name:
        :return:
        """
        if self.indexes[dataset_name] >= len(self.cached_datatset[dataset_name]):
            self.indexes[dataset_name] = 0
            self.cached_datatset[dataset_name] = self.load_data_cache(self.datatset[dataset_name])
        next_batch = self.cached_datatset[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target

