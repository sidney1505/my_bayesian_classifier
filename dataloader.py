import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import code # code.interact(local=dict(globals(), **locals()))


# DATA preprocessing
def load_and_preprocess_binary_data(config):
    positive_images = []
    negative_images = []
    #
    print('load images')
    for it, file in enumerate(os.listdir(config['dataset_dir'] + '/positive_samples')):
        if config['debug_mode'] and it >= 500:
            print('careful, its the debug moad, which doent load all data!')
            break
        img = Image.open(config['dataset_dir'] + '/positive_samples/' + file)
        img = img.resize([224,224])
        #img = img.resize([299,299])
        positive_images.append(img)

    for it, file in enumerate(os.listdir(config['dataset_dir'] + '/negative_samples')):
        if config['debug_mode'] and it >= 500:
            break
        img = Image.open(config['dataset_dir'] + '/negative_samples/' + file)
        img = img.resize([224,224])
        #img = img.resize([299,299])
        negative_images.append(img)
    #

    print('preprocess data')
    positive_train = positive_images[:-int(config['val_split'] * len(positive_images))]
    positive_test = positive_images[-int(config['val_split'] * len(positive_images)):]

    negative_train = negative_images[:-int(config['val_split'] * len(negative_images))]
    negative_test = negative_images[-int(config['val_split'] * len(negative_images)):]

    train_images = np.concatenate([np.stack(positive_train), np.stack(negative_train)])
    test_images = np.concatenate([np.stack(positive_test), np.stack(negative_test)])

    if len(positive_train) <= len(negative_train):
        train_labels = np.concatenate([np.ones(len(positive_train), dtype=np.int32), np.zeros(len(negative_train), dtype=np.int32)])
        test_labels = np.concatenate([np.ones(len(positive_test), dtype=np.int32), np.zeros(len(negative_test), dtype=np.int32)])
    else:
        train_labels = np.concatenate([np.zeros(len(positive_train), dtype=np.int32), np.ones(len(negative_train), dtype=np.int32)])
        test_labels = np.concatenate([np.zeros(len(positive_test), dtype=np.int32), np.ones(len(negative_test), dtype=np.int32)])

    train_images = preprocess_input(train_images)
    test_images = preprocess_input(test_images)
    #
    train_images = np.transpose(train_images, [0,3,1,2])
    test_images = np.transpose(test_images, [0,3,1,2])
    #
    return train_images, test_images, train_labels, test_labels



# for datasets small enough to fit in the RAM!
def load_and_preprocess_multiclass_data_with_numpy(config):
    # TODO better let this do by the dataloader!!!
    images_by_class = config['num_classes'] * [[]]
    #
    print('load images')
    for class_nr, class_path in enumerate(os.listdir(config['dataset_dir'])):
        print(class_nr)
        for it, file in enumerate(os.listdir(config['dataset_dir'] + '/' + class_path)):
            img = Image.open(config['dataset_dir'] + '/' + class_path + '/' + file)
            img = img.resize([224,224])
            images_by_class[class_nr].append(img)

    print('preprocess data')
    code.interact(local=dict(globals(), **locals()))
    #
    train_images_by_class = list(map(lambda class_nr: images_by_class[class_nr][:-int(config['val_split'] * len(images_by_class[class_nr]))], range(config['num_classes'])))
    val_images_by_class = list(map(lambda class_nr: images_by_class[class_nr][-int(config['val_split'] * len(images_by_class[class_nr])):], range(config['num_classes'])))
    #
    train_images = np.concatenate(list(map(lambda elem: np.stack(elem), train_images_by_class)))
    val_images = np.concatenate(list(map(lambda elem: np.stack(elem), val_images_by_class)))
    #
    train_labels = np.concatenate(list(map(lambda class_nr: class_nr * np.ones(len(train_images_by_class[class_nr]), dtype=np.int32), range(config['num_classes']))))
    val_labels = np.concatenate(list(map(lambda class_nr: class_nr * np.ones(len(val_images_by_class[class_nr]), dtype=np.int32), range(config['num_classes']))))
    #
    train_images = preprocess_input(train_images)
    val_images = preprocess_input(val_images)
    #
    train_images = np.transpose(train_images, [0,3,1,2])
    val_images = np.transpose(val_images, [0,3,1,2])
    #
    return train_images, val_images, train_labels, test_labels


# for datasets to big to fit into the RAM
def load_and_preprocess_multiclass_validation_data(config):
    # TODO better let this do by the dataloader!!!
    images_by_class = config['num_classes'] * [[]]
    #
    print('load images')
    for class_nr, class_path in enumerate(os.listdir(config['dataset_dir'])):
        print(class_nr)
        for it, file in enumerate(os.listdir(config['dataset_dir'] + '/' + class_path)):
            if config['debug_mode'] and it >= 500:
                break
            img = Image.open(config['dataset_dir'] + '/' + class_path + '/' + file)
            img = img.resize([224,224])
            images_by_class[class_nr].append(img)

    print('preprocess data')
    #
    val_images_by_class = list(map(lambda class_nr: images_by_class[class_nr][-int(config['val_split'] * len(images_by_class[class_nr])):], range(config['num_classes'])))
    #
    val_images = np.concatenate(list(map(lambda elem: np.stack(elem), val_images_by_class)))
    #
    val_images = preprocess_input(val_images)
    # channels first as demanded
    val_images = np.transpose(val_images, [0,3,1,2])
    #
    val_labels = np.concatenate(list(map(lambda class_nr: class_nr * np.ones(len(val_images_by_class[class_nr]), dtype=np.int32), range(config['num_classes']))))
    #
    return val_images, val_labels


# for a feature extraction before training in order to avoid forward pass through CNN each step
def extract_data_features(dataset_dir, save_features=True):
    print('extract features with pretrained inception net')
    #
    train_images, test_images, train_labels, test_labels = load_and_preprocess_binary_data(dataset_dir)
    # resnet = ResNet50(include_top=False, weights='imagenet')
    # resnet.predict(np.zeros([5,224,224,3]))
    # code.interact(local=dict(globals(), **locals()))
    train_images = InceptionResNetV2(include_top=False, weights='imagenet').predict(train_images)
    test_images = InceptionResNetV2(include_top=False, weights='imagenet').predict(test_images)
    # TODO store these + the groundtruth somewhere in order to avoid long computation times!!!
    # code.interact(local=dict(globals(), **locals()))
    if save_data:
        np.savez(open(dataset_dir + '/preprocessed.npz','wb'), train_images=train_images, test_images=test_images, train_labels=train_labels, test_labels=test_labels)
    #
    return train_images, test_images, train_labels, test_labels


# loading of the saved features
def load_features(dataset_dir):
    print('load data!')
    data = np.load(open(dataset_dir + '/preprocessed.npz','rb'))
    train_images = data['train_images']
    test_images = data['test_images']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    return train_images, test_images, train_labels, test_labels
