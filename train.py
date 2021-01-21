import argparse
import os
import numpy as np
import json
from scripts.voc import parse_voc_annotation
from core.yolov4_tiny import YOLOV4_tiny, dummy_loss
from scripts.generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from scripts.callbacks import CustomModelCheckpoint, CustomTensorBoard
from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
from core.activation import Mish, Mish6
from keras.models import load_model
from utils.draw_loss import LossHistory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_training_instances(
    train_annot_folder,
    train_image_folder,
    train_cache,
    valid_annot_folder,
    valid_image_folder,
    valid_cache,
    labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("[INFO] valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8*len(train_ints))
        np.random.seed(0)
        np.random.shuffle(train_ints)
        np.random.seed()

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in defect_config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('[INFO] Seen labels: \t'  + str(train_labels) + '\n')
        print('[INFO] Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('[INFO] Some labels have no annotations! Please revise the list of labels in the defect_config.json.')
            return None, None, None
    else:
        print('[INFO] No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save, valid_data, labels, loss_pic_name):
    makedirs(tensorboard_logs)
    
    early_stop = EarlyStopping(
        monitor     = 'loss', 
        min_delta   = 0.01, 
        patience    = 5, 
        mode        = 'min', 
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = model_to_save,
        filepath        = saved_weights_name.split('.')[0] + '_best.h5',
        monitor         = 'loss', 
        verbose         = 1, 
        save_best_only  = True,
        mode            = 'min', 
        period          = 1,
        addtion_save    = saved_weights_name,
        valid_data      = valid_data,
        labels          = labels
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 5,
        verbose  = 1,
        mode     = 'min',
        min_delta= 0.01,
        cooldown = 0,
        min_lr   = 0
    )
    tensorboard = CustomTensorBoard(
        log_dir                = tensorboard_logs,
        write_graph            = True,
        write_images           = True,
    )
    logs_loss = LossHistory(loss_pic_name=loss_pic_name)
    # return [early_stop, checkpoint, reduce_on_plateau, tensorboard, logs_loss]
    return [checkpoint, reduce_on_plateau, tensorboard, logs_loss]

def create_model(
    config,
    max_box_per_image,
    warmup_batches,
    multi_gpu,
    saved_weights_name,
    lr
):
    if config["model"]["model_name"] == "YOLOV4_tiny":
        print('[INFO] YOLOV4_tiny Model Creating...')
        if multi_gpu > 1:
            with tf.device('/cpu:1'):
                yolo_model = YOLOV4_tiny(
                    config=config,
                    max_box_per_image=max_box_per_image,
                    batch_size=config["train"]["batch_size"] // multi_gpu,
                    warmup_batches=warmup_batches)
                template_model, infer_model = yolo_model.model()
        else:
            yolo_model = YOLOV4_tiny(
                config=config,
                max_box_per_image=max_box_per_image,
                batch_size=config["train"]["batch_size"],
                warmup_batches=warmup_batches)
            template_model, infer_model = yolo_model.model()
    else:
        pass

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print("[INFO] Find pretrained weights...")
        print("\n[INFO] Loading pretrained weights...\n")
        template_model.load_weights(saved_weights_name)
    # else:
        # template_model.load_weights("backend.h5", by_name=True)

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model

    optimizer = Adam(lr=lr, clipnorm=0.001)
    # optimizer = SGD(lr=lr, momentum=0.9, decay=0.0005)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


def _main_(args):
    config_path = args.conf

    with open(config_path, encoding='UTF-8') as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print('\n[INFO] Training on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators
    ###############################
    train_generator = BatchGenerator(
        instances           = train_ints,
        anchors             = config['model']['anchors'],
        labels              = labels,
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],
        shuffle             = True,
        jitter              = 0.3,
        norm                = normalize
    )

    valid_generator = BatchGenerator(
        instances           = valid_ints,
        anchors             = config['model']['anchors'],
        labels              = labels,
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],
        shuffle             = True,
        jitter              = 0.0,
        norm                = normalize
    )
    print("[INFO] Creating Model...")
    ###############################
    #   Create the model
    ###############################
    if os.path.exists(config['train']['saved_weights_name']):
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        config=config,
        max_box_per_image=max_box_per_image,
        warmup_batches= warmup_batches,
        multi_gpu=multi_gpu,
        saved_weights_name=config['train']['saved_weights_name'],
        lr=config['train']['learning_rate']
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(config['train']['saved_weights_name'],
                                 config['train']['tensorboard_dir'],
                                 infer_model,
                                 valid_generator,
                                 labels,
                                 config_path.split('_')[0])
    print("[INFO] Training Model...")
    train_model.fit_generator(
        generator        = train_generator,
        steps_per_epoch  = len(train_generator) * config['train']['train_times'],
        epochs           = config['train']['nb_epochs'],
        validation_data  = valid_generator,
        validation_steps = len(train_generator) / config['train']['batch_size'],
        verbose          = 2 if config['train']['debug'] else 1,
        callbacks        = callbacks,
        workers          = 4,
        max_queue_size   = 8
    )
    print("[INFO] Saving Final Model...")
    infer_model.save(filepath=config['train']['saved_weights_name'])

    print("[INFO] Start Evalutating Model...")
    # make a GPU version of infer_model for evaluation
    if multi_gpu > 1:
        infer_model = load_model(config['train']['saved_weights_name'],
                                 custom_objects={'Mish': Mish,
                                                 'Mish6': Mish6,
                                                 'tf': tf},
                                 compile=False)

    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)
    ap = []

    # print the mAP score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ' average precision(AP): {:.6f}'.format(average_precision['ap']))
        ap.append(average_precision['ap'])
        print(labels[label] + ' recall: {:.6f}'.format(average_precision['recall']))
        print(labels[label] + ' precision: {:.6f}'.format(average_precision['precision']))
    print('[INFO] mAP: {:.6f}'.format(sum(ap) / len(ap)))
    print("[INFO] Completed...")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate keras YOLOV4_tiny on any dataset')
    argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
