import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import time

batch_size = 20
max_len = 10


def load_jpg_to_bytes(filename):
    path_prefex = "../../../tmp/ICDAR2019/LSVT_dataset/train_full_images/"
    file_path = path_prefex + filename+".jpg"
    img = Image.open(file_path, 'r')
    img_w = img.size[0]
    img_h = img.size[1]
    img_mode = img.mode
    img_raw = img.tobytes()
    return img_w, img_h, img_mode, img_raw


def load_json(file_path):
    with open(file_path, 'r') as f:
        label_list = json.load(f)
    print(len(label_list))
    transcription = []
    points = []
    illegibility = []
    ids = []
    # TODO
    for i in range(0, 100):
        id_i = "gt_" + str(i)
        if id_i not in label_list:
            print(id_i + " not find")
            continue
        else:
            instance = label_list[id_i]
            instance_transcription = []
            instance_points = []
            instance_illegibility = []
            for i in range(len(instance)):
                instance_transcription.append(instance[i]['transcription'])
                instance_points.append(instance[i]['points'])
                instance_illegibility.append(instance[i]['illegibility'])
            transcription.append(instance_transcription)
            points.append(instance_points)
            illegibility.append(instance_illegibility)
            ids.append(id_i)
    return transcription, points, illegibility, ids


def create_tfrecord(file_path, tf_name):
    transcription, points, illegibility, ids = load_json(file_path)
    writer = tf.python_io.TFRecordWriter(tf_name)

    for i in range(len(transcription)):
        _filename = tf.train.Feature(bytes_list=tf.train.BytesList(value=[ids[i].encode()]))
        _frame_transcription = list(map(lambda trans: tf.train.Feature(bytes_list=tf.train.BytesList(value=[trans.encode()])), transcription[i]))
        points_per_file = [np.array(j).astype(np.int8).tobytes() for j in points[i]]
        _frame_points = list(map(lambda p: tf.train.Feature(bytes_list=tf.train.BytesList(value=[p])), points_per_file))
        _frame_illegibility = list(map(lambda ill: tf.train.Feature(bytes_list=tf.train.BytesList(value=[ill.to_bytes(length=1,byteorder="little")])), illegibility[i]))
        img_w, img_h, img_mode, img_raw = load_jpg_to_bytes(ids[i])
        _img_w = tf.train.Feature(int64_list=tf.train.Int64List(value=[img_w]))
        _img_h = tf.train.Feature(int64_list=tf.train.Int64List(value=[img_h]))
        _img_mode = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_mode.encode()]))
        _frame_img_raw = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))]
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'filename': _filename,
                'img_w': _img_w,
                'img_h': _img_h,
                'img_mode': _img_mode
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'transcription': tf.train.FeatureList(feature=_frame_transcription),
                'points': tf.train.FeatureList(feature=_frame_points),
                'illegibility': tf.train.FeatureList(feature=_frame_illegibility),
                'img_raw': tf.train.FeatureList(feature=_frame_img_raw)
            })
        )
        writer.write(example.SerializeToString())
    writer.close()
    print("Create TFRecord Done! Length:", len(transcription))


def parase_function(serialized_example):
    context_features = {
        'filename': tf.FixedLenFeature([], dtype=tf.string),
        'img_w': tf.FixedLenFeature([], dtype=tf.int64),
        'img_h': tf.FixedLenFeature([], dtype=tf.int64),
        'img_mode': tf.FixedLenFeature([], dtype=tf.string),
    }

    sequence_features = {
        'transcription': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'points': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'illegibility': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'img_raw': tf.FixedLenSequenceFeature([], dtype=tf.string),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)
    # transcription = tf.cast(sequence_parsed['transcription'], tf.int32)
    filename = context_parsed['filename']
    img_w = context_parsed['img_w']
    img_h = context_parsed['img_h']
    img_mode = context_parsed['img_mode']
    # transcription = list(tf.map_fn(lambda trans: trans.decode(), sequence_parsed['transcription']))
    # points = tf.decode_raw(sequence_parsed['points'], tf.int8)
    # illegibility = list(tf.map_fn(lambda ill: ill.decode(), sequence_parsed['illegibility']))
    transcription = sequence_parsed['transcription']
    points = tf.decode_raw(sequence_parsed['points'], tf.int8)
    illegibility = sequence_parsed['illegibility']
    img_raw = tf.decode_raw(sequence_parsed['img_raw'], tf.uint8)
    return transcription, points, illegibility, img_raw, img_w, img_h, img_mode


def read_tf_record(filename):
    dataset = tf.data.TFRecordDataset(filename)
    new_dataset = dataset.map(parase_function)
    dataset = new_dataset
    # dataset = new_dataset.repeat().shuffle(10*batch_size)
    # dataset = dataset.padded_batch(batch_size, ([max_len], [max_len], [max_len]))
    iterator = dataset.make_one_shot_iterator()

    sess = tf.InteractiveSession()
    k = 0
    while True:
        try:
            transcription, points, illegibility, img_raw, img_w, img_h, img_mode = iterator.get_next()
            transcription, points, illegibility, img_raw, img_w, img_h, img_mode = sess.run([transcription, points, illegibility, img_raw, img_w, img_h, img_mode])
            transcription = list(map(lambda trans: trans.decode(), transcription))
            points = points.reshape((-1, 4, 2))
            illegibility = list(map(lambda ill: bool.from_bytes(ill, byteorder='little'), illegibility))
            img = Image.frombytes(img_mode.decode(), (img_w, img_w), img_raw)
            img.show()
            if k == 0:
                print(transcription, points)
                k = 1
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break
        else:
            # print(label1)
            pass


def shuffle_batch(filaname, batch_size):
    dataset = tf.data.TFRecordDataset(filaname)
    new_dataset = dataset.map(parase_function)
    dataset_shuffle = new_dataset.shuffle(buffer_size=6000).batch(batch_size).repeat(5000)
    iterator = dataset_shuffle.make_one_shot_iterator()
    next_elem = iterator.get_next()
    return next_elem


if __name__ == '__main__':
    start = time.time()
    # step 1
    # read_dic("train.json")
    # create_tfrecord("../../../tmp/ICDAR2019/LSVT_dataset/train_full_labels.json", "train.tfrecord")
    read_tf_record("train.tfrecord")
    # print("Time: %f s."%(time.time()-start))
    # load_json("../../../tmp/ICDAR2019/LSVT_dataset/train_full_labels.json")