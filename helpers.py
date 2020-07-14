import os
from os import listdir
import numpy as np
from tensorflow import keras
import string
import json

params = json.load("params.json")
num_epochs = params["NUM_EPOCHS"]
batch_sz = params["BATCH_SIZE"]


def make_image_features(image_directory="Flickr8k_Dataset"):
    if os.path.exists("features.json"):
        with open("features.json") as f:
            my_data = f.read()
            features = json.load(my_data)
    else:
        features = {}
        model = keras.applications.vgg16.VGG16()
        model = keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].outputs)

        for i, name in enumerate(listdir(image_directory)):
            image_path = image_directory + "/" + name
            image_id = name.split(".")[0]

            image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            x = keras.preprocessing.image.img_to_array(image)

            image_feature = model.predict(x, verbose=0)
            features[image_id] = image_feature

        json.dump(features, "features.json")

    return features


def get_image_features(image_directory="Flickr8k_text"):
    with open(image_directory + '/Flickr_8k.trainImages.txt', 'r') as f:
        trainImages = f.read()
    with open(image_directory + '/Flickr_8k.testImages.txt', 'r') as f:
        testImages = f.read()

    train_img_names = []
    for line in trainImages.split('\n'):
        train_img_names.append(line.split('.')[0])
    train_img_names = set(train_img_names)

    test_img_names = []
    for line in testImages.split('\n'):
        test_img_names.append(line.split('.')[0])
    test_img_names = set(test_img_names)

    features = make_image_features()

    train_img_features = {k: features[k] for k in train_img_names if not (k == '')}
    test_img_features = {k: features[k] for k in test_img_names if not (k == '')}

    return train_img_features, test_img_features


def make_captions_and_vocab(path_to_text="Flickr8k_text/Flickr8k.token.txt", start_token="START", stop_token="STOP"):
    if os.path.exists("captions.json") and os.path.exists("vocabulary.json"):

        with open("captions.json", "r") as f:
            my_data = f.read()
            captions = json.load(my_data)

        with open("vocabulary.json", "r") as f:
            my_data = f.read()
            vocab = json.load(my_data)

    else:
        captions = {}
        vocab = set()

        with open(path_to_text, "r") as f:
            text = f.read()

        lines = text.split("\n")
        for line in lines:
            if len(line) < 2:
                continue
            words = line.split()
            img_id = words[0].split(".")[0]

            caption = words[1:]
            caption = [x.lower() for x in caption]
            caption = [x.translate(str.maketrans('', '', string.punctuation)) for x in caption]
            caption = [x for x in caption if x.isalpha()]
            caption = [x for x in caption if len(x) > 1]

            [vocab.add(x) for x in caption]

            caption = " ".join(caption)
            caption = start_token + " " + caption + " " + stop_token
            if img_id not in captions:
                captions[img_id] = [caption]
            else:
                captions[img_id].append(caption)

        json.dump(captions, "captions.json")
        json.dump(vocab, "vocabulary.json")

    return captions, vocab


def get_vocab():
    _, vocab = make_captions_and_vocab()
    return vocab


def get_captions(image_directory="Flickr8k_text"):
    with open(image_directory + '/Flickr_8k.trainImages.txt', 'r') as f:
        trainImages = f.read()
    with open(image_directory + '/Flickr_8k.testImages.txt', 'r') as f:
        testImages = f.read()

    train_img_names = []
    for line in trainImages.split('\n'):
        train_img_names.append(line.split('.')[0])
    train_img_names = set(train_img_names)

    test_img_names = []
    for line in testImages.split('\n'):
        test_img_names.append(line.split('.')[0])
    test_img_names = set(test_img_names)
    captions, _ = make_captions_and_vocab()

    train_captions = {k: captions[k] for k in train_img_names if not (k == '')}
    test_captions = {k: captions[k] for k in test_img_names if not (k == '')}

    return train_captions, test_captions


def get_max_length_captions():
    train_captions, _ = get_captions()
    captions_list = [caption for captions in train_captions.values() for caption in captions]
    max_caption_length = max(len(s.split()) for s in captions_list)
    return max_caption_length


def get_tokenizer():
    if os.path.exists("tokenizer.json"):
        with open("tokenizer.json") as f:
            my_data = f.read()
            tokenizer = json.load(my_data)
            return tokenizer
    else:
        train_captions, _ = get_captions()
        captions_list = [caption for captions in train_captions.values() for caption in captions]
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(captions_list)
        json.dump(tokenizer, "tokenizer.json")
        return tokenizer


def get_vocab_size():
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer.word_index) + 1
    return vocab_size


def make_batch(tokenizer, max_caption_length, captions_list, image):
    input1 = []
    input2 = []
    output = []
    for caption in captions_list:
        sequence = tokenizer.text_to_sequence([caption])
        for i in range(1, len(sequence)):
            in_, out_ = sequence[:i], sequence[i]
            in_ = keras.preprocessing.sequence.pad_sequences([in_], maxlen=max_caption_length)[0]
            out_ = keras.utils.to_categorical([out_], num_classes=get_vocab_size())[0]
            input1.append(image)
            input2.append(in_)
            output.append(out_)
    return np.array(input1), np.array(input2), np.array(output)


def generate_data(captions, images, tokenizer, max_caption_length):
    while True:
        for image_id, captions_list in captions.items():
            image = images[image_id][0]
            encoder_input, decoder_input, decoder_output = make_batch(tokenizer, max_caption_length, captions_list,
                                                                      image)

            yield [[encoder_input, decoder_input], decoder_output]


def word2id(num, tokenizer):
    for word, i in tokenizer.word_index.items():
        if i == num:
            return word
    return None
