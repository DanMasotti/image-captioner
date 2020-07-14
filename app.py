from model import *

if __name__ == "__main__":
    vocab = get_vocab()
    vocab_size = len(vocab)
    max_caption_length = get_max_length_captions()
    train_captions, test_captions = get_captions()
    train_image_features, test_image_features = get_image_features()
    tokenizer = get_tokenizer()

    model = Model(tokenizer, vocab_size, max_caption_length)

    model.train(train_captions, train_image_features)

    BLEU_1, BLEU_2, BLEU_3, BLEU_4 = model.test(test_captions, test_image_features)
    print("&&&&&&&&&&&&&&&&&&&& TEST RESULTS &&&&&&&&&&&&&&&&&&&&&&")
    print('BLEU-1: ', BLEU_1)
    print('BLEU-2: ', BLEU_2)
    print('BLEU-3: ', BLEU_3)
    print('BLEU-4: ', BLEU_4)
