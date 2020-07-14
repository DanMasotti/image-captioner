import tensorflow as tf
from tensorflow import keras
from helpers import *


class Model:

	def __init__(self, my_tokenizer, vocab_sz, max_length):

		'''
		part of the model that uses the image features
		'''
		image_inputs = keras.layers.Input(shape=(4096,))
		dropout_image = keras.layers.Dropout(0.5)(image_inputs)
		layer1 = keras.layers.Dense(256, activation="relu")(dropout_image)

		''' 
		part of the model that uses the text features
		'''

		text_inputs = keras.layers.Input(shape=(max_length,))
		embeddings = keras.layers.Embedding(vocab_sz, 256, mask_zero=True)(text_inputs)
		dropout_text = keras.layers.Dropout(0.5)(embeddings)
		layer2 = keras.layers.LSTM(256)(dropout_text)

		'''
			Add the outputs of the image layer and the caption layer into
			the decoder, then pass through dense layer with ReLu activation, 
			then pass through another dense layer for a softmax prediction over 
			all the vocab for the next word in the outputted sequence caption.
		'''

		decoder1 = keras.layers.merge.add([layer1, layer2])
		decoder2 = keras.layers.Dense(256, activation="relu")(decoder1)
		output = keras.layers.Dense(vocab_sz, activation="softmax")(decoder2)

		'''
			Build the model with the above architecture. For conceptuals:
			inputs = [image_features, caption]
			output = word
		'''
		self.model = keras.models.Model(inputs=[image_inputs, text_inputs], outputs=output)

		'''
			Minimize the model's cross-entropy loss with the Adam Optimizer, and 
			the default learning rate.
		'''

		self.model.compile(loss="categorical-cross_entropy", optimizer="adam")

		self.max_caption_length = max_caption_length
		self.tokenizer = my_tokenizer

	def train(self, captions, image_features):
		steps = len(image_features)
		generator = data_generator(captions, image_features, self.tokenizer, self.max_caption_length)
		self.model.fit_generator(generator, epochs=num_epochs, steps_per_epoch=len(steps), verbose=1)
		self.model.save("model.h5")

	def predict_caption(self, image, start_token="START", stop_token="STOP"):

		prediction = start_token
		for _ in range(self.max_caption_length):
			sequence = self.tokenizer.texts_to_sequences([prediction])[0]
			sequence = keras.preprocessing.sequence.pad_sequences([sequence], maxlen=self.max_caption_length)
			y_hat = self.model.predict([image, sequence], verbose=0)
			y_hat = np.argmax(y_hat, tokenizer)
			word = word2id(y_hat, tokenizer)
			if word is None:
				break
			prediction += " " + word
			if word == stop_token:
				break
		return prediction

	def test(self, captions, image_features, start_token="START", stop_token="STOP"):
		actual, predicted = [], []

		tot = len(captions.keys())
		i = 0
		for image_id, caption_list in test_captions.items():
			y_hat = predict_caption(self.tokenizer, image, self.max_caption_length)
			y_true = [x.split() for x in captions_list]
			actual.append(y_true)
			predicted.append(y_hat)

			if i % 100 == 0:
				print(str(i) + " / " + str(tot))
			i += 1

		BLEU_1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
		BLEU_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
		BLEU_3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
		BLEU_4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

		return BLEU_1, BLEU_2, BLEU_3, BLEU_4
