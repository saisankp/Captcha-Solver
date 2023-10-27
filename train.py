import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import cv2
import numpy
import string
import random
import argparse
import time
import tensorflow as tf
import tensorflow.keras as keras
import re

def parseArguments(parser):
    parser = argparse.ArgumentParser()
    argumentsAndDescriptions = {
    '--width': ('Width of captcha image', int),
    '--height': ('Height of captcha image', int),
    '--length': ('Length of captchas in characters', int),
    '--batch-size': ('How many images in each training captcha batch', int),
    '--train-dataset': ('Where to look for the training image dataset', str),
    '--validate-dataset': ('Where to look for the validation image dataset', str),
    '--output-model-name': ('Where to save the trained model', str),
    '--input-model': ('Where to look for the input model to continue training', str),
    '--epochs': ('How many training epochs to run', int),
    '--symbols': ('File with the symbols to use in captchas', str)
    }

    for argument, (description, argument_type) in argumentsAndDescriptions.items():
        parser.add_argument(argument, help=description, type=argument_type)

    arguments = parser.parse_args()

    for argument, (description, _) in argumentsAndDescriptions.items():
            if getattr(arguments, argument.replace("--", "").replace("-", "_")) is None:
                print("Error: Please specify {}".format(argument))
                exit(1)

    return arguments


# To predict captcha text, we need to create a keras model to predict captchas
# We can base the model on using a maximum length, with '$' used for captcha characters less than the maximum length
# This way, our model can predict variable length captchas without using LSTM (long short-term memory networks)
def create_model(max_captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
  input_tensor = keras.Input(input_shape)
  output_tensor = input_tensor
  for i, module_length in enumerate([module_size] * model_depth):
      for j in range(module_length):
          output_tensor = keras.layers.Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(output_tensor)
          output_tensor = keras.layers.BatchNormalization()(output_tensor)
          output_tensor = keras.layers.Activation('relu')(output_tensor)
      output_tensor = keras.layers.MaxPooling2D(2)(output_tensor)
  output_tensor = keras.layers.Flatten()(output_tensor)
  output_tensor = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name='char_%d'%(i+1))(output_tensor) for i in range(max_captcha_length)]
  model = keras.Model(inputs=input_tensor, outputs=output_tensor)
  return model


# To train our model we need to use a class that signifies our folder of training images (batches of captchas to train)
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, max_captcha_length, captcha_symbols, captcha_width, captcha_height):
        # Set items in class
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.max_captcha_length = max_captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height
        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))-1

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for i in range(self.max_captcha_length)]

        for i in range(self.batch_size):
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]
            # Pop off image as we have used it
            self.used_files.append(self.files.pop(random_image_label))
            # Divide by 255 to scale the 8-bit RBG input from [0 -> 1] for keras
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = numpy.array(rgb_data) / 255.0
            X[i] = processed_data

            # We swapped : and / for o and L respectively as they are not allowed in filenames
            # Let's put them back so our model predicts accurately
            random_image_label = random_image_label.replace('o', ':')
            random_image_label = re.sub("L", "\\\\", random_image_label)
            # We add '$' to the right-side of the label until it is length 6 (if it is length 6 already, we don't do anything)
            random_image_label = '{:$<6}'.format(random_image_label.split('_')[0])
            for j, ch in enumerate(random_image_label):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1
        return X, y


def main():
    start_time = time.time()
    # Parse arguments and exit if any are msising
    arguments = parseArguments(argparse.ArgumentParser())

    captcha_symbols = None
    with open(arguments.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()

    # Only use the CPU as our local device is limited
    with tf.device('/device:CPU:0'):
        model = create_model(arguments.length, len(captcha_symbols), (arguments.height, arguments.width, 3))
        if arguments.input_model is not None:
            model.load_weights(arguments.input_model)
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])
        model.summary()
        training_data = ImageSequence(arguments.train_dataset, arguments.batch_size, arguments.length, captcha_symbols, arguments.width, arguments.height)
        validation_data = ImageSequence(arguments.validate_dataset, arguments.batch_size, arguments.length, captcha_symbols, arguments.width, arguments.height)
        callbacks = [keras.callbacks.EarlyStopping(patience=3), keras.callbacks.ModelCheckpoint(args.output_model_name+'.h5', save_best_only=False)]
        with open(arguments.output_model_name+".json", "w") as json_file:
            json_file.write(model.to_json())
        try:
            model.fit_generator(generator=training_data,
                                validation_data=validation_data,
                                epochs=arguments.epochs,
                                callbacks=callbacks,
                                use_multiprocessing=True)
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name+'_resume.h5')
            model.save_weights(args.output_model_name+'_resume.h5')
    
    end_time = time.time()
    time_taken = end_time - start_time
    with open('train_time_taken.txt', 'w') as file:
        file.write('Train: {:.2f} seconds'.format(time_taken))
        print('Time taken for training the model:', time_taken, 'seconds')


if __name__ == '__main__':
    main()
