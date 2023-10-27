import warnings
import time
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
from tensorflow import keras


def parseArguments(parser):
    parser = argparse.ArgumentParser()
    argumentsAndDescriptions = {
    '--model-name': ('Model name to use for classification', str),
    '--captcha-dir': ('Where to read the captchas from', str),
    '--output': ('File where the classifications should be saved', str),
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


def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


def main():
    start_time = time.time()
    # Parse arguments and exit if any are missing
    arguments = parseArguments(argparse.ArgumentParser())
    symbols_file = open(arguments.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()
    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
    # Only using CPU because of limited throttled computational power on the Raspberry Pi
    with tf.device('/cpu:0'):
        with open(arguments.output, 'w') as output_file:
            json_file = open(arguments.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(arguments.model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])
            for x in sorted(os.listdir(arguments.captcha_dir)):
                raw_data = cv2.imread(os.path.join(arguments.captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])
                prediction = model.predict(image)
                output_file.write(x + "," + decode(captcha_symbols, prediction).replace("$","") + "\n")
                print('Classified: ' + x)
                print('The model predicts: ' + decode(captcha_symbols, prediction).replace("$","") )
    end_time = time.time()
    time_taken = end_time - start_time
    with open('classify_time_taken.txt', 'w') as file:
        file.write('Classify: {:.2f} seconds'.format(time_taken))
        print('Time taken for classifying captchas from server:', time_taken, 'seconds')


if __name__ == '__main__':
    main()
