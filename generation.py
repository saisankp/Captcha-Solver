import os
import numpy
import random
import cv2
import argparse
import captcha.image
from tqdm import tqdm
import time

# We don't want any duplicates in our training data and we need atleast 11 captchas generated since we split training vs validation data with 10/11 vs 1/11
# If the count of captchas requested is greater than the number of possible unique captchas with size 'length', then exit
def verifyCaptchaRequest(length, count):
    # 26 uppercase letters, 26 lowercase letters, 10 digits
    possibleCharacters = 26 + 26 + 10
    uniqueCaptchasPossible = length ** possibleCharacters
    if(count > uniqueCaptchasPossible):
       print("Error: You are requesting " + str(count) + " images to be generated but only " + str(uniqueCaptchasPossible) + " images can have unique characters")
       exit(1)
    elif(count < 11):
        print("Error: Minimum number of captchas for training and validation images is 11")
        exit(1)


# Parse arguments from the user to generate captchas
def parseArguments(parser):
    parser = argparse.ArgumentParser()
    argumentsAndDescriptions = {
        '--width': ('Width of the generated captcha images', int),
        '--height': ('Height of the generated captcha images', int),
        '--length': ('Number of characters in the generated captchas', int),
        '--count': ('Number of captchas to generate', int),
        '--output-dir': ('Directory to store the generated captchas', str),
        '--symbols': ('File which contains the symbol set for the generated captchas', str)
    }

    for argument, (description, argument_type) in argumentsAndDescriptions.items():
        parser.add_argument(argument, help=description, type=argument_type)

    arguments = parser.parse_args()

    for argument, (description, _) in argumentsAndDescriptions.items():
            if getattr(arguments, argument.replace("--", "").replace("-", "_")) is None:
                print("Error: Please specify {}".format(argument))
                exit(1)

    verifyCaptchaRequest(arguments.length, arguments.count)

    return arguments


def main():
    start_time = time.time()
    # Parse arguments and exit if any are msising
    arguments = parseArguments(argparse.ArgumentParser())

    # Create a captcha generator for the specified width and height of our required captchas
    captcha_generator = captcha.image.ImageCaptcha(width=arguments.width, height=arguments.height)

    # Read the symbol set to generate our required captchas
    symbols_file = open(arguments.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()
    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    # Creating folder where the generated captchas will go
    if not os.path.exists(arguments.output_dir):
        print("Creating output directory " + arguments.output_dir)
        os.makedirs(arguments.output_dir)

    # Creating 4 subdirectories where the generated images and corresponding valid output should be
    training_dir = os.path.join(arguments.output_dir, 'training/images')
    validation_dir = os.path.join(arguments.output_dir, 'validation/images')
    trainingoutput_dir = os.path.join(arguments.output_dir, 'training/output')
    validationoutput_dir = os.path.join(arguments.output_dir, 'validation/output')
    for dir_path in [training_dir, validation_dir, trainingoutput_dir, validationoutput_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Write mode enabled for the output files as we iterate through each captcha we generate
    with open(os.path.join(trainingoutput_dir, 'output.txt'), 'w') as training_output_file, (
        open(os.path.join(validationoutput_dir, 'output.txt'), 'w')) as validation_output_file:

        # Generate training and validation data with a split of 10/11 vs 1/11, same ratio to the example shown in the PDF given to us
        trainingdatasplit = arguments.count // 11 * 10
        for i in tqdm(range(arguments.count), desc="Generating Captchas"):
            random_str = ''.join([random.choice(captcha_symbols) for j in range(arguments.length)])
            image_path = os.path.join(training_dir if i < trainingdatasplit else validation_dir,
                                      random_str + '.png')
            
            # We have randomly generated this captcha before, we need a new one.
            while os.path.exists(image_path):
                random_str = ''.join([random.choice(captcha_symbols) for j in range(arguments.length)])
                image_path = os.path.join(training_dir if i < trainingdatasplit else validation_dir,
                                      random_str + '.png')
            
            # Generate the image with a unique captcha
            image = numpy.array(captcha_generator.generate_image(random_str))
            cv2.imwrite(image_path, image)
            filename = os.path.basename(image_path)

            # # Write the corresponding line for the captcha in the output file
            output_file = training_output_file if i < arguments.count // 11 * 10 else validation_output_file
            output_file.write("{},{}\n".format(filename, random_str))
    end_time = time.time()
    time_taken = end_time - start_time

    with open('time_taken.txt', 'w') as file:
        file.write('Training data generation for {0} images: {1} seconds'.format(arguments.count, time_taken))
        print('Time taken for generating training images:', time_taken, 'seconds')

if __name__ == '__main__':
    main()
