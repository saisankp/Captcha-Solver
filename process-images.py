import cv2
import argparse
import os
from tqdm import tqdm
import time


def parseArguments(parser):
    parser = argparse.ArgumentParser()
    argumentsAndDescriptions = {
    '--image-folder': ('Where the generated images are', str),
    '--processed-image-folder': ('Where the processed images will be stored', str),
     '--type-of-data': ('Whether the data is training or validation', str)
    }

    for argument, (description, argument_type) in argumentsAndDescriptions.items():
        parser.add_argument(argument, help=description, type=argument_type)

    arguments = parser.parse_args()

    for argument, (description, _) in argumentsAndDescriptions.items():
            if getattr(arguments, argument.replace("--", "").replace("-", "_")) is None:
                print("Error: Please specify {}".format(argument))
                exit(1)

    if(arguments.type_of_data.lowercase != "training" and arguments.type_of_data.lowercase != "validation"):
        print("Error: Please specify whether the data is training or validation")
        exit(1)

    return arguments


# After using contouring to remove black dots, we found that the model struggles to find the character ":"
# So, we simply make the image black and white 
def process_image(path_of_image, path_to_write_new_image):
    image = cv2.imread(path_of_image)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (threshold, image_in_black_and_white) = cv2.threshold(grey_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(path_to_write_new_image, image_in_black_and_white)


def main():
    start_time = time.time()
    # Parse arguments and exit if any are msising
    arguments = parseArguments(argparse.ArgumentParser())

    # If the resulting directory doesn't exist already, make it!
    if not os.path.exists(arguments.processed_image_folder):
        os.makedirs(arguments.processed_image_folder)

    # Iterate through every file in the specified image folder
    for filename in tqdm(os.listdir(arguments.image_folder), total=len(os.listdir(arguments.image_folder)), desc='Processing generated captchas'):
        file_path_of_initial_image = os.path.join(arguments.image_folder, filename)
        file_path_of_new_image = str(os.path.join(arguments.processed_image_folder, filename))
        process_image(file_path_of_initial_image, file_path_of_new_image)

    end_time = time.time()
    time_taken = end_time - start_time

    if(arguments.type_of_data.lowercase() == "training"):
        with open('process_training_images_time_taken.txt', 'w') as file:
            file.write('Processed training images: {1} seconds'.format(time_taken))
            print('Time taken for processing training images:', time_taken, 'seconds')
    else:
        with open('process_validation_images_time_taken.txt', 'w') as file:
            file.write('Processed validation images: {1} seconds'.format(time_taken))
            print('Time taken for processing validation images:', time_taken, 'seconds')


if __name__ == '__main__':
    main()
