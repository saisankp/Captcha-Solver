import cv2
import argparse
import os


def process_image(path_of_image, path_to_write_new_image):
    originalImage = cv2.imread(path_of_image)
    gray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < 5:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
    result = 255 - thresh
    cv2.imwrite(path_to_write_new_image, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', help='Where the generated images are', type=str)
    parser.add_argument('--processed-image-folder', help='Where the processed images will be stored', type=str)
    args = parser.parse_args()

    if args.image_folder is None:
        print("Please specify where the generated images are located")
        exit(1)
    if args.processed_image_folder is None:
        print("Please specify where the processed images should be stored")
        exit(1)

    # If resulting directory doesn't exist already, make it
    if not os.path.exists(args.processed_image_folder):
        os.makedirs(args.processed_image_folder)

     # Iterate through every file in the specified image folder
    for filename in os.listdir(args.image_folder):
        file_path_of_initial_image = os.path.join(args.image_folder, filename)
        file_path_of_new_image = str(os.path.join(args.processed_image_folder, filename))
        process_image(file_path_of_initial_image, file_path_of_new_image)

if __name__ == '__main__':
    main()
