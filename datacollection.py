import os
import requests
import csv
from io import StringIO
import argparse
import time

def parseArguments(parser):
    parser = argparse.ArgumentParser()
    argumentsAndDescriptions = {
        '--url': ('Width of the generated captcha images', str),
        '--output-directory': ('Height of the generated captcha images', str),
    }

    for argument, (description, argument_type) in argumentsAndDescriptions.items():
        parser.add_argument(argument, help=description, type=argument_type)

    arguments = parser.parse_args()

    for argument, (description, _) in argumentsAndDescriptions.items():
            if getattr(arguments, argument.replace("--", "").replace("-", "_")) is None:
                print("Error: Please specify {}".format(argument))
                exit(1)

    return arguments

def getCSV(csv_url):
    response = requests.get(csv_url)
    if response.status_code == 200:
         csv_reader = csv.reader(StringIO(response.text))
         return csv_reader
    else:
         print('Error: Invalid response from API call to get CSV: {0}'.format(response))

def getImages(base_url, csv, captcha_directory):
    for row in csv:
        filename = row[0]
        response = requests.get('{0}&myfilename={1}'.format(base_url, filename))
        if response.status_code == 200:
            image_path = os.path.join(captcha_directory, filename)
            with open(image_path, 'wb') as image:
                image.write(response.content)
                print('Stored image {0}'.format(filename))
        else:
             print('Error: Invalid response from API call to get image: {0}'.format(response))


def main():
    start_time = time.time()
     # Parse arguments and exit if any are msising
    arguments = parseArguments(argparse.ArgumentParser())

    # Firstly, make the output directory for the images if it doesn't exist
    if not os.path.exists(arguments.output_directory):
        os.makedirs(arguments.output_directory)

    # Secondly, we need to get the CSV
    csv = getCSV(arguments.url)

    # Thirdly, we need to store the images from each filename in the csv into the "images" directory
    getImages(arguments.url, csv, arguments.output_directory)
    end_time = time.time()
    time_taken = end_time - start_time

    with open('time_taken.txt', 'w') as file:
        file.write('Data collection: {:.2f} seconds'.format(time_taken))
        print('Time taken:', time_taken, 'seconds')

if __name__ == '__main__':
    main()