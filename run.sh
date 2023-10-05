# Preparing for running our code
source env/bin/activate
source .xsessionrc

# 1. Data collection from server
python datacollection.py --url https://cs7ns1.scss.tcd.ie/?shortname=saisankp --output-directory individual-images

# 2. Training data generation
python generation.py --width 128 --height 64 --length 5 --symbols symbols.txt --count 140800 --output-dir generated-images

# Todo more...
