# Preparing for running our code
source env/bin/activate
source .environment

# 1. Data collection from server (run "ls | wc -l" in /individual-image to find amount of files downloaded, it should return 2000)
python datacollection.py --url https://cs7ns1.scss.tcd.ie/?shortname=saisankp --output-directory individual-images

#Cross checking from someone elses dataset
python datacollection.py --url https://cs7ns1.scss.tcd.ie/?shortname=cheny28 --output-directory cross-check
python datacollection.py --url https://cs7ns1.scss.tcd.ie/?shortname=corralp --output-directory cross-check2

# 2. Training data generation
python generation.py --width 128 --height 64 --min-length 1 --max-length 6 --symbols symbols.txt --count 140800 --output-dir generated-images --font EamonU

# Todo more...
