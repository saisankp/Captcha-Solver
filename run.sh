# Activate environment and variables
source env/bin/activate
source .environment

# 1. Data collection from server, run "ls | wc -l" in folders to find amount of files downloaded, it should return 2000 (RUN USING RASPBERRY PI)
python datacollection.py --url https://cs7ns1.scss.tcd.ie/?shortname=saisankp --output-directory saisankp-images
python datacollection.py --url https://cs7ns1.scss.tcd.ie/?shortname=hakhan --output-directory hakhan-images

# 2. Generate captchas (RUN USING WSL)
python generation.py --width 128 --height 64 --min-length 1 --max-length 6 --symbols symbols_for_generation.txt --count 140800 --output-dir generated-images --font EamonU

# 3. Process the generated captchas (RUN USING WSL)
python process-images.py --image-folder generated-images/training/images --processed-image-folder processed-training-images
python process-images.py --image-folder generated-images/validation/images --processed-image-folder processed-validation-images

# 3. Train our model with the generated images (RUN USING WSL)
python train.py --width 128 --height 64 --length 6 --batch-size 32 --train-dataset processed-training-images --validate-dataset processed-validation-images --output-model-name model.h5 --symbols symbols.txt --epochs 5

# 4. Classify the captchas to get our Submitty file (RUN USING RASPBERRY PI)
python classify.py --model-name model --captcha-dir saisankp-images --output saisankp.csv --symbols symbols.txt
python classify.py --model-name model --captcha-dir hakhan-images --output hakhan.csv --symbols symbols.txt
