# NOTE: Make sure to open this file in WSL i.e. Windows Subsystem for Linux (not command prompt/git bash)

# Activate the environment (use .\windowsEnv\Scripts\activate.bat on command prompt if you insist on using that)
source windowsEnv/Scripts/activate

# 1. Data collection from server, run "ls | wc -l" in folders to find amount of files downloaded, it should return 2000 (EXPERIMENT CONCLUSION: RUN USING RASPBERRY PI FOR THE BEST TIME)
python datacollection.py --url https://cs7ns1.scss.tcd.ie/?shortname=saisankp --output-directory saisankp-images
python datacollection.py --url https://cs7ns1.scss.tcd.ie/?shortname=hakhan --output-directory hakhan-images

# 2. Generate captchas (EXPERIMENT CONCLUSION: RUN USING WSL FOR THE BEST TIME)
python generation.py --width 128 --height 64 --min-length 1 --max-length 6 --symbols symbols_for_generation.txt --count 140800 --output-dir generated-images --font EamonU

# 3. Process the generated captchas (EXPERIMENT CONCLUSION: RUN USING RUN USING WSL FOR THE BEST TIME)
python process-images.py --image-folder generated-images/training/images --processed-image-folder processed-training-images
python process-images.py --image-folder generated-images/validation/images --processed-image-folder processed-validation-images

# 3. Train our model with the generated images (EXPERIMENT CONCLUSION: RUN USING WSL FOR ENOUGH TIME TO TRAIN THE MODEL)
python train.py --width 128 --height 64 --length 6 --batch-size 32 --train-dataset processed-training-images --validate-dataset processed-validation-images --output-model-name model.h5 --symbols symbols.txt --epochs 8

# 4. Classify the captchas to get our Submitty file (EXPERIMENT CONCLUSION: RUN USING RASPBERRY PI FOR THE BEST TIME)
python classify.py --model-name model --captcha-dir saisankp-images --output saisankp.csv --symbols symbols.txt
python classify.py --model-name model --captcha-dir hakhan-images --output hakhan.csv --symbols symbols.txt
