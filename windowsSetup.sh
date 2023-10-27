# NOTE: Make sure to open this file in WSL i.e. Windows Subsystem for Linux (not command prompt/git bash)

python -m pip install virtualenv 

# Activate the environment (use .\windowsEnv\Scripts\activate.bat on command prompt if you insist on using that)
source windowsEnv/Scripts/activate

# Install pip for python
mkdir -p pip
cd pip
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py -O get-pip.py
python get-pip.py
cd ..

# Install requests with pip
python -m pip install requests

# Install numpy with pip
python -m pip install numpy

# Install captcha with pip
python -m pip install captcha

# Install tqdm with pip
python -m pip install tqdm

# Install OpenCV with pip (we need 4.2.0.32 specifically because of python 2.7.18 support in OpenCV, reference: https://stackoverflow.com/questions/63346648/python-2-7-installing-opencv-via-pip-virtual-environment)
python -m pip install opencv-python==4.2.0.32

# We download a wheel specific for 64bit python 2.7.18 on Windows (reference: https://github.com/fo40225/tensorflow-windows-wheel/tree/master)
mkdir -p tensorflow
cd tensorflow
wget https://github.com/fo40225/tensorflow-windows-wheel/raw/master/1.5.0/py27/CPU/avx2/tensorflow-1.5.0-cp27-cp27m-win_amd64.whl -O tensorflow-1.5.0-cp27-cp27m-win_amd64.whl
python -m pip install tensorflow-1.5.0-cp27-cp27m-win_amd64.whl
cd ..

# Now, everything is installed :)
