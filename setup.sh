# NOTE: Make sure to open this file after ssh-ing into the Raspberry Pi

# Since we are on a Raspberry Pi (ARMV7 32bit), installing the packages we need is extremely tricky.
# This relates to our major learning in this module which is that IoT devices are extremely limited!
# However, our other major learning in this module is that there are always workarounds to everything, so this script shows how we did it :)

# This Raspberry Pi comes installed with Python 2.7.18. After many sleepless nights of trialing Python 3.5, 3.6, 3.7 and 3.8, keeping the original Python version 2.7.18 is best as it's compatible for using Tensorflow AND OpenCV on ARMV7 without errors with the quickest "import".
# Furthermore, installing other Python 3 versions often require sudo commands (a permission we DON'T have) to be in the /usr/lib directory (technically it can be installed in other directories but strange errors show up later such as missing header files, it's best to stay clear!).
# Here is how we installed the packages:

python -m pip install virtualenv 

# Activate the environment
source env/bin/activate

# Install pip for python
mkdir -p pip
cd pip
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
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

# Install cython version 0.29.36 with pip (through painful trial and error I found that you need this version in particular for installing Tensorflow without errors later)
python -m pip install Cython==0.29.36

# Install h5py with pip (we need this to avoid errors when installing Tensorflow later)
python -m pip install h5py

# Install Tensorflow from a wheel specific for Python 2.7 and ARMV7 (reference: https://github.com/PINTO0309/Tensorflow-bin/tree/main)
# We install via a wheel without pip as we are restricted with compatability for ARMV7 architectures via pip
# Furthermore, I went for Tensorflow 1 compared to Tensorflow 2 as my online research showed that they are both similar speeds but Tensorflow 1 is usually faster (reference: https://stackoverflow.com/a/58653636/15590483)
mkdir -p tensorflow
cd tensorflow
wget https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/main/previous_versions/download_tensorflow-1.15.0-cp27-cp27mu-linux_armv7l.sh
chmod +x download_tensorflow-1.15.0-cp27-cp27mu-linux_armv7l.sh
./download_tensorflow-1.15.0-cp27-cp27mu-linux_armv7l.sh
python -m pip install tensorflow-1.15.0-cp27-cp27mu-linux_armv7l.whl
cd ..

# Install OpenCV from source
# We install from source without pip because while we do not have permissions for "sudo apt get", OpenCV requires various packages such as libgtqui4, libjasper, libjpeg64, ilmbase, openexr etc for image processing
mkdir -p opencv
cd opencv
git clone https://github.com/opencv/opencv.git
git -C opencv checkout 3.4.0
# The command below is to fix a bug with cmake (reference: https://stackoverflow.com/a/67428275/15590483)
sed -i 's/char\* str = PyString_AsString(obj);/const char* str = PyString_AsString(obj);/g' opencv/modules/python/src2/cv2.cpp
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ../opencv
make
make install
cd ../..

# Activate environment variables
# Now that OpenCV is installed from source, we can add "$HOME/.local/" to the "LD_LIBRARY_PATH" environment variable in our shell's session environment file (this is already done in .environment)
# Since we don't have sudo commands, we can't do "sudo ln -s /usr/local/lib64/pkgconfig/opencv.pc /usr/share/pkgconfig/" to symlink this OpenCV installation to our system
# We can overcome this by adding an environment variable "PKG_CONFIG_PATH" in our shell's session environment (this is already done in .environment)
source .environment

# Allow python 2.7 to find our OpenCV installation
# Now that we have OpenCV installed and the .pc file can be found, we can add the .so file to our environment
# Alternatively, we can just do "import sys" and "sys.path.append('path/to/$HOME/.local/lib/python2.7.site-packages')" every time before we "import cv2" but using the .so file saves us from that.
opencv_path=$(find $HOME/.local -name "cv2.so")
cp "$opencv_path" env/lib/python2.7/site-packages/

# Now, everything is installed :)
