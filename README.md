# hackathon-global_motion_estimation
Repository for global motion estimation implementation

## Steps to Build and Run
Workspace setup on WSL ubuntu

### Essesntial Package installation
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential 
sudo apt-get install cmake 
sudo apt-get install git 
sudo apt-get install unzip 
sudo apt-get install pkg-config
sudo apt install libgtk2.0-dev
```

### Build and install OpenCV and configure pkg-config
```
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_BUILD_TYPE=Release -DWITH_GTK=ON -DCMAKE_INSTALL_PREFIX=/usr/local ../opencv
cmake --build .
sudo make install
sudo ldconfig

PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
```



