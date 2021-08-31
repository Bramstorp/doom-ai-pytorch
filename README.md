# doom-ai-pytorch
doom ai pytorch

## <a name="deps"></a> Dependencies

Even if you plan to install ViZDoom via PyPI or LuaRocks, you need to install some dependencies in your system first.


### <a name="linux_deps"></a> Linux
* CMake 3.1+
* Make
* GCC 6.0+
* Boost libraries 1.65.0+
* Python 3.5+ with Numpy for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

To get all dependencies on Ubuntu (we recommend using Ubuntu 18.04+) execute the following commands in the shell (requires root access). `scripts/linux_check_dependencies.sh` installs these for Python3:
```bash
# ZDoom dependencies
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 3 dependencies
sudo apt-get install python3-dev python3-pip
pip3 install numpy
# or install Anaconda 3 and add it to PATH

# Julia dependencies
sudo apt-get install julia
julia
julia> using Pkg
julia> Pkg.add("CxxWrap")
```


