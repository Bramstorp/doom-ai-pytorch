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

### <a name="macos_deps"></a> MacOS
* CMake 3.1+
* Clang 5.0+
* Boost libraries 1.65.0+
* Python 3.5+ with Numpy for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Mac_OS_X) are needed.

To get dependencies install [homebrew](https://brew.sh/)

```sh
# ZDoom dependencies and Boost libraries
brew install cmake boost openal-soft sdl2

# Python 3 dependencies
brew install python3
pip3 install numpy
# or install Anaconda 3 and add it to PATH

# Julia dependencies
brew cask install julia
julia
julia> using Pkg
julia> Pkg.add("CxxWrap")
```

### <a name="windows_deps"></a> Windows
* CMake 3.1+
* Visual Studio 2012+
* Boost 1.65+
* Python 3.5+ with Numpy for Python binding (optional)

Additionally, [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Windows) are needed.
Most of them are gathered in this repository: [ViZDoomWinDepBin](https://github.com/mwydmuch/ViZDoomWinDepBin).


## <a name="pypi"></a> Installation via PyPI (recommended for Python users)

ViZDoom for Python can be installed via **pip/conda** on Linux and MacOS, and it is strongly recommended.
However you will still need to install **[Linux](#linux_deps)/[MacOS](#macos_deps) dependencies**.

> Pip installation is not supported on Windows at the moment, but we hope some day it will.

To install the most stable official release from [PyPI](https://pypi.python.org/pypi):
```bash
pip install vizdoom
```

To install the newest version from the repository:
```bash
pip install git+https://github.com/mwydmuch/ViZDoom.git
```


## <a name="windows_bin"></a> Installation of Windows binaries

For Windows we are providing a compiled environment that can be download from [releases](https://github.com/mwydmuch/ViZDoom/releases) page.
To install it for Python, copy files to `site-packages` folder.

Location of `site-packages` depends on Python distribution:
- Python: `python_root\Lib\site-packges`
- Anaconda: `anaconda_root\lib\pythonX.X\site-packages`


