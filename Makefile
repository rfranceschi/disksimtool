LOCAL_FOLDER = $(shell pwd)
PARENT_FOLDER = $(shell dirname $(LOCAL_FOLDER))

# Variables for optool
OPTOOL_REPO_URL = https://github.com/cdominik/optool
OPTOOL_CLONE_DIR = $(PARENT_FOLDER)/optool
OPTOOL_BIN = $(shell command -v optool 2>/dev/null)

# Variables for radmc3d
RADMC3D_REPO_URL = https://github.com/dullemond/radmc3d-2.0.git
RADMC3D_CLONE_DIR = $(PARENT_FOLDER)/radmc3d
RADMC3D_BIN = $(shell command -v radmc3d 2>/dev/null)

# Default target
all: check_optool check_radmc3d install_python_package

# Target to check if optool is installed
check_optool:
ifeq ($(OPTOOL_BIN),)
    @echo "optool is not installed. Cloning and installing..."
    $(MAKE) install_optool
else
    @echo "optool is already installed."
endif

# Target to check if radmc3d is installed
check_radmc3d:
ifeq ($(RADMC3D_BIN),)
    @echo "radmc3d is not installed. Cloning and installing..."
    $(MAKE) install_radmc3d
else
    @echo "radmc3d is already installed."
endif

# Target to clone and install optool
install_optool: clone_optool build_optool install_optool_bin
    @echo "optool installed successfully."

# Target to clone and install radmc3d
install_radmc3d: clone_radmc3d build_radmc3d
    @echo "radmc3d installed successfully."

# Target to clone the optool repository
clone_optool:
    @if [ ! -d $(OPTOOL_CLONE_DIR) ]; then \
        git clone $(OPTOOL_REPO_URL) $(OPTOOL_CLONE_DIR); \
    else \
        echo "optool repository already cloned."; \
    fi

# Target to build the optool repository
build_optool:
    @cd $(OPTOOL_CLONE_DIR) && \
    make

# Target to install optool binaries
install_optool_bin:
    @cd $(OPTOOL_CLONE_DIR) && \
    make install bindir=~/bin/ && \
    pip install -e .

# Target to clone the radmc3d repository
clone_radmc3d:
    @if [ ! -d $(RADMC3D_CLONE_DIR) ]; then \
        git clone $(RADMC3D_REPO_URL) $(RADMC3D_CLONE_DIR); \
    else \
        echo "radmc3d repository already cloned."; \
    fi

# Target to build the radmc3d repository
build_radmc3d:
    @cd $(RADMC3D_CLONE_DIR) && \
    make install && \
    echo "export PATH=~/bin:$$PATH" >> ~/.bashrc && \
    echo "Please restart your terminal or run 'source ~/.bashrc' to update your PATH."

install_python_package:
	@echo "Installing Python package..."
	@pip install -e .

.PHONY: all check_optool check_radmc3d install_optool clone_optool build_optool install_optool_bin clone_radmc3d build_radmc3d install_python_package
