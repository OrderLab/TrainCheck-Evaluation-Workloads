# Please copy this file to the root directory of the pipeline
# and complete the TODO part to fit the pipeline's installation process.

# Exit immediately if a command exits with a non-zero status
set -e

third_party_libs_dir=$(grep -oP '(?<=third_party_libs_dir=).+' machine-learning-issues/INSTALL_CONFIG)

# print the third_party_libs_dir
echo "third party libraries would be installed at: $third_party_libs_dir"
# create the third_party_libs_dir if it does not exist
if [ ! -d ${third_party_libs_dir} ]; then
    mkdir -p ${third_party_libs_dir}
fi

pkg_name="lightning-thunder"
venv_name="lightning-thunder"
repo_url="https://github.com/Lightning-AI/lightning-thunder.git"
buggy_commit="c816506"
# buggy_pr="XXX"
# fixed_commit="bdb59d7"
fixed_pr="810"

# TODO: change the buggy_commit/buggy_pr, fixed_commit/fixed_pr to the correct values

# check directory

current_dir=$(pwd)

# check if the current dir contains traincheck
if [ -d "traincheck" ]; then
    echo "install.sh executed in the traincheck directory"
else
    echo "install.sh executed in the wrong directory, please execute it in the traincheck directory"
    exit 1
fi

run_cmd_in_conda_env() {
    env_name=$1
    cmd=$2
    bash -c "source ~/miniconda3/bin/activate ${env_name}; ${cmd}"
}


install() {
    conda create --name ${venv_name} python==3.10
    echo "virtual environment ${venv_name} created"


    ###########################################
    
    # TODO: Customized installation logic here

    # 1) Install nvFuser and PyTorch dependencies:
    python -m pip install --pre nvfuser-cu121-torch24
    
    ###########################################

    # install from source
    # if the third_party_libs_dir does not exist, create it
    if [ ! -d ${third_party_libs_dir} ]; then
        mkdir -p ${third_party_libs_dir}
    fi
    cd ${third_party_libs_dir}

    git clone ${repo_url}
    cd ${pkg_name}
    run_cmd_in_conda_env $venv_name "python -m pip install -e ."
    cd ${current_dir}
}

build_buggy(){
    check_installation
    cd ${third_party_libs_dir}
    cd ${pkg_name}
    # if have defined the buggy version, checkout to the buggy version
    if [ -n "$buggy_commit" ]; then
        git checkout ${buggy_commit}
    elif [ -n "$buggy_pr" ]; then
        git fetch origin pull/${buggy_pr}/head:buggy_pr_branch
        git checkout buggy_pr_branch
    fi

    cd ${current_dir}

    # TODO: add customization logic below
}

build_fixed(){
    check_installation
    cd ${third_party_libs_dir}
    cd ${pkg_name}
    # if have defined the fixed version, checkout to the fixed version
    if [ -n "$fixed_commit" ]; then
        git checkout ${fixed_commit}
    elif [ -n "$fixed_pr" ]; then
        git fetch origin pull/${fixed_pr}/head:fixed_pr_branch
        git checkout fixed_pr_branch
    fi

    cd ${current_dir}

    # TODO: add customization logic below
}

check_installation() {
    # check if the virtual environment exists
    if conda env list | grep -q ${venv_name}; then
        echo "virtual environment ${venv_name} exists"
    else
        echo "virtual environment ${venv_name} does not exist, begin installation"
        install
    fi
    cd ${third_party_libs_dir}
    # check if the cloned repo exists
    if [ -d ${pkg_name} ]; then
        echo "repo ${pkg_name} exists"
    else
        echo "repo ${pkg_name} does not exist, begin installation"
        uninstall
        install
    fi
    cd ${current_dir}
}

uninstall() {
    if conda env list | grep -q ${venv_name}; then
        conda remove --name ${venv_name} --all
        echo "virtual environment ${venv_name} removed"
    fi
    cd ${current_dir}
    cd ${third_party_libs_dir}
    # remove the cloned repo if it exists
    if [ -d ${pkg_name} ]; then
        rm -rf ${pkg_name}
    fi
    cd ${current_dir}
}

# check out the argument passed, could be "install", "uninstall", "build_buggy", and "build_fixed"

if [ "$1" = "install" ]; then
    echo "installing the dependencies for the pipeline"
    install
elif [ "$1" = "uninstall" ]; then
    echo "uninstalling the dependencies for the pipeline"
    uninstall
elif [ "$1" = "build_buggy" ]; then
    echo "building the buggy version of the pipeline"
    build_buggy
elif [ "$1" = "build_fixed" ]; then
    echo "building the fixed version of the pipeline"
    build_fixed
else
    echo "invalid argument, please use 'install', 'uninstall', 'build_buggy', or 'build_fixed'"
    exit 1
fi

cd ${current_dir}