# Parse arguments
overwrite_venv=
while (( $# )); do
    case $1 in
        --overwrite-venv)
            overwrite_venv=1
            ;;
        --*)
            echo 'bad option: '$1
            ;;
        *)
            echo $1' not parsed'
            ;;
    esac
    shift
done

# Set up environment variables
export PROJECT_ROOT=`git rev-parse --show-toplevel`

# Set up script variables
proj_name=ai-basics
venv_path=${PROJECT_ROOT}/${proj_name}_python3

# Create virtual environment
if [ $overwrite_venv ]; then
    rm -rf $venv_path
fi

if [ ! -d ${venv_path} ]; then
    virtualenv -p /usr/bin/python ${venv_path}
    source ${PROJECT_ROOT}/${proj_name}_python3/bin/activate
    python -m pip install -r ${PROJECT_ROOT}/requirements.txt
else
    source ${PROJECT_ROOT}/${proj_name}_python3/bin/activate
fi
