# some setup
WORKSPACE=/workspaces/$RepositoryName

# directory permission
find . -type d -print | xargs chmod 755

# python modules
python3 -m pip install -r .devcontainer/python-packages.txt
python3 -m gnuplot_kernel install --user

cd $WORKSPACE
if [ ! -d "fortran" ]; then
    mkdir fortran && cd fortran
    git clone https://github.com/amanotk/fortran-resume-sample.git sample
    git clone https://github.com/amanotk/fortran-resume-answer.git answer
fi

#cd $WORKSPACE
#if [ ! -d "python" ]; then
#    mkdir python && cd python
#    git clone https://github.com/amanotk/python-resume-sample.git sample
#    git clone https://github.com/amanotk/python-resume-answer.git answer
#fi

