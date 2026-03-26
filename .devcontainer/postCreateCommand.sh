# some setup
WORKSPACE=/workspaces/$RepositoryName

# directory permission
find . -type d -print | xargs chmod 755

# python modules
sudo /usr/local/bin/uv pip sync --system .devcontainer/python-packages.txt
python3 -m gnuplot_kernel install --user

cd $WORKSPACE
if [ ! -d "fortran" ]; then
    mkdir fortran && cd fortran
    git clone https://github.com/amanotk/fortran-resume-sample.git sample
    git clone https://github.com/amanotk/fortran-resume-answer.git answer
fi

# AI coding assistants (optional)
# Run .devcontainer/install-all-agents.sh to install all AI agents

# Ensure ~/.local/bin is in PATH
export PATH="$HOME/.local/bin:$PATH"
