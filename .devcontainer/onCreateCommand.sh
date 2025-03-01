# debian packages
export DEBIAN_FRONTEND=noninteractive
apt-get update
xargs -a .devcontainer/debian-packages.txt apt-get install -y

# acl
chown -R vscode:vscode .
setfacl -bnR .
