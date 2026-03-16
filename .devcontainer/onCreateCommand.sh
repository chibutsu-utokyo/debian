# debian packages
export DEBIAN_FRONTEND=noninteractive
apt-get update
xargs -a .devcontainer/debian-packages.txt apt-get install -y

# uv
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/0.10.9/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh
fi

# acl
chown -R vscode:vscode .
setfacl -bnR .
