#!/bin/bash

docker build  -t ssh_container .
docker run -itd --rm -p 22222:22 --name ssh_container ssh_container

clear

echo "dockerコンテナの準備が完了しました。"
echo "'ssh ユーザー名@localhost -p 22222'でSSHログインができます。"
