# 地球惑星物理学演習

これは東京大学理学部地球惑星物理学科3年生向けの演習科目
「[地球惑星物理学演習](https://chibutsu-utokyo.github.io/)」
のために用意されたLinuxコンテナ環境です．  
ブラウザとGithubアカウントがあればすぐに使うことができます．  
また各自のPCのVS CodeにRemote Developmentという拡張機能をインストールすることでVS Codeから直接接続することもできます．

## 起動方法
初めて起動するときは  
「Code」　→　「Codespaces」タブ　→　「Create codespace on main」  
をクリックするとブラウザ上でVSCodeが立ち上がります．  

<div align="center">
<img src="https://github.com/amanotk/codespaces/assets/46679145/2a9e5d35-086d-408f-b678-3cf6a1c8cbf1" width="400px">
</div>

初回は環境構築に少し時間がかかりますが，2回目以降は既存の環境を使うことができます．  
作成された環境には自動で名前（以下の例では"crispy bassoon"）が付与されますので，
その名前をクリックすれば起動することができます．

<div align="center">
<img src="https://github.com/amanotk/codespaces/assets/46679145/5a8ba104-a963-41b4-aa04-9a5fb48c97c5" width="400px">
</div>

## 注意点
これは
[Github Codespaces](https://docs.github.com/ja/codespaces/overview)
と呼ばれる開発環境（コンテナ）です．  
無料での利用には制限があり，コア数に応じて変わりますが，デフォルト（2コア使用）では
1ヶ月あたり60時間，ストレージは15GBまで使用ができます．
東大ECCSのメールアドレスを登録して
[Github Education](https://education.github.com/)
に申請すると1ヶ月あたり90時間，ストレージは20GBまで使用できるようになります．
いずれにしても演習での使用程度であれば特に問題にはならないと思います．

しばらく使われていないcodespaceは自動で削除されることになっています．削除される数日前にはGithubからメールが来ますので，必要であればその時に一度起動をしておいてください．

## codespace内の環境について
codespaceのVSCode上のターミナルでは，デフォルトでカレントディレクトリがワークスペース（リポジトリ）のルートディレクトリとなります．ただし，それ以外のディレクトリ（例えばホームディレクトリ）のファイルも操作することができます．codespace内のファイルやディレクトリに加えた変更はそのcodespaceを削除しない限りは保存されます．ホームディレクトリやワークスペースディレクトリは以下の通りです．
  
| 名前 | 環境変数名 | パス |
|---|---|---|
| ホーム | `HOME` | `/home/vscode` |
| ワークスペース | `CODESPACE_VSCODE_FOLDER` | `/workspaces/debian` |

