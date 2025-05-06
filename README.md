# poke-envの使い方
以下のリポジトリの使い方をまとめています。
https://github.com/hsahovic/poke-env


## Installation
環境を汚さないようcondaを使用する。`annaconda`と`Node.js`を使用するのでそれぞれインストールしておくこと。特にはまりどころはないです。

1. python3.9の仮想環境を作成
```
$ conda create -n poke_env python=3.9
```
Enter or yで先に進んで仮想環境を作成。以下で起動。
```
$ conda activate poke_env
```
後で出てくる強化学習の環境は`poke-env`と`numpy`と`tensorflow`の依存関係が厳しいので注意すること。
`poke-env`は最新バージョン（0.9.0）だと`numpy`2.0.2以上で`tensorflow`との衝突が起きるので
0.8.3を使用する。
```
$ pip install numpy==1.26.4
$ pip install tensorflow==2.15.0
$ pip install poke_env==0.8.3
$ pip install keras-rl2==1.0.5
```

2. pokemon-showdownをインストール
対戦環境サーバのインストールを以下で実施。
```
$ git clone https://github.com/smogon/pokemon-showdown.git
$ cd pokemon-showdown
$ npm install
$ cp config/config-example.js config/config.js
$ node pokemon-showdown start --no-security
```

3. poke-envのインストール
以下でインストール。torchも必要なので注意。自分のWSL環境ではtorchのインストールに時間がかかりkillされてしまっていたので引数追加。
```
$ git clone https://github.com/hsahovic/poke-env.git
$ cd poke-env
$ git checkout 0.8.3
$ pip install .[dev]
$ pip install --no-cache-dir torch

```
[【python】Torch をインストールしようとすると[Killed]となりインストールに失敗するので回避方法を探る](https://a1026302.hatenablog.com/entry/2021/01/12/150748#google_vignette)

## Random Playerと自分でのバトルのさせ方
#### ローカル環境でのRandom Player起動
WSLを起動し、以下の手順でRandomPlayerを起動させる。この際の起動アカウントは事前にhttps://play.pokemonshowdown.com/で作成しておくこと。
```
$ source ~/.bashrc
$ conda activate poke_env
$ python3 poke-env/examples/connectiong_an_agent_to_showdown.py
```

実行後`"Init player"`だけが出て特にエラー出てなければ成功。ちなみに自分の環境だとスマホのテザリングでないと実行できないので注意すること。
wifi使用するとプロキシ関係で引っかかる（プロキシ使っていないのだが）

#### ブラウザ側の処理
https://play.pokemonshowdown.com/ にアクセスし、適当な名前でログインする。ログイン場所は右上の`choose name`。
その後左下`Find user`にてホスト環境での起動アカウントを検索する。
`Challenge battle`したあと、右上の`Hide`ボタンを押すと選択したユーザが出てくるので、そのユーザを選んで`Challenge`するとバトルが開始する。
バトル開始しても動かない場合はネットワークの問題を疑うこと。

# 強化学習実行方法
1. サーバを立てる。
```
$ node pokemon-showdown start --no-security
```
2. 強化学習script実行
```
 python rl_with_new_open_ai_gym_wrapper.py
```