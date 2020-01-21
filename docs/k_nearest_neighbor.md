# k近傍法

## 概要

簡単に説明するとk近傍法というのはその名の通り，あるデータに着目した時に近隣k個のデータの平均もしくは多数決の値を予測として出力するアルゴリズム．  

以下の図のように赤と青の2つのグループがあり，緑の点がどちらのグループに属しているのかを予測したい場合を考える．  

まずはk=3の場合を考える．緑の点から緑の点自身を除く3点が入るように円を描く(図1中の内側の円)．このとき赤の点が2つ，青の点が1つとなり，多数決の結果，予測値は赤となる．  
次にk=5の場合を考えると(図1中の外側の円)，赤の点が4つ，青の点が1つとなり，k近傍法では赤と予測する．

k近傍法のkは人間が決めるべきハイパーパラメータであり，一般的には交差検証法などを用いて，精度が高くかつノイズに強くなる値を決める．  

## k近傍法のメリット・デメリット

### メリット
- 精度が悪くない
- 非常に直感的でブラックボックス的な予測ではない

### デメリット
- 毎回学習データの数だけ距離の計算を行うため計算量が大きい
- 入力データの次元数が大きくなると距離の扱いが難しくなる