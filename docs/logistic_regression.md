# ロジスティック回帰
## 概要
- 以下の式で表される線形識別関数の関数値を区間(0, 1)に制限したもの．
```math
y = \textbf{w}^T\textbf{x}
```
- 出力を区間(0, 1)に制限することで2クラス分類において確率的な解釈が可能になった．
- 線形識別関数の出力にロジスティック関数(ロジスティック・シグモイド関数)を適用することで出力を区間(0, 1)に制限したものがロジスティック回帰．

## ロジスティック回帰モデル
ロジスティック回帰は以下の式で表せる．  

```math
f(\textbf{x}) = \frac{1}{1+exp(-\textbf{w}^T\textbf{x})}
```

線形関数を非線形変換しても識別境界は超平面となるため，ロジスティック回帰モデルは線形識別関数に分類される．一般化線形モデル(generalized linear model)と呼ばれるモデルのうちの1つである．  

## パラメータ推定
モデルの出力を確率変数tで表し，tが1となる確率をP(t=1)=y，tが0となる確率をP(t=0)=1-yで表す．このとき確率変数tはパラメータyを持つベルヌーイ試行に従う．  

```math
f(t|y) = y^t(1-y)^{1-t}
```

N回の試行に基づく尤度関数は，

```math
L(y_1,...,y_N)=\prod_{i=1}^{N}f(t_i|y_i)=\prod_{i=1}^{N}y_i^{t_i}(1-y_i)^{(1-t_i)}
```

となる．また負の対数尤度は，  

```math
-lnL(y_1,...,y_N)=-\sum_{i=1}^{N}(t_ilny_i+(1-t_i)ln(1-y_i))
```

となり，これは交差エントロピー誤差(cross entropy error)と呼ばれる．  

```math
Cross Entropy Error = -\sum_{i=1}^{N}(t_ilny_i+(1-t_i)ln(1-y_i))
```

ここでy=wxより，

```math
-lnL(y_1,...,y_N) = -lnL(\textbf{w}) = -\sum_{i=1}^{N}(t_i\textbf{w}^T \textbf{x}_i-ln(1+exp(\textbf{w}^T\textbf{x}_i)))
```

最尤推定ではこの誤差を最小化するようにパラメータwを求める．負の対数尤度をwで微分すると以下の式のようになる．  

```math
\frac{\partial L(\textbf{w})}{\partial \textbf{w}} = -\sum_{i=1}^{N}(t_i \textbf{x}_i - \frac{\textbf{x}_i exp(\textbf{w}^T\textbf{x}_i)}{1+exp(\textbf{w}^T\textbf{x}_i)}) = \sum_{i=1}^{N}\textbf{x}_i(y_i-t_i)
```

この式が0となるwが解となる．  
解析的に解を求めることができないため，最急降下法などで解を求める．