# 線形回帰(Linear Regression)
## 概要
- 線形回帰とは以下の数式で表現される機械学習モデルである
- 単純な数式で表現できることから，データやモデルの解釈性が高い

![線形回帰](https://latex.codecogs.com/gif.latex?%5Chat%7By_i%7D%3Dw_0&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7Dx_iw_i)

## 線形回帰モデルの学習
線形回帰モデルで学習すべきパラメータは重みのwであり，一般にMean Squared Errorを最小化する様に学習を行う．  

![Mean Squared Error](https://latex.codecogs.com/gif.latex?E%28w%29%3D%5Cfrac%7B1%7D%7B2N%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5C%7B%5Chat%7By%7D_i-y_i%5C%7D%5E2)

与えられた入力![x](https://latex.codecogs.com/gif.latex?x)に対し，対応する目的変数![y](https://latex.codecogs.com/gif.latex?y)は，平均が![線形回帰](https://latex.codecogs.com/gif.latex?%5Chat%7By_i%7D%3Dw_0&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7Dx_iw_i)のガウス分布に従うとする(ノイズを分散が未知のガウス分布として扱う)．このとき，訓練データから線形回帰のパラメータを最尤推定したものとMSEを最小化することは等価となり，MSEを利用することの理論的な裏付けが得られる．  

## パラメータの最尤推定
前節で述べたパラメータの最尤推定について本節で示す．  

### ガウス分布

ガウス分布は以下の数式で表現される．  

![ガウス分布](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D%28x%20%7C%20%5Cmu%2C%20%5Csigma%5E2%29%3D%5Cfrac%7B1%7D%7B%282%5Cpi%5Csigma%5E2%29%5E%7B1/2%7D%7Dexp%5C%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%28x-%5Cmu%29%5E2%5C%7D)

- ![mu](https://latex.codecogs.com/gif.latex?%5Cmu) : 平均
- ![sigma](https://latex.codecogs.com/gif.latex?%5Csigma%5E2) : 分散

また分散![sigma](https://latex.codecogs.com/gif.latex?%5Csigma%5E2)の逆数を![precision parameter](https://latex.codecogs.com/gif.latex?%5Cbeta%3D1/%5Csigma%5E2)で表し，精度パラメータ(precision parameter)と呼ぶ．  

### 線形回帰の最尤推定
与えられた入力![x](https://latex.codecogs.com/gif.latex?x)に対し，対応する目的変数![y](https://latex.codecogs.com/gif.latex?y)は，平均が![線形回帰](https://latex.codecogs.com/gif.latex?%5Chat%7By_i%7D%3Dw_0&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7Dx_iw_i)のガウス分布に従う線形回帰モデルは以下のように表現される．  

![ガウス分布](https://latex.codecogs.com/gif.latex?p%28t%7Cx%2C%20%5Cbold%7Bw%7D%2C%20%5Cbeta%29%3D%5Cmathcal%7BN%7D%28t%20%7C%20y%28x%2C%5Cbold%7Bw%7D%29%2C%20%5Cbeta%5E%7B-1%7D%29)

この線形回帰モデルの未知のパラメータ![w](https://latex.codecogs.com/gif.latex?%5Cbold%7Bw%7D)と![beta](https://latex.codecogs.com/gif.latex?%5Cbeta)を訓練データ![訓練データ](https://latex.codecogs.com/gif.latex?%28%5Cbold%7Bx%7D%2C%5Cbold%7Bt%7D%29)を用いて最尤推定する．各データが上記確率分布から独立に取られたものと仮定すると尤度関数は以下の式で与えられる．  

![](https://latex.codecogs.com/gif.latex?p%28%5Cbold%7Bx%7D%20%7C%20%5Cbold%7Bt%7D%2C%20%5Cbold%7Bw%7D%2C%20%5Cbeta%29%3D%5Cprod_%7BN%7D%5E%7Bn%3D1%7D%5Cmathcal%7BN%7D%28t_n%20%7C%20y%28x_n%2C%20%5Cbold%7Bw%7D%29%2C%20%5Cbeta%5E%7B-1%7D%29)

尤度関数を最小化するかわりに尤度関数の対数を最大化することを考え，先ほどの尤度関数の対数を取ると，

![](https://latex.codecogs.com/gif.latex?lnp%28%5Cbold%7Bx%7D%20%7C%20%5Cbold%7Bt%7D%2C%20%5Cbold%7Bw%7D%2C%20%5Cbeta%29%3D-%5Cfrac%7B%5Cbeta%7D%7B2%7D%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5C%7By%28x_n%2C%5Cbold%7Bw%7D%29-t_n%5C%7D%5E2%20&plus;%20%5Cfrac%7BN%7D%7B2%7Dln%5Cbeta%20-%20%5Cfrac%7BN%7D%7B2%7Dln%282%5Cpi%29)

となる．線形回帰のパラメータ![](https://latex.codecogs.com/gif.latex?%5Cbold%7Bw%7D)を求める場合は上記式を最大化する．このとき第2項以降は無視することができるので，最尤推定による![](https://latex.codecogs.com/gif.latex?%5Cbold%7Bw%7D)の推定はノイズがガウス分布に従うという仮定のもとで，二乗和誤差の最小化と等価であるとみなせる．  
