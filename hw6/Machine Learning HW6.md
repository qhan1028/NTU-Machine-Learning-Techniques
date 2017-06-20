# Machine Learning HW6

B03902089 資工三 林良翰

### Descent Methods for Probabilistic SVM

* Probabilistic SVM: $\min\limits _{ A,B }{ \frac { 1 }{ N } \sum\limits _{ n=1 }^{ N }{ \ln { \left( 1+\exp\left( -{ y }_{ n }\left( A \cdot \left( { w }_{ SVM }^{ T }\phi \left( { x }_{ n } \right) +{ b }_{ SVM } \right) +B \right)  \right)  \right)  }  }  }$

* Let: 

  ${ z }_{ n }={ w }_{ SVM }^{ T }\phi \left( { x }_{ n } \right) +{ b }_{ SVM }$

  ${ p }_{ n }=\theta \left( -{ y }_{ n }\left( A{ z }_{ n }+B \right)  \right)$, where $\theta \left( s \right) =\frac { \exp\left( s \right)  }{ 1+\exp\left( s \right)  }$

#### 1.

* Let ${ s }_{ n }=-{ y }_{ n }\big( A{ z }_{ n }+B \big)$

  $\Rightarrow F\big( A,B \big) =\frac { 1 }{ N } \sum\limits _{ n=1 }^{ N }{ \ln\big( 1+\exp\left( { s }_{ n } \right)  \big)  }$

* $\frac { \partial F\left( A,B \right)  }{ \partial A } =\frac { 1 }{ N } \sum\limits _{ n=1 }^{ N }{ \frac { 1 }{ 1+\exp\left( { s }_{ n } \right)  }  } \exp\left( { s }_{ n } \right) \frac { \partial { s }_{ n } }{ \partial A } =\frac { 1 }{ N } \sum\limits _{ n=1 }^{ N }{ -{ p }_{ n }{ y }_{ n }{ z }_{ n } }$

  $\frac { \partial F\left( A,B \right)  }{ \partial B } =\frac { 1 }{ N } \sum\limits _{ n=1 }^{ N }{ \frac { 1 }{ 1+\exp\left( { s }_{ n } \right)  }  } \exp\left( { s }_{ n } \right) \frac { \partial { s }_{ n } }{ \partial B } =\frac { 1 }{ N } \sum\limits _{ n=1 }^{ N }{ -{ p }_{ n }{ y }_{ n } }$

* $∇F\left( A,B \right) =\frac { 1 }{ N } \sum\limits _{ n=1 }^{ N }{ { \left[ -{ p }_{ n }{ y }_{ n }{ z }_{ n },-{ p }_{ n }{ y }_{ n } \right]  }^{ T } }$

#### 2.
* Definition of Hessian Matrix $H \big( f \big)$ of $f\big( { x }_{ 1 },{ x }_{ 2 },\dots ,{ x }_{ n } \big)$

  $H\big( f \big) =\begin{bmatrix} \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ 1 }^{ 2 } }  & \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ 1 }{ x }_{ 2 } }  & \cdots  & \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ 1 }{ x }_{ n } }  \\ \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ 2 }{ x }_{ 1 } }  & \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ 2 }^{ 2 } }  & \cdots  & \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ 2 }{ x }_{ n } }  \\ \vdots  & \vdots  & \ddots  & \vdots  \\ \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ n }{ x }_{ 1 } }  & \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ n }{ x }_{ 2 } }  & \cdots  & \frac { { \partial  }^{ 2 }f }{ \partial { x }_{ n }^{ 2 } }  \end{bmatrix}$

* Let ${ s }_{ n }=-{ y }_{ n }\big( A{ z }_{ n }+B \big)$

  $\begin{eqnarray} \frac { \partial -{ p }_{ n }{ y }_{ n }{ z }_{ n } }{ \partial A }  & = & \frac { \partial -\theta \left( \partial { s }_{ n } \right) { y }_{ n }{ z }_{ n } }{ \partial { s }_{ n } } \frac { \partial { s }_{ n } }{ \partial A }  \\  & = & \frac { { e }^{ { s }_{ n } } }{ 1+{ e }^{ { s }_{ n } } } \left( 1-\frac { { e }^{ { s }_{ n } } }{ 1+{ e }^{ { s }_{ n } } }  \right) { \left( { y }_{ n }{ z }_{ n } \right)  }^{ 2 } \\  & = & { \left( { y }_{ n }{ z }_{ n } \right)  }^{ 2 }{ p }_{ n }\left( 1-{ p }_{ n } \right)  \end{eqnarray}$

  $\begin{eqnarray} \frac { \partial -{ p }_{ n }{ y }_{ n }{ z }_{ n } }{ \partial B }  & = & \frac { \partial -\theta \left( \partial { s }_{ n } \right) { y }_{ n }{ z }_{ n } }{ \partial { s }_{ n } } \frac { \partial { s }_{ n } }{ \partial B }  \\  & = & \frac { { e }^{ { s }_{ n } } }{ 1+{ e }^{ { s }_{ n } } } \left( 1-\frac { { e }^{ { s }_{ n } } }{ 1+{ e }^{ { s }_{ n } } }  \right) { \left( { { y }_{ n }^{ 2 }z }_{ n } \right)  } \\  & = & \left( { { y }_{ n }^{ 2 }z }_{ n } \right) { p }_{ n }\left( 1-{ p }_{ n } \right)  \end{eqnarray}$

  $\begin{eqnarray} \frac { \partial -{ p }_{ n }{ y }_{ n } }{ \partial B }  & = & \frac { \partial -\theta \left( \partial { s }_{ n } \right) { y }_{ n } }{ \partial { s }_{ n } } \frac { \partial { s }_{ n } }{ \partial B }  \\  & = & \frac { { e }^{ { s }_{ n } } }{ 1+{ e }^{ { s }_{ n } } } \left( 1-\frac { { e }^{ { s }_{ n } } }{ 1+{ e }^{ { s }_{ n } } }  \right) { { \left( { y }_{ n } \right)  }^{ 2 } } \\  & = & { \left( { y }_{ n } \right)  }^{ 2 }{ p }_{ n }\left( 1-{ p }_{ n } \right)  \end{eqnarray}$

* $H(F(A,B))=\frac { 1 }{ N } \sum _{ n=1 }^{ N }{ \begin{bmatrix} { \left( { y }_{ n }{ z }_{ n } \right)  }^{ 2 }{ p }_{ n }\left( 1-{ p }_{ n } \right)  & \left( { { y }_{ n }^{ 2 }z }_{ n } \right) { p }_{ n }\left( 1-{ p }_{ n } \right)  \\ \left( { { y }_{ n }^{ 2 }z }_{ n } \right) { p }_{ n }\left( 1-{ p }_{ n } \right)  & { { { \left( { y }_{ n } \right)  }^{ 2 } }p }_{ n }\left( 1-{ p }_{ n } \right)  \end{bmatrix} }$

### Kernel Ridge Regression

- Guassian Kernel $K\left( x,{ x }^{ \prime  } \right) =\exp\left( -\gamma { \left\| x-{ x }^{ \prime  } \right\|  }^{ 2 } \right) $
- Kernel Ridge Regression:
  - Want to Minimize: $\min\limits _{ \beta  }{ { E }_{ aug }\left( \beta  \right)  } =\min\limits _{ \beta  }{ \frac { \lambda  }{ N } { \beta  }^{ T }K\beta +\frac { 1 }{ N } \left( { \beta  }^{ T }{ K }^{ T }K\beta -2{ \beta  }^{ T }{ K }^{ T }y+{ y }^{ T }y \right)  }$
  - Solving: $∇{ E }_{ aug }\left( \beta  \right) =\frac { 2 }{ N } { K }^{ T }\left( \left( \lambda I+K \right) \beta -y \right) =0$
  - Obtain: $\beta ={ \left( \lambda I+K \right)  }^{ -1 }y$

#### 3.

* $\gamma \rightarrow \infty$ 
* $\lim\limits _{ \gamma \rightarrow \infty  }{ K\left( x,{ x }^{ \prime  } \right)  } =\lim\limits _{ \gamma \rightarrow \infty  }{ { e }^{ -\gamma { \left\| x-{ x }^{ \prime  } \right\|  }^{ 2 } } } =I$
* $\beta ={ \left( \lambda I + I \right)  }^{ -1 }y$

#### 4.

* $\gamma \rightarrow 0$
* $\lim\limits _{ \gamma \rightarrow 0 }{ K\left( x,{ x }^{ \prime  } \right)  } =\lim\limits _{ \gamma \rightarrow 0 }{ { e }^{ -\gamma { \left\| x-{ x }^{ \prime  } \right\|  }^{ 2 } } } =J$ 
  $J$ is the matrix of all ones.
* $\beta ={ \left( \lambda I+J \right)  }^{ -1 }y$

### Support Vector Regression

* $\left( { P }_{ 2 } \right) \min\limits _{ b,w,{ \xi  }^{ \vee  },{ \xi  }^{ \wedge  } }{ \frac { 1 }{ 2 } { w }^{ T }w+C\sum\limits _{ n=1 }^{ N }{ \left( { \left( { \xi  }_{ n }^{ \vee  } \right)  }^{ 2 }+{ \left( { \xi  }_{ n }^{ \wedge  } \right)  }^{ 2 } \right)  }  } $
  s.t. $-\epsilon -{ \xi  }_{ n }^{ \vee  }\le { y }_{ n }-{ w }^{ T }\phi \left( { x }_{ n } \right) -b\le \epsilon +{ \xi  }_{ n }^{ \wedge  }$

#### 5.

* Let ${ A }_{ n }={ y }_{ n }-{ w }^{ T }\phi \left( { x }_{ n } \right) -b$
* Lagrange Multiplier Method:
  $\mathcal {L}\left( { P }_{ 2 } \right) \min\limits _{ b,w,{ \xi  }^{ \vee  },{ \xi  }^{ \wedge  } }{ \max\limits _{ { \alpha  }^{ \vee  },{ \alpha  }^{ \wedge  } }{ \frac { 1 }{ 2 } { w }^{ T }w+C\sum\limits _{ n=1 }^{ N }{ \left( { \left( { \xi  }_{ n }^{ \vee  } \right)  }^{ 2 }+{ \left( { \xi  }_{ n }^{ \wedge  } \right)  }^{ 2 } \right)  } \\ \quad\quad+{ \alpha  }^{ \vee  }\sum\limits _{ n=1 }^{ N }{ \left( A_n+\left( \epsilon +{ \xi  }_{ n }^{ \vee  } \right)  \right)  } +{ \alpha  }^{ \wedge  }\sum\limits _{ n=1 }^{ N }{ \left( A_n-\left( \epsilon +{ \xi  }_{ n }^{ \wedge  } \right)  \right)  }  }  }$
* Partial derivative on $\xi$
  $\frac { \partial \mathcal{L}}{ \partial { \xi  }_{ n }^{ \vee  } } =2C{ \xi  }_{ n }^{ \vee  }+{ \alpha  }^{ \vee  }=0 \Rightarrow { \alpha  }^{ \vee  }=-2C{ \xi  }_{ n }^{ \vee  }$
  $\frac { \partial \mathcal{L}}{ \partial { \xi  }_{ n }^{ \wedge  } } =2C{ \xi  }_{ n }^{ \wedge  }-{ \alpha  }^{ \wedge  }=0 \Rightarrow { \alpha  }^{ \wedge  }=2C{ \xi  }_{ n }^{ \wedge  }$
  $\Rightarrow { L }\left( { P }_{ 2 } \right) \min\limits _{ b,w,{ \xi  }^{ \vee  },{ \xi  }^{ \wedge  } }{ \frac { 1 }{ 2 } { w }^{ T }w-C\sum\limits _{ n=1 }^{ N }{ \left( { \left( { \xi  }_{ n }^{ \vee  } \right)  }^{ 2 }+{ \left( { \xi  }_{ n }^{ \wedge  } \right)  }^{ 2 } \right)  } \\ \quad\quad -2C\sum\limits _{ n=1 }^{ N }{ { \xi  }_{ n }\left( A_{ n }+\epsilon  \right)  } +2C\sum\limits _{ n=1 }^{ N }{ { \xi  }_{ n }^{ \wedge  }\left( A_{ n }-\epsilon  \right)  }  }$
* Partial derivative on $\xi$ again
  $\frac { \partial \mathcal{L}}{ \partial { \xi  }_{ n }^{ \vee  } } =-2C{ \xi  }_{ n }^{ \vee  }-2C\left( { A }_{ n }+\epsilon  \right) =0 \Rightarrow { \xi  }_{ n }^{ \vee  }=-\left( { A }_{ n }+\epsilon  \right)$ and ${ A }_{ n }\le -\epsilon $
  $\frac { \partial \mathcal{L}}{ \partial { \xi  }_{ n }^{ \wedge  } } =-2C{ \xi  }_{ n }^{ \wedge  }+2C\left( { A }_{ n }-\epsilon  \right) =0 \Rightarrow { \xi  }_{ n }^{ \wedge  }=\left( { A }_{ n }-\epsilon  \right)$ and ${ A }_{ n }\ge +\epsilon$
  $\Rightarrow { L }\left( { P }_{ 2 } \right) \min\limits _{ b,w }{ \frac { 1 }{ 2 } { w }^{ T }w+C\sum\limits _{ n=1 }^{ N }{ \left( { \left[ { A }_{ n }\le -\epsilon  \right] \left( { A }_{ n }+\epsilon  \right)  }^{ 2 }+{ \left[ { A }_{ n }\ge +\epsilon  \right]  }{ \left( { A }_{ n }-\epsilon  \right)  }^{ 2 } \right) }  }$
* Transform the above equation into non-linear
  $\mathcal{L}\left( { P }_{ 2 } \right) \min\limits _{ b,w }{ \frac { 1 }{ 2 } { w }^{ T }w+C\sum\limits _{ n=1 }^{ N }{ { \left( \max { \left( 0,\left| A_n \right| -\epsilon  \right)  }  \right)  }^{ 2 } }  }$
  $\Rightarrow \mathcal{L}\left( { P }_{ 2 } \right) \min\limits _{ b,w }{ \frac { 1 }{ 2 } { w }^{ T }w+C\sum\limits _{ n=1 }^{ N }{ { \left( \max { \left( 0,\left| { y }_{ n }-{ w }^{ T }\phi \left( { x }_{ n } \right) -b \right| -\epsilon  \right)  }  \right)  }^{ 2 } }  }$

#### 6.

* Let ${ s }_{ n }=\sum\limits _{ m=1 }^{ N }{ \left( { \beta  }_{ m }K\left( { x }_{ n },{ x }_{ m } \right) +b \right)  }$
* From 5.
  $F\left( b,{ \beta  } \right) =\frac { 1 }{ 2 } \sum\limits _{ m=1 }^{ N }{ \left( \sum\limits _{ n=1 }^{ N }{ { \beta  }_{ n }{ \beta  }_{ m }K\left( { x }_{ n },{ x }_{ m } \right)  }  \right)  } +C\sum\limits _{ n=1 }^{ N }{ { \left( \max\limits { \left( 0,\left| { y }_{ n }-{ w }^{ T }\phi \left( { x }_{ n } \right) -b \right| -\epsilon  \right)  }  \right)  }^{ 2 } }$
* $\frac { \partial { s }_{ n } }{ \partial { \beta  }_{ m } } =K\left( { x }_{ n },{ x }_{ m } \right)$ denote as $K$
  $\begin{eqnarray} \frac { \partial F\left( b,{ \beta  } \right)  }{ \partial { \beta  }_{ m } }  & = & \frac { 1 }{ 2 } \sum\limits _{ n=1 }^{ N }{ { \beta  }_{ n }K } +C\sum\limits _{ n=1 }^{ N }{ \left[ \left| { y }_{ n }-{ s }_{ n } \right| \ge \epsilon  \right] \frac { \partial { \left( \left| { y }_{ n }-{ s }_{ n } \right| -\epsilon  \right)  }^{ 2 } }{ \partial { \beta  }_{ m } }  }  \\  & = & \frac { 1 }{ 2 } \sum\limits _{ n=1 }^{ N }{ { \beta  }_{ n }K } +\begin{cases} { y }_{ n }-{ s }_{ n }\ge 0,C\sum\limits _{ n=1 }^{ N }{ \left[ \left| { y }_{ n }-{ s }_{ n } \right| \ge \epsilon  \right] \frac { \partial { \left( { y }_{ n }-{ s }_{ n }-\epsilon  \right)  }^{ 2 } }{ \partial { \beta  }_{ m } }  }  \\ { y }_{ n }-{ s }_{ n }\le 0,C\sum\limits _{ n=1 }^{ N }{ \left[ \left| { y }_{ n }-{ s }_{ n } \right| \ge \epsilon  \right] \frac { \partial { \left( { s }_{ n }-{ y }_{ n }-\epsilon  \right)  }^{ 2 } }{ \partial { \beta  }_{ m } }  }  \end{cases} \\  & = & \frac { 1 }{ 2 } \sum\limits _{ n=1 }^{ N }{ { \beta  }_{ n }K } +\begin{cases} { y }_{ n }-{ s }_{ n }\ge 0,2C\sum\limits _{ n=1 }^{ N }{ \left[ \left| { y }_{ n }-{ s }_{ n } \right| \ge \epsilon  \right] \left( { y }_{ n }-{ s }_{ n }-\epsilon  \right) \left( -K \right)  }  \\ { y }_{ n }-{ s }_{ n }\le 0,2C\sum\limits _{ n=1 }^{ N }{ \left[ \left| { y }_{ n }-{ s }_{ n } \right| \ge \epsilon  \right]  } \left( { s }_{ n }-{ y }_{ n }-\epsilon  \right) K \end{cases} \\  & = & \frac { 1 }{ 2 } \sum\limits _{ n=1 }^{ N }{ { \beta  }_{ n }K } +2C\sum\limits _{ n=1 }^{ N }{ \left[ \left| { y }_{ n }-{ s }_{ n } \right| \ge \epsilon  \right]  } \left( \left| { y }_{ n }-{ s }_{ n } \right| -\epsilon  \right) sign\left( { y }_{ n }-{ s }_{ n } \right) K \end{eqnarray}$

### Blending

#### 7.

* Since there are only 2 points, the best hypothesis is simply the line passing through these 2 points. Suppose the 2 points are $\left( { x }_{ 1 },{ x }_{ 1 }^{ 2 } \right)$ and $\left( { x }_{ 2 },{ x }_{ 2 }^{ 2 } \right)$
* The best hypothesis can be represented as
  $h\left( x \right) =\frac { { x }_{ 1 }^{ 2 }-{ x }_{ 2 }^{ 2 } }{ { x }_{ 1 }-{ x }_{ 2 } } \left( x-{ x }_{ 1 } \right) +{ x }_{ 1 }^{ 2 }=\left( { x }_{ 1 }+{ x }_{ 2 } \right) x-{ x }_{ 1 }{ x }_{ 2 }$
* $\bar { g } \left( x \right) =E\left[ h\left( x \right)  \right] =E\left[ { x }_{ 1 }+{ x }_{ 2 } \right] x+E\left[ { x }_{ 1 }{ x }_{ 2 } \right]$
* Since the $x$ value of points are sampled from unifrom distribution over $\left[ 0,1 \right]$
  $\Rightarrow \bar { g } \left( x \right) =E\left[ { x }_{ 1 }+{ x }_{ 2 } \right] x+E\left[ { x }_{ 1 }{ x }_{ 2 } \right] =x-\frac { 1 }{ 4 }$

### Test Set Linear Regression

#### 8.

* Define a cheating hypothesis.
  ${ g }_{ i }\left( { x }_{ j } \right) =\left[ i=j \right]$, where $1\le i,j\le N$. The function will output 1 if $ i=j $, else output 0.
  Define a special hypothesis that will always output 0.
  ${ g }_{ 0 }\left( { x }_{ j } \right) =0$, where $1\le j\le N$. 
* Construct a series of cheating hypothesis
  $\left[ { g }_{ 0 },{ g }_{ 1 },{ g }_{ 2 },\dots ,{ g }_{ n-2 },{ g }_{ n-1 } \right]$
* Query RMSE for N times to obtain $RMSE\left( { g }_{ i } \right)$, where $0\le i\le N-1$
* Now we can compute every ${ \tilde { y }  }_{ i }$
  ${ \tilde { y }  }_{ i }=\frac { 1 }{ 2 } \left( N\left( { \left[ RMSE\left( { g }_{ 0 } \right)  \right]  }^{ 2 }-{ \left[ RMSE\left( { g }_{ i } \right)  \right]  }^{ 2 } \right) +1 \right)$
* ${ \tilde { y }  }_{ n }$ can be computed from all the other ${ \tilde { y }  }_{ i }$ with $RMSE\left( { g }_{ 0 } \right)$
  Thus we need N queries.

#### 9.

* Continue from 8., we use $g_0$ again
  ${ g }_{ 0 }\left( { x }_{ j } \right) =0$, where $1\le j\le N$. 
* List out two equations below
  ${ \left[ RMSE\left( { g }_{ 0 } \right)  \right]  }^{ 2 }=\frac { 1 }{ N } \sum\limits _{ i=1 }^{ N }{ { \left( { \tilde { y }  }_{ i } \right)  }^{ 2 } } =\frac { 1 }{ N } { \tilde { y }  }^{ T }\tilde { y }$
  ${ \left[ RMSE\left( { g } \right)  \right]  }^{ 2 }=\frac { 1 }{ N } \sum\limits _{ i=1 }^{ N }{ { \left( { \tilde { y }  }_{ i }-g\left( { x }_{ i } \right)  \right)  }^{ 2 } } =\frac { 1 }{ N } \left( { \tilde { y }  }^{ T }\tilde { y } -2{ g }^{ T }\tilde { y } +{ g }^{ T }g \right)$
* We can obtain ${ g }^{ T }\tilde { y }$ by the equations above
  ${ g }^{ T }\tilde { y } =\frac { 1 }{ 2 } \left( N\left( { \left[ RMSE\left( { g }_{ 0 } \right)  \right]  }^{ 2 }-{ \left[ RMSE\left( { g } \right)  \right]  }^{ 2 } \right) +{ g }^{ T }g \right)$
* Thus we only need 2 queries.

#### 10.

* Continue from 8. 9., we use $g_0$,  ${ g }^{ T }\tilde { y }$ again

  ${ g }^{ T }\tilde { y } =\frac { 1 }{ 2 } \left( N\left( { \left[ RMSE\left( { g }_{ 0 } \right)  \right]  }^{ 2 }-{ \left[ RMSE\left( { g } \right)  \right]  }^{ 2 } \right) +{ g }^{ T }g \right)$

* The problem is to obtain optimal $\left[ { \alpha  }_{ 1 },{ \alpha  }_{ 2 },\dots ,{ \alpha  }_{ K } \right]$ for
  $\min\limits _{ { \alpha  }_{ 1 },{ \alpha  }_{ 2 },\dots ,{ \alpha  }_{ K } }{ RMSE\left( \sum\limits _{ k=1 }^{ K }{ { \alpha  }_{ k }{ g }_{ k } }  \right)  }$
  $RMSE\left( \sum\limits _{ k=1 }^{ K }{ { \alpha  }_{ k }{ g }_{ k } }  \right) =\frac { 1 }{ N } \left( { \tilde { y }  }^{ T }\tilde { y } -2{ \left( \sum\limits _{ k=1 }^{ K }{ { \alpha  }_{ k }{ g }_{ k } }  \right)  }^{ T }\tilde { y } +{ \left( \sum\limits _{ k=1 }^{ K }{ { \alpha  }_{ k }{ g }_{ k } }  \right)  }^{ T }\left( \sum\limits _{ k=1 }^{ K }{ { \alpha  }_{ k }{ g }_{ k } }  \right)  \right)$

* Partial derivative by ${ \alpha  }_{ s }$, where $1\le s\le K$
  $\frac { \partial RMSE\left( \sum\limits _{ k=1 }^{ K }{ { \alpha  }_{ k }{ g }_{ k } }  \right)  }{ { \partial \alpha  }_{ s } } =-2\left( { g }_{ s } \right) ^{ T }\tilde { y } +2\left( { g }_{ s } \right) ^{ T }\sum\limits _{ k=1 }^{ K }{ { \alpha  }_{ k }{ g }_{ k } } =0$
  From 9., we can calculate $\left( { g }_{ s } \right) ^{ T }\tilde { y }$

* We can obtain $K$ equations to solve all $\alpha$
  $\Rightarrow$ Require $K+1$ queries.

### Experiment with Kernel Ridge Regression

#### 11.

* 400 training data, 100 testing data.


* $E_{in}$

  | $\lambda$ \ $\gamma$ | $32$  |  $2$  | $0.125$  |
  | :------------------: | :---: | :---: | :------: |
  |       $0.001$        | $0.0$ | $0.0$ |  $0.0$   |
  |         $1$          | $0.0$ | $0.0$ |  $0.03$  |
  |        $1000$        | $0.0$ | $0.0$ | $0.2425$ |

* $\gamma=32, 2$ or $(\lambda, \gamma)=(0.001, 0.125)$ have the minimum $E_{in}=0.0$

#### 12.

- $E_{out}$

  | $\lambda$ \ $\gamma$ |  $32$  |  $2$   | $0.125$ |
  | :------------------: | :----: | :----: | :-----: |
  |       $0.001$        | $0.45$ | $0.44$ | $0.46$  |
  |         $1$          | $0.45$ | $0.44$ | $0.45$  |
  |        $1000$        | $0.45$ | $0.44$ | $0.39$  |

- ​$(\lambda, \gamma)=(1000, 0.125)$ has the minimum $E_{in}=0.39$

### Experiment with Support Vector Regression

#### 13.

* 400 training data, 100 testing data.


* $E_{in}$

  | $\lambda$ \ $\gamma$ | $32$  |  $2$  | $0.125$ |
  | :------------------: | :---: | :---: | :-----: |
  |       $0.001$        | $0.4$ | $0.4$ |  $0.4$  |
  |         $1$          | $0.0$ | $0.0$ | $0.035$ |
  |        $1000$        | $0.0$ | $0.0$ |  $0.0$  |

* $\gamma=1000$ or $(\lambda, \gamma)=(1,32)$ or $(1, 2)$ have minimum $E_{in}=0.0$

#### 14.

* $E_{out}$

  | $\lambda$ \ $\gamma$ |  $32$  |  $2$   | $0.125$ |
  | :------------------: | :----: | :----: | :-----: |
  |       $0.001$        | $0.48$ | $0.48$ | $0.48$  |
  |         $1$          | $0.48$ | $0.48$ | $0.42$  |
  |        $1000$        | $0.48$ | $0.48$ | $0.47$  |

* $(\lambda, \gamma)=(1, 0.125)$ has the minimum $E_{in}=0.42$

### Experiment with Bagging Ridge Regression

#### 15. 16.

* Bootstrap aggregation on 400 training data, 200 iterations.
* 100 testing data.

- $E_{in}$ and $E_{out}$

  | $\lambda$ | $E_{in}$ | $E_{out}$ |
  | :-------: | :------: | :-------: |
  |  $0.01$   | $0.3115$ | $0.3710$  |
  |   $0.1$   | $0.3105$ | $0.3677$  |
  |    $1$    | $0.3126$ | $0.3693$  |
  |   $10$    | $0.3109$ | $0.3690$  |
  |   $100$   | $0.3128$ | $0.3692$  |

- $\lambda=0.1$ has the smallest $E_{in}=0.3105$ and $E_{out}=0.3677$