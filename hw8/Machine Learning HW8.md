## Machine Learning Techniques HW8

---

* b03902089 林良翰

### Random Forest

#### 1.

* Every example has ${ \left( 1-\frac { 1 }{ N }  \right)  }^{ N^\prime  }={ \left( 1-\frac { 1 }{ N }  \right)  }^{ Np }$ probability not to be sampled at all.
* $\lim\limits _{ N\rightarrow \infty  }{ { \left( 1-\frac { 1 }{ N }  \right)  }^{ Np }={ e }^{ -p } }$
* Thus, there are approximately ${ e }^{ -p }N$ examples will not be sampled at all.

#### 2.

* If all the examples predicted wrongly by at most one classification tree (one of $g_1$, $g_2$, $g_3$), $E_{out}\left(G\right)$ has its minimum $0$.
* The maximum ratio of examples which is predicted wrongly by at least two classification tree (two of  $g_1$, $g_2$, $g_3$, or all of them) is $0.375$ (e.g. $g_1$ and $g_2$ have $0.35$ same error with $g_3$, $g_1$ and $g_2$ have more $0.025$ same error but different with $g_3$).
* The range of $E_{out}\left(G\right)$ is $\left[0,0.375\right]$

#### 3. ?

* Assume that there are $K$ ($K$ is odd) binary classification trees $g_1,\dots,g_k$ with $E_{out}$ $e_1,\dots,e_k$, if there is an error on an example, it's necessary that there are at least $\frac{K+1}{2}$ (more than half) trees make the mistake.

* To obtain the maximum $E_{out}\left(G\right)$, we can consider that we divide all classification trees into $\frac{K+1}{2}=M$ groups $G_1,\dots,G_M$, while each group contains at least one tree, and errors of every tree in the same group are mutual exclusive (maximize the error).

* With the assumption above, we can know
  $$
  E_{out}\left(G\right)=\min\limits _{ i }{ E_{ out }\left( { G }_{ i } \right)  }
  $$
  where $i$ is integer and $1\le i\le M$.

* "Minimum is always smaller than mean", by this simple theory, we could derive 
  $$
  \min _{ i }{ E_{ out }\left( { G }_{ i } \right)  } \le \frac { 1 }{ M } \sum _{ i=1 }^{ M }{ E_{ out }\left( { G }_{ i } \right)  }
  $$

* The error of each group $G_1\dots G_M$ is not necessary mutual exclusive, thus if all trees $g_1\dots g_k$ have mutual exclusive errors, then
  $$
  \frac { 1 }{ M } \sum _{ i=1 }^{ M }{ E_{ out }\left( { G }_{ i } \right)  } \le \frac { 1 }{ M } \sum _{ k=1 }^{ K }{ E_{ out }\left( g_{ k } \right)  } \\
  \frac { 1 }{ M } \sum _{ j=1 }^{ N }{ E_{ out }\left( g_{ j } \right)  } =\frac { 2 }{ K+1 } \sum _{ k=1 }^{ K }{ { e }_{ k } }
  $$

* $\frac { 2 }{ K+1 } \sum\limits _{ k=1 }^{ K }{ { e }_{ k } }$ upper bounds $E_{out}\left(G\right)$

### Gradient Boosting

* The algroithm of gradient boosting
  * Initialize $s_1=s_2=\dots=s_N=0$.
  * For $t=\left[1,2,\dots,N\right]$ :
    * Obtain $g_t$ by ${\mathcal{A}}\{\left(x_n,y_n-s_n\right)\}$, where ${\mathcal A}$ is a squared-error regression algorithm.
    * Compute $\alpha_t={\rm OneVariableLinearRegression}\{\left(g_t\left(x_n\right),y_n-s_n\right)\}$
    * Update $s_n\leftarrow s_n+\alpha_t g_t\left(x_n\right)$
  * Return $G\left(x\right)=\sum\limits_{t=1}^{T}{\alpha_t g_t\left(x\right)}$

#### 4.

* By computing the gradient of squared-error of $\{\left(g_t\left(x_n\right),y_n-s_n\right)\}$ over $\alpha_t$
  $\frac { \partial \sum\limits _{ n=1 }^{ N }{ { \left( \left( { y }_{ n }-{ s }_{ n } \right) -{ \alpha  }_{ 1 }{ g }_{ 1 }\left( { x }_{ n } \right)  \right)  }^{ 2 } }  }{ \partial { \alpha  }_{ 1 } } =0$
  $\Rightarrow 2\sum\limits _{ n=1 }^{ N }{ \left( \left( { y }_{ n }-{ s }_{ n } \right) -{ \alpha  }_{ 1 }{ g }_{ 1 }\left( { x }_{ n } \right)  \right) \left( -{ g }_{ 1 }\left( { x }_{ n } \right)  \right)  } =0$
  $\Rightarrow 2\sum\limits _{ n=1 }^{ N }{ \left( { y }_{ n }-0-2{ \alpha  }_{ 1 } \right) \left( -2 \right)  } =0$
  $\Rightarrow { \alpha  }_{ 1 }=\frac { 1 }{ 2N } \sum\limits _{ n=1 }^{ N }{ { y }_{ n } } $
* By the update function
  $s_n\leftarrow s_n+\alpha_1 g_1\left(x_n\right)=0+\frac{1}{2N}\sum\limits_{n=1}^{N}{(y_n)}\cdot 2=\frac{1}{N}\sum\limits_{n=1}^{N}{(y_n)}$

#### 5.

* The steepest $\eta$ has the smallest gradient $0$.
  $\frac { \partial \sum\limits _{ n=1 }^{ N }{ { \left( \left( { y }_{ n }-{ s }_{ n } \right) -\eta { g }_{ t }\left( { x }_{ n } \right)  \right)  }^{ 2 } }  }{ \partial \eta  } =0$
  $\Rightarrow 2\sum\limits _{ n=1 }^{ N }{ { \left( \left( { y }_{ n }-{ s }_{ n } \right) -\eta { g }_{ t }\left( { x }_{ n } \right)  \right)  }\left( -{ g }_{ t }\left( { x }_{ n } \right)  \right)  } =0$
  $\Rightarrow \sum\limits _{ n=1 }^{ N }{ \left( -{ y }_{ n }{ g }_{ t }\left( { x }_{ n } \right) +{ s }_{ n }{ g }_{ t }\left( { x }_{ n } \right) +\eta { \left( { g }_{ t }\left( { x }_{ n } \right)  \right)  }^{ 2 } \right)  } =0$
  $\Rightarrow \sum\limits _{ n=1 }^{ N }{ { s }_{ n }{ g }_{ t }\left( { x }_{ n } \right)  } =\sum\limits _{ n=1 }^{ N }{ \left( { y }_{ n }{ g }_{ t }\left( { x }_{ n } \right) -\eta { \left( { g }_{ t }\left( { x }_{ n } \right)  \right)  }^{ 2 } \right)  } =\sum\limits _{ n=1 }^{ N }{ { g }_{ t }\left( { x }_{ n } \right) \left( { y }_{ n }-\eta { { g }_{ t }\left( { x }_{ n } \right)  } \right)  }$

#### 6.

* ${\mathcal A}$ is an algorithm of linear regression, and we found the result ${ g }_{ 1 }\left( x \right) ={ w }_{ 1 }x+{ b }_{ 1 }$
  $w_1$, $b_1$ are the optimal solution from linear regression, and must conform to the following equations
  $$
  \begin{cases} \frac { \partial { \sum\limits _{ n=1 }^{ N }{ { \left( { g }_{ 1 }\left( { x }_{ n } \right) -\left( { y }_{ n }-{ s }_{ n } \right)  \right)  }^{ 2 } }  } }{ \partial { w }_{ 1 } } =\frac { \partial \sum\limits _{ n=1 }^{ N }{ { \left( { w }_{ 1 }{ x }_{ n }+{ b }_{ 1 }-\left( { y }_{ n }-{ s }_{ n } \right)  \right)  }^{ 2 } }  }{ \partial { w }_{ 1 } } =0 \\ \frac { \partial \sum\limits _{ n=1 }^{ N }{ { \left( { g }_{ 1 }\left( { x }_{ n } \right) -\left( { y }_{ n }-{ s }_{ n } \right)  \right)  }^{ 2 } }  }{ \partial { b }_{ 1 } } =\frac { \partial \sum\limits _{ n=1 }^{ N }{ { \left( { w }_{ 1 }{ x }_{ n }+{ b }_{ 1 }-\left( { y }_{ n }-{ s }_{ n } \right)  \right)  }^{ 2 } }  }{ \partial { b }_{ 1 } } =0 \end{cases}
  $$
  And obtain the following result
  $$
  \begin{cases} { w }_{ 1 }=\frac { \sum\limits _{ n=1 }^{ N }{ { x }_{ n }\left( { y }_{ n }-{ s }_{ n } \right)  } -{ b }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { x }_{ n } }  }{ \sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }  }  \\ 
  { b }_{ 1 }=\sum\limits _{ n=1 }^{ N }{ { x }_{ n }\left( { y }_{ n }-{ s }_{ n } \right)  } -{ w }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }  \end{cases}
  $$

* Suppose $\alpha_1\ne1​$, then we can derive
  $$
  \begin{cases} \frac { \partial { \sum\limits _{ n=1 }^{ N }{ { \left( { \alpha  }_{ 1 }{ g }_{ 1 }\left( { x }_{ n } \right) -\left( { y }_{ n }-{ s }_{ n } \right)  \right)  }^{ 2 } }  } }{ \partial { w }_{ 1 } } =\frac { \partial \sum\limits _{ n=1 }^{ N }{ { \left( { \alpha  }_{ 1 }{ w }_{ 1 }{ x }_{ n }+{ \alpha  }_{ 1 }{ b }_{ 1 }-\left( { y }_{ n }-{ s }_{ n } \right)  \right)  }^{ 2 } }  }{ \partial { w }_{ 1 } } =0 \\ \frac { \partial \sum\limits _{ n=1 }^{ N }{ { \left( { \alpha  }_{ 1 }{ g }_{ 1 }\left( { x }_{ n } \right) -\left( { y }_{ n }-{ s }_{ n } \right)  \right)  }^{ 2 } }  }{ \partial { b }_{ 1 } } =\frac { \partial \sum\limits _{ n=1 }^{ N }{ { \left( { \alpha  }_{ 1 }{ w }_{ 1 }{ x }_{ n }+{ \alpha  }_{ 1 }{ b }_{ 1 }-\left( { y }_{ n }-{ s }_{ n } \right)  \right)  }^{ 2 } }  }{ \partial { b }_{ 1 } } =0 \end{cases}
  $$
  And obtain
  $$
  \begin{cases} { w }_{ 1 }=\frac { \sum\limits _{ n=1 }^{ N }{ { x }_{ n }\left( { y }_{ n }-{ s }_{ n } \right)  } -{ \alpha  }_{ 1 }{ b }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { x }_{ n } }  }{ { \alpha  }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }  }  \ne\frac { \sum\limits _{ n=1 }^{ N }{ { x }_{ n }\left( { y }_{ n }-{ s }_{ n } \right)  } -{ b }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { x }_{ n } }  }{ \sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }  }\\ 
  { b }_{ 1 }=\frac { 1 }{ { \alpha  }_{ 1 } } \sum\limits _{ n=1 }^{ N }{ { x }_{ n }\left( { y }_{ n }-{ s }_{ n } \right)  } -{ w }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }\ne \sum\limits _{ n=1 }^{ N }{ { x }_{ n }\left( { y }_{ n }-{ s }_{ n } \right)  } -{ w }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }  \end{cases}
  $$
  We found a more optimal $w_1$ and $b_1$

  $\Rightarrow$ Linear Regression $\mathcal A$ doesn't obtain the optimal solution.

  $\Rightarrow$ Contradiction.

* $\alpha_1$ must be $1$.

#### 7.

* From 6., we have the result of first iteration
  $$
  \begin{cases} { w }_{ 1 }=\frac { \sum\limits _{ n=1 }^{ N }{ { x }_{ n }\left( { y }_{ n }-{ s }_{ n } \right)  } -{ b }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { x }_{ n } }  }{ \sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }  } =\frac { \sum\limits _{ n=1 }^{ N }{ { x }_{ n }{ y }_{ n } } -{ b }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { x }_{ n } }  }{ \sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }  }  \\
  { b }_{ 1 }=\sum\limits _{ n=1 }^{ N }{ { x }_{ n }\left( { y }_{ n }-{ s }_{ n } \right)  } -{ w }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } } =\sum\limits _{ n=1 }^{ N }{ { x }_{ n }{ y }_{ n } } -{ w }_{ 1 }\sum\limits _{ n=1 }^{ N }{ { { \left( { x }_{ n } \right)  } }^{ 2 } }  \end{cases}
  $$
  where $s_n=s_n^1=0$, and $w_1$, $b_1$ are the optimal solution from $\mathcal A$.

  And we get the new $s_n^2=s_n+\alpha_1 g_1(x_n)=w_1 x_n+b_1$ at second iteration.

* Suppose we find $g_2(x_n)=w_2x_n+b_2\ne0$.
  By the linear regression algorithm $\mathcal A$, we want the gradient of the error function to be $0$.
  $$
  \frac { \partial \sum _{ n=1 }^{ N }{ { \left( { g }_{ 2 }\left( { x }_{ n } \right) -\left( { y }_{ n }-{ s }_{ n }^{ 1 } \right)  \right)  }^{ 2 } }  }{ \partial { w }_{ 2 } } =\frac { \partial \sum _{ n=1 }^{ N }{ { \left( { g }_{ 2 }\left( { x }_{ n } \right) -\left( { y }_{ n }-{ g }_{ 1 }\left( { x }_{ n } \right)  \right)  \right)  }^{ 2 } }  }{ \partial { w }_{ 2 } } =0\\
  \Rightarrow{ w }_{ 1 }+{ w }_{ 2 }=\frac { \sum\limits _{ n=1 }^{ N }{ { y }_{ n }x_{ n }-\left( { b }_{ 1 }+{ b }_{ 2 } \right) \sum\limits _{ n=1 }^{ N }{ { x }_{ n } }  }  }{ \sum\limits _{ n=1 }^{ N }{ { \left( { x }_{ n } \right)  }^{ 2 } }  }
  $$

* Let ${ w }^{ \prime  }={ w }_{ 1 }+{ w }_{ 2 }$ and ${ b }^{ \prime  }={ b }_{ 1 }+{ b }_{ 2 }$.
  $$
  w_2x_n+b_2\ne0\Rightarrow \begin{cases} { w }_{ 2 }=0,b_{ 2 }\neq 0 \\ { w }_{ 2 }\neq 0,b_{ 2 }=0 \\ { w }_{ 2 }\neq 0,b_{ 2 }\neq 0 \end{cases}\Rightarrow w^\prime\ne w_1\vee b^\prime\ne b_1
  $$
  We found a more optimal solution $w^\prime$, $b^\prime$ from $\mathcal A$, which is different from $w_1$, $b_1\Rightarrow$ Contradiction.

* $g_2(x_n)$ must be $0$.

### Neural Network

#### 8.

* To implement $OR\left(x_1,x_2,\dots,x_d\right)$, we need to make $w_0=d-1$, and $w_1=w_2=\dots =w_d=+1$, meaning that the output of ${\rm sign}\left( \sum\limits _{ i=0 }^{ d }{ { w }_{ i }{ x }_{ i } }  \right)$ will be positive if there is at least one $x_i$ which is positive.
* $\left(w_0,w_1,\dots,w_d\right)=\left( d-1,\underbrace { +1,\dots ,+1 }_{ d }  \right)$

#### 9.

* The meaning of $XOR$ is testing if there is odd number of positive examples.

* Thus we can use the formula of combinations
  $$
  \begin{pmatrix} 5 \\ 5 \end{pmatrix}+\begin{pmatrix} 5 \\ 3 \end{pmatrix}+\begin{pmatrix} 5 \\ 1 \end{pmatrix}=1+10+5=16
  $$

* $D\ge16$ 

#### 10. ?

* The Error function
  $$
  { e }_{ n }={ \left( { y }_{ n }-{ x }_{ 1 }^{ (L) } \right)  }^{ 2 }={ \left( { y }_{ n }-\tanh { \left( { s }_{ 1 }^{ (L) } \right)  }  \right)  }^{ 2 }
  $$
  $s_1^{\left(L\right)}$ is the linear combination from $\left(L-1\right)$ layer
  $$
  { s }_{ 1 }^{ (L) }=\sum _{ i=0 }^{ { d }^{ \left( L-1 \right)  } }{ { w }_{ i1 }^{ (L) }{ x }_{ i }^{ \left( L-1 \right)  } }
  $$
  $x_1^{\left(L\right)}$ is the final output
  $$
  { x }_{ 1 }^{ (L) }=\tanh { \left( { s }_{ 1 }^{ (L) } \right)  } =\tanh { \left( \sum _{ i=0 }^{ { d }^{ \left( L-1 \right)  } }{ { w }_{ i1 }^{ (L) }{ x }_{ i }^{ \left( L-1 \right)  } }  \right)  } 
  $$

* Gradient of weights before output layer

  $\begin{matrix} \Rightarrow  & \frac { \partial { e }_{ n } }{ \partial { w }_{ i1 }^{ (L) } } = \frac { \partial { \left( { y }_{ n }-{ x }_{ 1 }^{ (L) } \right)  }^{ 2 } }{ \partial { x }_{ 1 }^{ (L) } } \frac { \partial { x }_{ 1 }^{ (L) } }{ \partial { s }_{ 1 }^{ (L) } } \frac { \partial { s }_{ 1 }^{ (L) } }{ \partial { w }_{ i1 }^{ (L) } }   \end{matrix}$

  $\begin{matrix} = & \frac { \partial { \left( { y }_{ n }-\tanh { \left( { s }_{ 1 }^{ (L) } \right)  }  \right)  }^{ 2 } }{ \partial \tanh { \left( { s }_{ 1 }^{ (L) } \right)  }  } \frac { \partial \tanh { \left( { s }_{ 1 }^{ (L) } \right)  }  }{ \partial { s }_{ 1 }^{ (L) } } \frac { \partial { s }_{ 1 }^{ (L) } }{ \partial { w }_{ i1 }^{ (L) } }  \end{matrix}$

  $\begin{matrix} = & -2\left( { y }_{ n }-\tanh { \left( { s }_{ 1 }^{ (L) } \right)  }  \right) \cdot \left( 1-\tanh ^{ 2 }{ \left( { s }_{ 1 }^{ (L) } \right)  }  \right) \cdot { x }_{ i }^{ \left( L-1 \right)  } \end{matrix}$

  $\begin{matrix} = & { \delta  }_{ 1 }^{ \left( L \right)  }\cdot { x }_{ i }^{ \left( L-1 \right)  } \end{matrix}$

  * The notation ${ \delta  }_{ 1 }^{ \left( L \right)  }=\frac { \partial { e }_{ n } }{ \partial { s }_{ 1 }^{ (L) } } =-2\left( { y }_{ n }-\tanh { \left( { s }_{ 1 }^{ (L) } \right)  }  \right) \cdot \left( 1-\tanh ^{ 2 }{ \left( { s }_{ 1 }^{ (L) } \right)  }  \right)$
  * Because all the weights are $0$.
    $\Rightarrow { x }_{ i }^{ \left( L-1 \right)  }=\tanh { \left( \sum\limits _{ j=0 }^{ { d }^{ \left( L-2 \right)  } }{ { w }_{ ji }^{ (L-1) }{ x }_{ j }^{ \left( L-2 \right)  } }  \right)  } =\tanh { \left( 0 \right)  } =0$
  * $\Rightarrow \frac { \partial { e }_{ n } }{ \partial { w }_{ i1 }^{ (L) } }=0$

* Gradient of other layers
  $\begin{matrix} \Rightarrow  & \frac { \partial { e }_{ n } }{ \partial { w }_{ ij }^{ (\ell ) } } = \frac { \partial { e }_{ n } }{ \partial { s }_{ j }^{ (\ell ) } } \frac { \partial { s }_{ j }^{ (\ell ) } }{ \partial { w }_{ ij }^{ (\ell ) } }   \end{matrix}$

  $\begin{matrix} = & \left( \sum\limits _{ k=1 }^{ { d }^{ \left( \ell +1 \right)  } }{ \frac { \partial { e }_{ n } }{ \partial { s }_{ k }^{ (\ell +1) } } \cdot \frac { \partial { s }_{ k }^{ (\ell +1) } }{ \partial { x }_{ j }^{ \left( \ell  \right)  } } \cdot \frac { { \partial x }_{ j }^{ \left( \ell  \right)  } }{ \partial { s }_{ j }^{ (\ell ) } }  }  \right) \cdot { x }_{ i }^{ \left( \ell -1 \right)  } \end{matrix}$

  $\begin{matrix} = & \left( \sum\limits _{ k=1 }^{ { d }^{ \left( \ell +1 \right)  } }{ \frac { \partial { e }_{ n } }{ \partial { s }_{ k }^{ (\ell +1) } } \cdot \frac { \partial \left( \sum\limits _{ j=0 }^{ { d }^{ \left( \ell  \right)  } }{ { w }_{ jk }^{ (\ell +1) }{ x }_{ j }^{ \left( \ell  \right)  } }  \right)  }{ \partial { x }_{ j }^{ \left( \ell  \right)  } } \cdot \frac { \partial \tanh { \left( { s }_{ j }^{ (\ell ) } \right)  }  }{ \partial { s }_{ j }^{ (\ell ) } }  }  \right) \cdot { x }_{ i }^{ \left( \ell -1 \right)  } \end{matrix}$

  $\begin{matrix} = & \left( \sum\limits _{ k=1 }^{ { d }^{ \left( \ell +1 \right)  } }{ \frac { \partial { e }_{ n } }{ \partial { s }_{ k }^{ (\ell +1) } } \cdot { w }_{ jk }^{ (\ell +1) }\cdot \left( 1-\tanh ^{ 2 }{ \left( { s }_{ j }^{ (\ell ) } \right)  }  \right)  }  \right) \cdot { x }_{ i }^{ \left( \ell -1 \right)  } \end{matrix}$

  $\begin{matrix} = & \left( \sum\limits _{ k=1 }^{ { d }^{ \left( \ell +1 \right)  } }{ { \delta  }_{ k }^{ \left( \ell +1 \right)  }\cdot { w }_{ jk }^{ (\ell +1) }\cdot \left( 1-\tanh ^{ 2 }{ \left( { s }_{ j }^{ (\ell ) } \right)  }  \right)  }  \right) \cdot { x }_{ i }^{ \left( \ell -1 \right)  } \end{matrix}$

  * Because all the weights are $0$.
    $\Rightarrow \frac { \partial { e }_{ n } }{ \partial { w }_{ ij }^{ (\ell ) } }=0$

* All the gradient components are $0$.

#### 11.

* From 10.
  $$
  \frac { \partial { e }_{ n } }{ \partial { w }_{ ij }^{ (\ell ) } }=\left( \sum\limits _{ k=1 }^{ { d }^{ \left( \ell +1 \right)  } }{ { \delta  }_{ k }^{ \left( \ell +1 \right)  }\cdot { w }_{ jk }^{ (\ell +1) }\cdot \left( 1-\tanh ^{ 2 }{ \left( { s }_{ j }^{ (\ell ) } \right)  }  \right)  }  \right) \cdot { x }_{ i }^{ \left( \ell -1 \right)  }
  $$

* While $\ell=1$
  $$
  \frac { \partial { e }_{ n } }{ \partial { w }_{ ij }^{ (1) } } =\left( \sum\limits _{ k=1 }^{ { d }^{ \left( 2 \right)  } }{ { \delta  }_{ k }^{ \left( 2 \right)  }\cdot { w }_{ jk }^{ (2) }\cdot \left( 1-\tanh ^{ 2 }{ \left( { s }_{ j }^{ (1) } \right)  }  \right)  }  \right) \cdot { x }_{ i }^{ \left( 0 \right)  }\\
  \Rightarrow w_{jk}^{(2)}=w_{(j+1)k}^{(2)}=1
  $$

* Thus we have to determine whether $s_j^{(1)}$ and $s_{j+1}^{(1)}$ are the same.

  After one iteration from initial weights with all ones
  $$
  { s }_{ j }^{ (1) }=\sum _{ i=0 }^{ { d }^{ \left( 0 \right)  } }{ { w }_{ ij }^{ (1) }{ x }_{ i }^{ \left( 0 \right)  } }\\
  { s }_{ j+1 }^{ (1) }=\sum _{ i=0 }^{ { d }^{ \left( 0 \right)  } }{ { w }_{ i(j+1) }^{ (1) }{ x }_{ i }^{ \left( 0 \right)  } }\\
  { w }_{ ij }^{ (1) }={ w }_{ i(j+1) }^{ (1) }=1
  $$
  $\Rightarrow { s }_{ j }^{ (1) }$and ${ s }_{ j+1 }^{ (1) }$ are the same because the previous ${ w }_{ ij }^{ (1) }$ and ${ w }_{ i(j+1) }^{ (1) }$ are the same.

  $\Rightarrow$ The gradient of ${ w }_{ ij }^{ (1) }$ and ${ w }_{ i(j+1) }^{ (1) }$ are the same.

* After one iteration,  ${ w }_{ ij }^{ (1) }$ and ${ w }_{ i(j+1) }^{ (1) }$ were updated to the same value.

  $\Rightarrow$ Throughout the training preocess, the weights of  ${ w }_{ ij }^{ (1) }$ and ${ w }_{ i(j+1) }^{ (1) }$ are always the same.

### Experiments with Random Forest

#### 12.

![12](/Users/Qhan/mlt2017/hw8/12.png)

#### 13.

![13](/Users/Qhan/Desktop/13.png)

#### 14.

![14](/Users/Qhan/Desktop/14.png)

* The $E_{in}$ goes down to $0$ immediately, but $E_{out}$ converges to a value instead.

#### 15.

![15](/Users/Qhan/mlt2017/hw8/15.png)

#### 16.

![16](/Users/Qhan/mlt2017/hw8/16.png)

* The $E_{out}(G_t)$ is higher than $E_{in}(G_t)$, and both converge to some values.