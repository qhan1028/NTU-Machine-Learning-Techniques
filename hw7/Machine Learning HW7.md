# Machine Learning Techniques HW7

B03902089 資工三 林良翰

### Boosting

#### 1. 

* $\min\limits _{ w }{ { E }_{ in }^{ u }\left( w \right)  } =\frac { 1 }{ N } \sum\limits _{ n=1 }^{ N }{ { u }_{ n }{ \left( { y }_{ n }-{ w }^{ T }{ x }_{ n } \right)  }^{ 2 } }$


* For every $\left( { x }_{ n },{ y }_{ n } \right)$
  ${ u }_{ n }{ \left( { y }_{ n }-{ w }^{ T }{ x }_{ n } \right)  }^{ 2 }={ \left( { \sqrt { { u }_{ n } } y }_{ n }-{ w }^{ T }\sqrt { { u }_{ n } } { x }_{ n } \right)  }^{ 2 }={ \left( { \tilde { y }  }_{ n }-{ w }^{ T }{ \tilde { x }  }_{ n } \right)  }^{ 2 }$
* $\Rightarrow { \left\{ \left( { \tilde { x }  }_{ n },{ \tilde { y }  }_{ n } \right)  \right\}  }_{ n=1 }^{ N }={ \left\{ \left( \sqrt { { u }_{ n } } { x }_{ n },\sqrt { { u }_{ n } } { y }_{ n } \right)  \right\}  }_{ n=1 }^{ N }$

#### 2.

* ${ u }^{ \left( 1 \right)  }=\left[ \frac { 1 }{ N } ,\frac { 1 }{ N } ,\dots m,\frac { 1 }{ N }  \right]$
* 99% of the examples are positive $\Rightarrow { \epsilon  }_{ 1 }=0.99$
* ${ u }_{ - }^{ \left( 2 \right)  }\propto \frac { 1 }{ N } \cdot { \epsilon  }_{ 1 }\Rightarrow \frac { k\cdot 0.99 }{ N } \\ { u }_{ - }^{ \left( 2 \right)  }\propto \frac { 1 }{ N } \cdot { \left( 1-\epsilon _{ 1 } \right)  }\Rightarrow \frac { k\cdot 0.01 }{ N }$
* $\frac { { u }_{ + }^{ \left( 2 \right)  } }{ { u }_{ - }^{ \left( 2 \right)  } } =\frac { \left( 1-0.99 \right)  }{ 0.99 } =\frac { 1 }{ 99 }$ 

### Kernel for Decision Stumps

#### 3.

* ${ g }_{ s,i,\theta  }\left( x \right) =s\cdot \rm{sign}\left( { x }_{ i }-\theta  \right)$

  $i\in \left\{ 1,2,\dots ,d \right\}$, $d$ is the finite dimensionality of the input space.

  $s\in \left\{ +1,-1 \right\}$, $\theta \in \rm{I\!R}$, and $\rm{sign}\left( 0\right)=+1$


* $L=1$ and $R=6 \Rightarrow$ There are 6 kinds of $\theta$.
* $d=2 \Rightarrow$ Two dimensions.
* $s\in \left\{ +1,-1 \right\} \Rightarrow$ Positive ray or negative ray.
* Thus, there are $6 \times 2 \times 2 = 24$ kinds of $g_{s,i,\theta}\left(x\right)$.

#### 4.

### Decision Tree

${ u }_{ + }$ : Fraction of positive examples.

${ u }_{ - }=1-{ u }_{ + }$ : Fraction of Negative examples.

#### 5.

* Find the maximum value of Gini index
  $\max\limits _{ { u }_{ + } }{ 1-{ u }_{ + }^{ 2 }-{ u }_{ - }^{ 2 } }$, where $u_{+}\in \left[0, 1\right]$
* By ${ u }_{ - }=1-{ u }_{ + }$
  $\begin{matrix}  & 1-{ u }_{ + }^{ 2 }-{ u }_{ - }^{ 2 } \end{matrix}\\ \begin{matrix} = & 1-{ u }_{ + }^{ 2 }-{ \left( 1-{ u }_{ + } \right)  }^{ 2 } \end{matrix}\\ \begin{matrix} = & -2{ u }_{ + }^{ 2 }+2{ u }_{ + } \end{matrix}$
* $\frac { d\left( -2{ u }_{ + }^{ 2 }+2{ u }_{ + } \right)  }{ d{ u }_{ + } } =-4{ u }_{ + }+2=0$
  ${ u }_{ + }=\frac { 1 }{ 2 }$ can obtain maximum value $1-{ \left( \frac { 1 }{ 2 }  \right)  }^{ 2 }-{ \left( \frac { 1 }{ 2 }  \right)  }^{ 2 }=\frac { 1 }{ 2 }$

#### 6.

* Normalized Gini index is
  $\frac { 1-{ u }_{ + }^{ 2 }-{ u }_{ - }^{ 2 } }{ \frac { 1 }{ 2 }  } =2\left( 1-{ u }_{ + }^{ 2 }-{ u }_{ - }^{ 2 } \right) =-4{ u }_{ + }^{ 2 }+4{ u }_{ + }=-4{ \left( { u }_{ + }-\frac { 1 }{ 2 }  \right)  }^{ 2 }+1$


* [a] 
  As stated in the problem, the normalized classification error is $2\min { \left( { u }_{ + },{ u }_{ - } \right)  }$

* [b] 
  $\begin{matrix}  & { u }_{ + }{ \left( 1-\left( { u }_{ + }-{ u }_{ - } \right)  \right)  }^{ 2 }+{ u }_{ - }{ \left( -1-\left( { u }_{ + }-{ u }_{ - } \right)  \right)  }^{ 2 } \end{matrix}\\ \begin{matrix} = & { u }_{ + }{ \left( 1-{ u }_{ + }+{ u }_{ - } \right)  }^{ 2 } \end{matrix}+{ u }_{ - }{ \left( -1-{ u }_{ + }+{ u }_{ - } \right)  }^{ 2 }\\ \begin{matrix} = & { u }_{ + }{ \left( 2-2{ u }_{ + } \right)  }^{ 2 }+\left( 1-{ u }_{ + } \right) { \left( -2{ u }_{ + } \right)  }^{ 2 } \end{matrix}\\ \begin{matrix} = & 4{ u }_{ + }\left( 1-{ u }_{ + } \right) =-4{ u }_{ + }^{ 2 }+4{ u }_{ + }=-4{ \left( { u }_{ + }-\frac { 1 }{ 2 }  \right)  }^{ 2 }+1 \end{matrix}$

  $\Rightarrow$ Maximum is $1$ when $u_+ = \frac{1}{2}$, $\Rightarrow$ Normalized form is $-4{ \left( { u }_{ + }-\frac { 1 }{ 2 }  \right)  }^{ 2 }+1$

* [c]
  Find the derivative on $u_+$
  $\begin{matrix}  & \frac { d\left( -{ u }_{ + }\ln { \left( { u }_{ + } \right)  } -{ u }_{ - }\ln { \left( { u }_{ - } \right)  }  \right)  }{ d{ u }_{ + } }  \end{matrix}\\ \begin{matrix} = & \frac { d\left( -{ u }_{ + }\ln { \left( { u }_{ + } \right)  } -\left( 1-{ u }_{ + } \right) \ln { \left( 1-{ u }_{ + } \right)  }  \right)  }{ d{ u }_{ + } }  \end{matrix}\\ \begin{matrix} = & -\ln { \left( { u }_{ + } \right)  } -1+\ln { \left( 1-{ u }_{ + } \right)  } +1 \end{matrix}\\ \begin{matrix} = & -\ln { \left( { u }_{ + } \right)  } +\ln { \left( 1-{ u }_{ + } \right)  } =0 \end{matrix}$

  $\Rightarrow$ Maximum entropy is $-\ln { \left( 2 \right)  }$ when $u_+=\frac{1}{2}$

  $\Rightarrow$ Normalized form is $\frac { { u }_{ + }\ln { \left( { u }_{ + } \right)  } +{ u }_{ - }\ln { \left( { u }_{ - } \right)  }  }{ \ln { \left( 2 \right)  }  }$

* [d]
  The maximum of $1-\left| { u }_{ + }-{ u }_{ - } \right|$ is $1$ when $u_+ = u_- = \frac{1}{2}$
  $\Rightarrow$ Normalized form is still $1-\left| { u }_{ + }-{ u }_{ - } \right|$

* The answer is [b]

### Experiments with Adaptive Boosting

* Training data: 100
* Testing data: 1000
* Iteration $T=300$, dimension $D=2$

#### 7.

* $E_{in}\left(g_1\right)=0.24$, $\alpha_1=0.57634$

#### 8.

* There is no obvious trend that $E_{in}\left(g_t\right)$ is decreasing or increasing, because it goes up and down frequently.

#### 9.

* $E_{in}\left(G_T\right)=0.0$

#### 10.

* $U_2=0.85417$, $U_T=0.00547$

#### 11.

* $\min\limits_{t}{\epsilon_t}=0.17873$, $t=1$

#### 12.

* $E_{out}\left(g_1\right)=0.29$

#### 13.

* $E_{out}\left(G_T\right)=0.132$

### Experiments with Unpruned Decision Tree

#### 14.

#### 15.

#### 16.