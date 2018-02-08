# Deep FeedForward Networks
**Deep FeedForward Networks**은 **feedforward neural networks**와 **Multilayer Perceptrons**(**MLP**)라는 명칭을 사용하기도 하며, 주어진 입력값 X에 대한 결과값 Y를 구하는 함수 *f* 를 계산할 때 사용된다. 예시로, Linear regression에서 y = *f*(x) 라고 정의 할때, *f*(x)=*W*x+*b*이고, Feedforward Networks는 *W*의 값을 배움으로 써 주어진 입력x에 대한 알맞는 결과값 y를 구할 수 있다. 

FeedForward라고 불리는 이유는 입력값 x로 부터 함수 *f* 를 통해 결과값 y로 나올때 어떠한 정보들도 피드백(**Feedback**)되지 않기 때문이다. 만약 결과값 Y가 함수 *f* 에 연결되거나 함수 *f* 자체 내에서 Feedback이 사용된 경우 **Recurrent Neural Networks**로 정의된다. 

또한, Deep으로 불리는 이유는 함수 *f* 에 들어가는 값이 입력값 x뿐만 아니라 함수 *f* 의 결과값이 될수 있기 때문이다. 예로, ![equation](https://latex.codecogs.com/gif.latex?y%3Df%5E%7B%283%29%7D%28f%5E%7B%282%29%7D%28f%5E%7B%281%29%7D%28x%29%29%29) 함수 처럼, 함수의 입력값이 다른 함수의 값이 되어 각 함수들을 연결 시킬 수 있다.

이러한 구조는 신경망에서 가장 많이 쓰이고 3단계의 레이어(**Input Layer**, **Hidden Layer**, **Output Layer**)로 정의 할 수 있다. 예제에서, x는 입력값 레이어(**Input layer**), ![equation](https://latex.codecogs.com/gif.latex?f%5E%7B%281%29%7D)는 두번째 레이어, ![equation](https://latex.codecogs.com/gif.latex?f%5E%7B%282%29%7D)는 세번째 레이어, 마지막 레이어(y)는 결과값 레이어(**Output Layer**)라 부른다. 이때 입력값 레이어와 결과값 레이어 사이에 있는 레이어들이 **Hidden Layer**이다.

### Linear vs. nonLinear
Linear 모델(linear Regession, Logistic Regression)은 닫힌 형태의 수식 또는 convex형태의 그래프일 때 안정적이고 효과적인 결과값을 구할수있지만, 일차 함수에서만 사용 할 수 있고, 결국 linear 모델은 두개의 입력값 사이의 관계를 이해할 수 없다는 단점을 가지고 있다. 이를 극복하기 위해 linear 모델에 입력값 x를 바로 대입하지 않고 x를 커널 함수(**Kernel function**) ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)에 대입하여 nonlinear로 변경 후 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi%28x%29)를 모델의 입력값으로 대입한다. 이 함수 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 nonlinear transformation이라 한다.

![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 결정하는 3가지 방법이 존재한다:
1. Radial Basis Function(RBF) kernel과 같이 커널 머신에서 암묵적으로 사용하는 일반적인 함수를 사용한다. 이 함수의 차원이 크면 훈련 데이터(training set)를 전부 수용 할 수 있지만 실험 데이터(test set)에선 일반화가 재대로 이뤄지지 않는다. 이러한 함수들은 local Smoothness에만 기준을 두었기 때문에 과거의 정보를 충분히 내재하지 못하고 복잡한 문제를 해결하지 못하는 단점이 존재한다.

2. 다른 방법으론 개발자들이 직접 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 설계한다. Deep learning이 나오기 이전 기계학습에서 많은 개발자들이 사용하던 방법이다. 각 분야의 전문가들이 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 직접 구현해야 하는 단점이 존재한다.

3. Deep Learning은 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 학습을 통해 배운다. 이를 수식으로 보면 ![equation](https://latex.codecogs.com/gif.latex?y%3D%20f%28x%3B%5Ctheta%2C%20w%29%20%3D%20%5Cphi%28x%3B%5Ctheta%29%5E%7BT%7Dw)으로 표현한다. ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta)는 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 배울때 사용 되는 parameter이고 *w*는 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi%28x%29)에서 결과값을 산출할때 쓰이는 parameter이다. 딥 러닝에선 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)는 Hidden Layer라고 생각 할 수 있다. 이러한 방법은 3가지 방법들 중 유일하게 훈련시 볼록한(convex) 그래프가 되지 않지만, 이를 통해 얻을 수 있는 이점이 더 크다. ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 학습을 통해 배우는 방법은 위에 서술한 두가지 방법의 이점을 다 가지고 있다. ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 일반적인 함수(RBF)로 사용 할 경우 첫번째 방법과 유사하고, 개발자가 직접 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi)를 구현하면 두번쨰 방법과 유사하다. 두번째 방법과 다른 점은 개발자가 정확한 ![equation](https://latex.codecogs.com/gif.latex?%5Cphi) 함수를 정하지 않고 유사한 함수만 정해도 된다는 이점이 있다.

## Example: Learning XOR
FeedForward network를 좀 더 이해하기 쉽게 XOR함수를 학습하는 과정을 설명 하면서 위의 내용을 정리해 보자. XOR함수는 두개의 입력값이 서로 다를 때만 1을 반환하고 서로 다를 경우 0을 반환하는 함수이다.

입력값은 정리하면 ![equation](https://latex.codecogs.com/gif.latex?X%20%3D%20%7B%5B0%2C0%5D%2C%5B0%2C1%5D%2C%5B1%2C0%5D%2C%5B1%2C1%5D%7D)이고 모댈은 Linear 모델을 사용 하여 ![equation](https://latex.codecogs.com/gif.latex?f%28x%3Bw%2Cb%29%20%3D%20xw&plus;b)으로 수식을 정의 한다. loss/cost 함수는 Mean Square Error(MSE)를 사용하고 수식으로 다음과 같이 표현한다.

![equation](https://latex.codecogs.com/gif.latex?c%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7B4%7D%5Csum_%7Bx%5Cin%20X%7D%28f%5E%7B*%7D%28x%29%20-%20f%28x%3B%5Ctheta%20%29%29%5E%7B2%7D)

위의 모델에 입력값을 넣어 W와 b를 구하면 w = 0, b = ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D)이 된다. 결국 이 모델의 정확성은 0.5가 된다. 그 이유는 아래의 그림을 보면 알 수 있다.

![XOR graph](http://solarisailab.com/wp-content/uploads/2017/05/xor_limitation.gif)

AND와 OR의 경우 직선 하나로 두개의 결과값(0,1)을 구분 할수 있지만, XOR의 경우 직선 하나로 두개의 결과값을 구분 할 수 없다. 직선 하나로 구분 시 0과 1이 모두 들어 있기 때문에 정확성이 0.5일 수 밖에 없다. 이를 해결하기 위해 Hidden Layer 하나를 사용한 feedforward network를 구성해 보자. 이를 수식으로 표현하면 다음과 같다. 

![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20f%28h%3BW%2Cb%29%2C%20h%20%3D%20f%28x%3Bw%2C%20c%29)

![equation](https://latex.codecogs.com/gif.latex?%5Ctherefore%20y%20%3D%20f%5E%7B%282%29%7D%28f%5E%7B%281%29%7D%28x%3Bw%2Cc%29%3BW%2Cb%29)

Hidden Layer에서 산출한 값을 결과값으로 표현하기 위해 Activation 함수가 필요하다. **Hyper Tangent**(**tanh**) 또는 **Rectified linear Unit**(**ReLU**)함수가 가장 많이 쓰인다. 이 예제에서는 ReLU를 사용해서 결과값을 구하였다.

이제 만들어진 nonLinear 모델을 계산하여 결과값이 정확한지 확인 해보자.
각각의 wight와 bias를 정의하면 다음과 같다. b = 0 이다.

![equation](https://latex.codecogs.com/gif.latex?X%3D%5Cbegin%7Bbmatrix%7D%200%20%26%200%5C%5C%200%20%26%201%5C%5C%201%20%26%200%5C%5C%201%20%26%201%20%5Cend%7Bbmatrix%7D%2C%20w%3D%5Cbegin%7Bbmatrix%7D%201%20%26%201%5C%5C%201%20%26%201%20%5Cend%7Bbmatrix%7D%2C%20W%3D%5Cbegin%7Bbmatrix%7D%201%20%5C%5C%20-2%20%5Cend%7Bbmatrix%7D%2C%20c%3D%5Cbegin%7Bbmatrix%7D%200%5C%5C-1%20%5Cend%7Bbmatrix%7D)

먼저 *h*를 계산하면 

![equation](https://latex.codecogs.com/gif.latex?xw%3D%5Cbegin%7Bbmatrix%7D%200%20%26%200%5C%5C%201%20%26%201%5C%5C%201%20%26%201%5C%5C%202%20%26%202%20%5Cend%7Bbmatrix%7D)

구한 값에 c를 더하면,

![equation](https://latex.codecogs.com/gif.latex?xw%20&plus;%20c%3D%5Cbegin%7Bbmatrix%7D%200%20%26%20-1%5C%5C%201%20%26%200%5C%5C%201%20%26%200%5C%5C%202%20%26%201%20%5Cend%7Bbmatrix%7D)

구한 *h*값을 Activation 함수 ReLU에 적용 시키면,

![equation](https://latex.codecogs.com/gif.latex?ReLu%28h%29%3D%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%5C%5C%201%20%26%200%5C%5C%201%20%26%200%5C%5C%202%20%26%201%20%5Cend%7Bbmatrix%7D)

마지막으로 *ReLU*(*h*)에 *W*를 곱하면, 

![equation](https://latex.codecogs.com/gif.latex?hW%3D%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%201%20%5C%5C%201%20%5C%5C%200%20%5Cend%7Bbmatrix%7D)

모든 입력값의 결과가 재대로 나왔다. 단순 Linear에선 0.5의 정확성이 였지만, Hidden Layer를 사용한 FeedForward Networks에선 100%의 정확성을 보여준다. 현실에선 위와 같이 정확한 *wight*와 *bias*값을 찾긴 불가능 하지만, **Gradient Descent**를 사용하면 높은 정확성을 기대 할 수 있다. 

## Gradient-based Learning
Linear 모델과 신경망 모델의 가장 큰 차이점은 신경망의 비선형(nonlinearity) 특징 때문에 cost 함수의 그래프가 convex형태로 나타나지 않는다는 점이다. 결국 cost의 값이 매우 낮게 나오긴 하지만 linear 함수처럼 cost의 값이 0이 될 수는 없다. 또한, 신경망 모델에선 초기 parameter 값이 최종 결과값에 큰 영향을 미침으로 초기 값 설정을 알맞게 설정해야 한다.기본적으로 FeedForward Networks에선 weight값은 작은 무작위 값으로 설정하고 bias의 값은 0 또는 낮은 값으로 설정한다.

### Cost 함수
Deep 신경망에선 어떤 cost 함수를 사용하는지가 중요하다. 다행이도 신경망에서 사용되는 cost함수는 linear 모델이나 다른 parameter를 사용하는 모델이 사용하는 cost 함수와 유사하거나 같다. 대부분 입력값에 대한 결과값의 확율분포를 정하고 최대가능도(Maximum likelihood) 원리를 사용하여 계산한다. 또한, 좀더 쉽게 계산하는 방법으로 결과값의 전체 확율 분포를 계산해서 결과값에 대한 입력값 조건부 확율을 계산하기도 한다.

#### 최대가능도(Maximum likelihood)를 사용한 조건부 분포도(Conditional Distribution) 학습
현대 대부분의 신경망은 최대가능도 조건부확율을 사용하여 훈련을 한다. 최대가능도 조건불 확율을 간단히 설명하면 negative log-likelihood로 표현 할 수 있다. 익숙한 표현으론 negative cross-entropy이다. 수식으로 표현하면,

![equation](https://latex.codecogs.com/gif.latex?C%28%5Ctheta%20%29%20%3D%20-%5Cboldsymbol%7BE%7D_%7Bx%2Cy%7Dlog_%7Bp_%7Bmodel%7D%7D%28y%7Cx%29)

두가지 예로, Mean Square Error cost를 적용하면, 

![equation](https://latex.codecogs.com/gif.latex?C%28%5Ctheta%20%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%5Cboldsymbol%7BE%7D_%7Bx%2Cy%7D%7C%7Cy%20-%20%5Cboldsymbol%7Bf%7D%28x%3B%5Ctheta%20%29%29%7C%7C%5E%7B2%7D%20&plus;%20const)

이때 const는 Gaussian 분포도에 따라 변하며 분수 또한 ![equation](https://latex.codecogs.com/gif.latex?%5Ctheta)에 영향을 받지 않아, 위의 식과 동일하다 볼 수 있다.

Logistic regression의 cost 함수 또한 최대가능도를 활용한 함수이다:

![equation](https://latex.codecogs.com/gif.latex?C%28%5Ctheta%20%29%20%3D%20-%5Csum%20ylog%28xw%29%29&plus;%20%281-y%29log%281-wx%29%29)

최대가능도를 사용한 cost 함수의 장점은 각각의 모델에 맞춘 cost함수를 만들 필요성이 사라지는 것이다. 모델의 *P*(*y*|*x*)만 정해주면 cost 함수로 바로 적용 할 수 있다. 또한, log 함수를 사용 함으로 써 지수함수(exponential function)을 사용한 결과값 산출에서 발생되는 vanishing gredient 문제를 해결할 수 있다.

#### 조건부 통계(Conditional Statistics) 학습
주어진 입력값으로 결과값의 조건부 통계를 알고 싶을때 사용 되는 방법을 설명한다. 예를 들어 입력값 하나를 주어졌을때 결과값의 평균을 구하고자 할때가 있다. 특정한 특징을 parameter로 사용 하는 것이 아닌 연속성(countinuity) 또는 경계(boundedness)와 같은 특징을 사용하여 신경망의 cost 함수가 범함수(**Functional**)의 역활을 하는 방식으로 생각 할 수 있다. 정리하면, 범함수란 함수로부터 실제값을 구하는 함수로 생각 할수 있고 딥 러닝에서 parameter를 선택해 학습하는 개념이 아닌 함수를 선택해 학습하는 개념으로 생각 할 수 있다. cost 범함수를 설계할 때 cost의 값이 우리가 원하는 특정 함수일 때 최소가 될 수 있도록 한다. 최적화 문제(optimization)를 해결 하기 위해선 변분법(**Calculus of variations**)가 필요하다. 아래에서 두가지 최적화 방법을 설명 하겠다.

* 각 입력값 x로 결과값의 평균을 예측하는 함수를 구하는 방법

![equation](https://latex.codecogs.com/gif.latex?f%5E%7B*%7D%20%3D%20argmin%5Cboldsymbol%7BE%7D_%7Bx%2Cy%5Csim%20P_%7Bdata%7D%7D%7C%7Cy%20-%20f%28x%29%29%7C%7C%5E%7B2%7D)

또는 

![equation](https://latex.codecogs.com/gif.latex?f%5E%7B*%7D%28x%29%20%3D%20%5Cboldsymbol%7BE%7D_%7By%5Csim%20P_%7Bdata%7D%28y%7Cx%29%7D%5By%5D)

위의 식으로 표현 되며, mean squared error cost함수를 최소화 하면 각 입력값 x에 대한 y의 평균값을 예측하는 함수가 된다.

* 각 입력값 x로 결과값의 중앙값(median)을 예측하는 함수를 구하는 방법

![equation](https://latex.codecogs.com/gif.latex?f%5E%7B*%7D%20%3D%20argmin%5Cboldsymbol%7BE%7D_%7By%5Csim%20P_%7Bdata%7D%7D%5Cleft%20%5C%7C%20y-f%28x%29%29%20%5Cright%20%5C%7C)

위의 식은 각 입력값 x에 대한 결과값 y의 중앙값을 예측하는 함수이다. 이 cost 함수는 **mean absolute error**라 부른다.

불행히도, mean square error와 mean absolute error는 gradient-based 최적화시 안좋은 결과를 나타낸다. 이유는 이 두가지의 cost 함수를 사용 시 output units에서 gradients가 너무 작게 생성되어 학습이 재대로 이뤄지지 않는다. 그래서 *P*(y|x)의 전체 분포도가 필요하지 않는 상황에서도 위의 두 cost 함수보단 cross-entropy cost 함수를 더 많이 사용 한다.

### 결과값 유닛 (Output Unit)
cost 함수를 정하는 것은 output unit을 정하는 것과 상당히 밀첩한 관계가 있다. 보통, 데이터 분포도와 모델 분포도를 가지고 cross-entropy 함수를 사용 하여 cost 값을 구하고, 결과값을 어떻게 표현 하느냐에 따라 cross-entropy 함수의 유형이 정해진다. Layer를 설명 할때도 언급했듯이, 마지막 레이어가 output layer, hidden unit이 output unit이 될수 있고 output unit이 hidden unit이 될수 있다. 

#### 가우시안 분포를 사용한 Linear Unit
Linear output layer는 대부분 조전부 가우시안 분포도의 평균(conditional Gaussian distribution)을 나타낸다. log-likelihood를 최대화를 시킨 값은 mean squated error를 최소화한 값이랑 일치하고 최대가능도를 사용 해서 가우시안의 공분산(covariance)를 구할수 있다. 하지만, 공분산은 모든 입력값이 양수 matrix에 정의 되어야 하는 제한이 있다. 이러한 한계를 극복하기 위해 다른 output unit으로 공분산을 parameter로 적용 시킨다. 또한 linear unit은 gradient를 작게 만들지 않기 때문에 여러 최적화 알고리즘와 사용 할수 있는 장점이있다.

#### Bernoulli 분포를 사용한 Sigmoid Unit
결과값을 이진법으로 나타낼때 또는 Classification 문제를 해결하기 위해 사용된다.  Bernoulli 분포를 사용 하면 신경만은 오직 *P*(y = 1 | x)만 구하면 된다. 또한 이 확율이 알맞는 값이기 위해선 [0,1]사이의 값이여야만 한다. 만약 Linear unit을 사용한 경우 아래와 같은 식을 쓰고,

![equation](https://latex.codecogs.com/gif.latex?P%28y%20%3D%201%20%7C%20x%29%20%3D%20max%5C%7B0%2Cmin%5C%7B1%2Cwx&plus;b%5C%7D%5C%7D)

조건부 분도포(conditional distribution)에는 알맞지만 실제 훈련시 gadient descent가 재대로 작동하지 않는다. 그이유는 *wx*+*c*의 값이 [0,1]사이의 값에서 벗어나면 gradient는 0으로 변하게 되고, gadient가 0일때 학습 알고리즘은 더나은 parameter를 선택할 수 없는 상황이 되어버린다. 

따라서, Linear unit을 사용하기 보단 Sigmoid unit을 사용 하여 알맞는 답이 아닐시 gradient를 크게 만들어 학습을 시킬 수 있도록 하는 방법을 사용한다. 
