# Deep FeedForward Networks
**Deep FeedForward Networks**은 **feedforward neural networks**와 **Multilayer 		Perceptrons**(**MLP**)라는 명칭을 사용하기도 하며, 주어진 입력값 X에 대한 결과값 Y를 구하는 함수 *f* 를 계산할 때 사용된다. 예시로, Linear regression에서 y = *f*(x) 라고 정의 할때, *f*(x)=*W*x+*b*이고, Feedforward Networks는 *W*의 값을 배움으로 써 주어진 입력x에 대한 알맞는 결과값 y를 구할 수 있다. 

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

입력값은 정리하면 ![equation](https://latex.codecogs.com/gif.latex?X%20%3D%20%7B%5B0%2C0%5D%2C%5B0%2C1%5D%2C%5B1%2C0%5D%2C%5B1%2C1%5D%7D)이고 모댈은 Linear 모델을 사용 하여 ![equation](https://latex.codecogs.com/gif.latex?f%28x%3Bw%2Cb%29%20%3D%20xw&plus;b)으로 수식을 정의 한다. loss/cost 함수는 Mean Square Error(MSE)를 사용하고 수식으론

![equation](https://latex.codecogs.com/gif.latex?c%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7B4%7D%5Csum_%7Bx%5Cin%20X%7D%28f%5E%7B*%7D%28x%29%20-%20f%28x%3B%5Ctheta%20%29%29%5E%7B2%7D)

으로 표현한다. 

위의 모델에 입력값을 넣어 W와 b를 구하면 w = 0, b = ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2%7D)이 된다. 결국 이 모델의 정확성은 0.5가 된다. 그 이유는 아래의 그림을 보면 알 수 있다.

![XOR graph](http://solarisailab.com/wp-content/uploads/2017/05/xor_limitation.gif)

AND와 OR의 경우 직선 하나로 두개의 결과값(0,1)을 구분 할수 있지만, XOR의 경우 직선 하나로 두개의 결과값을 구분 할 수 없다. 직선 하나로 구분 시 0과 1이 모두 들어 있기 때문에 정확성이 0.5일 수 밖에 없다. 이를 해결하기 위해 Hidden Layer 하나를 사용한 feedforward network를 구성해 보자. 이를 수식으로 표현하면, 

![equation](https://latex.codecogs.com/gif.latex?h%20%3D%20f%28x%3Bw%2Cc%29%2C%20y%20%3D%20f%28h%3Bw%2Cb%29)

![equation](https://latex.codecogs.com/gif.latex?y%20%3D%20f%5E%7B%282%29%7D%28f%5E%7B%281%29%7D%28x%3Bw%2Cc%29%3BW%2Cb%29%29)


## Gradient-based Learning
Linear 모델과 신경망 모델의 가장 큰 차이점은 신경망의 비선형(nonlinearity) 특징 때문에 cost 함수의 그래프가 convex형태로 나타나지 않는다는 점이다. 결국
