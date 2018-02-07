# Deep FeedForward Networks
**Deep FeedForward Networks**은 **feedforward neural networks**와 **Multilayer 		Perceptrons**(**MLP**)라는 명칭을 사용하기도 하며, 주어진 입력값 X에 대한 결과값 Y를 구하는 함수 *f* 를 계산할 때 사용된다. 예시로, Linear regression에서 y = *f*(x) 라고 정의 할때, *f*(x)=*W*x+*b*이고, Feedforward Networks는 *W*의 값을 배움으로 써 주어진 입력x에 대한 알맞는 결과값 y를 구할 수 있다. 

FeedForward라고 불리는 이유는 입력값 x로 부터 함수 *f* 를 통해 결과값 y로 나올때 어떠한 정보들도 피드백(**Feedback**)되지 않기 때문이다. 만약 결과값 Y가 함수 *f* 에 연결되거나 함수 *f* 자체 내에서 Feedback이 사용된 경우 **Recurrent Neural Networks**로 정의된다. 

또한, Deep으로 불리는 이유는 함수 *f* 에 들어가는 값이 입력값 x뿐만 아니라 함수 *f* 의 결과값이 될수 있기 때문이다. 예로, ![equation](https://latex.codecogs.com/gif.latex?y%3Df%5E%7B%283%29%7D%28f%5E%7B%282%29%7D%28f%5E%7B%281%29%7D%28x%29%29%29) 함수 처럼, 함수의 입력값이 다른 함수의 값이 되어 각 함수들을 연결 시킬 수 있다.

이러한 구조는 신경망에서 가장 많이 쓰이고 3단계의 레이어(**Input Layer**, **Hidden Layer**, **Output Layer**)로 정의 할 수 있다. 예제에서, x는 입력값 레이어(**Input layer**), ![equation](https://latex.codecogs.com/gif.latex?f%5E%7B%281%29%7D)는 두번째 레이어, ![equation](https://latex.codecogs.com/gif.latex?f%5E%7B%282%29%7D)는 세번째 레이어, 마지막 레이어(y)는 결과값 레이어(**Output Layer**)라 부른다. 이때 입력값 레이어와 결과값 레이어 사이에 있는 레이어들이 **Hidden Layer**이다.

## Linear Vs. nonLinear
Linear 모델(linear Regession, Logistic Regression)은 닫힌 형태의 수식 또는 convex형태의 그래프일 때 안정적이고 효과적인 결과값을 구할수있지만, 일차 함수에서만 사용 할 수 있고, 결국 linear 모델은 두개의 입력값 사이의 관계를 이해할 수 없다는 단점을 가지고 있다. 이를 극복하기 위해 linear 모델에 입력값 x를 바로 대입하지 않고 x를 커널 함수(**Kernel function**) \phi(x)로 변경 후 \phi(x)를 모델의 입력값으로 대입한다. 이 함수 \alpha를 nonlinear transformation이라 한다. 
