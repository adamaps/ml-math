# Linear Algebra

Solving for unknowns within a system of linear equations.

There are only three options in linear algebra: `one`, `none`, or `infinite` solutions. Two different linear equations will therefore only intersect on a graph either once, none, or infinite times. It is impossible for linear equations to intersect other linear equations on a graph multiple times.

### Example:
- Sheriff has 180 km/h car
- Bank robber has 150 km/h car and a five-minute head start
- How long does it take the sheriff to catch the robber?
- What distance will they have travelled at that point?
- Ignore acceleration, traffic etc.

Convert to a system of linear equations
- 180 km/h = 3.0 km/min
- 150 km/h = 2.5 km/min
- Sherrif: d = 3(t - 5)
- Robber: d = 2.5t
- d = 3t - 15 and d = 2.5t
- 2.5t = 3t - 15

Therefore:
`t = 30 min` and `d = 75 km`

### In a system of linear equations:
- There could be many equations
- There could be many unknowns in each equation

Example:
We want to predict house prices based on a number of variables. This is a regression model. A deep learning model may consists of millions of rows `n` and millions of features `m` in an `n` x `m` matrix.

$\y1 = a + b_x11 + c_x12 + ... + m_x1m
$\y1 = a + b_x21 + c_x22 + ... + m_x2m
$\...
$\yn = a + b_xn1 + c_xn2 + ... + m_xnm

y: house price we want to predict
a: y-intercept (allows us to have an average over al other predicted house prices)
b: distance to school
c: number of bedrooms
m: any other relevant features
x: the values associated with each feature (inputs into the model)

We are trying to solve for the unknown paramters `a`, `b`, `c` and `m` in order to predict y.

Weights and biases are tuned by machine learning algorithms in order to map the input values into a predicted result.