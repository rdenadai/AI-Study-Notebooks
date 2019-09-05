# Q-Learning

![RL Schema : Sutton & Barto](https://github.com/rdenadai/AI-Study-Notebooks/tree/master/images/simple_RL_schema.png)

Q-Learning equation:

<img src="https://latex.codecogs.com/svg.latex?Q(s_t,&space;a_t)&space;\leftarrow&space;((1&space;-&space;\alpha)&space;\cdot&space;Q(s_t,&space;a_t))&space;&plus;&space;(\alpha&space;\cdot&space;(r_t&space;&plus;&space;\gamma&space;\cdot&space;max(Q(S_{t&plus;1},&space;a))))" title="Q(s_t, a_t) \leftarrow ((1 - \alpha) \cdot Q(s_t, a_t)) + (\alpha \cdot (r_t + \gamma \cdot max(Q(S_{t+1}, a))))" />


## Running

```bash
$> python -m code.qlearning.train
```

```bash
$> python -m code.qlearning.play
```

## Running agent




## References

 - [Reinforcement Learning (Sutton & Barto)](http://incompleteideas.net/book/RLbook2018.pdf)

 - [A Beginner's Guide to Deep Reinforcement Learning](https://skymind.ai/wiki/deep-reinforcement-learning)

 - [Deep Reinforcement Learning Course](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)

 - [Reinforcement Learning in Pacman](http://cs229.stanford.edu/proj2017/final-reports/5241109.pdf)

 - [Introduction to Reinforcement Learning (Coding Q-Learning) — Part 3](https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0)

 - [Reinforcement learning: Temporal-Difference, SARSA, Q-Learning & Expected SARSA in python](https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e)