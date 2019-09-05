<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.6/MathJax.js?config=TeX-AMS_HTML-full"></script>

# Q-Learning

![RL Schema : Sutton & Barto](https://github.com/rdenadai/AI-Study-Notebooks/tree/master/images/simple_RL_schema.png)



## Running

```bash
$> python -m code.qlearning.train
```

```bash
$> python -m code.qlearning.play
```

## References

 - [Reinforcement Learning (Sutton & Barto)](http://incompleteideas.net/book/RLbook2018.pdf)

 - [Deep Reinforcement Learning Course](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)

 - [Reinforcement Learning in Pacman](http://cs229.stanford.edu/proj2017/final-reports/5241109.pdf)

 - [Introduction to Reinforcement Learning (Coding Q-Learning) — Part 3](https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0)

 - [Reinforcement learning: Temporal-Difference, SARSA, Q-Learning & Expected SARSA in python](https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e)