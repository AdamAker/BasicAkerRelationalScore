[![CI](https://github.com/JuliaRegistries/General/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaRegistries/General/actions/workflows/ci.yml)

# BasicAkerRelationalScore
This is a dimensionality reduction algorithm which has the goal of maintaining interpretability i.e we eliminate variables directly from potential models that don't seem to add any predictive power. This is accomplished by the use of decision trees to approximate a function between two variables. This is a modified version of the [*Predictive Power Score*](#1) inspired by Florian Wetschoreck's article 

## The Problem
We'll start with a set of observations which can be further split into a set of features $F$ (**things we want to use to predict**) and targets $T$ (**things we want to predict**). The elements $x\in F$ and $y\in T$ are time-series of some measurable quantity. The main goal will be to minimize the set of features and targets we want to use to build models based on how well a feature does at predicting all the targets. How can we choose a good minimal set of observables to build models with? If we can potentially identify that there is a function between $x$ and $y$, then we can say that $x$ has predictive power with respect to $y$. So, how can identify if a function potentially exists between $x$ and $y$? which basically means we can build 

### Universal Function Approximators
Decision Trees are universal function approximators which basically means, we can split two dimensional subset of our data into different bins which are chosen based on minimizing a cost function. In this case the boundaries of the bins are chosen so as to minimize the error of the tree model makes when making predicitons. Spliting the data into different bins is constructing a function, but we need to understand how well this function does compared to a more naive model of prediction: taking the median of the target $y$ and always guessing that any $x$ will map to the median. 
 
### Comparing Model Performance
If we have two different models $g_1$ and $g_2$ mapping feature $x$ to  target $y$, then we will need a way to choose which model does a better job at predicting $y$ from $x$. One way to do this is to look at the mean absolute error of each model which is defined as

$$\text{MAE}=\sum\limits_{i=1}^{N}|y_i-g(x_i)|$$ 

We can compare the how well the "smart model" does as compared to the "naive model" by looking at the ratio 

$$r=\frac{\text{MAE}_{\text{smart}}}{\text{MAE}_{\text{naive}}}$$

as the smart model does better, this ratio becomes smaller and as the smart model starts doing as good or worse than the naive model, this ratio becomes larger. Up to this point, this is pretty much just the predictive power score. If our smart model is doing better than the naive model, then we have at least established that constructing a function between $x$ and $y$ is useful which means that we should include it in whatever models that we wish to build.

## Making the BARS 

There are a number of features that would be nice to have to make the process for judging how well a variable does at predicting another

The first thing we can do is to use a gaussian to map $r$ to $[0,1]$ so that we now have

$$e^{-r^2}$$

It'd be nice if when comparing how well a variable predicts itself, then its score would be $1$. We can make this happen by subtracting 

$$r_0=\frac{\text{MAE}_{\text{self}}}{\text{MAE}_{\text{naive}}}$$

from $r$ so that the score will be $1$. This makes sense because when we compare a variable to itself $r=r_0$ which means

$$e^{-(r-r_0)^2}$$

will become $1$. Finally, it'd be nice to be able to make it harder or easier for a feature-target pair to have a high BARS. This might be useful in case you'd like to see which variables tend to stick. This would make you feel more confident that there is a function between the feature-target pair. So we'll define the acceptence $\alpha$ as the hyperparameter which dictates how easy or difficult it is to get a high BARS. So, our final BARS is...

$$\text{BARS}(r,r_0;\alpha)=e^{-\frac{(r-r_0)^2}{\alpha^2}}$$



## References
<a id="1">[1]</a> 
Wetschoreck, Florian. (Apr 23, 2020). 
*RIP correlation. Introducing the Predictive Power Score.*
https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598

<a id="2">[2]</a>
Mathonline
*The Simple Function Approximation Theorem.*
http://mathonline.wikidot.com/the-simple-function-approximation-theorem

<a id="3">[3]</a>
kenndanielso Blog
*Universal Function Approximation.*
https://kenndanielso.github.io/mlrefined/blog_posts/12_Nonlinear_intro/12_5_Universal_approximation.html


