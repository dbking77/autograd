# Overview
Minimal pytorch-like automatic gradient (autograd) compution system.

## Gradient
For certain calculations (AI/ML) automatically computing the gradient of an
equation with respect its inputs is a very useful feature. 

For example a autograd system would be able to compute that the gradient of *OUT* with respect to *A* is 2.0
and the gradient with respect to *B* is 3.0  
```
OUT = A*2 + B*3
```

## Computation Graph
For a pytorch-like system to compute gradients it mast create a computation graph.
This graph tracks how the output was calculated from different inputs.
A graph for ```OUT = A*2 + B*3``` might look like
```
A-->(*)--
     ^   \
     2   (+)-->OUT
         /
B-->(*)--
     ^
     3
```

With the computation graph it is possible to back-propogate how the output is effected
by each input.

## Merging Backprop Paths
With some equations, the compatation graph is simply a tree.  
Because of this, there is only a single path from the output back to an input.

However, in many cases the equation is actually a graph where the may be many paths from the
output back to any input.

With the following equation there are multiple path from output *OUT* to itermediate value *B*
```
B = A*3
OUT = (B+1) * (B+2)
```

```
            (+)---
           / ^    \
A-->(*)-->B  1    (*)-->OUT
     ^     \      /
     3      (+)---
             ^
             2
```

When back-propogating the gradient from *OUT* to *B* there are multiple paths, so gradient
must be computed over each path and accumulated at *B*.

When back-propogating from *OUT* to *A* there are also multiple paths, however all those route go through
*B* before reaching *A*.   Ideally the back-propgation would only compute *A* gradient from the single path from *B* (saving computation time)
This can be done by have a counting pass that effectively counts the branches from any intermediate operation.




