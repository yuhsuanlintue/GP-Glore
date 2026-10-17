# GP-Glore


## Install

Install uv package manager

``` ps
uv lock --upgrade
uv sync
```

then run 
``` ps
uv run test.py
```

you should see two table being printed 
=====================================================================================
method         DFS-Full DFS-Simple GP-GLORE speedup vs DFS-Full speedup vs DFS-Simple
dataset  depth                                                                       
adults   2        18.86      18.56     1.82               10.36                 10.19
         3        345.4     337.18     3.38              102.27                 99.84
         4         >780       >780     7.62              102.36                102.36
         5        >1200      >1200     16.2               74.07                 74.07
credit-g 2          1.4       0.27     0.32                4.38                  0.85
         3        11.11       3.35     0.99               11.17                  3.37
         4        47.45      18.11     2.43               19.51                  7.45
         5       122.51      58.46     4.78               25.65                 12.24
diabetes 2         0.24       0.03     0.03                8.33                   1.1
         3         0.35        0.1     0.09                3.85                  1.12
         4         0.36       0.11      0.1                3.74                  1.18
         5         0.34       0.11     0.09                3.73                  1.21
spambase 2        17.39        1.6     6.44                 2.7                  0.25
         3        360.0      59.77    83.78                 4.3                  0.71
         4         >360       >360    360.0                 1.0                   1.0
         5         >690       >690     >690                 1.0                   1.0

geometric mean speedup vs DFS-Full:
depth
2     5.65
3    11.73
4     9.30
5     9.17

geometric mean speedup vs DFS-Simple:
depth
2    1.24
3    4.05
4    5.48
5    5.76