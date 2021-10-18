## GCNII code, 基于dgl
GCNII的官方代码实现：[GCNII官方实现](https://github.com/chennnM/GCNII) 

## 环境
```
dgl 0.6.1
pytorch 1.7.0
```

## 例子

```
>>> import dgl
>>> import numpy as np
>>> import torch as th
>>> # Case 2: Unidirectional bipartite graph
>>> u = [0, 1, 0, 0, 1]
>>> v = [0, 1, 2, 3, 2]
>>> g = dgl.bipartite((u, v))
>>> u_fea = th.rand(2, 5)
>>> v_fea = th.rand(4, 5)
>>> conv = GraphConvII(5, 2, norm='both', weight=True, bias=True)
>>> res = convii(g, (u_fea, v_fea))
```
