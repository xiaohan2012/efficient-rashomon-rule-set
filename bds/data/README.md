# examples of loading (binarized) data


``` python
import sys
sys.path.append("./AIX360-master/")
from bds import data
ds = data.BinaryDataset('iris')
ds.load()
# access the data using ds.X_train_b, etc
```
