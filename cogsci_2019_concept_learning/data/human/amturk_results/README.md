# Describing `spec.pkl`

```python
In [9]: import pickle

In [10]: with open('spec.pkl', 'rb') as f:
    ...:     spec = pickle.load(f)
    ...:

// There are 180 experimental trials associated with this dataset sample, stored in a (int->list)
dictionary of trial ID to trial info.
// Each participant should see exactly one of these trials.
In [11]: len(spec)
Out[11]: 180

// Each experimental trial has 8 tasks (corresponding to a task sampled from each of the 8 superordinate categories).
// The order of these tasks is already randomized.
In [12]: len(spec[0])
Out[12]: 8

// The task information is a dictionary containing the task parameters, and the names of
// The lists of training and test images should be randomized
In [13]: spec[0][0]
Out[13]:
{'condition': 'subordinate_condition',
 'sample node': 'n04380533',
 'test images': ['n04380533_3441.JPEG',
  'n04380533_1944.JPEG',
  'n03367059_9691.JPEG',
  'n03367059_6724.JPEG',
  'n03001627_1562.JPEG',
  'n03001627_3887.JPEG',
  'n04379243_4719.JPEG',
  'n03001627_7029.JPEG',
  'n04451818_7297.JPEG',
  'n13104059_1006.JPEG',
  'n13104059_5463.JPEG',
  'n01503061_10180.JPEG',
  'n04451818_12156.JPEG',
  'n13134947_4568.JPEG',
  'n13134947_6827.JPEG',
  'n04524313_5494.JPEG',
  'n04524313_18574.JPEG',
  'n04524313_7713.JPEG',
  'n01503061_11243.JPEG',
  'n03800933_2931.JPEG',
  'n04524313_5459.JPEG',
  'n04524313_1964.JPEG',
  'n03800933_22838.JPEG',
  'n01503061_11220.JPEG'],
 'training images': ['n04380533_2022.JPEG',
  'n04380533_10.JPEG',
  'n04380533_1290.JPEG',
  'n04380533_3.JPEG',
  'n04380533_4283.JPEG',
  'n04380533_1893.JPEG',
  'n04380533_6438.JPEG',
  'n04380533_4034.JPEG',
  'n04380533_4343.JPEG',
  'n04380533_9082.JPEG'],
 'trial type': '10_ex'}
```
