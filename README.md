# SGNS: A new method for word2vec
## Put your data
Put your corpus data in "data/enwik/". For example:
```
data/enwik/enwik9
```
## Data Preprocessing
Enwik data has html labels such as <xxx=xxx>
So we need to handle the data by following commands:
```
cd data/enwik/
perl main_.pl enwik9 > enwik9.txt
```
Now we can train by enwik9.txt.
## Training
```
python test.py enwik9.txt 0
# zero means training without previous model
```
## Handle the data of your model
name your model data with the same name of your training data, in this example, should be "enwik9.txt"
```
cd enwik-200/
mv PS10iter_fromSVD_dim200_step5e-05_factors/ enwik9.txt
```
## Test Result
using exp2.py, the second argument is the MAX_ITER，it should be the same as the value in the your training.
The value was set in the last command in test.py
```
python exp2.py enwik9.txt 10
```
