# computer-vision-pj1
This is an implementation of two-layer mlp model without pytorch or tensorflow.

To train the model from scratch and test different parameters, delete line 155, 157 and 159 in file 'main.py', the best parameters for hidden layer dim, learning rate and weight decay ratio will be automatically chosen and trained model will be saved.

To directly use the trained model, delete '#' in line 163 and 164 in file 'main.py' and define xtest.
