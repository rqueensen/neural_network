This is a neural network designed to classify hand-written digits. The test data was
provided by the class via a kaggle contest, I have had to remove the test data
due to school IP rules. All of this code was written by myself, so by school rules it
is my IP to make public. 

This neural network was found to be 97.5% accurate in identifying which number a 
hand-written digit was meant to be on kaggle, putting the network in the top 10% of
my class. 

All of the code is contained in main.py. It uses libraries from Anaconda 2.4.1, 
using the latest version of Anaconda causes the loadmat("training_data") to crash.
Anaconda3-2.4.1-Windows-x86_64.exe cane be found here:
https://repo.continuum.io/archive/index.html

Copy main.py into the directory containing the Training and Testing Data. 


python main.py train

Prints info to the console for training/validation accuracy per epoch, the graph
was made via excel. Each epoch takes about 4 minutes, so testing from 1 to 10 epochs
takes a very long time. 



python main.py kaggle

Produces the file predict.csv which contains the predictions for
the testing data. Runs on 50 epochs, took about 3 hours on my machine.
