Ronald Queensen

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
