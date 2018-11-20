1. Running Environment: 

at least python3 edition with matplotlib module installed.
 
if no installed matplotlib module, please use the command of 'pip -install matplotlib' for installation.  

2. How to run the program:
 
2.1 decompress the 'file.zip' to obtain the files: 

autos.arff     // data
ionosphere.arff //data
KNN_classification.py   // the program that can process the 					      classification task
WNN_classification.py   // Weighted KNN program that can 					          process the classification task
KNN_prediction_15.py       // the program that can process the 						prediction task
WNN_prediction_15.py // weighted KNN program that can process 				  the prediction task for 15 features
WNN_prediction_25.py // weighted WNN program that can process                 			       the prediction task for 25 features

2.2 run the program in Linux Command Line

open terminal, change current working directory into that with all the data files 

input command of "python3 <xxx.py>" to run the program of "<xxx.py>"

when terminal returns "Please wait..., program is running....".
This means that the program is running, please wait for output.

After a little while, the terminal can return a plotting graph with the title "trend of k from 0 to 50", which is the desired output.
When the running finishes, the terminal can return "program finished!"

If users want to run another program file, please close the current prediction plotting graph, or the program wll not stop!