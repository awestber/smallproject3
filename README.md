# smallproject3
Predicting the Probability of Deaths from Health and Demographic Features
					Alyssa Westberg
The problem that I am solving is trying to figure out the probability of a death based on the patient's health and demographic features. I hope that this would in turn help future providers be able to help give patients an accurate prognosis. I want to do this by finding correlations between all the factors and then build a model to show what the predictions are. I then need to make sure that my program has accuracy, specificity and sensitivity. The ML method that was used was KNN.
The packages I have implemented are
Pandas- data manipulation
Sciklit-learn- machine learning
matplotlib/seaborn- data visualization
Dataset: Annual cause death numbers
Code Explanation
	To start I had to import all the libraries that are needed for the code. Then upload the dataset from the CSV file. I then renamed the columns of the dataset so that it was easier for the training and testing. There were a lot of different types of causes of death so I decided to group some of them together since I was running into the issue of having too many unique values with them all separated. I had to clean the data which means deleting the missing values. Then training the data and predicting, then doing the performance evaluations of accuracy, AUC, sensitivity, etc. Finally I did all the data visualization showing a bar graph of all the groupings compared to each other. Since the Non-communicable disease was the highest I had the program show a table for what each result was for that grouping.


Code Performance
	After running this code I am able to come to the conclusion that the most common types of causes of death are Non-communicable diseases like cardiovascular, dementia, and diabetes. By grouping the causes of deaths into categories I was also able to see what types needed to be further evaluated. To go more in depth within that grouping the table shows the actual results of deaths from that specific type of disease. The leading cause of death being cardiovascular and neoplasms (cancers). These visualizations help to show what areas of healthcare need more studying on how to decrease the rate of deaths. 

Performance Comparison
I was unable to do the performance comparisons between multiple VMs because my configurations were having issues and it was having issues with Java.


I also did this code on my local computer because of all the issues that I was encountering. I have been emailing back and forth with the TA about this and he had stated to just do it on the local computer to show what my code was. Since I had to do it this way I was not able to use Spark which I know was a big part of these projects, however I did not have two VMs to speak to since the configurations were not working and the TA could not figure out what was going on either. 

