
This is an outline of what i did in the project.
Simple Description : The questions in Q&A column of Stack Overflow website are present in the dataset. One more dataset including all the available tags in the website is collected. Now for an easy intrepretation, Using Multinomial Navie Bayes the tags are tagged to Questions.
In this text Classification and Tag Identification Project, I worked out the following things:
1.Data Collection : Read the dataset into Pandas Dataframe
2. Data Wrangling: removed unnecessary columns. Cleared NA values and coverted the text into simple means.
3. Vectorizing : Used Countvectorizer to convert the text into vectors.
4. Building the Model : Used MULTINOMIAL NAIVE BAYES model for this DATASET
5. Prediction and Metrics : Predictions are made on test dataset and Metrics such as Accuracy are evaluated.
