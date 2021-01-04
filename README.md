# Offensive-Language-Detection-DK

This is the R-code for the bachelor project of Aske Bredahl & Johan Horsmans; "Offensive Language Detection in Danish". In this tutorial we will train the following models: Support Vector Machines, Naive Bayes, Logistic Regression, Random Forest, LSTM, Bi-LSTM, C-LSTM and a CNN. Furthermore, we will design the ensemble support system described in the assignment.

The supplied .md-file is the R-pipeline for the ensemble support system for the DKhate data in section 1 of the thesis. The model pipeline is the same for section 2 and 3 except that the data used for training/prediction varied.py

The python code for the thesis was run from a console, and, as such, it is not possible to commit that part of our pipeline to this repository. 

Furthermore, our fully expanded dataset is included in this repository as "fully_expanded_data.csv", with the following columns:
id: Unique ID for each comment
comment: The comment
class: The class of comments; Offensive (OFF) or non-offensive (NOT)
machine_man: Who labeled the comment; our model (machine) or humans (man).

The dataset contains a total of 106,369 comments (3,962 annotated by humans).
