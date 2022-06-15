# Credit_Risk_Analysis

## Purpose

Fast Lending, a peer-to peer lending services company wants to use machine learning to predict credit risk. My assignment is to build several machine learning models and evaluate their performance on how well they predict credit risk.

## Overview of Machine Learning Model Metrics

In general, building a machine learning model involves 1) creating the model, 2) training the model, 2) creating predictions, and 4) validating the model.

The <ins>Balanced Accuracy Score</ins> is used to validate model performance; the score is the percentage of predictions that are correct. Per industry standard, an accuracy score of 70%-90% is considered ideal model performance (source: https://www.obviously.ai/post/machine-learning-model-performance).

<ins>Precision</ins>  attempts to answer the question ‘What proportion of positive identifications was actually correct?’ A low precision score indicates that there is a large number of false positives. 

<ins>Recall,</ins> also known as <ins>Sensitivity</ins>, attempts to answer the question ‘What proportion of actual positives were identified correctly?’ A low recall score indicates a large number of false negatives.

Highly *sensitive* models do well detecting the intended targets but risk resulting in a number of false positives. High *precision* models are more conservative in that the predicted positives are likely to be truly positive, but a number of other true positives may not be predicted. The ideal model would have a roughly equal balance of precision and recall.

The <ins>F1 score</ins> statistic is a summary of Precision and Recall; scores range between 1.0 (best) and 0.0 (worst). A significant imbalance between Precision and Recall produces a low F1 score. According to data scientist Stephen Allwright (https://stephenallwright.com/good-f1-score/), the general rule of thumb for F1 score results are as follows:

![F1_Rule_of_Thumb](https://user-images.githubusercontent.com/97558998/173933907-effeca18-4681-4dbf-b235-027cfd57cf93.png)

<ins>Support</ins> is the number of occurrences in the dataset. There were 87 high-risk loans and 17,118 low-risk loans in the dataset for this exercise, as shown in the Classification Reports below.

## Results

**1. Naive Random Oversampling Classification Report:**

>![NaiveRandomOversampling](https://user-images.githubusercontent.com/97558998/173934206-29e9fcf7-741a-41cf-9ea2-fb662c33144d.png)

- Balanced accuracy score – the accuracy score of the Naïve Random Oversampling model was 0.653, meaning that approximately 65% of predictions were correct:

>![NRO_balAcurScore](https://user-images.githubusercontent.com/97558998/173934376-9ee112ac-9270-4014-8753-f9b7c28343ad.png)

- Precision score – as shown on the classification report, the Precision score of the Naïve Random Oversampling model was low for high-risk loans (0.01), signifying a large number of false positives, and high for low-risk loans (1.0), meaning that the low-risk loans were predicted accurately.

- Recall score – as shown in the classification report, the Recall score of the Naïve Random Oversampling model was 0.63 for high-risk and 0.67 for low-risk loans.

**2. SMOTE (Synthetic Minority Oversampling Technique) Oversampling Classification Report:**

>![SMOTE](https://user-images.githubusercontent.com/97558998/173935446-80415e15-1def-4dec-87dc-f48801087463.png)

- Balanced accuracy score - the accuracy score of the SMOTE Oversampling model was 0.651, meaning that approximately 65% of predictions were correct – slightly less than the Naïve Random Oversampling model:

>![SMOTE_bas](https://user-images.githubusercontent.com/97558998/173935410-4f6813bd-b739-451c-827f-bd13d0262cdf.png)

- Precision score – the Precision score for the SMOTE Oversampling model was low for high-risk loans (0.01) and high for low-risk loans (1.0) – the same as the Naïve Random Oversampling model.

- Recall score - the Recall score of the SMOTE Oversampling model was 0.64 for high-risk and 0.66 for low-risk loans, similar to the Oversampling models.

**3. Undersampling Classification Report:**

>![Undersampling](https://user-images.githubusercontent.com/97558998/173935818-ce9db797-dfff-454f-9c16-b7f90a265c0e.png)

- Balanced accuracy score – the accuracy score of the Undersampling model was 0.651, meaning that approximately 65% of predictions were correct, similar to the above 2 models.

>![Undersample_bas](https://user-images.githubusercontent.com/97558998/173936008-5551df37-248d-4994-9589-9080517df70a.png)

- Precision score – the Precision score for the Undersampling model was low for high-risk loans (0.01) and high for low-risk loans (1.0), the same as the other models.

- Recall score - the Recall score of the Undersampling model was 0.61 for high-risk and 0.45 for low-risk loans, meaning that overall, this Undersampling model is less accurate than the Oversampling models.

**4. Combination (Over and Under) Sampling Classification Report:**

>![Combo](https://user-images.githubusercontent.com/97558998/173936265-efe24844-bb27-4088-aa14-8f083100f040.png)

- Balanced accuracy score – the accuracy score of the Combination model was 0.529, meaning that approximately 53% of predictions were correct – much less accurate than the above models.

>![Combo_bas](https://user-images.githubusercontent.com/97558998/173936714-4361842c-06d2-4d5f-8350-368a75f78bfb.png)

- Precision score – the Precision score for the Combination model was low for high-risk loans (0.01) and high for low-risk loans (1.0), the same as the above models.

- Recall score - the Recall score of the Combination model was 0.69 for high-risk and 0.60 for low-risk loans, meaning that overall, this Combination model is less accurate than the Oversampling models.

**5. Balanced Random Forest Classifier Classification Report:**

>![BRF](https://user-images.githubusercontent.com/97558998/173937470-68c07354-7595-4a11-a31c-0cceae0d8036.png)

- Balanced accuracy score – the accuracy score of the Balanced Random Forest Classifier model was 0.789, meaning that approximately 79% of predictions were correct – substantially better than the above models.

>![BRF_bas](https://user-images.githubusercontent.com/97558998/173937558-b22ba3b1-4d53-466a-a80d-16ef94bdb78f.png)

- Precision score – the Precision score for the Balanced Random Forest Classifier model was low for high-risk loans (0.03) and high for low-risk loans (1.0), similar to the above models.

- Recall score – the Recall score of the Balanced Random Forest Classifier model was 0.87 for high-risk and 0.70 for low-risk loans, meaning that overall, this Balanced Random Forest Classifier model is substantially more accurate than the other models.

**6. Easy Ensemble AdaBoost Classifier Classification Report:**

>![Easy](https://user-images.githubusercontent.com/97558998/173937785-1c28d859-53b9-4854-905a-6eee29d4815f.png)

- Balanced accuracy score – the accuracy score of the Easy Ensemble AdaBoost Classifier model was 0.931, meaning that approximately 93% of predictions were correct – substantially better than all other models.

>![Easy_bas](https://user-images.githubusercontent.com/97558998/173938038-db3d46e5-5d1b-4887-84a2-80352aceb538.png)

- Precision score – the Precision score for the Easy Ensemble AdaBoost Classifier model was low for high-risk loans (0.09) and high for low-risk loans (1.0), similar to the above models.

- Recall score – the Recall score of the Easy Ensemble AdaBoost Classifier model was 0.92 for high-risk and 0.94 for low-risk loans, meaning that overall, this Easy Ensemble AdaBoost Classifier model is substantially more accurate than the other models.

## Summary

In summary, metrics for all models:

>![Summary stats](https://user-images.githubusercontent.com/97558998/173938367-b1704514-e8ec-4b78-a385-393c01989ab2.png)

The Easy Ensemble AdaBoost Classifier would be the best model to choose for this dataset, with an accuracy rate of 93%. However, the model still has a low precision score for high-risk loans, and even though it’s better than the other models, we would still see a large number of false positives for high-risk loans. The high recall scores for both high- and low-risk loans indicate that we would see a very few false negatives.





