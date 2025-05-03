# VAST 2021 - Question 2



**Characterize any biases you identify in these news sources, with respect to their representation of specific people, places, and events. Give examples. Please limit your answer to 6 images and 500 words.**


The process is as follows:

1. Train zero shot NER and sentiment analysis models for baslines
2. Pick the 300 lowest performers by score
3. Manually annotate them in doccano to create gold standard documents
4. Use these to fine-tune the models used in step 1
5. Deploy models to a flask backend which will take inputs from the frontend AKA the user and generates its NER and stance predictions


**The backend, frontend, and model training pipeline will live in this repository**




