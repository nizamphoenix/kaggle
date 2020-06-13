With the advent of artificial intelligence, computers have become adept at answering factoid questions, questions with only one or at most a few correct answers, and the answer is a single word token or a short noun phrase; but struggle to achieve a decent performance while answering subjective questions that require comprehension of the context like opinions, recommendations, or personal experiences.  

Humans are better at addressing such questions that require a deeper, multidimensional understanding of context - something computers aren't trained to do well, yet. Questions can take multiple forms, or intents, better known as linguistic level of pragmatics.  

This competetion faciliates using modern artificial intelligence tools, like deep learning and others developed in the field of Natural language processing to push the frontiers of question-answering efficacy of the machines, by working on diverse data from stackoverflow and its sister websites.  

More information about competition available [here](https://www.kaggle.com/c/google-quest-challenge).  

### Approach:-  
- DistilBERT was used to encode the raw text, BERTbase was overlooked dueto computational constraints.  
- 5-fold cross-validation was used to train the network to minimise error along with MultiTasknet since it was a multi-output regression problem.  
- Data consisted of *question title, question body, and answer body* of stackoverflow posts to predict scores of 30 variables, like *question_asker_understanding, question_expect_short_answer* etc
