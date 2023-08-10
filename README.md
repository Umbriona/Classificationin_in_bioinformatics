# Methodology for making classification models for biological sequence data.
With the next generatin sequenceing technologies sequence data in biotechnology has grown exponentially. With this new vast amount of data it is important to know how to use this data effectivly.  While the volume of raw sequence data as increased, charictarised (labled) data has not increased the same way. Further more the labled data we do have is ofthen heavily unbalanced. It is impo

When evaluating classification/regression models for sequence data it is important to take the evolutionary nature of the data in to consideration. Since the evolutionary process will make the data points corrolate with one another. Thus one needs to correct for this when splitting the data. 

# Introduction

# Background
## Large language models

### Masked language task

### Large Language Models for Proteins
In recent years, the field of bioinformatics has witnessed remarkable advancements driven by the application of machine learning techniques. Among these, large language models have emerged as a powerful tool for analyzing biological data, particularly in the context of proteins. Just as these models have revolutionized natural language processing tasks, they have also shown promise in unraveling the intricate language of protein sequences and structures.

The architecture of large language models draws inspiration from the Transformer model, which has proven to be highly effective for capturing contextual relationships within sequences. These models, such as the BERT (Bidirectional Encoder Representations from Transformers) architecture, have been adapted and extended to accommodate the unique characteristics of protein sequences and structures. The underlying principle of these models lies in their ability to learn complex patterns and dependencies by considering the interplay of amino acids within a sequence or residues within a protein structure.

### Pretraining with Masked Sentence Tasks
The key to the success of large language models lies in their pretraining phase, during which they learn rich representations from massive amounts of unlabeled data. For proteins, this involves encoding sequence information to capture both local and long-range dependencies. The pretraining process involves masked sentence tasks, wherein a portion of the input sequence is masked, and the model is trained to predict the masked elements based on the surrounding context. This encourages the model to learn features that are essential for understanding protein sequences and structures.

In the context of proteins, the masked sentence task involves predicting missing amino acids within a sequence or residues within a protein structure. This process forces the model to grasp the nuances of sequence motifs, secondary structure elements, and potential functional sites. By training on large datasets, these models become adept at extracting intricate features that human-designed algorithms might miss.

### Utilizing Embeddings for Protein Analysis
Once pretrained, these language models can be fine-tuned for specific tasks, such as protein structure prediction, function annotation, or classification. An invaluable output of these models is the embeddings—vector representations of protein sequences or structures in a high-dimensional space. These embeddings encode the acquired knowledge about proteins, capturing their structural, functional, and evolutionary characteristics.

These embeddings can be harnessed for diverse downstream applications. For instance, researchers can use them to cluster similar proteins, classify protein families, predict protein functions, or assess structural similarities. The high-dimensional nature of these embeddings enables the model to differentiate subtle differences between proteins, leading to enhanced accuracy and performance across a range of tasks.

In conclusion, large language models designed for proteins leverage the power of pretraining with masked sentence tasks to learn rich representations from unlabeled data. These embeddings can then be utilized for various downstream analyses, offering new avenues for understanding protein sequences and structures in ways previously unattainable by traditional methods.


# Results

## Principal component analysis and dimensionality reduction of embeddings

To visulise the embeddings of the sequences we can project the dat on to its princial components. This will alow us to view the linear representations of the data containing the most information. In figure 1 we have plotted the embedded data in the first two prinsiple components


## Stochastic gradient decent regression

## Regression with neural network

## Resampling techniques

# Descussion

# Conclusion

# Methodology



## Protein Data Preprocessing
The protein dataset used in this study consisted of 60,000 protein sequences obtained from [source]. Prior to analysis, the dataset underwent initial preprocessing steps to ensure data quality and suitability for subsequent clustering and analysis. These preprocessing steps included [briefly describe any data preprocessing steps, such as sequence cleaning or feature extraction].

## Protein Clustering with CD-HIT
To reduce redundancy within the protein dataset and capture representative sequences, the CD-HIT software was employed for clustering. CD-HIT is a widely-used tool for sequence clustering based on sequence similarity, effectively grouping similar sequences into clusters while eliminating duplications. The clustering process was executed using a sequence similarity threshold of 95%, resulting in the formation of distinct clusters representing diverse protein sequences.

The first clustering step at 95% sequence similarity was intended to remove redundant sequences from the dataset, ensuring that only representative sequences were retained for downstream analysis.

## Construction of Independent Data Splits
For the purpose of model training, validation, and testing, an additional clustering step was performed using the CD-HIT tool. This time, the dataset was clustered at a lower sequence similarity threshold of 30%. The clusters obtained from this step were employed to construct independent data splits for training, validation, and testing.

Specifically, for the validation and test sets, a subset of clusters containing a single representative sequence was selected. The total number of selected clusters was set to 2000 for each of the validation and test sets. The remaining sequences, which were not included in the selected clusters, were designated as the training set.

The rationale behind this approach was to ensure that the validation and test sets were representative of the diversity present in the protein dataset while maintaining independence from the training set. The clustering-based data split construction enabled the model to be trained on distinct sequences that were not present in the validation or test sets, thereby preventing information leakage and ensuring unbiased evaluation.

## Extracting embeddings from ESM

The ESM-1v (Evolutionary Scale Modeling) model has demonstrated significant capabilities in capturing complex patterns and features within protein sequences. To extract embeddings from this model, we employed a process that involved utilizing the final layer's outputs to obtain a meaningful representation of each protein sequence.

The ESM-1v model was initially pretrained on a vast corpus of protein sequences, learning to encode various sequence motifs, structural properties, and evolutionary relationships. The final layer of the model contains high-dimensional embeddings that encapsulate these learned features. In our study, we extracted embeddings for individual protein sequences by focusing on the averaged embeddings from this last layer.

## Embedding Extraction Procedure
The following steps outline the procedure employed to extract embeddings from the ESM-1v model:

Data Preparation: The protein sequence dataset used in this study was preprocessed to ensure uniformity and suitability for input to the ESM-1v model. Preprocessing steps included [describe any data preprocessing steps, such as sequence encoding or padding].
Model Loading: The pretrained ESM-1v model was loaded into the analysis environment. This model had been trained on a diverse set of protein sequences, enabling it to capture a wide array of sequence features.
Embedding Extraction: For each protein sequence in the dataset, we fed the sequence through the ESM-1v model and extracted the embeddings from the last layer. Specifically, we retrieved the embeddings generated by the final layer, resulting in a sequence of high-dimensional vectors.
Averaging Embeddings: To obtain a single embedding vector for each protein sequence, we performed element-wise averaging of the embeddings extracted from the last layer. This process generated a representative embedding that encapsulated the sequence's essential features captured by the model.

Embedding Utilization in Analysis
The obtained averaged embeddings served as meaningful representations of the protein sequences. These embeddings were utilized in various downstream analyses, including [describe the specific analyses or tasks where the embeddings were used, such as classification, clustering, or similarity computation].

## Establishing base line performance of classical models

Logistic Regression Model as Baseline
To establish a baseline for comparison in our analysis, we employed a logistic regression model. Logistic regression is a fundamental and interpretable machine learning algorithm often used for binary classification tasks. In this study, we utilized logistic regression as a starting point to assess the predictive power of the features on the target variable.

Model Training
The following steps outline the process of training the logistic regression model:

Feature Selection: Relevant features were chosen based on [describe the criteria for feature selection, such as domain knowledge or data exploration].
Data Preparation: The dataset was divided into a training set and a separate validation or testing set. Preprocessing steps were applied to the data, which included [describe any data preprocessing steps, such as normalization or handling missing values].
Model Initialization: The logistic regression model was initialized, creating a linear combination of the selected features.
Model Training: The model was trained using the training set. During training, the algorithm adjusted the model's parameters to minimize the logistic loss function. This process aimed to find the optimal weights for each feature.

Model Evaluation
The performance of the logistic regression model was evaluated using appropriate metrics, such as accuracy, precision, recall, F1-score, or ROC-AUC, depending on the nature of the classification task. The evaluation process included the following steps:

Prediction: The trained model was used to make predictions on the validation or testing set.
Metric Calculation: The predictions were compared to the actual target labels, and relevant evaluation metrics were calculated to measure the model's performance.
Interpretation: The calculated metrics provided insights into the model's ability to correctly classify instances. Additionally, a confusion matrix was generated to visualize the distribution of true positives, true negatives, false positives, and false negatives.
Baseline Comparison
The performance metrics obtained from the logistic regression model served as a baseline against which more complex models were compared. This comparison allowed us to assess whether more advanced algorithms or feature engineering strategies provided significant improvements over the simple logistic regression approach.

## Training Neural network

Neural Network Architecture for Model Improvement
To enhance predictive capabilities beyond the baseline, a neural network was employed with a more intricate architecture. The neural network was designed to capture complex patterns and relationships within the data, thus enabling more advanced feature representations and improved predictive power.

The architecture of the neural network comprised the following layers:

The input layer, which accepted the selected relevant features for the classification task.
Three fully connected hidden layers, each equipped with 200 nodes. These hidden layers were integrated to capture increasingly intricate and abstract features within the data. Each hidden layer was accompanied by rectified linear unit (ReLU) activations, introducing non-linearity and allowing the network to capture intricate relationships.
Batch normalization was applied after each hidden layer to stabilize and expedite the training process. Dropout regularization was also incorporated to prevent overfitting, achieved by randomly deactivating a subset of neurons during each forward pass.
The output layer consisted of a single node with linear activation. This configuration was apt for regression tasks, facilitating the generation of continuous predictions.

Model Training and Hyperparameters
The neural network underwent training to optimize its weights and biases for accurate predictions. The training process encompassed the following steps:

Data Splitting: The dataset was partitioned into training and validation sets, with the independent test set reserved for the final evaluation. This partitioning facilitated model training and the subsequent assessment of its generalization performance.

Loss Function: A suitable loss function for regression tasks was chosen to quantify the disparity between predicted and actual target values.

Optimization: The Adam optimization algorithm was employed, along with a learning rate of 0.001, beta1 of 0.9, and beta2 of 0.99, to minimize the loss function and iteratively update the model's parameters.

Hyperparameter Tuning: Hyperparameters, such as dropout rate and batch size, were fine-tuned to promote model convergence and performance. The chosen batch size was 64.

Training and Early Stopping: Training was executed using the independent test set. The process employed early stopping to mitigate overfitting, halting training if validation loss ceased to improve.

Model Evaluation
The trained neural network's performance was evaluated using appropriate regression metrics, such as mean squared error (MSE) or mean absolute error (MAE), contingent on the task's nature. This evaluation process consisted of predicting values on the independent test set and calculating relevant regression metrics to gauge model accuracy and predictive power.

Comparison with Baseline
Performance metrics from the neural network were juxtaposed with those of the baseline model, providing insight into whether the neural network's heightened complexity led to significant improvements over the simpler logistic regression approach.

## Dealing with data with long tails

Over-sampling for Enhanced Recall Performance
To address the challenge of sparse samples and imbalanced distribution within the dataset, we implemented an over-sampling strategy aimed at boosting the recall performance in regions with limited data representation. Over-sampling involves increasing the instances of underrepresented classes or samples, thereby mitigating the class imbalance issue and allowing the model to better learn patterns from these sparse regions.

Data Distribution Analysis
Prior to the application of over-sampling, a comprehensive analysis of the data distribution was conducted. This analysis identified regions of the distribution that exhibited sparse sample representation. The goal was to target these specific areas for over-sampling, thus improving the model's ability to make accurate predictions for instances that would otherwise be overlooked due to the scarcity of data.

Over-sampling Procedure
The following steps outline the procedure employed for over-sampling:

Class Identification: The classes or regions of the data distribution with sparse samples were identified through the previously conducted data distribution analysis.
Sampling Strategy: Over-sampling was applied to these identified classes or regions by generating synthetic samples using various techniques. We employed random over-sampling, SMOTE (Synthetic Minority Over-sampling Technique), and ADASYN (Adaptive Synthetic Sampling) to create synthetic instances while preserving the inherent patterns within the original data.
Sampling Levels: Over-sampling was performed at various levels, allowing us to assess the impact of different levels of data augmentation on the model's performance. We considered increments such as 100%, 200%, and 300% over-sampling, signifying the proportionate increase in the number of synthetic instances added to the existing data.
Model Training and Evaluation
Following the over-sampling process, model training and evaluation were conducted to gauge the effect of the strategy on recall performance. The following steps elucidate the process:

Data Splitting: The dataset was divided into training, validation, and testing sets, ensuring that the original distribution of classes within each set was maintained.
Model Training: The model was trained using the over-sampled training data, utilizing appropriate machine learning algorithms such as neural networks, random forests, or support vector machines.
Evaluation Metrics: The evaluation process focused on assessing the recall performance in the regions of the data distribution with sparse samples. Evaluation metrics such as precision, recall, F1-score, and confusion matrices were computed to quantify the impact of
over-sampling on the model's ability to correctly classify instances from these regions.
