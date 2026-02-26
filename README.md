\documentclass[12pt, a4paper]{article}

% Essential packages
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{amsmath}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{parskip}

% Code formatting style
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegray},
    keywordstyle=\color{blue},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=4
}
\lstset{style=mystyle}

\title{\textbf{Binary Image Classification Using Transfer Learning}}
\author{}
\date{}

\begin{document}

\maketitle

\section{Introduction}
This report details the implementation of a binary image classifier (e.g., classifying images as either Cats or Dogs). Building a deep neural network from scratch requires immense computational power and massive datasets. To bypass this limitation, the provided codebase utilizes Transfer Learning, specifically adapting a highly efficient pretrained model called MobileNetV2. This document elucidates the underlying theory, details the libraries utilized, and provides a systematic, step-by-step breakdown of the implemented code.

\section{Theoretical Foundation}
Before examining the implementation, it is imperative to establish the three core theoretical concepts governing this architecture.

\subsection{Convolutional Neural Networks (CNNs)}
Deep learning models tailored for computer vision are known as CNNs. As an image propagates through a CNN, the model applies mathematical filters (convolutions) to detect spatial patterns. Early layers identify rudimentary features such as edges and color gradients, while deeper layers aggregate these to recognize complex, object-specific anatomical features.

\subsection{Transfer Learning}
Rather than training a CNN from random initialization (which necessitates substantial computational time and data), this approach leverages a model previously trained on millions of images. By "freezing" the pre-trained model's internal layers, its established feature-extraction capabilities are preserved. Only the final classification layer is replaced and optimized to address the specific binary classification problem.

\subsection{Binary Classification \& The Sigmoid Function}
The objective is to output a singular probability: $0$ for the first class (e.g., Cat) and $1$ for the second class (e.g., Dog). To achieve this, the final layer of the network utilizes a Sigmoid activation function. The mathematical formulation is:

\begin{equation}
S(x) = \frac{1}{1 + e^{-x}}
\end{equation}

This function maps any raw continuous numerical output from the network into a strict probability distribution spanning from $0.0$ to $1.0$. A standard decision boundary is applied: outputs strictly below $0.5$ are assigned to class 0, whereas outputs of $0.5$ or greater are assigned to class 1.

\section{Libraries Used}
The implementation relies on two foundational libraries within the Python data science ecosystem:

\begin{itemize}
    \item \textbf{TensorFlow (tf):} An open-source machine learning framework developed by Google. The high-level Keras API (\texttt{tf.keras}) is utilized to instantiate, compile, and train the neural networks.
    \item \textbf{Scikit-Learn (sklearn):} Utilized strictly for its \texttt{metrics} module. It computes the final evaluation metrics (Accuracy, Precision, Recall) via the \texttt{classification\_report} and \texttt{confusion\_matrix} functions.
\end{itemize}

\section{Code Breakdown and Explanation}
The following section outlines the step-by-step methodology for constructing and evaluating the model.

\subsection{Step 1: Loading the Pretrained Model}
\begin{lstlisting}[language=Python]
base_model = tf.keras.applications.MobileNetV2(input_shape=(128,128,3), include_top=False, weights='imagenet')
base_model.trainable = False 
\end{lstlisting}
\begin{itemize}
    \item \texttt{MobileNetV2}: The chosen base architecture, optimized for computational efficiency across standard and mobile hardware.
    \item \texttt{input\_shape=(128,128,3)}: Defines the expected input tensor dimensions: 128x128 pixels with 3 spatial color channels (RGB).
    \item \texttt{include\_top=False}: Excludes the original 1,000-class prediction head, retaining solely the feature-extraction layers.
    \item \texttt{weights='imagenet'}: Initializes the network with weights optimally learned from the ImageNet dataset.
    \item \texttt{base\_model.trainable = False}: Freezes the base model, preventing updates to the pre-learned weights during the training phase.
\end{itemize}

\subsection{Step 2: Building the Custom Classifier}
\begin{lstlisting}[language=Python]
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid') 
])
\end{lstlisting}
\begin{itemize}
    \item \texttt{Sequential}: Initializes a linear pipeline for data propagation.
    \item \texttt{base\_model}: Acts as the foundational feature extractor.
    \item \texttt{GlobalAveragePooling2D()}: Reduces the spatial dimensions of the base model's 3D output tensor by computing the average, resulting in a 1D feature vector.
    \item \texttt{Dense(1, activation='sigmoid')}: The final output layer consisting of a single neuron to output the binary probability.
\end{itemize}

\subsection{Step 3: Compiling the Model}
\begin{lstlisting}[language=Python]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
\end{lstlisting}
\begin{itemize}
    \item \texttt{compile}: Configures the model for training.
    \item \texttt{optimizer='adam'}: Employs the Adam optimization algorithm to iteratively update the network weights.
    \item \texttt{loss='binary\_crossentropy'}: The chosen objective function to compute the loss for the two-class classification task.
    \item \texttt{metrics=['accuracy']}: Specifies that the model should track and output classification accuracy during training.
\end{itemize}

\subsection{Step 4: Training the Model}
\begin{lstlisting}[language=Python]
history = model.fit(train_iterator, epochs=5, validation_data=val_iterator)
\end{lstlisting}
\begin{itemize}
    \item \texttt{fit}: Initiates the optimization process.
    \item \texttt{train\_iterator}: The generator yielding batches of training inputs and targets.
    \item \texttt{epochs=5}: Defines the number of complete passes through the training dataset. Due to Transfer Learning, convergence is typically achieved rapidly.
    \item \texttt{validation\_data}: Evaluates the model's generalization capabilities on unseen data at the end of each epoch.
\end{itemize}

\subsection{Step 5: Evaluation}
\begin{lstlisting}[language=Python]
val_iterator.reset()
y_true = val_iterator.classes
y_pred = (model.predict(val_iterator) > 0.5).astype("int32")

print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))
\end{lstlisting}
\begin{itemize}
    \item \texttt{reset()}: Re-initializes the validation generator index.
    \item \texttt{y\_true}: Extracts the ground-truth labels.
    \item \texttt{model.predict()}: Generates probability estimations for the validation set.
    \item \texttt{> 0.5}: Applies the threshold to convert continuous probabilities into binary class decisions.
    \item \texttt{.astype("int32")}: Casts boolean values to standard integer arrays for metric computation.
\end{itemize}

\section{Understanding the Output Metrics}
The \texttt{classification\_report} function yields a comprehensive evaluation table based on the following metrics:

\begin{table}[h]
\centering
\renewcommand{\arraystretch}{1.5}
\begin{tabular}{@{}lp{12cm}@{}}
\toprule
\textbf{Metric} & \textbf{Definition in this Context} \\ \midrule
\textbf{Accuracy} & The overall proportion of correctly classified images across the entire dataset. \\
\textbf{Precision} & Out of all images predicted as the positive class (e.g., "Dog"), the proportion that were actually the positive class. Quantifies false positives. \\
\textbf{Recall} & Out of all actual positive class images, the proportion successfully identified by the model. Quantifies false negatives. \\
\textbf{F1-Score} & The harmonic mean of Precision and Recall. It offers a balanced evaluation metric, particularly crucial when dealing with imbalanced datasets. \\ \bottomrule
\end{tabular}
\end{table}

\section{Conclusion}
By leveraging Transfer Learning via MobileNetV2, an efficient binary classification model was successfully constructed. Instead of randomly initializing weights and training over an extended duration, freezing the pre-trained ImageNet base allowed the network to act as a highly capable feature extractor immediately. The model solely required learning the final mapping to the specific target classes, yielding rapid convergence, high precision, and minimal computational overhead within just 5 training epochs.

\end{document}
