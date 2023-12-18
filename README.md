# Transformers

This repo documents my understanding of Transformers. 

Table of Contents:
1. [Introduction](#1)
2. [Transformers architecture](#2)
   1. [Encoder](#3)
      1. [Input Embeddings](#4)
      2. [Positional information](#5)
         1. [Positional encoding](#6)
         2. [Positional embedding](#7)
         3. [Positional encoding vs. positional embedding](#8)
      3. [The attention mechanism](#9)

<a name="1"></a>
## Introduction

Transformers have been proven to handle sequence-to-sequence tasks, particularly in NLP. Transformers, developed in 2017 by <a href="https://arxiv.org/abs/1706.03762">Vaswani et al.</a>, have become the backbone for many state-of-the-art models. Transformers are not just for NLP but also computer vision and computational biology. Transformers borrowed the transfer learning technique from computer vision, which allows us to use the weights from one task and apply them to a different but similar task for state-of-the-art performance. In computer vision, transformers do a great job by combining image and text representation to generate amazing art based solely on a single text description (like DALL-E2). 

Transformers are very efficient for tasks such as translation, question-answering, and generating human-level text. Previously, sequential tasks were based on recurrent or convolutional layers. Self-attention mechanism is the most important part of transformers, which can process input data (which will be converted to embeddings) in parallel which in turn, allows more efficient and more scalable models. 


<a name="2"></b>
## Transformers architecture 

At the core, transformers consists of two primary components: encoder and decoder (Fig. 1). Encoder helps the model understand the input in a sense of machine understanding, we can also think of the encoder as a translator that listens to some input and converts it into numbers that capture the important parts of the input. It processes the input data and extracts meaning from it and this meaning is represented by numbers in vectors called embeddings. On the other hand, decoder is in charge of generating the output. It takes the output of the encoder, which is the understanding of the input, and starts generating text token by token. The decoder takes the numbers produced by the encoder and converts them into words. This architecture allows for many tasks like translation, summarization, and image recognition. 


![](https://github.com/DanialArab/images/blob/main/Transformers/transformers%20architecture.png)

Fig. 1: Transformers architecture

Both encoders and decoders are implemented through layers. Although transformer architecture presented in the original paper consists of 6 layers in each encoder and decoder, modern transformers have different number of layers: for example, BERT (which is an encoder only model) has 12 and 24 layers in the small and large versions, respectively, the standard BART has 12 layers in each encoder and decoder, the original GPT has 12 decoder only layers while GPT-3 has 96 layers. The layers work in series like the output of one serves as the input of the next layer (Fig. 2). At the end of encoder, we have the encoder output, which goes to the decoder layer. 


![](https://github.com/DanialArab/images/blob/main/Transformers/encoder-decoder.png)

Fig. 2: Encoder-decoder architecture 

<a name="3"></c>
### Encoder 
 
A more detailed representation of the encoder and decoder is depicted in Fig. 3.

![](https://github.com/DanialArab/images/blob/main/Transformers/transformers%20architecture%20detailed.png)

Fig. 3: More detailed depiction of encoder and decore in transformers 

<a name="4"></d>
#### Input Embeddings

The first step in the encoder path is to convert the words into numbers. The words are first tokenized, i.e., each word is either totally or partially matched with a token, as shown in Fig. 4.


![](https://github.com/DanialArab/images/blob/main/Transformers/tokenization.png)

Fig. 4: Words tokenization 

Tokenozers are the bridge between humans and computers at the very basic level. Tokenizers just convert the words into numbers. So words are first converted into tokens and each token is associated with a number called a token ID (Fig. 5). 


![](https://github.com/DanialArab/images/blob/main/Transformers/tokenizer.png)

Fig. 5: Words first are converted into tokens which are associated numbers called token ID

This token ID usually goes from zero to the number of tokens that the model can handle. In the example shown in Fig. 4, the tokenizer takes each of the words and converts them into tokens. In this simple example, we assume each word is a token, which is not exact: the relationship between tokens and words can vary depending on the language and the specific content. As a rough estimate, 1000 tokens often correspond to around 700 words in English. This is just a general approximation, and the actual word count may vary based on factors such as sentence structure, vocabulary, and writing style. We have special tokens in the above example: one at the beginning and one at the end. Each one of these tokens is mapped with a number. So the words that are understandable to humans are now converted into some numbers that are understandable to machines. Each token ID is in turn associated with a vector called embedding. The set of vectors of all tokens is called the word embedding matrix, as shown in Fig. 6.


![](https://github.com/DanialArab/images/blob/main/Transformers/word%20embedding%20matrix.png)

Fig. 6: Word embedding matrix 

The word embeddings matrix, which is learned during the training of a transformer, has learned what each token means/represents. This matrix now contains the meaning of each word or each token. In any language, words carry context, sentiment, and syntax. Word embeddings encapsulate these attributes enabling algorithms to discern the relationships and meaning among words. From this perspective, word embeddings are almost at least one main repository of the knowledge of transformers. So every transformer has a word embedding. 

The word embeddings are like the look-up tables. Once we have matched each one of the token IDs with their vectors we now have vectors that represent each token of the input, which is called an input embedding (Fig. 6). 


![](https://github.com/DanialArab/images/blob/main/Transformers/vectorized%20lookup%20tables.png)

Fig. 7: Word embeddings are like the look-up tables

Embeddings in transformers are learned during training and they capture the nuances of the language as per the training objective. In transformers, the embedding layer is a look-up table that translates token IDs to their corresponding vectors, which is a very powerful look-up table. 

<a name="5"></d>
#### Positional information

Previous models like RNNs and LSTMs process sequences word by word, which helps the model to keep the order of the words in a sentence. However, transformers do not process the sequential information step by step and instead, they handle elements of a sequence all at once in parallel. To maintain the order of words in a sequence, transformers need to track the positional information. This is achieved by adding a unique number to each input embedding (the positional encodings have the same dimension as the input embeddings so that the two can be summed) (Fig. 8). 

![](https://github.com/DanialArab/images/blob/main/Transformers/positional%20embedding.png)

Fig. 8: Simplified diagram depicting positional encoding 

There are two approaches to retain this positional information:

+ Positional encoding, which was utilized in the original paper
+ Positional embedding, which was utilized in the more modern architectures 

The objective is to provide the model with information about the position of each individual token in the sequence. This positional information is crucial for the model to understand the sequence order when processing input embeddings. 

<a name="6"></d>
##### Positional encoding 

The original transformer used a mathematical function involving sines and cosines to perform positional encoding:

![](https://github.com/DanialArab/images/blob/main/Transformers/positional%20encoding%20original%20formula.png)

Fig. 9: Sine and cosine functions suggested in the original paper to perform positional encoding

where pos is the position and i is the dimension. Each position of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 X 2π. The positional encoding visualization is depicted in the <a href="https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers#positional-encoding-visualization">Positional encoding visualizer application</a>. 

<a name="7"></d>
##### Positional embedding 

In this approach the positional information is learned during training, offering adaptability and flexibility. Most modern transformers use positional embeddings. 

<a name="8"></d>
##### Positional encoding vs. positional embedding 

![](https://github.com/DanialArab/images/blob/main/Transformers/table.png)

Afterward, we add the positional encoding to the input embeddings to get the embedding matrix, as shown in Fig. 10.

![](https://github.com/DanialArab/images/blob/main/Transformers/input%20emdebbings%20plus%20positional%20encoding.png)

Fig. 10: Positional encoding + input embeddings = embedding matrix

<a name="9"></d>
#### The attention mechanism

here


References

 <a href="https://mlbootcamp.ai/course.html?guid=d105240a-94e1-405b-be80-60056659c24c">The Transformers Layer by Layer</a>
 
  










### from before needs to be polished 

+ Attention is all you need 
  + BLEU (Bilingual Evaluation Understudy) is a metric used for evaluating the quality of machine-generated text, especially in the context of machine translation. It was proposed as a metric for automatic evaluation of machine translation output. BLEU compares a candidate translation (produced by a machine translation system) to one or more reference translations (human-generated translations) by computing a similarity score.
  + The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder: In the context of the passage you provided, "sequence transduction" refers to the task of transforming one sequence of data into another sequence. This often occurs in the context of natural language processing (NLP) and machine translation.
  + Sequence modeling is a type of machine learning task where the goal is to generate or predict a sequence of values. **A sequence is an ordered set of elements, and this could be a sequence of words in a sentence, time-series data, musical notes, or any other ordered set of data points.** In the context of neural networks, sequence modeling involves training models to understand the dependencies and relationships between elements in a sequence. The models are designed to capture patterns, trends, and structures within the sequential data. **Sequence modeling is crucial for tasks where the order of elements in the data matters.**
  + Embedding: The concept of embeddings in computer vision is analogous to that in natural language processing or other domains. It involves transforming high-dimensional input data into a lower-dimensional space where meaningful patterns and relationships can be better captured and utilized by the model.
  + In the context of a U-Net architecture or similar encoder-decoder structures in computer vision, the **latent representation obtained after encoding is often referred to as an embedding.** The term "embedding" here denotes a **lower-dimensional representation of the input data that captures relevant features and patterns.**
  
  In the U-Net architecture:
  
  Encoder: This part of the network captures hierarchical features from the input image, gradually reducing the spatial dimensions while increasing the number of channels. The final output of the encoder is a condensed representation often referred to as the "latent space" or "embedding."
  
  Decoder: The decoder part of the U-Net takes this latent representation and reconstructs the original input or performs a specific task, such as segmentation. The decoder upsamples the latent representation to generate an output that matches the input size.
  
  **The latent representation obtained after encoding can indeed be considered an embedding because it represents the essential features of the input image in a more compact form.** This representation is expected to capture relevant information for the given computer vision task, making it easier for the decoder to generate accurate outputs.
  
  **The concept of embeddings in computer vision is analogous to that in natural language processing or other domains. It involves transforming high-dimensional input data into a lower-dimensional space where meaningful patterns and relationships can be better captured and utilized by the model.**

  + Embeddings in machine learning refer to the representation of objects or features in a **lower-dimensional space**. This is often done to capture meaningful relationships and similarities between the objects. Embeddings are used in various tasks such as natural language processing, computer vision, and recommendation systems. Here are some key points about embeddings in machine learning:

  + Word Embeddings: In natural language processing, words are often represented as high-dimensional vectors. Word embeddings are dense vector representations that capture semantic relationships between words. Techniques like Word2Vec, GloVe, and FastText are commonly used to learn word embeddings. These embeddings are trained on large corpora and can be used to find similarities between words.

  + When referring to a "lower-dimensional space" in the context of embeddings in machine learning, it does mean a representation with **fewer dimensions compared to the original space**. This reduction in dimensionality often serves several purposes:

Compact Representation: The lower-dimensional space is more compact, requiring fewer parameters to represent each object or feature. This compactness is useful for efficiency in storage and computation.

Semantic Meaning: The lower-dimensional space is designed to capture the essential semantic meaning or relationships between objects or features. The idea is to retain important information while discarding redundant or less informative details.

Generalization: Embeddings aim to generalize well to unseen data by capturing the underlying structure and patterns. The lower-dimensional space is expected to contain meaningful representations that facilitate generalization to new examples.

Computational Efficiency: Training models with lower-dimensional representations can be computationally more efficient, both in terms of memory usage and computational complexity. This efficiency is crucial, especially when dealing with large datasets and complex models.

It's important to note that the term "lower-dimensional" doesn't necessarily imply a decrease in the quality or informativeness of the representation. Instead, it indicates a more concise representation that retains the key characteristics of the data. The process of learning embeddings involves training a model to project the original high-dimensional data into this lower-dimensional space in a way that maximizes the utility of the representation for the given task.

In summary, "lower-dimensional space" in the context of embeddings refers to a representation that is more concise, retains essential information, and is computationally efficient for learning and generalization.


+ https://mlbootcamp.ai/myCourse.html?guid=d105240a-94e1-405b-be80-60056659c24c

  + Let's break down the concept of a sequence-to-sequence task further:

Sequence:

**A sequence is an ordered list of elements**. These elements could be anything like words, characters, or other data points depending on the context of the problem.

Task:

The task refers to what you want the machine learning model to do. In a sequence-to-sequence task, you want the model to perform an operation or generate an output sequence based on an input sequence.

Sequence-to-Sequence:

In a sequence-to-sequence task, the model takes an input sequence, processes it, and produces an output sequence. The lengths of the input and output sequences can vary.
Example - Machine Translation:

One common application is machine translation, where you input a sequence of words in one language and want the model to output a sequence of words in another language.

Input Sequence (e.g., English): "Hello, how are you?"

Output Sequence (e.g., French): "Bonjour, comment ça va?"

Encoder-Decoder Architecture:

This task often involves an encoder-decoder architecture. The encoder processes the input sequence and creates a condensed representation (context vector). The decoder then uses this context vector to generate the output sequence.

Encoder: Understands the input sequence.

Decoder: Uses the understanding to produce the output sequence.

In summary, a sequence-to-sequence task is a type of machine learning problem where the goal is to transform an input sequence into an output sequence, and the lengths of the sequences can vary. This is widely used in tasks like language translation, summarization, and more.


  + Self-attention mechanism can process input data in parallel allowing for more efficient and scalable models
  + We use masking in the masked attention in the decoder to prevent future information leakage
 
  + The Attention Mechanism: we have a self-attention module in the encoder and we have two: a masked attention module and another attention module, called the cross-attention module, in the decoder.
  + attention mechanism is called the heart or brain of the transformers. self-attention allows the model to weight the importance of different words in a sentence. more to lean!!!!
  +  
 


References:

https://mlbootcamp.ai/course.html?guid=d105240a-94e1-405b-be80-60056659c24c

https://www.youtube.com/watch?v=TQQlZhbC5ps

https://www.linkedin.com/posts/armand-ruiz_the-paper-that-changed-everything-in-ai-activity-7134149759048577024-n_Q6?utm_source=share&utm_medium=member_android
