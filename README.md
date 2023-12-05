# Transformers

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
  + Encoder helps the model understand the input in the sense of machine understanding, we can also think of the encoder as a translator that listens to some input and converts it into numbers that capture the important part of the input, it processes the input data and extracts meaning from it, this meaning is represented by numbers in vectors called embeddings 
  + decoder is in charge of generating the output, it takes the output of the encoder, which is the understanding of the input, and starts generating text token by token, decoder takes the numbers produced by the encoder and converts them into words
  + Input embeddings: each word in the input is converted into a token and each token is associated with a number called token id. ?????????
  + Positional encoding: Transformers do not process sequences step by step, instead transformers process the inputs all at once in parallel. So to retain the order of the sequence transformers use something called positional encoding, which is basically a number that is added to the input embeddings, the goal is to provide the model with information about the position of each individual word actually each individual token in the sequence
  + Positional encoding is fixed b/c it comes from a deterministic function, that uses cousin and sin, while positional embedding is adaptive and learned during training and it is flexible
  + The Attention Mechanism: we have a self-attention module in the encoder and we have two: a masked attention module and another attention module, called the cross-attention module, in the decoder.
  + attention mechanism is called the heart or brain of the transformers. self-attention allows the model to weight the importance of different words in a sentence. more to lean!!!!
  +  
 


References:

https://mlbootcamp.ai/course.html?guid=d105240a-94e1-405b-be80-60056659c24c

https://www.youtube.com/watch?v=TQQlZhbC5ps
