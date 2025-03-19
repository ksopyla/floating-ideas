---
{"dg-publish":true,"permalink":"/research/concept-encoder/","created":"2025-03-16T18:06:00.805+01:00","updated":"2025-03-19T22:28:17.341+01:00"}
---

#research/concept-encoding #publish/seed 

Concept Encoder 🧠 💎

# Research ideas

This work tries to address the problem of generating the coherent text and chat responses with use of encoder-decoder or diffusion basedapproach. 
The main idea lies in intuition that just predicting the next word is not enough and written text has as deeper structure that could be represent better with encoder model which will try to uncover this deeper meaning by using the concept tokens instead word tokens. Then the decoder (autoregressive or diffusion) will try to generate the text based on the concept tokens.


In addition to properly represent the underlying meaning of the text, we want to increase the context length of the model by using the concept tokens.

Its better to multiply the concept tensor by the sequence tensor than the sequence tensor by itself because $\text{concept\_length} \ll \text{sequence\_length}$.
While computing the attention we need to multiply $(Q \cdot K^T) \cdot V$

**Self attention computation** between the sequence "token" tensors, ends up with matrix multiplication: 

* In the case of the self attention we have $Q = K = V$ of shape $[sequence\_length, embed\_dim]$, we aim to have sequence_length be ~128K - 2M tokens, so the $Q \cdot K^T$ is very expensive operation in terms of time and memory $[128K \times embed\_dim] \cdot [embed\_dim \times 128K]$ - as a result we get $[128K \times 128K]$ matrix, 
* for 128k context length this needs storing $128*1024*128*1024/(1024*1024*1024)=16G$ float numbers


**Cross attention computation** between the concept and sequence "token" tensors, ends up with less expensive operation: 
* concept tokens are stored as $Q = [concept\_length, embed\_dim]$ where concept_length could be in a range of 32-2048, sequence "tokens" are stored as $K = V = [sequence\_length, embed\_dim]$ 
* this leads to matrix multiplication $[concept\_length \times embed\_dim] \cdot [embed\_dim \times sequence\_length]$ - as a result we get $[concept\_length \times sequence\_length]$ matrix which is **much smaller**

When we add batch dimension to the tensors we get:

$Q=[concept\_length, batch\_size, embed\_dim]$ 

$K=V=[sequence\_length, batch\_size, embed\_dim]$ 



This work was initially inspired by the papers:

* "Memory Transformer"
*  "ConceptBERT: A Concept-based Framework for Pre-training Language Models"
* "Large Concept Models" by Meta
* "LLaDA diffusion model"