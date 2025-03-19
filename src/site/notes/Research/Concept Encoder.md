---
{"dg-publish":true,"permalink":"/research/concept-encoder/","created":"2025-03-16T18:06:00.805+01:00","updated":"2025-03-19T22:22:57.866+01:00"}
---

#research/concept-encoding #publish/seed 

Concept Encoder 🧠 💎

# Research ideas

This work tries to address the problem of generating the coherent text and chat responses with use of encoder-decoder or diffusion basedapproach. 
The main idea lies in intuition that just predicting the next word is not enough and written text has as deeper structure that could be represent better with encoder model which will try to uncover this deeper meaning by using the concept tokens instead word tokens. Then the decoder (autoregressive or diffusion) will try to generate the text based on the concept tokens.


In addition to properly represent the underlying meaning of the text, we want to increase the context length of the model by using the concept tokens.

Its better to multiply the concept tensor by the sequence tensor than the sequence tensor by itself because $\text{concept\_length} \ll \text{sequence\_length}$.
While computing the attention we need to multiply $(Q \cdot K^T) \cdot V$
