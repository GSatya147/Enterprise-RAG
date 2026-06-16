### Embedding 
#### 1. The Mental Model
- Embedding is where most consequential architectural decisions live, which are expensive to reverse.  

**Mental model**  
- An embedding is a compression. you're taking a chunk of text, potentially hundreds of tokens and collapsing it into a single fixed-size vector, say 1024 numbers.   
- That vector has to somehow encode everything semantically relevant about that text so that when a query vector lands nearby in the same space, it means those two pieces of text are meaningfully related for the purposes of your specific task.  
- The key word is your *specific* task. embedding models are trained on specific data distributions with specific objectives. when you use one for your RAG pipeline you're betting that its notion of "semantically similar" aligns with what your users actually mean when they search. sometimes that bet is correct. often it isn't, and you won't know until you measure.  

**Asymmetry problem, yes it matters**  
- Queries and documents are fundamentally different kinds of text.
1. a query is short, often telegraphic, sometimes poorly formed 
2. a document chunk is longer, more formal, richer in context — a full paragraph from an earnings report.   
- If you embed both with the same model and compare cosine similarity, you're assuming the model maps both into the same semantic neighborhood. that assumption breaks more than you'd expect.

**Solution:**  
- `Asymmetric embedding`, where you use different prefixes or even different model endpoints like query embedding vs document embedding. 
- Models like `E5`, `BGE`, and `Instructor-XL` explicitly support this: you prepend "query: " to queries and "passage: " to documents, and the model was trained to produce embeddings that align across this asymmetry. 

**What a vector actually captures**  
- `Cosine similarity` in embedding space measures something like `topical relatedness.` it does not measure `factual correctness`, `temporal relevance`, `document authority`, or `logical entailment`. 
- Two chunks can be highly similar vectors while one is from 2019 and factually outdated, and the model has no way to know this. a chunk about "how to reset your password using the old portal" and a chunk about "how to reset your password using the new SSO system" may have nearly identical embeddings because the surface-level topic is identical. but one of them is wrong for current users.
- This is not a flaw you fix in embedding, it's a fundamental limitation of the approach that you compensate for at the retrieval and metadata filtering layer. knowing this is important because it tells you what embedding is responsible for `topical proximity` and what it isn't `factual currency`, `logical relevance`, `intent precision`.

#### 2. The Architecture choices that matter
**Single vector vs multi-vector (`bi-encoder` vs `colBERT`)**  


