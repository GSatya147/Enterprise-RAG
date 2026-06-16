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
- *bi-encoder:* one vector per chunk, one vector per query, cosine similarity between them. fast, scalable, cheap to store. the problem is information compression, you're collapsing potentially 512 tokens into one vector. information is lost. two chunks about the same topic but with different specific claims will end up close in vector space, and the model has no way to tell them apart.
- *`colBERT` and (`colBERTv2`, `ModerncolBERT` in 2026):* instead of one vector per chunk, you store one vector per token in the chunk. at query time you also get one vector per query token. relevance score is computed by taking each query token's embedding and finding its maximum similarity against all document token embeddings, called MaxSim. then you sum these across all query tokens.
- *intuition:* so instead of asking `Is this chunk broadly about the same as query?`, you are asking `For every word in this query does the document has something that matches?`. It's much fine-grained.
- The cost is storage: say a 512 token chunk with 128-dim colBERT model at 4 bytes takes `512 x 128 x 4 = 256 kB` compared to a 1516-dim vector chunk at float 32 is `6 KB`, thats roughly `40x`.
- `colBERT` introduced some compression techniques (centroid + residuals) to mitigate this. PLAID engine makes ANN search over multi-vector practical.  
- *Note:* `BGE-M3` is notable here because it's the one open-source model that produces all three simultaneously: dense single-vector embeddings, sparse lexical weights (like `BM25` but learned), and ColBERT-style multi-vector embeddings from the same model. it's a strong choice when you want to experiment with hybrid retrieval without managing multiple separate models.

**Matryoshka embeddings (underused)**  
- Matryoshka Representation Learning (MRL) is the idea that you train a model such that the first N dimensions of the embedding are themselves a valid, high-quality embedding. the model front-loads the most semantically important information into early dimensions. 
- *The result:* you can truncate a 3072-dimensional embedding to 256 dimensions and retain roughly 93-95% of retrieval quality, at one-sixth the storage and much faster dot-product computation.  
- OpenAI's `text-embedding-3-large` and `text-embedding-3-small` were trained with MRL. you literally pass a dimensions parameter to the API and get whatever size you want. `Cohere` and `Jina` also support this.  
- The production implication is significant. if you're storing 100 million vectors at 3072 dims float32, that's about 1.2TB. at 256 dims you're at roughly 100GB. same retrieval quality, order of magnitude less storage and faster ANN search. the decision to use MRL embeddings should be made at the architecture stage, not discovered later when the vectorDB bill arrives.

**The MTEB trap**
- MTEB (Massive Text Embedding Benchmark) is the standard leaderboard for embedding models. everyone looks at it, picks the model at the top, and calls it done. this is a mistake and understanding why is a genuine differentiator.
- MTEB is an average across 56 tasks spanning `classification`, `clustering`, `semantic similarity`, `retrieval`, and more. your RAG pipeline is exclusively a `retrieval` task on your specific domain. two models within one point on MTEB regularly sit eight to twelve points apart on domain-specific `retrieval` benchmarks. a model that wins MTEB on the strength of its `classification` performance can be mediocre for your specific legal or medical `retrieval` use case.
- The correct move: filter MTEB to `retrieval` tasks only, look at the BEIR sub-benchmark which is retrieval-specific across multiple domains, find the domain closest to yours, and use that ranking as your starting point. then build a small eval set from your actual data — 100-200 query/chunk relevance pairs and benchmark candidate models on it. that eval set is worth more than any leaderboard number.
- As of mid-2026, for English RAG the top performers are `Qwen3-Embedding-8B` (open source, outperforms all API models on MTEB at 70.6), Voyage `voyage-3-large` (strong on code/finance/legal domain variants), OpenAI `text-embedding-3-large` (best default for English with MRL support), and `BGE-M3` (best for hybrid retrieval with one model). for multilingual, `Cohere-Embed-v4` has the broadest language coverage with the smallest per-language performance floor.


