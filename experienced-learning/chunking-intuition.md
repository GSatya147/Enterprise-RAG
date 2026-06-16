## Chunking
### 1. The core problem and misleading intuition  
Chunking is not a text processing decision. it's an information retrieval contract you're making at index time that every downstream component has to live with. you chunk once, you embed those chunks, you index those chunks and from that point on, your retrieval system can only surface what you gave it. if you cut the wrong way, no amount of clever retrieval, no reranker, no LLM prompt engineering will recover the context you threw away. the mistake is baked in.  
So the real question is: *What is the smallest unit of information that is independently meaningful for a query?*  

**Fundamental tension**  
Every chunking strategy is actively trying to solve this tension  
- `context` vs `precision`: small chunks are precise, less noise. But they don't have surrounding context.  large chunks carry context, but they pollute LLM attention.  

**Naive approach**  
Naive approach is not as naive as you might think it is. It's ranked first among 7 different strategies for general corpora in Feb 2026 benchmarks.  
1. fixed-size splitting, with maybe 10-20% of overlapping. This is what langchain's `RecursiveTextSplitter` does under the hood.  
2. Good for: FAQ pages, product documentation, blog posts, short-form content with consistent paragraph lengths - fixed-size works because the content is already roughly uniform in density  
2. Fails when: document has variable information density, a legal contract has dense clauses followed by boilerplate. a research paper has an abstract, methodology, results, appendix - cutting at 512 fixed chunks is catastrophical.  

**Why overlap doesn't solve context problem**  
Everyone thinks overlap is the fix. add 10-20% overlap and your context loss goes away. it doesn't. overlap helps with boundary context, preventing a sentence from being cut in half. it does nothing for semantic context the fact that "its revenue" refers to Apple, which was mentioned 400 tokens ago. overlap is a band-aid, not a solution. knowing this is already a differentiator because most candidates think overlap is the answer when it's just noise reduction at the boundaries.  

#### The question you should be asking about any document before choosing a strategy  
Three questions, in this order:  
- **Query based**  
1. What's the query type?  
2. Is the user asking lookup question (precision - smaller chunk strategy) like *what is the refund policy?* or reasoning question (context- larger chunk strategy) like *compare q1 and q2 across the product lines*.    
3. *Strategy:* For loop-up: precision type - smaller chunks startegy. For reasoning: context type - larger chunks strategy  

- **Document structure based**      
1. What's the document structure?  
2. Does it have natural semantic boundaries (sections, headings, paragraphs, clauses)? or  
3. Is it dense prose where boundaries are implicit?  
4. *Strategy:* Structured docs → structure-aware splitting. Dense prose → semantic splitting.    

- **Document information based**   
1. what's the document's information density?   
2. *strategy:* Uniform (FAQ, blog) → fixed-size is fine. Variable (legal, academic, financial) → you need a strategy that respects semantic units.  

### 2. The strategies, their intuition, and where they earn their cost  
#### Different chunking strategies  
Let's go as a progression of strategies, improving from the previous one.  

**1. Fixed-sizing/recursive character splitting**    
- "Cut every N tokens, add some overlap."  
- Recursive is slightly smarter, it tries to split on paragraph breaks first, then sentences, then words, then characters, falling back down the hierarchy until it hits your size limit. it respects natural language breaks where it can.  
*When does it earn it's cost?*  
- When your corpus is homogeneous and your queries are lookup-style. if you're building a chatbot over a product FAQ or a single-domain knowledge base with consistent document structure, this is the right call. fast, deterministic, no external dependencies, easy to debug. don't overthink it here.  
*When does it break?*  
- The moment your corpus has variable information density. a financial report has three paragraphs of boilerplate legal disclaimer followed by one dense paragraph that contains everything meaningful. fixed-size will split that dense paragraph in half and bury the boilerplate in multiple chunks nobody should ever retrieve.  

**2. Semantic chunking**  
- Instead of cutting by token count, you cut by meaning shift. the idea: embed every sentence, then slide a window across and measure cosine similarity between adjacent sentences. when similarity drops sharply, that's a topic boundary cut there.
- You're letting the content tell you where it wants to be split rather than imposing an arbitrary size. 
*Honest caveats*  
- It is roughly 14x slower than former one, cus of indexing and embedding the large corpus at chunking level itself.
- The similarity drop can never happen, considering academic research's every sentence is related - undersplitting
- The benchmark from Feb 2026 resulted 43 tokens chunks - the opposite, over-splitting  
*Closing notes*  
- Semantic chunking works gracefully on long-form narratives where context/topic switching is real and gradual like novels, transcripts, books, essays.
- It struggels on dense technical corpus
- And is over-kill for homogeneous content

**3. Hierarical/parent-child chunking**  
- Most elegant solution for context vs precision tension and mostly used.  
- The idea = you maintain both simulataneously
- Small child chunks: maybe 128-256 tokens goes into your vector index. these are what get retrieved because they're precise and tightly scoped. but when you retrieve a child chunk, you don't send it to the LLM. instead you look up its parent, a larger 512-1024 token chunk that contains the child, send that to the LLM for generation.
- Insight: the retrieval and generation has two different optimal chunk sizes, retrieval needs precision(child chunk) and generation needs context (parent chunk).
*Cheap alternative*  
- The `sentence-window` variant is the simpler implementation of the same idea: you index individual sentences but when you retrieve one, you return the surrounding window (say 2 sentences before and after). no separate parent document store needed, just a metadata pointer.  
*Honest caveats*  
- Maintaining two index levels means updates in the corpus gonna be expensive.
- If document changes, you need to change indices both in sync
- And parent chunk size needs tuning per domain, cus it's the same problem again too large means noise, too small no learning at all.

**4. Late chunking**  
- In standard chunking you split first, then embed. the problem is each chunk is embedded in isolation. so a chunk containing "its revenue grew 3%" has no idea that "its" refers to Apple, because Apple was in a different chunk that got processed separately.
- Late chunking flips the order. you take the entire document, pass it through a long-context embedding model (Jina's long-context models are the reference implementation here), get token-level embeddings for every token in the document all at once, with full document context. then you pool those token embeddings within your chunk boundaries to produce chunk embeddings.
- The key: when you pool, each token embedding already carries global document context because the attention mechanism saw the whole document. 
- The practical implication is significant for documents with lots of anaphoric references pronouns, "the company," "the aforementioned clause" things that only make sense in context. late chunking improved retrieval on these by 10-12% in research settings.
*Honest caveats*  
- You need long context embeddings models
- The arxiv paper from April 2025 noted that late chunking trades semantic coherence for efficiency, `contextual retrieval` preserves coherence better, late chunking is faster but can sacrifice relevance on complex queries.

**5. Contextual Retrieval (Anthropic's approach)**  
- Chunk first normally, then for every chunk, call an LLM with the full document and the chunk and ask it to generate a 50-100 token context summary describing what this chunk is about in the context of the whole document. prepend that summary to the chunk before embedding.
- So instead of embedding "revenue grew 3% last quarter", you embed "This chunk is from Apple's Q3 2024 earnings report. It describes quarterly revenue performance relative to the previous quarter. Revenue grew 3% last quarter."
- The embedding is now massively more informative. Anthropic's own numbers: 49% reduction in retrieval failures with hybrid search, 67% when combined with reranking. that's not a marginal gain.
*Honest Caveats*  
- For a corpus of 100,000 chunks that's 100,000 LLM calls. that's expensive. 
- The mitigation is prompt caching, since the document prefix is the same across all calls for a given document, you cache it and only pay for the chunk-specific part. Anthropic claims this brings the cost down by up to 90% with their caching implementation.  

**Notes:** Late chunking is more cheap but needs long context embedding model, degrades for very long documents. Contextual retrieval is expensive but works on any embedding model and document.  

**6. Structure aware/element type routing**  
- Tables are atomic. embed a table as one unit, optionally with its caption and column headers prepended as context.
- Headings should be propagated into their child chunks as metadata or as a prefix.
- code blocks/snippets are atomic.  
- If your parser gave you typed elements (which it should, from loading), your chunker should consume those types and apply different rules per type. this is the architecture most people don't build because they treat chunking as a generic text operation rather than a document-structure-aware operation.  

### 3. Failure modes and the Decision Framework  
#### Failure modes that matter  

**1. Semantic Orphaning**  
- Passed retrieval (precision), failed generation (context).
- The chunk missing it's semantic context which got retrieved correctly through similarity but generation failed due to lack of context.
- This is not a retrieval problem, it's a chunking problem masquerading as a generation problem. 
- Fix the chunking do not blame LLM or prompts.  

**2. Split entity problem**  
- An entity: a person, a company, a legal clause, a definition gets split across two chunks. both halves retrieve with moderate similarity to related queries but neither half contains the complete entity. the LLM sees half a definition and either hallucinates the rest or says it doesn't have enough information. this is especially brutal in legal and medical domains where precision of a full clause is everything.

**3. Chunk size mismatch with query type**
- Your queries are multi-hop reasoning questions ("compare the refund policy across product lines A, B, and C") but your chunks are 128-token precision chunks. each individual chunk correctly contains one piece of the answer but the LLM receives three tiny isolated fragments with no connective tissue
- *Flip side*: your queries are precise lookups ("what is the late payment fee?") but your chunks are 1024-token parent chunks. you retrieve correctly but you hand the LLM a wall of text where the answer is one sentence buried in the middle.

**4. Overlap-induced duplication**
- you added 20% overlap to be safe. your top-k retrieval returns chunk 7 and chunk 8, which overlap by 100 tokens. the LLM now sees the same 100 tokens twice in its context. for factual questions this is just waste. for questions that require aggregation or counting, double-counted information can cause wrong answers. "how many times did X occur?" the LLM counts your overlap as two occurrences.
- *Fix*: deduplication at retrieval time before context assembly, not reducing overlap at chunk time. but most pipelines don't have that step.

**Stale chunk problem**
- Your document updates. a policy changes, a price changes, a date changes. you re-index the new document. but your vector store has chunks from both the old and new version because you inserted without deleting, or your deletion logic missed some chunks. now retrieval returns a mix of old and new chunks. the LLM sees contradictory information and either hedges or picks the wrong one.
- This is an operational failure that starts at chunking architecture. if you assign chunk IDs that are deterministic and document-scoped (hash of document ID + chunk position), then updates become delete-by-document-ID then re-insert, which is clean. if your chunk IDs are random UUIDs, you've lost the ability to do clean updates.

**The embedding model - chunk size mismatch**
- Different embedding models have different optimal input lengths. a model trained predominantly on short texts (like early sentence-transformers) will produce poor embeddings for 1024-token chunks, the representation degrades as length increases because the model wasn't trained to compress that much information into one vector. 
- Conversely, a long-context embedding model like text-embedding-3-large or Jina's long-context models can handle larger chunks and may actually underperform on very short chunks because they're not meaningfully utilizing their capacity.
- Pick embedding model and chunk size as a shared decision.

####