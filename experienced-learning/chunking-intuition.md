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
3. *Strategy:* 
- For loop-up: precision type - smaller chunks startegy. 
- For reasoning: context type - larger chunks strategy

- **Document structure based**  
1. What's the document structure? 
2. Does it have natural semantic boundaries (sections, headings, paragraphs, clauses)? or 
3. Is it dense prose where boundaries are implicit? 
4. *Strategy:* 
- Structured docs → structure-aware splitting. 
- Dense prose → semantic splitting.  

- **Document information based**
1. what's the document's information density? 
2. *strategy:*
- Uniform (FAQ, blog) → fixed-size is fine. 
- Variable (legal, academic, financial) → you need a strategy that respects semantic units.

### 2. The strategies, their intuition, and where they earn their cost
#### Different chunking strategies
Let's go as a progression of strategies

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

