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

#### **The question you should be asking about any document before choosing a strategy**
Three questions, in this order:
- **Query based**  
1. What's the query type? 
2. Is the user asking lookup question (precision - smaller chunk strategy) like *what is the refund policy?* or reasoning question (context- larger chunk strategy) like *compare q1 and q2 across the product lines*.  
3. *Strategy:* for loop-up: precision type - smaller chunks startegy. for reasoning: context type - larger chunks strategy

- **Document structure based**  
1. What's the document structure? 
2. Does it have natural semantic boundaries (sections, headings, paragraphs, clauses)? or 
3. Is it dense prose where boundaries are implicit? 
4. *Strategy:* structured docs → structure-aware splitting. dense prose → semantic splitting.  

- **Document information based**
1. what's the document's information density? 
2. Uniform (FAQ, blog) → fixed-size is fine. 
3. Variable (legal, academic, financial) → you need a strategy that respects semantic units.

