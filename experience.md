#### failure experience:

1. **Loading document corpus:** Tried langchain components explicitly for loading document corpus of pdfs which has 2 column styled, 1 column styles arxiv pdf, `PDFPlumber` loader with custom configurations performed awful. tried others `PymuPDF` performed significantly better with minimal/default configurations, this shows pymupdf has more layout level awareness even tho it doesn't respect the spaces and layout decors, it gets the job done with the text.

2. **Data cleaning:** Sometimes we need to overengineer, tbh whatever we do in dataset engineering is never over engineering at all. Ignoring references/bibliographies in the data, automatically pollutes the individual chunks, even tho reranker ranks them low and prevents them never contaminating LLM context, still indexing-wasted-resources are not worth for individual chunks which are not useful. especially at scale. SPEND SOME TIME IN DATASET ENGINEERING.

3. **Embedding:** Local CPU embedding voyage model: 30 mins +, Managed voyageAi API: 10 seconds. hmmmm.

4. **Voyage API Rate Limits:** Free tier = 3 RPM, 10K TPM. Silently failed mid-embedding loop, exception swallowed, shape mismatch only caught by assert (7 embeddings vs 333 chunks). Fix: add payment method (free tokens still apply), rate limits unlock in seconds not "several minutes" as advertised. Always assert len(embeddings) == len(chunks) before upsert.

5. **Pinecone index naming:** Uppercase letters in index name silently skipped index creation. No error thrown. Vectors never landed. Fix: always lowercase. Silent failures are the worst failures, add a vector count check after upsert to validate.

6. **LangChain wrapper vs raw libraries:** LangChain's PDFPlumberLoader passes no configuration to the underlying pdfplumber, default extraction with zero column awareness. The abstraction hides the knobs you actually need. Use LangChain where it earns its keep (RecursiveCharacterTextSplitter), go raw everywhere else.

7. **Local model vs API architecture decision:** Removed SentenceTransformer and CrossEncoder local models entirely. Voyage API handles both embedding and reranking. Same models, same output, no CPU bottleneck, no local dependency. Production RAG has no local models in the critical path.