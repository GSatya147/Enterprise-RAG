#### failure experience:

1. **Loading document corpus:** Tried langchain components explicitly for loading document corpus of pdfs which has 2 column styled, 1 column styles arxiv pdf, `PDFPlumber` loader with custom configurations performed awful. tried others `PymuPDF` performed significantly better with minimal/default configurations, this shows pymupdf has more layout level awareness even tho it doesn't respect the spaces and layout decors, it gets the job done with the text.

2. **Data cleaning:** Sometimes we need to overengineer, tbh whatever we do in dataset engineering is never over engineering at all. Ignoring references/bibliographies in the data, automatically pollutes the individual chunks, even tho reranker ranks them low and prevents them never contaminating LLM context, still indexing-wasted-resources are not worth for individual chunks which are not useful. especially at scale. SPEND SOME TIME IN DATASET ENGINEERING.

3. **Embedding:**