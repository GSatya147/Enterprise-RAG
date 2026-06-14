### Loading
**Loading** may sound boring - but it's a slow poison which contaminates the entire pipeline at initial stage, most pipelines use `PyPDFLoader` and moves on.

**Best Practice :**  
- `PyPDF2`, `PDFPlumber` (Awful personal experience with preserving two column layout), `PymuPDF` (best for two columned arXiv style pdf - tested personally): Works for simple digital PDFs not scanned ones.  
- Always eyeball the loaded pages with actual content, moving forward without spot-checking is unreasonable.  

**Advanced Practice**  
- For any PDF other than clean text-only ones, we need a layout-aware parser, i.e a parser with spatial awareness not just etxt exctraction ability.  
- Tools like `Unstructured` (with hi_res strategy), `Docling`, `Marker-PDF`, `LlamaParse` operate at this level. They give us backetyped elements like `Title`, `NarrativeText`, `Table`, `ListItem`: with metadata like page number and element position, perfect.  

**Personal Suggestions**  
Don't just plug one loader, make it decisional on the data type - Make a Router + Parser, not a Loader
1. Clean born-digital PDF → `pdfplumber` or `PyMuPDF`, fast and cheap  
2. Complex PDF with tables/charts/multi-column → `Docling` or `Marker-PDF` (open source, self-hosted, reliable)  
3. Scanned or image-heavy documents → you need OCR in the loop, either `Unstructured hi_res` with tesseract, or AWS Textract / Google Document AI if cost isn't a concern  
4. Web content → `Firecrawl` or `Trafilatura`, not `BeautifulSoup`, because those are designed for clean content extraction not just tag stripping  
5. Office docs (Word, PowerPoint) → Unstructured's `partition()` handles the breadth well, it supports 25+ formats with a single API  

**Differentiator Practices**  
- Most pipelines doesn't preserve structure as metadata, they extract text and throw away the document's hierarchy. the differentiator is that when you extract a Table element, we need to store it differently. we convert it to markdown table format, we tag it as `element_type: table`, we keep its page number, section heading, and position. why? because later when we chunk, we treat tables as atomic units, we never split a table across chunks. and when we retrieve, we can filter or boost by element type. "give me tables only" is a valid retrieval strategy for financial documents.  
- Pipelines should also store the document-level metadata at load time. author, creation date, document title, section headings as a list. this becomes filterable metadata in our vector store later. most pipelines never do this and then wonder why the retrieval has no selectivity.  

**Personal Creative Practice**  
I'd build a loading layer that does three things upfront:  
- Run a quick document classifier before parsing. a simple heuristic: count image objects vs text blocks in the PDF.   
1. if images dominate → route to VLM-assisted parsing or `hi_res` OCR.   
2. if text dominates → route to fast parser.   
3. don't send everything through expensive OCR when 80% of your documents are clean text   
- Parse once, store the raw structured output (the typed elements with metadata) in a intermediate format like JSONL. this is your "parse cache". now if you want to experiment with different chunking strategies later, you don't re-parse. re-parsing is expensive, especially with cloud parsers. parse once, chunk many times.  
- Validate at load time. before a document enters your pipeline I'd run a quick sanity check: did we extract at least N characters? did table count match expected? are there encoding issues (`\x00` null bytes, garbled `unicode`)? if a document fails validation, it goes to a quarantine queue, not silently into your index. silent bad documents are the worst failure mode because you only discover them when a user asks a question that should be answerable and isn't.  

