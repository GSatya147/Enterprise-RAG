### Loading
**Loading** may sound boring - but it's a slow poison which contaminates the entire pipeline at initial stage, most pipelines use `PyPDFLoader` and moves on.

**Best Practice :**
- `PyPDF2`, `PDFPlumber` (Awful personal experience with preserving two column layout), `PymuPDF` (best for two columned arXiv style pdf - tested personally): Works for simple digital PDFs not scanned ones.
- Always eyeball the loaded pages with actual content, moving forward without spot-checking is unreasonable.

**Advanced Practice :**
