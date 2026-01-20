# VocalizeWeb SRS Document

## Project Information
- **Project Title:** Voice-to-Voice AI Assistant
- **Product Name:** VocalizeWeb
- **Team Members:**
  - Sarmad Sultan (NUM-BSCS-2022-15)
  - Shehzana Bibi (NUM-BSCS-2022-44)
- **Supervisor:** Mam Sonia Hassan
- **Year:** 2026

## Document Structure

```
latex_srs/
├── main.tex                  # Main LaTeX document
├── sections/                 # Individual chapter files
│   ├── 00_cover.tex         # Cover page
│   ├── 01_declaration.tex   # Declaration
│   ├── 02_abstract.tex      # Abstract
│   ├── 03_introduction.tex  # Chapter 1: Introduction
│   ├── 04_system_description.tex  # Chapter 2: System Description
│   ├── 05_analysis_design.tex     # Chapter 3: Analysis and Design
│   ├── 06_remaining_work.tex      # Chapter 4: Remaining Work
│   ├── 07_references.tex    # References
│   └── 08_appendices.tex    # Appendices
└── diagrams/                # PlantUML diagram source files
    ├── dfd_level0.puml      # Context Diagram
    ├── dfd_level1.puml      # Level 1 DFD
    ├── dfd_level2.puml      # Level 2 DFD (RAG Pipeline)
    ├── usecase.puml         # Use Case Diagram
    ├── erd.puml             # Entity-Relationship Diagram
    ├── sequence_query.puml  # Voice Query Sequence
    └── sequence_upload.puml # Knowledge Upload Sequence
```

## Compiling the LaTeX Document

### Option 1: Using Overleaf (Recommended)
1. Go to [Overleaf](https://www.overleaf.com)
2. Create a new blank project
3. Upload all `.tex` files maintaining the directory structure
4. Set the main document to `main.tex`
5. Click "Recompile" to generate the PDF

### Option 2: Windows with MiKTeX
1. Install MiKTeX from https://miktex.org/download
2. Install TeXstudio or TeX Editor of choice
3. Open `main.tex` in the editor
4. Click "Build & View" or press F5
5. PDF will be generated in the same directory

### Option 3: Command Line (LaTeX installed)
```bash
cd D:\stt-tts\SRS\srs_reports\latex_srs
pdflatex main.tex
pdflatex main.tex  # Run twice for TOC and references
```

## Generating Diagrams from PlantUML

### Online (No Installation Required)
1. Go to http://www.plantuml.com/plantuml/uml/
2. Copy the content from any `.puml` file
3. Paste into the text area
4. Click "Submit" to generate the diagram
5. Download as PNG or SVG

### Using PlantUML Jar (Recommended for batch processing)
```bash
# Install Java if not already installed
# Download plantuml.jar from https://plantuml.com/download

# Generate all diagrams
cd diagrams
java -jar plantuml.jar *.puml

# This will create .png files for all diagrams
```

### VS Code Extension
1. Install "PlantUML" extension in VS Code
2. Open any `.puml` file
3. Press `Alt+D` to preview
4. Right-click and select "Export Current Diagram" to save as image

## Document Formatting Specifications

- **Font:** Times New Roman, 12pt
- **Line Spacing:** 1.5 with space after paragraphs
- **Alignment:** Justified
- **Margins:** Left 3cm, Right/Top/Bottom 2.5cm
- **Headings:**
  - Chapter: 16pt Bold
  - Section (Level 1): 14pt Bold  - Subsection (Level 2): 12pt Bold
  - Subsubsection (Level 3): 12pt Bold
- **References:** IEEE format
- **Citations:** Numbered [1], [2], etc.

## Embedding Diagrams in LaTeX

Once you generate PNG images from PlantUML files, you can embed them:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{diagrams/dfd_level0.png}
    \caption{\textit{Level 0 Data Flow Diagram (Context Diagram)}}
    \label{fig:dfd0}
\end{figure}
```

Place diagram PNG files in the `diagrams/` directory and uncomment the figure blocks in the respective sections.

## Making Changes

### To Update Content
1. Edit the relevant `.tex` file in the `sections/` directory
2. Recompile `main.tex` to see changes
3. Keep language natural and professional

### To Add Diagrams
1. Edit or create `.puml` files in `diagrams/` directory
2. Generate PNG images using PlantUML
3. Add figure blocks in the appropriate chapter file
4. Reference using `Figure~\ref{fig:label}`

### To Add References
1. Edit `sections/07_references.tex`
2. Add new `\bibitem{key}` entry in IEEE format
3. Cite in text using `\cite{key}` or manually as [#]

## Quick Fixes

### Missing Package Errors
If you get missing package errors during compilation:
- **Overleaf:** Packages install automatically
- **MiKTeX:** Click "Install" when prompted
- **Manual:** Run `tlmgr install <package-name>`

### Table of Contents Not Updated
Run `pdflatex main.tex` twice - first pass generates TOC data, second pass incorporates it.

### Images Not Showing
1. Ensure PNG files are in the correct `diagrams/` folder
2. Check file paths in `\includegraphics` commands 
3. Verify the `graphicx` package is loaded

## Converting to Word (if needed)

```bash
# Using pandoc (install from https://pandoc.org)
pandoc main.tex -o VocalizeWeb_SRS.docx --bibliography=references.bib
```

Note: LaTeX to Word conversion may require formatting adjustments.

## Final Checks Before Submission

- [ ] All sections complete with no placeholder text
- [ ] All diagrams generated and embedded
- [ ] References properly formatted in IEEE style
- [ ] Table of Contents, List of Figures, List of Tables generated
- [ ] Page numbers correct
- [ ] No LaTeX compilation errors
- [ ] Team names and details correct on cover page
- [ ] Supervisor name spelled correctly
- [ ] Year is correct (2026)
- [ ] Abstract is one page or less
- [ ] All technical terms defined in Chapter 1.3
- [ ] Document prints correctly with proper margins

## Contact

For technical issues with the SRS document:
- Sarmad Sultan: NUM-BSCS-2022-15
- Shehzana Bibi: NUM-BSCS-2022-44

---

**Document Created:** January 12, 2026  
**Last Updated:** January 12, 2026  
**Template Compliance:** Industrial FYPs SRS Report Template (IEEE Format)
