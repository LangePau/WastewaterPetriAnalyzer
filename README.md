# README

## Description
Automated image processing for top-down photos of Petri dishes from wastewater samples.  
Detects, segments, and measures bacterial colonies. Exports results as CSV/JSON and optionally as PDF.

## Prerequisites
- Python 3.10+
- Git

## Installation & Setup

1. Clone the repository  
   
```bash
   git clone https://github.com/LangePau/WastewaterPetriAnalyzer.git
```
  

2. Create and activate a virtual environment  
``` bash
   python3 -m venv .venv  
   source .venv/bin/activate               # Linux/macOS  
   source .venv\Scripts\activate           # Windows PowerShell
```

3. Install dependencies  
```
   pip install --upgrade pip  
   pip install -r requirements.txt
```
## Directory Structure

