import re
import pandas as pd

def clean_redacted_text(text):
    """
    Cleans de-identification tokens (XXXX) using semantic regex replacement.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Map age indicators
    text = re.sub(r'\bXXXX\s*-\s*year\s*-\s*old\b', '[Age]-year-old', text, flags=re.IGNORECASE)
    text = re.sub(r'\bXXXX\s+year\s+old\b', '[Age]-year-old', text, flags=re.IGNORECASE)
    text = re.sub(r'\bXXXX\s*-year-old\b', '[Age]-year-old', text, flags=re.IGNORECASE)
    
    # 2. Map date indicators (e.g., "dated XXXX", "on XXXX")
    text = re.sub(r'\b(dated|date of|on|from)\s+XXXX\b', r'\1 [Date]', text, flags=re.IGNORECASE)
    
    # 3. Map time indicators (e.g., "at XXXX hours")
    text = re.sub(r'\b(at|around)\s+XXXX\s+(hours|hrs)\b', r'\1 [Time]', text, flags=re.IGNORECASE)
    
    # 4. Remove generic / isolated XXXX tokens that don't fit context
    text = re.sub(r'\bXXXX\b', '', text)
    
    # 5. Clean up grammar and punctuation anomalies left by deletions
    text = re.sub(r'\s+', ' ', text)                 # Collapses double/triple spaces
    text = re.sub(r'\s+([.,;?])', r'\1', text)       # Removes space before punctuation
    text = re.sub(r',+', ',', text)                  # Removes double commas
    text = re.sub(r'\s*,+\s*\.', '.', text)          # Cleans up sequences like " ,." to "."
    
    return text.strip()

def preprocess_reports_df(reports_path):
    """
    Loads, cleans, and standardises the radiology reports DataFrame.
    """
    df = pd.read_csv(reports_path)
    
    # Lowercase column names for consistency
    df.columns = [col.lower() for col in df.columns]
    
    # Clean text columns
    for col in ['findings', 'impression', 'indication']:
        if col in df.columns:
            df[col] = df[col].apply(clean_redacted_text)
            
    # Standardise column spacing/nulls
    df = df.fillna("")
    return df
