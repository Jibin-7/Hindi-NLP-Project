import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from collections import defaultdict, Counter
import torch

# ==========================================
# 1. LOAD ABSTRACTIVE AI MODEL
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    return tokenizer, model

# ==========================================
# 2. CLASSIC NLP ASSETS 
# ==========================================
HINDI_STOPWORDS = set([
    "के", "का", "एक", "में", "की", "है", "यह", "और", "से", "हैं", "को", "पर", "इस", "होता", "कि", "जो", 
    "कर", "मे", "गया", "करने", "किया", "लिये", "अपने", "ने", "बनी", "नहीं", "तो", "ही", "या", "एवं", 
    "दिया", "हो", "इसका", "था", "द्वारा", "हुआ", "तक", "साथ", "करना", "वाले", "बाद", "लिए", "आप", "कुछ", 
    "सकते", "किसी", "ये", "इसके", "सबसे", "इसमें", "थे", "दो", "होने", "वह", "वे", "करते", "बहुत", "कहा", "गई"
])

def clean_text(text):
    return re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s।,\|]', '', text)

def hindi_stemmer(word):
    suffixes = ['ा', 'ी', 'े', 'ों', 'ें', 'कर', 'ना', 'ता', 'ती', 'ते']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > 3:
            return word[:-len(suffix)]
    return word

def enforce_hindi_fullstop(text):
    return text.replace('.', '।').replace('|', '।')

# ==========================================
# 3. ACCURACY & EVALUATION LOGIC
# ==========================================
def calculate_rouge_1(system_summary, reference_text):
    sys_tokens = [w for w in clean_text(system_summary).split() if w not in HINDI_STOPWORDS]
    ref_tokens = [w for w in clean_text(reference_text).split() if w not in HINDI_STOPWORDS]
    
    sys_counter = Counter(sys_tokens)
    ref_counter = Counter(ref_tokens)
    
    overlap = sum((sys_counter & ref_counter).values())
    
    precision = overlap / len(sys_tokens) if len(sys_tokens) > 0 else 0
    recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return round(precision * 100, 1), round(recall * 100, 1), round(f1 * 100, 1)

def calculate_compression_ratio(original, summary):
    orig_len = len(original.split())
    sum_len = len(summary.split())
    if orig_len == 0: return 0
    return round((1 - (sum_len / orig_len)) * 100, 1)

# ==========================================
# 4. CORE PIPELINE & MAPPING
# ==========================================
def extract_keywords_and_pipeline(text):
    """Retained for the Teacher's Evaluation Tab"""
    text = enforce_hindi_fullstop(text)
    sentences = [s.strip() for s in text.split('।') if len(s.strip()) > 5]
    
    word_frequencies = defaultdict(int)
    stem_to_original = {} 
    pipeline_steps = {}
    
    for i, sentence in enumerate(sentences):
        raw_tokens = clean_text(sentence).split()
        no_stopwords = [w for w in raw_tokens if w not in HINDI_STOPWORDS and len(w) > 2]
        
        stemmed_words = []
        for word in no_stopwords:
            stemmed = hindi_stemmer(word)
            stemmed_words.append(stemmed)
            if stemmed not in stem_to_original:
                stem_to_original[stemmed] = word
        
        if i == 0:
            pipeline_steps = {
                "sentence": sentence,
                "tokens": raw_tokens,
                "no_stopwords": no_stopwords,
                "stemmed": stemmed_words
            }
            
        for word in stemmed_words:
            word_frequencies[word] += 1
            
    sorted_stems = sorted(word_frequencies, key=word_frequencies.get, reverse=True)
    top_original_keywords = [stem_to_original[stem] for stem in sorted_stems if len(stem) > 2]
    
    return top_original_keywords[:5], pipeline_steps


def highlight_source_statements(original_text, abstract_summary, top_percent=0.4):
    original_text = enforce_hindi_fullstop(original_text)
    sentences = [s.strip() for s in original_text.split('।') if len(s.strip()) > 5]
    
    if not sentences:
        return original_text
        
    summary_words = set(clean_text(abstract_summary).split()) - HINDI_STOPWORDS
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        sent_words = set(clean_text(sentence).split()) - HINDI_STOPWORDS
        overlap = len(summary_words.intersection(sent_words))
        sentence_scores[i] = overlap
        
    num_to_highlight = max(1, int(len(sentences) * top_percent))
    top_indices = sorted(sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_to_highlight])
    
    highlighted_text = original_text
    for i, sentence in enumerate(sentences):
        if i in top_indices:
            highlight = f"<mark style='background-color: #ffe066; color: #000000; padding: 3px 6px; border-radius: 4px; font-weight: 500;'>{sentence}</mark>"
            highlighted_text = highlighted_text.replace(sentence, highlight)
            
    return highlighted_text.replace('</mark>।', '</mark> ।')

# ==========================================
# 5. STREAMLIT UI DESIGN
# ==========================================
st.set_page_config(page_title="Abstractive Hindi Summarizer", layout="wide")

with st.sidebar:
    st.header("⚙️ System Architecture")
    st.info("**Methodology:** Dual-Pass Abstractive Generation (mT5)")
    st.info("**Mapping:** Lexical Overlap highlights the source statements.")
    st.markdown("---")

st.title("🧠 Neural Abstractive Hindi Summarizer")
st.markdown("Generates AI-written summaries, distinct dynamic headlines, and evaluates accuracy.")

with st.spinner("Initializing Abstractive Model... (Please wait)"):
    tokenizer, model = load_model()

news_input = st.text_area("Paste Hindi News Article Here:", height=180)

if st.button("Generate AI Summary & Headline", type="primary"):
    if len(news_input.strip()) < 30:
        st.warning("Please enter a longer text for abstractive summarization.")
    else:
        with st.spinner("AI is analyzing text to write the summary and headline..."):
            
            keywords, pipeline_steps = extract_keywords_and_pipeline(news_input)
            inputs = tokenizer(news_input, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                # PASS 1: Generate the full Abstractive Summary
                summary_ids = model.generate(
                    inputs["input_ids"], 
                    max_length=120, 
                    min_length=25, 
                    length_penalty=1.5, 
                    num_beams=4,
                    early_stopping=True
                )
                
                # PASS 2: Generate the Distinct AI Headline (Forcing a shorter output)
                title_ids = model.generate(
                    inputs["input_ids"], 
                    max_length=18, 
                    min_length=5, 
                    length_penalty=0.5, # Low penalty forces brevity
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode the summary
            abstract_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            abstract_summary = enforce_hindi_fullstop(abstract_summary)
            
            # Decode and format the title
            raw_ai_title = tokenizer.decode(title_ids[0], skip_special_tokens=True)
            raw_ai_title = raw_ai_title.replace('।', '').strip()
            final_title = f"{raw_ai_title}... : जानिए पूरी खबर"
            
            # Accuracy & Mapping
            highlighted_original = highlight_source_statements(news_input, abstract_summary)
            precision, recall, f1_score = calculate_rouge_1(abstract_summary, news_input)
            compression = calculate_compression_ratio(news_input, abstract_summary)
            
            st.markdown("---")
            
            tab1, tab2, tab3 = st.tabs(["📊 Final Output", "⚙️ NLP Pipeline (Evaluation)", "📈 Accuracy & Metrics"])
            
            with tab1:
                st.markdown(f"<h3 style='color: #ef4444;'>{final_title}</h3>", unsafe_allow_html=True)
                st.write("")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📝 Abstractive Summary (AI)")
                    st.success(abstract_summary)
                    
                with col2:
                    st.subheader("🔍 Source Statements")
                    st.markdown(f"<div style='line-height: 1.8; padding: 10px;'>{highlighted_original}</div>", unsafe_allow_html=True)

            with tab2:
                st.markdown("### Preprocessing Pipeline (System Evaluation)")
                if pipeline_steps:
                    st.code(f"1. Original Sentence: {pipeline_steps['sentence']}", language="text")
                    st.code(f"2. Tokenization: {pipeline_steps['tokens']}", language="python")
                    st.code(f"3. Stopwords Removed: {pipeline_steps['no_stopwords']}", language="python")
                    st.code(f"4. Stemming Applied: {pipeline_steps['stemmed']}", language="python")
                    st.info(f"**Top Extracted Core Entities:** {', '.join(keywords)}")
                    
            with tab3:
                st.markdown("### System Accuracy & ROUGE Evaluation")
                st.markdown("The system calculates the **ROUGE-1 F1-Score** by measuring the unigram lexical overlap between the AI's abstractive generation and the original source text.")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("ROUGE-1 F1 Score", f"{f1_score}%")
                col_m2.metric("Precision", f"{precision}%")
                col_m3.metric("Recall", f"{recall}%")
                col_m4.metric("Data Compression", f"{compression}%")
