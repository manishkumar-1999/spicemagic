"""
SpiceMagic Web Application v2
Enhanced with:
1. Live prediction interface (PubMedBERT model)
2. PubMed link integration (clickable references)
3. Dark mode + UI polish

Usage:
    python -m streamlit run spicerx_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64
import warnings
warnings.filterwarnings('ignore')

# Helper to embed local image as base64 (works in Streamlit)
def img_to_base64(path):
    """Convert local image to base64 string for HTML embedding."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    ext = path.split('.')[-1].lower()
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="SpiceMagic — Spice-Disease Database",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# Force all external links to open in a new tab
# ==========================================================
components.html("""
<script>
setInterval(() => {
    try {
        const doc = window.parent.document;
        doc.querySelectorAll('a').forEach(a => {
            const href = a.getAttribute('href') || '';
            // Only patch real external URLs (http/https), skip Streamlit's internal # links
            if (href.startsWith('http') && a.target !== '_blank') {
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
            }
        });
    } catch (e) {}
}, 400);
</script>
""", height=0)

# ==========================================================
# Global UI Helpers: Toast Notifications, Copy-to-Clipboard, Skeleton Loaders
# ==========================================================
st.markdown("""
<style>
/* --- Skeleton loader (shimmering placeholder) --- */
.skeleton {
    background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
    background-size: 200% 100%;
    animation: skeleton-shimmer 1.4s ease-in-out infinite;
    border-radius: 6px;
    display: block;
}
.skeleton-line { height: 14px; margin: 8px 0; }
.skeleton-title { height: 28px; margin: 12px 0; width: 60%; }
.skeleton-card {
    height: 100px;
    margin: 12px 0;
    border-radius: 10px;
}
@keyframes skeleton-shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
[data-theme-dark] .skeleton,
.stApp[style*="background-color: rgb(30, 30, 46)"] .skeleton {
    background: linear-gradient(90deg, #2a2a3e 25%, #3a3a4e 50%, #2a2a3e 75%);
    background-size: 200% 100%;
}

/* --- Toast notification container --- */
.toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 999999;
    display: flex;
    flex-direction: column;
    gap: 10px;
    pointer-events: none;
}
.toast {
    background: #FFFFFF;
    color: #1E293B;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    border-left: 4px solid #10B981;
    animation: toast-in 0.3s ease-out, toast-out 0.3s ease-in 3s forwards;
    font-size: 0.9rem;
    font-weight: 500;
    pointer-events: auto;
    min-width: 240px;
}
.toast-success { border-left-color: #10B981; }
.toast-error   { border-left-color: #EF4444; }
.toast-info    { border-left-color: #3B82F6; }
@keyframes toast-in {
    from { transform: translateX(120%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
@keyframes toast-out {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(120%); opacity: 0; }
}

/* --- Copy button --- */
.copy-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: #F1F5F9;
    color: #475569;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.8rem;
    font-family: monospace;
    transition: all 0.15s ease;
    text-decoration: none !important;
}
.copy-btn:hover {
    background: #E2E8F0;
    border-color: #94A3B8;
    color: #1E293B;
}
.copy-btn.copied {
    background: #D1FAE5;
    border-color: #10B981;
    color: #047857;
}
</style>
""", unsafe_allow_html=True)

# Install toast + copy helpers on the parent window (real script needs iframe)
components.html("""
<script>
(function() {
    try {
        const top = window.parent;
        if (top.__toastInit) return;
        top.__toastInit = true;
        const doc = top.document;
        let container = doc.getElementById('toast-container-root');
        if (!container) {
            container = doc.createElement('div');
            container.id = 'toast-container-root';
            container.className = 'toast-container';
            doc.body.appendChild(container);
        }
        top.showToast = function(msg, kind) {
            kind = kind || 'success';
            const t = doc.createElement('div');
            t.className = 'toast toast-' + kind;
            t.textContent = msg;
            container.appendChild(t);
            setTimeout(function() { t.remove(); }, 3500);
        };
        top.copyToClipboard = function(text, btnEl) {
            const done = function() {
                if (btnEl) {
                    const orig = btnEl.innerHTML;
                    btnEl.classList.add('copied');
                    btnEl.innerHTML = '✓ Copied';
                    setTimeout(function() {
                        btnEl.classList.remove('copied');
                        btnEl.innerHTML = orig;
                    }, 1500);
                }
                if (top.showToast) top.showToast('Copied to clipboard!', 'success');
            };
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(done).catch(function() {
                    if (top.showToast) top.showToast('Copy failed', 'error');
                });
            } else {
                const ta = doc.createElement('textarea');
                ta.value = text; doc.body.appendChild(ta);
                ta.select(); doc.execCommand('copy'); ta.remove();
                done();
            }
        };
    } catch (e) {}
})();
</script>
""", height=0)

def render_skeleton(kind='card', count=3):
    """Render CSS skeleton placeholder while data loads."""
    html_parts = []
    for _ in range(count):
        if kind == 'card':
            html_parts.append('<div class="skeleton skeleton-card"></div>')
        elif kind == 'title':
            html_parts.append('<div class="skeleton skeleton-title"></div>')
        else:
            html_parts.append('<div class="skeleton skeleton-line"></div>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)

def render_copy_button(text_to_copy, label="📋 Copy", key=None):
    """Render a small copy-to-clipboard button.

    Uses window.parent.copyToClipboard defined above. Escapes text for JS.
    """
    safe = str(text_to_copy).replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
    btn_id = f"cpybtn_{key or abs(hash(text_to_copy))}"
    st.markdown(
        f"""<a href="javascript:void(0)" id="{btn_id}"
              class="copy-btn"
              onclick="window.parent.copyToClipboard('{safe}', this); return false;">
              {label}
        </a>""",
        unsafe_allow_html=True,
    )

def show_toast(msg, kind='success'):
    """Trigger a toast notification (success/error/info)."""
    safe = str(msg).replace("'", "\\'")
    components.html(
        f"<script>if(window.parent.showToast){{window.parent.showToast('{safe}','{kind}');}}</script>",
        height=0,
    )

# ==========================================================
# Visitor Counter (uses CounterAPI — free, no signup, no backend)
# ==========================================================
@st.cache_data(ttl=60)  # Cache count for 60s to avoid hammering the API
def _fetch_visitor_count():
    """Read current visitor count without incrementing."""
    import requests
    try:
        resp = requests.get(
            "https://api.counterapi.dev/v1/spicemagic/visits",
            timeout=4,
        )
        if resp.ok:
            data = resp.json()
            return int(data.get('count', 0))
    except Exception:
        pass
    return None

def _increment_visitor_count():
    """Increment count once per session, return new count."""
    import requests
    try:
        resp = requests.get(
            "https://api.counterapi.dev/v1/spicemagic/visits/up",
            timeout=4,
        )
        if resp.ok:
            data = resp.json()
            return int(data.get('count', 0))
    except Exception:
        pass
    return None

def get_visitor_count():
    """Get visitor count — increment once per session, cache reads."""
    if 'visitor_counted' not in st.session_state:
        st.session_state.visitor_counted = True
        count = _increment_visitor_count()
        if count is not None:
            st.session_state.visitor_total = count
    if 'visitor_total' not in st.session_state:
        # Fallback: try a read-only fetch
        count = _fetch_visitor_count()
        st.session_state.visitor_total = count
    return st.session_state.get('visitor_total')

def render_visitor_badge():
    """Render an inline visitor-count badge with live/fallback states."""
    total = get_visitor_count()
    if total is None:
        # Network / API unreachable — hide silently rather than show broken UI
        return
    _is_dark = st.session_state.get('theme', 'Light') == 'Dark'
    _bg    = 'linear-gradient(135deg, #1E293B, #334155)' if _is_dark else 'linear-gradient(135deg, #EEF2F7, #F8FAFC)'
    _text  = '#F1F5F9' if _is_dark else '#1E293B'
    _label = '#94A3B8' if _is_dark else '#64748B'
    _accent = '#A78BFA' if _is_dark else '#7C3AED'
    st.markdown(f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 12px;
        background: {_bg};
        padding: 10px 18px;
        border-radius: 10px;
        border: 1px solid {'#334155' if _is_dark else '#E5E7EB'};
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin: 8px 0;
    ">
        <span style="font-size: 1.5rem;">👁️</span>
        <div>
            <div style="color: {_label}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">
                Total Visitors
            </div>
            <div style="color: {_text}; font-size: 1.4rem; font-weight: 800; line-height: 1.2;">
                <span style="color: {_accent};">{total:,}</span>
            </div>
        </div>
        <div style="width: 1px; height: 40px; background: {'#475569' if _is_dark else '#CBD5E1'}; margin: 0 4px;"></div>
        <div>
            <div style="color: {_label}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">
                Live
            </div>
            <div style="color: {_text}; font-size: 0.95rem; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; background: #10B981; border-radius: 50%; box-shadow: 0 0 8px rgba(16, 185, 129, 0.6); animation: pulse-live 2s infinite;"></span>
                You + others
            </div>
        </div>
    </div>
    <style>
    @keyframes pulse-live {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# Theme Selection (Dark Mode Toggle)
# ==========================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

# ==========================================================
# Dynamic CSS based on theme
# ==========================================================
def get_theme_css(theme='Light'):
    if theme == 'Dark':
        return """
        <style>
            /* Dark mode overrides */
            .stApp {
                background-color: #1E1E2E;
                color: #E0E0E0;
            }
            /* Force all headings visible in dark mode */
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4,
            [data-testid="stMarkdownContainer"] h5,
            [data-testid="stMarkdownContainer"] h6 {
                color: #F1F5F9 !important;
            }
            /* Markdown paragraphs, lists, labels */
            [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stMarkdownContainer"] span,
            [data-testid="stMarkdownContainer"] strong,
            .stMarkdown, .stMarkdown p, .stMarkdown li,
            .stApp label, .stApp p, .stApp li {
                color: #E0E0E0 !important;
            }
            /* Radio, checkbox, selectbox labels */
            .stRadio label, .stCheckbox label, .stSelectbox label,
            .stSlider label, .stTextInput label, .stTextArea label,
            .stNumberInput label, .stDateInput label {
                color: #E0E0E0 !important;
            }
            /* Radio option text */
            .stRadio div[role="radiogroup"] label {
                color: #E0E0E0 !important;
            }
            /* Tab labels */
            .stTabs [data-baseweb="tab"] {
                color: #CBD5E1 !important;
            }
            .stTabs [aria-selected="true"] {
                color: #A78BFA !important;
            }
            /* Metric widget text */
            [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
            [data-testid="stMetricDelta"] {
                color: #E0E0E0 !important;
            }
            /* Dataframe / table text */
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                color: #E0E0E0 !important;
            }
            /* Expander header */
            .streamlit-expanderHeader, [data-testid="stExpander"] summary {
                color: #E0E0E0 !important;
            }
            /* Info/warning/success box text */
            .stAlert, .stAlert p, .stAlert span {
                color: #F1F5F9 !important;
            }
            /* Sidebar — visibly separated from main area with border + darker bg */
            [data-testid="stSidebar"] {
                background-color: #14141F !important;
                border-right: 1px solid #3B3B5A !important;
                box-shadow: 4px 0 12px rgba(0, 0, 0, 0.35) !important;
            }
            /* Optional accent bar on the inside of the sidebar edge */
            [data-testid="stSidebar"]::after {
                content: '';
                position: absolute;
                top: 0; right: 0; bottom: 0;
                width: 3px;
                background: linear-gradient(180deg, #8B5CF6 0%, #6366F1 100%);
                opacity: 0.6;
            }
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] h5,
            [data-testid="stSidebar"] h6 {
                color: #F1F5F9 !important;
            }
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"],
            [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
                color: #E0E0E0 !important;
            }
            /* Radio options in sidebar */
            [data-testid="stSidebar"] .stRadio label,
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
                color: #E0E0E0 !important;
            }
            /* Metric widgets in sidebar */
            [data-testid="stSidebar"] [data-testid="stMetricLabel"],
            [data-testid="stSidebar"] [data-testid="stMetricValue"] {
                color: #E0E0E0 !important;
            }
            [data-testid="stSidebar"] [data-testid="stMetricValue"] {
                color: #A78BFA !important;
            }
            .main-header {
                background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
            }
            .main-header h1 {
                color: white; margin: 0; font-size: 2.5rem; font-weight: bold;
            }
            .main-header p {
                color: #E0E7FF; margin: 0.5rem 0 0 0; font-size: 1.1rem;
            }
            .metric-card {
                background: #2A2A3E;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.4);
                border-left: 4px solid #8B5CF6;
                margin-bottom: 1rem;
                color: #E0E0E0;
            }
            .metric-value {
                font-size: 2.5rem; font-weight: bold; color: #A78BFA; margin: 0;
            }
            .metric-label {
                color: #A0A0B0; font-size: 0.95rem; margin: 0;
            }
            .prediction-card {
                background: #2A2A3E;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.4);
                margin: 1rem 0;
            }
            .stTextInput input, .stTextArea textarea {
                background-color: #2A2A3E !important;
                color: #E0E0E0 !important;
                border: 1px solid #4B5563 !important;
            }
            /* Placeholder text visibility in dark mode (multiple selectors for cross-browser + Streamlit variants) */
            input::placeholder,
            textarea::placeholder,
            .stApp input::placeholder,
            .stApp textarea::placeholder,
            [data-testid="stTextInput"] input::placeholder,
            [data-testid="stTextArea"] textarea::placeholder,
            [data-baseweb="input"] input::placeholder,
            [data-baseweb="textarea"] textarea::placeholder {
                color: #CBD5E1 !important;
                opacity: 1 !important;
                -webkit-text-fill-color: #CBD5E1 !important;
            }
            input::-webkit-input-placeholder,
            textarea::-webkit-input-placeholder {
                color: #CBD5E1 !important;
                opacity: 1 !important;
            }
            input::-moz-placeholder,
            textarea::-moz-placeholder {
                color: #CBD5E1 !important;
                opacity: 1 !important;
            }
            .stSelectbox > div > div {
                background-color: #2A2A3E !important;
                color: #E0E0E0 !important;
            }
            .stButton>button {
                background-color: #8B5CF6;
                color: white;
                border-radius: 8px;
                padding: 0.5rem 1.5rem;
                border: none;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #7C3AED;
            }
            .css-1d391kg {
                background-color: #1A1A2E;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            /* Hide only specific header items — keep sidebar toggle visible */
            .stDeployButton {display: none !important;}
            [data-testid="stAppDeployButton"] {display: none !important;}
            [data-testid="stDecoration"] {display: none !important;}
            [data-testid="stStatusWidget"] {display: none !important;}
            /* Make sure sidebar collapse/expand button is always visible */
            [data-testid="stSidebarCollapseButton"] {visibility: visible !important; display: block !important;}
            [data-testid="collapsedControl"] {visibility: visible !important; display: block !important; z-index: 999999 !important;}
            [data-testid="baseButton-headerNoPadding"] {visibility: visible !important; display: block !important;}
        </style>
        """
    else:
        return """
        <style>
            /* Light mode (default) */
            .main-header {
                background: linear-gradient(90deg, #1F4E79 0%, #2E75B6 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .main-header h1 {
                color: white; margin: 0; font-size: 2.5rem; font-weight: bold;
            }
            .main-header p {
                color: #E1F0FF; margin: 0.5rem 0 0 0; font-size: 1.1rem;
            }
            .metric-card {
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                border-left: 4px solid #2E75B6;
                margin-bottom: 1rem;
            }
            .metric-value {
                font-size: 2.5rem; font-weight: bold; color: #1F4E79; margin: 0;
            }
            .metric-label {
                color: #666; font-size: 0.95rem; margin: 0;
            }
            .prediction-card {
                background: white;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin: 1rem 0;
            }
            .stButton>button {
                background-color: #2E75B6;
                color: white;
                border-radius: 8px;
                padding: 0.5rem 1.5rem;
                border: none;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #1F4E79;
            }
            .css-1d391kg {
                background-color: #F5F9FC;
            }
            /* Sidebar — visibly separated in light mode too */
            [data-testid="stSidebar"] {
                background-color: #EEF2F7 !important;
                border-right: 1px solid #CBD5E1 !important;
                box-shadow: 4px 0 12px rgba(0, 0, 0, 0.06) !important;
            }
            [data-testid="stSidebar"]::after {
                content: '';
                position: absolute;
                top: 0; right: 0; bottom: 0;
                width: 3px;
                background: linear-gradient(180deg, #2E75B6 0%, #1F4E79 100%);
                opacity: 0.5;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            /* Hide only specific header items — keep sidebar toggle visible */
            .stDeployButton {display: none !important;}
            [data-testid="stAppDeployButton"] {display: none !important;}
            [data-testid="stDecoration"] {display: none !important;}
            [data-testid="stStatusWidget"] {display: none !important;}
            /* Make sure sidebar collapse/expand button is always visible */
            [data-testid="stSidebarCollapseButton"] {visibility: visible !important; display: block !important;}
            [data-testid="collapsedControl"] {visibility: visible !important; display: block !important; z-index: 999999 !important;}
            [data-testid="baseButton-headerNoPadding"] {visibility: visible !important; display: block !important;}
        </style>
        """

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ==========================================================
# Load Data (Cached)
# ==========================================================
@st.cache_data
def load_database():
    for filename in ['final_database_v2.csv', 'final_database.csv']:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['label'] = df['Association Nature'].astype(str).str.strip().str.lower()
            df['Spice'] = df['Spice'].astype(str).str.strip()
            df['Disease'] = df['Disease'].astype(str).str.strip()
            if 'Publication Date' in df.columns:
                df['Year'] = pd.to_datetime(df['Publication Date'], errors='coerce').dt.year
            return df
    return None

# ==========================================================
# Load Model (Cached) - for live prediction
# ==========================================================
@st.cache_resource
def load_model():
    """Load PubMedBERT model for live prediction."""
    try:
        # Patch torch security check for older torch versions
        # (safe: we control the model file we're loading)
        try:
            import transformers.utils.import_utils as _tu
            _tu.check_torch_load_is_safe = lambda: None
        except Exception:
            pass
        try:
            import transformers.modeling_utils as _mu
            _mu.check_torch_load_is_safe = lambda: None
        except Exception:
            pass

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
        ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
        )

        # Load fine-tuned weights if available
        if os.path.exists('pubmedbert_final.pt'):
            try:
                # Try safe loading first (torch >= 2.6)
                state_dict = torch.load('pubmedbert_final.pt', map_location=device,
                                         weights_only=True)
            except Exception:
                # Fallback for older torch versions
                state_dict = torch.load('pubmedbert_final.pt', map_location=device,
                                         weights_only=False)
            model.load_state_dict(state_dict)
            model_status = "✅ Fine-tuned model loaded (90% accuracy)"
        else:
            model_status = "⚠️ Fine-tuned weights not found — using base PubMedBERT (accuracy may be lower)"

        model.to(device)
        model.eval()

        return {
            'model': model,
            'tokenizer': tokenizer,
            'device': device,
            'id2label': ID2LABEL,
            'status': model_status,
            'available': True
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}

def predict(text, model_dict):
    """Make prediction on input text."""
    import torch
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    device = model_dict['device']
    id2label = model_dict['id2label']

    inputs = tokenizer(text, max_length=256, padding='max_length',
                        truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = id2label[pred_idx]

    return {
        'label': pred_label,
        'confidence': float(probs[pred_idx]),
        'probs': {id2label[i]: float(probs[i]) for i in range(3)},
    }

df = load_database()

if df is None:
    st.error("❌ Database file not found. Please ensure `final_database_v2.csv` or `final_database.csv` exists.")
    st.stop()

# ==========================================================
# TOP INFO BAR (rotating fact + social icons)
# ==========================================================
from datetime import date as _date

def _get_daily_fact(df):
    """Pick a rotating 'did you know' fact based on day-of-year."""
    top_spice = df.groupby('Spice').size().sort_values(ascending=False)
    top_disease = df.groupby('Disease').size().sort_values(ascending=False)
    total = len(df)
    positive_pct = (df['label'] == 'positive').mean() * 100
    facts = [
        f"<b>{top_spice.index[0].title()}</b> is the most-studied spice with <b>{top_spice.iloc[0]:,}</b> associations",
        f"<b>{top_disease.index[0].title()}</b> tops the disease chart with <b>{top_disease.iloc[0]:,}</b> associations",
        f"Our database has <b>{total:,}</b> spice-disease associations mined from PubMed",
        f"<b>{positive_pct:.1f}%</b> of studied associations show positive/beneficial effects",
        f"We track <b>{df['Spice'].nunique():,}</b> unique spices across <b>{df['Disease'].nunique():,}</b> diseases",
        f"Our PubMedBERT model achieves <b>90.00%</b> accuracy — beating the SpiceRx baseline by 3.30%",
        f"The dataset spans <b>{df['PubMed ID'].nunique():,}</b> unique PubMed papers" if 'PubMed ID' in df.columns else f"Over <b>{total:,}</b> curated entries — largest of its kind",
    ]
    idx = _date.today().timetuple().tm_yday % len(facts)
    return facts[idx]

_fact = _get_daily_fact(df)

# Inline SVGs for social icons (uniform 18x18, brand-accurate paths)
_sch_svg = '<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M5.242 13.769L0 9.5 12 0l12 9.5-5.242 4.269C17.548 11.249 14.978 9.5 12 9.5c-2.977 0-5.548 1.748-6.758 4.269zM12 10a7 7 0 100 14 7 7 0 000-14z"/></svg>'
_li_svg  = '<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.063 2.063 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>'
_x_svg   = '<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231 5.451-6.231zm-1.161 17.52h1.833L7.084 4.126H5.117l11.966 15.644z"/></svg>'
_gh_svg  = '<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23a11.5 11.5 0 016 0c2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>'
_wk_svg  = '<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12.09 13.119c-.936 1.932-2.217 4.548-2.853 5.728-.616 1.074-1.127.931-1.532.029-1.406-3.321-4.293-9.144-5.651-12.409-.251-.601-.441-.987-.611-1.139-.171-.15-.674-.291-1.51-.421-.081-.02-.121-.079-.111-.181v-.591l.011-.032c.202-.011.982.03 3.146.03 1.673 0 3.083-.05 3.334-.03.181.011.231.06.231.181v.552c0 .12-.05.19-.161.19l-.482.03c-.381.03-.567.16-.567.42 0 .119.031.269.111.449.708 1.629 2.997 6.813 4.09 8.876.14.279.15.279.269.03l1.982-4.038-.271-.549-1.531-3.226c-.38-.771-1.06-2.309-1.302-2.749a1.148 1.148 0 00-.729-.66l-.681-.16c-.09-.02-.13-.09-.13-.191v-.6c0-.13.03-.19.14-.19h4.531c.17 0 .21.06.21.198v.516c0 .16-.03.24-.13.24l-.48.03c-.343.03-.532.152-.532.362 0 .18.09.4.18.6l1.98 4.35c.09.011 1.412-2.912 1.751-3.703.111-.243.211-.451.211-.622 0-.259-.14-.507-.68-.578l-.373-.028c-.152-.02-.171-.09-.171-.242v-.514c0-.121.04-.221.171-.221 1.51.03 3.081 0 3.293 0 .112 0 .213.061.213.221v.541c0 .19-.032.24-.183.24l-.323.028c-.612.061-.984.28-1.294.822-.32.61-1.732 3.373-2.442 4.86l2.204 4.66h.111l4.171-9.3c.161-.372.271-.755.271-1.007 0-.351-.144-.61-.752-.652l-.352-.03c-.13-.02-.15-.06-.15-.19v-.552c0-.121.04-.221.19-.221 1.031.028 1.632 0 2.732 0 .642 0 1.032-.019 1.833-.028.111 0 .21.038.21.221v.588c0 .09-.061.152-.161.152l-.652.06c-.472.06-.601.311-.821.784-.161.331-1.411 2.842-2.751 5.585-1.331 2.751-2.492 5.15-3.343 6.899-.352.75-.673 1.311-1.164 1.311-.291 0-.629-.15-.821-.522-.42-.782-1.371-2.882-2.033-4.312z"/></svg>'
_fb_svg  = '<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>'
_ig_svg  = '<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12 2.163c3.204 0 3.584.012 4.849.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.849.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z"/></svg>'

_info_bar_html = f"""
<div class="info-bar">
    <div class="info-bar-fact">
        <span class="info-bar-icon">💡</span>
        <span class="info-bar-label">Did you know?</span>
        <span class="info-bar-text">{_fact}</span>
    </div>
    <div class="info-bar-social">
        <a href="https://scholar.google.com/citations?hl=en&user=qyth_0QAAAAJ" target="_blank" rel="noopener noreferrer" class="social-link social-sch" title="Google Scholar">{_sch_svg}</a>
        <a href="https://in.linkedin.com/in/ganeshbagler" target="_blank" rel="noopener noreferrer" class="social-link social-li" title="LinkedIn">{_li_svg}</a>
        <a href="https://x.com/gansbags" target="_blank" rel="noopener noreferrer" class="social-link social-x" title="X (Twitter)">{_x_svg}</a>
        <a href="https://github.com/cosylabiiit" target="_blank" rel="noopener noreferrer" class="social-link social-gh" title="GitHub">{_gh_svg}</a>
        <a href="https://en.wikipedia.org/wiki/Ganesh_Bagler" target="_blank" rel="noopener noreferrer" class="social-link social-wk" title="Wikipedia">{_wk_svg}</a>
        <a href="https://www.facebook.com/ganesh.bagler" target="_blank" rel="noopener noreferrer" class="social-link social-fb" title="Facebook">{_fb_svg}</a>
        <a href="https://www.instagram.com/gansbags/?hl=bg" target="_blank" rel="noopener noreferrer" class="social-link social-ig" title="Instagram">{_ig_svg}</a>
    </div>
</div>
"""

st.markdown("""
<style>
.info-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
    color: #FFFFFF;
    padding: 8px 20px;
    border-radius: 8px;
    margin: 0 0 12px 0;
    font-size: 0.88rem;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
    flex-wrap: wrap;
    gap: 10px;
}
.info-bar-fact {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 0;
}
.info-bar-icon {
    font-size: 1.05rem;
    flex-shrink: 0;
}
.info-bar-label {
    font-weight: 700;
    color: #FEF3C7;
    flex-shrink: 0;
}
.info-bar-text {
    color: #FFFFFF !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.info-bar-text b {
    color: #FDE68A !important;
    font-weight: 700;
}
.info-bar-social {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
}
.social-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    color: #FFFFFF !important;
    background: rgba(255, 255, 255, 0.12);
    border-radius: 50%;
    text-decoration: none !important;
    transition: all 0.2s ease;
}
.social-link:hover {
    background: #FFFFFF;
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}
.social-sch:hover { color: #4285F4 !important; }
.social-li:hover  { color: #0A66C2 !important; }
.social-x:hover   { color: #000000 !important; }
.social-gh:hover  { color: #24292E !important; }
.social-wk:hover  { color: #202122 !important; }
.social-fb:hover  { color: #1877F2 !important; }
.social-ig:hover  { color: #E1306C !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(_info_bar_html, unsafe_allow_html=True)

# ==========================================================
# TOP NAVIGATION BAR (CoSyLab-style)
# ==========================================================
# Reduce Streamlit's default top padding
st.markdown("""
<style>
/* Give the nav bar breathing room from Streamlit's toolbar */
.main .block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
[data-testid="stMain"] .block-container {
    padding-top: 2.5rem !important;
}

/* Top nav bar — dark theme, balanced padding, no clipping */
.top-nav {
    background: linear-gradient(90deg, #1E293B 0%, #334155 100%);
    padding: 18px 28px;
    border-radius: 10px;
    margin: 0 0 24px 0;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    border: 1px solid #475569;
    width: 100%;
    min-height: 72px;
    box-sizing: border-box;
    flex-wrap: nowrap;
    overflow: visible;
    position: relative;
}
.top-nav-brand {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 14px;
    text-decoration: none !important;
    flex-shrink: 0;
}
.top-nav-brand:hover {
    opacity: 0.85;
    text-decoration: none !important;
}
.top-nav-logo {
    width: 48px;
    height: 48px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    border-radius: 50%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    overflow: hidden;
}
.top-nav-logo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}
.top-nav-text {
    display: flex;
    flex-direction: column;
    justify-content: center;
    line-height: 1;
}
.top-nav-title,
.top-nav .top-nav-title {
    color: #FFFFFF !important;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.top-nav-subtitle,
.top-nav .top-nav-subtitle {
    color: #CBD5E1 !important;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    margin: 3px 0 0 0 !important;
    padding: 0 !important;
    line-height: 1;
    text-transform: uppercase;
}
.top-nav-contact {
    background: #3B82F6;
    color: #FFFFFF !important;
    padding: 10px 28px;
    border-radius: 6px;
    text-decoration: none !important;
    font-weight: 600;
    font-size: 0.95rem;
    display: inline-block;
    flex-shrink: 0;
    white-space: nowrap;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
}
.top-nav-contact:hover {
    background: #2563EB;
    text-decoration: none !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(59, 130, 246, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Header HTML — uses the real CoSyLab logo (downloaded from cosylab.iiitd.edu.in)
_cosylab_logo_b64 = img_to_base64("cosylab_logo.png")
if _cosylab_logo_b64:
    _logo_img = f'<img src="{_cosylab_logo_b64}" alt="CoSyLab Logo" style="width:100%;height:100%;object-fit:contain;"/>'
else:
    # Fallback to remote URL if local file missing
    _logo_img = '<img src="https://cosylab.iiitd.edu.in/assets/images/cosylab_logo.png" alt="CoSyLab Logo" style="width:100%;height:100%;object-fit:contain;"/>'

top_nav_html = f"""
<div class="top-nav">
    <a href="https://cosylab.iiitd.edu.in/#" target="_blank" rel="noopener noreferrer" class="top-nav-brand">
        <span class="top-nav-logo">{_logo_img}</span>
        <div class="top-nav-text">
            <div class="top-nav-title">CoSyLab</div>
            <div class="top-nav-subtitle">COMPLEX SYSTEMS LAB</div>
        </div>
    </a>
    <a href="https://cosylab.iiitd.edu.in/#contact" target="_blank" rel="noopener noreferrer" class="top-nav-contact">Contact</a>
</div>
"""

st.markdown(top_nav_html, unsafe_allow_html=True)

# ==========================================================
# Main Header
# ==========================================================
st.markdown("""
<div class="main-header">
    <h1>🌿 SpiceMagic</h1>
    <p>AI-Powered Spice-Disease Association Database — 228,750 Labeled Associations from PubMed</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("🗂️ Navigation")

# Theme toggle
theme_choice = st.sidebar.radio(
    "🌓 Theme:",
    ["Light", "Dark"],
    index=0 if st.session_state.theme == 'Light' else 1,
    horizontal=True,
)
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    st.rerun()

st.sidebar.markdown("---")

PAGES = ["📊 Dashboard", "🧠 Live Prediction", "🔎 Search & Explore",
         "🌶️ Spice Profile", "🧬 Disease Profile", "⚖️ Compare",
         "🌊 Sankey Flow", "📈 Analytics", "💬 AI Chat",
         "✉️ Contact", "💡 About"]

# Initialize navigation history
if 'nav_history' not in st.session_state:
    st.session_state.nav_history = [PAGES[0]]

def _on_page_change():
    """Track user's page selection in history."""
    new_page = st.session_state.page_radio
    if not st.session_state.nav_history or st.session_state.nav_history[-1] != new_page:
        st.session_state.nav_history.append(new_page)
        # Keep history reasonably sized
        if len(st.session_state.nav_history) > 20:
            st.session_state.nav_history = st.session_state.nav_history[-20:]

page = st.sidebar.radio(
    "Select View:",
    PAGES,
    label_visibility="collapsed",
    key="page_radio",
    on_change=_on_page_change,
)

# Show ← Back button in sidebar when history has prior pages
if len(st.session_state.nav_history) > 1:
    def _go_back():
        st.session_state.nav_history.pop()
        st.session_state.page_radio = st.session_state.nav_history[-1]
    st.sidebar.button("← Back to previous", on_click=_go_back, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Database Stats")
st.sidebar.metric("Total Associations", f"{len(df):,}")
st.sidebar.metric("Unique Spices", f"{df['Spice'].nunique():,}")
st.sidebar.metric("Unique Diseases", f"{df['Disease'].nunique():,}")
if 'PubMed ID' in df.columns:
    st.sidebar.metric("Unique Papers", f"{df['PubMed ID'].nunique():,}")

# Visitor counter in sidebar (compact, always visible)
_visitor_total = get_visitor_count()
if _visitor_total is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👁️ Site Visitors")
    st.sidebar.metric("Total", f"{_visitor_total:,}", help="Cumulative visits across all users")

# ==========================================================
# Color scheme (adapts to theme)
# ==========================================================
COLORS = {
    'positive': '#27AE60',
    'negative': '#E74C3C',
    'neutral':  '#F39C12',
}

def make_pubmed_link(pubmed_id):
    """Return clickable PubMed link."""
    if pd.isna(pubmed_id) or str(pubmed_id).lower() in ['nan', '']:
        return ''
    pmid = str(int(float(pubmed_id))) if isinstance(pubmed_id, (int, float)) else str(pubmed_id).strip()
    return f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'

# ==========================================================
# PAGE: LIVE PREDICTION (NEW!)
# ==========================================================
if page == "🧠 Live Prediction":
    st.markdown("## 🧠 Explore Spice-Disease Associations")

    st.markdown("---")

    # ==================================================
    # QUICK SEARCH — Just spice + disease, no text
    # ==================================================
    if True:
        st.markdown("### 🔎 Quick Search — Instant Results from 228k Papers")
        st.markdown("**No text needed!** Just pick a spice and disease.")

        col1, col2 = st.columns(2)

        with col1:
            # Autocomplete-style spice selector
            all_spices_list = sorted(df['Spice'].unique().tolist())
            spice_sel = st.selectbox(
                "🌶️ Select Spice:",
                options=[''] + all_spices_list,
                index=0,
                placeholder="Type or select a spice..."
            )

        with col2:
            # Disease selector (filtered if spice is selected)
            if spice_sel:
                # Only show diseases that exist for this spice (top 100)
                spice_diseases = df[df['Spice'] == spice_sel]['Disease'].value_counts().head(200).index.tolist()
                disease_options = [''] + spice_diseases
                placeholder = f"Choose disease (showing {len(spice_diseases)} for {spice_sel})..."
            else:
                # Show top 200 most common diseases
                top_diseases_list = df['Disease'].value_counts().head(200).index.tolist()
                disease_options = [''] + top_diseases_list
                placeholder = "Choose disease..."

            disease_sel = st.selectbox(
                "🧬 Select Disease:",
                options=disease_options,
                index=0,
                placeholder=placeholder
            )

        # Quick suggestions
        st.markdown("**💡 Quick suggestions:**")
        sug_col1, sug_col2, sug_col3, sug_col4 = st.columns(4)

        suggestions = [
            ('Turmeric', 'inflammatory'),
            ('Ginger', 'nausea'),
            ('Cinnamon', 'diabetes'),
            ('Garlic', 'cholesterol'),
        ]

        for i, (sug_spice, sug_disease) in enumerate(suggestions):
            col = [sug_col1, sug_col2, sug_col3, sug_col4][i]
            with col:
                if st.button(f"{sug_spice} → {sug_disease}", key=f"sug_{i}", use_container_width=True):
                    st.session_state.quick_spice = sug_spice
                    st.session_state.quick_disease = sug_disease
                    st.rerun()

        # Apply session state
        if 'quick_spice' in st.session_state:
            spice_sel = st.session_state.quick_spice
            disease_sel = st.session_state.quick_disease
            del st.session_state.quick_spice
            del st.session_state.quick_disease

        # Show results
        if spice_sel and disease_sel:
            st.markdown("---")

            # Filter database
            results = df[
                (df['Spice'].str.lower() == spice_sel.lower()) &
                (df['Disease'].str.contains(disease_sel, case=False, na=False))
            ]

            if len(results) == 0:
                st.warning(f"⚠️ No papers found for **{spice_sel}** and **{disease_sel}**. Try different combination.")
            else:
                st.markdown(f"### 📊 Found **{len(results):,}** papers about **{spice_sel}** → **{disease_sel}**")

                # Overall verdict
                label_counts = results['label'].value_counts()
                top_label = label_counts.idxmax()
                top_pct = (label_counts[top_label] / len(results)) * 100

                emoji = '🟢' if top_label == 'positive' else ('🔴' if top_label == 'negative' else '🟡')
                color = COLORS[top_label]

                st.markdown(f"""
                <div class="prediction-card" style="border-left: 8px solid {color};">
                    <h2 style="margin: 0; color: {color};">{emoji} Consensus: {top_label.upper()}</h2>
                    <h4 style="margin: 0.5rem 0 0 0;">
                        {top_pct:.1f}% of {len(results):,} papers say <b>{top_label}</b>
                    </h4>
                </div>
                """, unsafe_allow_html=True)

                # Distribution bars
                st.markdown("#### 📊 Sentiment Distribution")
                col1, col2, col3 = st.columns(3)

                pos_count = (results['label'] == 'positive').sum()
                neu_count = (results['label'] == 'neutral').sum()
                neg_count = (results['label'] == 'negative').sum()

                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(39,174,96,0.1); border-radius: 10px; border-left: 4px solid #27AE60;">
                        <h1 style="color: #27AE60; margin: 0;">🟢 {pos_count:,}</h1>
                        <p style="margin: 0;">Positive ({pos_count/len(results)*100:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(243,156,18,0.1); border-radius: 10px; border-left: 4px solid #F39C12;">
                        <h1 style="color: #F39C12; margin: 0;">🟡 {neu_count:,}</h1>
                        <p style="margin: 0;">Neutral ({neu_count/len(results)*100:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(231,76,60,0.1); border-radius: 10px; border-left: 4px solid #E74C3C;">
                        <h1 style="color: #E74C3C; margin: 0;">🔴 {neg_count:,}</h1>
                        <p style="margin: 0;">Negative ({neg_count/len(results)*100:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Pie chart
                st.markdown("#### 🥧 Visual Distribution")
                fig = px.pie(
                    values=label_counts.values,
                    names=[l.capitalize() for l in label_counts.index],
                    color=[l.capitalize() for l in label_counts.index],
                    color_discrete_map={
                        'Positive': COLORS['positive'],
                        'Negative': COLORS['negative'],
                        'Neutral':  COLORS['neutral']
                    },
                    hole=0.5,
                )
                fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
                fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

                # Top papers with PubMed links
                st.markdown("#### 📚 Top 20 Papers")

                if 'max_confidence' in results.columns:
                    top_papers = results.nlargest(20, 'max_confidence')
                else:
                    top_papers = results.head(20)

                display_papers = top_papers[['Spice', 'Disease', 'label']].copy()
                display_papers = display_papers.rename(columns={'label': 'Sentiment'})

                if 'max_confidence' in top_papers.columns:
                    display_papers['Confidence'] = (top_papers['max_confidence'] * 100).round(1).astype(str) + '%'

                if 'Info (English)' in top_papers.columns:
                    display_papers['Info (English)'] = top_papers['Info (English)']

                if 'PubMed ID' in top_papers.columns:
                    display_papers['PubMed ID'] = top_papers['PubMed ID'].apply(
                        lambda x: str(int(float(x))) if pd.notna(x) and str(x).lower() != 'nan' else ''
                    )
                    display_papers['PubMed Link'] = top_papers['PubMed ID'].apply(make_pubmed_link)

                st.dataframe(
                    display_papers,
                    use_container_width=True,
                    height=500,
                    column_config={
                        "PubMed Link": st.column_config.LinkColumn(
                            "🔗 PubMed", display_text="Open ↗"),
                    }
                )

                # Export button
                csv = results.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download all {len(results):,} results (CSV)",
                    data=csv,
                    file_name=f'{spice_sel}_{disease_sel}_papers.csv',
                    mime='text/csv',
                    use_container_width=True,
                )

        elif spice_sel and not disease_sel:
            # Show spice profile mini-view
            st.markdown("---")
            spice_data = df[df['Spice'] == spice_sel]
            st.info(f"💡 **{spice_sel}** appears in **{len(spice_data):,} papers** across {spice_data['Disease'].nunique():,} diseases. Select a disease above to see specific results.")

            top_10 = spice_data['Disease'].value_counts().head(10)
            st.markdown(f"#### Top 10 diseases studied with {spice_sel}:")
            for i, (dis, cnt) in enumerate(top_10.items(), 1):
                st.markdown(f"{i}. **{dis}** — {cnt:,} papers")

        else:
            st.info("👆 Select a **spice** and **disease** above to see results.")

# ==========================================================
# PAGE: DASHBOARD
# ==========================================================
elif page == "📊 Dashboard":
    st.markdown("## 📊 Database Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pos_count = (df['label'] == 'positive').sum()
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #27AE60;">
            <p class="metric-value" style="color: #27AE60;">{pos_count:,}</p>
            <p class="metric-label">🟢 Positive Associations</p>
            <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.85rem;">{pos_count/len(df)*100:.1f}% of database</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        neu_count = (df['label'] == 'neutral').sum()
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #F39C12;">
            <p class="metric-value" style="color: #F39C12;">{neu_count:,}</p>
            <p class="metric-label">🟡 Neutral Associations</p>
            <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.85rem;">{neu_count/len(df)*100:.1f}% of database</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        neg_count = (df['label'] == 'negative').sum()
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #E74C3C;">
            <p class="metric-value" style="color: #E74C3C;">{neg_count:,}</p>
            <p class="metric-label">🔴 Negative Associations</p>
            <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.85rem;">{neg_count/len(df)*100:.1f}% of database</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_conf = df['max_confidence'].mean() if 'max_confidence' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{avg_conf*100:.1f}%</p>
            <p class="metric-label">🎯 Avg Confidence</p>
            <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.85rem;">Model prediction quality</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Sentiment Distribution")
        label_counts = df['label'].value_counts()
        fig = px.pie(
            values=label_counts.values,
            names=[l.capitalize() for l in label_counts.index],
            color=[l.capitalize() for l in label_counts.index],
            color_discrete_map={
                'Positive': COLORS['positive'],
                'Negative': COLORS['negative'],
                'Neutral':  COLORS['neutral'],
            },
            hole=0.4,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
        fig.update_layout(showlegend=True, height=400, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🌶️ Top 10 Most-Studied Spices")
        top_spices = df.groupby('Spice').size().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_spices.values, y=top_spices.index, orientation='h',
                      color=top_spices.values, color_continuous_scale='Blues')
        fig.update_layout(
            xaxis_title="Number of Papers", yaxis_title="",
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False, height=400, coloraxis_showscale=False,
            margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 Top Associations")
    tab1, tab2, tab3 = st.tabs(["🟢 Top Positive", "🔴 Top Negative", "🟡 Top Neutral"])

    for tab, label in zip([tab1, tab2, tab3], ['positive', 'negative', 'neutral']):
        with tab:
            subset = df[df['label'] == label]
            top_pairs = (subset.groupby(['Spice', 'Disease']).size()
                         .reset_index(name='Papers')
                         .sort_values('Papers', ascending=False).head(15))
            fig = px.bar(
                top_pairs, x='Papers',
                y=top_pairs.apply(lambda r: f"{r['Spice']} → {r['Disease']}", axis=1),
                orientation='h', color_discrete_sequence=[COLORS[label]])
            fig.update_layout(
                xaxis_title="Number of Papers", yaxis_title="",
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False, height=500,
                margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    if 'Year' in df.columns:
        st.markdown("### 📈 Research Growth Over Time")
        year_data = df.dropna(subset=['Year'])
        year_data = year_data[(year_data['Year'] >= 1980) & (year_data['Year'] <= 2025)]
        time_stats = year_data.groupby(['Year', 'label']).size().unstack(fill_value=0)

        fig = go.Figure()
        for label in ['positive', 'neutral', 'negative']:
            if label in time_stats.columns:
                fig.add_trace(go.Scatter(
                    x=time_stats.index, y=time_stats[label],
                    mode='lines', name=label.capitalize(),
                    line=dict(color=COLORS[label], width=2), stackgroup='one'))
        fig.update_layout(
            xaxis_title="Year", yaxis_title="Number of Papers",
            hovermode='x unified', height=400,
            margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PAGE: SEARCH & EXPLORE
# ==========================================================
elif page == "🔎 Search & Explore":
    st.markdown("## 🔎 Search & Explore")

    # Autocomplete option lists (cached, cleaned)
    @st.cache_data
    def _get_all_options(_df):
        import re
        # Only keep entries that start with a letter and are at least 4 chars long,
        # to filter out ~2,000 text-extraction artifacts like "-2", "AA", ", and"
        def _clean(series):
            s = series.dropna().astype(str).str.strip()
            valid = s[s.str.match(r'^[A-Za-z][\w\s\-\.\,\(\)\/\']{2,}$', na=False)]
            return sorted(valid.unique().tolist())
        return ([''] + _clean(_df['Spice']),
                [''] + _clean(_df['Disease']))
    _spice_options, _disease_options = _get_all_options(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        spice_filter = st.selectbox(
            "🌶️ Spice (autocomplete)",
            options=_spice_options,
            index=0,
            help="Start typing to filter — pick a specific spice or leave blank for all",
        )
    with col2:
        disease_filter = st.selectbox(
            "🧬 Disease (autocomplete)",
            options=_disease_options,
            index=0,
            help="Start typing to filter — pick a specific disease or leave blank for all",
        )
    with col3:
        sentiment_filter = st.multiselect(
            "😊 Sentiment", options=['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral'])
    with col4:
        if 'max_confidence' in df.columns:
            conf_threshold = st.slider("🎯 Min Confidence", 0.75, 1.0, 0.85, 0.05)
        else:
            conf_threshold = 0.75

    # Timeline slider (year range) — only if Year column exists
    if 'Year' in df.columns and df['Year'].notna().any():
        year_min = int(df['Year'].min())
        year_max = int(df['Year'].max())
        year_range = st.slider(
            "🕰️ Publication Year Range",
            min_value=year_min, max_value=year_max,
            value=(year_min, year_max),
            help="Filter results to papers published in this year range"
        )
    else:
        year_range = None

    filtered = df.copy()
    if spice_filter:
        filtered = filtered[filtered['Spice'] == spice_filter]
    if disease_filter:
        filtered = filtered[filtered['Disease'] == disease_filter]
    if sentiment_filter:
        filtered = filtered[filtered['label'].isin(sentiment_filter)]
    if 'max_confidence' in filtered.columns:
        filtered = filtered[filtered['max_confidence'] >= conf_threshold]
    if year_range and 'Year' in filtered.columns:
        filtered = filtered[
            (filtered['Year'] >= year_range[0]) &
            (filtered['Year'] <= year_range[1])
        ]

    st.markdown(f"### 📋 Found **{len(filtered):,}** associations")

    if len(filtered) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive", f"{(filtered['label']=='positive').sum():,}")
        with col2:
            st.metric("Neutral", f"{(filtered['label']=='neutral').sum():,}")
        with col3:
            st.metric("Negative", f"{(filtered['label']=='negative').sum():,}")

        col1, col2 = st.columns([1, 3])
        with col1:
            sort_by = st.selectbox("Sort by", ["Confidence", "Spice", "Disease"])
        with col2:
            n_show = st.slider("Show top N results", 10, 200, 50, 10)

        if sort_by == "Confidence" and 'max_confidence' in filtered.columns:
            display = filtered.nlargest(n_show, 'max_confidence')
        elif sort_by == "Spice":
            display = filtered.sort_values('Spice').head(n_show)
        else:
            display = filtered.sort_values('Disease').head(n_show)

        # Build display with PubMed ID + clickable link
        display_cols = display[['Spice', 'Disease', 'label']].copy()
        if 'max_confidence' in display.columns:
            display_cols['Confidence'] = (display['max_confidence'] * 100).round(1).astype(str) + '%'
        if 'Info (English)' in display.columns:
            display_cols['Info (English)'] = display['Info (English)']
        if 'PubMed ID' in display.columns:
            display_cols['PubMed ID'] = display['PubMed ID'].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x).lower() != 'nan' else ''
            )
            display_cols['PubMed Link'] = display['PubMed ID'].apply(make_pubmed_link)

        display_cols = display_cols.rename(columns={'label': 'Sentiment'})

        # Show table with clickable PubMed links
        st.dataframe(
            display_cols,
            use_container_width=True,
            height=500,
            column_config={
                "PubMed Link": st.column_config.LinkColumn(
                    "🔗 PubMed",
                    help="Open the article on PubMed",
                    validate="^https://pubmed.ncbi.nlm.nih.gov/",
                    display_text="Open ↗"
                ),
                "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            }
        )

        csv = filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered results (CSV)",
            data=csv,
            file_name=f'spicerx_search_results_{len(filtered)}.csv',
            mime='text/csv')
    else:
        st.info("🔎 No results found. Try adjusting your filters.")

# ==========================================================
# PAGE: SPICE PROFILE
# ==========================================================
elif page == "🌶️ Spice Profile":
    st.markdown("## 🌶️ Spice Profile")

    all_spices = sorted(df['Spice'].unique().tolist())
    selected_spice = st.selectbox("Select a spice:", all_spices,
                                    index=all_spices.index('Turmeric') if 'Turmeric' in all_spices else 0)

    spice_data = df[df['Spice'] == selected_spice]

    if len(spice_data) > 0:
        st.markdown(f"### 📊 {selected_spice} — Full Profile")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Papers", f"{len(spice_data):,}")
        with col2:
            pos_pct = (spice_data['label']=='positive').mean() * 100
            st.metric("Positive %", f"{pos_pct:.1f}%")
        with col3:
            unique_diseases = spice_data['Disease'].nunique()
            st.metric("Unique Diseases", f"{unique_diseases:,}")
        with col4:
            if 'PubMed ID' in spice_data.columns:
                st.metric("Unique Papers", f"{spice_data['PubMed ID'].nunique():,}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🥧 Sentiment Distribution")
            counts = spice_data['label'].value_counts()
            fig = px.pie(
                values=counts.values,
                names=[l.capitalize() for l in counts.index],
                color=[l.capitalize() for l in counts.index],
                color_discrete_map={
                    'Positive': COLORS['positive'],
                    'Negative': COLORS['negative'],
                    'Neutral':  COLORS['neutral']},
                hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 🧬 Top 15 Diseases Associated")
            top_diseases = spice_data.groupby('Disease').size().sort_values(ascending=False).head(15)
            fig = px.bar(x=top_diseases.values, y=top_diseases.index,
                          orientation='h', color=top_diseases.values,
                          color_continuous_scale='Blues')
            fig.update_layout(
                xaxis_title="Papers", yaxis_title="",
                yaxis={'categoryorder': 'total ascending'},
                height=350, coloraxis_showscale=False,
                margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        # Sample papers with PubMed ID + link
        st.markdown("#### 📚 Sample Recent Papers")
        recent = spice_data.head(20)
        if 'PubMed ID' in recent.columns:
            sample_df = recent[['Spice', 'Disease', 'label']].copy()
            sample_df['PubMed ID'] = recent['PubMed ID'].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x).lower() != 'nan' else ''
            )
            sample_df['PubMed Link'] = recent['PubMed ID'].apply(make_pubmed_link)
            if 'max_confidence' in recent.columns:
                sample_df['Confidence'] = (recent['max_confidence'] * 100).round(1).astype(str) + '%'
            sample_df = sample_df.rename(columns={'label': 'Sentiment'})
            st.dataframe(sample_df, use_container_width=True,
                          column_config={
                              "PubMed Link": st.column_config.LinkColumn(
                                  "🔗 PubMed", display_text="Open ↗"),
                          })

        st.markdown("#### 📊 Sentiment Breakdown for Top Diseases")
        top10_disease = spice_data.groupby('Disease').size().sort_values(ascending=False).head(10).index.tolist()
        subset = spice_data[spice_data['Disease'].isin(top10_disease)]
        pivot = subset.groupby(['Disease', 'label']).size().unstack(fill_value=0)
        pivot = pivot.reindex(top10_disease)

        fig = go.Figure()
        for label in ['positive', 'neutral', 'negative']:
            if label in pivot.columns:
                fig.add_trace(go.Bar(
                    y=pivot.index, x=pivot[label],
                    name=label.capitalize(), orientation='h',
                    marker_color=COLORS[label]))
        fig.update_layout(barmode='stack', xaxis_title="Number of Papers",
                          height=400, margin=dict(t=20, b=20, l=20, r=20),
                          yaxis={'autorange': 'reversed'})
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PAGE: DISEASE PROFILE
# ==========================================================
elif page == "🧬 Disease Profile":
    st.markdown("## 🧬 Disease Profile")

    all_diseases = sorted(df['Disease'].unique().tolist())
    default_idx = 0
    if 'cancer' in all_diseases:
        default_idx = all_diseases.index('cancer')
    selected_disease = st.selectbox("Select a disease:", all_diseases, index=default_idx)

    disease_data = df[df['Disease'] == selected_disease]

    if len(disease_data) > 0:
        st.markdown(f"### 📊 {selected_disease} — Full Profile")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Papers", f"{len(disease_data):,}")
        with col2:
            pos_pct = (disease_data['label']=='positive').mean() * 100
            st.metric("Positive %", f"{pos_pct:.1f}%")
        with col3:
            unique_spices = disease_data['Spice'].nunique()
            st.metric("Unique Spices Tested", f"{unique_spices:,}")
        with col4:
            if 'PubMed ID' in disease_data.columns:
                st.metric("Unique Papers", f"{disease_data['PubMed ID'].nunique():,}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🥧 Sentiment Distribution")
            counts = disease_data['label'].value_counts()
            fig = px.pie(
                values=counts.values,
                names=[l.capitalize() for l in counts.index],
                color=[l.capitalize() for l in counts.index],
                color_discrete_map={
                    'Positive': COLORS['positive'],
                    'Negative': COLORS['negative'],
                    'Neutral':  COLORS['neutral']},
                hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 🌶️ Top 15 Spices Studied")
            top_spices = disease_data.groupby('Spice').size().sort_values(ascending=False).head(15)
            fig = px.bar(x=top_spices.values, y=top_spices.index,
                          orientation='h', color=top_spices.values,
                          color_continuous_scale='Greens')
            fig.update_layout(
                xaxis_title="Papers", yaxis_title="",
                yaxis={'categoryorder': 'total ascending'},
                height=350, coloraxis_showscale=False,
                margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        # Sample papers with PubMed ID + link
        st.markdown("#### 📚 Sample Recent Papers")
        recent = disease_data.head(20)
        if 'PubMed ID' in recent.columns:
            sample_df = recent[['Spice', 'Disease', 'label']].copy()
            sample_df['PubMed ID'] = recent['PubMed ID'].apply(
                lambda x: str(int(float(x))) if pd.notna(x) and str(x).lower() != 'nan' else ''
            )
            sample_df['PubMed Link'] = recent['PubMed ID'].apply(make_pubmed_link)
            if 'max_confidence' in recent.columns:
                sample_df['Confidence'] = (recent['max_confidence'] * 100).round(1).astype(str) + '%'
            sample_df = sample_df.rename(columns={'label': 'Sentiment'})
            st.dataframe(sample_df, use_container_width=True,
                          column_config={
                              "PubMed Link": st.column_config.LinkColumn(
                                  "🔗 PubMed", display_text="Open ↗"),
                          })

        st.markdown("#### 📊 Sentiment Breakdown for Top Spices")
        top10_spice = disease_data.groupby('Spice').size().sort_values(ascending=False).head(10).index.tolist()
        subset = disease_data[disease_data['Spice'].isin(top10_spice)]
        pivot = subset.groupby(['Spice', 'label']).size().unstack(fill_value=0)
        pivot = pivot.reindex(top10_spice)

        fig = go.Figure()
        for label in ['positive', 'neutral', 'negative']:
            if label in pivot.columns:
                fig.add_trace(go.Bar(
                    y=pivot.index, x=pivot[label],
                    name=label.capitalize(), orientation='h',
                    marker_color=COLORS[label]))
        fig.update_layout(barmode='stack', xaxis_title="Number of Papers",
                          height=400, margin=dict(t=20, b=20, l=20, r=20),
                          yaxis={'autorange': 'reversed'})
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PAGE: COMPARE (Side-by-Side)
# ==========================================================
elif page == "⚖️ Compare":
    st.markdown("## ⚖️ Compare — Side-by-Side Spice Analysis")
    st.markdown("**Pick any two spices to see their profiles side-by-side.**")

    all_spices = sorted(df['Spice'].unique().tolist())

    def _stats_for(sub_df):
        """Compute standard stats for a filtered dataframe."""
        if len(sub_df) == 0:
            return None
        return {
            'total': len(sub_df),
            'positive': (sub_df['label'] == 'positive').sum(),
            'negative': (sub_df['label'] == 'negative').sum(),
            'neutral':  (sub_df['label'] == 'neutral').sum(),
            'pos_pct': (sub_df['label'] == 'positive').mean() * 100,
            'neg_pct': (sub_df['label'] == 'negative').mean() * 100,
            'unique_papers': sub_df['PubMed ID'].nunique() if 'PubMed ID' in sub_df.columns else 0,
        }

    def _render_stats_column(name, sub_df, entity_type):
        """Render one side of the comparison."""
        stats = _stats_for(sub_df)
        if stats is None:
            st.warning(f"No data found for **{name}**")
            return
        st.markdown(f"### {name}")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Papers", f"{stats['total']:,}")
            st.metric("Positive %", f"{stats['pos_pct']:.1f}%")
        with c2:
            st.metric("Unique PubMed", f"{stats['unique_papers']:,}")
            st.metric("Negative %", f"{stats['neg_pct']:.1f}%")
        # Sentiment pie
        counts = sub_df['label'].value_counts()
        fig = px.pie(
            values=counts.values,
            names=[l.capitalize() for l in counts.index],
            color=[l.capitalize() for l in counts.index],
            color_discrete_map={'Positive': COLORS['positive'],
                                'Negative': COLORS['negative'],
                                'Neutral':  COLORS['neutral']},
            hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"cmp_pie_{entity_type}_{name}")
        # Top-5 related entities
        other_col = 'Disease' if entity_type == 'Spice' else 'Spice'
        top5 = sub_df.groupby(other_col).size().sort_values(ascending=False).head(5)
        st.markdown(f"**Top 5 {other_col}s:**")
        for entity, count in top5.items():
            st.markdown(f"- {entity} — `{count:,}` papers")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        spice_a = st.selectbox("Select first spice", all_spices,
                                index=all_spices.index('Turmeric') if 'Turmeric' in all_spices else 0,
                                key="cmp_spice_a")
    with c2:
        spice_b = st.selectbox("Select second spice", all_spices,
                                index=all_spices.index('Cinnamon') if 'Cinnamon' in all_spices else min(1, len(all_spices)-1),
                                key="cmp_spice_b")
    if spice_a == spice_b:
        st.warning("⚠️ Please pick two different spices.")
    else:
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            _render_stats_column(spice_a, df[df['Spice'] == spice_a], 'Spice')
        with col_b:
            _render_stats_column(spice_b, df[df['Spice'] == spice_b], 'Spice')

# ==========================================================
# PAGE: SANKEY FLOW (replaces Network Graph)
# ==========================================================
elif page == "🌊 Sankey Flow":
    st.markdown("## 🌊 Sankey Flow — How Spices Flow Through Sentiment to Diseases")
    st.markdown("**A publication-quality flow visualization. Widths of the streams represent paper counts.**")

    # Filters — smaller defaults for cleaner labels
    col1, col2, col3 = st.columns(3)
    with col1:
        top_n_spices = st.slider("🌶️ Top N Spices", 3, 15, 8, 1, key="sky_ns",
                                  help="Fewer = clearer labels")
    with col2:
        top_n_diseases = st.slider("🧬 Top N Diseases", 3, 15, 8, 1, key="sky_nd",
                                    help="Fewer = clearer labels")
    with col3:
        sentiment_scope = st.selectbox(
            "🎨 Sentiment flows to show",
            ["All (Positive, Neutral, Negative)", "Positive only", "Negative only", "Neutral only"],
            index=0, key="sky_sent")

    # Get top entities by paper count
    top_spices_list = df.groupby('Spice').size().sort_values(ascending=False).head(top_n_spices).index.tolist()
    top_diseases_list = df.groupby('Disease').size().sort_values(ascending=False).head(top_n_diseases).index.tolist()

    # Restrict to sentiments requested
    if sentiment_scope == "Positive only":
        sentiments_to_show = ['positive']
    elif sentiment_scope == "Negative only":
        sentiments_to_show = ['negative']
    elif sentiment_scope == "Neutral only":
        sentiments_to_show = ['neutral']
    else:
        sentiments_to_show = ['positive', 'neutral', 'negative']

    subset = df[
        df['Spice'].isin(top_spices_list) &
        df['Disease'].isin(top_diseases_list) &
        df['label'].isin(sentiments_to_show)
    ].copy()

    if len(subset) == 0:
        st.warning("⚠️ No rows match these filters — try widening them.")
    else:
        # ==============================
        # Build Sankey structure
        # Nodes: [top spices...] + [sentiments...] + [top diseases...]
        # Links: Spice → Sentiment + Sentiment → Disease
        # ==============================
        sentiment_labels = [s.capitalize() for s in sentiments_to_show]
        spice_labels = [s for s in top_spices_list if s in subset['Spice'].unique()]
        disease_labels = [d for d in top_diseases_list if d in subset['Disease'].unique()]

        node_labels = spice_labels + sentiment_labels + disease_labels

        # Node index mapping
        spice_idx    = {s: i for i, s in enumerate(spice_labels)}
        sent_idx     = {s.lower(): len(spice_labels) + i for i, s in enumerate(sentiment_labels)}
        disease_idx  = {d: len(spice_labels) + len(sentiment_labels) + i for i, d in enumerate(disease_labels)}

        # Theme-aware node colors
        is_dark = st.session_state.get('theme', 'Light') == 'Dark'
        spice_node_col   = '#F97316'   # orange for spices
        disease_node_col = '#3B82F6'   # blue for diseases
        sent_colors = {
            'positive': '#10B981',   # green
            'neutral':  '#F59E0B',   # amber
            'negative': '#EF4444',   # red
        }
        sent_link_colors_rgba = {
            'positive': 'rgba(16, 185, 129, 0.35)',
            'neutral':  'rgba(245, 158, 11, 0.35)',
            'negative': 'rgba(239, 68, 68, 0.35)',
        }

        node_colors = (
            [spice_node_col] * len(spice_labels) +
            [sent_colors[s.lower()] for s in sentiment_labels] +
            [disease_node_col] * len(disease_labels)
        )

        # Build links
        sources, targets, values, link_colors, link_labels = [], [], [], [], []

        # Left half: Spice → Sentiment
        left_group = subset.groupby(['Spice', 'label']).size().reset_index(name='Papers')
        for _, row in left_group.iterrows():
            spice, sentiment, papers = row['Spice'], row['label'], row['Papers']
            if spice in spice_idx and sentiment in sent_idx:
                sources.append(spice_idx[spice])
                targets.append(sent_idx[sentiment])
                values.append(int(papers))
                link_colors.append(sent_link_colors_rgba[sentiment])
                link_labels.append(f"{spice} → {sentiment.capitalize()}: {papers:,} papers")

        # Right half: Sentiment → Disease
        right_group = subset.groupby(['label', 'Disease']).size().reset_index(name='Papers')
        for _, row in right_group.iterrows():
            sentiment, disease, papers = row['label'], row['Disease'], row['Papers']
            if sentiment in sent_idx and disease in disease_idx:
                sources.append(sent_idx[sentiment])
                targets.append(disease_idx[disease])
                values.append(int(papers))
                link_colors.append(sent_link_colors_rgba[sentiment])
                link_labels.append(f"{sentiment.capitalize()} → {disease}: {papers:,} papers")

        text_color = '#F1F5F9' if is_dark else '#1E293B'
        paper_bg = 'rgba(0,0,0,0)'

        # Sharp text needs solid backgrounds — avoid transparent for label clarity
        node_line_col = 'rgba(255,255,255,0.85)' if is_dark else 'rgba(0,0,0,0.55)'

        # Compute dynamic height so labels never overlap: min 640px, add 45px per max node
        max_col_nodes = max(len(spice_labels), len(disease_labels))
        chart_height = max(640, 90 + max_col_nodes * 55)

        # Build the Sankey figure
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            valueformat=',',
            valuesuffix=' papers',
            textfont=dict(color=text_color, size=15, family='Arial Black'),
            node=dict(
                pad=28,
                thickness=26,
                line=dict(color=node_line_col, width=1),
                label=node_labels,
                color=node_colors,
                hovertemplate='<b>%{label}</b><br>Total flow: %{value:,} papers<extra></extra>',
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                customdata=link_labels,
                hovertemplate='%{customdata}<extra></extra>',
                line=dict(color='rgba(0,0,0,0)', width=0),
            ),
        )])

        # Solid paper background so anti-aliased text renders crisply (transparent
        # backgrounds cause the browser's compositor to smear label pixels)
        solid_bg = '#1E1E2E' if is_dark else '#FFFFFF'

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Spice → Sentiment → Disease Flow</b><br>"
                    f"<span style='font-size:13px;'>Showing {len(spice_labels)} spices × "
                    f"{len(disease_labels)} diseases · {len(subset):,} associations</span>"
                ),
                x=0.5,
                xanchor='center',
                font=dict(size=20, color=text_color, family='Arial Black'),
            ),
            height=chart_height,
            paper_bgcolor=solid_bg,
            plot_bgcolor=solid_bg,
            font=dict(color=text_color, size=15, family='Arial'),
            margin=dict(l=40, r=40, t=110, b=30),
        )

        # Higher-quality rendering for crisper labels
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                'displaylogo': False,
                'toImageButtonOptions': {'scale': 3},  # 3x DPI when exporting
                'responsive': True,
            },
        )

        # ==============================
        # Info cards below the chart
        # ==============================
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Spices Shown",   f"{len(spice_labels)}")
        with c2: st.metric("Diseases Shown", f"{len(disease_labels)}")
        with c3: st.metric("Total Papers",   f"{len(subset):,}")
        with c4: st.metric("Flows Drawn",    f"{len(sources)}")

        st.markdown("---")
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("### 📖 How to Read")
            st.markdown("""
            - **Left column** — top spices (🟠 orange)
            - **Middle column** — sentiment classes (🟢 positive, 🟠 neutral, 🔴 negative)
            - **Right column** — top diseases (🔵 blue)
            - **Stream width** = number of papers
            - **Colored streams** flow by sentiment
            """)
        with h2:
            st.markdown("### 🎯 How to Interact")
            st.markdown("""
            - **Hover** on any stream to see exact paper count
            - **Hover** on nodes to see total flow through that node
            - **Drag nodes** vertically to rearrange for clarity
            - **Adjust filters** above to explore different subsets
            """)

        # Detailed flow table
        with st.expander("📋 View all flows as table"):
            # Combine both sides
            flow_rows = []
            for s, sent, p in left_group[['Spice', 'label', 'Papers']].itertuples(index=False):
                if s in spice_idx and sent in sent_idx:
                    flow_rows.append({'From': s, 'To': sent.capitalize(), 'Type': 'Spice → Sentiment', 'Papers': int(p)})
            for sent, d, p in right_group[['label', 'Disease', 'Papers']].itertuples(index=False):
                if sent in sent_idx and d in disease_idx:
                    flow_rows.append({'From': sent.capitalize(), 'To': d, 'Type': 'Sentiment → Disease', 'Papers': int(p)})
            flow_df = pd.DataFrame(flow_rows).sort_values('Papers', ascending=False)
            st.dataframe(flow_df, use_container_width=True, height=400)

            csv = flow_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {len(flow_df)} flows (CSV)",
                data=csv,
                file_name=f'sankey_flows_{sentiment_scope.split()[0].lower()}.csv',
                mime='text/csv',
            )

# ==========================================================
# PAGE: ANALYTICS
# ==========================================================
elif page == "📈 Analytics":
    st.markdown("## 📊 Analytics")

    tab1, tab2, tab3 = st.tabs(["🌶️ Beneficial Spices", "📈 Time Trends", "🎯 Confidence Analysis"])

    with tab1:
        st.markdown("### 🌶️ Most Beneficial Spices (Highest Positive %)")
        min_papers = st.slider("Minimum papers", 100, 5000, 500, 100)
        spice_stats = df.groupby('Spice').agg(
            Total=('label', 'count'),
            Positive=('label', lambda x: (x == 'positive').sum())).reset_index()
        spice_stats['Pos %'] = (spice_stats['Positive'] / spice_stats['Total'] * 100).round(1)
        beneficial = spice_stats[spice_stats['Total'] >= min_papers].sort_values('Pos %', ascending=False).head(20)

        if len(beneficial) > 0:
            fig = px.bar(beneficial, x='Pos %', y='Spice', orientation='h',
                          color='Total', color_continuous_scale='Greens',
                          hover_data=['Total', 'Positive'])
            fig.update_layout(
                xaxis_title="% Positive Associations", yaxis_title="",
                yaxis={'categoryorder': 'total ascending'},
                height=600, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 📈 Research Trends by Spice")
        top_spices_list = df.groupby('Spice').size().sort_values(ascending=False).head(20).index.tolist()
        selected_spices = st.multiselect("Select spices to compare:",
                                          options=top_spices_list,
                                          default=top_spices_list[:5])

        if 'Year' in df.columns and selected_spices:
            year_data = df[df['Spice'].isin(selected_spices)].dropna(subset=['Year'])
            year_data = year_data[(year_data['Year'] >= 1990) & (year_data['Year'] <= 2025)]
            trends = year_data.groupby(['Year', 'Spice']).size().reset_index(name='Papers')
            fig = px.line(trends, x='Year', y='Papers', color='Spice', markers=True)
            fig.update_layout(
                xaxis_title="Year", yaxis_title="Number of Papers",
                hovermode='x unified', height=500,
                margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'max_confidence' in df.columns:
            st.markdown("### 🎯 Model Confidence Analysis")
            fig = px.histogram(df, x='max_confidence', color='label',
                                nbins=40, opacity=0.7,
                                color_discrete_map={
                                    'positive': COLORS['positive'],
                                    'negative': COLORS['negative'],
                                    'neutral':  COLORS['neutral']})
            fig.update_layout(
                xaxis_title="Prediction Confidence",
                yaxis_title="Number of Predictions",
                height=450, barmode='overlay',
                margin=dict(t=20, b=20, l=20, r=20))
            fig.add_vline(x=0.75, line_dash="dash", line_color="red",
                          annotation_text="Threshold: 0.75")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Median Confidence", f"{df['max_confidence'].median()*100:.1f}%")
            with col2:
                st.metric("High Confidence (>0.95)",
                          f"{(df['max_confidence']>0.95).sum():,}",
                          f"{(df['max_confidence']>0.95).mean()*100:.1f}%")
            with col3:
                st.metric("Very High (>0.99)",
                          f"{(df['max_confidence']>0.99).sum():,}",
                          f"{(df['max_confidence']>0.99).mean()*100:.1f}%")

# ==========================================================
# PAGE: AI CHAT
# ==========================================================
elif page == "💬 AI Chat":
    st.markdown("## 💬 AI Chat — Ask About Spices & Diseases")
    st.markdown("**Natural language interface. Ask questions in plain English — I'll query the database for you.**")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Example queries
    with st.expander("💡 Example questions to try", expanded=False):
        st.markdown("""
        - "What spices help with **diabetes**?"
        - "Show me negative effects of **turmeric**"
        - "How many papers on **cinnamon** and **cancer**?"
        - "Top spices for **inflammation**"
        - "Compare **turmeric** vs **ginger**"
        - "What is the most-studied disease?"
        """)

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # NL parser: extract intent + entities from user query
    def _parse_query(q, df):
        q_low = q.lower()
        spice_list = df['Spice'].unique().tolist()
        disease_list = df['Disease'].unique().tolist()

        found_spices = [s for s in spice_list if s.lower() in q_low]
        found_diseases = [d for d in disease_list if d.lower() in q_low]

        # Sentiment intent
        if any(w in q_low for w in ['help', 'benefit', 'positive', 'good', 'cure', 'treat']):
            sentiment = 'positive'
        elif any(w in q_low for w in ['harm', 'negative', 'adverse', 'bad', 'side effect', 'toxic']):
            sentiment = 'negative'
        elif 'neutral' in q_low or 'no effect' in q_low:
            sentiment = 'neutral'
        else:
            sentiment = None

        return {
            'spices': found_spices,
            'diseases': found_diseases,
            'sentiment': sentiment,
            'compare': 'compare' in q_low or ' vs ' in q_low or 'versus' in q_low,
            'count':   any(w in q_low for w in ['how many', 'count', 'number of']),
            'top':     any(w in q_low for w in ['top', 'best', 'most', 'highest']),
            'most_studied': 'most' in q_low and 'stud' in q_low,
        }

    def _answer_query(q, df):
        p = _parse_query(q, df)

        # Case: most-studied
        if p['most_studied']:
            if 'disease' in q.lower():
                top = df.groupby('Disease').size().sort_values(ascending=False).head(5)
                lines = [f"**Top 5 most-studied diseases:**"]
                for name, cnt in top.items():
                    lines.append(f"- **{name}** — {cnt:,} papers")
                return '\n'.join(lines)
            else:
                top = df.groupby('Spice').size().sort_values(ascending=False).head(5)
                lines = [f"**Top 5 most-studied spices:**"]
                for name, cnt in top.items():
                    lines.append(f"- **{name}** — {cnt:,} papers")
                return '\n'.join(lines)

        # Case: compare two entities
        if p['compare'] and len(p['spices']) >= 2:
            s1, s2 = p['spices'][:2]
            d1 = df[df['Spice'] == s1]
            d2 = df[df['Spice'] == s2]
            return (
                f"**Comparing {s1} vs {s2}:**\n"
                f"| Metric | {s1} | {s2} |\n"
                f"|---|---|---|\n"
                f"| Papers | {len(d1):,} | {len(d2):,} |\n"
                f"| Positive % | {(d1['label']=='positive').mean()*100:.1f}% | {(d2['label']=='positive').mean()*100:.1f}% |\n"
                f"| Unique diseases | {d1['Disease'].nunique():,} | {d2['Disease'].nunique():,} |\n\n"
                f"👉 See the **⚖️ Compare** page for full side-by-side view."
            )

        # Case: spice + disease pair
        if p['spices'] and p['diseases']:
            s, d = p['spices'][0], p['diseases'][0]
            pair = df[(df['Spice'] == s) & (df['Disease'] == d)]
            if len(pair) == 0:
                return f"No direct papers found linking **{s}** and **{d}**."
            top_sent = pair['label'].value_counts().idxmax()
            return (
                f"Found **{len(pair):,}** papers on **{s}** × **{d}**.\n\n"
                f"- Dominant sentiment: **{top_sent.capitalize()}**\n"
                f"- Positive: {(pair['label']=='positive').sum():,}\n"
                f"- Negative: {(pair['label']=='negative').sum():,}\n"
                f"- Neutral: {(pair['label']=='neutral').sum():,}"
            )

        # Case: spices for a disease
        if p['diseases'] and not p['spices']:
            d = p['diseases'][0]
            sub = df[df['Disease'] == d]
            if p['sentiment']:
                sub = sub[sub['label'] == p['sentiment']]
            if len(sub) == 0:
                return f"No {p['sentiment'] or 'matching'} results found for **{d}**."
            top = sub.groupby('Spice').size().sort_values(ascending=False).head(8)
            label = p['sentiment'] or 'associated'
            lines = [f"**Top spices with {label} effect on {d}:**"]
            for s, cnt in top.items():
                lines.append(f"- **{s}** — {cnt:,} papers")
            return '\n'.join(lines)

        # Case: diseases for a spice
        if p['spices'] and not p['diseases']:
            s = p['spices'][0]
            sub = df[df['Spice'] == s]
            if p['sentiment']:
                sub = sub[sub['label'] == p['sentiment']]
            if len(sub) == 0:
                return f"No {p['sentiment'] or 'matching'} results found for **{s}**."
            top = sub.groupby('Disease').size().sort_values(ascending=False).head(8)
            label = p['sentiment'] or 'associated'
            lines = [f"**Top diseases {label} to {s}:**"]
            for d, cnt in top.items():
                lines.append(f"- **{d}** — {cnt:,} papers")
            return '\n'.join(lines)

        # Fallback
        return (
            "I couldn't find a specific spice or disease in your question. "
            "Try mentioning a **spice** (e.g. turmeric, cinnamon) or a **disease** "
            "(e.g. diabetes, cancer). See the examples above for query patterns."
        )

    # Chat input
    user_q = st.chat_input("Ask a question about spices, diseases, or associations...")
    if user_q:
        st.session_state.chat_history.append({'role': 'user', 'content': user_q})
        with st.chat_message('user'):
            st.markdown(user_q)
        answer = _answer_query(user_q, df)
        st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# ==========================================================
# PAGE: CONTACT
# ==========================================================
elif page == "✉️ Contact":
    st.markdown("## ✉️ Contact Us")
    st.markdown("**Have questions, feedback, or want to collaborate? Reach out below.**")

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("contact_form", clear_on_submit=True):
            st.markdown("### 📝 Send a message")
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Your name *", placeholder="John Doe")
            with c2:
                email = st.text_input("Your email *", placeholder="you@example.com")
            subject = st.selectbox("Subject *", [
                "General Inquiry", "Research Collaboration", "Bug Report",
                "Feature Request", "Data Access", "Citation Request", "Other"])
            message = st.text_area("Message *", placeholder="Type your message...", height=180)
            agree = st.checkbox("I agree to be contacted about this inquiry.")
            submitted = st.form_submit_button("📤 Send Message", type="primary", use_container_width=True)

            if submitted:
                if not name.strip() or not email.strip() or not message.strip():
                    st.error("❌ Please fill in all required fields (name, email, message).")
                elif '@' not in email or '.' not in email:
                    st.error("❌ Please enter a valid email address.")
                elif not agree:
                    st.error("❌ Please agree to be contacted.")
                else:
                    # Send real email via Formsubmit.co (no signup, forwards to bagler@iiitd.ac.in)
                    import requests as _rq
                    with st.spinner("📤 Sending your message..."):
                        try:
                            resp = _rq.post(
                                "https://formsubmit.co/ajax/bagler@iiitd.ac.in",
                                data={
                                    "name":    name,
                                    "email":   email,
                                    "subject": f"[SpiceMagic] {subject}",
                                    "message": message,
                                    "_subject": f"[SpiceMagic Contact] {subject} — from {name}",
                                    "_template": "table",
                                    "_captcha": "false",
                                },
                                headers={"Accept": "application/json"},
                                timeout=15,
                            )
                            data = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {}
                            success = resp.status_code == 200 and data.get('success') in (True, "true", "True")
                        except Exception as _e:
                            success = False
                            data = {"error": str(_e)}

                    if success:
                        st.success(f"✅ Thanks **{name}**! Your message was sent to Prof. Bagler at **bagler@iiitd.ac.in**. He'll respond to **{email}** shortly.")
                        show_toast(f"Message sent from {name}!", 'success')
                        # First-time activation notice
                        if 'formsubmit_activated' not in st.session_state:
                            st.session_state.formsubmit_activated = True
                            st.info("🔔 **First-time note:** Prof. Bagler will receive an activation email from Formsubmit — one click and all future messages arrive directly in his inbox.")
                    else:
                        st.error(
                            f"⚠️ Couldn't send via Formsubmit ({data.get('message', data.get('error', 'unknown error'))}). "
                            f"Please use the mailto link below or email **bagler@iiitd.ac.in** directly."
                        )
                        show_toast("Message sending failed", 'error')
                        # Mailto fallback
                        import urllib.parse as _up
                        body = f"From: {name} <{email}>\nSubject: {subject}\n\n{message}"
                        mailto = f"mailto:bagler@iiitd.ac.in?subject={_up.quote(subject)}&body={_up.quote(body)}"
                        st.markdown(f"[📧 Open in your email client]({mailto})")

    with col2:
        st.markdown("### 📍 Contact Info")
        st.markdown("""
        **CoSyLab**
        Complex Systems Laboratory
        IIIT-Delhi
        Okhla Industrial Estate, Phase III
        New Delhi 110020, India

        **Email:** bagler@iiitd.ac.in
        **Website:** [cosylab.iiitd.edu.in](https://cosylab.iiitd.edu.in/)
        """)

        st.markdown("### 🌐 Follow")
        # Theme-aware backgrounds (matches app session_state theme)
        _cs_is_dark = st.session_state.get('theme', 'Light') == 'Dark'
        _cs_bg     = '#1E293B' if _cs_is_dark else '#F8FAFC'
        _cs_border = '#475569' if _cs_is_dark else '#E5E7EB'
        _cs_text   = '#F1F5F9' if _cs_is_dark else '#1E293B'
        st.markdown(f"""
        <style>
        .contact-social {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .contact-social a {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            background: {_cs_bg};
            border: 1px solid {_cs_border};
            color: {_cs_text} !important;
            text-decoration: none !important;
            font-size: 0.92rem;
            font-weight: 500;
            transition: all 0.15s ease;
        }}
        .contact-social a:hover {{
            transform: translateX(4px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .contact-social .icon-wrap {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            color: #FFFFFF;
        }}
        .contact-social .ic-sch  {{ background: #4285F4; }}
        .contact-social .ic-li   {{ background: #0A66C2; }}
        .contact-social .ic-x    {{ background: #000000; }}
        .contact-social .ic-gh   {{ background: #24292E; }}
        .contact-social .ic-wk   {{ background: #202122; }}
        .contact-social .ic-fb   {{ background: #1877F2; }}
        .contact-social .ic-ig   {{ background: linear-gradient(45deg, #F58529, #DD2A7B, #8134AF); }}
        </style>
        <div class="contact-social">
            <a href="https://scholar.google.com/citations?hl=en&user=qyth_0QAAAAJ" target="_blank" rel="noopener noreferrer">
                <span class="icon-wrap ic-sch">{_sch_svg}</span>Google Scholar
            </a>
            <a href="https://in.linkedin.com/in/ganeshbagler" target="_blank" rel="noopener noreferrer">
                <span class="icon-wrap ic-li">{_li_svg}</span>LinkedIn
            </a>
            <a href="https://x.com/gansbags" target="_blank" rel="noopener noreferrer">
                <span class="icon-wrap ic-x">{_x_svg}</span>X (Twitter)
            </a>
            <a href="https://github.com/cosylabiiit" target="_blank" rel="noopener noreferrer">
                <span class="icon-wrap ic-gh">{_gh_svg}</span>GitHub
            </a>
            <a href="https://en.wikipedia.org/wiki/Ganesh_Bagler" target="_blank" rel="noopener noreferrer">
                <span class="icon-wrap ic-wk">{_wk_svg}</span>Wikipedia
            </a>
            <a href="https://www.facebook.com/ganesh.bagler" target="_blank" rel="noopener noreferrer">
                <span class="icon-wrap ic-fb">{_fb_svg}</span>Facebook
            </a>
            <a href="https://www.instagram.com/gansbags/?hl=bg" target="_blank" rel="noopener noreferrer">
                <span class="icon-wrap ic-ig">{_ig_svg}</span>Instagram
            </a>
        </div>
        """, unsafe_allow_html=True)

# ==========================================================
# PAGE: ABOUT
# ==========================================================
elif page == "💡 About":
    st.markdown("## ℹ️ About SpiceMagic")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🎯 Project Overview

        **SpiceMagic** is an AI-powered database of spice-disease associations mined from
        PubMed biomedical literature. Using **PubMedBERT** (a biomedical transformer model),
        we've classified 228,750 spice-disease relationships into positive (beneficial),
        negative (harmful), or neutral (no clear effect).

        ### 🧪 Methodology

        - **Data Source:** 250,255 spice-disease associations from PubMed abstracts
        - **Model:** PubMedBERT (fine-tuned on biomedical text)
        - **Training:** 4,500 manually annotated examples
        - **Evaluation:** 90.00% accuracy (5-seed ensemble)
        - **Confidence Filter:** Only associations with ≥75% model confidence

        ### 🏆 Performance

        | Metric | Value |
        |--------|-------|
        | Accuracy | **90.00%** |
        | F1 Score | 89.95% |
        | MCC | 0.818 |
        | Sensitivity | 90.00% |
        | Specificity | 94.30% |

        ### 📊 Database Stats

        - **228,750** high-confidence labeled associations
        - **164** unique spices covered
        - **37,871** unique diseases
        - **47,858** unique PubMed papers
        - **Mean confidence:** 98.0%

        ### 🚀 Technology Stack

        - **Frontend:** Streamlit + Plotly
        - **Backend:** PubMedBERT (Hugging Face Transformers)
        - **Language:** Python 3
        - **ML Framework:** PyTorch
        """)

    with col2:
        st.markdown("### 📈 Quick Stats")
        st.info(f"""
        **Total Associations:** {len(df):,}

        **Positive:** {(df['label']=='positive').sum():,} ({(df['label']=='positive').mean()*100:.1f}%)

        **Neutral:** {(df['label']=='neutral').sum():,} ({(df['label']=='neutral').mean()*100:.1f}%)

        **Negative:** {(df['label']=='negative').sum():,} ({(df['label']=='negative').mean()*100:.1f}%)
        """)

        st.markdown("### ✨ Features")
        st.success("""
        - 🧠 Live AI Prediction (Quick Search)
        - 🔎 Search & Explore with autocomplete
        - 🌶️ Spice Profiles
        - 🧬 Disease Profiles
        - ⚖️ Compare (Spice vs Spice)
        - 🌊 Sankey Flow diagram
        - 📈 Analytics dashboard
        - 💬 AI Chat (natural language queries)
        - ✉️ Contact form
        - 🔗 PubMed integration
        - 🌙 Dark mode
        """)

        st.markdown("### 🔗 Related Work")
        st.markdown("""
        - SpiceRx CNN (2018): 86.70%
        """)

    st.markdown("---")

    # ============================
    # About the Author section
    # ============================
    st.markdown("### 👤 About the Author")
    _bagler_b64 = img_to_base64('bagler_photo.jpg')
    if _bagler_b64:
        _author_img = f'<img src="{_bagler_b64}" alt="Prof. Ganesh Bagler"/>'
    else:
        _author_img = '<img src="https://cosylab.iiitd.edu.in/assets/images/team/bagler.jpg" alt="Prof. Ganesh Bagler"/>'

    # Theme-aware colors (matches app's session_state theme, not OS preference)
    _is_dark = st.session_state.get('theme', 'Light') == 'Dark'
    _bg_grad   = 'linear-gradient(135deg, #1E293B 0%, #0F172A 100%)' if _is_dark else 'linear-gradient(135deg, #F8FAFC 0%, #EEF2F7 100%)'
    _border    = '#475569' if _is_dark else '#E5E7EB'
    _img_ring  = '#334155' if _is_dark else '#FFFFFF'
    _name_col  = '#F1F5F9' if _is_dark else '#1E293B'
    _title_col = '#CBD5E1' if _is_dark else '#4A5568'
    _inst_col  = '#94A3B8' if _is_dark else '#64748B'
    _bio_col   = '#E2E8F0' if _is_dark else '#334155'

    st.markdown(f"""
    <style>
    .author-card {{
        display: flex;
        gap: 24px;
        align-items: center;
        background: {_bg_grad};
        border: 1px solid {_border};
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 12px 0 20px 0;
    }}
    .author-card img {{
        width: 130px;
        height: 130px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid {_img_ring};
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        flex-shrink: 0;
    }}
    .author-info {{ flex: 1; min-width: 0; }}
    .author-name  {{ font-size: 1.4rem; font-weight: 700; color: {_name_col} !important; margin: 0 0 4px 0; }}
    .author-title {{ color: {_title_col} !important; font-size: 0.95rem; margin: 0 0 6px 0; }}
    .author-inst  {{ color: {_inst_col} !important; font-size: 0.9rem; margin: 0 0 12px 0; font-style: italic; }}
    .author-bio   {{ color: {_bio_col} !important; font-size: 0.95rem; line-height: 1.55; margin: 0; }}
    .author-links {{
        margin-top: 14px;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }}
    .author-links a {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        color: #FFFFFF !important;
        text-decoration: none !important;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }}
    .author-links a:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.3);
    }}
    .author-links .al-sch {{ background: #4285F4; }}
    .author-links .al-li  {{ background: #0A66C2; }}
    .author-links .al-x   {{ background: #000000; }}
    .author-links .al-gh  {{ background: #24292E; }}
    .author-links .al-wk  {{ background: #202122; }}
    .author-links .al-web {{ background: linear-gradient(135deg, #4A90E2, #F5A623); }}

    @media (max-width: 640px) {{
        .author-card {{ flex-direction: column; text-align: center; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="author-card">
        {_author_img}
        <div class="author-info">
            <p class="author-name">Prof. Ganesh Bagler</p>
            <p class="author-title">Associate Professor · Head, Complex Systems Laboratory (CoSyLab)</p>
            <p class="author-inst">Indraprastha Institute of Information Technology, Delhi (IIIT-Delhi)</p>
            <p class="author-bio">
                Prof. Bagler's research combines computational biology, data science, and food computing.
                His lab pioneered computational gastronomy — bringing quantitative rigor to food, flavor,
                and health. He has published in Nature Scientific Reports, PLOS ONE, and Bioinformatics,
                and his work on spice-disease associations (SpiceRx) is the foundation of this project.
            </p>
            <div class="author-links">
                <a href="https://scholar.google.com/citations?hl=en&user=qyth_0QAAAAJ" target="_blank" rel="noopener noreferrer" class="al-sch" title="Google Scholar">{_sch_svg}</a>
                <a href="https://in.linkedin.com/in/ganeshbagler" target="_blank" rel="noopener noreferrer" class="al-li" title="LinkedIn">{_li_svg}</a>
                <a href="https://x.com/gansbags" target="_blank" rel="noopener noreferrer" class="al-x" title="X (Twitter)">{_x_svg}</a>
                <a href="https://github.com/cosylabiiit" target="_blank" rel="noopener noreferrer" class="al-gh" title="GitHub">{_gh_svg}</a>
                <a href="https://en.wikipedia.org/wiki/Ganesh_Bagler" target="_blank" rel="noopener noreferrer" class="al-wk" title="Wikipedia">{_wk_svg}</a>
                <a href="https://cosylab.iiitd.edu.in/" target="_blank" rel="noopener noreferrer" class="al-web" title="CoSyLab Website" style="font-size:18px;">🌐</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# FOOTER (CoSyLab-style)
# ==========================================================
# CSS for footer (added first, separately)
st.markdown("""
<style>
.footer-box {
    background: #2A9D8F;
    padding: 40px 20px 30px 20px;
    color: white;
    text-align: center;
    border-radius: 12px;
    margin-top: 40px;
    margin-bottom: -20px;
}
.footer-box h2 {
    color: white;
    font-size: 2rem;
    letter-spacing: 3px;
    margin: 0 0 15px 0;
}
.footer-box p {
    color: white;
    margin: 10px 0;
}
.footer-box a {
    color: white;
    text-decoration: underline;
}
.footer-cc-badge {
    display: inline-block;
    background: white;
    color: black;
    padding: 6px 12px;
    border-radius: 5px;
    font-family: monospace;
    margin: 10px 0;
    font-size: 0.9rem;
    font-weight: bold;
}
.footer-disc {
    max-width: 900px;
    margin: 20px auto;
    font-size: 0.9rem;
    line-height: 1.6;
}
.footer-inst {
    font-weight: bold;
    font-size: 1.05rem;
    margin-top: 20px;
}
.footer-icons {
    margin-top: 20px;
    display: flex;
    justify-content: center;
    gap: 14px;
}
.footer-icons a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    background: #1A1A2E;
    color: #FFFFFF !important;
    border-radius: 50%;
    text-decoration: none !important;
    transition: all 0.25s ease;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}
.footer-icons a:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.3);
}
.footer-icons a svg {
    width: 20px;
    height: 20px;
}
.footer-icons .fi-fb:hover { background: #1877F2; }
.footer-icons .fi-x:hover  { background: #000000; }
.footer-icons .fi-li:hover { background: #0A66C2; }
.footer-icons .fi-ig:hover { background: linear-gradient(45deg, #F58529 0%, #DD2A7B 50%, #8134AF 100%); }
</style>
""", unsafe_allow_html=True)

# Try to load the local IIIT-D logo
iiitd_logo_b64 = img_to_base64('iiitd_logo.png')

if iiitd_logo_b64:
    IIITD_LOGO_HTML = f'<img src="{iiitd_logo_b64}" alt="IIIT-Delhi" style="height: 70px; border-radius: 4px;">'
else:
    # Fallback: CSS version
    IIITD_LOGO_HTML = '''
    <div style="display: inline-flex; align-items: center; gap: 4px; padding: 10px 15px; background: transparent;">
        <span style="display: inline-block; width: 8px; height: 45px; background: #A0AEC0;"></span>
        <span style="display: inline-block; width: 8px; height: 45px; background: #A0AEC0;"></span>
        <span style="display: inline-block; width: 8px; height: 45px; background: #A0AEC0;"></span>
        <span style="display: inline-block; width: 20px; height: 45px; border: 4px solid #4FB39F; border-left: none; border-radius: 0 25px 25px 0; margin-left: 2px;"></span>
    </div>
    '''

# Try to load local Creative Commons badge
cc_badge_b64 = img_to_base64('cc_badge.png')

# Footer HTML content with real image assets
footer_html = f"""
<div class="footer-box">
    <div style="margin-bottom: 20px;">
        {IIITD_LOGO_HTML}
    </div>
    <p style="font-size: 1.1rem;">Copyright © 2026 &nbsp;&nbsp; All rights reserved.</p>
    <div style="margin: 15px 0;">
        <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/" target="_blank" rel="noopener noreferrer">
            <img src="{cc_badge_b64 if cc_badge_b64 else 'https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png'}" alt="Creative Commons BY-NC-SA" style="height: 40px; border-radius: 4px;">
        </a>
    </div>
    <p>
        This work is licensed under a
        <b><a href="https://creativecommons.org/licenses/by-nc-sa/3.0/" target="_blank" rel="noopener noreferrer">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a></b>.
    </p>
    <p class="footer-disc">
        All material on this website is a product of research and is provided for your information only and may not be construed as medical advice or instruction.<br>
        No action or inaction should be taken based solely on the contents of this information; instead, readers should consult appropriate health professionals on any matter relating to their health and well-being.
    </p>
    <p class="footer-inst">
        Indraprastha Institute of Information Technology Delhi (IIIT-Delhi) | Dr. Ganesh Bagler
    </p>
    <div class="footer-icons">
        <a href="https://www.facebook.com/ganesh.bagler" target="_blank" rel="noopener noreferrer" class="fi-fb" title="Facebook">{_fb_svg}</a>
        <a href="https://x.com/gansbags" target="_blank" rel="noopener noreferrer" class="fi-x" title="X (Twitter)">{_x_svg}</a>
        <a href="https://in.linkedin.com/in/ganeshbagler" target="_blank" rel="noopener noreferrer" class="fi-li" title="LinkedIn">{_li_svg}</a>
        <a href="https://www.instagram.com/gansbags/?hl=bg" target="_blank" rel="noopener noreferrer" class="fi-ig" title="Instagram">{_ig_svg}</a>
    </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
