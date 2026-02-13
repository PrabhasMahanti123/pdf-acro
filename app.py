import streamlit as st
import fitz
import tempfile
import os
from converter_core import PDFConverter

st.set_page_config(page_title="PDF to AcroForm Converter", layout="wide")

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .stApp { color: #e0e0e0; }
    h1 { color: #00d2ff !important; }
    .stFileUploader { border: 2px dashed #00d2ff; border-radius: 10px; padding: 10px; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        color: white;
        font-weight: bold;
        border: none;
        font-size: 1.1em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,210,255,0.4);
    }
    .success-box {
        background: rgba(0,210,255,0.1);
        border: 1px solid #00d2ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 PDF to AcroForm Converter")
st.markdown("**Convert any static PDF form into an editable AcroForm.** Supports text-based forms (with labels/underscores) and vector-only forms automatically.")

uploaded_file = st.file_uploader("Upload a static PDF template", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_path = tempfile.mktemp(suffix="_editable.pdf")

    if st.button("🔄 Convert to Editable PDF"):
        with st.spinner("Analyzing form structure and detecting fields..."):
            try:
                converter = PDFConverter(input_path)
                field_count = converter.convert(output_path)
                
                with open(output_path, "rb") as f:
                    pdf_bytes = f.read()
                
                st.markdown(f"""
                <div class="success-box">
                    <h3 style="color: #00d2ff; margin:0;">✅ Conversion Successful!</h3>
                    <p style="margin: 8px 0 0 0;">Detected and created <strong>{field_count}</strong> editable fields across all pages.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.download_button(
                    label=f"⬇️ Download Editable PDF ({field_count} fields)",
                    data=pdf_bytes,
                    file_name=f"editable_{uploaded_file.name}",
                    mime="application/pdf"
                )
                
                st.info("💡 **Tip:** Open the downloaded PDF in Adobe Acrobat or a browser. Fill in the fields, then **Save** to keep your data.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(input_path):
                    os.remove(input_path)
else:
    st.info("👆 Upload a PDF form template to begin.")
