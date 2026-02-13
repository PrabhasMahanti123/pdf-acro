import fitz
import re
import os
import io
import numpy as np
from PIL import Image
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

class PDFConverter:
    """Universal PDF to AcroForm converter with Mistral OCR + EasyOCR positioning."""
    
    CHECKBOX_CHARS = ["□", "☐", "☑", "☒", "▢", "◻", "◯", "■"]
    CHECKBOX_ENCODED = ["Γÿé", "Γÿí", "Γûí", "Γûá", "ΓÿÉ"]
    
    def __init__(self, input_path):
        self.input_path = input_path
        self.doc = fitz.open(input_path)
        self._mistral_client = None
        self._ocr_reader = None
    
    def _get_mistral_client(self):
        if self._mistral_client is None:
            self._mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        return self._mistral_client
    
    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._ocr_reader
    
    def _get_page_text_blocks(self, page):
        blocks = page.get_text("dict")["blocks"]
        return [b for b in blocks if b.get("type") == 0]
    
    def _mistral_ocr_pdf(self):
        client = self._get_mistral_client()
        with open(self.input_path, "rb") as f:
            uploaded_file = client.files.upload(
                file={"file_name": os.path.basename(self.input_path), "content": f},
                purpose="ocr"
            )
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
        return client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed_url.url}
        )
    
    def _ocr_page_positions(self, page):
        """Get text positions from EasyOCR."""
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        img_np = np.array(img)
        reader = self._get_ocr_reader()
        results = reader.readtext(img_np)
        
        scale_x = page.rect.width / img.width
        scale_y = page.rect.height / img.height
        
        positions = []
        for bbox, text, conf in results:
            x0 = bbox[0][0] * scale_x
            y0 = bbox[0][1] * scale_y
            x1 = bbox[2][0] * scale_x
            y1 = bbox[2][1] * scale_y
            positions.append({"text": text, "rect": fitz.Rect(x0, y0, x1, y1), "conf": conf})
        return positions
    
    def _extract_form_fields_from_markdown(self, markdown_text):
        """Parse Mistral's markdown to extract every field label and checkbox."""
        fields_info = []
        
        # All checkbox chars to search for
        ALL_CB = self.CHECKBOX_CHARS + self.CHECKBOX_ENCODED
        
        # Split by lines in table cells
        lines = markdown_text.replace("|", "\n").split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("---"):
                continue
            
            # Find checkbox markers (both Unicode and encoded)
            for cb in ALL_CB:
                if cb in line:
                    parts = line.split(cb)
                    for part in parts[1:]:
                        option = part.strip()
                        # Clean: take text up to next checkbox or end of line
                        for other_cb in ALL_CB:
                            if other_cb in option:
                                option = option.split(other_cb)[0].strip()
                        option = re.sub(r'\s+', ' ', option).strip()
                        if option and len(option) > 1 and len(option) < 60:
                            fields_info.append({"type": "checkbox", "label": option[:30]})
            
            # Find "Label:" patterns for text fields
            colon_matches = re.finditer(r'([A-Za-z\s/\'()\[\]#*\d.]+?)\*?\s*:', line)
            for match in colon_matches:
                label = match.group(0).strip()
                # Skip instruction text, URLs, timestamps, very long labels
                skip_keywords = ["NOTE:", "---", "http", "sfhp.org", "1(", "Methods:", 
                                 "Services:", "values.", "below:", "service."]
                if len(label) < 40 and not any(sk in label for sk in skip_keywords):
                    fields_info.append({"type": "text", "label": label})
        
        return fields_info
    
    def _find_label_position(self, label_text, ocr_positions, used_positions):
        """Find the OCR position that best matches a label, with fuzzy matching."""
        label_clean = label_text.rstrip(":*").strip().lower()
        
        if len(label_clean) < 2:
            return None
        
        best_match = None
        best_score = 0
        
        for i, tp in enumerate(ocr_positions):
            if i in used_positions:
                continue
            
            ocr_clean = tp["text"].lower().strip()
            
            # Exact substring match
            if label_clean in ocr_clean or ocr_clean in label_clean:
                score = len(label_clean) / max(len(ocr_clean), 1)
                if score > best_score:
                    best_score = score
                    best_match = (i, tp)
            
            # Word-level match: check if main words match
            label_words = set(label_clean.split())
            ocr_words = set(ocr_clean.split())
            common = label_words & ocr_words
            if len(common) >= 1 and len(label_words) > 0:
                score = len(common) / len(label_words)
                if score > best_score:
                    best_score = score
                    best_match = (i, tp)
        
        if best_match and best_score > 0.3:
            return best_match
        return None
    
    def _detect_fields_ocr(self, page, page_markdown):
        """Use Mistral markdown to determine fields, EasyOCR for positions."""
        fields = []
        page_width = page.rect.width
        
        # Get OCR positions
        ocr_positions = self._ocr_page_positions(page)
        
        # Parse fields from Mistral markdown
        form_fields = self._extract_form_fields_from_markdown(page_markdown)
        
        used_positions = set()
        
        for field_info in form_fields:
            if field_info["type"] == "checkbox":
                # Find the checkbox option text in OCR positions
                match = self._find_label_position(field_info["label"], ocr_positions, used_positions)
                if match:
                    idx, tp = match
                    used_positions.add(idx)
                    # Place checkbox to the left of the option text
                    cb_rect = fitz.Rect(tp["rect"].x0 - 12, tp["rect"].y0, tp["rect"].x0 - 1, tp["rect"].y1)
                    fields.append({"type": "checkbox", "rect": cb_rect})
            
            elif field_info["type"] == "text":
                label = field_info["label"]
                match = self._find_label_position(label, ocr_positions, used_positions)
                if match:
                    idx, tp = match
                    used_positions.add(idx)
                    
                    # Place text field to the right of the label
                    field_x0 = tp["rect"].x1 + 2
                    
                    # Find next OCR text on the same line to limit field width
                    field_x1 = page_width - 20
                    for j, other in enumerate(ocr_positions):
                        if j != idx and abs(other["rect"].y0 - tp["rect"].y0) < 5 and other["rect"].x0 > tp["rect"].x1 + 5:
                            field_x1 = min(field_x1, other["rect"].x0 - 2)
                    
                    field_x1 = max(field_x0 + 20, min(field_x1, field_x0 + 200))
                    
                    if field_x1 - field_x0 > 15:
                        fields.append({
                            "type": "text",
                            "rect": fitz.Rect(field_x0, tp["rect"].y0, field_x1, tp["rect"].y1)
                        })
        
        return fields
    
    # ---- Native text strategies ----
    
    def _find_checkbox_locations(self, page):
        locations = []
        for char in self.CHECKBOX_CHARS + self.CHECKBOX_ENCODED:
            for inst in page.search_for(char):
                locations.append(inst)
        return locations
    
    def _find_underscore_fields(self, page):
        fields = []
        instances = page.search_for("___")
        if not instances:
            return fields
        merged = []
        for inst in sorted(instances, key=lambda r: (r.y0, r.x0)):
            if merged and abs(inst.y0 - merged[-1].y0) < 3 and inst.x0 < merged[-1].x1 + 5:
                merged[-1] = fitz.Rect(
                    min(merged[-1].x0, inst.x0), min(merged[-1].y0, inst.y0),
                    max(merged[-1].x1, inst.x1), max(merged[-1].y1, inst.y1))
            else:
                merged.append(fitz.Rect(inst))
        for m in merged:
            fields.append({"type": "text", "rect": fitz.Rect(m.x0, m.y0, m.x1, m.y1 + 2)})
        return fields
    
    def _find_label_fields(self, page):
        fields = []
        blocks = self._get_page_text_blocks(page)
        pw = page.rect.width
        for block in blocks:
            for line in block["lines"]:
                lt = "".join([s["text"] for s in line["spans"]]).strip()
                for label in re.findall(r'([A-Za-z\s/\'()#*]+)\s*:\s*', lt):
                    for li in page.search_for(label.strip() + ":"):
                        if abs(li.y0 - line["bbox"][1]) < 5:
                            fx0 = li.x1 + 2
                            fx1 = min(fx0 + 150, pw - 20)
                            if fx1 - fx0 > 20:
                                fields.append({"type": "text", "rect": fitz.Rect(fx0, li.y0, fx1, li.y1)})
                            break
        return fields
    
    # ---- Main detection orchestrator ----
    
    def detect_fields(self, page_num, mistral_pages=None):
        page = self.doc[page_num]
        text_blocks = self._get_page_text_blocks(page)
        fields = []
        
        if len(text_blocks) > 0:
            for loc in self._find_checkbox_locations(page):
                fields.append({"type": "checkbox", "rect": loc})
            uf = self._find_underscore_fields(page)
            fields.extend(uf)
            if not uf:
                fields.extend(self._find_label_fields(page))
        elif mistral_pages is not None:
            page_md = ""
            for mp in mistral_pages:
                if mp.index == page_num:
                    page_md = mp.markdown
                    break
            if page_md:
                fields = self._detect_fields_ocr(page, page_md)
        
        # Dedup
        unique = []
        for f in fields:
            if f["rect"].width <= 0 or f["rect"].height <= 0:
                continue
            is_dup = any(f["type"] == u["type"] and f["rect"].intersects(u["rect"]) for u in unique)
            if not is_dup:
                unique.append(f)
        return unique
    
    def convert(self, output_path):
        try:
            self.doc.need_appearances = True
        except:
            pass
        
        # Check if any page needs OCR
        needs_ocr = any(len(self._get_page_text_blocks(self.doc[i])) == 0 for i in range(len(self.doc)))
        mistral_pages = None
        if needs_ocr:
            try:
                ocr_response = self._mistral_ocr_pdf()
                mistral_pages = ocr_response.pages
            except Exception as e:
                print(f"Mistral OCR failed: {e}")
        
        total_fields = 0
        for i in range(len(self.doc)):
            page = self.doc[i]
            fields = self.detect_fields(i, mistral_pages)
            for j, field in enumerate(fields):
                widget = fitz.Widget()
                widget.rect = field["rect"]
                widget.field_name = f"page{i+1}_field{j+1}"
                widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX if field["type"] == "checkbox" else fitz.PDF_WIDGET_TYPE_TEXT
                if field["type"] == "text":
                    widget.field_flags = 0
                page.add_widget(widget)
                total_fields += 1
        
        self.doc.save(output_path, garbage=3, deflate=True)
        self.doc.close()
        return total_fields


if __name__ == "__main__":
    base = r"c:\Users\VH0000540\Downloads\pdf-acro-conv"
    tests = [
        ("Aetna_Arizona Standard Prior Authorisation Request Form for Health Care Services_2025.pdf", "output_aetna.pdf"),
        ("ALS - Agents Radicava_part-b-pa-request-form-2025.pdf", "output_als.pdf"),
        ("San Francisco Health Plan -Pre-Authorization request form.pdf", "output_sfhp.pdf"),
    ]
    for inp, out in tests:
        inp_path = os.path.join(base, inp)
        out_path = os.path.join(base, out)
        if os.path.exists(inp_path):
            print(f"Converting: {inp[:50]}...")
            converter = PDFConverter(inp_path)
            count = converter.convert(out_path)
            print(f"  -> {count} fields detected\n")
        else:
            print(f"Not found: {inp}")
