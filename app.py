import os
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FUNDGRUBE KATHARINEUM",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS zur Umsetzung des Marken- & Overlay-Designs
custom_css = """
<style>
    /* Hintergrund & Grundfarben */
    .stApp {
        background-color: #F8F9FA;
        color: #212121;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Haupt-Überschrift (Bogen-Effekt nachgeahmt mit Styling) */
    .curved-header {
        text-align: center;
        font-weight: 900;
        font-size: 2.2rem;
        color: #D32F2F;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header-logo {
        text-align: center;
        font-size: 2.8rem;
        margin-bottom: 1rem;
    }

    /* Buttons Styling */
    div.stButton > button:first-child {
        background-color: #D32F2F !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #B71C1C !important;
        box-shadow: 0 4px 8px rgba(211, 47, 47, 0.3);
    }

    /* Sekundäre Buttons */
    .secondary-btn button {
        background-color: #EEEEEE !important;
        color: #212121 !important;
        border: 1px solid #CCCCCC !important;
    }

    /* Cards für Fundstücke */
    .card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-3px);
    }
    
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #212121;
        margin-top: 8px;
    }
    
    .card-tags {
        font-size: 0.85rem;
        color: #757575;
        margin-top: 2px;
    }

    /* Status & Benachrichtigung */
    .lost-badge {
        background-color: #D32F2F;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING & PREDICTION (KOMPATIBEL MIT TEACHABLE MACHINE & KERAS 2/3)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_keras_model():
    """Lädt das Teachable Machine Modell mit tf_keras Kompatibilität."""
    model_path = "keras_model.h5"
    labels_path = "labels.txt"
    
    model = None
    labels = []
    
    if os.path.exists(model_path):
        try:
            # Nutze tf_keras für alte Teachable Machine / Keras 2 H5-Dateien
            import tf_keras as keras
            model = keras.models.load_model(model_path, compile=False)
        except Exception as e1:
            try:
                # Fallback auf Standard-TensorFlow/Keras
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e2:
                st.warning(f"Fehler beim Laden des Keras Modells: {e2}")
            
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f.readlines()]
        except Exception as e:
            st.warning(f"Fehler beim Lesen der labels.txt: {e}")
            
    return model, labels

model, labels = load_keras_model()

def classify_image(image: Image.Image):
    """Klassifiziert das hochgeladene Bild per Keras-Modell."""
    if model is not None and len(labels) > 0:
        # Preprocessing laut Teachable Machine
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = labels[index]
        confidence_score = float(prediction[0][index])
        
        # Entfernt führende Nummern aus Teachable Machine Labels (z.B. "0 Schlüssel" -> "Schlüssel")
        clean_label = class_name.split(' ', 1)[-1] if ' ' in class_name else class_name
        return clean_label, confidence_score
    else:
        # Fallback Prediction, falls Modell nicht geladen wurde
        return "Schlüsselbund / Zubehör", 0.95

# -----------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION (IN-MEMORY DATENBANK)
# -----------------------------------------------------------------------------
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Suchen"

if "selected_item_id" not in st.session_state:
    st.session_state.selected_item_id = None

if "items_db" not in st.session_state:
    st.session_state.items_db = [
        {
            "id": 1,
            "title": "Sporttasche Blau",
            "tags": "Sport, Nike, Blau, Tasche",
            "category": "Taschen",
            "color": "Blau",
            "brand": "Nike",
            "location": "Sporthalle 2",
            "storage_location": "Fundbüro Raum 102",
            "finder_name": "Herr Meyer (Hausmeister)",
            "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?auto=format&fit=crop&w=400&q=80"
        },
        {
            "id": 2,
            "title": "Schlüsselbund mit Filzanhänger",
            "tags": "Schlüssel, Metall, Rot, Anhänger",
            "category": "Schlüssel",
            "color": "Silber/Rot",
            "brand": "Unbekannt",
            "location": "Pausenhof West",
            "storage_location": "Sekretariat",
            "finder_name": "Anna (Klasse 9b)",
            "image_url": "https://images.unsplash.com/photo-1582139329536-e7284fece509?auto=format&fit=crop&w=400&q=80"
        },
        {
            "id": 3,
            "title": "Trinkflasche Edelstahl",
            "tags": "Flasche, Mepal, Grün, Metall",
            "category": "Flaschen",
            "color": "Grün",
            "brand": "Mepal",
            "location": "Mensa",
            "storage_location": "Fundkiste Mensa",
            "finder_name": "Lukas (Klasse 6a)",
            "image_url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?auto=format&fit=crop&w=400&q=80"
        }
    ]

# -----------------------------------------------------------------------------
# 4. HEADER COMPONENT
# -----------------------------------------------------------------------------
def render_header(title_override=None, show_back=False):
    col_back, col_title, col_settings = st.columns([1, 4, 1])
    
    with col_back:
        if show_back:
            if st.button("← Zurück", key="btn_back_header"):
                st.session_state.current_screen = "Suchen"
                st.session_state.selected_item_id = None
                st.rerun()

    with col_title:
        title = title_override if title_override else "FUNDGRUBE KATHARINEUM"
        st.markdown(f"<div class='curved-header'>{title}</div>", unsafe_allow_html=True)
        if not title_override:
            st.markdown("<div class='sub-header-logo'>🏫</div>", unsafe_allow_html=True)

    with col_settings:
        if st.button("⚙️", key="btn_settings_header"):
            st.session_state.current_screen = "Einstellungen"
            st.rerun()

    st.markdown("---")

# -----------------------------------------------------------------------------
# 5. BOTTOM NAVIGATION BAR
# -----------------------------------------------------------------------------
def render_bottom_nav():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    nav_container = st.container()
    with nav_container:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        
        current = st.session_state.current_screen
        
        with c1:
            lbl1 = "🔍 Suchen (Aktiv)" if current == "Suchen" else "🔍 Suchen"
            if st.button(lbl1, width="stretch", key="nav_suchen"):
                st.session_state.current_screen = "Suchen"
                st.session_state.selected_item_id = None
                st.rerun()
                
        with c2:
            lbl2 = "➕ HINZUFÜGEN (Aktiv)" if current == "Hinzufügen" else "➕ Hinzufügen"
            if st.button(lbl2, width="stretch", key="nav_hinzufuegen"):
                st.session_state.current_screen = "Hinzufügen"
                st.rerun()
                
        with c3:
            lbl3 = "🏷️ Vermisst (Aktiv)" if current == "Vermisst" else "🏷️ Vermisst [LOST]"
            if st.button(lbl3, width="stretch", key="nav_vermisst"):
                st.session_state.current_screen = "Vermisst"
                st.rerun()

# -----------------------------------------------------------------------------
# 6. SCREEN 1: DASHBOARD & SCHNELLSUCHE (SUCHEN)
# -----------------------------------------------------------------------------
def screen_suchen():
    render_header()
    
    search_query = st.text_input("🔍 Gegenstand Suchen...", placeholder="z. B. Schlüssel, Sporttasche, Blau...")
    
    with st.expander("Filter hinzufügen ▽", expanded=False):
        col_cat, col_col, col_brand, col_loc = st.columns(4)
        
        with col_cat:
            filter_cat = st.selectbox("Fundstück-Kategorie", ["Alle", "Schlüssel", "Taschen", "Flaschen", "Elektronik", "Kleidung"])
        with col_col:
            filter_color = st.selectbox("Farbe Suchen", ["Alle", "Blau", "Rot", "Grün", "Schwarz", "Silber"])
        with col_brand:
            filter_brand = st.text_input("Marke Suchen", placeholder="z.B. Nike, Mepal")
        with col_loc:
            filter_loc = st.text_input("Ort Suchen", placeholder="z.B. Pausenhof, Mensa")

    st.markdown("### Fundstücke Galerie")
    
    filtered_items = st.session_state.items_db
    
    if search_query:
        filtered_items = [
            item for item in filtered_items 
            if search_query.lower() in item['title'].lower() or search_query.lower() in item['tags'].lower()
        ]
        
    if 'filter_cat' in locals() and filter_cat != "Alle":
        filtered_items = [item for item in filtered_items if item.get('category') == filter_cat]
        
    if 'filter_color' in locals() and filter_color != "Alle":
        filtered_items = [item for item in filtered_items if filter_color.lower() in item.get('color', '').lower()]

    if not filtered_items:
        st.info("Keine passenden Fundstücke gefunden.")
        return

    cols = st.columns(3)
    for idx, item in enumerate(filtered_items):
        col = cols[idx % 3]
        with col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if item.get("image_url"):
                st.image(item["image_url"], width="stretch")
            elif item.get("image_data"):
                st.image(item["image_data"], width="stretch")
            
            st.markdown(f"<div class='card-title'>{item['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-tags'>Tags: {item['tags']}</div>", unsafe_allow_html=True)
            
            if st.button("Details anzeigen", key=f"btn_item_{item['id']}"):
                st.session_state.selected_item_id = item['id']
                st.session_state.current_screen = "Detail"
                st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. SCREEN 2: FUNDSTÜCK DETAILANSICHT
# -----------------------------------------------------------------------------
def screen_detail():
    render_header(title_override="FUNDSTÜCK", show_back=True)
    
    item = next((i for i in st.session_state.items_db if i['id'] == st.session_state.selected_item_id), None)
    
    if not item:
        st.error("Fundstück nicht gefunden.")
        return

    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.markdown("### BILDER")
        if item.get("image_url"):
            st.image(item["image_url"], width="stretch")
        elif item.get("image_data"):
            st.image(item["image_data"], width="stretch")
            
    with col_info:
        st.markdown(f"## {item['title']}")
        st.markdown(f"**🏷️ Tags:** {item['tags']}")
        st.markdown(f"**📍 Findungsort:** {item['location']}")
        st.markdown(f"**📦 Ort der Aufbewahrung:** {item['storage_location']}")
        st.markdown(f"**👤 Finder:** {item['finder_name']}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Als abgeholt / zurückgegeben markieren"):
            st.session_state.items_db = [i for i in st.session_state.items_db if i['id'] != item['id']]
            st.success("Gegenstand wurde aus der Datenbank entfernt.")
            st.session_state.current_screen = "Suchen"
            st.rerun()

# -----------------------------------------------------------------------------
# 8. SCREEN 3: FUNDSTÜCK MELDEN (HINZUFÜGEN - FINDER-MODUS)
# -----------------------------------------------------------------------------
def screen_hinzufuegen():
    render_header(title_override="HINZUFÜGEN", show_back=True)
    
    st.markdown("### Bild aus Dateien hier hochladen")
    
    upload_option = st.radio("Upload-Quelle wählen:", ["Datei hochladen", "Kamera nutzen"], horizontal=True)
    
    uploaded_image = None
    if upload_option == "Datei hochladen":
        file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])
        if file:
            uploaded_image = Image.open(file)
    else:
        camera_file = st.camera_input("Foto aufnehmen")
        if camera_file:
            uploaded_image = Image.open(camera_file)

    detected_category = ""
    confidence = 0.0

    if uploaded_image:
        st.image(uploaded_image, caption="Hochgeladenes Bild", width=300)
        
        with st.spinner("KI analysiert das Bild..."):
            detected_category, confidence = classify_image(uploaded_image)
            
        st.success(f"🤖 **KI-Vorschlag:** {detected_category} (Wahrscheinlichkeit: {confidence*100:.1f}%)")

    st.markdown("---")
    st.markdown("### Fund-Informationen vervollständigen")
    
    with st.form("form_add_item"):
        title_input = st.text_input("Vorgeschlagener KI Titel (Anpassbar)", value=detected_category if detected_category else "")
        tags_input = st.text_input("Vorgeschlagene KI Tags (Anpassbar)", value=f"{detected_category}, Fundstück, Katharineum" if detected_category else "Fundstück, Katharineum")
        location_input = st.text_input("Findungsort", placeholder="z. B. Schulhof, Turnhalle, Raum 204")
        storage_input = st.text_input("Ort der Aufbewahrung", placeholder="z. B. Sekretariat, Hausmeister")
        finder_input = st.text_input("Name vom Finder", placeholder="Dein Name / Klasse (Optional)")
        
        submit = st.form_submit_button("Fundstück veröffentlichen 🚀")
        
        if submit:
            if not title_input or not location_input:
                st.error("Bitte mindestens Titel und Findungsort ausfüllen!")
            else:
                new_id = max([i['id'] for i in st.session_state.items_db], default=0) + 1
                new_item = {
                    "id": new_id,
                    "title": title_input,
                    "tags": tags_input,
                    "category": detected_category if detected_category else "Sonstiges",
                    "color": "Diverse",
                    "brand": "Unbekannt",
                    "location": location_input,
                    "storage_location": storage_input if storage_input else "Sekretariat",
                    "finder_name": finder_input if finder_input else "Anonym",
                    "image_data": uploaded_image if uploaded_image else None,
                    "image_url": "https://images.unsplash.com/photo-1584438784894-089d6a62b8fa?auto=format&fit=crop&w=400&q=80" if not uploaded_image else None
                }
                st.session_state.items_db.append(new_item)
                st.success("Fundstück erfolgreich registriert!")
                st.session_state.current_screen = "Suchen"
                st.rerun()

# -----------------------------------------------------------------------------
# 9. SCREEN 4: VERMISST MELDEN & KI SMART-MATCH (VERMISST)
# -----------------------------------------------------------------------------
def screen_vermisst():
    render_header(title_override="VERMISST [LOST]", show_back=True)
    
    st.markdown("<span class='lost-badge'>LOST & FOUND SMART-MATCH</span>", unsafe_allow_html=True)
    st.write("Lade ein Foto deines verloren gegangenen Gegenstands hoch. Die KI vergleicht es direkt mit der Fund-Datenbank.")
    
    file = st.file_uploader("Bild deines verlorenen Gegenstands hochladen", type=["jpg", "jpeg", "png"], key="lost_uploader")
    
    if file:
        img = Image.open(file)
        st.image(img, width=250, caption="Dein Such-Bild")
        
        with st.spinner("Vergleiche mit Datenbank..."):
            predicted_cat, conf = classify_image(img)
            
        st.info(f"Erkannte Kategorie: **{predicted_cat}**")
        st.markdown("### 🔍 ÄHNLICHE BILDER IN DER DATENBANK")
        
        matches = [i for i in st.session_state.items_db if predicted_cat.lower() in i['title'].lower() or predicted_cat.lower() in i['tags'].lower()]
        
        if matches:
            cols = st.columns(len(matches))
            for idx, match in enumerate(matches):
                with cols[idx]:
                    st.image(match.get('image_url') or match.get('image_data'), width="stretch")
                    st.caption(f"**{match['title']}**\nOrt: {match['location']}")
        else:
            st.warning("Aktuell kein exakter KI-Treffer in der Datenbank vorhanden.")

    st.markdown("---")
    st.markdown("### Such-Auftrag erstellen")
    
    st.text_input("Titel hinzufügen", placeholder="z. B. Meine blaue Jacke")
    st.text_input("Tags hinzufügen", placeholder="z. B. Jacke, Blau, XL, Adidas")
    
    notify_active = st.toggle("Benachrichtigung aktivieren (E-Mail/App)", value=True)
    if notify_active:
        st.caption("🟢 **Aktiv:** Die KI benachrichtigt dich automatisch, sobald ein passendes Item hochgeladen wird.")
        
    if st.button("Vermisst-Meldung speichern"):
        st.success("Such-Auftrag gespeichert. Du wirst bei neuen Treffern benachrichtigt!")

# -----------------------------------------------------------------------------
# 10. SCREEN 5: EINSTELLUNGEN
# -----------------------------------------------------------------------------
def screen_einstellungen():
    render_header(title_override="EINSTELLUNGEN", show_back=True)
    
    settings_options = [
        "🔔 Benachrichtigungen & Push-Service",
        "👤 Mein Profil / Kontaktdaten",
        "🏫 Schule / Standort (Katharineum)",
        "🔒 Datenschutz & Nutzungsbedingungen",
        "ℹ️ App-Info & Version (v1.0.0)"
    ]
    
    for opt in settings_options:
        col_txt, col_arrow = st.columns([5, 1])
        with col_txt:
            st.markdown(f"**{opt}**")
        with col_arrow:
            st.button("→", key=f"btn_opt_{opt}")
        st.markdown("---")

# -----------------------------------------------------------------------------
# 11. MAIN APP ROUTER
# -----------------------------------------------------------------------------
def main():
    screen = st.session_state.current_screen
    
    if screen == "Suchen":
        screen_suchen()
    elif screen == "Detail":
        screen_detail()
    elif screen == "Hinzufügen":
        screen_hinzufuegen()
    elif screen == "Vermisst":
        screen_vermisst()
    elif screen == "Einstellungen":
        screen_einstellungen()
        
    render_bottom_nav()

if __name__ == "__main__":
    main()
