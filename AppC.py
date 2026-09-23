import os
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuración de la API Key de Google
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(
    page_title="AutoPartes AI - Buscador Inteligente",
    page_icon="🚗",
    layout="wide"
)

# Modelos en orden de preferencia.
# Los gemini-1.5 y gemini-2.0 ya fueron retirados por Google (dan NotFound).
MODELOS_PREFERIDOS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=600&q=80",
             caption="AutoPartes AI - Sistema Cloud",
             use_container_width=True)
    st.markdown("### 🛠️ Panel de Control")
    st.info("Sistema RAG Híbrido Directo + Generador Visual.")
    st.markdown("#### 📂 Catálogos Activos:")
    st.markdown("- Mazda 3 (2010-2013)")
    st.markdown("- Toyota Corolla (2014-2019)")
    st.markdown("- Ford Mustang (2015+)")
    st.divider()
    st.markdown("**Proyecto de Examen:** IA Aplicada 🎓")

col1, col2 = st.columns([4, 1])
with col1:
    st.title("🚗 AutoPartes AI: Buscador Visual de Repuestos")
    st.markdown("Asistente inteligente con recuperación directa y visualización exacta de componentes.")

with col2:
    st.image("https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?auto=format&fit=crop&w=300&q=80",
             use_container_width=True)

DIRECTORIO_DB = "./chroma_db_repuestos"


def elegir_modelo() -> str:
    """Elige el primer modelo preferido que esté realmente disponible para tu API key."""
    try:
        disponibles = {
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        }
    except Exception:
        disponibles = set()

    for nombre in MODELOS_PREFERIDOS:
        if nombre in disponibles:
            return nombre
    return MODELOS_PREFERIDOS[0]


@st.cache_resource
def inicializar_or_cargar_rag():
    catalogo_default = "catalogo_repuestos.txt"

    if not os.path.exists(catalogo_default):
        with open(catalogo_default, "w", encoding="utf-8") as f:
            f.write(
                "Catálogo Técnico de Repuestos y Equivalencias Automotrices.\n"
                "Vehículo: Mazda 3 (Segunda Generación, 2010-2013).\n"
                "- Pastillas de freno delanteras OEM: Código MZ-301FR. Equivalencia Bosch: BP-450.\n"
                "Vehículo: Toyota Corolla (E170, 2014-2019).\n"
                "- Pastillas de freno delanteras OEM: Código TY-C140. Equivalencia Akebono: ACT-1211.\n"
                "Vehículo: Ford Mustang (S550, 2015-2023).\n"
                "- Pastillas de freno delanteras OEM (GT / EcoBoost): Código FR-BRK-15. Equivalencia Brembo Performance: P-59-088.\n"
            )

    loader = TextLoader(catalogo_default, encoding="utf-8")
    documentos = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = text_splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DIRECTORIO_DB
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    nombre_modelo = elegir_modelo()
    model = genai.GenerativeModel(nombre_modelo)
    return retriever, model, nombre_modelo


def generar_respuesta(model, nombre_modelo: str, prompt: str) -> str:
    """Intenta con el modelo elegido y, si no existe, prueba los demás de la lista."""
    candidatos = [nombre_modelo] + [m for m in MODELOS_PREFERIDOS if m != nombre_modelo]
    ultimo_error = None

    for nombre in candidatos:
        try:
            modelo = model if nombre == nombre_modelo else genai.GenerativeModel(nombre)
            return modelo.generate_content(prompt).text
        except Exception as e:
            ultimo_error = e
            continue

    return f"⚠️ No pude generar la respuesta ({type(ultimo_error).__name__}). Intenta de nuevo en unos minutos."


def obtener_imagen_repuesto(pregunta: str):
    p = pregunta.lower()
    if "mustang" in p:
        return "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?auto=format&fit=crop&w=600&q=80", "Sistema de Frenos de Alto Rendimiento - Ford Mustang"
    elif "mazda" in p:
        return "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=600&q=80", "Repuesto Específico - Mazda"
    elif "toyota" in p or "corolla" in p:
        return "https://images.unsplash.com/photo-1590362891991-f776e747a588?auto=format&fit=crop&w=600&q=80", "Repuesto Específico - Toyota"

    if "freno" in p or "pastillas" in p or "disco" in p:
        return "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=600&q=80", "Sistema de Frenos / Pastillas"
    elif "amortiguador" in p or "suspension" in p:
        return "https://images.unsplash.com/photo-1486006920555-c77dce18193b?auto=format&fit=crop&w=600&q=80", "Amortiguador y Suspensión"
    elif "bomba" in p or "gasolina" in p or "combustible" in p:
        return "https://images.unsplash.com/photo-1508974239320-0a029497e820?auto=format&fit=crop&w=600&q=80", "Bomba de Combustible"
    else:
        return "https://images.unsplash.com/photo-1489824904134-891ab64532f1?auto=format&fit=crop&w=600&q=80", "Componente Mecánico Automotriz"


with st.spinner("Inicializando motor híbrido directo..."):
    retriever, model, nombre_modelo = inicializar_or_cargar_rag()

st.sidebar.caption(f"Modelo activo: {nombre_modelo}")

st.divider()
st.subheader("💬 Consulta Interactiva con Soporte Visual")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if "imagen" in mensaje and mensaje["imagen"]:
            st.markdown(f'<img src="{mensaje["imagen"]}" width="380" style="border-radius: 8px; margin-top: 10px;">', unsafe_allow_html=True)
            st.caption(mensaje["caption"])

pregunta_usuario = st.chat_input("Ej: ¿Qué pastillas de freno le sirven al Mustang 2015?")

if pregunta_usuario:
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Procesando consulta y seleccionando imagen del componente..."):
            docs_relacionados = retriever.invoke(pregunta_usuario)
            contexto_texto = "\n\n".join([doc.page_content for doc in docs_relacionados])

            prompt_final = (
                "Eres un experto asesor de repuestos automotrices, mecánico en jefe e historiador de vehículos.\n"
                "Primero, revisa los fragmentos de contexto del catálogo local. Si la respuesta está ahí, úsala. "
                "Si no, utiliza tu conocimiento general sobre la industria automotriz mundial para responder con precisión.\n\n"
                f"Contexto del Catálogo Local:\n{contexto_texto}\n\n"
                f"Pregunta del Usuario: {pregunta_usuario}"
            )

            respuesta = generar_respuesta(model, nombre_modelo, prompt_final)

            url_img, caption_img = obtener_imagen_repuesto(pregunta_usuario)

            st.markdown(respuesta)
            st.markdown(f'<img src="{url_img}" width="380" style="border-radius: 8px; margin-top: 10px;">', unsafe_allow_html=True)
            st.caption(caption_img)

            st.session_state.mensajes.append({
                "rol": "assistant",
                "contenido": respuesta,
                "imagen": url_img,
                "caption": caption_img
            })
