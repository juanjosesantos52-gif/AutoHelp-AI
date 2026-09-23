import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configura tu API Key de Gemini como un Secret de Streamlit Cloud (o ponla temporalmente aquí para pruebas)
# En Streamlit Cloud lo pondrás en la sección "Secrets" como GOOGLE_API_KEY
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(
    page_title="AutoPartes AI - Buscador Inteligente",
    page_icon="🚗",
    layout="wide"
)

# ... (Tu barra lateral y diseño se mantienen igual) ...

DIRECTORIO_DB = "./chroma_db_repuestos"

@st.cache_resource
def inicializar_or_cargar_rag():
    catalogo_default = "catalogo_repuestos.txt"
    
    if not os.path.exists(catalogo_default):
        with open(catalogo_default, "w", encoding="utf-8") as f:
            f.write(
                "Catálogo Técnico de Repuestos y Equivalencias Automotrices.\n"
                "Vehículo: Mazda 3 (Segunda Generación, 2010-2013).\n"
                "- Pastillas de freno delanteras OEM: Código MZ-301FR. Equivalencia Bosch: BP-450.\n"
                "Vehículo: Ford Mustang (S550, 2015-2023).\n"
                "- Pastillas de freno delanteras OEM (GT / EcoBoost): Código FR-BRK-15. Equivalencia Brembo: P-59-088.\n"
            )

    loader = TextLoader(catalogo_default, encoding="utf-8")
    documentos = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = text_splitter.split_documents(documentos)

    # Usamos embeddings y LLM en la nube de Google (Gratis y 24/7)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DIRECTORIO_DB
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    system_prompt = (
        "Eres un experto asesor de repuestos automotrices, mecánico en jefe e historiador de vehículos.\n"
        "Revisa los fragmentos de contexto del catálogo local. Si la respuesta está ahí, úsala. "
        "Si no, utiliza tu conocimiento general sobre la industria automotriz mundial para responder con precisión.\n\n"
        "Contexto del Catálogo Local:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Función inteligente para asociar la imagen correcta según el vehículo y la pieza mencionada
def obtener_imagen_repuesto(pregunta: str):
    p = pregunta.lower()
    
    # Detección específica por vehículo y componente
    if "mustang" in p:
        if "freno" in p or "pastillas" in p or "disco" in p:
            return "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?auto=format&fit=crop&w=600&q=80", "Sistema de Frenos de Alto Rendimiento - Ford Mustang"
        else:
            return "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?auto=format&fit=crop&w=600&q=80", "Componente Mecánico - Ford Mustang"
            
    elif "mazda" in p:
        return "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=600&q=80", "Repuesto Específico - Mazda"
        
    elif "toyota" in p or "corolla" in p:
        return "https://images.unsplash.com/photo-1590362891991-f776e747a588?auto=format&fit=crop&w=600&q=80", "Repuesto Específico - Toyota"
        
    # Filtrado genérico por tipo de pieza si no especifica marca exacta
    if "freno" in p or "pastillas" in p or "disco" in p:
        return "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=600&q=80", "Sistema de Frenos / Pastillas"
    elif "amortiguador" in p or "suspension" in p:
        return "https://images.unsplash.com/photo-1486006920555-c77dce18193b?auto=format&fit=crop&w=600&q=80", "Amortiguador y Suspensión"
    elif "bomba" in p or "gasolina" in p or "combustible" in p:
        return "https://images.unsplash.com/photo-1508974239320-0a029497e820?auto=format&fit=crop&w=600&q=80", "Bomba de Combustible"
    else:
        return "https://images.unsplash.com/photo-1489824904134-891ab64532f1?auto=format&fit=crop&w=600&q=80", "Componente Mecánico Automotriz"

with st.spinner("Inicializando motor híbrido..."):
    cadena_rag = inicializar_or_cargar_rag()

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
            resultado = cadena_rag.invoke({"input": pregunta_usuario})
            respuesta = resultado["answer"]
            
            # Ahora la imagen toma en cuenta estrictamente lo que preguntó el usuario
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