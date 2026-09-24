import os
import threading
import uvicorn
import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image

# Importaciones de FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- CONFIGURACIÓN DE LA API KEY DE GOOGLE ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(
    page_title="AutoPartes AI - Buscador Global Multimarca",
    page_icon="🚗",
    layout="wide"
)

MODELOS_PREFERIDOS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

DIRECTORIO_DB = "./chroma_db_repuestos"

# --- LISTA COMPLETA Y GLOBAL DE MARCAS DEL MUNDO ---
MARCAS_GLOBALES = {
    "🇺🇸 Norteamérica (EE.UU., Canadá, México)": [
        "Acura", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "Eagle", "Ford", 
        "GMC", "Hennessey", "Hummer", "Jeep", "Lincoln", "Lucid Motors", "Mercury", "Oldsmobile", 
        "Panoz", "Plymouth", "Pontiac", "RAM", "Rivian", "Saleen", "Saturn", "Shelby", 
        "SRT", "Tesla", "VLF Automotive"
    ],
    "🇯🇵 Japón": [
        "Abarth (JP Spec)", "Daihatsu", "Datsun", "Honda", "Infiniti", "Isuzu", "Lexus", 
        "Mazda", "Mitsubishi", "Mitsuoka", "Nissan", "Scion", "Subaru", "Suzuki", "Toyota", "Tommykaira"
    ],
    "🇩🇪 Alemania": [
        "Audi", "Alpina", "BMW", "Borgward", "Bitter", "Gumpert / Apollo", "Maybach", 
        "Mercedes-Benz", "Opel", "Porsche", "RUF", "Smart", "Volkswagen", "Wiesmann"
    ],
    "🇮🇹 Italia": [
        "Abarth", "Alfa Romeo", "Autobianchi", "Bizzarrini", "Bugatti (Origen)", "De Tomaso", 
        "Ferrari", "Fiat", "Fornasari", "Innocenti", "ISO Rivolta", "Lamborghini", "Lancia", 
        "Maserati", "Mazzanti", "Pagani", "Pininfarina"
    ],
    "🇬🇧 Reino Unido": [
        "AC Cars", "Aston Martin", "Austin", "Austin-Healey", "Bentley", "BAC (Mono)", 
        "Caterham", "Ginetta", "Jaguar", "Jensen", "Land Rover", "Lister", "Lotus", 
        "Marcos", "McLaren", "MG", "Mini", "Morgan", "Noble", "Radical", "Rolls-Royce", 
        "Rover", "Triumph", "TVR", "Vauxhall", "Westfield"
    ],
    "🇫🇷 Francia": [
        "Alpine", "Bugatti", "Citroën", "DS Automobiles", "Ligier", "Microcar", "Peugeot", "Renault", "Venturi"
    ],
    "🇨🇳 China": [
        "Aiways", "BAIC", "Baojun", "BYD", "Changan", "Chery", "Dongfeng", "FAW", 
        "Foton", "GAC Group", "Geely", "Great Wall (GWM)", "Haval", "Hongqi", "JAC Motors", 
        "Jetour", "Leapmotor", "Li Auto", "Lynk & Co", "NIO", "Omoda", "SAIC Motor", 
        "Voyah", "Wuling", "XPeng", "Zeekr"
    ],
    "🇰🇷 Corea del Sur": [
        "Daewoo", "Genesis", "Hyundai", "KG Mobility (SsangYong)", "Kia", "Renault Korea (Samsung)"
    ],
    "🇪🇸 🇸🇪 🇳🇱 Resto de Europa": [
        "Abarth", "Artega", "Donkervoort", "GTA Spano", "Koenigsegg", "KTM (X-Bow)", 
        "Lada (Rusia)", "Rimac (Croacia)", "SEAT", "Skoda (Rep. Checa)", "Spyker", "Troller", "Volvo", "Zastava"
    ],
    "🚚 Camiones Pesados & Vehículos Comerciales": [
        "Actros (Mercedes)", "Astra", "Avia", "DAF", "DONGFENG Commercial", "ERF", 
        "Faw Jiefang", "Foton Auman", "Freightliner", "Hino", "Howo (Sinotruk)", "Hyundai Commercial", 
        "International / Navistar", "Isuzu Truck", "IVECO", "Kamaz", "Kenworth", 
        "Mack Trucks", "MAN", "Mercedes-Benz Trucks", "Nissan Diesel / UD Trucks", "Peterbilt", 
        "Renault Trucks", "Scania", "Shacman", "Tatra", "Terex", "Volvo Trucks", "Western Star"
    ],
    "🏎️ Hiperautos & Marcas Exclusivas Globales": [
        "Apollo Automobili", "Ariel", "Czinger", "Devel Sixteen", "Drako", "KTM", "Naran", 
        "SSC North America", "Zenvo"
    ]
}

# --- CONFIGURACIÓN DE FASTAPI BACKEND ---
api = FastAPI(
    title="AutoPartes AI Backend API",
    description="API REST para consumo del catálogo global, búsqueda de tiendas y diagnóstico multimodal.",
    version="1.0.0"
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    region: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None

@api.get("/api/health")
def api_health():
    return {"status": "online", "message": "Backend API corriendo junto a Streamlit."}

@api.get("/api/brands")
def api_get_brands():
    return {"categories": MARCAS_GLOBALES}

@api.post("/api/chat")
def api_chat(req: ChatRequest):
    info_vehiculo = f"{req.brand or ''} {req.model or ''} ({req.year or ''})".strip()
    return {
        "reply": f"Solicitud recibida para {info_vehiculo}: {req.message}",
        "vehicle": info_vehiculo,
        "status": "success"
    }

def run_fastapi():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="error")

# Iniciar FastAPI en segundo plano para que no bloquee la interfaz de Streamlit
@st.cache_resource
def start_backend_thread():
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()

start_backend_thread()

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=600&q=80",
             caption="AutoPartes AI - Catálogo Global",
             use_container_width=True)
    st.markdown("### 🛠️ Panel de Control")
    st.info("Sistema RAG Híbrido Multimarca: Cobertura Global de Autos, Lujo, EV y Camiones.")
    st.success("🟢 API REST Backend activa en el puerto 8000 (`/api/health`)")
    
    st.markdown("#### 🚘 Garaje Virtual Global")
    
    region_sel = st.selectbox("1. Selecciona Región o Tipo:", list(MARCAS_GLOBALES.keys()))
    marca_sel = st.selectbox("2. Selecciona la Marca Exacta:", MARCAS_GLOBALES[region_sel])
    
    modelo_vehiculo = st.text_input("3. Modelo (ej. Civic, Mustang, Cascadia, F-150):", "")
    año_vehiculo = st.number_input("4. Año del Vehículo / Camión:", min_value=1950, max_value=2027, value=2020)
    
    st.divider()
    st.markdown("**Proyecto de Examen:** IA Aplicada 🎓")

col1, col2 = st.columns([4, 1])
with col1:
    st.title("🚗🚛 AutoPartes AI: Buscador Universal de Repuestos")
    st.markdown("Asistente inteligente con recuperación técnica para todas las marcas mundiales, comerciales e industriales.")

with col2:
    st.image("https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?auto=format&fit=crop&w=300&q=80",
             use_container_width=True)

def elegir_modelo() -> str:
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
    return "gemini-1.5-flash"

@st.cache_resource
def inicializar_or_cargar_rag():
    catalogo_default = "catalogo_repuestos.txt"

    if not os.path.exists(catalogo_default):
        with open(catalogo_default, "w", encoding="utf-8") as f:
            f.write(
                "Catálogo Técnico Universal de Repuestos y Equivalencias Automotrices e Industriales.\n"
                "Mazda 3 (2010-2013): Pastillas delanteras OEM: MZ-301FR / Bosch: BP-450.\n"
                "Toyota Corolla (2014-2019): Pastillas delanteras OEM: TY-C140 / Akebono: ACT-1211.\n"
                "Ford Mustang (2015+): Pastillas delanteras OEM: FR-BRK-15 / Brembo: P-59-088.\n"
                "Freightliner Cascadia (DD15): Filtro de Aceite OEM: A4721800609 / Donaldson: P550821.\n"
                "BYD Han / Tang: Pastillas de freno Brembo OEM: BYD-BRK-EV9 / Bosch EV Grade.\n"
                "Volvo FH / VNL: Filtro de Combustible OEM: 21707133 / Fleetguard: FF5785.\n"
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

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    nombre_modelo = elegir_modelo()
    model = genai.GenerativeModel(nombre_modelo)
    return retriever, model, nombre_modelo

def generar_respuesta_estructurada(model, nombre_modelo: str, prompt_final: str, imagen=None):
    candidatos = [nombre_modelo] + [m for m in MODELOS_PREFERIDOS if m != nombre_modelo]
    
    for nombre in candidatos:
        try:
            m = genai.GenerativeModel(nombre)
            inputs = [prompt_final]
            if imagen:
                inputs.append(imagen)
            response = m.generate_content(inputs)
            return response.text
        except Exception:
            continue
    return "⚠️ No se pudo procesar la solicitud con el modelo activo."

def obtener_imagen_repuesto(pregunta: str):
    p = pregunta.lower()
    if any(k in p for k in ["camion", "freightliner", "kenworth", "peterbilt", "volvo truck", "man", "scania", "iveco", "mack", "howo"]):
        return "https://images.unsplash.com/photo-1601584115197-04ecc0da31d7?auto=format&fit=crop&w=600&q=80", "Componente Heavy-Duty / Camiones"
    elif any(k in p for k in ["byd", "tesla", "nio", "xpeng", "zeekr", "lucid", "rivian"]):
        return "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=600&q=80", "Componente de Alta Tecnología / EV"
    else:
        return "https://images.unsplash.com/photo-1508974239320-0a029497e820?auto=format&fit=crop&w=600&q=80", "Pieza de Repuesto Sugerida"

def obtener_tiendas_recomendadas(pregunta: str, region: str, marca: str) -> str:
    p = (pregunta + " " + region + " " + marca).lower()
    tiendas = []
    
    if "camione" in p or "heavy" in p or any(k in p for k in ["freightliner", "kenworth", "peterbilt", "international", "volvo truck", "mack", "scania", "man", "iveco"]):
        tiendas.append("🚛 **FinditParts / FleetPride:** Líderes mundiales en inventario de piezas para camiones de carga pesada, suspensión de aire y filtración.")
        tiendas.append("🚛 **Raney's Truck Parts:** Especialistas en repuestos de carrocería, iluminación y mecánica de cabina.")
    elif any(k in p for k in ["byd", "changan", "geely", "great wall", "cheri", "mg", "jetour", "haima", "jac"]):
        tiendas.append("🇨🇳 **AliExpress / Alibaba Motors:** La fuente directa primaria para códigos OEM y módulos de marcas chinas.")
        tiendas.append("🛒 **CarPartsChina / BuyAutoParts:** Tiendas especializadas en importación de repuestos para marcas asiáticas emergentes.")
    elif any(k in p for k in ["bmw", "mercedes", "audi", "volkswagen", "porsche", "volvo"]):
        tiendas.append("🇪🇺 **FCP Euro / ECS Tuning:** Especialistas en marcas europeas con garantía de por vida en componentes mecánicos.")
        tiendas.append("🛒 **Pelican Parts:** El mejor catálogo para mantenimientos y despieces técnicos europeos.")
    else:
        tiendas.append("🛒 **RockAuto:** Catálogo global con precios de liquidación por código de pieza para marcas americanas, japonesas y coreanas.")
        tiendas.append("🛒 **CarParts.com / PartsGeek:** Gran cobertura con envíos rápidos en piezas de reemplazo directo.")

    tiendas.append("📦 **Amazon / eBay Motors:** Recomendado para búsqueda directa por número de parte OEM (Cross-Reference).")
    
    formato = "#### 🛒 ¿Dónde encontrar las mejores ofertas y precios para esta pieza?\n"
    for t in tiendas:
        formato += f"- {t}\n"
    formato += "\n💡 *Consejo de experto:* Copia el número de parte OEM de la tabla técnica y pégalo en la barra de búsqueda de las tiendas indicadas para garantizar 100% de compatibilidad."
    return formato

with st.spinner("Cargando base de datos multimarca global..."):
    retriever, model, nombre_modelo = inicializar_or_cargar_rag()

st.sidebar.caption(f"Modelo en ejecucion: {nombre_modelo}")
st.divider()

# --- INTERFAZ PRINCIPAL DE BÚSQUEDA ---
st.subheader("💬 Asistente de Consulta y Diagnóstico Universal")

imagen_subida = st.file_uploader("📷 Adjunta una foto de la pieza rota o desgastada (opcional):", type=["jpg", "jpeg", "png"])
if imagen_subida:
    img_preview = Image.open(imagen_subida)
    st.image(img_preview, caption="Imagen adjuntada por el usuario", width=250)

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if "imagen" in mensaje and mensaje["imagen"]:
            st.image(mensaje["imagen"], width=350, caption=mensaje.get("caption", ""))

pregunta_usuario = st.chat_input("Ej: ¿Qué pastillas de freno y filtro le sirven a mi auto?")

if pregunta_usuario:
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_usuario})
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en catálogos globales y analizando proveedores..."):
            
            info_vehiculo = f"{marca_sel} {modelo_vehiculo} ({año_vehiculo})".strip()
            consulta_rag = f"{pregunta_usuario} {info_vehiculo}"
                
            docs_relacionados = retriever.invoke(consulta_rag)
            contexto_texto = "\n\n".join([doc.page_content for doc in docs_relacionados])

            prompt_final = f"""
            Eres un mecánico master y especialista senior en catálogo de repuestos automotrices globales.
            Analiza el contexto y responde la consulta para el vehículo especificado.

            Contexto del Catálogo Local:
            {contexto_texto}

            Vehículo / Camión Seleccionado: {info_vehiculo} (Origen/Clase: {region_sel})
            Consulta del Usuario: {pregunta_usuario}

            INSTRUCCIONES DE RESPUESTA:
            1. Responde con precisión técnica en formato Markdown.
            2. Incluye una tabla o lista detallada con:
               - Nombre técnico exacto de la pieza.
               - Código de pieza OEM y equivalencias comerciales (Bosch, Denso, Donaldson, Brembo, Fleetguard, Akebono, etc.).
               - Recomendación de torque o instalación si aplica.
            """

            imagen_pil = Image.open(imagen_subida) if imagen_subida else None
            respuesta_tecnica = generar_respuesta_estructurada(model, nombre_modelo, prompt_final, imagen_pil)
            informacion_tiendas = obtener_tiendas_recomendadas(pregunta_usuario, region_sel, marca_sel)
            
            respuesta_completa = f"{respuesta_tecnica}\n\n---\n{informacion_tiendas}"
            url_img, caption_img = obtener_imagen_repuesto(pregunta_usuario + " " + marca_sel)

            st.markdown(respuesta_completa)
            st.image(url_img, width=380, caption=caption_img)

            st.session_state.mensajes.append({
                "rol": "assistant",
                "contenido": respuesta_completa,
                "imagen": url_img,
                "caption": caption_img
            })
