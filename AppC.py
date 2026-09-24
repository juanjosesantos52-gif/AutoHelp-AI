import os
import io
import numpy as np
from typing import Optional
from PIL import Image

# Importaciones de FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Librería ligera oficial de Google Gemini
import google.generativeai as genai

# --- CONFIGURACIÓN DE API KEY ---
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

MODELOS_PREFERIDOS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# --- LISTA COMPLETA Y GLOBAL DE MARCAS DEL MUNDO (CONSERVADA INTACTA) ---
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

# --- DOCUMENTOS Y RAG ULTRA-LIGERO (<30MB RAM) ---
DOCUMENTOS_CATALOGO = [
    "Catálogo Técnico Universal de Repuestos y Equivalencias Automotrices e Industriales.",
    "Mazda 3 (2010-2013): Pastillas delanteras OEM: MZ-301FR / Bosch: BP-450.",
    "Toyota Corolla (2014-2019): Pastillas delanteras OEM: TY-C140 / Akebono: ACT-1211.",
    "Ford Mustang (2015+): Pastillas delanteras OEM: FR-BRK-15 / Brembo: P-59-088.",
    "Freightliner Cascadia (DD15): Filtro de Aceite OEM: A4721800609 / Donaldson: P550821.",
    "BYD Han / Tang: Pastillas de freno Brembo OEM: BYD-BRK-EV9 / Bosch EV Grade.",
    "Volvo FH / VNL: Filtro de Combustible OEM: 21707133 / Fleetguard: FF5785."
]

EMBEDDINGS_CACHE = []

def inicializar_rag():
    global EMBEDDINGS_CACHE
    if not API_KEY or EMBEDDINGS_CACHE:
        return
    try:
        for doc in DOCUMENTOS_CATALOGO:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=doc
            )
            EMBEDDINGS_CACHE.append(res['embedding'])
    except Exception as e:
        print(f"Error inicializando embeddings RAG: {e}")

def buscar_contexto_rag(query: str, k: int = 3) -> str:
    if not API_KEY or not EMBEDDINGS_CACHE:
        return "\n".join(DOCUMENTOS_CATALOGO)
    try:
        q_res = genai.embed_content(
            model="models/text-embedding-004",
            content=query
        )
        q_vec = np.array(q_res['embedding'])
        
        similitudes = []
        for doc_vec in EMBEDDINGS_CACHE:
            d_vec = np.array(doc_vec)
            score = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            similitudes.append(score)
            
        indices = np.argsort(similitudes)[-k:][::-1]
        return "\n\n".join([DOCUMENTOS_CATALOGO[i] for i in indices])
    except Exception:
        return "\n\n".join(DOCUMENTOS_CATALOGO[:3])

# --- FUNCIONES AUXILIARES (TIENDAS E IMÁGENES) ---
def obtener_imagen_repuesto(pregunta: str):
    p = pregunta.lower()
    if any(k in p for k in ["camion", "freightliner", "kenworth", "peterbilt", "volvo truck", "man", "scania", "iveco", "mack", "howo"]):
        return "https://images.unsplash.com/photo-1601584115197-04ecc0da31d7?auto=format&fit=crop&w=600&q=80", "Componente Heavy-Duty / Camiones"
    elif any(k in p for k in ["byd", "tesla", "nio", "xpeng", "zeekr", "lucid", "rivian"]):
        return "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=600&q=80", "Componente de Alta Tecnología / EV"
    else:
        return "https://images.unsplash.com/photo-1508974239320-0a029497e820?auto=format&fit=crop&w=600&q=80", "Pieza de Repuesto Sugerida"

def obtener_tiendas_recomendadas(pregunta: str, region: str, marca: str) -> str:
    p = (pregunta + " " + str(region) + " " + str(marca)).lower()
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

def generar_respuesta_estructurada(nombre_modelo: str, prompt_final: str, imagen=None):
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

# --- SERVIDOR FASTAPI ---
app = FastAPI(title="AutoPartes AI Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    inicializar_rag()

@app.get("/api/health")
def api_health():
    return {"status": "online", "message": "Backend API corriendo en Render."}

@app.get("/api/brands")
def api_get_brands():
    return {"categories": MARCAS_GLOBALES}

@app.post("/api/chat")
async def api_chat(
    message: str = Form(...),
    region: Optional[str] = Form(None),
    brand: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY / GOOGLE_API_KEY no encontrada en las variables del entorno.")

    info_vehiculo = f"{brand or ''} {model or ''} ({year or ''})".strip()
    consulta_rag = f"{message} {info_vehiculo}"
    
    contexto_texto = buscar_contexto_rag(consulta_rag)

    prompt_final = f"""
    Eres un mecánico master y especialista senior en catálogo de repuestos automotrices globales.
    Analiza el contexto y responde la consulta para el vehículo especificado.

    Contexto del Catálogo Local:
    {contexto_texto}

    Vehículo / Camión Seleccionado: {info_vehiculo if info_vehiculo else 'No especificado'} (Origen/Clase: {region or 'General'})
    Consulta del Usuario: {message}

    INSTRUCCIONES DE RESPUESTA:
    1. Responde con precisión técnica en formato Markdown.
    2. Incluye una tabla o lista detallada con:
       - Nombre técnico exacto de la pieza.
       - Código de pieza OEM y equivalencias comerciales (Bosch, Denso, Donaldson, Brembo, Fleetguard, Akebono, etc.).
       - Recomendación de torque o instalación si aplica.
    """

    imagen_pil = None
    if file:
        img_bytes = await file.read()
        imagen_pil = Image.open(io.BytesIO(img_bytes))

    nombre_modelo = elegir_modelo()
    respuesta_tecnica = generar_respuesta_estructurada(nombre_modelo, prompt_final, imagen_pil)
    informacion_tiendas = obtener_tiendas_recomendadas(message, region or "", brand or "")
    
    respuesta_completa = f"{respuesta_tecnica}\n\n---\n{informacion_tiendas}"
    url_img, caption_img = obtener_imagen_repuesto(message + " " + (brand or ""))

    return {
        "reply": respuesta_completa,
        "vehicle": info_vehiculo,
        "image_url": url_img,
        "image_caption": caption_img,
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
