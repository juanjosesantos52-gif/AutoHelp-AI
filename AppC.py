"""AutoPartes AI - Backend para Render (FastAPI).
Sirve la interfaz (index.html) en "/" y la API en "/api/*" desde el mismo servicio.
Start Command en Render:  uvicorn main:app --host 0.0.0.0 --port $PORT
Variable de entorno requerida: GOOGLE_API_KEY
"""
import base64
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

BASE = Path(__file__).parent
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

# gemini-1.5-* ya fue apagado por Google: por eso daba NotFound. Se prueban en orden.
MODELOS_PREFERIDOS = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest"]
MODELO_EMBEDDING = "gemini-embedding-001"
CATALOGO = BASE / "catalogo_repuestos.txt"

REGIONES = {"usa": "Norteamérica", "jp": "Japón", "de": "Alemania", "cn": "China", "trucks": "Camiones pesados"}

MARCAS_GLOBALES = {
    "🇺🇸 Norteamérica (EE.UU., Canadá, México)": ["Acura", "Buick", "Cadillac", "Chevrolet", "Chrysler", "Dodge", "Eagle", "Ford", "GMC", "Hennessey", "Hummer", "Jeep", "Lincoln", "Lucid Motors", "Mercury", "Oldsmobile", "Panoz", "Plymouth", "Pontiac", "RAM", "Rivian", "Saleen", "Saturn", "Shelby", "SRT", "Tesla", "VLF Automotive"],
    "🇯🇵 Japón": ["Abarth (JP Spec)", "Daihatsu", "Datsun", "Honda", "Infiniti", "Isuzu", "Lexus", "Mazda", "Mitsubishi", "Mitsuoka", "Nissan", "Scion", "Subaru", "Suzuki", "Toyota", "Tommykaira"],
    "🇩🇪 Alemania": ["Audi", "Alpina", "BMW", "Borgward", "Bitter", "Gumpert / Apollo", "Maybach", "Mercedes-Benz", "Opel", "Porsche", "RUF", "Smart", "Volkswagen", "Wiesmann"],
    "🇮🇹 Italia": ["Abarth", "Alfa Romeo", "Autobianchi", "Bizzarrini", "Bugatti (Origen)", "De Tomaso", "Ferrari", "Fiat", "Fornasari", "Innocenti", "ISO Rivolta", "Lamborghini", "Lancia", "Maserati", "Mazzanti", "Pagani", "Pininfarina"],
    "🇬🇧 Reino Unido": ["AC Cars", "Aston Martin", "Austin", "Austin-Healey", "Bentley", "BAC (Mono)", "Caterham", "Ginetta", "Jaguar", "Jensen", "Land Rover", "Lister", "Lotus", "Marcos", "McLaren", "MG", "Mini", "Morgan", "Noble", "Radical", "Rolls-Royce", "Rover", "Triumph", "TVR", "Vauxhall", "Westfield"],
    "🇫🇷 Francia": ["Alpine", "Bugatti", "Citroën", "DS Automobiles", "Ligier", "Microcar", "Peugeot", "Renault", "Venturi"],
    "🇨🇳 China": ["Aiways", "BAIC", "Baojun", "BYD", "Changan", "Chery", "Dongfeng", "FAW", "Foton", "GAC Group", "Geely", "Great Wall (GWM)", "Haval", "Hongqi", "JAC Motors", "Jetour", "Leapmotor", "Li Auto", "Lynk & Co", "NIO", "Omoda", "SAIC Motor", "Voyah", "Wuling", "XPeng", "Zeekr"],
    "🇰🇷 Corea del Sur": ["Daewoo", "Genesis", "Hyundai", "KG Mobility (SsangYong)", "Kia", "Renault Korea (Samsung)"],
    "🇪🇸 🇸🇪 🇳🇱 Resto de Europa": ["Abarth", "Artega", "Donkervoort", "GTA Spano", "Koenigsegg", "KTM (X-Bow)", "Lada (Rusia)", "Rimac (Croacia)", "SEAT", "Skoda (Rep. Checa)", "Spyker", "Troller", "Volvo", "Zastava"],
    "🚚 Camiones Pesados & Vehículos Comerciales": ["Actros (Mercedes)", "Astra", "Avia", "DAF", "DONGFENG Commercial", "ERF", "Faw Jiefang", "Foton Auman", "Freightliner", "Hino", "Howo (Sinotruk)", "Hyundai Commercial", "International / Navistar", "Isuzu Truck", "IVECO", "Kamaz", "Kenworth", "Mack Trucks", "MAN", "Mercedes-Benz Trucks", "Nissan Diesel / UD Trucks", "Peterbilt", "Renault Trucks", "Scania", "Shacman", "Tatra", "Terex", "Volvo Trucks", "Western Star"],
    "🏎️ Hiperautos & Marcas Exclusivas Globales": ["Apollo Automobili", "Ariel", "Czinger", "Devel Sixteen", "Drako", "KTM", "Naran", "SSC North America", "Zenvo"],
}

CATALOGO_DEFAULT = (
    "Catálogo Técnico Universal de Repuestos y Equivalencias Automotrices e Industriales.\n"
    "Mazda 3 (2010-2013): Pastillas delanteras OEM: MZ-301FR / Bosch: BP-450.\n"
    "Toyota Corolla (2014-2019): Pastillas delanteras OEM: TY-C140 / Akebono: ACT-1211.\n"
    "Ford Mustang (2015+): Pastillas delanteras OEM: FR-BRK-15 / Brembo: P-59-088.\n"
    "Freightliner Cascadia (DD15): Filtro de Aceite OEM: A4721800609 / Donaldson: P550821.\n"
    "BYD Han / Tang: Pastillas de freno Brembo OEM: BYD-BRK-EV9 / Bosch EV Grade.\n"
    "Volvo FH / VNL: Filtro de Combustible OEM: 21707133 / Fleetguard: FF5785.\n"
)

app = FastAPI(title="AutoPartes AI Backend API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    region: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    image: Optional[str] = None  # data URL base64 (opcional)


# ---------- RAG (catálogo + embeddings de Gemini, sin torch: cabe en el plan gratis de Render) ----------
_indice = {"lineas": None, "vecs": None}


def cargar_catalogo():
    if not CATALOGO.exists():
        CATALOGO.write_text(CATALOGO_DEFAULT, encoding="utf-8")
    return [l.strip() for l in CATALOGO.read_text(encoding="utf-8").splitlines() if l.strip()]


def _embed(textos, tarea):
    r = client.models.embed_content(model=MODELO_EMBEDDING, contents=textos, config=types.EmbedContentConfig(task_type=tarea))
    v = np.array([e.values for e in r.embeddings], dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def buscar_contexto(consulta, k=3):
    lineas = cargar_catalogo()
    try:
        if _indice["vecs"] is None or _indice["lineas"] != lineas:
            _indice["vecs"], _indice["lineas"] = _embed(lineas, "RETRIEVAL_DOCUMENT"), lineas
        q = _embed([consulta], "RETRIEVAL_QUERY")[0]
        top = np.argsort(-(_indice["vecs"] @ q))[:k]
        return "\n".join(lineas[i] for i in top)
    except Exception as e:  # si falla el embedding, el catálogo es corto: se envía completo
        print(f"[RAG] embeddings no disponibles, usando catálogo completo: {e}")
        return "\n".join(lineas)


# ---------- Gemini ----------
def parsear_imagen(data_url):
    m = re.match(r"data:(image/[\w.+-]+);base64,(.+)", data_url or "", re.S)
    if not m:
        return None
    return types.Part.from_bytes(data=base64.b64decode(m.group(2)), mime_type=m.group(1))


def generar(prompt, imagen=None):
    contenido = [prompt] + ([imagen] if imagen else [])
    for nombre in MODELOS_PREFERIDOS:
        try:
            r = client.models.generate_content(model=nombre, contents=contenido)
            if r.text:
                return r.text
        except Exception as e:
            print(f"[Gemini] {nombre} falló: {e}")
    raise HTTPException(status_code=502, detail="No se pudo generar la respuesta con Gemini. Revisa los logs de Render y tu GOOGLE_API_KEY.")


# ---------- Extras del catálogo (igual que en la versión Streamlit) ----------
def obtener_imagen_repuesto(pregunta):
    p = pregunta.lower()
    if any(k in p for k in ["camion", "freightliner", "kenworth", "peterbilt", "volvo truck", "man", "scania", "iveco", "mack", "howo"]):
        return "https://images.unsplash.com/photo-1601584115197-04ecc0da31d7?auto=format&fit=crop&w=600&q=80", "Componente Heavy-Duty / Camiones"
    if any(k in p for k in ["byd", "tesla", "nio", "xpeng", "zeekr", "lucid", "rivian"]):
        return "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=600&q=80", "Componente de Alta Tecnología / EV"
    return "https://images.unsplash.com/photo-1508974239320-0a029497e820?auto=format&fit=crop&w=600&q=80", "Pieza de Repuesto Sugerida"


def obtener_tiendas_recomendadas(pregunta, region, marca):
    p = f"{pregunta} {region} {marca}".lower()
    t = []
    if "camione" in p or "heavy" in p or any(k in p for k in ["freightliner", "kenworth", "peterbilt", "international", "volvo truck", "mack", "scania", "man", "iveco"]):
        t += ["🚛 **FinditParts / FleetPride:** inventario de piezas para camiones de carga pesada, suspensión de aire y filtración.",
              "🚛 **Raney's Truck Parts:** repuestos de carrocería, iluminación y mecánica de cabina."]
    elif any(k in p for k in ["byd", "changan", "geely", "great wall", "chery", "mg", "jetour", "haima", "jac"]):
        t += ["🇨🇳 **AliExpress / Alibaba Motors:** fuente directa de códigos OEM y módulos de marcas chinas.",
              "🛒 **CarPartsChina / BuyAutoParts:** importación de repuestos para marcas asiáticas emergentes."]
    elif any(k in p for k in ["bmw", "mercedes", "audi", "volkswagen", "porsche", "volvo"]):
        t += ["🇪🇺 **FCP Euro / ECS Tuning:** especialistas en marcas europeas con garantía en componentes mecánicos.",
              "🛒 **Pelican Parts:** catálogo para mantenimiento y despieces técnicos europeos."]
    else:
        t += ["🛒 **RockAuto:** catálogo global con buenos precios por código de pieza (marcas americanas, japonesas y coreanas).",
              "🛒 **CarParts.com / PartsGeek:** gran cobertura con envíos rápidos en reemplazos directos."]
    t.append("📦 **Amazon / eBay Motors:** útil para buscar directo por número de parte OEM (cross-reference).")
    return ("#### 🛒 ¿Dónde encontrar las mejores ofertas para esta pieza?\n" + "\n".join(f"- {x}" for x in t)
            + "\n\n💡 *Consejo:* copia el número OEM de la tabla técnica y pégalo en el buscador de la tienda para confirmar compatibilidad.")


# ---------- Rutas ----------
@app.get("/api/health")
def api_health():
    return {"status": "online", "gemini_key": bool(API_KEY), "models": MODELOS_PREFERIDOS}


@app.get("/api/brands")
def api_get_brands():
    return {"categories": MARCAS_GLOBALES}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="Falta la variable de entorno GOOGLE_API_KEY en Render.")
    region = REGIONES.get(req.region or "", req.region or "")
    vehiculo = f"{req.brand or ''} {req.model or ''} ({req.year or ''})".strip()
    imagen = parsear_imagen(req.image)

    contexto = buscar_contexto(f"{req.message} {vehiculo}")
    prompt = f"""Eres un mecánico master y especialista senior en catálogo de repuestos automotrices globales.
Responde siempre en español, en Markdown, para el vehículo indicado.

Contexto del catálogo local (puede no incluir la pieza; si no aparece, dilo y da orientación general indicando que se verifique el código OEM):
{contexto}

Vehículo / camión: {vehiculo} (Origen/Clase: {region})
Consulta del usuario: {req.message}
{"Se adjunta una foto de la pieza: identifícala y úsala para responder." if imagen else ""}

Incluye una tabla o lista con:
- Nombre técnico exacto de la pieza.
- Código OEM y equivalencias comerciales (Bosch, Denso, Donaldson, Brembo, Fleetguard, Akebono, etc.).
- Recomendación de torque o instalación si aplica."""

    tecnica = generar(prompt, imagen)
    tiendas = obtener_tiendas_recomendadas(req.message, region, req.brand or "")
    url_img, caption = obtener_imagen_repuesto(f"{req.message} {req.brand or ''}")
    return {"reply": f"{tecnica}\n\n---\n{tiendas}", "vehicle": vehiculo, "status": "success",
            "image_url": url_img, "image_caption": caption}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "index.html")
