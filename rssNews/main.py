# requirements.txt
# Instala estas dependencias primero: pip install -r requirements.txt
"""
requests==2.31.0
beautifulsoup4==4.12.2
newspaper3k==0.2.8
lxml==4.9.3
html5lib==1.1
"""

import json
import requests
from bs4 import BeautifulSoup
import time
from newspaper import Article
import os
from datetime import datetime

def extraer_texto_noticia(url):
    """
    Extrae el texto completo de una noticia usando múltiples métodos
    """
    try:
        # Método 1: Intentar con newspaper3k (más efectivo para noticias)
        try:
            articulo = Article(url, language='es')
            articulo.download()
            articulo.parse()
            
            if articulo.text and len(articulo.text) > 200:
                return articulo.text
        except Exception as e:
            print(f"    Newspaper error: {str(e)[:50]}...")
            pass
        
        # Método 2: BeautifulSoup como respaldo
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        respuesta = requests.get(url, timeout=15, headers=headers)
        respuesta.encoding = 'utf-8'
        sopa = BeautifulSoup(respuesta.content, 'html.parser')
        
        # Buscar contenido en elementos comunes de noticias
        selectores = [
            'article',
            '.article-content',
            '.story-content',
            '.news-content',
            '.entry-content',
            '.post-content',
            '[class*="content"]',
            '[class*="body"]',
            '.nota-content',
            '.note-content',
            'main',
            '.news-body',
            '.article-body',
            '.post-body'
        ]
        
        for selector in selectores:
            elementos = sopa.select(selector)
            for elemento in elementos:
                # Limpiar elementos no deseados
                for tag in elemento(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'button', 'form']):
                    tag.decompose()
                
                texto = elemento.get_text(separator='\n', strip=True)
                if len(texto) > 300:  # Umbral más alto para contenido real
                    # Limpiar líneas muy cortas
                    lineas_limpias = [linea.strip() for linea in texto.split('\n') 
                                    if len(linea.strip()) > 30 and not linea.strip().startswith('©')]
                    texto_final = '\n'.join(lineas_limpias)
                    if len(texto_final) > 300:
                        return texto_final
        
        # Método 3: Buscar en divs principales
        divs_principales = sopa.find_all('div', class_=lambda x: x and any(palabra in x.lower() for palabra in 
                                                                          ['content', 'article', 'story', 'main', 'body', 'nota']))
        for div in divs_principales:
            for tag in div(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                tag.decompose()
            
            texto = div.get_text(separator='\n', strip=True)
            lineas_limpias = [linea.strip() for linea in texto.split('\n') 
                            if len(linea.strip()) > 40 and not linea.strip().startswith('©')]
            texto_final = '\n'.join(lineas_limpias)
            if len(texto_final) > 300:
                return texto_final
        
        return None
        
    except Exception as e:
        print(f"    Error extrayendo {url}: {str(e)[:80]}")
        return None

def procesar_json_noticias(archivo_json):
    """
    Procesa un archivo JSON con la estructura específica y extrae el texto completo
    """
    # Cargar el archivo JSON
    with open(archivo_json, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    # Verificar la estructura del JSON
    if isinstance(datos, list):
        # Si es una lista de noticias
        print("📖 Estructura detectada: Lista de noticias")
        grupos_noticias = {'noticias': datos}
    elif isinstance(datos, dict):
        # Si es un diccionario con temas
        print("📖 Estructura detectada: Diccionario por temas")
        grupos_noticias = datos
    else:
        raise ValueError("Formato JSON no reconocido")
    
    resultados_completos = {}
    estadisticas = {}
    
    for tema, noticias in grupos_noticias.items():
        print(f"\n📖 Procesando: {tema}")
        print(f"   📊 Encontradas {len(noticias)} noticias")
        
        resultados_tema = []
        exitosas = 0
        
        for i, noticia in enumerate(noticias):
            # Verificar que tenga la estructura esperada
            if not isinstance(noticia, dict):
                print(f"   ⚠️  Noticia {i+1} no es un diccionario, saltando...")
                continue
                
            if 'enlace' not in noticia:
                print(f"   ⚠️  Noticia {i+1} sin enlace, saltando...")
                continue
            
            titulo = noticia.get('titulo', f'Noticia {i+1}')
            print(f"   📄 [{i+1}/{len(noticias)}] Extrayendo: {titulo[:70]}...")
            
            texto_completo = extraer_texto_noticia(noticia['enlace'])
            
            if texto_completo:
                # Crear copia de la noticia con el texto completo
                noticia_completa = noticia.copy()
                noticia_completa['texto_completo'] = texto_completo
                noticia_completa['longitud_texto'] = len(texto_completo)
                noticia_completa['palabras_texto'] = len(texto_completo.split())
                
                resultados_tema.append(noticia_completa)
                exitosas += 1
                print(f"      ✅ Éxito ({len(texto_completo)} caracteres, {len(texto_completo.split())} palabras)")
            else:
                print(f"      ❌ No se pudo extraer texto completo")
                # Guardar igual pero sin texto completo
                noticia_completa = noticia.copy()
                noticia_completa['texto_completo'] = ''
                noticia_completa['longitud_texto'] = 0
                noticia_completa['palabras_texto'] = 0
                resultados_tema.append(noticia_completa)
            
            time.sleep(1)  # Espera entre peticiones para ser respetuoso
        
        resultados_completos[tema] = resultados_tema
        estadisticas[tema] = {
            'total': len(noticias),
            'exitosas': exitosas,
            'porcentaje': (exitosas / len(noticias)) * 100 if noticias else 0
        }
        
        print(f"   📊 {exitosas}/{len(noticias)} noticias extraídas exitosamente")
    
    return resultados_completos, estadisticas

def guardar_resultados_txt(resultados_completos, nombre_base):
    """
    Guarda los resultados en archivos de texto
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta_salida = f"noticias_completas_{timestamp}"
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # 1. Archivo único con todas las noticias
    ruta_todas = os.path.join(carpeta_salida, f'{nombre_base}_TODAS.txt')
    with open(ruta_todas, 'w', encoding='utf-8') as f:
        f.write("📰 COLECCIÓN COMPLETA DE NOTICIAS\n")
        f.write("=" * 70 + "\n\n")
        
        for tema, noticias in resultados_completos.items():
            f.write(f"🎯 TEMA: {tema.upper()}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, noticia in enumerate(noticias, 1):
                f.write(f"📖 NOTICIA {i}: {noticia['titulo']}\n")
                f.write(f"🔗 Enlace: {noticia['enlace']}\n")
                f.write(f"📅 Fecha: {noticia.get('fecha', 'N/A')}\n")
                if noticia.get('resumen'):
                    f.write(f"📝 Resumen: {noticia['resumen'][:200]}...\n")
                f.write(f"📊 Longitud: {noticia['longitud_texto']} caracteres, {noticia['palabras_texto']} palabras\n")
                f.write("-" * 50 + "\n")
                
                if noticia['texto_completo']:
                    f.write(noticia['texto_completo'])
                else:
                    f.write("❌ No se pudo extraer el texto completo de esta noticia\n")
                
                f.write("\n\n" + "★" * 70 + "\n\n")
    
    # 2. Archivos separados por tema
    for tema, noticias in resultados_completos.items():
        if noticias:
            ruta_tema = os.path.join(carpeta_salida, f'{nombre_base}_{tema}.txt')
            with open(ruta_tema, 'w', encoding='utf-8') as f:
                f.write(f"📰 NOTICIAS SOBRE: {tema.upper()}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, noticia in enumerate(noticias, 1):
                    f.write(f"NOTICIA {i}:\n")
                    f.write(f"Título: {noticia['titulo']}\n")
                    f.write(f"Enlace: {noticia['enlace']}\n")
                    f.write(f"Fecha: {noticia.get('fecha', 'N/A')}\n")
                    if noticia.get('resumen'):
                        f.write(f"Resumen: {noticia['resumen']}\n")
                    f.write("-" * 40 + "\n")
                    
                    if noticia['texto_completo']:
                        f.write(noticia['texto_completo'])
                    else:
                        f.write("❌ No se pudo extraer el texto completo de esta noticia\n")
                    
                    f.write("\n\n" + "=" * 60 + "\n\n")
    
    # 3. Solo textos (limpio para LLM)
    ruta_solo_texto = os.path.join(carpeta_salida, f'{nombre_base}_SOLO_TEXTO.txt')
    with open(ruta_solo_texto, 'w', encoding='utf-8') as f:
        for tema, noticias in resultados_completos.items():
            f.write(f"[TEMA: {tema.upper()}]\n")
            f.write("=" * 50 + "\n\n")
            
            for i, noticia in enumerate(noticias, 1):
                if noticia['texto_completo']:
                    f.write(f"[NOTICIA {i}: {noticia['titulo']}]\n")
                    f.write(noticia['texto_completo'])
                    f.write("\n\n[FIN_NOTICIA]\n\n")
    
    # 4. JSON actualizado con textos completos
    ruta_json = os.path.join(carpeta_salida, f'{nombre_base}_COMPLETO.json')
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(resultados_completos, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Resultados guardados en la carpeta: {carpeta_salida}")
    return carpeta_salida

def main():
    """
    Función principal para ejecutar localmente
    """
    print("🚀 EXTRACTOR DE TEXTO COMPLETO DE NOTICIAS")
    print("=" * 50)
    
    # Solicitar archivo JSON
    archivo_json = input("📁 Ingresa el nombre de tu archivo JSON (ej: noticias.json): ").strip()
    
    if not os.path.exists(archivo_json):
        print(f"❌ El archivo '{archivo_json}' no existe")
        return
    
    # Procesar el archivo JSON
    print(f"\n📖 Cargando y procesando: {archivo_json}")
    resultados, stats = procesar_json_noticias(archivo_json)
    
    # Guardar resultados
    nombre_base = os.path.splitext(os.path.basename(archivo_json))[0]
    carpeta_salida = guardar_resultados_txt(resultados, nombre_base)
    
    # Mostrar estadísticas
    print("\n📊 ESTADÍSTICAS FINALES")
    print("=" * 50)
    
    total_noticias = 0
    total_exitosas = 0
    
    for tema, stat in stats.items():
        print(f"{tema.upper():<20}: {stat['exitosas']:>2}/{stat['total']:>2} ({stat['porcentaje']:.1f}%)")
        total_noticias += stat['total']
        total_exitosas += stat['exitosas']
    
    print(f"\nTOTAL: {total_exitosas}/{total_noticias} noticias extraídas")
    print(f"PORCENTAJE GLOBAL: {(total_exitosas/total_noticias)*100:.1f}%")
    
    print(f"\n🎉 ¡Proceso completado!")
    print(f"📂 Los archivos están en: {carpeta_salida}/")
    print(f"📄 Archivos generados:")
    print(f"   - {nombre_base}_TODAS.txt (completo con metadatos)")
    print(f"   - {nombre_base}_SOLO_TEXTO.txt (limpio para LLM)")
    print(f"   - Archivos por tema individuales")
    print(f"   - {nombre_base}_COMPLETO.json (JSON actualizado)")

# Ejecutar el script
if __name__ == "__main__":
    main()