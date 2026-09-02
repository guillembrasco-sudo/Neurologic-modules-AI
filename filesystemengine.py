from pathlib import Path
import subprocess
import os
import json
import re
from typing import Dict, Any, Tuple
import sys

# Directorio raíz restringido (Directorio de trabajo actual)
WORKSPACE_DIR = Path.cwd().resolve()

def obtener_ruta_segura(nombre_archivo: str) -> Path:
    """
    Resuelve la ruta absoluta y verifica mediante inspección jerárquica
    que el archivo permanezca estrictamente dentro de WORKSPACE_DIR.
    """
    ruta_objetivo = (WORKSPACE_DIR / nombre_archivo).resolve()
    if WORKSPACE_DIR not in ruta_objetivo.parents and ruta_objetivo != WORKSPACE_DIR:
        raise PermissionError(
            f"VULNERABILIDAD DETECTADA: La ruta '{nombre_archivo}' viola el límite del Sandbox."
        )
    return ruta_objetivo


class FileSystemEngine:
    """Ejecutor de operaciones de disco y código para AgentCLI."""

    @staticmethod
    def leer_archivo(nombre_archivo: str) -> str:
        ruta = obtener_ruta_segura(nombre_archivo)
        if not ruta.is_file():
            return f"Error: El archivo '{nombre_archivo}' no existe."
        return ruta.read_text(encoding="utf-8")

    @staticmethod
    def escribir_archivo(nombre_archivo: str, contenido: str) -> str:
        ruta = obtener_ruta_segura(nombre_archivo)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(contenido, encoding="utf-8")
        return f"✓ Archivo '{nombre_archivo}' guardado correctamente ({len(contenido)} bytes)."

    @staticmethod
    def listar_archivos() -> str:
        elementos = [item.name for item in WORKSPACE_DIR.iterdir()]
        return "Archivos en workspace:\n" + "\n".join(f"- {e}" for e in elementos)

    @staticmethod
    def borrar_archivo(nombre_archivo: str) -> str:
        ruta = obtener_ruta_segura(nombre_archivo)
        if not ruta.exists():
            return f"Error: El archivo '{nombre_archivo}' no existe."
        if ruta.is_dir():
            return f"Error: '{nombre_archivo}' es un directorio."
        ruta.unlink()
        return f"✓ Archivo '{nombre_archivo}' eliminado del disco."

    @staticmethod
    def ejecutar_python(nombre_archivo: str, timeout: int = 15) -> str:
        """Ejecuta un script Python usando subprocess.run con límites de recursos."""
        ruta = obtener_ruta_segura(nombre_archivo)
        if not ruta.is_file():
            return f"Error de ejecución: El script '{nombre_archivo}' no existe."

        try:
            proceso = subprocess.run(
                [sys.executable, str(ruta)],
                cwd=WORKSPACE_DIR,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            salida = proceso.stdout.strip()
            errores = proceso.stderr.strip()
            
            res = f"--- STDOUT (Código de salida: {proceso.returncode}) ---\n"
            res += salida if salida else "(Sin salida estándar)"
            if errores:
                res += f"\n--- STDERR ---\n{errores}"
            return res

        except subprocess.TimeoutExpired:
            return f"Error: La ejecución de '{nombre_archivo}' excedió el límite de {timeout} segundos."
        except Exception as e:
            return f"Error crítico de subprocess: {str(e)}"