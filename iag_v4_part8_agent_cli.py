"""
IAG Modular V4 - Parte 8
Agente completo con chat, persistencia y comandos.

Este archivo monta BrainCore sobre una CLI interactiva:
- conversación
- guardar / cargar estado
- memoria
- estado del sistema
- objetivos
- razonamiento explícito
- consolidación manual
- salida limpia

Requiere que existan las partes anteriores en /mnt/data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import sys
import json
import time
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# ============================================================
# Dispositivo y Tipos Globales
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


BASE_DIR = Path("E:/IA/Brain")
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

AVAILABLE_TOOLS = {
    "escribir_archivo": {
        "description": "Crea o sobrescribe un archivo en el sistema de archivos.",
        "args": {
            "nombre_archivo": "(string) Nombre o ruta relativa del archivo.",
            "contenido": "(string) Texto o código que se escribirá en el archivo.",
        },
    },
    "leer_archivo": {
        "description": "Lee y devuelve el contenido de un archivo de texto.",
        "args": {"nombre_archivo": "(string) Nombre del archivo a leer."},
    },
    "borrar_archivo": {
        "description": "Elimina un archivo del sistema.",
        "args": {"nombre_archivo": "(string) Nombre del archivo a eliminar."},
    },
    "listar_archivos": {
        "description": "Devuelve la lista de archivos disponibles en el directorio de trabajo.",
        "args": {},
    },
    "ejecutar_python": {
        "description": "Ejecuta un script de Python en el entorno sandbox y devuelve la salida (stdout/stderr).",
        "args": {"nombre_archivo": "(string) Nombre del script .py a ejecutar."},
    },
    "think": {
        "description": "Permite pensar durante más tiempo para obtener respuestas de mayor calidad",
        "args": {"theme": "Tema a tratar, si no se pone nada, se trata el input directo del usuario"},
    },
}


def _import_from_file(module_name: str, file_name: str):
    path = BASE_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Falta el archivo requerido: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar: {file_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


p7 = _import_from_file("iag_v4_part7_braincore", "iag_v4_part7_braincore.py")
p_fileengine = _import_from_file("filesystemengine", "filesystemengine.py")


@dataclass(slots=True)
class SessionCommandResult:
    text: str
    should_exit: bool = False


def _make_json_serializable(obj: Any) -> Any:
    """
    Convierte de forma recursiva cualquier objeto cognitivo complejo, tensor de PyTorch
    o estructura de datos protegida (dataclass con slots) en un formato compatible con JSON.
    """
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        return obj.detach().cpu().tolist()
    if hasattr(obj, "compact") and callable(obj.compact):
        return _make_json_serializable(obj.compact())
    if hasattr(obj, "export") and callable(obj.export):
        return _make_json_serializable(obj.export())
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        try:
            from dataclasses import asdict

            return _make_json_serializable(asdict(obj))
        except TypeError:
            return {f: _make_json_serializable(getattr(obj, f)) for f in obj.__dataclass_fields__}
    return obj

DANGEROUS_TOOLS = {"ejecutar_python", "borrar_archivo", "escribir_archivo"}

class AgentCLI:
    def __init__(self, brain: Optional[Any] = None):
        self.brain = brain if brain is not None else p7.BrainCore(device=DEVICE)
        self.fs_engine = p_fileengine.FileSystemEngine()
        self.running = True

    # --------------------------------------------------------
    # Comandos Cognitivos y de Control
    # --------------------------------------------------------

    def help_text(self) -> str:
        return (
            "\n=== PANEL DE COMANDOS COGNITIVOS ===\n"
            "  /help                           Muestra esta guía de comandos.\n"
            "  /status                         Reporte del estado actual de neuromodulación y memorias.\n"
            "  /correct [texto]                Inserta una corrección explícita sobre el último turno.\n"
            "  /explain                        Muestra las señales lógicas y causales de la última inferencia.\n"
            "  /save [archivo.pkl]             Persiste el estado actual del agente en disco.\n"
            "  /load [archivo.pkl]             Restaura un estado cognitivo guardado.\n"
            "  /history [n]                    Muestra los últimos 'n' turnos de conversación con sus scores.\n"
            "  /mem [n]                        Inspecciona los últimos 'n' recuerdos en orden cronológico real.\n"
            "  /graph                          Resumen topológico del Grafo Cognitivo GNN.\n"
            "  /goals                          Lista de objetivos existenciales activos.\n"
            "  /goal [texto]                   Inserta un nuevo objetivo existencial.\n"
            "  /remember [clave]: [valor]      Inserta un recuerdo conceptual de forma manual.\n"
            "  /recall [clave]                 Recupera de forma directa un valor de la base de hechos.\n"
            "  /search [texto]                 Búsqueda asociativa en memoria mediante similitud de embeddings.\n"
            "  /consolidate                    Fuerza la consolidación episódico-semántica y purga de memorias.\n"
            "  /train [archivo.txt]            Ingesta masiva de conocimiento estructurado por bloques.\n"
            "  /reset                          Reinicia por completo la instancia de BrainCore.\n"
            "  /export_json [archivo.json]     Exporta un dump matemático limpio de todo el cerebro.\n"
            "  /exit                           Finaliza de forma segura la sesión del agente."
        )

    def handle_command(self, text: str) -> SessionCommandResult:
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in {"/exit", "exit", "quit"}:
            return SessionCommandResult("Saliendo.", should_exit=True)

        if cmd == "/help":
            return SessionCommandResult(self.help_text())

        if cmd == "/status":
            return SessionCommandResult(self.brain.status())

        if cmd == "/explain":
            return SessionCommandResult(self.brain.explain_last())

        if cmd == "/save":
            filename = arg.strip() or "braincore.pkl"
            path = self.brain.save(filename)
            return SessionCommandResult(f"Guardado en: {path}")

        if cmd == "/load":
            filename = arg.strip() or "braincore.pkl"
            try:
                self.brain.load(filename)
                return SessionCommandResult(f"Estado cargado con éxito.")
            except Exception as e:
                return SessionCommandResult(f"Error crítico al restaurar estado: {str(e)}")

        if cmd == "/history":
            n = 10
            if arg.strip().isdigit():
                n = int(arg.strip())
            hist = self.brain.history[-n:]
            if not hist:
                return SessionCommandResult("Sin historial.")
            lines = []
            for row in hist:
                lines.append(
                    f"[{row.get('turn')}] {row.get('mode')} conf={row.get('confidence', 0.0):.3f} :: {row.get('text')}"
                )
            return SessionCommandResult("\n".join(lines))

        if cmd == "/mem":
            n = 8
            if arg.strip().isdigit():
                n = int(arg.strip())

            valid_size = self.brain.hdc._size
            if valid_size == 0:
                return SessionCommandResult("La memoria semántica y episódica se encuentra vacía.")

            active_timestamps = self.brain.hdc._timestamps[:valid_size]

            if isinstance(active_timestamps, torch.Tensor):
                sorted_indices = torch.argsort(active_timestamps).cpu().tolist()
            else:
                sorted_indices = sorted(range(len(active_timestamps)), key=lambda i: active_timestamps[i])

            most_recent_indices = sorted_indices[-n:]

            lines = []
            for idx in reversed(most_recent_indices):
                k = self.brain.hdc._keys[idx]
                payload = self.brain.hdc._payloads[idx]
                ts = active_timestamps[idx]
                if isinstance(ts, torch.Tensor):
                    ts = ts.item()
                time_str = time.strftime("%H:%M:%S", time.localtime(ts))
                lines.append(f"[{time_str}] {k} -> {payload}")

            return SessionCommandResult("\n".join(lines))

        if cmd == "/graph":
            return SessionCommandResult(self.brain.graph.summary())

        if cmd == "/correct":
            true_answer = arg.strip()
            if not true_answer:
                return SessionCommandResult("Formato: /correct [corrección de la última respuesta]")
            
            if not self.brain.history:
                return SessionCommandResult("No hay historial previo para corregir.")

            last_turn = self.brain.history[-1]
            last_prompt = last_turn.get("text", "unknown_context")
            
            # Registrar corrección en la memoria semántica
            self.brain.base.remember(f"correct::{last_prompt}", true_answer, tags=("correction",))
            self.brain.base.neuromodulators.boost("serotonina", 0.05)
            
            return SessionCommandResult(f"✓ Corrección registrada para el último turno: '{true_answer}'")
            

        if cmd == "/goals":
            goals = self.brain.base.goals
            if not goals:
                return SessionCommandResult("No hay objetivos existenciales registrados en el núcleo.")
            lines = []
            for gid, info in goals.items():
                t_str = time.strftime("%H:%M:%S", time.localtime(info.get("created", time.time())))
                lines.append(f" - [{gid}] ({t_str}) [{info.get('state').upper()}]: {info.get('goal')}")
            return SessionCommandResult("Objetivos activos:\n" + "\n".join(lines))

        if cmd == "/goal":
            if not arg.strip():
                return SessionCommandResult("Formato incorrecto. Uso: /goal [descripción del objetivo]")
            gid = f"g{len(self.brain.base.goals) + 1}"
            self.brain.base.goals[gid] = {"goal": arg.strip(), "created": time.time(), "state": "active"}
            self.brain.base.neuromodulators.boost("dopamina", 0.03)
            return SessionCommandResult(f"✓ Nuevo objetivo registrado bajo el identificador: {gid}")

        if cmd == "/remember":
            content = arg.strip()
            if ":" not in content:
                return SessionCommandResult("Formato incorrecto. Uso: /remember [clave]: [valor]")
            k, v = content.split(":", 1)
            latent_dummy = torch.zeros(self.brain.cfg.perception_dim, device=self.brain.device, dtype=DTYPE)
            hdc_vec = self.brain._latent_to_hdc(
                torch.zeros(128, device=self.brain.device, dtype=DTYPE), text=v.strip()
            )
            self.brain.remember_observation(f"manual_{k.strip()}", latent_dummy, hdc_vec)
            self.brain.base.remember(k.strip(), v.strip(), tags=("manual",))
            return SessionCommandResult(f"✓ Conocimiento consolidado en bases: {k.strip()}")

        if cmd == "/recall":
            key = arg.strip()
            if not key:
                return SessionCommandResult("Formato: /recall clave")
            value = self.brain.base.recall(key, default="No encontrado")
            if value is None:
                return SessionCommandResult(f"Clave '{key}' no encontrada en el núcleo factual.")
            return SessionCommandResult(f"Fila factual [{key}]: {value}")

        if cmd == "/search":
            query = arg.strip()
            if not query:
                return SessionCommandResult("Formato: /search texto")
            matches = self.brain.base.search_memory(query, topk=5)
            if not matches:
                return SessionCommandResult("Sin coincidencias.")
            lines = [f"{k} ({s:+.3f}) = {v}" for k, v, s in matches]
            return SessionCommandResult("\n".join(lines))

        if cmd == "/consolidate":
            self.brain.consolidate()
            return SessionCommandResult("Consolidación ejecutada.")

        if cmd == "/reset":
            self.brain = p7.BrainCore(device=DEVICE)
            return SessionCommandResult("Cerebro reiniciado.")

        if cmd == "/train":
            data_file = arg.strip()
            if not data_file:
                return SessionCommandResult("Formato: /train (nombre_de_archivo)")
            try:
                t0 = time.time()
                chunks = self.brain.train_on_file(data_file)
                elapsed = time.time() - t0
                return SessionCommandResult(
                    f"✓ Entrenamiento masivo completado con éxito.\n"
                    f" - Bloques procesados: {chunks}\n"
                    f" - Tiempo de cómputo:  {elapsed:.2f} segundos\n"
                    f" - Velocidad media:    {chunks / (elapsed + 1e-8):.2f} bloques/seg"
                )
            except FileNotFoundError:
                return SessionCommandResult(f"✗ Error de lectura: No existe el archivo indicado en '{data_file}'")
            except Exception as exc:
                return SessionCommandResult(f"✗ Error crítico durante el parseo de entrenamiento: {exc}")

        if cmd == "/export_json":
            filename = arg.strip() or "braincore_export.json"
            payload = _make_json_serializable(self.brain.export())
            path = Path(filename)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return SessionCommandResult(f"✓ Dump del estado del cerebro exportado correctamente a: {path.absolute()}")

        return SessionCommandResult(f"Comando desconocido: {cmd}")

    # --------------------------------------------------------
    # Módulo de Detección e Invocación de Herramientas
    # --------------------------------------------------------

    def _parse_and_execute_tool(self, text_response: str, user_input: str) -> Tuple[bool, str, str]:
        """
        Busca si la respuesta del cerebro contiene una solicitud de herramienta.
        Devuelve: (se_ejecutó_tool, nombre_tool, resultado_ejecucion)
        """
        pattern = r"<<<TOOL:\s*({.*?})\s*>>>"
        match = re.search(pattern, text_response, re.DOTALL)

        if not match:
            return False, "", ""

        payload = json.loads(match.group(1))
        tool_name = payload.get("tool")
    
        if tool_name in DANGEROUS_TOOLS:
            confirm = input(f"⚠ El sistema quiere ejecutar '{tool_name}' con {payload.get('args')}. ¿Confirmar? [s/N] ")
            if confirm.strip().lower() != "s":
                return True, tool_name, "Ejecución cancelada por el usuario."

        try:
            raw_json = match.group(1)
            payload = json.loads(raw_json)
            tool_name = payload.get("tool")
            args = payload.get("args", {})

            if tool_name == "escribir_archivo":
                res = self.fs_engine.escribir_archivo(args.get("nombre_archivo"), args.get("contenido", ""))
            elif tool_name == "leer_archivo":
                res = self.fs_engine.leer_archivo(args.get("nombre_archivo"))
            elif tool_name == "borrar_archivo":
                res = self.fs_engine.borrar_archivo(args.get("nombre_archivo"))
            elif tool_name == "listar_archivos":
                res = self.fs_engine.listar_archivos()
            elif tool_name == "ejecutar_python":
                res = self.fs_engine.ejecutar_python(args.get("nombre_archivo"))
            elif tool_name == "think":
                theme = args.get("theme") if args.get("theme") else user_input
                res = self.think_loop(theme)
            else:
                res = f"Error: Herramienta desconocida '{tool_name}'."

            return True, str(tool_name), res

        except Exception as err:
            return True, "error_parser", f"Error parseando llamada a herramienta: {str(err)}"

    def think_loop(self, theme: str, max_iterations: int= 10) -> str:
        LOOP_INSTRUCTIONS = "cuando hayas terminado de pensar, escribe <<<EXIT_THINK_LOOP>>>"
        facts = LOOP_INSTRUCTIONS + theme
        iteration = 0
        while True:
            new_fact = self.brain.think(facts, in_loop=True)
            facts += f"\n{new_fact}"
            iteration += 1

            if "<<<EXIT_THINK_LOOP>>>" in new_fact:
                break

            if iteration >= max_iterations:
                facts += "\n[SISTEMA: Cierre forzado por límite de iteraciones de pensamiento]."
                break

        return self.brain.think(facts)

    def _build_system_prompt(self) -> str:
        """Genera las instrucciones de herramientas formateadas para el modelo."""
        tools_desc = []
        for name, info in AVAILABLE_TOOLS.items():
            args_str = ", ".join([f'"{k}": {v}' for k, v in info["args"].items()])
            tools_desc.append(f"- {name}: {info['description']}\n  Parámetros: {{{args_str}}}")

        formatted_tools = "\n".join(tools_desc)

        return f"""
        [INSTRUCCIONES DE SISTEMA Y HERRAMIENTAS]
        Tienes acceso a las siguientes herramientas para interactuar con el sistema de archivos y ejecutar código:

        {formatted_tools}

        [REGLAS DE INVOCACIÓN]
        1. Si necesitas usar una herramienta para responder al usuario, DEBES incluir la llamada en tu respuesta usando EXACTAMENTE este formato de delimitador:
           <<<TOOL: {{"tool": "NOMBRE_HERRAMIENTA", "args": {{"PARAMETRO": "VALOR"}}}}>>>

        2. Solo puedes emitir UNA llamada a herramienta por turno.
        3. Asegúrate de que el contenido dentro de <<<TOOL: ... >>> sea un JSON totalmente válido.
        4. Si no necesitas usar ninguna herramienta, responde directamente al usuario en texto plano.
        """.strip()

    # --------------------------------------------------------
    # Bucle Principal de Conversación
    # --------------------------------------------------------

    def chat(self, text: str) -> str:
        if text.startswith("/"):
            result = self.handle_command(text)
            return result.text

        system_instruction = self._build_system_prompt()

        formatted_history = []
        for turn in self.brain.history:
            role = getattr(turn, "role", turn.get("role", "unknown") if isinstance(turn, dict) else "unknown")
            t_text = getattr(turn, "text", turn.get("text", "") if isinstance(turn, dict) else "")
            formatted_history.append(f"{role.upper()}: {t_text}")
        
        history_str = "\n".join(formatted_history) if formatted_history else "Sin historial previo."

        input_prompt = (
            f"{system_instruction}\n\n"
            f"[HISTORIAL]\n"
            f"{history_str}\n"
            f"[USER INPUT]\n"
            f"{text}"
        )

        # 1. Primera pasada de pensamiento en BrainCore
        first_thought = self.brain.think(input_prompt)

        # 2. Verificar si BrainCore decidió usar el sistema de archivos
        has_tool, tool_name, tool_output = self._parse_and_execute_tool(first_thought, input_prompt)

        if not has_tool:
            return first_thought

        # 3. Si ejecutó una herramienta, consolidamos la observación en la memoria del cerebro
        obs_key = f"tool_result_{tool_name}_{int(time.time())}"
        payload_text = f"Herramienta [{tool_name}] ejecutada. Resultado:\n{tool_output}"

        latent_dummy = torch.zeros(self.brain.cfg.perception_dim, device=self.brain.device, dtype=DTYPE)
        hdc_vec = self.brain._latent_to_hdc(
            torch.zeros(128, device=self.brain.device, dtype=DTYPE), text=payload_text
        )

        self.brain.remember_observation(obs_key, latent_dummy, hdc_vec)

        # 4. Feedback loop: Le pasamos al cerebro el resultado técnico para que genere su respuesta final
        feedback_prompt = (
            f"[SISTEMA DE EJECUCIÓN - {tool_name.upper()}]\n"
            f"Resultado obtenido del disco/subprocess:\n{tool_output}\n\n"
            f"Con base en esta salida, responde al usuario final."
        )

        final_thought = self.brain.think(feedback_prompt)

        # Limpiamos los tags de tool en la respuesta que verá el usuario final
        clean_first = re.sub(r"<<<TOOL:.*?>>>", "", first_thought, flags=re.DOTALL).strip()

        return f"{clean_first}\n\n[Ejecución de disco]: {tool_output}\n\n{final_thought}"

    def run(self) -> None:
        print("IAG Modular V4 - Agente completo (PyTorch Native)")
        print("Escribe /help para ver comandos.\n")

        while self.running:
            try:
                user = input("Tu> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSaliendo.")
                break

            if not user:
                continue

            result = self.chat(user)
            print("IAG>", result)

            if user.strip().lower() in {"/exit", "exit", "quit"}:
                break


# ============================================================
# Demo / Arranque
# ============================================================

def build_agent() -> AgentCLI:
    brain = p7.BrainCore(device=DEVICE)
    return AgentCLI(brain=brain)


if __name__ == "__main__":
    agent = build_agent()
    agent.run()