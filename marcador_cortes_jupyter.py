from __future__ import annotations

import html as html_lib
import json
import mimetypes
import re
import subprocess
import threading
import uuid
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from IPython.display import HTML, display

__version__ = "6.1"


def _caminho_padrao_transcricao(arquivo: str | Path) -> Path:
    arquivo = Path(arquivo).resolve()
    return arquivo.with_name(f"{arquivo.stem}_transcricao.json")


def _texto_palavras(palavras: list[dict]) -> str:
    return "".join(item["word"] for item in palavras).strip()


def _agrupar_palavras_em_blocos(
    palavras: list[dict],
    pausa_bloco: float = 0.7,
    duracao_max_bloco: float = 10.0,
    caracteres_max_bloco: int = 150,
) -> list[dict]:
    """Agrupa palavras em blocos naturais, priorizando pausas no áudio."""
    if not palavras:
        return []

    blocos: list[dict] = []
    atual: list[dict] = []

    def concluir() -> None:
        nonlocal atual

        if not atual:
            return

        blocos.append(
            {
                "start": atual[0]["start"],
                "end": atual[-1]["end"],
                "text": _texto_palavras(atual),
                "words": atual,
            }
        )
        atual = []

    for palavra in palavras:
        if atual:
            anterior = atual[-1]
            pausa = max(0.0, palavra["start"] - anterior["end"])
            duracao = palavra["end"] - atual[0]["start"]
            tamanho = len(_texto_palavras(atual)) + len(palavra["word"])
            fim_frase = bool(
                re.search(r'[.!?…][\"\'”’)]*$', anterior["word"].strip())
            )

            quebrar = (
                pausa >= pausa_bloco
                or duracao >= duracao_max_bloco
                or tamanho >= caracteres_max_bloco
                or (fim_frase and pausa >= 0.25 and duracao >= 2.0)
            )

            if quebrar:
                concluir()

        atual.append(palavra)

    concluir()

    for indice, bloco in enumerate(blocos):
        bloco["id"] = indice

    return blocos


def transcrever_video(
    arquivo: str | Path,
    saida: str | Path | None = None,
    modelo: str = "small",
    idioma: str | None = "pt",
    dispositivo: str = "cpu",
    compute_type: str = "int8",
    pausa_bloco: float = 0.7,
    duracao_max_bloco: float = 10.0,
    caracteres_max_bloco: int = 150,
    pausa_vad_ms: int = 500,
    sobrescrever: bool = False,
) -> Path:
    """
    Transcreve o vídeo com timestamps por palavra e salva um JSON.

    A transcrição é agrupada em blocos definidos principalmente pelas pausas.
    O arquivo original não é alterado.
    """
    arquivo = Path(arquivo).resolve()

    if not arquivo.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {arquivo}")

    saida = (
        _caminho_padrao_transcricao(arquivo)
        if saida is None
        else Path(saida).resolve()
    )

    if saida.exists() and not sobrescrever:
        print(f"Transcrição já existente: {saida.name}")
        return saida

    try:
        from faster_whisper import WhisperModel
    except ImportError as erro:
        raise ImportError(
            "O faster-whisper não está instalado. Execute em uma célula: "
            "%pip install faster-whisper"
        ) from erro

    print(f"Carregando o modelo de transcrição: {modelo}")
    modelo_whisper = WhisperModel(
        modelo,
        device=dispositivo,
        compute_type=compute_type,
    )

    print(f"Transcrevendo: {arquivo.name}")
    segmentos_iterador, info = modelo_whisper.transcribe(
        str(arquivo),
        language=idioma,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": int(pausa_vad_ms),
        },
        condition_on_previous_text=True,
    )

    palavras: list[dict] = []
    segmentos: list[dict] = []

    for segmento in segmentos_iterador:
        palavras_segmento: list[dict] = []

        for palavra in segmento.words or []:
            if palavra.start is None or palavra.end is None:
                continue

            item = {
                "word": palavra.word,
                "start": round(float(palavra.start), 3),
                "end": round(float(palavra.end), 3),
                "probability": round(
                    float(getattr(palavra, "probability", 0.0) or 0.0),
                    4,
                ),
            }
            palavras.append(item)
            palavras_segmento.append(item)

        segmentos.append(
            {
                "start": round(float(segmento.start), 3),
                "end": round(float(segmento.end), 3),
                "text": segmento.text.strip(),
                "words": palavras_segmento,
            }
        )

    blocos = _agrupar_palavras_em_blocos(
        palavras,
        pausa_bloco=pausa_bloco,
        duracao_max_bloco=duracao_max_bloco,
        caracteres_max_bloco=caracteres_max_bloco,
    )

    dados = {
        "version": 1,
        "source": arquivo.name,
        "model": modelo,
        "language": getattr(info, "language", idioma),
        "language_probability": round(
            float(getattr(info, "language_probability", 0.0) or 0.0),
            4,
        ),
        "duration": round(
            float(getattr(info, "duration", 0.0) or 0.0),
            3,
        ),
        "settings": {
            "pause_block_seconds": pausa_bloco,
            "max_block_seconds": duracao_max_bloco,
            "max_block_characters": caracteres_max_bloco,
            "vad_min_silence_ms": pausa_vad_ms,
        },
        "blocks": blocos,
        "segments": segmentos,
    }

    saida.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Transcrição salva: {saida.name}")
    print(f"Blocos: {len(blocos)} | Palavras: {len(palavras)}")
    return saida


def carregar_transcricao(caminho: str | Path) -> dict:
    caminho = Path(caminho).resolve()

    if not caminho.exists():
        raise FileNotFoundError(f"Transcrição não encontrada: {caminho}")

    dados = json.loads(caminho.read_text(encoding="utf-8"))

    if not isinstance(dados.get("blocks"), list):
        raise ValueError("O JSON não contém uma lista válida de blocos.")

    return dados


_SERVIDORES: list[ThreadingHTTPServer] = []


def _executar_ffprobe(arquivo: Path) -> dict:
    comando = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,pix_fmt,avg_frame_rate,r_frame_rate:"
        "format=duration,format_name",
        "-of", "json",
        str(arquivo),
    ]

    resultado = subprocess.run(
        comando,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(resultado.stdout)


def informacoes_video(arquivo: str | Path) -> dict:
    arquivo = Path(arquivo).resolve()

    if not arquivo.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {arquivo}")

    dados = _executar_ffprobe(arquivo)
    streams = dados.get("streams", [])

    video = next(
        (item for item in streams if item.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (item for item in streams if item.get("codec_type") == "audio"),
        None,
    )

    if video is None:
        raise ValueError("O arquivo não possui uma faixa de vídeo.")

    taxa = video.get("avg_frame_rate") or video.get("r_frame_rate")
    fps = float(Fraction(taxa)) if taxa and taxa != "0/0" else 30.0

    return {
        "arquivo": arquivo,
        "duracao": float(dados["format"]["duration"]),
        "formato": dados["format"].get("format_name", ""),
        "codec_video": video.get("codec_name", ""),
        "pixel_format": video.get("pix_fmt", ""),
        "codec_audio": audio.get("codec_name", "") if audio else None,
        "possui_audio": audio is not None,
        "fps": fps,
    }


def _compativel_com_navegador(info: dict) -> bool:
    extensao = info["arquivo"].suffix.lower()
    codec_video = info["codec_video"]
    pixel_format = info["pixel_format"]
    codec_audio = info["codec_audio"]

    recipiente_ok = extensao in {".mp4", ".m4v", ".mov"}
    video_ok = codec_video == "h264" and pixel_format in {
        "yuv420p",
        "yuvj420p",
    }
    audio_ok = codec_audio in {None, "aac", "mp3"}

    return recipiente_ok and video_ok and audio_ok


def criar_preview_web(
    arquivo: str | Path,
    saida: str | Path | None = None,
    sobrescrever: bool = False,
) -> Path:
    """
    Cria uma cópia apenas para visualização no navegador.

    O arquivo original não é alterado. Os cortes continuam usando os tempos
    marcados e podem ser aplicados ao vídeo original.
    """
    entrada = Path(arquivo).resolve()

    if not entrada.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {entrada}")

    if saida is None:
        saida = entrada.with_name(f"{entrada.stem}_preview_web.mp4")
    else:
        saida = Path(saida).resolve()

    if saida.exists() and not sobrescrever:
        return saida

    print("Criando uma cópia compatível com o navegador...")
    print(f"Saída: {saida.name}")

    comando = [
        "ffmpeg",
        "-y",
        "-i", str(entrada),
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(saida),
    ]

    subprocess.run(comando, check=True)

    print("Cópia de visualização criada.")
    return saida

# -----------------------------------------------------------------------------
# Recursos de projeto / legenda adicionados na v6
# -----------------------------------------------------------------------------


def _caminho_padrao_projeto(arquivo: str | Path) -> Path:
    arquivo = Path(arquivo).resolve()
    return arquivo.with_name(f"{arquivo.stem}_projeto_editor.json")


def _gravar_json_atomico(caminho: str | Path, dados: dict) -> Path:
    """Grava JSON por arquivo temporário para reduzir risco de corrupção."""
    caminho = Path(caminho).resolve()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f".{caminho.name}.{uuid.uuid4().hex}.tmp")
    temporario.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporario.replace(caminho)
    return caminho


def carregar_projeto(caminho: str | Path) -> dict:
    caminho = Path(caminho).resolve()
    if not caminho.exists():
        raise FileNotFoundError(f"Projeto não encontrado: {caminho}")
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    if not isinstance(dados, dict):
        raise ValueError("O projeto precisa ser um objeto JSON.")
    if not isinstance(dados.get("operations", []), list):
        raise ValueError("O projeto não contém uma lista válida de operações.")
    return dados


def salvar_projeto(caminho: str | Path, dados: dict) -> Path:
    if not isinstance(dados, dict):
        raise TypeError("dados precisa ser um dicionário.")
    if not isinstance(dados.get("operations", []), list):
        raise ValueError("dados['operations'] precisa ser uma lista.")
    dados = dict(dados)
    dados.setdefault("version", 1)
    dados.setdefault("time_reference", "source")
    return _gravar_json_atomico(caminho, dados)


def extrair_cortes_projeto(projeto: dict) -> list[tuple[float, float]]:
    """Extrai somente cortes de um projeto v6, para compatibilidade com a v5."""
    cortes: list[tuple[float, float]] = []
    for operacao in projeto.get("operations", []):
        if not operacao.get("enabled", True):
            continue
        if operacao.get("type") != "cut":
            continue
        inicio = float(operacao["start"])
        fim = float(operacao["end"])
        if fim > inicio:
            cortes.append((inicio, fim))
    cortes.sort(key=lambda item: item[0])
    return cortes


def _formatar_srt_tempo(segundos: float) -> str:
    milissegundos = max(0, round(float(segundos) * 1000))
    horas, resto = divmod(milissegundos, 3_600_000)
    minutos, resto = divmod(resto, 60_000)
    segundos_int, ms = divmod(resto, 1000)
    return f"{horas:02d}:{minutos:02d}:{segundos_int:02d},{ms:03d}"


def _formatar_vtt_tempo(segundos: float) -> str:
    return _formatar_srt_tempo(segundos).replace(",", ".")


def exportar_legendas(
    transcricao: str | Path | dict,
    saida: str | Path,
    formato: str | None = None,
) -> Path:
    """Exporta os blocos atuais da transcrição para SRT ou WebVTT."""
    if isinstance(transcricao, (str, Path)):
        dados = carregar_transcricao(transcricao)
    else:
        dados = transcricao

    saida = Path(saida).resolve()
    formato = (formato or saida.suffix.lstrip(".") or "srt").lower()
    if formato not in {"srt", "vtt"}:
        raise ValueError("formato deve ser 'srt' ou 'vtt'.")

    linhas: list[str] = []
    if formato == "vtt":
        linhas.extend(["WEBVTT", ""])

    for indice, bloco in enumerate(dados.get("blocks", []), start=1):
        palavras = bloco.get("words") or []
        texto = _texto_palavras(palavras) if palavras else str(bloco.get("text", "")).strip()
        if not texto:
            continue
        inicio = float(bloco.get("start", 0.0))
        fim = float(bloco.get("end", inicio))
        if formato == "srt":
            linhas.append(str(indice))
            linhas.append(f"{_formatar_srt_tempo(inicio)} --> {_formatar_srt_tempo(fim)}")
        else:
            linhas.append(f"{_formatar_vtt_tempo(inicio)} --> {_formatar_vtt_tempo(fim)}")
        linhas.extend([texto, ""])

    saida.write_text("\n".join(linhas), encoding="utf-8")
    return saida


def corrigir_palavra_transcricao(
    caminho: str | Path,
    bloco: int,
    palavra: int,
    novo_texto: str,
    salvar_em: str | Path | None = None,
) -> Path:
    """Corrige uma palavra preservando start/end; útil também fora da interface."""
    caminho = Path(caminho).resolve()
    dados = carregar_transcricao(caminho)
    alvo = dados["blocks"][bloco]["words"][palavra]
    antiga = str(alvo.get("word", ""))
    prefixo = antiga[: len(antiga) - len(antiga.lstrip())]
    alvo["word"] = prefixo + str(novo_texto).strip()
    dados["blocks"][bloco]["text"] = _texto_palavras(dados["blocks"][bloco]["words"])

    # Mantém a cópia em segments coerente, quando encontrada pelo timestamp.
    inicio = float(alvo.get("start", -1))
    fim = float(alvo.get("end", -1))
    for segmento in dados.get("segments", []):
        alterou = False
        for item in segmento.get("words", []):
            if abs(float(item.get("start", -2)) - inicio) < 1e-6 and abs(float(item.get("end", -2)) - fim) < 1e-6:
                item["word"] = alvo["word"]
                alterou = True
                break
        if alterou:
            segmento["text"] = _texto_palavras(segmento.get("words", []))
            break

    dados["version"] = max(int(dados.get("version", 1) or 1), 2)
    destino = caminho if salvar_em is None else Path(salvar_em).resolve()
    return _gravar_json_atomico(destino, dados)


class _VideoHandler(BaseHTTPRequestHandler):
    raiz: Path
    caminho_transcricao: Path | None = None
    caminho_projeto: Path | None = None
    bloqueio_api = threading.Lock()

    def log_message(self, formato: str, *args) -> None:
        return

    def _cabecalhos_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _responder_json(self, codigo: int, dados: dict) -> None:
        corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self._cabecalhos_cors()
        self.end_headers()
        try:
            self.wfile.write(corpo)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _ler_json(self) -> dict:
        tamanho = int(self.headers.get("Content-Length", "0") or "0")
        if tamanho <= 0:
            raise ValueError("Corpo vazio.")
        if tamanho > 20 * 1024 * 1024:
            raise ValueError("Corpo JSON grande demais.")
        corpo = self.rfile.read(tamanho)
        dados = json.loads(corpo.decode("utf-8"))
        if not isinstance(dados, dict):
            raise ValueError("O corpo precisa ser um objeto JSON.")
        return dados

    def _resolver_arquivo(self) -> Path | None:
        caminho_url = unquote(urlparse(self.path).path).lstrip("/")
        candidato = (self.raiz / caminho_url).resolve()
        try:
            candidato.relative_to(self.raiz)
        except ValueError:
            return None
        if not candidato.is_file():
            return None
        return candidato

    def _cabecalhos_comuns(
        self,
        arquivo: Path,
        tamanho: int,
        codigo: int,
        inicio: int = 0,
        fim: int | None = None,
    ) -> None:
        if fim is None:
            fim = tamanho - 1
        tipo, _ = mimetypes.guess_type(str(arquivo))
        tipo = tipo or "application/octet-stream"
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Accept-Ranges", "bytes")
        self._cabecalhos_cors()
        self.send_header("Cache-Control", "no-cache")
        if codigo == 206:
            self.send_header("Content-Range", f"bytes {inicio}-{fim}/{tamanho}")
        self.send_header("Content-Length", str(fim - inicio + 1))
        self.end_headers()

    def _enviar(self, somente_cabecalho: bool = False) -> None:
        arquivo = self._resolver_arquivo()
        if arquivo is None:
            self.send_error(404, "Arquivo não encontrado")
            return

        tamanho = arquivo.stat().st_size
        cabecalho_range = self.headers.get("Range")
        inicio = 0
        fim = tamanho - 1
        codigo = 200

        if cabecalho_range:
            correspondencia = re.match(r"bytes=(\d*)-(\d*)", cabecalho_range)
            if correspondencia:
                inicio_txt, fim_txt = correspondencia.groups()
                if inicio_txt:
                    inicio = int(inicio_txt)
                if fim_txt:
                    fim = min(int(fim_txt), tamanho - 1)
                if not inicio_txt and fim_txt:
                    quantidade = int(fim_txt)
                    inicio = max(0, tamanho - quantidade)
                    fim = tamanho - 1
                if inicio >= tamanho or inicio > fim:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{tamanho}")
                    self._cabecalhos_cors()
                    self.end_headers()
                    return
                codigo = 206

        self._cabecalhos_comuns(
            arquivo=arquivo,
            tamanho=tamanho,
            codigo=codigo,
            inicio=inicio,
            fim=fim,
        )
        if somente_cabecalho:
            return

        restante = fim - inicio + 1
        with arquivo.open("rb") as fluxo:
            fluxo.seek(inicio)
            while restante > 0:
                bloco = fluxo.read(min(1024 * 1024, restante))
                if not bloco:
                    break
                try:
                    self.wfile.write(bloco)
                except (BrokenPipeError, ConnectionResetError):
                    break
                except OSError as erro:
                    if getattr(erro, "winerror", None) in {10053, 10054}:
                        break
                    raise
                restante -= len(bloco)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cabecalhos_cors()
        self.end_headers()

    def do_HEAD(self) -> None:
        self._enviar(somente_cabecalho=True)

    def do_GET(self) -> None:
        if urlparse(self.path).path.startswith("/__editor_api__/"):
            self._responder_json(405, {"ok": False, "erro": "Use POST."})
            return
        self._enviar(somente_cabecalho=False)

    def do_POST(self) -> None:
        rota = urlparse(self.path).path
        try:
            dados = self._ler_json()
            with self.bloqueio_api:
                if rota == "/__editor_api__/transcricao":
                    if self.caminho_transcricao is None:
                        self._responder_json(400, {"ok": False, "erro": "Transcrição sem caminho de salvamento."})
                        return
                    if not isinstance(dados.get("blocks"), list):
                        raise ValueError("JSON de transcrição sem lista 'blocks'.")
                    dados["version"] = max(int(dados.get("version", 1) or 1), 2)
                    _gravar_json_atomico(self.caminho_transcricao, dados)
                    self._responder_json(200, {"ok": True, "arquivo": self.caminho_transcricao.name})
                    return

                if rota == "/__editor_api__/projeto":
                    if self.caminho_projeto is None:
                        self._responder_json(400, {"ok": False, "erro": "Projeto sem caminho de salvamento."})
                        return
                    salvar_projeto(self.caminho_projeto, dados)
                    self._responder_json(200, {"ok": True, "arquivo": self.caminho_projeto.name})
                    return

            self._responder_json(404, {"ok": False, "erro": "Rota desconhecida."})
        except Exception as erro:
            self._responder_json(400, {"ok": False, "erro": str(erro)})


def _iniciar_servidor(
    arquivo: Path,
    caminho_transcricao: Path | None = None,
    caminho_projeto: Path | None = None,
) -> tuple[str, str]:
    raiz = arquivo.parent.resolve()
    classe_handler = type(
        f"VideoHandler_{uuid.uuid4().hex}",
        (_VideoHandler,),
        {
            "raiz": raiz,
            "caminho_transcricao": caminho_transcricao,
            "caminho_projeto": caminho_projeto,
            "bloqueio_api": threading.Lock(),
        },
    )
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), classe_handler)
    thread = threading.Thread(target=servidor.serve_forever, daemon=True)
    thread.start()
    _SERVIDORES.append(servidor)
    porta = servidor.server_address[1]
    base = f"http://127.0.0.1:{porta}"
    nome = quote(arquivo.name)
    return f"{base}/{nome}", base


def encerrar_servidores_preview() -> None:
    while _SERVIDORES:
        servidor = _SERVIDORES.pop()
        servidor.shutdown()
        servidor.server_close()


def marcador_cortes(
    arquivo: str | Path,
    largura: int = 1000,
    preparar_preview: bool | str = "auto",
    transcricao: bool | str | Path = "auto",
    modelo_transcricao: str = "small",
    idioma_transcricao: str | None = "pt",
    dispositivo_transcricao: str = "cpu",
    compute_type_transcricao: str = "int8",
    pausa_bloco: float = 0.7,
    projeto: bool | str | Path = "auto",
    legendas: bool = True,
) -> None:
    """
    Marcador v6 para Jupyter.

    Mantém os recursos da v5 e acrescenta:
      - edição de palavras da transcrição preservando timestamps;
      - salvamento das correções no JSON da transcrição;
      - legenda sincronizada sobre o player + exportação SRT/VTT;
      - projeto JSON de operações não destrutivas;
      - operações adicionais registráveis e com prévia visual (mute, blur, texto, imagem,
        vídeo sobreposto, inserção/substituição, zoom, crop e tarja);
      - caixa visual arrastável/redimensionável com coordenadas normalizadas.

    Importante: o marcador REGISTRA as operações adicionais. A lista de cortes
    legada continua sendo produzida para o pipeline de corte já existente.
    Renderização das operações adicionais requer um pipeline FFmpeg compatível.

    projeto:
        "auto" -> carrega <video>_projeto_editor.json se existir e permite salvar;
        True   -> mesmo comportamento de "auto";
        False  -> não usa arquivo de projeto no servidor (download continua possível);
        caminho -> usa o JSON informado.
    """
    original = Path(arquivo).resolve()
    info_original = informacoes_video(original)

    dados_transcricao: dict | None = None
    caminho_transcricao: Path | None = None

    if transcricao is True:
        caminho_transcricao = transcrever_video(
            original,
            modelo=modelo_transcricao,
            idioma=idioma_transcricao,
            dispositivo=dispositivo_transcricao,
            compute_type=compute_type_transcricao,
            pausa_bloco=pausa_bloco,
        )
        dados_transcricao = carregar_transcricao(caminho_transcricao)
    elif transcricao == "auto":
        caminho_transcricao = _caminho_padrao_transcricao(original)
        if caminho_transcricao.exists():
            dados_transcricao = carregar_transcricao(caminho_transcricao)
    elif transcricao not in {False, None}:
        caminho_transcricao = Path(transcricao).resolve()
        dados_transcricao = carregar_transcricao(caminho_transcricao)

    caminho_projeto: Path | None = None
    dados_projeto: dict | None = None
    if projeto is not False and projeto is not None:
        if projeto in {True, "auto"}:
            caminho_projeto = _caminho_padrao_projeto(original)
        else:
            caminho_projeto = Path(projeto).resolve()
        if caminho_projeto.exists():
            dados_projeto = carregar_projeto(caminho_projeto)

    if dados_projeto is None:
        dados_projeto = {
            "version": 1,
            "source": original.name,
            "time_reference": "source",
            "operations": [],
        }

    if preparar_preview not in {True, False, "auto"}:
        raise ValueError('preparar_preview deve ser True, False ou "auto".')

    precisa_converter = (
        preparar_preview is True
        or (preparar_preview == "auto" and not _compativel_com_navegador(info_original))
    )
    exibido = criar_preview_web(original) if precisa_converter else original
    info_exibido = informacoes_video(exibido)
    origem, api_base = _iniciar_servidor(
        exibido,
        caminho_transcricao=caminho_transcricao if dados_transcricao is not None else None,
        caminho_projeto=caminho_projeto,
    )
    identificador = "mc_" + uuid.uuid4().hex[:10]

    resumo_codec = f'{info_original["codec_video"]} / {info_original["pixel_format"]}'
    if info_original["codec_audio"]:
        resumo_codec += f' / áudio {info_original["codec_audio"]}'

    aviso_preview = (
        f'<div class="aviso">Visualização por cópia compatível: '
        f'<strong>{html_lib.escape(exibido.name)}</strong>. '
        f'O original não foi alterado.</div>'
        if exibido != original else ""
    )

    aviso_v6 = (
        '<div class="info-v6"><strong>v6.1:</strong> interface compacta, prévias visuais e cortes compatíveis com a v5. '
        'Imagem, texto, vídeo sobreposto, desfoque e tarja possuem prévia no navegador. As operações são salvas no projeto JSON e ainda precisam de um pipeline '
        'FFmpeg compatível para serem renderizadas.</div>'
    )

    modelo = r"""
<div id="__ID__" class="marcador-cortes-v61">
<style>
#__ID__ {max-width:__WIDTH__px;padding:8px;border:1px solid #bbb;border-radius:8px;font-family:Arial,sans-serif;font-size:11.5px;line-height:1.25}
#__ID__ * {box-sizing:border-box}
#__ID__ .aviso, #__ID__ .info-v6, #__ID__ .status {margin:5px 0;padding:5px 7px;border:1px solid #d0a000;border-radius:5px;font-size:10.5px}
#__ID__ .info-v6 {border-color:#6699cc;background:rgba(80,130,220,.07)}
#__ID__ .status {border-color:#aaa;display:none}
#__ID__ .erro {display:none;margin:5px 0;padding:5px 7px;border:1px solid #b00020;border-radius:5px}
#__ID__ .palco-video {position:relative;width:100%;background:#000;overflow:hidden;line-height:0;user-select:none}
#__ID__ .video-principal {display:block;width:100%;max-height:55vh;background:#000}
#__ID__ .legenda-video {position:absolute;left:8%;right:8%;bottom:4%;z-index:20;text-align:center;color:white;font-weight:700;font-size:clamp(13px,1.65vw,22px);line-height:1.2;text-shadow:0 2px 4px #000,0 0 3px #000;pointer-events:none}
#__ID__ .legenda-video span {display:inline;background:rgba(0,0,0,.58);padding:2px 5px;border-radius:3px;box-decoration-break:clone;-webkit-box-decoration-break:clone}
#__ID__ .legenda-video.topo {top:4%;bottom:auto}
#__ID__ .legenda-video.meio {top:50%;bottom:auto;transform:translateY(-50%)}
#__ID__ .camada-operacoes {position:absolute;inset:0;z-index:9;pointer-events:none;overflow:hidden}
#__ID__ .preview-op {position:absolute;overflow:hidden;line-height:1.2;pointer-events:none}
#__ID__ .preview-op img, #__ID__ .preview-op video {width:100%;height:100%;display:block;background:transparent}
#__ID__ .preview-op .preview-texto {width:100%;height:100%;display:flex;align-items:center;justify-content:center;text-align:center;color:#fff;font-weight:700;font-size:clamp(11px,1.4vw,22px);text-shadow:0 1px 3px #000;padding:4px;white-space:pre-wrap;overflow:hidden}
#__ID__ .preview-op.preview-blur {background:rgba(255,255,255,.025);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
#__ID__ .preview-op.preview-tarja {background:#000}
#__ID__ .preview-op.preview-guia {border:1px dashed rgba(255,255,255,.75);background:rgba(255,255,255,.04)}
#__ID__ .preview-op .mini-rotulo {position:absolute;left:2px;top:2px;padding:1px 3px;border-radius:2px;background:rgba(0,0,0,.62);color:#fff;font:9px Arial,sans-serif;line-height:1.2}
#__ID__ .caixa-regiao {position:absolute;z-index:30;border:1.5px solid #ffd54f;background:rgba(255,213,79,.06);min-width:30px;min-height:24px;cursor:move;line-height:1.2;touch-action:none;overflow:visible}
#__ID__ .caixa-regiao[hidden] {display:none}
#__ID__ .preview-edicao {position:absolute;inset:0;overflow:hidden;pointer-events:none;background:transparent}
#__ID__ .preview-edicao img, #__ID__ .preview-edicao video {display:block;width:100%;height:100%;background:transparent}
#__ID__ .preview-edicao .texto-preview {width:100%;height:100%;display:flex;align-items:center;justify-content:center;text-align:center;padding:4px;color:#fff;font-weight:700;font-size:clamp(11px,1.4vw,22px);text-shadow:0 1px 3px #000;white-space:pre-wrap;overflow:hidden}
#__ID__ .preview-edicao .erro-preview {display:flex;width:100%;height:100%;align-items:center;justify-content:center;text-align:center;padding:4px;color:#ffe082;background:rgba(0,0,0,.55);font-size:9px;line-height:1.2}
#__ID__ .caixa-regiao .rotulo-regiao {position:absolute;left:0;top:0;transform:translateY(-100%);background:#ffd54f;color:#111;padding:1px 4px;font:9px Arial,sans-serif;white-space:nowrap;line-height:1.25}
#__ID__ .caixa-regiao .alca {position:absolute;right:-5px;bottom:-5px;width:10px;height:10px;border:1px solid #111;background:#ffd54f;cursor:nwse-resize}
#__ID__ .tempo {margin:5px 0;font-family:Consolas,monospace;font-size:14px;font-weight:bold}
#__ID__ .grupo-controles, #__ID__ .painel {margin:5px 0}
#__ID__ .painel {padding:6px;border:1px solid #bbb;border-radius:6px}
#__ID__ .rotulo-controles {display:block;margin-bottom:2px;font-size:10px;font-weight:bold;opacity:.72}
#__ID__ .botoes {display:flex;flex-wrap:wrap;gap:3px;margin:3px 0}
#__ID__ button {display:inline-flex;align-items:center;gap:3px;padding:3px 5px;min-height:24px;cursor:pointer;white-space:nowrap;font-size:10.5px}
#__ID__ button.ativo {outline:1.5px solid currentColor;outline-offset:1px}
#__ID__ button:disabled {opacity:.5;cursor:not-allowed}
#__ID__ kbd {padding:0 2px;border:1px solid #aaa;border-radius:2px;font-family:Consolas,monospace;font-size:7.5px;opacity:.65}
#__ID__ .marcas {margin:4px 0;font-family:Consolas,monospace;font-size:10.5px}
#__ID__ table {width:100%;border-collapse:collapse;margin-top:5px;font-size:10.5px}
#__ID__ th, #__ID__ td {border:1px solid #ccc;padding:3px 4px;text-align:left;vertical-align:top}
#__ID__ textarea {width:100%;min-height:74px;margin-top:4px;font-family:Consolas,monospace;font-size:9.5px}
#__ID__ input[type=text], #__ID__ input[type=number], #__ID__ select {padding:3px 4px;min-height:25px;max-width:100%;font-size:10.5px}
#__ID__ .grade-campos {display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:5px;margin:5px 0}
#__ID__ .campo {display:flex;flex-direction:column;gap:2px}
#__ID__ .campo label {font-size:9.5px;font-weight:bold;opacity:.72}
#__ID__ .painel-transcricao {margin:5px 0;border:1px solid #bbb;border-radius:6px;overflow:hidden}
#__ID__ .cabecalho-transcricao {display:flex;flex-wrap:wrap;justify-content:space-between;gap:5px;padding:4px 6px;font-size:10.5px;font-weight:bold;background:rgba(127,127,127,.10)}
#__ID__ .transcricao {max-height:190px;overflow-y:auto;padding:3px;scroll-behavior:smooth;font-size:10.5px}
#__ID__ .bloco-transcricao {display:grid;grid-template-columns:58px 1fr;gap:4px;padding:3px 4px;border-radius:4px}
#__ID__ .bloco-transcricao.ativo {background:rgba(80,130,220,.15)}
#__ID__ .tempo-transcricao {align-self:start;padding:1px 2px;border:0;background:transparent;font-family:Consolas,monospace;font-size:9px;text-decoration:underline;min-height:auto}
#__ID__ .texto-transcricao {line-height:1.45}
#__ID__ .palavra-transcricao {cursor:pointer;border-radius:2px}
#__ID__ .palavra-transcricao:hover {text-decoration:underline;background:rgba(127,127,127,.14)}
#__ID__ .palavra-transcricao.ativa {background:rgba(255,190,0,.32);outline:1px solid rgba(150,100,0,.35)}
#__ID__ .palavra-transcricao.editada {text-decoration:underline dotted;text-decoration-thickness:1px}
#__ID__ .ajuda {font-size:9.5px;opacity:.75;margin-top:3px}
#__ID__ .tipo-op {font-weight:bold}
#__ID__ .badge {display:inline-block;padding:1px 4px;border:1px solid #aaa;border-radius:999px;font-size:8.5px}
#__ID__ details {margin:4px 0}
#__ID__ summary {cursor:pointer;font-weight:bold;font-size:10.5px}
</style>

__PREVIEW_NOTICE__
__V6_NOTICE__
<div>Arquivo original: <strong>__ORIGINAL__</strong><br>Codec original: <code>__CODEC__</code></div>
<div id="__ID___erro" class="erro">O navegador não conseguiu reproduzir este arquivo. Execute novamente com <code>preparar_preview=True</code>.</div>
<div id="__ID___status" class="status"></div>

<div id="__ID___palco" class="palco-video">
  <video id="__ID___video" class="video-principal" controls preload="metadata"><source src="__SRC__" type="video/mp4">Seu navegador não conseguiu abrir o vídeo.</video>
  <div id="__ID___camada_operacoes" class="camada-operacoes"></div>
  <div id="__ID___legenda" class="legenda-video" __SUBTITLE_HIDDEN__><span></span></div>
  <div id="__ID___regiao" class="caixa-regiao" hidden>
    <div id="__ID___preview_regiao" class="preview-edicao"></div>
    <div id="__ID___rotulo_regiao" class="rotulo-regiao">região</div><div class="alca"></div>
  </div>
</div>

<div class="tempo"><span id="__ID___tempo">00:00:00.000</span> · velocidade: <span id="__ID___velocidade">1×</span> · FPS: __FPS_TEXT__</div>

<div id="__ID___painel_transcricao" class="painel-transcricao" hidden>
  <div class="cabecalho-transcricao">
    <span>Transcrição sincronizada · clique para ir ao tempo · <strong>duplo clique para corrigir</strong></span>
    <span id="__ID___estado_transcricao">sem alterações</span>
  </div>
  <div id="__ID___transcricao" class="transcricao"></div>
  <div class="botoes" style="padding:6px">
    <button data-acao="salvar-transcricao">Salvar correções no JSON</button>
    <button data-acao="baixar-transcricao">Baixar transcrição JSON</button>
    <button data-acao="baixar-srt">Baixar SRT</button>
    <button data-acao="baixar-vtt">Baixar VTT</button>
    <button data-acao="toggle-legenda">Mostrar/ocultar legenda</button>
    <select id="__ID___posicao_legenda" title="Posição da legenda"><option value="baixo">Legenda embaixo</option><option value="meio">Legenda no meio</option><option value="topo">Legenda no topo</option></select>
  </div>
</div>

<div class="grupo-controles"><span class="rotulo-controles">Reprodução e deslocamento</span><div class="botoes">
<button data-acao="play"><span>Reproduzir/Pausar</span><kbd>Espaço</kbd></button>
<button data-passo="-10"><span>−10 s</span><kbd>J</kbd></button><button data-passo="-1"><span>−1 s</span><kbd>←</kbd></button><button data-passo="-0.1"><span>−0,1 s</span><kbd>⇧←</kbd></button><button data-passo="-0.01"><span>−0,01 s</span><kbd>Ctrl←</kbd></button><button data-passo="-0.001"><span>−0,001 s</span><kbd>Ctrl⇧←</kbd></button>
<button data-passo="0.001"><span>+0,001 s</span><kbd>Ctrl⇧→</kbd></button><button data-passo="0.01"><span>+0,01 s</span><kbd>Ctrl→</kbd></button><button data-passo="0.1"><span>+0,1 s</span><kbd>⇧→</kbd></button><button data-passo="1"><span>+1 s</span><kbd>→</kbd></button><button data-passo="10"><span>+10 s</span><kbd>L</kbd></button>
</div></div>

<div class="grupo-controles"><span class="rotulo-controles">Velocidade de reprodução</span><div class="botoes">
<button data-velocidade="0.1"><span>0,1×</span><kbd>1</kbd></button><button data-velocidade="0.25"><span>0,25×</span><kbd>2</kbd></button><button data-velocidade="0.5"><span>0,5×</span><kbd>3</kbd></button><button data-velocidade="0.75"><span>0,75×</span><kbd>4</kbd></button><button data-velocidade="1" class="ativo"><span>1×</span><kbd>5</kbd></button>
</div></div>

<div class="grupo-controles"><span class="rotulo-controles">Marcação de tempo</span><div class="botoes">
<button data-acao="inicio"><span>Marcar início</span><kbd>I</kbd></button><button data-acao="fim"><span>Marcar fim</span><kbd>O</kbd></button><button data-acao="adicionar"><span>Adicionar corte</span><kbd>Enter</kbd></button><button data-acao="desfazer"><span>Desfazer corte</span><kbd>U</kbd></button><button data-acao="limpar"><span>Limpar cortes</span><kbd>Del</kbd></button>
</div></div>
<div class="marcas">Início: <span id="__ID___inicio">—</span><br>Fim: <span id="__ID___fim">—</span></div>

<details>
  <summary>Lista de cortes compatível com a v5</summary>
  <div class="painel">
    <table><thead><tr><th>#</th><th>Início</th><th>Fim</th><th>Removido</th></tr></thead><tbody id="__ID___tabela"></tbody></table>
    <textarea id="__ID___saida" readonly></textarea>
    <div class="botoes"><button data-acao="copiar"><span>Copiar lista</span><kbd>C</kbd></button><button data-acao="baixar"><span>Baixar cortes.txt</span><kbd>B</kbd></button></div>
  </div>
</details>

<div class="painel">
  <strong>Edição / operações</strong>
  <div class="ajuda">Use as mesmas marcas de início/fim. Imagem, texto e vídeo aparecem na própria caixa durante a edição; operações já adicionadas reaparecem no tempo correspondente. Mídias devem estar na pasta do vídeo ou em uma subpasta.</div>
  <div class="grade-campos">
    <div class="campo"><label>Tipo de operação</label><select id="__ID___tipo_operacao">
      <option value="cut" selected>Corte</option><option value="mute">Silenciar áudio</option><option value="blur_region">Desfoque de região</option><option value="black_bar">Tarja sobre região</option><option value="overlay_text">Texto sobreposto</option><option value="overlay_image">Imagem sobreposta</option><option value="overlay_video">Vídeo sobreposto</option><option value="zoom_region">Zoom em região</option><option value="crop_region">Crop de região</option><option value="insert_video">Inserir vídeo em um ponto</option><option value="replace_video">Substituir trecho por vídeo</option>
    </select></div>
    <div class="campo" id="__ID___campo_texto"><label>Texto</label><input id="__ID___op_texto" type="text" placeholder="Texto que aparecerá no vídeo"></div>
    <div class="campo" id="__ID___campo_media"><label>Arquivo de mídia</label><input id="__ID___op_media" type="text" placeholder="ex.: logo.png ou camera.mp4"></div>
    <div class="campo" id="__ID___campo_fit"><label>Ajuste da mídia</label><select id="__ID___op_fit"><option value="contain">Conter / preservar proporção</option><option value="cover">Preencher / cortar bordas</option><option value="fill">Esticar</option></select></div>
    <div class="campo" id="__ID___campo_audio"><label>Áudio (vídeo sobreposto)</label><select id="__ID___op_audio"><option value="base">Só principal</option><option value="overlay">Só sobreposto</option><option value="both">Os dois</option><option value="none">Nenhum</option></select></div>
    <div class="campo" id="__ID___campo_playback"><label>Reprodução (vídeo sobreposto)</label><select id="__ID___op_playback"><option value="both">Os dois rodam</option><option value="base_only">Só principal roda</option><option value="overlay_only">Só sobreposto roda</option></select></div>
    <div class="campo" id="__ID___campo_intensidade"><label>Intensidade</label><input id="__ID___op_intensidade" type="number" min="1" max="100" step="1" value="16"></div>
  </div>
  <div class="botoes"><button data-acao="mostrar-regiao">Caixa</button><button data-acao="resetar-regiao">Resetar caixa</button><button data-acao="adicionar-operacao"><strong>Adicionar operação</strong></button><button data-acao="limpar-operacoes">Limpar adicionais</button></div>

  <table><thead><tr><th>#</th><th>Tipo</th><th>Início</th><th>Fim</th><th>Detalhes</th><th></th></tr></thead><tbody id="__ID___tabela_operacoes"></tbody></table>
  <div class="botoes"><button data-acao="salvar-projeto">Salvar projeto</button><button data-acao="copiar-projeto">Copiar JSON</button><button data-acao="baixar-projeto">Baixar JSON</button></div>
  <details><summary>Ver JSON do projeto</summary><textarea id="__ID___saida_operacoes" readonly></textarea></details>
</div>

<div class="ajuda">Clique uma vez dentro da interface para ativar os atalhos. A edição de palavras não altera os timestamps. A caixa mostra uma prévia visual antes de adicionar a operação. Sempre mantenha uma cópia da última versão funcional dos arquivos antes de testar uma nova versão.</div>
</div>

<script>
(() => {
const raiz=document.getElementById("__ID__"), video=document.getElementById("__ID___video"), palco=document.getElementById("__ID___palco"), camadaOperacoes=document.getElementById("__ID___camada_operacoes"), erroEl=document.getElementById("__ID___erro"), statusEl=document.getElementById("__ID___status"), tempoEl=document.getElementById("__ID___tempo"), velocidadeEl=document.getElementById("__ID___velocidade"), inicioEl=document.getElementById("__ID___inicio"), fimEl=document.getElementById("__ID___fim"), tabelaEl=document.getElementById("__ID___tabela"), saidaEl=document.getElementById("__ID___saida"), painelTranscricaoEl=document.getElementById("__ID___painel_transcricao"), transcricaoEl=document.getElementById("__ID___transcricao"), estadoTranscricaoEl=document.getElementById("__ID___estado_transcricao"), legendaEl=document.getElementById("__ID___legenda"), posicaoLegendaEl=document.getElementById("__ID___posicao_legenda"), caixaRegiao=document.getElementById("__ID___regiao"), previewRegiao=document.getElementById("__ID___preview_regiao"), rotuloRegiao=document.getElementById("__ID___rotulo_regiao"), tipoOperacaoEl=document.getElementById("__ID___tipo_operacao"), tabelaOperacoesEl=document.getElementById("__ID___tabela_operacoes"), saidaOperacoesEl=document.getElementById("__ID___saida_operacoes"), campoTexto=document.getElementById("__ID___campo_texto"), campoMedia=document.getElementById("__ID___campo_media"), campoFit=document.getElementById("__ID___campo_fit"), campoAudio=document.getElementById("__ID___campo_audio"), campoPlayback=document.getElementById("__ID___campo_playback"), campoIntensidade=document.getElementById("__ID___campo_intensidade"), opTexto=document.getElementById("__ID___op_texto"), opMedia=document.getElementById("__ID___op_media"), opFit=document.getElementById("__ID___op_fit"), opAudio=document.getElementById("__ID___op_audio"), opPlayback=document.getElementById("__ID___op_playback"), opIntensidade=document.getElementById("__ID___op_intensidade");
const fps=__FPS__, duracaoEsperada=__DURATION__, dadosTranscricao=__TRANSCRIPT_JSON__, dadosProjetoInicial=__PROJECT_JSON__, apiBase="__API_BASE__", podeSalvarTranscricao=__CAN_SAVE_TRANSCRIPT__, podeSalvarProjeto=__CAN_SAVE_PROJECT__;
let inicio=null,fim=null,cortes=[],operacoes=Array.isArray(dadosProjetoInicial?.operations)?JSON.parse(JSON.stringify(dadosProjetoInicial.operations)):[],blocosTranscricao=[],blocoAtivo=-1,palavraAtiva=null,transcricaoSuja=false,legendasVisiveis=__SUBTITLES_VISIBLE__;
let regiaoAtual={x:.66,y:.06,width:.29,height:.22}, interacaoRegiao=null;

function status(msg,erro=false){statusEl.style.display="block";statusEl.style.borderColor=erro?"#b00020":"#3a8f5c";statusEl.textContent=msg;clearTimeout(statusEl._timer);statusEl._timer=setTimeout(()=>statusEl.style.display="none",4500)}
function limitar(v){const d=Number.isFinite(video.duration)?video.duration:duracaoEsperada;return Math.max(0,Math.min(v,d))}
function formatar(s){s=Math.max(0,Number(s)||0);const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),r=s%60;return String(h).padStart(2,"0")+":"+String(m).padStart(2,"0")+":"+r.toFixed(3).padStart(6,"0")}
function formatarCurto(s){s=Math.max(0,Number(s)||0);const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),si=Math.floor(s%60);return h>0?String(h).padStart(2,"0")+":"+String(m).padStart(2,"0")+":"+String(si).padStart(2,"0"):String(m).padStart(2,"0")+":"+String(si).padStart(2,"0")}
function textoPalavras(words){return (words||[]).map(w=>w.word||"").join("").trim()}
function buscar(s){video.pause();video.currentTime=limitar(s);atualizarTempo(video.currentTime)}
function moverSegundos(d){buscar(video.currentTime+d)}
function alternarReproducao(){if(video.paused){video.play().then(()=>{camadaOperacoes.querySelectorAll("video").forEach(v=>{v.muted=true;v.playbackRate=video.playbackRate;v.play().catch(()=>{})})}).catch(()=>erroEl.style.display="block")}else{video.pause();camadaOperacoes.querySelectorAll("video").forEach(v=>v.pause())}}
function definirVelocidade(t){video.playbackRate=t;video.defaultPlaybackRate=t;camadaOperacoes.querySelectorAll("video").forEach(v=>v.playbackRate=t);velocidadeEl.textContent=String(t).replace(".",",")+"×";raiz.querySelectorAll("button[data-velocidade]").forEach(b=>b.classList.toggle("ativo",Number(b.dataset.velocidade)===Number(t)))}
function downloadTexto(nome,conteudo,tipo="text/plain;charset=utf-8"){const blob=new Blob([conteudo],{type:tipo}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=nome;a.click();setTimeout(()=>URL.revokeObjectURL(url),250)}
async function copiarTexto(texto){await navigator.clipboard.writeText(texto);status("Copiado para a área de transferência.")}

function atualizarSegmentoCorrespondente(palavra){if(!dadosTranscricao?.segments)return;for(const seg of dadosTranscricao.segments){const item=(seg.words||[]).find(w=>Math.abs(Number(w.start)-Number(palavra.start))<1e-6&&Math.abs(Number(w.end)-Number(palavra.end))<1e-6);if(item){item.word=palavra.word;seg.text=textoPalavras(seg.words);break}}}
function corrigirPalavra(span){if(!dadosTranscricao)return;const bi=Number(span.dataset.bloco),pi=Number(span.dataset.palavra),palavra=dadosTranscricao.blocks[bi].words[pi],atual=String(palavra.word||""),prefixo=(atual.match(/^\s*/)||[""])[0],novo=prompt("Corrigir palavra (o tempo será preservado):",atual.trim());if(novo===null)return;const limpo=novo.trim();if(!limpo)return;palavra.word=prefixo+limpo;dadosTranscricao.blocks[bi].text=textoPalavras(dadosTranscricao.blocks[bi].words);atualizarSegmentoCorrespondente(palavra);dadosTranscricao.version=Math.max(Number(dadosTranscricao.version||1),2);span.textContent=palavra.word;span.classList.add("editada");transcricaoSuja=true;estadoTranscricaoEl.textContent="alterações não salvas";atualizarLegenda(video.currentTime)}
async function salvarTranscricao(){if(!dadosTranscricao){return}if(!podeSalvarTranscricao){status("Sem caminho de transcrição para salvar. Use 'Baixar transcrição JSON'.",true);return}try{const r=await fetch(apiBase+"/__editor_api__/transcricao",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(dadosTranscricao)}),j=await r.json();if(!r.ok||!j.ok)throw new Error(j.erro||"Falha ao salvar");transcricaoSuja=false;estadoTranscricaoEl.textContent="salva em "+j.arquivo;status("Transcrição salva: "+j.arquivo)}catch(e){status("Erro ao salvar transcrição: "+e.message,true)}}
function srtTempo(s){let ms=Math.max(0,Math.round(Number(s)*1000)),h=Math.floor(ms/3600000);ms%=3600000;let m=Math.floor(ms/60000);ms%=60000;let si=Math.floor(ms/1000),mil=ms%1000;return String(h).padStart(2,"0")+":"+String(m).padStart(2,"0")+":"+String(si).padStart(2,"0")+","+String(mil).padStart(3,"0")}
function gerarSRT(vtt=false){if(!dadosTranscricao)return "";let linhas=vtt?["WEBVTT",""]:[];let n=1;for(const b of dadosTranscricao.blocks||[]){const t=textoPalavras(b.words)||String(b.text||"").trim();if(!t)continue;if(!vtt)linhas.push(String(n++));let a=srtTempo(b.start),z=srtTempo(b.end);if(vtt){a=a.replace(",",".");z=z.replace(",",".")}linhas.push(a+" --> "+z,t,"")}return linhas.join("\n")}

function renderizarTranscricao(){if(!dadosTranscricao||!Array.isArray(dadosTranscricao.blocks))return;painelTranscricaoEl.hidden=false;transcricaoEl.innerHTML="";blocosTranscricao=[];dadosTranscricao.blocks.forEach((b,bi)=>{const linha=document.createElement("div");linha.className="bloco-transcricao";const tempo=document.createElement("button");tempo.type="button";tempo.className="tempo-transcricao";tempo.dataset.irTempo=String(b.start);tempo.textContent=formatarCurto(b.start);tempo.title=formatar(b.start);const texto=document.createElement("div");texto.className="texto-transcricao";(b.words||[]).forEach((p,pi)=>{const span=document.createElement("span");span.className="palavra-transcricao";span.dataset.start=String(p.start);span.dataset.end=String(p.end);span.dataset.bloco=String(bi);span.dataset.palavra=String(pi);span.title=formatar(p.start)+" · duplo clique para corrigir";span.textContent=p.word;texto.appendChild(span)});linha.append(tempo,texto);transcricaoEl.appendChild(linha);blocosTranscricao.push({dados:b,elemento:linha,palavras:Array.from(texto.querySelectorAll(".palavra-transcricao"))})})}
function atualizarTranscricao(s){if(!blocosTranscricao.length)return;let nb=blocoAtivo;if(nb<0||s<blocosTranscricao[nb].dados.start||s>blocosTranscricao[nb].dados.end+.25)nb=blocosTranscricao.findIndex(({dados})=>s>=dados.start-.05&&s<=dados.end+.25);if(nb!==blocoAtivo){if(blocoAtivo>=0)blocosTranscricao[blocoAtivo].elemento.classList.remove("ativo");blocoAtivo=nb;if(blocoAtivo>=0){const el=blocosTranscricao[blocoAtivo].elemento;el.classList.add("ativo");el.scrollIntoView({block:"nearest"})}}let np=null;if(blocoAtivo>=0)np=blocosTranscricao[blocoAtivo].palavras.find(span=>s>=Number(span.dataset.start)-.03&&s<=Number(span.dataset.end)+.05)||null;if(np!==palavraAtiva){if(palavraAtiva)palavraAtiva.classList.remove("ativa");palavraAtiva=np;if(palavraAtiva)palavraAtiva.classList.add("ativa")}}
function atualizarLegenda(s){if(!dadosTranscricao||!legendasVisiveis){legendaEl.hidden=true;return}const b=(dadosTranscricao.blocks||[]).find(x=>s>=Number(x.start)-.03&&s<=Number(x.end)+.1);if(!b){legendaEl.hidden=true;return}legendaEl.querySelector("span").textContent=textoPalavras(b.words)||b.text||"";legendaEl.hidden=false}
function atualizarTempo(s=video.currentTime){tempoEl.textContent=formatar(s);atualizarTranscricao(s);atualizarLegenda(s);renderizarOverlaysAtivos(s)}

function sincronizarCortes(){cortes=operacoes.filter(o=>o.type==="cut"&&o.enabled!==false&&Number(o.end)>Number(o.start)).map(o=>[Number(o.start),Number(o.end)]).sort((a,b)=>a[0]-b[0])}
function novaId(tipo){return tipo+"_"+Date.now().toString(36)+"_"+Math.random().toString(36).slice(2,6)}
function criarOp(tipo,a,b,params={}){return {id:novaId(tipo),type:tipo,start:Number(a),end:Number(b),enabled:true,track:tipo==="cut"?"main":"editor_v6",params:params,note:""}}
function listaPython(){if(!cortes.length)return "[]";return "[\n"+cortes.map(([a,b])=>`    ("${formatar(a)}", "${formatar(b)}"),`).join("\n")+"\n]"}
function projetoAtual(){return {version:1,source:__SOURCE_JSON__,time_reference:"source",operations:operacoes}}
function resumoOp(o){const p=o.params||{};if(o.type==="overlay_text")return p.text||"";if(["overlay_image","overlay_video","insert_video","replace_video"].includes(o.type))return p.media||"";if(o.type==="blur_region")return "intensidade "+(p.intensity||"");if(o.type==="mute")return "áudio principal";if(p.region)return `x=${p.region.x.toFixed(3)}, y=${p.region.y.toFixed(3)}, w=${p.region.width.toFixed(3)}, h=${p.region.height.toFixed(3)}`;return ""}
function nomeOp(t){return ({cut:"Corte",mute:"Silenciar",blur_region:"Desfoque",black_bar:"Tarja",overlay_text:"Texto",overlay_image:"Imagem",overlay_video:"Vídeo sobreposto",zoom_region:"Zoom",crop_region:"Crop",insert_video:"Inserir vídeo",replace_video:"Substituir por vídeo"})[t]||t}
function renderizar(){sincronizarCortes();inicioEl.textContent=inicio===null?"—":formatar(inicio);fimEl.textContent=fim===null?"—":formatar(fim);tabelaEl.innerHTML="";cortes.forEach(([a,b],i)=>{const tr=document.createElement("tr");tr.innerHTML=`<td>${i+1}</td><td>${formatar(a)}</td><td>${formatar(b)}</td><td>${formatar(b-a)}</td>`;tabelaEl.appendChild(tr)});saidaEl.value=listaPython();tabelaOperacoesEl.innerHTML="";operacoes.forEach((o,i)=>{const tr=document.createElement("tr");tr.innerHTML=`<td>${i+1}</td><td><span class="tipo-op">${nomeOp(o.type)}</span></td><td>${formatar(o.start)}</td><td>${formatar(o.end)}</td><td>${resumoOp(o)}</td><td><button data-remover-op="${i}">×</button></td>`;tabelaOperacoesEl.appendChild(tr)});saidaOperacoesEl.value=JSON.stringify(projetoAtual(),null,2);renderizarOverlaysAtivos(video.currentTime,true)}
function marcarInicio(){inicio=video.currentTime;renderizar()}
function marcarFim(){fim=video.currentTime;renderizar()}
function adicionarCorte(){if(inicio===null||fim===null){alert("Marque o início e o fim antes de adicionar.");return}if(fim<=inicio){alert("O fim precisa ser posterior ao início.");return}operacoes.push(criarOp("cut",inicio,fim,{}));operacoes.sort((a,b)=>a.start-b.start);inicio=null;fim=null;renderizar()}
function desfazerCorte(){for(let i=operacoes.length-1;i>=0;i--){if(operacoes[i].type==="cut"){operacoes.splice(i,1);break}}renderizar()}
function limparCortes(){if(confirm("Apagar todos os cortes? As operações adicionais serão mantidas.")){operacoes=operacoes.filter(o=>o.type!=="cut");inicio=null;fim=null;renderizar()}}

const tiposComRegiao=new Set(["blur_region","black_bar","overlay_text","overlay_image","overlay_video","zoom_region","crop_region"]);
const tiposMediaRegiao=new Set(["overlay_image","overlay_video"]);
let chaveOverlaysAtivos="";
function clamp01(v){return Math.max(0,Math.min(1,v))}
function aplicarRegiao(){const r=palco.getBoundingClientRect();if(!r.width||!r.height)return;caixaRegiao.style.left=(regiaoAtual.x*100)+"%";caixaRegiao.style.top=(regiaoAtual.y*100)+"%";caixaRegiao.style.width=(regiaoAtual.width*100)+"%";caixaRegiao.style.height=(regiaoAtual.height*100)+"%"}
function lerRegiao(){const p=palco.getBoundingClientRect(),c=caixaRegiao.getBoundingClientRect();regiaoAtual={x:clamp01((c.left-p.left)/p.width),y:clamp01((c.top-p.top)/p.height),width:clamp01(c.width/p.width),height:clamp01(c.height/p.height)};if(regiaoAtual.x+regiaoAtual.width>1)regiaoAtual.width=1-regiaoAtual.x;if(regiaoAtual.y+regiaoAtual.height>1)regiaoAtual.height=1-regiaoAtual.y;return {...regiaoAtual,unit:"normalized"}}
function resetarRegiao(){regiaoAtual={x:.66,y:.06,width:.29,height:.22};aplicarRegiao();atualizarPreviewEdicao()}
function caminhoMediaUrl(caminho){const bruto=String(caminho||"").trim().replace(/\\/g,"/");if(!bruto)return "";if(/^https?:\/\//i.test(bruto))return bruto;const partes=bruto.split("/").filter(p=>p&&p!==".");if(partes.some(p=>p===".."))return "";return apiBase+"/"+partes.map(encodeURIComponent).join("/")}
function limparPreviewEdicao(){previewRegiao.innerHTML="";previewRegiao.style.backdropFilter="";previewRegiao.style.webkitBackdropFilter="";previewRegiao.style.background=""}
function erroPreview(msg){limparPreviewEdicao();const d=document.createElement("div");d.className="erro-preview";d.textContent=msg;previewRegiao.appendChild(d)}
function atualizarPreviewEdicao(){
    limparPreviewEdicao();
    const t=tipoOperacaoEl.value;
    if(!tiposComRegiao.has(t))return;
    if(t==="overlay_text"){
        const d=document.createElement("div");d.className="texto-preview";d.textContent=opTexto.value.trim()||"Texto";previewRegiao.appendChild(d);return;
    }
    if(t==="overlay_image"){
        const url=caminhoMediaUrl(opMedia.value);if(!url){erroPreview("Informe uma imagem da pasta do vídeo.");return}
        const img=document.createElement("img");img.alt="Prévia";img.style.objectFit=opFit.value||"contain";img.src=url;img.onerror=()=>erroPreview("Imagem não encontrada ou não suportada.");previewRegiao.appendChild(img);return;
    }
    if(t==="overlay_video"){
        const url=caminhoMediaUrl(opMedia.value);if(!url){erroPreview("Informe um vídeo da pasta do vídeo.");return}
        const v=document.createElement("video");v.muted=true;v.playsInline=true;v.preload="metadata";v.style.objectFit=opFit.value||"contain";v.src=url;v.addEventListener("loadedmetadata",()=>{try{v.currentTime=Math.min(.05,Math.max(0,(v.duration||1)/100))}catch(_){}});v.onerror=()=>erroPreview("Vídeo não encontrado ou não suportado no navegador.");previewRegiao.appendChild(v);return;
    }
    if(t==="blur_region"){
        const px=Math.max(1,Number(opIntensidade.value)||16);previewRegiao.style.backdropFilter=`blur(${px}px)`;previewRegiao.style.webkitBackdropFilter=`blur(${px}px)`;previewRegiao.style.background="rgba(255,255,255,.025)";return;
    }
    if(t==="black_bar"){previewRegiao.style.background="#000";return}
    if(t==="zoom_region"||t==="crop_region"){const d=document.createElement("div");d.className="erro-preview";d.textContent=t==="zoom_region"?"Área do zoom":"Área do crop";previewRegiao.appendChild(d)}
}
function estiloRegiao(el,r){el.style.left=(Number(r.x)*100)+"%";el.style.top=(Number(r.y)*100)+"%";el.style.width=(Number(r.width)*100)+"%";el.style.height=(Number(r.height)*100)+"%"}
function criarPreviewOperacao(o){
    const p=o.params||{},r=p.region;if(!r)return null;
    const el=document.createElement("div");el.className="preview-op";estiloRegiao(el,r);el.dataset.opId=o.id||"";
    if(o.type==="overlay_text"){
        const d=document.createElement("div");d.className="preview-texto";d.textContent=p.text||"";el.appendChild(d);
    }else if(o.type==="overlay_image"){
        const img=document.createElement("img");img.src=caminhoMediaUrl(p.media);img.style.objectFit=p.fit||"contain";img.alt="";el.appendChild(img);
    }else if(o.type==="overlay_video"){
        const v=document.createElement("video");v.src=caminhoMediaUrl(p.media);v.muted=true;v.playsInline=true;v.preload="auto";v.style.objectFit=p.fit||"contain";v.dataset.opStart=String(o.start);el.appendChild(v);
    }else if(o.type==="blur_region"){
        el.classList.add("preview-blur");const px=Math.max(1,Number(p.intensity)||16);el.style.backdropFilter=`blur(${px}px)`;el.style.webkitBackdropFilter=`blur(${px}px)`;
    }else if(o.type==="black_bar"){
        el.classList.add("preview-tarja");el.style.opacity=String(p.opacity??1);
    }else if(o.type==="zoom_region"||o.type==="crop_region"){
        el.classList.add("preview-guia");const l=document.createElement("span");l.className="mini-rotulo";l.textContent=o.type==="zoom_region"?"zoom":"crop";el.appendChild(l);
    }else{return null}
    return el;
}
function renderizarOverlaysAtivos(s,forcar=false){
    const ativos=operacoes.filter(o=>o.enabled!==false&&tiposComRegiao.has(o.type)&&Number(o.start)<=s&&s<=Number(o.end));
    const chave=ativos.map(o=>o.id+":"+o.type+":"+JSON.stringify(o.params||{})).join("|");
    if(forcar||chave!==chaveOverlaysAtivos){
        chaveOverlaysAtivos=chave;camadaOperacoes.innerHTML="";ativos.forEach(o=>{const el=criarPreviewOperacao(o);if(el)camadaOperacoes.appendChild(el)});
    }
    camadaOperacoes.querySelectorAll("video[data-op-start]").forEach(v=>{if(v.readyState<1)return;const rel=Math.max(0,s-Number(v.dataset.opStart||0));const alvo=v.duration&&Number.isFinite(v.duration)?Math.min(rel,Math.max(0,v.duration-.04)):rel;if(Math.abs((v.currentTime||0)-alvo)>.12){try{v.currentTime=alvo}catch(_){}}});
}
function atualizarCamposOperacao(){
    const t=tipoOperacaoEl.value,reg=tiposComRegiao.has(t);
    caixaRegiao.hidden=!reg;rotuloRegiao.textContent=nomeOp(t);
    campoTexto.style.display=t==="overlay_text"?"flex":"none";
    campoMedia.style.display=["overlay_image","overlay_video","insert_video","replace_video"].includes(t)?"flex":"none";
    campoFit.style.display=tiposMediaRegiao.has(t)?"flex":"none";
    campoAudio.style.display=t==="overlay_video"?"flex":"none";
    campoPlayback.style.display=t==="overlay_video"?"flex":"none";
    campoIntensidade.style.display=["blur_region","zoom_region"].includes(t)?"flex":"none";
    if(reg)requestAnimationFrame(()=>{aplicarRegiao();atualizarPreviewEdicao()});else limparPreviewEdicao();
}
function adicionarOperacao(){
    const t=tipoOperacaoEl.value;
    if(t==="cut"){adicionarCorte();return}
    if(inicio===null){alert("Marque pelo menos o início.");return}
    let a=inicio,b=fim;
    if(t==="insert_video")b=a;else if(b===null||b<=a){alert("Para esta operação, marque início e fim válidos.");return}
    const p={};
    if(tiposComRegiao.has(t))p.region=lerRegiao();
    if(t==="overlay_text"){if(!opTexto.value.trim()){alert("Informe o texto.");return}p.text=opTexto.value.trim()}
    if(["overlay_image","overlay_video","insert_video","replace_video"].includes(t)){if(!opMedia.value.trim()){alert("Informe o arquivo de mídia.");return}p.media=opMedia.value.trim()}
    if(tiposMediaRegiao.has(t))p.fit=opFit.value||"contain";
    if(t==="overlay_video"){p.audio_policy=opAudio.value;p.playback_policy=opPlayback.value}
    if(t==="blur_region")p.intensity=Number(opIntensidade.value)||16;
    if(t==="zoom_region")p.factor=Math.max(1,Number(opIntensidade.value)||2);
    if(t==="black_bar")p.opacity=1;
    if(t==="crop_region")p.restore_canvas=true;
    operacoes.push(criarOp(t,a,b,p));operacoes.sort((x,y)=>x.start-y.start);renderizar();status("Operação registrada. A prévia aparece no intervalo marcado.")
}
async function salvarProjeto(){const dados=projetoAtual();if(!podeSalvarProjeto){status("Sem caminho de projeto no servidor. Use 'Baixar projeto JSON'.",true);return}try{const r=await fetch(apiBase+"/__editor_api__/projeto",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(dados)}),j=await r.json();if(!r.ok||!j.ok)throw new Error(j.erro||"Falha ao salvar");status("Projeto salvo: "+j.arquivo)}catch(e){status("Erro ao salvar projeto: "+e.message,true)}}

caixaRegiao.addEventListener("pointerdown",e=>{if(caixaRegiao.hidden)return;const p=palco.getBoundingClientRect(),c=caixaRegiao.getBoundingClientRect(),resize=e.target.classList.contains("alca");interacaoRegiao={modo:resize?"resize":"move",pointerId:e.pointerId,startX:e.clientX,startY:e.clientY,left:c.left-p.left,top:c.top-p.top,width:c.width,height:c.height,pw:p.width,ph:p.height};caixaRegiao.setPointerCapture(e.pointerId);e.preventDefault()});
caixaRegiao.addEventListener("pointermove",e=>{const q=interacaoRegiao;if(!q||q.pointerId!==e.pointerId)return;const dx=e.clientX-q.startX,dy=e.clientY-q.startY;if(q.modo==="move"){let l=Math.max(0,Math.min(q.left+dx,q.pw-q.width)),t=Math.max(0,Math.min(q.top+dy,q.ph-q.height));regiaoAtual.x=l/q.pw;regiaoAtual.y=t/q.ph}else{let w=Math.max(30,Math.min(q.width+dx,q.pw-q.left)),h=Math.max(24,Math.min(q.height+dy,q.ph-q.top));regiaoAtual.width=w/q.pw;regiaoAtual.height=h/q.ph}aplicarRegiao();e.preventDefault()});
caixaRegiao.addEventListener("pointerup",e=>{if(interacaoRegiao&&interacaoRegiao.pointerId===e.pointerId){lerRegiao();atualizarPreviewEdicao();interacaoRegiao=null;try{caixaRegiao.releasePointerCapture(e.pointerId)}catch(_){}}});
window.addEventListener("resize",()=>{if(!caixaRegiao.hidden)aplicarRegiao()});

raiz.addEventListener("dblclick",e=>{const p=e.target.closest(".palavra-transcricao");if(p){e.preventDefault();e.stopPropagation();corrigirPalavra(p)}});
raiz.addEventListener("click",async e=>{const alvo=e.target;const interativo=alvo.closest("input,select,textarea,button,[contenteditable=true]");if(!interativo)raiz.focus();const palavra=e.target.closest(".palavra-transcricao");if(palavra){buscar(Number(palavra.dataset.start));return}const remover=e.target.closest("button[data-remover-op]");if(remover){operacoes.splice(Number(remover.dataset.removerOp),1);renderizar();return}const b=e.target.closest("button");if(!b)return;if(b.dataset.irTempo){buscar(Number(b.dataset.irTempo));return}if(b.dataset.passo){moverSegundos(Number(b.dataset.passo));return}if(b.dataset.velocidade){definirVelocidade(Number(b.dataset.velocidade));return}const a=b.dataset.acao;if(a==="play")alternarReproducao();else if(a==="inicio")marcarInicio();else if(a==="fim")marcarFim();else if(a==="adicionar")adicionarCorte();else if(a==="desfazer")desfazerCorte();else if(a==="limpar")limparCortes();else if(a==="copiar")await copiarTexto(saidaEl.value);else if(a==="baixar")downloadTexto("cortes.txt",saidaEl.value);else if(a==="salvar-transcricao")await salvarTranscricao();else if(a==="baixar-transcricao")downloadTexto("transcricao_corrigida.json",JSON.stringify(dadosTranscricao,null,2),"application/json;charset=utf-8");else if(a==="baixar-srt")downloadTexto("legendas.srt",gerarSRT(false));else if(a==="baixar-vtt")downloadTexto("legendas.vtt",gerarSRT(true));else if(a==="toggle-legenda"){legendasVisiveis=!legendasVisiveis;atualizarLegenda(video.currentTime)}else if(a==="mostrar-regiao"){caixaRegiao.hidden=!caixaRegiao.hidden;if(!caixaRegiao.hidden)requestAnimationFrame(aplicarRegiao)}else if(a==="resetar-regiao")resetarRegiao();else if(a==="adicionar-operacao")adicionarOperacao();else if(a==="limpar-operacoes"){if(confirm("Apagar todas as operações adicionais? Os cortes serão mantidos.")){operacoes=operacoes.filter(o=>o.type==="cut");renderizar()}}else if(a==="salvar-projeto")await salvarProjeto();else if(a==="copiar-projeto")await copiarTexto(saidaOperacoesEl.value);else if(a==="baixar-projeto")downloadTexto("projeto_editor.json",saidaOperacoesEl.value,"application/json;charset=utf-8")});

tipoOperacaoEl.addEventListener("change",atualizarCamposOperacao);
opTexto.addEventListener("input",atualizarPreviewEdicao);
opMedia.addEventListener("input",atualizarPreviewEdicao);
opFit.addEventListener("change",atualizarPreviewEdicao);
opIntensidade.addEventListener("input",atualizarPreviewEdicao);
posicaoLegendaEl.addEventListener("change",()=>{legendaEl.classList.remove("topo","meio");if(posicaoLegendaEl.value!=="baixo")legendaEl.classList.add(posicaoLegendaEl.value);atualizarLegenda(video.currentTime)});
video.addEventListener("error",()=>erroEl.style.display="block");video.addEventListener("loadedmetadata",()=>{erroEl.style.display="none";video.preservesPitch=true;definirVelocidade(1);atualizarTempo(0);aplicarRegiao()});video.addEventListener("timeupdate",()=>atualizarTempo(video.currentTime));video.addEventListener("seeked",()=>atualizarTempo(video.currentTime));
if("requestVideoFrameCallback" in HTMLVideoElement.prototype){const acompanhar=(_a,m)=>{atualizarTempo(m.mediaTime);video.requestVideoFrameCallback(acompanhar)};video.requestVideoFrameCallback(acompanhar)}
raiz.tabIndex=0;raiz.addEventListener("keydown",async e=>{if(["TEXTAREA","INPUT","SELECT"].includes(e.target.tagName))return;const t=e.key.toLowerCase();if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();let p=1;if(e.ctrlKey&&e.shiftKey)p=.001;else if(e.ctrlKey)p=.01;else if(e.shiftKey)p=.1;moverSegundos((e.key==="ArrowLeft"?-1:1)*p);return}if(t===" "){e.preventDefault();alternarReproducao()}else if(t==="j"){e.preventDefault();moverSegundos(-10)}else if(t==="l"){e.preventDefault();moverSegundos(10)}else if(t==="i"){e.preventDefault();marcarInicio()}else if(t==="o"){e.preventDefault();marcarFim()}else if(t==="u"){e.preventDefault();desfazerCorte()}else if(e.key==="Delete"){e.preventDefault();limparCortes()}else if(t==="c"&&!e.ctrlKey&&!e.metaKey){e.preventDefault();await copiarTexto(saidaEl.value)}else if(t==="b"){e.preventDefault();downloadTexto("cortes.txt",saidaEl.value)}else if(e.key==="Enter"){e.preventDefault();adicionarCorte()}else if(["1","2","3","4","5"].includes(e.key)){e.preventDefault();definirVelocidade({"1":.1,"2":.25,"3":.5,"4":.75,"5":1}[e.key])}});

sincronizarCortes();renderizar();renderizarTranscricao();atualizarCamposOperacao();atualizarLegenda(0);
})();
</script>
"""

    conteudo = (
        modelo
        .replace("__ID__", identificador)
        .replace("__WIDTH__", str(int(largura)))
        .replace("__SRC__", html_lib.escape(origem, quote=True))
        .replace("__FPS_TEXT__", f'{info_exibido["fps"]:.3f}')
        .replace("__FPS__", repr(info_exibido["fps"]))
        .replace("__DURATION__", repr(info_exibido["duracao"]))
        .replace("__TRANSCRIPT_JSON__", json.dumps(dados_transcricao, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__PROJECT_JSON__", json.dumps(dados_projeto, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__SOURCE_JSON__", json.dumps(original.name, ensure_ascii=False))
        .replace("__API_BASE__", html_lib.escape(api_base, quote=True))
        .replace("__CAN_SAVE_TRANSCRIPT__", "true" if caminho_transcricao is not None and dados_transcricao is not None else "false")
        .replace("__CAN_SAVE_PROJECT__", "true" if caminho_projeto is not None else "false")
        .replace("__SUBTITLES_VISIBLE__", "true" if legendas and dados_transcricao is not None else "false")
        .replace("__SUBTITLE_HIDDEN__", "" if legendas and dados_transcricao is not None else "hidden")
        .replace("__ORIGINAL__", html_lib.escape(original.name))
        .replace("__CODEC__", html_lib.escape(resumo_codec))
        .replace("__PREVIEW_NOTICE__", aviso_preview)
        .replace("__V6_NOTICE__", aviso_v6)
    )
    display(HTML(conteudo))

