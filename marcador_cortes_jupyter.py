from __future__ import annotations

import html as html_lib
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import uuid
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from IPython.display import HTML, display

__version__ = "6.9.3"


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
        "stream=index,codec_type,codec_name,pix_fmt,avg_frame_rate,r_frame_rate,width,height:"
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
        "largura": int(video.get("width") or 0),
        "altura": int(video.get("height") or 0),
    }


def _informacoes_midia(arquivo: str | Path) -> dict:
    """Informações mínimas de qualquer mídia (vídeo ou áudio)."""
    arquivo = Path(arquivo).resolve()
    if not arquivo.exists():
        raise FileNotFoundError(f"Mídia não encontrada: {arquivo}")
    dados = _executar_ffprobe(arquivo)
    streams = dados.get("streams", [])
    return {
        "arquivo": arquivo,
        "duracao": float(dados.get("format", {}).get("duration") or 0.0),
        "possui_video": any(x.get("codec_type") == "video" for x in streams),
        "possui_audio": any(x.get("codec_type") == "audio" for x in streams),
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
    lado_maximo: int | None = None,
    crf: int = 20,
    bitrate_audio: str = "128k",
) -> Path:
    """Cria uma cópia apenas para visualização no navegador.

    ``lado_maximo`` permite gerar um preview mais leve sem alterar o vídeo
    original. Isso é usado no Google Colab para reduzir o tráfego pelo proxy
    do iframe. Os tempos permanecem os mesmos e o render final continua sendo
    feito a partir do arquivo original.
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

    filtros: list[str] = []
    if lado_maximo is not None and int(lado_maximo) > 0:
        info = informacoes_video(entrada)
        w, h = int(info["largura"]), int(info["altura"])
        maior = max(w, h)
        if maior > int(lado_maximo):
            fator = float(lado_maximo) / float(maior)
            novo_w = max(2, int(round((w * fator) / 2.0) * 2))
            novo_h = max(2, int(round((h * fator) / 2.0) * 2))
            filtros.append(f"scale={novo_w}:{novo_h}")

    comando = [
        "ffmpeg",
        "-y",
        "-i", str(entrada),
        "-map", "0:v:0",
        "-map", "0:a:0?",
    ]
    if filtros:
        comando += ["-vf", ",".join(filtros)]
    comando += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough",
        "-c:a", "aac",
        "-b:a", str(bitrate_audio),
        "-movflags", "+faststart",
        str(saida),
    ]

    subprocess.run(comando, check=True)

    print("Cópia de visualização criada.")
    return saida


def _preparar_preview_para_ambiente(
    original: Path,
    info_original: dict,
    preparar_preview: bool | str,
    modo_colab: bool,
) -> Path:
    """Escolhe o arquivo usado apenas no player do editor.

    No Colab, ``auto`` sempre cria uma cópia leve com ``faststart``. O proxy
    do iframe é mais confiável com arquivos menores e com os metadados do MP4
    no início. No Jupyter local, ``auto`` conserva o comportamento anterior.
    """
    if preparar_preview not in {True, False, "auto"}:
        raise ValueError('preparar_preview deve ser True, False ou "auto".')

    if preparar_preview is False:
        return original

    if modo_colab and preparar_preview == "auto":
        saida = original.with_name(f"{original.stem}_preview_colab.mp4")
        return criar_preview_web(
            original,
            saida=saida,
            lado_maximo=1280,
            crf=24,
            bitrate_audio="96k",
        )

    precisa_converter = (
        preparar_preview is True
        or (preparar_preview == "auto" and not _compativel_com_navegador(info_original))
    )
    return criar_preview_web(original) if precisa_converter else original

# -----------------------------------------------------------------------------
# Recursos de projeto / legenda e prévia de timeline adicionados na v6
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
    dados.setdefault("version", 4)
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



def _texto_bloco_legenda(bloco: dict) -> str:
    palavras = bloco.get("words") or []
    return _texto_palavras(palavras) if palavras else str(bloco.get("text", "")).strip()


def _quebrar_texto_legenda(texto: str, largura: int, tamanho_fonte: int) -> list[str]:
    """Quebra a legenda em linhas aproximando a largura disponível no vídeo."""
    texto = " ".join(str(texto or "").split())
    if not texto:
        return []
    # Aproximação conservadora: glifos médios ocupam ~0,55 da altura da fonte.
    max_chars = max(16, int((largura * 0.82) / max(1.0, tamanho_fonte * 0.55)))
    max_chars = min(64, max_chars)
    return textwrap.wrap(texto, width=max_chars, break_long_words=False, break_on_hyphens=False)


def _aplicar_legendas_no_segmento(
    filtros: list[str],
    v_atual: str,
    transcricao: dict | None,
    config: dict | None,
    inicio: float,
    fim: float,
    velocidade: float,
    largura: int,
    altura: int,
) -> str:
    """Queima a transcrição corrigida no trecho atual do vídeo principal."""
    if not transcricao or not config or not bool(config.get("burn_in", False)):
        return v_atual

    proporcao = config.get("font_size_ratio")
    if proporcao is not None:
        tamanho = int(round(max(0.001, float(proporcao)) * altura))
    else:
        tamanho = int(round(float(config.get("font_size", 32))))
    tamanho = max(4, min(max(4, altura // 4), tamanho))

    posicao = str(config.get("position", "baixo"))
    # Ajuste fino vertical em porcentagem da altura real do frame.
    # Valor positivo desloca a legenda para baixo; negativo, para cima.
    ajuste_vertical_pct = max(-20.0, min(20.0, float(config.get("vertical_offset_pct", 0.0) or 0.0)))
    ajuste_vertical = int(round(altura * ajuste_vertical_pct / 100.0))
    margem = max(10, int(round(altura * 0.045)))
    espacamento = max(2, tamanho // 8)
    atual = v_atual
    indice_leg = 0

    for bloco in transcricao.get("blocks", []):
        try:
            bs = float(bloco.get("start", 0.0))
            be = float(bloco.get("end", bs))
        except Exception:
            continue
        if be <= inicio + 1e-6 or bs >= fim - 1e-6:
            continue
        texto = _texto_bloco_legenda(bloco)
        linhas = _quebrar_texto_legenda(texto, largura, tamanho)
        if not linhas:
            continue

        local_ini = max(0.0, (max(bs, inicio) - inicio) / velocidade)
        local_fim = max(local_ini + 0.001, (min(be, fim) - inicio) / velocidade)
        altura_bloco = len(linhas) * tamanho + max(0, len(linhas) - 1) * espacamento
        if posicao == "topo":
            y0 = margem + ajuste_vertical
        elif posicao == "meio":
            y0 = int(round((altura - altura_bloco) / 2)) + ajuste_vertical
        else:
            y0 = altura - margem - altura_bloco + ajuste_vertical
        y0 = max(0, min(max(0, altura - altura_bloco), y0))

        for li, linha in enumerate(linhas):
            texto_ff = _escape_drawtext(linha)
            novo = f"vsub{indice_leg}"
            y = y0 + li * (tamanho + espacamento)
            boxborder = max(4, int(round(tamanho * 0.18)))
            filtros.append(
                f"[{atual}]drawtext=text='{texto_ff}':font='sans-serif':fontcolor=white:fontsize={tamanho}:"
                f"x=(w-text_w)/2:y={y}:"
                f"box=1:boxcolor=black@0.58:boxborderw={boxborder}:"
                f"enable='between(t,{local_ini:.8f},{local_fim:.8f})'[{novo}]"
            )
            atual = novo
            indice_leg += 1
    return atual


# -----------------------------------------------------------------------------
# Renderização real do projeto (v6.6)
# -----------------------------------------------------------------------------


def _executar_ffmpeg_render(comando: list[str], contexto: str) -> subprocess.CompletedProcess:
    """Executa o FFmpeg e transforma falhas em mensagens úteis no Jupyter."""
    resultado = subprocess.run(
        comando,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if resultado.returncode != 0:
        linhas = (resultado.stderr or "").strip().splitlines()
        resumo = "\n".join(linhas[-30:]) if linhas else "Sem detalhes no stderr do FFmpeg."
        raise RuntimeError(
            f"FFmpeg falhou durante {contexto} (código {resultado.returncode}).\n\n{resumo}"
        )
    return resultado


def _resolver_midia_projeto(raiz: Path, caminho: str | Path) -> Path:
    """Resolve uma mídia do projeto mantendo-a dentro da pasta do vídeo."""
    bruto = Path(str(caminho).strip())
    candidato = bruto if bruto.is_absolute() else (raiz / bruto)
    candidato = candidato.resolve()
    try:
        candidato.relative_to(raiz.resolve())
    except ValueError as erro:
        raise ValueError(
            f"A mídia precisa estar na pasta do vídeo ou em subpasta: {caminho}"
        ) from erro
    if not candidato.exists():
        raise FileNotFoundError(f"Mídia não encontrada: {candidato}")
    return candidato


def _atempo_cadeia(fator: float) -> str:
    """Monta uma cadeia atempo válida para fatores fora de 0.5–2.0."""
    fator = max(0.1, min(16.0, float(fator)))
    partes: list[float] = []
    restante = fator
    while restante > 2.0 + 1e-9:
        partes.append(2.0)
        restante /= 2.0
    while restante < 0.5 - 1e-9:
        partes.append(0.5)
        restante /= 0.5
    partes.append(restante)
    return ",".join(f"atempo={p:.8f}" for p in partes)


def _regiao_pixels(regiao: dict | None, largura: int, altura: int) -> tuple[int, int, int, int]:
    r = regiao or {}
    x = max(0.0, min(1.0, float(r.get("x", 0.0))))
    y = max(0.0, min(1.0, float(r.get("y", 0.0))))
    w = max(0.001, min(1.0 - x, float(r.get("width", 1.0))))
    h = max(0.001, min(1.0 - y, float(r.get("height", 1.0))))
    px = max(0, min(largura - 2, round(x * largura)))
    py = max(0, min(altura - 2, round(y * altura)))
    pw = max(2, min(largura - px, round(w * largura)))
    ph = max(2, min(altura - py, round(h * altura)))
    # H.264/yuv420p funciona melhor com dimensões pares.
    pw -= pw % 2
    ph -= ph % 2
    pw = max(2, pw)
    ph = max(2, ph)
    return px, py, pw, ph


def _janela_enquadramento_pixels(
    regiao: dict | None,
    largura: int,
    altura: int,
    fator: float | None = None,
) -> tuple[int, int, int, int]:
    """
    Retorna uma janela com o mesmo aspecto do frame que CONTÉM toda a região.

    - crop: fator=None -> menor janela possível sem perder a seleção;
    - zoom: fator>=1 -> tenta largura/altura 1/fator, mas reduz o zoom se
      isso cortaria qualquer parte da região selecionada.
    """
    r = regiao or {}
    rx = max(0.0, min(1.0, float(r.get("x", 0.0))))
    ry = max(0.0, min(1.0, float(r.get("y", 0.0))))
    rw = max(0.001, min(1.0 - rx, float(r.get("width", 1.0))))
    rh = max(0.001, min(1.0 - ry, float(r.get("height", 1.0))))

    # Em coordenadas normalizadas, uma janela com o mesmo aspect ratio do
    # frame é um quadrado (largura_norm == altura_norm).
    minimo = max(rw, rh)
    desejado = 0.0 if fator is None else 1.0 / max(1.0, float(fator))
    tamanho = min(1.0, max(minimo, desejado, 0.001))

    cx = rx + rw / 2.0
    cy = ry + rh / 2.0

    # Faixa admissível para que a janela permaneça no frame E contenha
    # integralmente a região selecionada.
    x_min = max(0.0, rx + rw - tamanho)
    x_max = min(rx, 1.0 - tamanho)
    y_min = max(0.0, ry + rh - tamanho)
    y_max = min(ry, 1.0 - tamanho)

    x = min(max(cx - tamanho / 2.0, x_min), x_max) if x_min <= x_max else max(0.0, min(1.0 - tamanho, cx - tamanho / 2.0))
    y = min(max(cy - tamanho / 2.0, y_min), y_max) if y_min <= y_max else max(0.0, min(1.0 - tamanho, cy - tamanho / 2.0))

    px = int(round(x * largura))
    py = int(round(y * altura))
    pw = max(2, int(round(tamanho * largura)))
    ph = max(2, int(round(tamanho * altura)))

    # yuv420p: dimensões pares. Expandimos, em vez de reduzir, para não
    # eliminar pixels da seleção nas bordas por arredondamento.
    if pw % 2:
        pw += 1
    if ph % 2:
        ph += 1
    pw = min(largura, pw)
    ph = min(altura, ph)
    px = max(0, min(largura - pw, px))
    py = max(0, min(altura - ph, py))
    return px, py, pw, ph


def _escape_drawtext(texto: str) -> str:
    return (
        str(texto)
        .replace("\\", r"\\\\")
        .replace(":", r"\\:")
        .replace("'", r"\\'")
        .replace("%", r"\\%")
        .replace("\n", r"\\n")
    )


def _cor_ffmpeg(cor: str, opacidade: float = 1.0) -> str:
    cor = str(cor or "#ffffff").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", cor):
        cor = "0x" + cor[1:]
    return f"{cor}@{max(0.0, min(1.0, float(opacidade))):.3f}"


def _filtro_fit(rotulo: str, largura: int, altura: int, fit: str) -> str:
    fit = fit or "contain"
    if fit == "fill":
        return f"[{rotulo}]scale={largura}:{altura}[{rotulo}_fit]"
    if fit == "cover":
        return (
            f"[{rotulo}]scale={largura}:{altura}:force_original_aspect_ratio=increase,"
            f"crop={largura}:{altura}[{rotulo}_fit]"
        )
    return (
        f"[{rotulo}]scale={largura}:{altura}:force_original_aspect_ratio=decrease,"
        f"pad={largura}:{altura}:(ow-iw)/2:(oh-ih)/2:color=black@0[{rotulo}_fit]"
    )


def _audio_silencio(rotulo: str, duracao: float) -> str:
    return (
        f"anullsrc=r=48000:cl=stereo,atrim=duration={duracao:.8f},"
        f"asetpts=PTS-STARTPTS[{rotulo}]"
    )


def _normalizar_audio(rotulo_in: str, rotulo_out: str, fator: float = 1.0, volume: float = 1.0) -> str:
    cadeia = [f"[{rotulo_in}]aresample=48000", "aformat=sample_fmts=fltp:channel_layouts=stereo"]
    if abs(float(fator) - 1.0) > 1e-6:
        cadeia.append(_atempo_cadeia(fator))
    if abs(float(volume) - 1.0) > 1e-6:
        cadeia.append(f"volume={max(0.0, float(volume)):.6f}")
    cadeia.append(f"asetpts=PTS-STARTPTS[{rotulo_out}]")
    return ",".join(cadeia)



def _rgba_cor(cor: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    cor = str(cor or "#ff3b30").strip()
    m = re.fullmatch(r"#([0-9a-fA-F]{6})", cor)
    if not m:
        cor = "#ff3b30"
    v = cor.lstrip("#")
    r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    return r, g, b, int(round(max(0.0, min(1.0, float(alpha))) * 255))


def _gerar_forma_png(params: dict, largura: int, altura: int, caminho: Path) -> None:
    """Gera uma camada PNG transparente para Forma/Destaque."""
    try:
        from PIL import Image, ImageDraw
    except Exception as erro:
        raise RuntimeError(
            "A operação Forma/Destaque requer Pillow. Instale com: pip install pillow"
        ) from erro

    img = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    x, y, w, h = _regiao_pixels(params.get("region"), largura, altura)
    tipo = str(params.get("shape", "rectangle"))
    cor = str(params.get("color", "#ff3b30"))
    opacidade = max(0.0, min(1.0, float(params.get("opacity", 1.0))))
    fill_op = max(0.0, min(1.0, float(params.get("fill_opacity", 0.15))))
    preencher = bool(params.get("fill", False))
    proporcao = params.get("stroke_ratio")
    if proporcao is not None:
        esp = max(1, int(round(float(proporcao) * altura)))
    else:
        esp = max(1, int(round(float(params.get("thickness", 4)))))

    stroke = _rgba_cor(cor, opacidade)
    fill = _rgba_cor(cor, fill_op) if preencher else None
    x2, y2 = x + w - 1, y + h - 1

    if tipo == "rectangle":
        draw.rectangle((x, y, x2, y2), outline=stroke, width=esp, fill=fill)
    elif tipo == "rounded_rectangle":
        raio = max(2, min(w, h) // 8)
        draw.rounded_rectangle((x, y, x2, y2), radius=raio, outline=stroke, width=esp, fill=fill)
    elif tipo == "ellipse":
        draw.ellipse((x, y, x2, y2), outline=stroke, width=esp, fill=fill)
    elif tipo == "circle":
        lado = max(2, min(w, h))
        cx, cy = x + w // 2, y + h // 2
        bx1, by1 = cx - lado // 2, cy - lado // 2
        bx2, by2 = bx1 + lado - 1, by1 + lado - 1
        draw.ellipse((bx1, by1, bx2, by2), outline=stroke, width=esp, fill=fill)
    elif tipo == "line":
        yy = y + h // 2
        draw.line((x, yy, x2, yy), fill=stroke, width=esp)
    elif tipo == "underline":
        yy = max(y, y2 - max(esp, int(round(h * 0.08))))
        draw.line((x, yy, x2, yy), fill=stroke, width=esp)
    elif tipo == "arrow":
        direcao = str(params.get("direction", "right"))
        margem = max(esp * 2, 2)
        if direcao in {"left", "right"}:
            sy = y + h // 2
            if direcao == "right":
                p1, p2 = (x + margem, sy), (x2 - margem, sy)
            else:
                p1, p2 = (x2 - margem, sy), (x + margem, sy)
            draw.line((*p1, *p2), fill=stroke, width=esp)
            head = max(esp * 4, min(w, h) // 5)
            tx, ty = p2
            s = 1 if direcao == "right" else -1
            pts = [(tx, ty), (tx - s * head, ty - head // 2), (tx - s * head, ty + head // 2)]
            draw.polygon(pts, fill=stroke)
        else:
            sx = x + w // 2
            if direcao == "down":
                p1, p2 = (sx, y + margem), (sx, y2 - margem)
            else:
                p1, p2 = (sx, y2 - margem), (sx, y + margem)
            draw.line((*p1, *p2), fill=stroke, width=esp)
            head = max(esp * 4, min(w, h) // 5)
            tx, ty = p2
            s = 1 if direcao == "down" else -1
            pts = [(tx, ty), (tx - head // 2, ty - s * head), (tx + head // 2, ty - s * head)]
            draw.polygon(pts, fill=stroke)
    elif tipo == "highlighter":
        alpha = fill_op if fill_op > 0 else 0.28
        draw.rectangle((x, y, x2, y2), fill=_rgba_cor(cor, alpha))
    else:
        draw.rectangle((x, y, x2, y2), outline=stroke, width=esp, fill=fill)

    img.save(caminho, "PNG")


def _renderizar_segmento_fonte(
    entrada: Path,
    saida: Path,
    inicio: float,
    fim: float,
    operacoes: list[dict],
    info_fonte: dict,
    raiz: Path,
    crf: int,
    preset: str,
    transcricao_legendas: dict | None = None,
    config_legendas: dict | None = None,
) -> None:
    """Renderiza um intervalo da fonte com o conjunto constante de operações ativas."""
    duracao_fonte = max(0.001, fim - inicio)
    largura = int(info_fonte["largura"] or 1280)
    altura = int(info_fonte["altura"] or 720)
    fps = float(info_fonte["fps"] or 30.0)

    velocidade = 1.0
    for o in operacoes:
        if o.get("type") == "speed_segment":
            velocidade *= max(0.1, min(16.0, float((o.get("params") or {}).get("factor", 1.0))))
    velocidade = max(0.1, min(16.0, velocidade))
    duracao_saida = duracao_fonte / velocidade

    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{inicio:.8f}", "-t", f"{duracao_fonte:.8f}", "-i", str(entrada)]
    filtros: list[str] = []

    # Vídeo base.
    if abs(velocidade - 1.0) > 1e-6:
        filtros.append(f"[0:v]setpts=(PTS-STARTPTS)/{velocidade:.8f}[vbase0]")
    else:
        filtros.append("[0:v]setpts=PTS-STARTPTS[vbase0]")
    v_atual = "vbase0"

    # Se apenas o vídeo secundário deve rodar, congela visualmente a base.
    op_video = next((o for o in operacoes if o.get("type") == "overlay_video"), None)
    if op_video and (op_video.get("params") or {}).get("playback_policy") == "overlay_only":
        # O filtro overlay usa o vídeo principal como relógio. Apenas tpad
        # sobre um único frame podia resultar em um stream com efetivamente
        # um só frame, congelando também o vídeo 2. Aqui criamos um stream
        # CFR real, repetindo o frame principal na taxa do projeto.
        filtros.append(
            f"[{v_atual}]trim=end_frame=1,"
            f"loop=loop=-1:size=1:start=0,"
            f"setpts=N/({fps:.8f}*TB),"
            f"trim=duration={duracao_saida:.8f}[vfreeze]"
        )
        v_atual = "vfreeze"

    # Crop e zoom alteram o enquadramento antes das sobreposições.
    # A janela sempre contém a região inteira marcada no editor.
    for indice, o in enumerate(operacoes):
        p = o.get("params") or {}
        if o.get("type") == "crop_region":
            x, y, w, h = _janela_enquadramento_pixels(p.get("region"), largura, altura, fator=None)
            novo = f"vcrop{indice}"
            filtros.append(f"[{v_atual}]crop={w}:{h}:{x}:{y},scale={largura}:{altura}[{novo}]")
            v_atual = novo
        elif o.get("type") == "zoom_region":
            fator = max(1.0, min(5.0, float(p.get("factor", 1.5))))
            x, y, w, h = _janela_enquadramento_pixels(p.get("region"), largura, altura, fator=fator)
            novo = f"vzoom{indice}"
            filtros.append(f"[{v_atual}]crop={w}:{h}:{x}:{y},scale={largura}:{altura}[{novo}]")
            v_atual = novo

    # Áudio base.
    if info_fonte.get("possui_audio"):
        filtros.append(_normalizar_audio("0:a", "abase", fator=velocidade, volume=1.0))
    else:
        filtros.append(_audio_silencio("abase", duracao_saida))
    a_atual = "abase"

    if any(o.get("type") == "mute" for o in operacoes):
        filtros.append(f"[{a_atual}]volume=0[amute]")
        a_atual = "amute"

    proximo_input = 1
    audio_overlay_rotulo: str | None = None
    op_overlay_video: dict | None = None

    # Primeiro prepara entradas de imagem/vídeo adicionais e aplica efeitos na ordem.
    for indice, o in enumerate(operacoes):
        tipo = o.get("type")
        p = o.get("params") or {}
        if tipo == "blur_region":
            x, y, w, h = _regiao_pixels(p.get("region"), largura, altura)
            intensidade = max(1.0, min(80.0, float(p.get("intensity", 16.0))))
            base = f"vbl_base{indice}"
            pedaco = f"vbl_piece{indice}"
            novo = f"vbl{indice}"
            filtros.append(f"[{v_atual}]split=2[{base}][{pedaco}]")
            filtros.append(f"[{pedaco}]crop={w}:{h}:{x}:{y},boxblur=luma_radius={intensidade:.3f}:luma_power=1[blur{indice}]")
            filtros.append(f"[{base}][blur{indice}]overlay={x}:{y}[{novo}]")
            v_atual = novo

        elif tipo == "black_bar":
            x, y, w, h = _regiao_pixels(p.get("region"), largura, altura)
            opacity = max(0.0, min(1.0, float(p.get("opacity", 1.0))))
            novo = f"vbar{indice}"
            filtros.append(f"[{v_atual}]drawbox=x={x}:y={y}:w={w}:h={h}:color=black@{opacity:.3f}:t=fill[{novo}]")
            v_atual = novo

        elif tipo == "overlay_text":
            x, y, w, h = _regiao_pixels(p.get("region"), largura, altura)
            texto = _escape_drawtext(p.get("text", ""))
            proporcao_fonte = p.get("font_size_ratio")
            if proporcao_fonte is not None:
                tamanho = max(8, min(int(altura * 0.5), int(round(float(proporcao_fonte) * altura))))
            else:
                # Compatibilidade com projetos antigos.
                tamanho = max(8, min(240, int(float(p.get("font_size", 36)))))
            cor = _cor_ffmpeg(p.get("color", "#ffffff"), 1.0)
            bg_op = max(0.0, min(1.0, float(p.get("background_opacity", 0.0))))
            bg = _cor_ffmpeg(p.get("background_color", "#000000"), bg_op)
            fonte = str(p.get("font", "Arial")).lower()
            fontfile = ""
            if fonte == "monospace":
                fontfile = ":font='monospace'"
            elif fonte == "serif":
                fontfile = ":font='serif'"
            else:
                fontfile = ":font='sans-serif'"
            novo = f"vtxt{indice}"
            filtros.append(
                f"[{v_atual}]drawtext=text='{texto}'{fontfile}:fontcolor={cor}:fontsize={tamanho}:"
                f"x={x}+({w}-text_w)/2:y={y}+({h}-text_h)/2:"
                f"box={'1' if bg_op > 0 else '0'}:boxcolor={bg}:boxborderw=6[{novo}]"
            )
            v_atual = novo

        elif tipo == "shape_highlight":
            camada = saida.with_name(f"{saida.stem}_shape_{indice}.png")
            _gerar_forma_png(p, largura, altura, camada)
            args.extend(["-loop", "1", "-t", f"{duracao_saida:.8f}", "-i", str(camada)])
            idx = proximo_input
            proximo_input += 1
            filtros.append(f"[{idx}:v]format=rgba[shape{indice}]")
            novo = f"vshape{indice}"
            filtros.append(f"[{v_atual}][shape{indice}]overlay=0:0:shortest=1[{novo}]")
            v_atual = novo

        elif tipo == "overlay_image":
            midia = _resolver_midia_projeto(raiz, p.get("media", ""))
            # PNG/JPG são imagens estáticas e usam -loop 1. GIF é uma
            # mídia animada: -loop não pertence ao demuxer GIF e fazia o
            # FFmpeg abortar com "Option loop not found". Para GIF,
            # repetimos o stream animado e limitamos ao intervalo da operação.
            if midia.suffix.lower() == ".gif":
                args.extend(["-stream_loop", "-1", "-t", f"{duracao_saida:.8f}", "-i", str(midia)])
            else:
                args.extend(["-loop", "1", "-t", f"{duracao_saida:.8f}", "-i", str(midia)])
            x, y, w, h = _regiao_pixels(p.get("region"), largura, altura)
            entrada_label = f"{proximo_input}:v"
            fit_label = f"img{indice}"
            filtros.append(_filtro_fit(entrada_label, w, h, p.get("fit", "contain")).replace(f"[{entrada_label}_fit]", f"[{fit_label}]") )
            novo = f"vimg{indice}"
            filtros.append(f"[{v_atual}][{fit_label}]overlay={x}:{y}:shortest=1[{novo}]")
            v_atual = novo
            proximo_input += 1

        elif tipo == "overlay_video" and op_overlay_video is None:
            op_overlay_video = o
            midia = _resolver_midia_projeto(raiz, p.get("media", ""))
            info_m = informacoes_video(midia)
            media_in = max(0.0, float(p.get("media_in", 0.0)))
            # Mapeia o offset do segmento atual para o vídeo 2.
            offset_fonte = max(0.0, inicio - float(o.get("start", inicio)))
            media_start = media_in + offset_fonte
            media_out = p.get("media_out")
            dur_disp = duracao_fonte
            if media_out is not None:
                dur_disp = min(dur_disp, max(0.001, float(media_out) - media_start))
            playback = p.get("playback_policy", "both")
            if playback == "base_only":
                args.extend(["-ss", f"{media_start:.8f}", "-i", str(midia)])
            else:
                args.extend(["-ss", f"{media_start:.8f}", "-t", f"{dur_disp:.8f}", "-i", str(midia)])
            idx = proximo_input
            proximo_input += 1
            modo = p.get("presentation_mode", "overlay")
            fit = p.get("fit", "contain")

            if playback == "base_only":
                # Congela apenas o vídeo 2, mas mantém um stream de quadros
                # válido durante todo o intervalo.
                filtros.append(
                    f"[{idx}:v]trim=end_frame=1,"
                    f"loop=loop=-1:size=1:start=0,"
                    f"setpts=N/({fps:.8f}*TB),"
                    f"trim=duration={duracao_saida:.8f}[ovraw{indice}]"
                )
            else:
                filtros.append(f"[{idx}:v]setpts=(PTS-STARTPTS)/{velocidade:.8f}[ovraw{indice}]")

            if modo == "side_by_side":
                filtros.append(_filtro_fit(v_atual, largura // 2, altura, "contain").replace(f"[{v_atual}_fit]", f"[left{indice}]"))
                filtros.append(_filtro_fit(f"ovraw{indice}", largura // 2, altura, fit).replace(f"[ovraw{indice}_fit]", f"[right{indice}]"))
                novo = f"vside{indice}"
                filtros.append(f"[left{indice}][right{indice}]hstack=inputs=2[{novo}]")
                v_atual = novo
            elif modo == "broll":
                filtros.append(_filtro_fit(f"ovraw{indice}", largura, altura, fit).replace(f"[ovraw{indice}_fit]", f"[vbroll{indice}]"))
                v_atual = f"vbroll{indice}"
            else:
                x, y, w, h = _regiao_pixels(p.get("region"), largura, altura)
                filtros.append(_filtro_fit(f"ovraw{indice}", w, h, fit).replace(f"[ovraw{indice}_fit]", f"[ovfit{indice}]"))
                novo = f"vov{indice}"
                filtros.append(f"[{v_atual}][ovfit{indice}]overlay={x}:{y}:eof_action=pass[{novo}]")
                v_atual = novo

            # Só cria a cadeia de áudio do vídeo 2 quando ela será realmente usada.
            # Na v6.4, mesmo com política "Só principal", a saída [aovX] era
            # criada e ficava desconectada do filter_complex, fazendo o FFmpeg
            # abortar com "unconnected output".
            audio_policy = p.get("audio_policy", "base")
            if (
                info_m.get("possui_audio")
                and playback != "base_only"
                and audio_policy in {"overlay", "both"}
            ):
                volume_ov = max(0.0, min(1.0, float(p.get("overlay_volume", 100)) / 100.0))
                filtros.append(_normalizar_audio(f"{idx}:a", f"aov{indice}", fator=velocidade, volume=volume_ov))
                audio_overlay_rotulo = f"aov{indice}"

    # Política de áudio do vídeo sobreposto.
    if op_overlay_video is not None:
        p = op_overlay_video.get("params") or {}
        policy = p.get("audio_policy", "base")
        base_volume = max(0.0, min(1.0, float(p.get("base_volume", 100)) / 100.0))
        a_base: str | None = None

        # A cadeia de áudio principal só é criada quando a política realmente
        # a utiliza. Isso evita saídas de filtro desconectadas no FFmpeg.
        if policy in {"base", "both"}:
            filtros.append(f"[{a_atual}]volume={base_volume:.6f}[abasevol]")
            a_base = "abasevol"

        if policy == "none":
            # O áudio principal já foi preparado acima; descarte-o explicitamente
            # para não deixar uma saída de filtro órfã.
            filtros.append(f"[{a_atual}]anullsink")
            filtros.append(_audio_silencio("aout", duracao_saida))
            a_atual = "aout"
        elif policy == "overlay":
            filtros.append(f"[{a_atual}]anullsink")
            if audio_overlay_rotulo:
                a_atual = audio_overlay_rotulo
            else:
                filtros.append(_audio_silencio("aout", duracao_saida))
                a_atual = "aout"
        elif policy == "both" and audio_overlay_rotulo and a_base:
            filtros.append(f"[{a_base}][{audio_overlay_rotulo}]amix=inputs=2:duration=first:normalize=0[aout]")
            a_atual = "aout"
        elif a_base:
            a_atual = a_base

    # Áudio de fundo/inserido. Pode substituir ou misturar com o áudio
    # resultante do vídeo principal/vídeo sobreposto.
    for indice, o in enumerate(operacoes):
        if o.get("type") != "background_audio":
            continue
        p = o.get("params") or {}
        policy = p.get("audio_policy", "both")
        base_volume = max(0.0, min(1.0, float(p.get("base_volume", 100)) / 100.0))
        bg_volume = max(0.0, min(1.0, float(p.get("overlay_volume", 30)) / 100.0))

        if policy == "base":
            if abs(base_volume - 1.0) > 1e-6:
                filtros.append(f"[{a_atual}]volume={base_volume:.6f}[abgbase{indice}]")
                a_atual = f"abgbase{indice}"
            continue
        if policy == "none":
            filtros.append(f"[{a_atual}]anullsink")
            filtros.append(_audio_silencio(f"abgsil{indice}", duracao_saida))
            a_atual = f"abgsil{indice}"
            continue

        midia = _resolver_midia_projeto(raiz, p.get("media", ""))
        info_bg = _informacoes_midia(midia)
        if not info_bg.get("possui_audio"):
            raise ValueError(f"A mídia de áudio não possui faixa de áudio: {midia.name}")

        media_in = max(0.0, float(p.get("media_in", 0.0)))
        # O áudio acompanha o mesmo relógio da fonte, inclusive em trechos
        # acelerados/desacelerados.
        offset_fonte = max(0.0, inicio - float(o.get("start", inicio)))
        media_start = media_in + offset_fonte
        media_out = p.get("media_out")
        disponivel = max(0.001, float(info_bg.get("duracao", 0.0)) - media_start)
        if media_out is not None:
            disponivel = min(disponivel, max(0.001, float(media_out) - media_start))
        dur_input = min(duracao_fonte, disponivel)

        args.extend(["-ss", f"{media_start:.8f}", "-t", f"{dur_input:.8f}", "-i", str(midia)])
        idx_bg = proximo_input
        proximo_input += 1
        rot_bg = f"abg{indice}"
        filtros.append(
            f"[{idx_bg}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"{_atempo_cadeia(velocidade)},volume={bg_volume:.6f},asetpts=PTS-STARTPTS,"
            f"apad,atrim=duration={duracao_saida:.8f}[{rot_bg}]"
        )

        if policy == "overlay":
            filtros.append(f"[{a_atual}]anullsink")
            a_atual = rot_bg
        else:  # both
            base_rot = f"abgbase{indice}"
            mix_rot = f"abgmix{indice}"
            filtros.append(f"[{a_atual}]volume={base_volume:.6f}[{base_rot}]")
            filtros.append(f"[{base_rot}][{rot_bg}]amix=inputs=2:duration=first:normalize=0[{mix_rot}]")
            a_atual = mix_rot

    # Legenda automática da transcrição pertence ao vídeo matriz/principal.
    # É aplicada depois das sobreposições visuais, portanto fica legível no resultado,
    # mas não é adicionada aos segmentos de vídeo inserido/substituto.
    v_atual = _aplicar_legendas_no_segmento(
        filtros, v_atual, transcricao_legendas, config_legendas,
        inicio, fim, velocidade, largura, altura,
    )

    # Normalização final do vídeo e áudio.
    filtros.append(f"[{v_atual}]fps={fps:.6f},scale={largura}:{altura},format=yuv420p[vout]")
    filtros.append(f"[{a_atual}]atrim=duration={duracao_saida:.8f},asetpts=PTS-STARTPTS[aout_final]")

    comando = args + [
        "-filter_complex", ";".join(filtros),
        "-map", "[vout]", "-map", "[aout_final]",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(saida),
    ]
    _executar_ffmpeg_render(comando, "renderizar um trecho do vídeo principal")


def _renderizar_segmento_midia(
    midia: Path,
    saida: Path,
    media_in: float,
    media_out: float | None,
    largura: int,
    altura: int,
    fps: float,
    fit: str,
    audio_policy: str,
    volume: float,
    duracao_alvo: float | None,
    crf: int,
    preset: str,
) -> None:
    info = informacoes_video(midia)
    media_in = max(0.0, media_in)
    if media_out is None:
        media_out = float(info["duracao"])
    media_out = min(float(info["duracao"]), max(media_in + 0.001, float(media_out)))
    duracao_natural = media_out - media_in
    fator = 1.0
    if duracao_alvo is not None and duracao_alvo > 0.001:
        fator = duracao_natural / duracao_alvo
    duracao_saida = duracao_natural / fator

    filtros: list[str] = []
    filtros.append(f"[0:v]setpts=(PTS-STARTPTS)/{fator:.8f}[mv0]")
    filtros.append(_filtro_fit("mv0", largura, altura, fit).replace("[mv0_fit]", "[mv1]"))
    filtros.append(f"[mv1]fps={fps:.6f},format=yuv420p[vout]")

    if info.get("possui_audio") and audio_policy not in {"none", "base"}:
        filtros.append(_normalizar_audio("0:a", "ma0", fator=fator, volume=max(0.0, min(1.0, volume))))
        a_atual = "ma0"
    else:
        filtros.append(_audio_silencio("ma0", duracao_saida))
        a_atual = "ma0"
    filtros.append(f"[{a_atual}]atrim=duration={duracao_saida:.8f},asetpts=PTS-STARTPTS[aout]")

    comando = [
        "ffmpeg", "-y", "-ss", f"{media_in:.8f}", "-t", f"{duracao_natural:.8f}", "-i", str(midia),
        "-filter_complex", ";".join(filtros), "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(saida),
    ]
    _executar_ffmpeg_render(comando, "renderizar um vídeo inserido/substituto")


def processar_projeto(
    entrada: str | Path,
    projeto: str | Path | dict | None = None,
    saida: str | Path | None = None,
    crf: int = 18,
    preset: str = "medium",
    manter_temporarios: bool = False,
) -> Path:
    """
    Renderiza um projeto v6.8 em MP4 usando FFmpeg.

    A implementação prioriza robustez para teste: divide a timeline em pequenos
    segmentos, aplica as operações de cada trecho e concatena os segmentos já
    normalizados. Isso é mais lento que um filter_complex monolítico, mas torna
    cortes, velocidade, inserção/substituição e sobreposições mais previsvisíveis.
    """
    entrada = Path(entrada).resolve()
    if not entrada.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {entrada}")
    raiz = entrada.parent.resolve()

    if projeto is None:
        caminho = _caminho_padrao_projeto(entrada)
        dados = carregar_projeto(caminho)
    elif isinstance(projeto, dict):
        dados = projeto
    else:
        dados = carregar_projeto(projeto)

    if saida is None:
        saida = entrada.with_name(f"{entrada.stem}_editado.mp4")
    else:
        saida = Path(saida)
        if not saida.is_absolute():
            saida = raiz / saida
        saida = saida.resolve()
    try:
        saida.relative_to(raiz)
    except ValueError as erro:
        raise ValueError("A saída deve ficar na pasta do vídeo ou em subpasta.") from erro
    saida.parent.mkdir(parents=True, exist_ok=True)

    info = informacoes_video(entrada)
    duracao = float(info["duracao"])
    largura = int(info["largura"] or 1280)
    altura = int(info["altura"] or 720)
    fps = float(info["fps"] or 30.0)

    operacoes = [o for o in dados.get("operations", []) if o.get("enabled", True)]

    config_legendas = dados.get("subtitles") if isinstance(dados.get("subtitles"), dict) else {}
    transcricao_legendas: dict | None = None
    if bool(config_legendas.get("burn_in", False)):
        caminho_legendas = _caminho_padrao_transcricao(entrada)
        if not caminho_legendas.exists():
            raise FileNotFoundError(
                "A opção de incorporar legenda está ativa, mas a transcrição não foi encontrada: "
                f"{caminho_legendas.name}. Abra o editor/transcreva o vídeo ou desative 'Incorporar no MP4'."
            )
        transcricao_legendas = carregar_transcricao(caminho_legendas)

    # Sanidade de substituições: não permite sobreposição de duas substituições.
    replaces = sorted((o for o in operacoes if o.get("type") == "replace_video"), key=lambda o: float(o.get("start", 0)))
    for a, b in zip(replaces, replaces[1:]):
        if float(b.get("start", 0)) < float(a.get("end", 0)) - 1e-6:
            raise ValueError("Há operações 'Substituir por vídeo' sobrepostas. Ajuste os intervalos.")

    limites = {0.0, duracao}
    tipos_intervalo = {"cut", "speed_segment", "mute", "blur_region", "black_bar", "overlay_text", "overlay_image", "overlay_video", "shape_highlight", "zoom_region", "crop_region", "replace_video", "background_audio"}
    for o in operacoes:
        t = o.get("type")
        a = max(0.0, min(duracao, float(o.get("start", 0.0))))
        b = max(0.0, min(duracao, float(o.get("end", a))))
        limites.add(a)
        if t in tipos_intervalo:
            limites.add(b)
        # Mídias temporais podem acabar antes do intervalo marcado.
        if t in {"overlay_video", "background_audio"}:
            p = o.get("params") or {}
            mi, mo = p.get("media_in"), p.get("media_out")
            if mi is not None and mo is not None and float(mo) > float(mi):
                limites.add(min(b, a + (float(mo) - float(mi))))
    pontos = sorted(x for x in limites if 0 <= x <= duracao)

    pasta_tmp_obj = tempfile.TemporaryDirectory(prefix="editor_video_v66_", dir=str(raiz))
    pasta_tmp = Path(pasta_tmp_obj.name)
    segmentos: list[Path] = []
    contador = 0

    def novo_segmento() -> Path:
        nonlocal contador
        contador += 1
        return pasta_tmp / f"seg_{contador:05d}.mp4"

    def inserir_operacoes_no_ponto(t: float) -> None:
        # Vídeo sobreposto com "Só vídeo 2 roda": congela o principal no
        # ponto de início, toca o vídeo 2 na região selecionada e depois
        # retoma o principal do MESMO ponto. Isso acrescenta duração ao
        # resultado, em vez de "roubar" tempo do vídeo principal.
        for o in sorted(
            (
                x for x in operacoes
                if x.get("type") == "overlay_video"
                and (x.get("params") or {}).get("playback_policy") == "overlay_only"
                and abs(float(x.get("start", 0)) - t) < 1e-5
            ),
            key=lambda x: x.get("id", ""),
        ):
            p = o.get("params") or {}
            midia = _resolver_midia_projeto(raiz, p.get("media", ""))
            info_m = informacoes_video(midia)
            media_in = max(0.0, float(p.get("media_in", 0.0)))
            media_out = p.get("media_out")
            if media_out is None:
                media_out = float(info_m["duracao"])
            else:
                media_out = min(float(info_m["duracao"]), float(media_out))
            dur_midia = max(0.001, media_out - media_in)
            dur_marcada = max(0.001, float(o.get("end", t)) - float(o.get("start", t)))
            dur_pausa = min(dur_midia, dur_marcada)

            # Reutiliza o renderizador de trecho: ele já sabe congelar a base
            # quando playback_policy == overlay_only. O intervalo de fonte
            # abaixo serve apenas para fornecer o frame/áudio-base; depois a
            # timeline principal será renderizada normalmente a partir de t.
            fim_aux = min(duracao, t + dur_pausa)
            if fim_aux <= t + 1e-6:
                fim_aux = min(duracao, t + 0.05)
            seg = novo_segmento()
            _renderizar_segmento_fonte(
                entrada, seg, t, fim_aux, [o], info, raiz, crf, preset,
                transcricao_legendas=None, config_legendas=None,
            )
            segmentos.append(seg)

        for o in sorted((x for x in operacoes if x.get("type") == "insert_video" and abs(float(x.get("start", 0)) - t) < 1e-5), key=lambda x: x.get("id", "")):
            p = o.get("params") or {}
            midia = _resolver_midia_projeto(raiz, p.get("media", ""))
            seg = novo_segmento()
            _renderizar_segmento_midia(
                midia, seg,
                float(p.get("media_in", 0.0)),
                None if p.get("media_out") is None else float(p.get("media_out")),
                largura, altura, fps, p.get("fit", "contain"),
                p.get("audio_policy", "overlay"),
                max(0.0, min(1.0, float(p.get("overlay_volume", 100)) / 100.0)),
                None, crf, preset,
            )
            segmentos.append(seg)

        for o in sorted((x for x in replaces if abs(float(x.get("start", 0)) - t) < 1e-5), key=lambda x: x.get("id", "")):
            p = o.get("params") or {}
            midia = _resolver_midia_projeto(raiz, p.get("media", ""))
            alvo = None
            if p.get("duration_policy") == "fit_interval":
                alvo = max(0.001, float(o.get("end", t)) - float(o.get("start", t)))
            seg = novo_segmento()
            _renderizar_segmento_midia(
                midia, seg,
                float(p.get("media_in", 0.0)),
                None if p.get("media_out") is None else float(p.get("media_out")),
                largura, altura, fps, p.get("fit", "contain"),
                p.get("audio_policy", "overlay"),
                max(0.0, min(1.0, float(p.get("overlay_volume", 100)) / 100.0)),
                alvo, crf, preset,
            )
            segmentos.append(seg)

    def operacao_ativa_no_segmento(o: dict, a: float, b: float) -> bool:
        if o.get("type") in {"insert_video", "replace_video", "cut"}:
            return False
        # "Só vídeo 2 roda" é uma pausa da timeline principal. Ela é
        # renderizada como um segmento adicional no ponto de início e não
        # como uma operação que consome o trecho correspondente da fonte.
        if (
            o.get("type") == "overlay_video"
            and (o.get("params") or {}).get("playback_policy") == "overlay_only"
        ):
            return False
        if float(o.get("start", 0)) > a + 1e-6 or float(o.get("end", a)) < b - 1e-6:
            return False
        if o.get("type") in {"overlay_video", "background_audio"}:
            p = o.get("params") or {}
            mi, mo = p.get("media_in"), p.get("media_out")
            if mi is not None and mo is not None and float(mo) > float(mi):
                fim_midia_na_fonte = float(o.get("start", 0)) + (float(mo) - float(mi))
                if a >= fim_midia_na_fonte - 1e-6:
                    return False
        return True

    for i in range(len(pontos) - 1):
        a, b = pontos[i], pontos[i + 1]
        inserir_operacoes_no_ponto(a)
        if b - a < 0.015:
            continue

        # Substituição e corte eliminam o conteúdo principal desse intervalo.
        if any(o.get("type") == "replace_video" and float(o.get("start", 0)) <= a + 1e-6 and float(o.get("end", 0)) >= b - 1e-6 for o in operacoes):
            continue
        if any(o.get("type") == "cut" and float(o.get("start", 0)) <= a + 1e-6 and float(o.get("end", 0)) >= b - 1e-6 for o in operacoes):
            continue

        ativos = [o for o in operacoes if operacao_ativa_no_segmento(o, a, b)]
        seg = novo_segmento()
        _renderizar_segmento_fonte(
            entrada, seg, a, b, ativos, info, raiz, crf, preset,
            transcricao_legendas=transcricao_legendas, config_legendas=config_legendas,
        )
        segmentos.append(seg)

    # Inserções no fim do vídeo.
    inserir_operacoes_no_ponto(duracao)

    if not segmentos:
        pasta_tmp_obj.cleanup()
        raise ValueError("O projeto não produziu nenhum trecho de vídeo.")

    lista = pasta_tmp / "concat.txt"
    lista.write_text("\n".join(f"file '{p.as_posix().replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'" for p in segmentos), encoding="utf-8")
    comando_concat = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lista), "-c", "copy", "-movflags", "+faststart", str(saida)]
    try:
        _executar_ffmpeg_render(comando_concat, "concatenar os segmentos do projeto")
    except subprocess.CalledProcessError:
        # Fallback mais tolerante se algum timebase do concat demuxer divergir.
        comando_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lista),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(saida),
        ]
        _executar_ffmpeg_render(comando_concat, "concatenar os segmentos do projeto")

    if manter_temporarios:
        print(f"Temporários preservados em: {pasta_tmp}")
        pasta_tmp_obj.cleanup = lambda: None  # type: ignore[method-assign]
    else:
        pasta_tmp_obj.cleanup()
    return saida

class _VideoHandler(BaseHTTPRequestHandler):
    raiz: Path
    arquivo_original: Path | None = None
    caminho_transcricao: Path | None = None
    caminho_projeto: Path | None = None
    bloqueio_api = threading.Lock()
    pagina_editor: str | None = None

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

    def _responder_html(self, html: str) -> None:
        corpo = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-cache")
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
        caminho = urlparse(self.path).path
        if caminho in {"", "/"} and self.pagina_editor is not None:
            corpo = self.pagina_editor.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.send_header("Cache-Control", "no-cache")
            self._cabecalhos_cors()
            self.end_headers()
            return
        self._enviar(somente_cabecalho=True)

    def do_GET(self) -> None:
        caminho = urlparse(self.path).path
        if caminho in {"", "/"} and self.pagina_editor is not None:
            self._responder_html(self.pagina_editor)
            return
        if caminho.startswith("/__editor_api__/"):
            self._responder_json(405, {"ok": False, "erro": "Use POST."})
            return
        self._enviar(somente_cabecalho=False)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        rota = parsed.path
        try:
            # Upload binário: permite escolher arquivo no navegador e copiá-lo
            # para uma subpasta do projeto sem carregar tudo em memória.
            if rota == "/__editor_api__/upload":
                nome = Path((parse_qs(parsed.query).get("name") or ["midia.bin"])[0]).name
                if not nome or nome in {".", ".."}:
                    raise ValueError("Nome de arquivo inválido.")
                tamanho = int(self.headers.get("Content-Length", "0") or "0")
                if tamanho <= 0:
                    raise ValueError("Arquivo vazio.")
                pasta = (self.raiz / "editor_midias").resolve()
                pasta.mkdir(parents=True, exist_ok=True)
                destino = (pasta / nome).resolve()
                destino.relative_to(pasta)
                restante = tamanho
                with destino.open("wb") as f:
                    while restante > 0:
                        bloco = self.rfile.read(min(1024 * 1024, restante))
                        if not bloco:
                            break
                        f.write(bloco)
                        restante -= len(bloco)
                if restante != 0:
                    destino.unlink(missing_ok=True)
                    raise IOError("Upload interrompido antes do fim.")
                self._responder_json(200, {"ok": True, "arquivo": f"editor_midias/{nome}"})
                return

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

                if rota == "/__editor_api__/render":
                    if self.arquivo_original is None:
                        self._responder_json(400, {"ok": False, "erro": "Vídeo original indisponível."})
                        return
                    projeto = dados.get("project")
                    if not isinstance(projeto, dict):
                        raise ValueError("Projeto inválido para renderização.")
                    nome_saida = Path(str(dados.get("output") or f"{self.arquivo_original.stem}_editado.mp4")).name
                    if not nome_saida.lower().endswith(".mp4"):
                        nome_saida += ".mp4"
                    destino = self.arquivo_original.parent / nome_saida
                    if self.caminho_projeto is not None:
                        salvar_projeto(self.caminho_projeto, projeto)
                    arquivo = processar_projeto(self.arquivo_original, projeto=projeto, saida=destino)
                    self._responder_json(200, {"ok": True, "arquivo": arquivo.name})
                    return

            self._responder_json(404, {"ok": False, "erro": "Rota desconhecida."})
        except Exception as erro:
            self._responder_json(400, {"ok": False, "erro": str(erro)})


def _iniciar_servidor(
    arquivo: Path,
    caminho_transcricao: Path | None = None,
    caminho_projeto: Path | None = None,
    arquivo_original: Path | None = None,
) -> tuple[str, str]:
    raiz = arquivo.parent.resolve()
    classe_handler = type(
        f"VideoHandler_{uuid.uuid4().hex}",
        (_VideoHandler,),
        {
            "raiz": raiz,
            "arquivo_original": (arquivo_original or arquivo).resolve(),
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


def _executando_no_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _iniciar_servidor_colab(
    arquivo: Path,
    caminho_transcricao: Path | None = None,
    caminho_projeto: Path | None = None,
    arquivo_original: Path | None = None,
) -> tuple[int, type[_VideoHandler]]:
    """Servidor do editor para o proxy/iframe do Google Colab.

    A página do editor é instalada em ``classe_handler.pagina_editor`` depois
    que o HTML é montado. URLs de vídeo, mídia e API são relativas à própria
    página, portanto atravessam o proxy do Colab sem depender de localhost no
    navegador do usuário.
    """
    raiz = arquivo.parent.resolve()
    classe_handler = type(
        f"VideoHandlerColab_{uuid.uuid4().hex}",
        (_VideoHandler,),
        {
            "raiz": raiz,
            "arquivo_original": (arquivo_original or arquivo).resolve(),
            "caminho_transcricao": caminho_transcricao,
            "caminho_projeto": caminho_projeto,
            "bloqueio_api": threading.Lock(),
            "pagina_editor": None,
        },
    )
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), classe_handler)
    thread = threading.Thread(target=servidor.serve_forever, daemon=True)
    thread.start()
    _SERVIDORES.append(servidor)
    return int(servidor.server_address[1]), classe_handler


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
    Marcador v6.9.3 para Jupyter e Google Colab.

    Mantém os recursos da v5 e acrescenta:
      - edição de palavras da transcrição preservando timestamps;
      - salvamento das correções no JSON da transcrição;
      - legenda sincronizada sobre o player + exportação SRT/VTT;
      - projeto JSON de operações não destrutivas;
      - prévia do projeto com cortes, aceleração/desaceleração por trecho,
        inserções/substituições, áudio de vídeo 2 e áudio de fundo;
      - seleção do trecho de origem (início/fim) de vídeo ou áudio secundário;
      - operações adicionais registráveis e com prévia visual (mute, velocidade, blur,
        texto, imagem, vídeo sobreposto, inserção/substituição, zoom, crop e tarja);
      - caixa visual arrastável/redimensionável com coordenadas normalizadas.

    Importante: o marcador mantém a lista de cortes legada, e a v6.9.3 também
    consegue renderizar o projeto atual por FFmpeg com processar_projeto() ou
    pela função processar_projeto() em uma célula do Jupyter.

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
            "version": 4,
            "source": original.name,
            "time_reference": "source",
            "region_reference": "video_frame",
            "text_size_reference": "frame_height",
            "subtitles": {
                "burn_in": False,
                "position": "baixo",
                "font_size": 22,
                "font_size_ratio": None,
            },
            "operations": [],
        }

    modo_colab = _executando_no_colab()
    exibido = _preparar_preview_para_ambiente(
        original=original,
        info_original=info_original,
        preparar_preview=preparar_preview,
        modo_colab=modo_colab,
    )
    info_exibido = informacoes_video(exibido)
    porta_colab: int | None = None
    handler_colab: type[_VideoHandler] | None = None
    if modo_colab:
        porta_colab, handler_colab = _iniciar_servidor_colab(
            exibido,
            caminho_transcricao=caminho_transcricao if dados_transcricao is not None else None,
            caminho_projeto=caminho_projeto,
            arquivo_original=original,
        )
        # A página será servida na raiz do próprio servidor. Tudo relativo.
        origem = quote(exibido.name)
        api_base = "."
    else:
        origem, api_base = _iniciar_servidor(
            exibido,
            caminho_transcricao=caminho_transcricao if dados_transcricao is not None else None,
            caminho_projeto=caminho_projeto,
            arquivo_original=original,
        )
    identificador = "mc_" + uuid.uuid4().hex[:10]

    resumo_codec = f'{info_original["codec_video"]} / {info_original["pixel_format"]}'
    if info_original["codec_audio"]:
        resumo_codec += f' / áudio {info_original["codec_audio"]}'

    aviso_preview = (
        f'<div class="aviso">'
        f'{"Colab: preview leve/faststart" if modo_colab else "Visualização por cópia compatível"}: '
        f'<strong>{html_lib.escape(exibido.name)}</strong>. '
        f'O original não foi alterado; o render final usa o original.</div>'
        if exibido != original else ""
    )

    aviso_v6 = (
        '<div class="info-v6"><strong>v6.9.3:</strong> interface compacta + prévia + renderização FFmpeg. '
        'Correções da transcrição são salvas automaticamente; mídias podem ser escolhidas pelo botão Selecionar arquivo; '
        'Zoom/Crop preservam integralmente a região marcada; Forma/Destaque inclui retângulo, círculo, elipse, linha, sublinhado, seta e marca-texto. A prévia do projeto move o player para junto dos controles de prévia. Áudio de fundo pode ser recortado e misturado por intervalo. A legenda corrigida pode ser incorporada ao vídeo principal, com tamanho e posição configuráveis. O MP4 final é gerado pela função processar_projeto() no Jupyter.</div>'
    )

    modelo = r"""
<div id="__ID__" class="marcador-cortes-v65">
<style>
#__ID__ {max-width:__WIDTH__px;padding:8px;border:1px solid #bbb;border-radius:8px;font-family:Arial,sans-serif;font-size:9.5px;line-height:1.2}
#__ID__ * {box-sizing:border-box}
#__ID__ .aviso, #__ID__ .info-v6, #__ID__ .status {margin:5px 0;padding:5px 7px;border:1px solid #d0a000;border-radius:5px;font-size:9.5px}
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
#__ID__ .camada-sequencial {position:absolute;inset:0;z-index:14;pointer-events:none;overflow:hidden;background:transparent}
#__ID__ .camada-sequencial.ativa {background:#000}
#__ID__ .camada-sequencial video {position:absolute;display:block;background:#000;object-fit:contain}
#__ID__ .preview-op.preview-side {inset:0!important;background:#000}
#__ID__ .preview-op.preview-side video {position:absolute;top:0;width:50%;height:100%;object-fit:contain;background:#000}
#__ID__ .preview-op.preview-side video.base-clone {left:0}
#__ID__ .preview-op.preview-side video.overlay-clone {right:0}
#__ID__ .preview-op.preview-broll {inset:0!important;background:#000}
#__ID__ .preview-op.preview-broll video {width:100%;height:100%;object-fit:contain}
#__ID__ .modo-preview {font-weight:700}
#__ID__ .preview-op {position:absolute;overflow:hidden;line-height:1.2;pointer-events:none}
#__ID__ .preview-op img, #__ID__ .preview-op video {width:100%;height:100%;display:block;background:transparent}
#__ID__ .preview-op .preview-texto {width:100%;height:100%;display:flex;align-items:center;justify-content:center;text-align:center;color:#fff;font-weight:700;font-size:clamp(11px,1.4vw,22px);text-shadow:0 1px 3px #000;padding:4px;white-space:pre-wrap;overflow:hidden}
#__ID__ .preview-op.preview-blur {background:rgba(255,255,255,.025);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}
#__ID__ .preview-op.preview-tarja {background:#000}
#__ID__ .preview-op .forma-svg, #__ID__ .preview-edicao .forma-svg {width:100%;height:100%;display:block;overflow:visible}
#__ID__ .preview-local {margin-top:6px;padding:5px;border:1px dashed #aaa;border-radius:6px;background:rgba(127,127,127,.035)}
#__ID__ .preview-local[hidden] {display:none}
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
#__ID__ button {display:inline-flex;align-items:center;gap:3px;padding:3px 5px;min-height:24px;cursor:pointer;white-space:nowrap;font-size:9.5px}
#__ID__ button.ativo {outline:1.5px solid currentColor;outline-offset:1px}
#__ID__ button:disabled {opacity:.5;cursor:not-allowed}
#__ID__ kbd {padding:0 2px;border:1px solid #aaa;border-radius:2px;font-family:Consolas,monospace;font-size:7.5px;opacity:.65}
#__ID__ .marcas {margin:4px 0;font-family:Consolas,monospace;font-size:9.5px}
#__ID__ table {width:100%;border-collapse:collapse;margin-top:5px;font-size:9.5px}
#__ID__ th, #__ID__ td {border:1px solid #ccc;padding:3px 4px;text-align:left;vertical-align:top}
#__ID__ textarea {width:100%;min-height:74px;margin-top:4px;font-family:Consolas,monospace;font-size:9.5px}
#__ID__ input[type=text], #__ID__ input[type=number], #__ID__ input[type=color], #__ID__ select {padding:2px 4px;min-height:23px;max-width:100%;font-size:9.5px}
#__ID__ input[type=range] {width:100%;height:16px;margin:1px 0}
#__ID__ .seek-linha {display:grid;grid-template-columns:46px 1fr 46px;gap:4px;align-items:center;font:9px Consolas,monospace;margin:2px 0}
#__ID__ .seek-linha span:last-child{text-align:right}
#__ID__ .inline-campos {display:flex;gap:4px;align-items:center;flex-wrap:wrap}
#__ID__ .inline-campos input[type=number]{width:62px}
#__ID__ .inline-campos input[type=color]{width:34px;padding:1px}
#__ID__ .resultado-render {display:none;margin-top:5px;padding:5px;border:1px solid #3a8f5c;border-radius:5px}
#__ID__ .resultado-render video {width:100%;max-height:48vh;background:#000;margin-top:4px}
#__ID__ .grade-campos {display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:5px;margin:5px 0}
#__ID__ .campo {display:flex;flex-direction:column;gap:2px}
#__ID__ .campo label {font-size:9.5px;font-weight:bold;opacity:.72}
#__ID__ .painel-video2 {display:none;grid-column:1/-1;padding:5px;border:1px dashed #aaa;border-radius:5px;background:rgba(127,127,127,.035)}
#__ID__ .video2-wrap {display:flex;align-items:flex-start;gap:6px;flex-wrap:wrap}
#__ID__ .mini-video2 {width:min(230px,100%);max-height:118px;background:#000}
#__ID__ .video2-info {flex:1;min-width:240px;font-size:9.5px}
#__ID__ .video2-tempo {font:10px Consolas,monospace;margin-bottom:3px}
#__ID__ .linha-tempos2 {display:grid;grid-template-columns:1fr 18px 1fr;gap:3px;align-items:center;margin:3px 0}
#__ID__ .linha-tempos2 span {text-align:center;font-size:9px;opacity:.7}
#__ID__ .painel-transcricao {margin:5px 0;border:1px solid #bbb;border-radius:6px;overflow:hidden}
#__ID__ .cabecalho-transcricao {display:flex;flex-wrap:wrap;justify-content:space-between;gap:5px;padding:4px 6px;font-size:10.5px;font-weight:bold;background:rgba(127,127,127,.10)}
#__ID__ .transcricao {max-height:190px;overflow-y:auto;padding:3px;scroll-behavior:smooth;font-size:9.5px}
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
#__ID__ #__ID___tabela_operacoes tr[data-editar-op] {cursor:pointer}
#__ID__ #__ID___tabela_operacoes tr[data-editar-op]:hover td {background:rgba(80,130,220,.08)}
#__ID__ #__ID___tabela_operacoes tr.editando td {background:rgba(255,190,0,.14)}
#__ID__ .badge {display:inline-block;padding:1px 4px;border:1px solid #aaa;border-radius:999px;font-size:8.5px}
#__ID__ details {margin:4px 0}
#__ID__ summary {cursor:pointer;font-weight:bold;font-size:9.5px}
</style>

__PREVIEW_NOTICE__
__V6_NOTICE__
<div>Arquivo original: <strong>__ORIGINAL__</strong><br>Codec original: <code>__CODEC__</code></div>
<div id="__ID___erro" class="erro">O navegador não conseguiu reproduzir este arquivo. Execute novamente com <code>preparar_preview=True</code>.</div>
<div id="__ID___status" class="status"></div>

<div id="__ID___palco" class="palco-video">
  <video id="__ID___video" class="video-principal" controls preload="metadata"><source src="__SRC__" type="video/mp4">Seu navegador não conseguiu abrir o vídeo.</video>
  <div id="__ID___camada_operacoes" class="camada-operacoes"></div>
  <div id="__ID___camada_sequencial" class="camada-sequencial"></div>
  <div id="__ID___legenda" class="legenda-video" __SUBTITLE_HIDDEN__><span></span></div>
  <div id="__ID___regiao" class="caixa-regiao" hidden>
    <div id="__ID___preview_regiao" class="preview-edicao"></div>
    <div id="__ID___rotulo_regiao" class="rotulo-regiao">região</div><div class="alca"></div>
  </div>
</div>
<div class="seek-linha"><span id="__ID___seek_inicio">00:00</span><input id="__ID___seek_principal" type="range" min="0" max="1000" step="1" value="0"><span id="__ID___seek_fim">--:--</span></div>

<div class="tempo"><span id="__ID___tempo">00:00:00.000</span> · velocidade: <span id="__ID___velocidade">1×</span> · FPS: __FPS_TEXT__</div>
<div class="ajuda"><label><input id="__ID___saltar_cortes" type="checkbox" checked> saltar automaticamente os cortes durante a reprodução</label></div>

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
    <label style="display:inline-flex;align-items:center;gap:3px">Tam. <input id="__ID___tamanho_legenda" type="number" min="4" max="120" step="1" value="22" style="width:48px"> px</label>
    <label style="display:inline-flex;align-items:center;gap:3px" title="Negativo sobe; positivo desce">Ajuste Y <input id="__ID___ajuste_legenda" type="range" min="-20" max="20" step="1" value="0" style="width:90px"><span id="__ID___ajuste_legenda_valor">0%</span></label>
    <label style="display:inline-flex;align-items:center;gap:3px"><input id="__ID___incorporar_legenda" type="checkbox"> Incorporar no MP4</label>
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
<button data-acao="inicio"><span>Marcar início</span><kbd>I</kbd></button><button data-acao="fim"><span>Marcar fim</span><kbd>O</kbd></button>
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
  <div class="ajuda">Use as mesmas marcas de início/fim. O vídeo 2 pode ser sobreposto, colocado lado a lado ou usado como B-roll. A Prévia do projeto simula cortes, velocidade por trecho, inserções, substituições e políticas de áudio antes da renderização final. Mídias devem estar na pasta do vídeo ou em uma subpasta.</div>
  <div class="grade-campos">
    <div class="campo"><label>Tipo de operação</label><select id="__ID___tipo_operacao">
      <option value="cut" selected>Corte</option><option value="speed_segment">Acelerar/desacelerar trecho</option><option value="mute">Silenciar áudio</option><option value="blur_region">Desfoque de região</option><option value="black_bar">Tarja sobre região</option><option value="overlay_text">Texto sobreposto</option><option value="overlay_image">Imagem sobreposta</option><option value="shape_highlight">Forma/Destaque</option><option value="overlay_video">Vídeo sobreposto</option><option value="zoom_region">Zoom em região</option><option value="crop_region">Crop/Reenquadrar região</option><option value="background_audio">Áudio de fundo/inserido</option><option value="insert_video">Inserir vídeo em um ponto</option><option value="replace_video">Substituir trecho por vídeo</option>
    </select></div>
    <div class="campo" id="__ID___campo_texto"><label>Texto</label><input id="__ID___op_texto" type="text" placeholder="Texto que aparecerá no vídeo"></div>
    <div class="campo" id="__ID___campo_texto_estilo"><label>Estilo do texto</label><div class="inline-campos"><span>Tam.</span><input id="__ID___op_texto_tamanho" type="number" min="8" max="240" step="1" value="36"><span>Cor</span><input id="__ID___op_texto_cor" type="color" value="#ffffff"><select id="__ID___op_texto_fonte"><option value="sans-serif">Sans</option><option value="serif">Serif</option><option value="monospace">Mono</option></select></div></div>
    <div class="campo" id="__ID___campo_texto_fundo"><label>Fundo do texto</label><div class="inline-campos"><input id="__ID___op_texto_fundo_cor" type="color" value="#000000"><span>Opac.</span><input id="__ID___op_texto_fundo_opacidade" type="range" min="0" max="100" step="5" value="0"><span id="__ID___texto_fundo_pct">0%</span></div></div>
    <div class="campo" id="__ID___campo_forma"><label>Forma/Destaque</label><select id="__ID___op_forma_tipo"><option value="rectangle">Retângulo</option><option value="rounded_rectangle">Retângulo arredondado</option><option value="circle">Círculo</option><option value="ellipse">Elipse</option><option value="line">Linha</option><option value="underline">Sublinhado</option><option value="arrow">Seta</option><option value="highlighter">Marca-texto</option></select></div>
    <div class="campo" id="__ID___campo_forma_estilo"><label>Estilo da forma</label><div class="inline-campos"><span>Cor</span><input id="__ID___op_forma_cor" type="color" value="#ff3b30"><span>Esp.</span><input id="__ID___op_forma_espessura" type="number" min="1" max="80" step="1" value="4"><span>Opac.</span><input id="__ID___op_forma_opacidade" type="range" min="5" max="100" step="5" value="100"></div></div>
    <div class="campo" id="__ID___campo_forma_preench"><label>Preenchimento</label><div class="inline-campos"><label><input id="__ID___op_forma_preencher" type="checkbox"> preencher</label><span>Opac.</span><input id="__ID___op_forma_fill_opacidade" type="range" min="0" max="100" step="5" value="20"><select id="__ID___op_forma_direcao" title="Direção da seta"><option value="right">→ direita</option><option value="left">← esquerda</option><option value="down">↓ baixo</option><option value="up">↑ cima</option></select></div></div>
    <div class="campo" id="__ID___campo_media"><label>Arquivo de mídia / áudio</label><div class="inline-campos"><input id="__ID___op_media" type="text" placeholder="logo.png / camera.mp4 / trilha.mp3" style="flex:1;min-width:120px"><button type="button" data-acao="selecionar-midia">Selecionar…</button><input id="__ID___arquivo_midia" type="file" accept="image/*,video/*,audio/*" hidden></div></div>
    <div class="campo" id="__ID___campo_fit"><label>Ajuste da mídia</label><select id="__ID___op_fit"><option value="contain">Conter / preservar proporção</option><option value="cover">Preencher / cortar bordas</option><option value="fill">Esticar</option></select></div>
    <div class="campo" id="__ID___campo_modo_video"><label>Apresentação do vídeo 2</label><select id="__ID___op_modo_video"><option value="overlay">Sobrepor na caixa</option><option value="side_by_side">Lado a lado automático</option><option value="broll">B-roll / tela cheia</option></select></div>
    <div class="campo" id="__ID___campo_audio"><label>Áudio</label><select id="__ID___op_audio"><option value="base">Só principal</option><option value="overlay">Só vídeo 2</option><option value="both">Os dois</option><option value="none">Nenhum</option></select></div>
    <div class="campo" id="__ID___campo_playback"><label>Reprodução</label><select id="__ID___op_playback"><option value="both">Os dois rodam</option><option value="base_only">Só principal roda</option><option value="overlay_only">Só vídeo 2 roda</option></select></div>
    <div class="campo" id="__ID___campo_volume_base"><label>Volume principal (%)</label><input id="__ID___op_volume_base" type="number" min="0" max="100" step="5" value="100"></div>
    <div class="campo" id="__ID___campo_volume_overlay"><label>Volume mídia/áudio 2 (%)</label><input id="__ID___op_volume_overlay" type="number" min="0" max="100" step="5" value="100"></div>
    <div class="campo" id="__ID___campo_duracao_replace"><label>Substituição</label><select id="__ID___op_duracao_replace"><option value="natural">Duração natural do vídeo 2</option><option value="fit_interval">Ajustar ao intervalo marcado</option></select></div>
    <div class="campo" id="__ID___campo_velocidade_trecho"><label>Velocidade do trecho (×)</label><input id="__ID___op_velocidade_trecho" type="number" min="0.1" max="16" step="0.05" value="2"></div>
    <div class="campo" id="__ID___campo_blur"><label>Desfoque (px)</label><div class="inline-campos"><input id="__ID___op_blur" type="range" min="1" max="60" step="1" value="16"><input id="__ID___op_blur_num" type="number" min="1" max="60" step="1" value="16"></div></div>
    <div class="campo" id="__ID___campo_zoom"><label>Zoom</label><div class="inline-campos"><input id="__ID___op_zoom" type="range" min="1" max="5" step="0.1" value="1.5"><input id="__ID___op_zoom_num" type="number" min="1" max="5" step="0.1" value="1.5"><span>×</span></div></div>
    <div id="__ID___painel_video2" class="painel-video2">
      <div class="video2-wrap">
        <video id="__ID___video2" class="mini-video2" preload="metadata" playsinline></video>
        <div class="video2-info">
          <div class="video2-tempo">Mídia 2: <span id="__ID___tempo_video2">00:00:00.000</span></div>
          <div class="seek-linha"><span>00:00</span><input id="__ID___seek_video2" type="range" min="0" max="1000" step="1" value="0"><span id="__ID___seek_video2_fim">--:--</span></div>
          <div class="linha-tempos2">
            <input id="__ID___op_media_inicio" type="text" value="00:00:00.000" title="Início usado do vídeo 2">
            <span>→</span>
            <input id="__ID___op_media_fim" type="text" placeholder="fim do vídeo" title="Fim usado do vídeo 2">
          </div>
          <div class="botoes">
            <button data-acao="play-video2">▶/Ⅱ mídia 2</button>
            <button data-acao="inicio-video2">Início mídia</button>
            <button data-acao="fim-video2">Fim mídia</button>
            <button data-acao="todo-video2">Mídia inteira</button>
          </div>
          <div class="ajuda">Escolha exatamente o trecho da mídia secundária. A barra permite saltar diretamente para qualquer ponto; início/fim preservam o arquivo original.</div>
        </div>
      </div>
    </div>
  </div>
  <div class="botoes"><button data-acao="mostrar-regiao">Caixa</button><button data-acao="resetar-regiao">Resetar caixa</button><button id="__ID___botao_adicionar_operacao" data-acao="adicionar-operacao"><strong>Adicionar operação</strong></button><button id="__ID___botao_cancelar_edicao" data-acao="cancelar-edicao" hidden>Cancelar edição</button><button data-acao="desfazer-operacao">Desfazer última</button><button data-acao="limpar-operacoes">Limpar adicionais</button></div>

  <table><thead><tr><th>#</th><th>Tipo</th><th>Início</th><th>Fim</th><th>Detalhes</th><th></th></tr></thead><tbody id="__ID___tabela_operacoes"></tbody></table>
  <div class="botoes"><button data-acao="preview-projeto"><strong>▶ Prévia projeto</strong></button><button data-acao="preview-daqui">▶ Daqui</button><button data-acao="parar-preview">■ Parar prévia</button><span id="__ID___modo_preview" class="badge modo-preview">edição</span></div>
  <div id="__ID___preview_local" class="preview-local" hidden><div class="ajuda">Prévia do projeto — o mesmo player de edição é movido para cá enquanto a prévia estiver ativa.</div><div id="__ID___preview_local_palco"></div></div>
  <div class="botoes"><button data-acao="salvar-projeto">Salvar projeto</button><button data-acao="copiar-projeto">Copiar JSON</button><button data-acao="baixar-projeto">Baixar JSON</button></div>
  <details><summary>Ver JSON do projeto</summary><textarea id="__ID___saida_operacoes" readonly></textarea></details>
</div>

<div class="ajuda">Clique uma vez dentro da interface para ativar os atalhos. Clique em uma linha da tabela de operações para reabrir e editar o objeto; o × apenas exclui. A edição de palavras não altera os timestamps. Sempre mantenha uma cópia da última versão funcional dos arquivos antes de testar uma nova versão.</div>
</div>

<script>
(() => {
const raiz=document.getElementById("__ID__"), video=document.getElementById("__ID___video"), palco=document.getElementById("__ID___palco"), camadaOperacoes=document.getElementById("__ID___camada_operacoes"), camadaSequencial=document.getElementById("__ID___camada_sequencial"), erroEl=document.getElementById("__ID___erro"), statusEl=document.getElementById("__ID___status"), tempoEl=document.getElementById("__ID___tempo"), velocidadeEl=document.getElementById("__ID___velocidade"), inicioEl=document.getElementById("__ID___inicio"), fimEl=document.getElementById("__ID___fim"), tabelaEl=document.getElementById("__ID___tabela"), saidaEl=document.getElementById("__ID___saida"), painelTranscricaoEl=document.getElementById("__ID___painel_transcricao"), transcricaoEl=document.getElementById("__ID___transcricao"), estadoTranscricaoEl=document.getElementById("__ID___estado_transcricao"), legendaEl=document.getElementById("__ID___legenda"), posicaoLegendaEl=document.getElementById("__ID___posicao_legenda"), tamanhoLegendaEl=document.getElementById("__ID___tamanho_legenda"), ajusteLegendaEl=document.getElementById("__ID___ajuste_legenda"), ajusteLegendaValorEl=document.getElementById("__ID___ajuste_legenda_valor"), incorporarLegendaEl=document.getElementById("__ID___incorporar_legenda"), caixaRegiao=document.getElementById("__ID___regiao"), previewRegiao=document.getElementById("__ID___preview_regiao"), rotuloRegiao=document.getElementById("__ID___rotulo_regiao"), tipoOperacaoEl=document.getElementById("__ID___tipo_operacao"), tabelaOperacoesEl=document.getElementById("__ID___tabela_operacoes"), saidaOperacoesEl=document.getElementById("__ID___saida_operacoes"), campoTexto=document.getElementById("__ID___campo_texto"), campoTextoEstilo=document.getElementById("__ID___campo_texto_estilo"), campoTextoFundo=document.getElementById("__ID___campo_texto_fundo"), campoMedia=document.getElementById("__ID___campo_media"), campoFit=document.getElementById("__ID___campo_fit"), campoModoVideo=document.getElementById("__ID___campo_modo_video"), campoAudio=document.getElementById("__ID___campo_audio"), campoPlayback=document.getElementById("__ID___campo_playback"), campoVolumeBase=document.getElementById("__ID___campo_volume_base"), campoVolumeOverlay=document.getElementById("__ID___campo_volume_overlay"), campoDuracaoReplace=document.getElementById("__ID___campo_duracao_replace"), campoVelocidadeTrecho=document.getElementById("__ID___campo_velocidade_trecho"), campoBlur=document.getElementById("__ID___campo_blur"), campoZoom=document.getElementById("__ID___campo_zoom"), painelVideo2=document.getElementById("__ID___painel_video2"), video2=document.getElementById("__ID___video2"), tempoVideo2El=document.getElementById("__ID___tempo_video2"), opMediaInicio=document.getElementById("__ID___op_media_inicio"), opMediaFim=document.getElementById("__ID___op_media_fim"), opTexto=document.getElementById("__ID___op_texto"), opMedia=document.getElementById("__ID___op_media"), opFit=document.getElementById("__ID___op_fit"), opModoVideo=document.getElementById("__ID___op_modo_video"), opAudio=document.getElementById("__ID___op_audio"), opPlayback=document.getElementById("__ID___op_playback"), opVolumeBase=document.getElementById("__ID___op_volume_base"), opVolumeOverlay=document.getElementById("__ID___op_volume_overlay"), opDuracaoReplace=document.getElementById("__ID___op_duracao_replace"), opVelocidadeTrecho=document.getElementById("__ID___op_velocidade_trecho"), opBlur=document.getElementById("__ID___op_blur"), opBlurNum=document.getElementById("__ID___op_blur_num"), opZoom=document.getElementById("__ID___op_zoom"), opZoomNum=document.getElementById("__ID___op_zoom_num"), opTextoTamanho=document.getElementById("__ID___op_texto_tamanho"), opTextoCor=document.getElementById("__ID___op_texto_cor"), opTextoFonte=document.getElementById("__ID___op_texto_fonte"), opTextoFundoCor=document.getElementById("__ID___op_texto_fundo_cor"), opTextoFundoOpacidade=document.getElementById("__ID___op_texto_fundo_opacidade"), textoFundoPct=document.getElementById("__ID___texto_fundo_pct"), arquivoMidia=document.getElementById("__ID___arquivo_midia"), saltarCortes=document.getElementById("__ID___saltar_cortes"), seekPrincipal=document.getElementById("__ID___seek_principal"), seekFim=document.getElementById("__ID___seek_fim"), seekVideo2=document.getElementById("__ID___seek_video2"), seekVideo2Fim=document.getElementById("__ID___seek_video2_fim"), modoPreviewEl=document.getElementById("__ID___modo_preview"), botaoAdicionarOperacao=document.getElementById("__ID___botao_adicionar_operacao"), botaoCancelarEdicao=document.getElementById("__ID___botao_cancelar_edicao");
const campoForma=document.getElementById("__ID___campo_forma"), campoFormaEstilo=document.getElementById("__ID___campo_forma_estilo"), campoFormaPreench=document.getElementById("__ID___campo_forma_preench"), opFormaTipo=document.getElementById("__ID___op_forma_tipo"), opFormaCor=document.getElementById("__ID___op_forma_cor"), opFormaEspessura=document.getElementById("__ID___op_forma_espessura"), opFormaOpacidade=document.getElementById("__ID___op_forma_opacidade"), opFormaPreencher=document.getElementById("__ID___op_forma_preencher"), opFormaFillOpacidade=document.getElementById("__ID___op_forma_fill_opacidade"), opFormaDirecao=document.getElementById("__ID___op_forma_direcao"), previewLocal=document.getElementById("__ID___preview_local"), previewLocalPalco=document.getElementById("__ID___preview_local_palco");
const palcoPaiOriginal=palco.parentNode, palcoProximoOriginal=palco.nextSibling;
const fps=__FPS__, duracaoEsperada=__DURATION__, dadosTranscricao=__TRANSCRIPT_JSON__, dadosProjetoInicial=__PROJECT_JSON__, apiBase="__API_BASE__", podeSalvarTranscricao=__CAN_SAVE_TRANSCRIPT__, podeSalvarProjeto=__CAN_SAVE_PROJECT__;
let inicio=null,fim=null,cortes=[],operacoes=Array.isArray(dadosProjetoInicial?.operations)?JSON.parse(JSON.stringify(dadosProjetoInicial.operations)):[],blocosTranscricao=[],blocoAtivo=-1,palavraAtiva=null,transcricaoSuja=false,legendasVisiveis=__SUBTITLES_VISIBLE__;
const configLegendaInicial=(dadosProjetoInicial?.subtitles&&typeof dadosProjetoInicial.subtitles==="object")?dadosProjetoInicial.subtitles:{};
if(["baixo","meio","topo"].includes(configLegendaInicial.position))posicaoLegendaEl.value=configLegendaInicial.position;
if(Number.isFinite(Number(configLegendaInicial.font_size)))tamanhoLegendaEl.value=String(Math.max(4,Math.min(120,Number(configLegendaInicial.font_size))));
if(Number.isFinite(Number(configLegendaInicial.vertical_offset_pct)))ajusteLegendaEl.value=String(Math.max(-20,Math.min(20,Number(configLegendaInicial.vertical_offset_pct))));
ajusteLegendaValorEl.textContent=(Number(ajusteLegendaEl.value)||0)+"%";
incorporarLegendaEl.checked=!!configLegendaInicial.burn_in;
let operacaoEditandoId=null;
let regiaoAtual={x:.66,y:.06,width:.29,height:.22}, interacaoRegiao=null;
let regioesLegadasMigradas=dadosProjetoInicial?.region_reference==="video_frame";
let modoPreviaProjeto=false,ultimoTempoProjeto=0,eventosConsumidos=new Set(),midiaSequencial=null;
let estadoAudioUsuario={muted:video.muted,volume:video.volume};
let velocidadeBaseProjeto=video.playbackRate||1;
let audioFundoAtual=null;
function pararAudioFundo(){if(audioFundoAtual?.audio){try{audioFundoAtual.audio.pause()}catch(_){}}audioFundoAtual=null}
function sincronizarAudioFundo(s){
    if(!modoPreviaProjeto||midiaSequencial){pararAudioFundo();return null}
    const op=operacoes.find(o=>o.enabled!==false&&o.type==="background_audio"&&s>=Number(o.start)&&s<=Number(o.end));
    if(!op){pararAudioFundo();return null}
    const p=op.params||{},pol=p.audio_policy||"both";
    if(pol==="base"||pol==="none"){pararAudioFundo();return op}
    const url=caminhoMediaUrl(p.media);if(!url)return op;
    if(!audioFundoAtual||audioFundoAtual.id!==op.id){pararAudioFundo();const a=new Audio(url);a.preload="auto";audioFundoAtual={id:op.id,audio:a,op};}
    const a=audioFundoAtual.audio,mi=Math.max(0,Number(p.media_in)||0),mo=Number(p.media_out),alvo=mi+Math.max(0,s-Number(op.start));
    if(Number.isFinite(mo)&&mo>mi&&alvo>=mo-.03){a.pause();return op}
    if(a.readyState>=1&&Math.abs((a.currentTime||0)-alvo)>.20){try{a.currentTime=alvo}catch(_){}}
    a.volume=Math.max(0,Math.min(1,Number(p.overlay_volume??30)/100));a.muted=false;
    if(!video.paused)a.play().catch(()=>{});else a.pause();
    return op
}

function status(msg,erro=false){statusEl.style.display="block";statusEl.style.borderColor=erro?"#b00020":"#3a8f5c";statusEl.textContent=msg;clearTimeout(statusEl._timer);statusEl._timer=setTimeout(()=>statusEl.style.display="none",4500)}
function limitar(v){const d=Number.isFinite(video.duration)?video.duration:duracaoEsperada;return Math.max(0,Math.min(v,d))}
function formatar(s){s=Math.max(0,Number(s)||0);const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),r=s%60;return String(h).padStart(2,"0")+":"+String(m).padStart(2,"0")+":"+r.toFixed(3).padStart(6,"0")}
function formatarCurto(s){s=Math.max(0,Number(s)||0);const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),si=Math.floor(s%60);return h>0?String(h).padStart(2,"0")+":"+String(m).padStart(2,"0")+":"+String(si).padStart(2,"0"):String(m).padStart(2,"0")+":"+String(si).padStart(2,"0")}
function parseTempo(txt){if(typeof txt==="number")return Math.max(0,txt);const v=String(txt??"").trim();if(!v)return null;const p=v.split(":").map(Number);if(p.some(x=>!Number.isFinite(x)))return null;if(p.length===1)return Math.max(0,p[0]);if(p.length===2)return Math.max(0,p[0]*60+p[1]);if(p.length===3)return Math.max(0,p[0]*3600+p[1]*60+p[2]);return null}
function hexRgba(hex,a){const h=String(hex||"#000000").replace("#","");const n=parseInt(h,16)||0;return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${Math.max(0,Math.min(1,a))})`}
function faixaMidia(p,duracao=null){let a=parseTempo(p?.media_in);if(a===null)a=0;let b=parseTempo(p?.media_out);if(b===null&&Number.isFinite(duracao))b=duracao;if(b!==null&&b<a)b=a;return {inicio:a,fim:b,duracao:b===null?null:Math.max(0,b-a)}}
function textoPalavras(words){return (words||[]).map(w=>w.word||"").join("").trim()}
function buscar(s){video.pause();video.currentTime=limitar(s);atualizarTempo(video.currentTime)}
function moverSegundos(d){buscar(video.currentTime+d)}
function alternarReproducao(){if(midiaSequencial){if(midiaSequencial.video.paused)midiaSequencial.video.play().catch(()=>{});else midiaSequencial.video.pause();return}if(video.paused){video.play().then(()=>{camadaOperacoes.querySelectorAll("video[data-op-start]").forEach(v=>{const op=operacoes.find(o=>o.id===v.dataset.opId),p=op?.params||{};v.playbackRate=video.playbackRate;if(p.playback_policy!=="base_only")v.play().catch(()=>{})});if(modoPreviaProjeto)aplicarPoliticaAudio(video.currentTime)}).catch(()=>erroEl.style.display="block")}else{video.pause();camadaOperacoes.querySelectorAll("video").forEach(v=>v.pause());if(audioFundoAtual?.audio)audioFundoAtual.audio.pause()}}
function definirVelocidade(t){t=Math.max(.1,Math.min(16,Number(t)||1));if(modoPreviaProjeto)velocidadeBaseProjeto=t;video.playbackRate=t;video.defaultPlaybackRate=t;camadaOperacoes.querySelectorAll("video").forEach(v=>v.playbackRate=t);velocidadeEl.textContent=String(t).replace(".",",")+"×";raiz.querySelectorAll("button[data-velocidade]").forEach(b=>b.classList.toggle("ativo",Number(b.dataset.velocidade)===Number(t)))}
function aplicarVelocidadeProjeto(s){if(!modoPreviaProjeto||midiaSequencial)return;const op=operacoes.find(o=>o.enabled!==false&&o.type==="speed_segment"&&s>=Number(o.start)&&s<=Number(o.end));const fator=op?Math.max(.1,Math.min(16,Number(op.params?.factor)||1)):1;const taxa=Math.max(.1,Math.min(16,velocidadeBaseProjeto*fator));if(Math.abs(video.playbackRate-taxa)>.001)video.playbackRate=taxa;velocidadeEl.textContent=String(taxa).replace(".",",")+"×"+(op?" (trecho ×"+String(fator).replace(".",",")+")":"")}
function downloadTexto(nome,conteudo,tipo="text/plain;charset=utf-8"){const blob=new Blob([conteudo],{type:tipo}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=nome;a.click();setTimeout(()=>URL.revokeObjectURL(url),250)}
async function copiarTexto(texto){await navigator.clipboard.writeText(texto);status("Copiado para a área de transferência.")}

function atualizarSegmentoCorrespondente(palavra){if(!dadosTranscricao?.segments)return;for(const seg of dadosTranscricao.segments){const item=(seg.words||[]).find(w=>Math.abs(Number(w.start)-Number(palavra.start))<1e-6&&Math.abs(Number(w.end)-Number(palavra.end))<1e-6);if(item){item.word=palavra.word;seg.text=textoPalavras(seg.words);break}}}
function corrigirPalavra(span){if(!dadosTranscricao)return;const bi=Number(span.dataset.bloco),pi=Number(span.dataset.palavra),palavra=dadosTranscricao.blocks[bi].words[pi],atual=String(palavra.word||""),prefixo=(atual.match(/^\s*/)||[""])[0],novo=prompt("Corrigir palavra (o tempo será preservado):",atual.trim());if(novo===null)return;const limpo=novo.trim();if(!limpo)return;palavra.word=prefixo+limpo;dadosTranscricao.blocks[bi].text=textoPalavras(dadosTranscricao.blocks[bi].words);atualizarSegmentoCorrespondente(palavra);dadosTranscricao.version=Math.max(Number(dadosTranscricao.version||1),2);span.textContent=palavra.word;span.classList.add("editada");transcricaoSuja=true;estadoTranscricaoEl.textContent="salvando…";atualizarLegenda(video.currentTime);if(podeSalvarTranscricao){clearTimeout(corrigirPalavra._timer);corrigirPalavra._timer=setTimeout(()=>salvarTranscricao(),250)}else{estadoTranscricaoEl.textContent="alterações não salvas"}}
async function salvarTranscricao(){if(!dadosTranscricao){return}if(!podeSalvarTranscricao){status("Sem caminho de transcrição para salvar. Use 'Baixar transcrição JSON'.",true);return}try{const r=await fetch(apiBase+"/__editor_api__/transcricao",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(dadosTranscricao)}),j=await r.json();if(!r.ok||!j.ok)throw new Error(j.erro||"Falha ao salvar");transcricaoSuja=false;estadoTranscricaoEl.textContent="salva em "+j.arquivo;status("Transcrição salva: "+j.arquivo)}catch(e){status("Erro ao salvar transcrição: "+e.message,true)}}
function srtTempo(s){let ms=Math.max(0,Math.round(Number(s)*1000)),h=Math.floor(ms/3600000);ms%=3600000;let m=Math.floor(ms/60000);ms%=60000;let si=Math.floor(ms/1000),mil=ms%1000;return String(h).padStart(2,"0")+":"+String(m).padStart(2,"0")+":"+String(si).padStart(2,"0")+","+String(mil).padStart(3,"0")}
function gerarSRT(vtt=false){if(!dadosTranscricao)return "";let linhas=vtt?["WEBVTT",""]:[];let n=1;for(const b of dadosTranscricao.blocks||[]){const t=textoPalavras(b.words)||String(b.text||"").trim();if(!t)continue;if(!vtt)linhas.push(String(n++));let a=srtTempo(b.start),z=srtTempo(b.end);if(vtt){a=a.replace(",",".");z=z.replace(",",".")}linhas.push(a+" --> "+z,t,"")}return linhas.join("\n")}

function renderizarTranscricao(){if(!dadosTranscricao||!Array.isArray(dadosTranscricao.blocks))return;painelTranscricaoEl.hidden=false;transcricaoEl.innerHTML="";blocosTranscricao=[];dadosTranscricao.blocks.forEach((b,bi)=>{const linha=document.createElement("div");linha.className="bloco-transcricao";const tempo=document.createElement("button");tempo.type="button";tempo.className="tempo-transcricao";tempo.dataset.irTempo=String(b.start);tempo.textContent=formatarCurto(b.start);tempo.title=formatar(b.start);const texto=document.createElement("div");texto.className="texto-transcricao";(b.words||[]).forEach((p,pi)=>{const span=document.createElement("span");span.className="palavra-transcricao";span.dataset.start=String(p.start);span.dataset.end=String(p.end);span.dataset.bloco=String(bi);span.dataset.palavra=String(pi);span.title=formatar(p.start)+" · duplo clique para corrigir";span.textContent=p.word;texto.appendChild(span)});linha.append(tempo,texto);transcricaoEl.appendChild(linha);blocosTranscricao.push({dados:b,elemento:linha,palavras:Array.from(texto.querySelectorAll(".palavra-transcricao"))})})}
function atualizarTranscricao(s){if(!blocosTranscricao.length)return;let nb=blocoAtivo;if(nb<0||s<blocosTranscricao[nb].dados.start||s>blocosTranscricao[nb].dados.end+.25)nb=blocosTranscricao.findIndex(({dados})=>s>=dados.start-.05&&s<=dados.end+.25);if(nb!==blocoAtivo){if(blocoAtivo>=0)blocosTranscricao[blocoAtivo].elemento.classList.remove("ativo");blocoAtivo=nb;if(blocoAtivo>=0){const el=blocosTranscricao[blocoAtivo].elemento;el.classList.add("ativo");}}let np=null;if(blocoAtivo>=0)np=blocosTranscricao[blocoAtivo].palavras.find(span=>s>=Number(span.dataset.start)-.03&&s<=Number(span.dataset.end)+.05)||null;if(np!==palavraAtiva){if(palavraAtiva)palavraAtiva.classList.remove("ativa");palavraAtiva=np;if(palavraAtiva)palavraAtiva.classList.add("ativa")}}
function aplicarEstiloLegenda(){
    const tam=Math.max(4,Math.min(120,Number(tamanhoLegendaEl.value)||22));
    tamanhoLegendaEl.value=String(tam);
    legendaEl.style.fontSize=tam+"px";
    const m=areaRealVideo();
    legendaEl.classList.remove("topo","meio");
    legendaEl.style.right="auto";
    if(m.width>0&&m.height>0){
        // A legenda deve ocupar a largura do FRAME real do vídeo, e não a
        // largura inteira do palco (que inclui barras pretas em vídeos 9:16).
        // Usamos 82% da largura útil, igual à regra do renderizador FFmpeg.
        const margemX=m.width*.09;
        const margemY=m.height*.045;
        const ajustePct=Math.max(-20,Math.min(20,Number(ajusteLegendaEl.value)||0));
        ajusteLegendaEl.value=String(ajustePct);
        ajusteLegendaValorEl.textContent=ajustePct+"%";
        const ajusteY=m.height*ajustePct/100;
        legendaEl.style.left=(m.left-m.palcoLeft+margemX)+"px";
        legendaEl.style.width=Math.max(20,m.width-2*margemX)+"px";
        if(posicaoLegendaEl.value==="topo"){
            legendaEl.style.top=(m.top-m.palcoTop+margemY+ajusteY)+"px";
            legendaEl.style.bottom="auto";
            legendaEl.style.transform="none";
            legendaEl.classList.add("topo");
        }else if(posicaoLegendaEl.value==="meio"){
            legendaEl.style.top=(m.top-m.palcoTop+m.height/2+ajusteY)+"px";
            legendaEl.style.bottom="auto";
            legendaEl.style.transform="translateY(-50%)";
            legendaEl.classList.add("meio");
        }else{
            legendaEl.style.top="auto";
            legendaEl.style.bottom=(m.palcoHeight-(m.top-m.palcoTop+m.height)+margemY-ajusteY)+"px";
            legendaEl.style.transform="none";
        }
    }else{
        // Fallback antes de os metadados do vídeo estarem disponíveis.
        legendaEl.style.left="9%";
        legendaEl.style.width="82%";
        legendaEl.style.top="auto";
        const ajustePct=Math.max(-20,Math.min(20,Number(ajusteLegendaEl.value)||0));
        ajusteLegendaValorEl.textContent=ajustePct+"%";
        if(posicaoLegendaEl.value==="baixo")legendaEl.style.bottom=(4-ajustePct)+"%";
        else legendaEl.style.bottom="auto";
        if(posicaoLegendaEl.value==="topo")legendaEl.style.top=(4+ajustePct)+"%";
        else if(posicaoLegendaEl.value==="meio")legendaEl.style.top=(50+ajustePct)+"%";
        legendaEl.style.transform=posicaoLegendaEl.value==="meio"?"translateY(-50%)":"none";
        if(posicaoLegendaEl.value!=="baixo")legendaEl.classList.add(posicaoLegendaEl.value);
    }
}
function atualizarLegenda(s){aplicarEstiloLegenda();if(!dadosTranscricao||!legendasVisiveis){legendaEl.hidden=true;return}const b=(dadosTranscricao.blocks||[]).find(x=>s>=Number(x.start)-.03&&s<=Number(x.end)+.1);if(!b){legendaEl.hidden=true;return}legendaEl.querySelector("span").textContent=textoPalavras(b.words)||b.text||"";legendaEl.hidden=false}
function atualizarTempo(s=video.currentTime){tempoEl.textContent=formatar(s);atualizarTranscricao(s);atualizarLegenda(s);renderizarOverlaysAtivos(s)}

function sincronizarCortes(){cortes=operacoes.filter(o=>o.type==="cut"&&o.enabled!==false&&Number(o.end)>Number(o.start)).map(o=>[Number(o.start),Number(o.end)]).sort((a,b)=>a[0]-b[0])}
function novaId(tipo){return tipo+"_"+Date.now().toString(36)+"_"+Math.random().toString(36).slice(2,6)}
function criarOp(tipo,a,b,params={}){return {id:novaId(tipo),type:tipo,start:Number(a),end:Number(b),enabled:true,track:tipo==="cut"?"main":"editor_v6",params:params,note:""}}
function listaPython(){if(!cortes.length)return "[]";return "[\n"+cortes.map(([a,b])=>`    ("${formatar(a)}", "${formatar(b)}"),`).join("\n")+"\n]"}
function configLegendaAtual(){const tam=Math.max(4,Math.min(120,Number(tamanhoLegendaEl.value)||22)),mh=areaRealVideo().height,ajuste=Math.max(-20,Math.min(20,Number(ajusteLegendaEl.value)||0));return {burn_in:!!incorporarLegendaEl.checked,position:posicaoLegendaEl.value||"baixo",vertical_offset_pct:ajuste,font_size:tam,font_size_ratio:mh>0?tam/mh:(configLegendaInicial.font_size_ratio??null)}}
function projetoAtual(){return {version:5,source:__SOURCE_JSON__,time_reference:"source",region_reference:"video_frame",text_size_reference:"frame_height",subtitles:configLegendaAtual(),operations:operacoes}}
function resumoOp(o){const p=o.params||{};if(o.type==="speed_segment")return "×"+(p.factor||1);if(o.type==="overlay_text")return p.text||"";if(o.type==="overlay_video"){const m=({overlay:"sobrepor",side_by_side:"lado a lado",broll:"B-roll"})[p.presentation_mode||"overlay"];const a=({base:"áudio principal",overlay:"áudio vídeo 2",both:"áudios juntos",none:"sem áudio"})[p.audio_policy||"base"];return `${p.media||""} · ${formatar(parseTempo(p.media_in)||0)}→${p.media_out?formatar(parseTempo(p.media_out)||0):"fim"} · ${m} · ${a}`}if(o.type==="insert_video")return `${p.media||""} · ${formatar(parseTempo(p.media_in)||0)}→${p.media_out?formatar(parseTempo(p.media_out)||0):"fim"} · inserir/retomar · ${p.audio_policy==="none"?"sem áudio":"áudio vídeo 2"}`;if(o.type==="replace_video")return `${p.media||""} · ${formatar(parseTempo(p.media_in)||0)}→${p.media_out?formatar(parseTempo(p.media_out)||0):"fim"} · substituir · ${p.duration_policy==="fit_interval"?"ajustar ao intervalo":"duração natural"}`;if(o.type==="background_audio"){const a=({base:"só original",overlay:"só inserido",both:"misturado",none:"sem áudio"})[p.audio_policy||"both"];return `${p.media||""} · ${formatar(parseTempo(p.media_in)||0)}→${p.media_out?formatar(parseTempo(p.media_out)||0):"fim"} · ${a}`}if(o.type==="overlay_image")return p.media||"";if(o.type==="shape_highlight")return ({rectangle:"retângulo",rounded_rectangle:"retângulo arredondado",circle:"círculo",ellipse:"elipse",line:"linha",underline:"sublinhado",arrow:"seta",highlighter:"marca-texto"})[p.shape||"rectangle"]||"forma";if(o.type==="blur_region")return "intensidade "+(p.intensity||"");if(o.type==="mute")return "áudio principal";if(p.region)return `x=${p.region.x.toFixed(3)}, y=${p.region.y.toFixed(3)}, w=${p.region.width.toFixed(3)}, h=${p.region.height.toFixed(3)}`;return ""}
function nomeOp(t){return ({cut:"Corte",speed_segment:"Velocidade",mute:"Silenciar",blur_region:"Desfoque",black_bar:"Tarja",overlay_text:"Texto",overlay_image:"Imagem",shape_highlight:"Forma/Destaque",overlay_video:"Vídeo sobreposto",zoom_region:"Zoom",crop_region:"Crop",insert_video:"Inserir vídeo",replace_video:"Substituir por vídeo",background_audio:"Áudio de fundo"})[t]||t}
function atualizarModoEdicao(){
    const ativa=!!operacaoEditandoId;
    botaoAdicionarOperacao.innerHTML=ativa?"<strong>Salvar alterações</strong>":"<strong>Adicionar operação</strong>";
    botaoCancelarEdicao.hidden=!ativa;
}
function renderizar(){sincronizarCortes();inicioEl.textContent=inicio===null?"—":formatar(inicio);fimEl.textContent=fim===null?"—":formatar(fim);tabelaEl.innerHTML="";cortes.forEach(([a,b],i)=>{const tr=document.createElement("tr");tr.innerHTML=`<td>${i+1}</td><td>${formatar(a)}</td><td>${formatar(b)}</td><td>${formatar(b-a)}</td>`;tabelaEl.appendChild(tr)});saidaEl.value=listaPython();tabelaOperacoesEl.innerHTML="";operacoes.forEach((o,i)=>{const tr=document.createElement("tr");tr.dataset.editarOp=o.id;tr.title="Clique na linha para editar esta operação";if(o.id===operacaoEditandoId)tr.classList.add("editando");tr.innerHTML=`<td>${i+1}</td><td><span class="tipo-op">${nomeOp(o.type)}</span></td><td>${formatar(o.start)}</td><td>${formatar(o.end)}</td><td>${resumoOp(o)}</td><td><button data-remover-op="${i}" title="Excluir">×</button></td>`;tabelaOperacoesEl.appendChild(tr)});atualizarModoEdicao();saidaOperacoesEl.value=JSON.stringify(projetoAtual(),null,2);renderizarOverlaysAtivos(video.currentTime,true)}
function marcarInicio(){inicio=video.currentTime;renderizar()}
function marcarFim(){fim=video.currentTime;renderizar()}
function adicionarCorte(){if(inicio===null||fim===null){alert("Marque o início e o fim antes de adicionar.");return}if(fim<=inicio){alert("O fim precisa ser posterior ao início.");return}operacoes.push(criarOp("cut",inicio,fim,{}));operacoes.sort((a,b)=>a.start-b.start);inicio=null;fim=null;renderizar()}
function desfazerCorte(){for(let i=operacoes.length-1;i>=0;i--){if(operacoes[i].type==="cut"){operacoes.splice(i,1);break}}renderizar()}
function limparCortes(){if(confirm("Apagar todos os cortes? As operações adicionais serão mantidas.")){if(operacoes.some(o=>o.type==="cut"&&o.id===operacaoEditandoId))operacaoEditandoId=null;operacoes=operacoes.filter(o=>o.type!=="cut");inicio=null;fim=null;renderizar()}}

const tiposComRegiao=new Set(["blur_region","black_bar","overlay_text","overlay_image","shape_highlight","overlay_video","zoom_region","crop_region"]);
const tiposMediaRegiao=new Set(["overlay_image","overlay_video"]);
let chaveOverlaysAtivos="";
function clamp01(v){return Math.max(0,Math.min(1,v))}
function janelaEnquadramento(r,fator=null){
    r=r||{x:0,y:0,width:1,height:1};
    const rx=clamp01(Number(r.x)||0),ry=clamp01(Number(r.y)||0),rw=Math.max(.001,Math.min(1-rx,Number(r.width)||1)),rh=Math.max(.001,Math.min(1-ry,Number(r.height)||1));
    const minimo=Math.max(rw,rh),desejado=fator===null?0:1/Math.max(1,Number(fator)||1),tam=Math.min(1,Math.max(.001,minimo,desejado));
    const cx=rx+rw/2,cy=ry+rh/2;
    const xmin=Math.max(0,rx+rw-tam),xmax=Math.min(rx,1-tam),ymin=Math.max(0,ry+rh-tam),ymax=Math.min(ry,1-tam);
    const x=xmin<=xmax?Math.min(Math.max(cx-tam/2,xmin),xmax):Math.max(0,Math.min(1-tam,cx-tam/2));
    const y=ymin<=ymax?Math.min(Math.max(cy-tam/2,ymin),ymax):Math.max(0,Math.min(1-tam,cy-tam/2));
    return {x,y,width:tam,height:tam};
}
function tamanhoTextoPx(p){const m=areaRealVideo();if(p?.font_size_ratio&&m.height)return Math.max(8,Number(p.font_size_ratio)*m.height);return Math.max(8,Number(p?.font_size)||36)}
function areaRealVideo(){
    const p=palco.getBoundingClientRect(),v=video.getBoundingClientRect();
    let left=v.left,top=v.top,width=v.width,height=v.height;
    const iw=Number(video.videoWidth)||0,ih=Number(video.videoHeight)||0;
    if(iw>0&&ih>0&&v.width>0&&v.height>0){
        const escala=Math.min(v.width/iw,v.height/ih);
        const rw=iw*escala,rh=ih*escala;
        left=v.left+(v.width-rw)/2;top=v.top+(v.height-rh)/2;width=rw;height=rh;
    }
    return {left,top,width,height,palcoLeft:p.left,palcoTop:p.top,palcoWidth:p.width,palcoHeight:p.height};
}
function estiloRegiao(el,r){
    const m=areaRealVideo();if(!m.width||!m.height)return;
    el.style.left=(m.left-m.palcoLeft+Number(r.x)*m.width)+"px";
    el.style.top=(m.top-m.palcoTop+Number(r.y)*m.height)+"px";
    el.style.width=(Number(r.width)*m.width)+"px";
    el.style.height=(Number(r.height)*m.height)+"px";
}
function aplicarRegiao(){estiloRegiao(caixaRegiao,regiaoAtual)}
function lerRegiao(){
    const m=areaRealVideo(),c=caixaRegiao.getBoundingClientRect();if(!m.width||!m.height)return {...regiaoAtual,unit:"normalized"};
    regiaoAtual={x:clamp01((c.left-m.left)/m.width),y:clamp01((c.top-m.top)/m.height),width:clamp01(c.width/m.width),height:clamp01(c.height/m.height)};
    if(regiaoAtual.x+regiaoAtual.width>1)regiaoAtual.width=1-regiaoAtual.x;
    if(regiaoAtual.y+regiaoAtual.height>1)regiaoAtual.height=1-regiaoAtual.y;
    return {...regiaoAtual,unit:"normalized"};
}
function migrarRegioesLegadas(){
    if(regioesLegadasMigradas||!video.videoWidth||!video.videoHeight)return false;
    const p=palco.getBoundingClientRect(),m=areaRealVideo();if(!p.width||!p.height||!m.width||!m.height)return false;
    let alterou=false;
    operacoes.forEach(o=>{const r=o?.params?.region;if(!r)return;
        const l=p.left+Number(r.x)*p.width,t=p.top+Number(r.y)*p.height,w=Number(r.width)*p.width,h=Number(r.height)*p.height;
        let nx=clamp01((l-m.left)/m.width),ny=clamp01((t-m.top)/m.height),nw=clamp01(w/m.width),nh=clamp01(h/m.height);
        if(nx+nw>1)nw=Math.max(0,1-nx);if(ny+nh>1)nh=Math.max(0,1-ny);
        o.params.region={x:nx,y:ny,width:nw,height:nh,unit:"normalized"};alterou=true;
    });
    regioesLegadasMigradas=true;
    if(alterou)status("Regiões de projeto antigo ajustadas para o frame real. Salve o projeto para persistir a migração.");
    return alterou;
}
function resetarRegiao(){regiaoAtual={x:.66,y:.06,width:.29,height:.22};aplicarRegiao();atualizarPreviewEdicao()}
function caminhoMediaUrl(caminho){const bruto=String(caminho||"").trim().replace(/\\/g,"/");if(!bruto)return "";if(/^https?:\/\//i.test(bruto))return bruto;const partes=bruto.split("/").filter(p=>p&&p!==".");if(partes.some(p=>p===".."))return "";return apiBase+"/"+partes.map(encodeURIComponent).join("/")}
function limparPreviewEdicao(){previewRegiao.innerHTML="";previewRegiao.style.backdropFilter="";previewRegiao.style.webkitBackdropFilter="";previewRegiao.style.background=""}
function erroPreview(msg){limparPreviewEdicao();const d=document.createElement("div");d.className="erro-preview";d.textContent=msg;previewRegiao.appendChild(d)}
function criarSvgForma(p={}){
    const ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg");svg.classList.add("forma-svg");svg.setAttribute("viewBox","0 0 100 100");svg.setAttribute("preserveAspectRatio","none");
    const tipo=p.shape||"rectangle",cor=p.color||"#ff3b30",op=Math.max(.05,Math.min(1,Number(p.opacity??1))),fillOp=Math.max(0,Math.min(1,Number(p.fill_opacity??.2))),preencher=!!p.fill,esp=Math.max(.6,Math.min(18,Number(p.thickness_preview??p.thickness??4)));
    if(tipo==="circle")svg.setAttribute("preserveAspectRatio","xMidYMid meet");
    const stroke=cor,fill=preencher?cor:"none";let el;
    if(tipo==="rectangle"||tipo==="rounded_rectangle"){el=document.createElementNS(ns,"rect");el.setAttribute("x","2");el.setAttribute("y","2");el.setAttribute("width","96");el.setAttribute("height","96");if(tipo==="rounded_rectangle"){el.setAttribute("rx","8");el.setAttribute("ry","8")}}
    else if(tipo==="ellipse"||tipo==="circle"){el=document.createElementNS(ns,"ellipse");el.setAttribute("cx","50");el.setAttribute("cy","50");el.setAttribute("rx",tipo==="circle"?"47":"47");el.setAttribute("ry",tipo==="circle"?"47":"42")}
    else if(tipo==="line"||tipo==="underline"){el=document.createElementNS(ns,"line");el.setAttribute("x1","3");el.setAttribute("x2","97");el.setAttribute("y1",tipo==="underline"?"88":"50");el.setAttribute("y2",tipo==="underline"?"88":"50")}
    else if(tipo==="arrow"){
        const dir=p.direction||"right",g=document.createElementNS(ns,"g"),line=document.createElementNS(ns,"line"),poly=document.createElementNS(ns,"polygon");
        let x1=8,y1=50,x2=86,y2=50,pts="86,50 70,36 70,64";
        if(dir==="left"){x1=92;x2=14;pts="14,50 30,36 30,64"}else if(dir==="down"){x1=50;y1=8;x2=50;y2=86;pts="50,86 36,70 64,70"}else if(dir==="up"){x1=50;y1=92;x2=50;y2=14;pts="50,14 36,30 64,30"}
        line.setAttribute("x1",x1);line.setAttribute("y1",y1);line.setAttribute("x2",x2);line.setAttribute("y2",y2);line.setAttribute("stroke",stroke);line.setAttribute("stroke-width",esp);line.setAttribute("stroke-opacity",op);line.setAttribute("stroke-linecap","round");poly.setAttribute("points",pts);poly.setAttribute("fill",stroke);poly.setAttribute("fill-opacity",op);g.append(line,poly);svg.appendChild(g);return svg;
    } else if(tipo==="highlighter"){el=document.createElementNS(ns,"rect");el.setAttribute("x","1");el.setAttribute("y","8");el.setAttribute("width","98");el.setAttribute("height","84");el.setAttribute("fill",cor);el.setAttribute("fill-opacity",fillOp||.28);svg.appendChild(el);return svg}
    else return svg;
    el.setAttribute("stroke",stroke);el.setAttribute("stroke-width",esp);el.setAttribute("stroke-opacity",op);el.setAttribute("fill",fill);if(preencher)el.setAttribute("fill-opacity",fillOp);if(el.tagName!=="rect"&&el.tagName!=="ellipse")el.setAttribute("stroke-linecap","round");svg.appendChild(el);return svg;
}

function atualizarPreviewEdicao(){
    limparPreviewEdicao();
    const t=tipoOperacaoEl.value;
    if(!tiposComRegiao.has(t))return;
    if(t==="overlay_text"){
        const d=document.createElement("div");d.className="texto-preview";d.textContent=opTexto.value.trim()||"Texto";d.style.fontSize=(Number(opTextoTamanho.value)||36)+"px";d.style.color=opTextoCor.value||"#ffffff";d.style.fontFamily=opTextoFonte.value||"sans-serif";const bgop=Math.max(0,Math.min(1,Number(opTextoFundoOpacidade.value||0)/100));d.style.background=hexRgba(opTextoFundoCor.value||"#000000",bgop);previewRegiao.appendChild(d);return;
    }
    if(t==="shape_highlight"){
        const p={shape:opFormaTipo.value,color:opFormaCor.value,thickness:Number(opFormaEspessura.value)||4,thickness_preview:Number(opFormaEspessura.value)||4,opacity:Number(opFormaOpacidade.value||100)/100,fill:opFormaPreencher.checked,fill_opacity:Number(opFormaFillOpacidade.value||0)/100,direction:opFormaDirecao.value};previewRegiao.appendChild(criarSvgForma(p));return;
    }
    if(t==="overlay_image"){
        const url=caminhoMediaUrl(opMedia.value);if(!url){erroPreview("Informe uma imagem da pasta do vídeo.");return}
        const img=document.createElement("img");img.alt="Prévia";img.style.objectFit=opFit.value||"contain";img.src=url;img.onerror=()=>erroPreview("Imagem não encontrada ou não suportada.");previewRegiao.appendChild(img);return;
    }
    if(t==="overlay_video"){
        const url=caminhoMediaUrl(opMedia.value);if(!url){erroPreview("Informe um vídeo da pasta do vídeo.");return}
        const modo=opModoVideo.value||"overlay";if(modo==="side_by_side")regiaoAtual={x:.5,y:0,width:.5,height:1};else if(modo==="broll")regiaoAtual={x:0,y:0,width:1,height:1};aplicarRegiao();
        const v=document.createElement("video");v.muted=true;v.playsInline=true;v.preload="metadata";v.style.objectFit=opFit.value||"contain";v.src=url;v.addEventListener("loadedmetadata",()=>{try{v.currentTime=Math.min(.05,Math.max(0,(v.duration||1)/100))}catch(_){}});v.onerror=()=>erroPreview("Vídeo não encontrado ou não suportado no navegador.");previewRegiao.appendChild(v);return;
    }
    if(t==="blur_region"){
        const px=Math.max(1,Number(opBlur.value)||16);previewRegiao.style.backdropFilter=`blur(${px}px)`;previewRegiao.style.webkitBackdropFilter=`blur(${px}px)`;previewRegiao.style.background="rgba(255,255,255,.025)";return;
    }
    if(t==="black_bar"){previewRegiao.style.background="#000";return}
    if(t==="zoom_region"||t==="crop_region"){
        const clone=document.createElement("video");clone.src=video.currentSrc||video.querySelector("source")?.src||"";clone.muted=true;clone.playsInline=true;clone.preload="auto";clone.style.position="absolute";clone.style.pointerEvents="none";clone.style.objectFit="fill";
        const r=lerRegiao(),janela=janelaEnquadramento(r,t==="zoom_region"?Math.max(1,Math.min(5,Number(opZoom.value)||1.5)):null);
        clone.style.left=(-janela.x/janela.width*100)+"%";clone.style.top=(-janela.y/janela.height*100)+"%";clone.style.width=(100/janela.width)+"%";clone.style.height=(100/janela.height)+"%";
        clone.addEventListener("loadedmetadata",()=>{try{clone.currentTime=video.currentTime}catch(_){}});previewRegiao.appendChild(clone);return;
    }
}

function criarPreviewOperacao(o){
    const p=o.params||{},r=p.region;if(!r)return null;
    const el=document.createElement("div");el.className="preview-op";estiloRegiao(el,r);el.dataset.opId=o.id||"";
    if(o.type==="overlay_text"){
        const d=document.createElement("div");d.className="preview-texto";d.textContent=p.text||"";d.style.fontSize=tamanhoTextoPx(p)+"px";d.style.color=p.color||"#ffffff";d.style.fontFamily=p.font||"sans-serif";d.style.background=hexRgba(p.background_color||"#000000",Number(p.background_opacity||0));el.appendChild(d);
    }else if(o.type==="overlay_image"){
        const img=document.createElement("img");img.src=caminhoMediaUrl(p.media);img.style.objectFit=p.fit||"contain";img.alt="";el.appendChild(img);
    }else if(o.type==="overlay_video"){
        const modo=p.presentation_mode||"overlay";
        if(modo==="side_by_side"){
            el.classList.add("preview-side");estiloRegiao(el,{x:0,y:0,width:1,height:1});
            const base=document.createElement("video");base.src=video.currentSrc||video.querySelector("source")?.src||"";base.muted=true;base.playsInline=true;base.preload="auto";base.className="base-clone";base.dataset.baseClone="1";
            const v=document.createElement("video");v.src=caminhoMediaUrl(p.media);v.muted=true;v.playsInline=true;v.preload="auto";v.className="overlay-clone";v.style.objectFit=p.fit||"contain";v.dataset.opStart=String(o.start);v.dataset.opId=o.id||"";v.dataset.mediaIn=String(p.media_in??0);v.dataset.mediaOut=String(p.media_out??"");
            el.append(base,v);
        }else{
            if(modo==="broll"){el.classList.add("preview-broll");estiloRegiao(el,{x:0,y:0,width:1,height:1})}
            const v=document.createElement("video");v.src=caminhoMediaUrl(p.media);v.muted=true;v.playsInline=true;v.preload="auto";v.style.objectFit=p.fit||"contain";v.dataset.opStart=String(o.start);v.dataset.opId=o.id||"";v.dataset.mediaIn=String(p.media_in??0);v.dataset.mediaOut=String(p.media_out??"");el.appendChild(v);
        }
    }else if(o.type==="shape_highlight"){
        const pp={...p};const mh=areaRealVideo().height;if(pp.stroke_ratio&&mh)pp.thickness_preview=Math.max(1,Number(pp.stroke_ratio)*mh);el.appendChild(criarSvgForma(pp));
    }else if(o.type==="blur_region"){
        el.classList.add("preview-blur");const px=Math.max(1,Number(p.intensity)||16);el.style.backdropFilter=`blur(${px}px)`;el.style.webkitBackdropFilter=`blur(${px}px)`;
    }else if(o.type==="black_bar"){
        el.classList.add("preview-tarja");el.style.opacity=String(p.opacity??1);
    }else if(o.type==="zoom_region"||o.type==="crop_region"){
        estiloRegiao(el,{x:0,y:0,width:1,height:1});const clone=document.createElement("video");clone.src=video.currentSrc||video.querySelector("source")?.src||"";clone.muted=true;clone.playsInline=true;clone.preload="auto";clone.dataset.baseClone="1";const rr=p.region||{x:0,y:0,width:1,height:1};clone.style.position="absolute";clone.style.objectFit="fill";const janela=janelaEnquadramento(rr,o.type==="zoom_region"?Math.max(1,Math.min(5,Number(p.factor)||1.5)):null);clone.style.left=(-janela.x/janela.width*100)+"%";clone.style.top=(-janela.y/janela.height*100)+"%";clone.style.width=(100/janela.width)+"%";clone.style.height=(100/janela.height)+"%";el.appendChild(clone);
    }else{return null}
    return el;
}
function renderizarOverlaysAtivos(s,forcar=false){
    const ativos=operacoes.filter(o=>{if(o.enabled===false||!tiposComRegiao.has(o.type)||Number(o.start)>s||s>Number(o.end))return false;if(o.type!=="overlay_video")return true;const p=o.params||{},a=Number(p.media_in??0),b=Number(p.media_out);if(Number.isFinite(b)&&b>a)return s<=Math.min(Number(o.end),Number(o.start)+(b-a)+.02);return true});
    const chave=ativos.map(o=>o.id+":"+o.type+":"+JSON.stringify(o.params||{})).join("|");
    if(forcar||chave!==chaveOverlaysAtivos){
        chaveOverlaysAtivos=chave;camadaOperacoes.innerHTML="";ativos.forEach(o=>{const el=criarPreviewOperacao(o);if(el)camadaOperacoes.appendChild(el)});
    }
    camadaOperacoes.querySelectorAll("video[data-base-clone]").forEach(v=>{if(v.readyState<1)return;if(Math.abs((v.currentTime||0)-s)>.12){try{v.currentTime=s}catch(_){}}});
    camadaOperacoes.querySelectorAll("video[data-op-start]").forEach(v=>{
        if(v.readyState<1)return;const op=operacoes.find(o=>o.id===v.dataset.opId)||ativos.find(o=>Math.abs(Number(o.start)-Number(v.dataset.opStart||0))<1e-6&&o.type==="overlay_video");const p=op?.params||{};
        const rel=Math.max(0,s-Number(v.dataset.opStart||0)),mi=Math.max(0,Number(v.dataset.mediaIn||0)),moRaw=Number(v.dataset.mediaOut),mo=Number.isFinite(moRaw)&&moRaw>mi?moRaw:(v.duration&&Number.isFinite(v.duration)?v.duration:null);const bruto=mi+rel;const alvo=mo!==null?Math.min(bruto,Math.max(mi,mo-.04)):bruto;
        if(p.playback_policy==="base_only"){if(Math.abs(v.currentTime-mi)>.08){try{v.currentTime=mi}catch(_){}}v.pause()}else if(Math.abs((v.currentTime||0)-alvo)>.18){try{v.currentTime=alvo}catch(_){}}
        if(modoPreviaProjeto&&!video.paused&&p.playback_policy!=="base_only"){v.play().catch(()=>{})}else if(video.paused){v.pause()}
    });
    aplicarPoliticaAudio(s,ativos);
}
function atualizarCaixaRegiaoNaPrevia(s){
    // Na edição, a caixa permanece disponível normalmente.
    // Na prévia, ela acompanha SOMENTE o intervalo atualmente marcado:
    // aparece enquanto o intervalo está ativo e some assim que ele termina.
    const reg=tiposComRegiao.has(tipoOperacaoEl.value);
    if(!modoPreviaProjeto){
        caixaRegiao.hidden=!reg;
        return;
    }
    if(!reg){caixaRegiao.hidden=true;return}
    const a=Number(inicio),b=Number(fim);
    const intervaloValido=Number.isFinite(a)&&Number.isFinite(b)&&b>a;
    caixaRegiao.hidden=!(intervaloValido&&s>=a-0.002&&s<=b+0.002);
    if(!caixaRegiao.hidden)requestAnimationFrame(()=>aplicarRegiao());
}
function atualizarCamposOperacao(){
    const t=tipoOperacaoEl.value,reg=tiposComRegiao.has(t),ehVideo=["overlay_video","insert_video","replace_video","background_audio"].includes(t);
    campoTexto.style.display=t==="overlay_text"?"flex":"none";campoTextoEstilo.style.display=t==="overlay_text"?"flex":"none";campoTextoFundo.style.display=t==="overlay_text"?"flex":"none";
    campoForma.style.display=t==="shape_highlight"?"flex":"none";campoFormaEstilo.style.display=t==="shape_highlight"?"flex":"none";campoFormaPreench.style.display=t==="shape_highlight"?"flex":"none";opFormaDirecao.style.display=(t==="shape_highlight"&&opFormaTipo.value==="arrow")?"inline-block":"none";
    campoMedia.style.display=["overlay_image","overlay_video","insert_video","replace_video","background_audio"].includes(t)?"flex":"none";
    campoFit.style.display=tiposMediaRegiao.has(t)?"flex":"none";
    campoModoVideo.style.display=t==="overlay_video"?"flex":"none";
    campoAudio.style.display=["overlay_video","insert_video","replace_video","background_audio"].includes(t)?"flex":"none";
    campoPlayback.style.display=t==="overlay_video"?"flex":"none";
    campoVolumeBase.style.display=["overlay_video","background_audio"].includes(t)?"flex":"none";
    campoVolumeOverlay.style.display=["overlay_video","insert_video","replace_video","background_audio"].includes(t)?"flex":"none";
    campoDuracaoReplace.style.display=t==="replace_video"?"flex":"none";
    campoVelocidadeTrecho.style.display=t==="speed_segment"?"flex":"none";
    painelVideo2.style.display=ehVideo?"block":"none";
    campoBlur.style.display=t==="blur_region"?"flex":"none";campoZoom.style.display=t==="zoom_region"?"flex":"none";
    if(t==="insert_video"||t==="replace_video"){
        opAudio.innerHTML='<option value="overlay">Áudio da mídia 2</option><option value="none">Sem áudio</option>';
        if(!["overlay","none"].includes(opAudio.value))opAudio.value="overlay";
    }else if(t==="background_audio"){
        const anterior=opAudio.value;opAudio.innerHTML='<option value="both">Original + áudio inserido</option><option value="overlay">Só áudio inserido</option><option value="base">Só áudio original</option><option value="none">Nenhum</option>';if(["base","overlay","both","none"].includes(anterior))opAudio.value=anterior;else opAudio.value="both";
        if(Number(opVolumeOverlay.value)===100)opVolumeOverlay.value="30";
    }else if(t==="overlay_video"){
        const anterior=opAudio.value;opAudio.innerHTML='<option value="base">Só principal</option><option value="overlay">Só vídeo 2</option><option value="both">Os dois</option><option value="none">Nenhum</option>';if(["base","overlay","both","none"].includes(anterior))opAudio.value=anterior;else opAudio.value="base";
    }
    if(t==="overlay_video"){
        const m=opModoVideo.value||"overlay";
        if(m==="side_by_side")regiaoAtual={x:.5,y:0,width:.5,height:1};
        else if(m==="broll")regiaoAtual={x:0,y:0,width:1,height:1};
        else if(regiaoAtual.width>=.95&&regiaoAtual.height>=.95)regiaoAtual={x:.66,y:.06,width:.29,height:.22};
    }
    caixaRegiao.hidden=!reg;rotuloRegiao.textContent=nomeOp(t);
    if(reg)requestAnimationFrame(()=>{aplicarRegiao();atualizarPreviewEdicao()});else limparPreviewEdicao();
    if(ehVideo)atualizarVideo2Fonte();else{try{video2.pause()}catch(_){}}
}
function atualizarVideo2Fonte(){
    const ehVideo=["overlay_video","insert_video","replace_video","background_audio"].includes(tipoOperacaoEl.value);if(!ehVideo)return;
    const url=caminhoMediaUrl(opMedia.value);if(!url){video2.removeAttribute("src");video2.load();tempoVideo2El.textContent="00:00:00.000";return}
    if(video2.dataset.url!==url){video2.dataset.url=url;video2.src=url;video2.load()}
}
function trechoVideo2Atual(){const a=parseTempo(opMediaInicio.value);let b=parseTempo(opMediaFim.value);if(a===null)return null;if(b===null&&Number.isFinite(video2.duration))b=video2.duration;if(b!==null&&b<=a)return null;return {inicio:a,fim:b}}
function cancelarEdicaoOperacao(){operacaoEditandoId=null;atualizarModoEdicao();renderizar();status("Edição cancelada.")}
function carregarOperacaoParaEdicao(id){
    const o=operacoes.find(x=>x.id===id);if(!o)return;
    operacaoEditandoId=o.id;inicio=Number(o.start);fim=Number(o.end);tipoOperacaoEl.value=o.type;const p=o.params||{};
    if(o.type==="overlay_video")opModoVideo.value=p.presentation_mode||"overlay";
    atualizarCamposOperacao();
    if(o.type==="overlay_text"){opTexto.value=p.text||"";opTextoTamanho.value=String(Math.max(8,Number(p.font_size)||36));opTextoCor.value=p.color||"#ffffff";opTextoFonte.value=p.font||"sans-serif";opTextoFundoCor.value=p.background_color||"#000000";opTextoFundoOpacidade.value=String(Math.round(Math.max(0,Math.min(1,Number(p.background_opacity)||0))*100));textoFundoPct.textContent=opTextoFundoOpacidade.value+"%"}
    if(o.type==="shape_highlight"){opFormaTipo.value=p.shape||"rectangle";opFormaCor.value=p.color||"#ff3b30";opFormaEspessura.value=String(Math.max(1,Number(p.thickness)||4));opFormaOpacidade.value=String(Math.round(Math.max(.05,Math.min(1,Number(p.opacity)||1))*100));opFormaPreencher.checked=!!p.fill;opFormaFillOpacidade.value=String(Math.round(Math.max(0,Math.min(1,Number(p.fill_opacity)||0))*100));opFormaDirecao.value=p.direction||"right"}
    if(["overlay_image","overlay_video","insert_video","replace_video","background_audio"].includes(o.type))opMedia.value=p.media||"";
    if(["overlay_video","insert_video","replace_video","background_audio"].includes(o.type)){opMediaInicio.value=formatar(Number(p.media_in)||0);opMediaFim.value=p.media_out==null?"":formatar(Number(p.media_out))}
    if(tiposMediaRegiao.has(o.type))opFit.value=p.fit||"contain";
    if(o.type==="overlay_video"){opModoVideo.value=p.presentation_mode||"overlay";opAudio.value=p.audio_policy||"base";opPlayback.value=p.playback_policy||"both";opVolumeBase.value=String(Number(p.base_volume??100));opVolumeOverlay.value=String(Number(p.overlay_volume??100))}
    if(o.type==="insert_video"||o.type==="replace_video"){opAudio.value=p.audio_policy||"overlay";opVolumeOverlay.value=String(Number(p.overlay_volume??100))}
    if(o.type==="background_audio"){opAudio.value=p.audio_policy||"both";opVolumeBase.value=String(Number(p.base_volume??100));opVolumeOverlay.value=String(Number(p.overlay_volume??30))}
    if(o.type==="replace_video")opDuracaoReplace.value=p.duration_policy||"natural";
    if(o.type==="speed_segment")opVelocidadeTrecho.value=String(Number(p.factor)||1);
    if(o.type==="blur_region"){opBlur.value=String(Number(p.intensity)||16);opBlurNum.value=opBlur.value}
    if(o.type==="zoom_region"){opZoom.value=String(Number(p.factor)||1.5);opZoomNum.value=opZoom.value}
    if(p.region){regiaoAtual={x:Number(p.region.x)||0,y:Number(p.region.y)||0,width:Math.max(.001,Number(p.region.width)||1),height:Math.max(.001,Number(p.region.height)||1)}}
    atualizarCamposOperacao();
    // atualizarCamposOperacao configura as opções válidas; restaura então os valores exatos salvos.
    if(o.type==="overlay_video"){opAudio.value=p.audio_policy||"base";opPlayback.value=p.playback_policy||"both";opVolumeBase.value=String(Number(p.base_volume??100));opVolumeOverlay.value=String(Number(p.overlay_volume??100))}
    if(o.type==="insert_video"||o.type==="replace_video"){opAudio.value=p.audio_policy||"overlay";opVolumeOverlay.value=String(Number(p.overlay_volume??100))}
    if(o.type==="background_audio"){opAudio.value=p.audio_policy||"both";opVolumeBase.value=String(Number(p.base_volume??100));opVolumeOverlay.value=String(Number(p.overlay_volume??30))}
    if(p.region){regiaoAtual={x:Number(p.region.x)||0,y:Number(p.region.y)||0,width:Math.max(.001,Number(p.region.width)||1),height:Math.max(.001,Number(p.region.height)||1)};caixaRegiao.hidden=false;requestAnimationFrame(()=>{aplicarRegiao();atualizarPreviewEdicao()})}
    atualizarVideo2Fonte();buscar(Number(o.start));renderizar();status("Editando "+nomeOp(o.type)+". Ajuste os campos/caixa e clique em Salvar alterações.")
}
function adicionarOperacao(){
    const t=tipoOperacaoEl.value;
    if(t==="cut"){
        if(operacaoEditandoId){if(inicio===null||fim===null||fim<=inicio){alert("Marque início e fim válidos para o corte.");return}const idx=operacoes.findIndex(o=>o.id===operacaoEditandoId);if(idx<0){operacaoEditandoId=null;adicionarCorte();return}const antigo=operacoes[idx];operacoes[idx]={...criarOp("cut",inicio,fim,{}),id:antigo.id,enabled:antigo.enabled!==false,track:antigo.track||"main",note:antigo.note||""};operacaoEditandoId=null;operacoes.sort((x,y)=>x.start-y.start);renderizar();status("Corte atualizado.");return}
        adicionarCorte();return
    }
    if(inicio===null){alert("Marque pelo menos o início.");return}
    let a=inicio,b=fim;
    if(t==="insert_video")b=a;else if(b===null||b<=a){alert("Para esta operação, marque início e fim válidos.");return}
    const p={};
    if(tiposComRegiao.has(t))p.region=lerRegiao();
    if(t==="overlay_text"){if(!opTexto.value.trim()){alert("Informe o texto.");return}p.text=opTexto.value.trim();p.font_size=Math.max(8,Math.min(240,Number(opTextoTamanho.value)||36));const mh=areaRealVideo().height;p.font_size_ratio=mh>0?p.font_size/mh:null;p.color=opTextoCor.value||"#ffffff";p.font=opTextoFonte.value||"sans-serif";p.background_color=opTextoFundoCor.value||"#000000";p.background_opacity=Math.max(0,Math.min(1,Number(opTextoFundoOpacidade.value||0)/100))}
    if(t==="shape_highlight"){p.shape=opFormaTipo.value||"rectangle";p.color=opFormaCor.value||"#ff3b30";p.thickness=Math.max(1,Math.min(80,Number(opFormaEspessura.value)||4));const mh=areaRealVideo().height;p.stroke_ratio=mh>0?p.thickness/mh:null;p.opacity=Math.max(.05,Math.min(1,Number(opFormaOpacidade.value||100)/100));p.fill=!!opFormaPreencher.checked;p.fill_opacity=Math.max(0,Math.min(1,Number(opFormaFillOpacidade.value||0)/100));p.direction=opFormaDirecao.value||"right"}
    if(["overlay_image","overlay_video","insert_video","replace_video","background_audio"].includes(t)){if(!opMedia.value.trim()){alert("Informe o arquivo de mídia.");return}p.media=opMedia.value.trim()}
    if(["overlay_video","insert_video","replace_video","background_audio"].includes(t)){const faixa=trechoVideo2Atual();if(!faixa){alert("Escolha um início/fim válido para a mídia 2.");return}p.media_in=faixa.inicio;p.media_out=faixa.fim}
    if(tiposMediaRegiao.has(t))p.fit=opFit.value||"contain";
    if(t==="overlay_video"){p.audio_policy=opAudio.value;p.playback_policy=opPlayback.value;p.presentation_mode=opModoVideo.value||"overlay";p.base_volume=Math.max(0,Math.min(100,Number(opVolumeBase.value)||0));p.overlay_volume=Math.max(0,Math.min(100,Number(opVolumeOverlay.value)||0))}
    if(t==="insert_video"||t==="replace_video"){p.audio_policy=opAudio.value||"overlay";p.overlay_volume=Math.max(0,Math.min(100,Number(opVolumeOverlay.value)))}
    if(t==="background_audio"){p.audio_policy=opAudio.value||"both";p.base_volume=Math.max(0,Math.min(100,Number(opVolumeBase.value)||0));p.overlay_volume=Math.max(0,Math.min(100,Number(opVolumeOverlay.value)||0))}
    if(t==="replace_video")p.duration_policy=opDuracaoReplace.value||"natural";
    if(t==="speed_segment"){const f=Math.max(.1,Math.min(16,Number(opVelocidadeTrecho.value)||1));p.factor=f}
    if(t==="blur_region")p.intensity=Number(opBlur.value)||16;
    if(t==="zoom_region")p.factor=Math.max(1,Math.min(5,Number(opZoom.value)||1.5));
    if(t==="black_bar")p.opacity=1;
    if(t==="crop_region")p.restore_canvas=true;
    const nova=criarOp(t,a,b,p);
    if(operacaoEditandoId){const idx=operacoes.findIndex(o=>o.id===operacaoEditandoId);if(idx>=0){const antiga=operacoes[idx];nova.id=antiga.id;nova.enabled=antiga.enabled!==false;nova.track=antiga.track||nova.track;nova.note=antiga.note||"";operacoes[idx]=nova}else operacoes.push(nova);operacaoEditandoId=null;operacoes.sort((x,y)=>x.start-y.start);renderizar();status("Alterações salvas na operação.")}
    else{operacoes.push(nova);operacoes.sort((x,y)=>x.start-y.start);renderizar();status("Operação registrada. A prévia aparece no intervalo marcado.")}
}

function restaurarAudioUsuario(){video.muted=estadoAudioUsuario.muted;video.volume=estadoAudioUsuario.volume}
function aplicarPoliticaAudio(s,ativos=null){
    if(!modoPreviaProjeto||midiaSequencial)return;
    ativos=ativos||operacoes.filter(o=>o.enabled!==false&&o.type==="overlay_video"&&Number(o.start)<=s&&s<=Number(o.end));
    const mutarPrincipal=operacoes.some(o=>o.enabled!==false&&o.type==="mute"&&Number(o.start)<=s&&s<=Number(o.end));
    const op=ativos.find(o=>o.type==="overlay_video");
    if(!op){restaurarAudioUsuario();if(mutarPrincipal)video.muted=true;}
    else{const p=op.params||{},pol=p.audio_policy||"base";video.muted=mutarPrincipal||!["base","both"].includes(pol);video.volume=Math.max(0,Math.min(1,Number(p.base_volume??100)/100));camadaOperacoes.querySelectorAll("video[data-op-start]").forEach(v=>{v.muted=!["overlay","both"].includes(pol);v.volume=Math.max(0,Math.min(1,Number(p.overlay_volume??100)/100))});}
    const bg=sincronizarAudioFundo(s);
    if(bg){const p=bg.params||{},pol=p.audio_policy||"both";if(pol==="overlay"||pol==="none"){video.muted=true;camadaOperacoes.querySelectorAll("video[data-op-start]").forEach(v=>v.muted=true)}else if(pol==="both"||pol==="base"){video.muted=mutarPrincipal;video.volume=Math.max(0,Math.min(1,Number(p.base_volume??100)/100))}if(pol==="none")pararAudioFundo()}
}
function limparSequencial(){
    if(midiaSequencial?.video){try{midiaSequencial.video.pause()}catch(_){}}
    midiaSequencial=null;camadaSequencial.innerHTML="";camadaSequencial.classList.remove("ativa");camadaSequencial.style.background="";
}
function finalizarSequencial(){
    if(!midiaSequencial)return;const q=midiaSequencial;limparSequencial();video.currentTime=limitar(q.retomar);ultimoTempoProjeto=video.currentTime;restaurarAudioUsuario();renderizarOverlaysAtivos(video.currentTime,true);if(modoPreviaProjeto)video.play().catch(()=>{})
}
function iniciarSequencial(op,retomar,regional=false){
    pararAudioFundo();if(midiaSequencial)return;const p=op.params||{},url=caminhoMediaUrl(p.media);if(!url){status("Mídia não encontrada para a prévia: "+(p.media||""),true);eventosConsumidos.add(op.id);return}
    eventosConsumidos.add(op.id);video.pause();camadaOperacoes.querySelectorAll("video").forEach(v=>v.pause());
    const cont=document.createElement("div");cont.style.position="absolute";cont.style.overflow="hidden";cont.style.background="#000";
    if(regional&&p.region){estiloRegiao(cont,p.region)}else{cont.style.inset="0"}
    const v=document.createElement("video");v.src=url;v.playsInline=true;v.preload="auto";v.style.width="100%";v.style.height="100%";v.style.objectFit=p.fit||"contain";v.volume=Math.max(0,Math.min(1,Number(p.overlay_volume??100)/100));v.muted=(p.audio_policy||"overlay")==="none";
    cont.appendChild(v);camadaSequencial.innerHTML="";camadaSequencial.appendChild(cont);camadaSequencial.classList.add("ativa");camadaSequencial.style.background=regional?"transparent":"#000";
    midiaSequencial={video:v,op:op,retomar:retomar};
    let fimMidiaSelecionada=null;
    v.addEventListener("loadedmetadata",()=>{const faixa=faixaMidia(p,v.duration);fimMidiaSelecionada=faixa.fim;try{v.currentTime=Math.min(faixa.inicio,Math.max(0,v.duration-.04))}catch(_){}if(op.type==="replace_video"&&p.duration_policy==="fit_interval"){const alvo=Math.max(.05,Number(op.end)-Number(op.start)),dur=faixa.duracao??Math.max(.05,v.duration-faixa.inicio);if(Number.isFinite(dur)&&dur>0)v.playbackRate=Math.max(.1,Math.min(16,dur/alvo))}});
    const limiteRegional=regional?Math.max(.05,Number(op.end)-Number(op.start)):null;
    v.addEventListener("timeupdate",()=>{if(fimMidiaSelecionada!==null&&v.currentTime>=fimMidiaSelecionada-.03){finalizarSequencial();return}if(limiteRegional!==null){const faixa=faixaMidia(p,v.duration);if(v.currentTime-faixa.inicio>=limiteRegional-.03)finalizarSequencial()}});
    v.addEventListener("ended",finalizarSequencial);v.addEventListener("error",()=>{status("Não foi possível reproduzir "+(p.media||"mídia")+" na prévia.",true);finalizarSequencial()});
    v.play().catch(()=>status("Clique em Reproduzir se o navegador bloquear o áudio do vídeo 2.",true));
}
function saltoCorteEdicao(s){if(modoPreviaProjeto||!saltarCortes?.checked)return false;const c=operacoes.find(o=>o.enabled!==false&&o.type==="cut"&&s>=Number(o.start)&&s<Number(o.end));if(c){video.currentTime=Math.min(Number(c.end)+.001,video.duration||duracaoEsperada);return true}return false}
function executarPreviaProjeto(s){
    if(!modoPreviaProjeto||midiaSequencial)return;
    aplicarVelocidadeProjeto(s);
    if(s<ultimoTempoProjeto-.35)eventosConsumidos.clear();
    const corte=operacoes.find(o=>o.enabled!==false&&o.type==="cut"&&s>=Number(o.start)&&s<Number(o.end)-.002);if(corte){video.currentTime=limitar(Number(corte.end)+.001);ultimoTempoProjeto=video.currentTime;return}
    const candidatos=operacoes.filter(o=>o.enabled!==false&&["insert_video","replace_video","overlay_video"].includes(o.type)&&!eventosConsumidos.has(o.id));
    for(const o of candidatos){const a=Number(o.start),atingiu=(ultimoTempoProjeto<=a&&s>=a-.04)||(s>=a&&s<a+.10);if(!atingiu)continue;
        if(o.type==="insert_video"){iniciarSequencial(o,a+.001,false);ultimoTempoProjeto=s;return}
        if(o.type==="replace_video"){iniciarSequencial(o,Number(o.end)+.001,false);ultimoTempoProjeto=s;return}
        if(o.type==="overlay_video"&&(o.params||{}).playback_policy==="overlay_only"){iniciarSequencial(o,a+.001,true);ultimoTempoProjeto=s;return}
    }
    ultimoTempoProjeto=s;
}
function moverPalcoParaPreviewLocal(){
    if(!previewLocal||!previewLocalPalco)return;previewLocal.hidden=false;if(palco.parentNode!==previewLocalPalco)previewLocalPalco.appendChild(palco);requestAnimationFrame(()=>{aplicarRegiao();aplicarEstiloLegenda();renderizarOverlaysAtivos(video.currentTime,true);previewLocal.scrollIntoView({behavior:"smooth",block:"nearest"})});
}
function restaurarPalcoOriginal(){
    if(!palcoPaiOriginal||palco.parentNode===palcoPaiOriginal)return;if(palcoProximoOriginal&&palcoProximoOriginal.parentNode===palcoPaiOriginal)palcoPaiOriginal.insertBefore(palco,palcoProximoOriginal);else palcoPaiOriginal.appendChild(palco);if(previewLocal)previewLocal.hidden=true;requestAnimationFrame(()=>{aplicarRegiao();aplicarEstiloLegenda();renderizarOverlaysAtivos(video.currentTime,true)});
}

function iniciarPreviaProjeto(doInicio=true){
    limparSequencial();estadoAudioUsuario={muted:video.muted,volume:video.volume};velocidadeBaseProjeto=video.playbackRate||1;modoPreviaProjeto=true;eventosConsumidos.clear();moverPalcoParaPreviewLocal();if(doInicio)video.currentTime=0;atualizarCaixaRegiaoNaPrevia(video.currentTime);ultimoTempoProjeto=video.currentTime;modoPreviewEl.textContent="prévia ativa";modoPreviewEl.style.borderColor="#3a8f5c";status(doInicio?"Prévia do projeto iniciada do começo.":"Prévia do projeto iniciada deste ponto.");renderizarOverlaysAtivos(video.currentTime,true);video.play().catch(()=>status("Clique novamente em Reproduzir se o navegador bloquear a reprodução.",true))
}
function pararPreviaProjeto(){
    pararAudioFundo();modoPreviaProjeto=false;atualizarCaixaRegiaoNaPrevia(video.currentTime);video.pause();limparSequencial();restaurarAudioUsuario();video.playbackRate=velocidadeBaseProjeto;video.defaultPlaybackRate=velocidadeBaseProjeto;velocidadeEl.textContent=String(velocidadeBaseProjeto).replace(".",",")+"×";eventosConsumidos.clear();modoPreviewEl.textContent="edição";modoPreviewEl.style.borderColor="";renderizarOverlaysAtivos(video.currentTime,true);status("Prévia encerrada. Você voltou ao modo de edição.");restaurarPalcoOriginal()
}
async function salvarProjeto(){const dados=projetoAtual();if(!podeSalvarProjeto){status("Sem caminho de projeto no servidor. Use 'Baixar projeto JSON'.",true);return}try{const r=await fetch(apiBase+"/__editor_api__/projeto",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(dados)}),j=await r.json();if(!r.ok||!j.ok)throw new Error(j.erro||"Falha ao salvar");status("Projeto salvo: "+j.arquivo)}catch(e){status("Erro ao salvar projeto: "+e.message,true)}}


let timerSalvarPreferenciasLegenda=null;
function agendarSalvarPreferenciasLegenda(){renderizar();if(!podeSalvarProjeto)return;clearTimeout(timerSalvarPreferenciasLegenda);timerSalvarPreferenciasLegenda=setTimeout(async()=>{try{await fetch(apiBase+"/__editor_api__/projeto",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(projetoAtual())})}catch(_){ }},220)}

async function uploadMidia(file){if(!file)return;status("Copiando mídia para o projeto…");try{const r=await fetch(apiBase+"/__editor_api__/upload?name="+encodeURIComponent(file.name),{method:"POST",headers:{"Content-Type":"application/octet-stream"},body:file}),j=await r.json();if(!r.ok||!j.ok)throw new Error(j.erro||"Falha no upload");opMedia.value=j.arquivo;status("Mídia selecionada: "+j.arquivo);atualizarPreviewEdicao();atualizarVideo2Fonte()}catch(e){status("Erro ao selecionar mídia: "+e.message,true)}}
function desfazerUltimaOperacao(){if(!operacoes.length)return;const removida=operacoes.pop();if(removida?.id===operacaoEditandoId)operacaoEditandoId=null;renderizar();status("Última operação removida.")}

caixaRegiao.addEventListener("pointerdown",e=>{if(caixaRegiao.hidden)return;const m=areaRealVideo(),c=caixaRegiao.getBoundingClientRect(),resize=e.target.classList.contains("alca");interacaoRegiao={modo:resize?"resize":"move",pointerId:e.pointerId,startX:e.clientX,startY:e.clientY,left:c.left-m.left,top:c.top-m.top,width:c.width,height:c.height,pw:m.width,ph:m.height};caixaRegiao.setPointerCapture(e.pointerId);e.preventDefault()});
caixaRegiao.addEventListener("pointermove",e=>{const q=interacaoRegiao;if(!q||q.pointerId!==e.pointerId)return;const dx=e.clientX-q.startX,dy=e.clientY-q.startY;if(q.modo==="move"){let l=Math.max(0,Math.min(q.left+dx,q.pw-q.width)),t=Math.max(0,Math.min(q.top+dy,q.ph-q.height));regiaoAtual.x=l/q.pw;regiaoAtual.y=t/q.ph}else{let w=Math.max(30,Math.min(q.width+dx,q.pw-q.left)),h=Math.max(24,Math.min(q.height+dy,q.ph-q.top));regiaoAtual.width=w/q.pw;regiaoAtual.height=h/q.ph}aplicarRegiao();e.preventDefault()});
caixaRegiao.addEventListener("pointerup",e=>{if(interacaoRegiao&&interacaoRegiao.pointerId===e.pointerId){lerRegiao();atualizarPreviewEdicao();interacaoRegiao=null;try{caixaRegiao.releasePointerCapture(e.pointerId)}catch(_){}}});
window.addEventListener("resize",()=>{if(!caixaRegiao.hidden)aplicarRegiao();aplicarEstiloLegenda();renderizarOverlaysAtivos(video.currentTime,true)});

raiz.addEventListener("dblclick",e=>{const p=e.target.closest(".palavra-transcricao");if(p){e.preventDefault();e.stopPropagation();corrigirPalavra(p)}});
raiz.addEventListener("click",async e=>{const alvo=e.target;const interativo=alvo.closest("input,select,textarea,button,[contenteditable=true]");if(!interativo)raiz.focus();const palavra=e.target.closest(".palavra-transcricao");if(palavra){buscar(Number(palavra.dataset.start));return}const remover=e.target.closest("button[data-remover-op]");if(remover){const idx=Number(remover.dataset.removerOp),removida=operacoes[idx];if(removida?.id===operacaoEditandoId)operacaoEditandoId=null;operacoes.splice(idx,1);renderizar();return}const linhaEditar=e.target.closest("tr[data-editar-op]");if(linhaEditar&&!interativo){carregarOperacaoParaEdicao(linhaEditar.dataset.editarOp);return}const b=e.target.closest("button");if(!b)return;if(b.dataset.irTempo){buscar(Number(b.dataset.irTempo));return}if(b.dataset.passo){moverSegundos(Number(b.dataset.passo));return}if(b.dataset.velocidade){definirVelocidade(Number(b.dataset.velocidade));return}const a=b.dataset.acao;if(a==="play")alternarReproducao();else if(a==="inicio")marcarInicio();else if(a==="fim")marcarFim();else if(a==="copiar")await copiarTexto(saidaEl.value);else if(a==="baixar")downloadTexto("cortes.txt",saidaEl.value);else if(a==="salvar-transcricao")await salvarTranscricao();else if(a==="baixar-transcricao")downloadTexto("transcricao_corrigida.json",JSON.stringify(dadosTranscricao,null,2),"application/json;charset=utf-8");else if(a==="baixar-srt")downloadTexto("legendas.srt",gerarSRT(false));else if(a==="baixar-vtt")downloadTexto("legendas.vtt",gerarSRT(true));else if(a==="toggle-legenda"){legendasVisiveis=!legendasVisiveis;atualizarLegenda(video.currentTime)}else if(a==="mostrar-regiao"){caixaRegiao.hidden=!caixaRegiao.hidden;if(!caixaRegiao.hidden)requestAnimationFrame(aplicarRegiao)}else if(a==="resetar-regiao")resetarRegiao();else if(a==="adicionar-operacao")adicionarOperacao();else if(a==="cancelar-edicao")cancelarEdicaoOperacao();else if(a==="desfazer-operacao")desfazerUltimaOperacao();else if(a==="selecionar-midia")arquivoMidia.click();else if(a==="limpar-operacoes"){if(confirm("Apagar todas as operações adicionais? Os cortes serão mantidos.")){operacoes=operacoes.filter(o=>o.type==="cut");operacaoEditandoId=null;renderizar()}}else if(a==="salvar-projeto")await salvarProjeto();else if(a==="copiar-projeto")await copiarTexto(saidaOperacoesEl.value);else if(a==="baixar-projeto")downloadTexto("projeto_editor.json",saidaOperacoesEl.value,"application/json;charset=utf-8");else if(a==="preview-projeto")iniciarPreviaProjeto(true);else if(a==="preview-daqui")iniciarPreviaProjeto(false);else if(a==="parar-preview")pararPreviaProjeto();else if(a==="play-video2"){if(video2.paused)video2.play().catch(()=>status("Não foi possível reproduzir a mídia 2.",true));else video2.pause()}else if(a==="inicio-video2"){opMediaInicio.value=formatar(video2.currentTime)}else if(a==="fim-video2"){opMediaFim.value=formatar(video2.currentTime)}else if(a==="todo-video2"){opMediaInicio.value="00:00:00.000";opMediaFim.value=Number.isFinite(video2.duration)?formatar(video2.duration):"";try{video2.currentTime=0}catch(_){}}});

tipoOperacaoEl.addEventListener("change",atualizarCamposOperacao);
[opFormaTipo,opFormaCor,opFormaEspessura,opFormaOpacidade,opFormaPreencher,opFormaFillOpacidade,opFormaDirecao].forEach(el=>el.addEventListener("input",()=>{atualizarCamposOperacao();atualizarPreviewEdicao()}));
opTexto.addEventListener("input",atualizarPreviewEdicao);[opTextoTamanho,opTextoCor,opTextoFonte,opTextoFundoCor].forEach(el=>el.addEventListener("input",atualizarPreviewEdicao));opTextoFundoOpacidade.addEventListener("input",()=>{textoFundoPct.textContent=opTextoFundoOpacidade.value+"%";atualizarPreviewEdicao()});arquivoMidia.addEventListener("change",()=>uploadMidia(arquivoMidia.files?.[0]));
let timerVideo2Fonte=null;
opMedia.addEventListener("input",()=>{atualizarPreviewEdicao();clearTimeout(timerVideo2Fonte);timerVideo2Fonte=setTimeout(atualizarVideo2Fonte,350)});
opMedia.addEventListener("change",atualizarVideo2Fonte);
opFit.addEventListener("change",atualizarPreviewEdicao);
opModoVideo.addEventListener("change",()=>{atualizarCamposOperacao();atualizarPreviewEdicao()});
opAudio.addEventListener("change",()=>{if(modoPreviaProjeto)aplicarPoliticaAudio(video.currentTime)});
opVolumeBase.addEventListener("input",()=>{if(modoPreviaProjeto)aplicarPoliticaAudio(video.currentTime)});
opVolumeOverlay.addEventListener("input",()=>{if(modoPreviaProjeto)aplicarPoliticaAudio(video.currentTime)});
function sincronizarBlur(orig){const v=Math.max(1,Math.min(60,Number(orig.value)||16));opBlur.value=v;opBlurNum.value=v;atualizarPreviewEdicao()}opBlur.addEventListener("input",()=>sincronizarBlur(opBlur));opBlurNum.addEventListener("input",()=>sincronizarBlur(opBlurNum));function sincronizarZoom(orig){const v=Math.max(1,Math.min(5,Number(orig.value)||1.5));opZoom.value=v;opZoomNum.value=v;atualizarPreviewEdicao()}opZoom.addEventListener("input",()=>sincronizarZoom(opZoom));opZoomNum.addEventListener("input",()=>sincronizarZoom(opZoomNum));
video2.addEventListener("loadedmetadata",()=>{tempoVideo2El.textContent=formatar(video2.currentTime);if(!opMediaFim.value.trim()&&Number.isFinite(video2.duration))opMediaFim.value=formatar(video2.duration);seekVideo2Fim.textContent=formatarCurto(video2.duration);seekVideo2.value="0"});
video2.addEventListener("timeupdate",()=>{tempoVideo2El.textContent=formatar(video2.currentTime);if(Number.isFinite(video2.duration)&&video2.duration>0)seekVideo2.value=String(Math.round(video2.currentTime/video2.duration*1000))});seekVideo2.addEventListener("input",()=>{if(Number.isFinite(video2.duration)&&video2.duration>0){video2.currentTime=Number(seekVideo2.value)/1000*video2.duration;tempoVideo2El.textContent=formatar(video2.currentTime)}});
video2.addEventListener("error",()=>status("Não foi possível abrir a mídia 2. Confira o nome/caminho.",true));
posicaoLegendaEl.addEventListener("change",()=>{aplicarEstiloLegenda();atualizarLegenda(video.currentTime);agendarSalvarPreferenciasLegenda()});
tamanhoLegendaEl.addEventListener("input",()=>{aplicarEstiloLegenda();atualizarLegenda(video.currentTime)});
tamanhoLegendaEl.addEventListener("change",agendarSalvarPreferenciasLegenda);
ajusteLegendaEl.addEventListener("input",()=>{ajusteLegendaValorEl.textContent=(Number(ajusteLegendaEl.value)||0)+"%";aplicarEstiloLegenda();atualizarLegenda(video.currentTime)});
ajusteLegendaEl.addEventListener("change",agendarSalvarPreferenciasLegenda);
incorporarLegendaEl.addEventListener("change",agendarSalvarPreferenciasLegenda);
video.addEventListener("error",()=>erroEl.style.display="block");video.addEventListener("ended",()=>{if(modoPreviaProjeto)pararPreviaProjeto()});video.addEventListener("loadedmetadata",()=>{erroEl.style.display="none";video.preservesPitch=true;definirVelocidade(1);const migrou=migrarRegioesLegadas();let migrouTexto=false;const mh=areaRealVideo().height;if(mh>0){operacoes.forEach(o=>{if(o.type==="overlay_text"&&o.params&&o.params.font_size_ratio==null&&o.params.font_size!=null){o.params.font_size_ratio=Math.max(.001,Number(o.params.font_size)/mh);migrouTexto=true}})}aplicarRegiao();aplicarEstiloLegenda();seekFim.textContent=formatarCurto(video.duration);seekPrincipal.value="0";if(migrou||migrouTexto){renderizar();if(podeSalvarProjeto)setTimeout(()=>salvarProjeto(),120)}else atualizarTempo(0)});seekPrincipal.addEventListener("input",()=>{if(Number.isFinite(video.duration)&&video.duration>0){buscar(Number(seekPrincipal.value)/1000*video.duration)}});video.addEventListener("timeupdate",()=>{const s=video.currentTime;if(saltoCorteEdicao(s)){atualizarTempo(video.currentTime);return}if(modoPreviaProjeto)atualizarCaixaRegiaoNaPrevia(s);executarPreviaProjeto(s);atualizarTempo(video.currentTime);if(Number.isFinite(video.duration)&&video.duration>0)seekPrincipal.value=String(Math.round(s/video.duration*1000))});video.addEventListener("seeked",()=>{if(modoPreviaProjeto&&!midiaSequencial)ultimoTempoProjeto=video.currentTime;atualizarTempo(video.currentTime)});
if("requestVideoFrameCallback" in HTMLVideoElement.prototype){const acompanhar=(_a,m)=>{if(modoPreviaProjeto)atualizarCaixaRegiaoNaPrevia(m.mediaTime);atualizarTempo(m.mediaTime);video.requestVideoFrameCallback(acompanhar)};video.requestVideoFrameCallback(acompanhar)}
raiz.tabIndex=0;raiz.addEventListener("keydown",async e=>{if(["TEXTAREA","INPUT","SELECT"].includes(e.target.tagName))return;const t=e.key.toLowerCase();if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();let p=1;if(e.ctrlKey&&e.shiftKey)p=.001;else if(e.ctrlKey)p=.01;else if(e.shiftKey)p=.1;moverSegundos((e.key==="ArrowLeft"?-1:1)*p);return}if(t===" "){e.preventDefault();alternarReproducao()}else if(t==="j"){e.preventDefault();moverSegundos(-10)}else if(t==="l"){e.preventDefault();moverSegundos(10)}else if(t==="i"){e.preventDefault();marcarInicio()}else if(t==="o"){e.preventDefault();marcarFim()}else if(t==="u"){e.preventDefault();desfazerUltimaOperacao()}else if(e.key==="Delete"){e.preventDefault();limparCortes()}else if(t==="c"&&!e.ctrlKey&&!e.metaKey){e.preventDefault();await copiarTexto(saidaEl.value)}else if(t==="b"){e.preventDefault();downloadTexto("cortes.txt",saidaEl.value)}else if(e.key==="Enter"){e.preventDefault();adicionarOperacao()}else if(["1","2","3","4","5"].includes(e.key)){e.preventDefault();definirVelocidade({"1":.1,"2":.25,"3":.5,"4":.75,"5":1}[e.key])}});

sincronizarCortes();renderizar();renderizarTranscricao();atualizarCamposOperacao();aplicarEstiloLegenda();atualizarLegenda(0);
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
    if modo_colab:
        if porta_colab is None or handler_colab is None:
            raise RuntimeError("Falha ao inicializar o servidor do editor no Colab.")
        handler_colab.pagina_editor = conteudo
        try:
            from google.colab import output as colab_output  # type: ignore
            # O editor permanece dentro do notebook e pode rolar internamente.
            colab_output.serve_kernel_port_as_iframe(
                porta_colab, path="/", width="100%", height="1150"
            )
        except Exception as erro:
            raise RuntimeError(
                "O editor iniciou, mas o Colab não conseguiu expor o servidor "
                "no iframe. Tente reiniciar a sessão do Colab."
            ) from erro
    else:
        display(HTML(conteudo))

