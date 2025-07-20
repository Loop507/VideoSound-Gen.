import streamlit as st
import numpy as np
import cv2
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional
import soundfile as sf # Assicurati di aver installato: pip install soundfile
from scipy.signal import butter, lfilter, iirfilter # Assicurati di aver installato: pip install scipy

# Costanti globali (puoi modificarle)
MAX_DURATION = 300  # Durata massima del video in secondi
MIN_DURATION = 1.0  # Durata minima del video in secondi
MAX_FILE_SIZE = 50 * 1024 * 1024  # Dimensione massima del file (50 MB)
AUDIO_SAMPLE_RATE = 44100 # Frequenza di campionamento per l'audio generato
AUDIO_FPS = 30 # Frame per secondo dell'audio (dovrebbe corrispondere al video per semplicità)


def check_ffmpeg() -> bool:
    """Verifica se FFmpeg è installato e disponibile nel PATH."""
    return shutil.which("ffmpeg") is not None

def validate_video_file(uploaded_file) -> bool:
    """Valida le dimensioni del file video caricato."""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def analyze_video_frames(video_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Carica il video, estrae i frame e analizza luminosità e dettaglio/contrasto.
    Restituisce array di luminosità e dettaglio per frame, e info sul video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il file video.")
        return None, None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.error("❌ Impossibile leggere il framerate del video.")
        cap.release()
        return None, None, None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    if video_duration < MIN_DURATION:
        st.error(f"❌ Il video deve essere lungo almeno {MIN_DURATION} secondi. Durata attuale: {video_duration:.2f}s")
        cap.release()
        return None, None, None, None, None, None
    
    if video_duration > MAX_DURATION:
        st.warning(f"⚠️ Video troppo lungo ({video_duration:.1f}s). Verranno analizzati solo i primi {MAX_DURATION} secondi.")
        total_frames = int(MAX_DURATION * fps)
        video_duration = MAX_DURATION

    brightness_data = []
    detail_data = [] # Useremo la varianza dei pixel per una stima del dettaglio/contrasto

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Converti il frame in scala di grigi per l'analisi
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Luminosità Media: Media dei pixel
        brightness = np.mean(gray_frame) / 255.0 # Normalizza tra 0 e 1
        brightness_data.append(brightness)

        # 2. Dettaglio/Contrasto: Varianza dei pixel (o deviazione standard)
        # Una varianza più alta indica più variazione di pixel, quindi più dettaglio/contrasto
        detail = np.std(gray_frame) / 255.0 # Normalizza tra 0 e 1
        detail_data.append(detail)

        progress_bar.progress((i + 1) / total_frames)
        status_text.text(f"📊 Analisi Frame {i + 1}/{total_frames} | Luminosità: {brightness:.2f} | Dettaglio: {detail:.2f}")

    cap.release()
    st.success("✅ Analisi video completata!")
    
    return np.array(brightness_data), np.array(detail_data), width, height, fps, video_duration

# Nuova funzione/classe per la generazione audio
class AudioGenerator:
    def __init__(self, sample_rate: int, fps: int):
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_per_frame = self.sample_rate // self.fps # Quanti campioni audio per ogni frame video

    def generate_base_waveform(self, duration_samples: int) -> np.ndarray:
        """Genera un'onda a dente di sega (sawtooth) come base, ricca di armoniche."""
        # Una frequenza fissa per la base, ad esempio 220 Hz (La3)
        base_freq = 220.0 
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)
        # Genera un'onda a dente di sega
        waveform = 2 * (t * base_freq - np.floor(t * base_freq + 0.5))
        return waveform.astype(np.float32)

    def apply_filter_dynamic(self, base_audio: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray, 
                             min_cutoff: float, max_cutoff: float, min_res: float, max_res: float) -> np.ndarray:
        """
        Applica un filtro passa-basso dinamico all'audio di base, modulato dai dati visivi.
        """
        st.info("🎶 Inizializzazione della generazione audio sperimentale (Sintesi Sottrattiva)...")
        
        generated_audio = np.zeros_like(base_audio)
        
        filter_order = 4 # Ordine del filtro (Butterworth di 4° ordine)
        
        # Inizializza lo stato del filtro per una transizione più fluida tra i frame
        # Il numero di stati (zi) dipende dall'ordine del filtro
        # Per un filtro di ordine N, lfilter con output='ba' richiede N stati
        zi = np.zeros(filter_order) 

        num_frames = len(brightness_data)
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(base_audio)) # Evita sforamenti
            
            # Preleva il segmento audio per il frame corrente
            audio_segment = base_audio[frame_start_sample:frame_end_sample]
            
            if audio_segment.size == 0:
                continue # Salta frame se non c'è audio

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]
            
            # Calcola la frequenza di taglio e la risonanza per questo frame
            # Normalizza current_brightness e current_detail (già tra 0 e 1)
            # e mappali ai range definiti dagli slider dell'utente
            
            cutoff_freq = min_cutoff + current_brightness * (max_cutoff - min_cutoff)
            resonance_q = min_res + current_detail * (max_res - min_res) # Q per la risonanza

            # Normalizza la frequenza di taglio per scipy.signal (frequenza di Nyquist)
            nyquist = 0.5 * self.sample_rate
            
            # Per i filtri Butterworth, non c'è un parametro 'Q' diretto come nei filtri biquad.
            # L'argomento `Wn` in `butter` è la frequenza di taglio normalizzata.
            # Tuttavia, possiamo usare `iirfilter` per un filtro di tipo 'peaking' o 'notch'
            # se vogliamo un controllo esplicito di Q, ma per un passa-basso,
            # la risonanza è più una caratteristica del filtro stesso.
            # Useremo il 'detail' per variare leggermente il cutoff o l'ordine, o magari un Q implicito se usiamo un biquad.
            
            # Per una sintesi sottrattiva standard con un passa-basso Butterworth:
            # la frequenza di taglio è il parametro principale.
            # Possiamo usare il "dettaglio" per rendere il cutoff più sensibile o per modulare un altro parametro.
            
            # Esempio: usiamo la risonanza (Q) per modificare l'andamento del cutoff o aggiungere un leggero boost.
            # Per ora, manteniamo `normal_cutoff` come parametro principale.
            normal_cutoff = cutoff_freq / nyquist
            
            # Assicurati che il cutoff non sia troppo vicino a 0 o 1 (limiti di stabilità)
            normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999) # Range leggermente più ampio e sicuro

            # Progetta il filtro Butterworth
            # `btype='lowpass'` è il tipo, `analog=False` per filtri digitali
            # `output='ba'` restituisce i coefficienti b e a, necessari per `lfilter`
            b, a = butter(filter_order, normal_cutoff, btype='lowpass', analog=False, output='ba')
            
            # Applica il filtro al segmento audio, mantenendo lo stato (zi)
            # `lfilter` restituisce la serie filtrata e il nuovo stato finale del filtro
            filtered_segment, zi = lfilter(b, a, audio_segment, zi=zi)
            
            generated_audio[frame_start_sample:frame_end_sample] = filtered_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎶 Generazione audio Frame {i + 1}/{num_frames} | Cutoff: {int(cutoff_freq)} Hz | Q: {resonance_q:.2f}")

        # Normalizza l'audio finale per evitare clipping
        if np.max(np.abs(generated_audio)) > 0:
            generated_audio = generated_audio / np.max(np.abs(generated_audio)) * 0.9 # Normalizza a 0.9 per sicurezza
            
        return generated_audio

def main():
    st.set_page_config(page_title="🎵 VideoSound Gen - Sperimentale", layout="centered")
    st.title("🎬 VideoSound Gen - Sperimentale")
    # Aggiungi la dicitura con caratteri più piccoli qui
    st.markdown("###### by Loop507") 
    st.markdown("### Genera musica sperimentale da un video muto")
    st.markdown("Carica un video e osserva come le sue proprietà visive creano un paesaggio sonoro dinamico attraverso la sintesi sottrattiva.")

    uploaded_file = st.file_uploader("🎞️ Carica un file video (.mp4, .mov, ecc.)", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        if not validate_video_file(uploaded_file):
            return
            
        # Salva il file video caricato localmente per OpenCV e FFmpeg
        # Genera un nome file unico per evitare conflitti e permettere upload multipli
        base_name_upload = os.path.splitext(uploaded_file.name)[0]
        # Creiamo un hash o un timestamp per rendere il nome file unico
        unique_id = str(np.random.randint(10000, 99999)) # Un ID semplice, per non usare datetime per brevità
        video_input_path = os.path.join("temp", f"{base_name_upload}_{unique_id}.mp4")
        os.makedirs("temp", exist_ok=True) # Assicurati che la directory 'temp' esista
        with open(video_input_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("🎥 Video caricato correttamente!")

        # Analizza i frame del video
        with st.spinner("📊 Analisi frame video (luminosità, dettaglio) in corso..."):
            brightness_data, detail_data, width, height, fps, video_duration = analyze_video_frames(video_input_path)
        
        if brightness_data is None: # Se l'analisi fallisce
            return
        
        st.info(f"🎥 Durata video: {video_duration:.2f} secondi | Risoluzione: {width}x{height} | FPS: {fps:.2f}")

        # Configurazione degli effetti audio
        st.markdown("---")
        st.subheader("🎶 Configurazione Sintesi Audio Sperimentale")

        # Slider per il controllo dei range di mappatura (opzionale per l'utente)
        st.sidebar.header("Parametri Sintesi Sottrattiva")
        min_cutoff_user = st.sidebar.slider("Min Frequenza Taglio (Hz)", 20, 5000, 100)
        max_cutoff_user = st.sidebar.slider("Max Frequenza Taglio (Hz)", 1000, 20000, 8000)
        min_resonance_user = st.sidebar.slider("Min Risonanza (Q)", 0.1, 5.0, 0.5) # Range più ragionevole per Q
        max_resonance_user = st.sidebar.slider("Max Risonanza (Q)", 1.0, 30.0, 10.0) # Range più ampio per sperimentazione

        # Verifichiamo FFmpeg prima di avviare il processo
        if not check_ffmpeg():
            st.warning("⚠️ FFmpeg non disponibile sul tuo sistema. Non sarà possibile unire l'audio al video. Assicurati che FFmpeg sia installato e nel PATH.")
            
        if st.button("🎵 Genera Audio e Unisci al Video"):
            # Genera nomi file unici per l'output
            base_name_output = os.path.splitext(uploaded_file.name)[0]
            audio_output_path = os.path.join("temp", f"{base_name_output}_{unique_id}_generated_audio.wav")
            final_video_path = os.path.join("temp", f"{base_name_output}_{unique_id}_final_videosound.mp4")
            
            # Assicurati che la directory 'temp' esista
            os.makedirs("temp", exist_ok=True)

            audio_gen = AudioGenerator(AUDIO_SAMPLE_RATE, int(fps)) # Usiamo gli FPS del video per l'audio
            
            # Genera la waveform di base per tutta la durata
            total_samples = int(video_duration * AUDIO_SAMPLE_RATE)
            base_waveform = audio_gen.generate_base_waveform(total_samples)

            # Applica il filtro dinamico e genera l'audio finale
            with st.spinner("🎧 Generazione audio sperimentale e applicazione filtri dinamici..."):
                generated_audio = audio_gen.apply_filter_dynamic(
                    base_waveform, 
                    brightness_data, 
                    detail_data,
                    min_cutoff=min_cutoff_user, 
                    max_cutoff=max_cutoff_user,
                    min_res=min_resonance_user, 
                    max_res=max_resonance_user
                )
            
            if generated_audio is None or generated_audio.size == 0:
                st.error("❌ Errore nella generazione dell'audio.")
                return

            try:
                sf.write(audio_output_path, generated_audio, AUDIO_SAMPLE_RATE)
                st.success(f"✅ Audio sperimentale generato e salvato in '{audio_output_path}'")
            except Exception as e:
                st.error(f"❌ Errore nel salvataggio dell'audio: {str(e)}")
                return
            
            # Unisci l'audio al video usando FFmpeg
            if check_ffmpeg():
                with st.spinner("🔗 Unione audio e video con FFmpeg..."):
                    try:
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", video_input_path, 
                            "-i", audio_output_path, 
                            "-c:v", "copy", 
                            "-c:a", "aac", "-b:a", "192k", 
                            "-map", "0:v:0", "-map", "1:a:0", 
                            "-shortest", 
                            final_video_path
                        ], capture_output=True, check=True)
                        st.success(f"✅ Video finale con audio salvato in '{final_video_path}'")
                        
                        with open(final_video_path, "rb") as f:
                            st.download_button(
                                "⬇️ Scarica il Video con Audio",
                                f,
                                file_name=f"videosound_sottrattiva_{base_name_output}.mp4",
                                mime="video/mp4"
                            )
                        
                        # Pulizia files temporanei
                        for temp_f in [video_input_path, audio_output_path, final_video_path]:
                            if os.path.exists(temp_f):
                                os.remove(temp_f)
                        st.info("🗑️ File temporanei puliti.")

                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Errore FFmpeg durante l'unione: {e.stderr.decode()}")
                        st.code(e.stdout.decode() + e.stderr.decode()) 
                    except Exception as e:
                        st.error(f"❌ Errore generico durante l'unione: {str(e)}")
            else:
                st.warning(f"⚠️ FFmpeg non trovato. Il video con audio non può essere unito. L'audio generato è disponibile in '{audio_output_path}'.")
                with open(audio_output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica Solo Audio (FFmpeg non trovato per video)",
                        f,
                        file_name=f"videosound_sottrattiva_audio_{base_name_output}.wav",
                        mime="audio/wav"
                    )


if __name__ == "__main__":
    main()
