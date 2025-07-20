import streamlit as st
import numpy as np
import cv2
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional
import soundfile as sf
from scipy.signal import butter, lfilter
import librosa # Importa librosa
import librosa.display # Per eventuali visualizzazioni future se necessarie

# Costanti globali (puoi modificarle)
MAX_DURATION = 300  # Durata massima del video in secondi
MIN_DURATION = 1.0  # Durata minima del video in secondi
MAX_FILE_SIZE = 50 * 1024 * 1024  # Dimensione massima del file (50 MB)
AUDIO_SAMPLE_RATE = 44100 # Frequenza di campionamento per l'audio generato
AUDIO_FPS = 30 # Frame per secondo dell'audio (dovrebbe corrispondere al video per semplicità)

# Definizioni delle risoluzioni per i formati
FORMAT_RESOLUTIONS = {
    "Originale": (0, 0),
    "1:1 (Quadrato)": (720, 720),
    "16:9 (Orizzontale)": (1280, 720),
    "9:16 (Verticale)": (720, 1280)
}

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

# Modifiche alla classe AudioGenerator
class AudioGenerator:
    def __init__(self, sample_rate: int, fps: int):
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_per_frame = self.sample_rate // self.fps # Quanti campioni audio per ogni frame video

    def generate_base_waveform(self, duration_samples: int) -> np.ndarray:
        """Genera un'onda a dente di sega (sawtooth) come base, ricca di armoniche."""
        base_freq = 220.0 
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)
        waveform = 2 * (t * base_freq - np.floor(t * base_freq + 0.5))
        return waveform.astype(np.float32)

    def apply_filter_dynamic(self, base_audio: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray, 
                             min_cutoff: float, max_cutoff: float, min_res: float, max_res: float,
                             min_pitch_shift_semitones: float, max_pitch_shift_semitones: float,
                             min_time_stretch_rate: float, max_time_stretch_rate: float) -> np.ndarray:
        """
        Applica un filtro passa-basso dinamico, pitch shifting e time stretching all'audio di base, 
        modulato dai dati visivi.
        """
        st.info("🎶 Inizializzazione della generazione audio sperimentale...")
        
        # Inizialmente, l'audio filtrato sarà la base per gli effetti successivi
        filtered_audio = np.zeros_like(base_audio)
        
        filter_order = 4
        zi = np.zeros(filter_order) 

        num_frames = len(brightness_data)
        
        # Pre-processamento per gli effetti pitch/time (se si applicano per frame)
        # useremo segmenti di audio filtrato per applicare questi effetti
        
        status_text = st.empty()
        
        # --- Fase 1: Filtro Dinamico (come prima) ---
        progress_bar_filter = st.progress(0, text="🎶 Applicazione Filtro Dinamico...")
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(base_audio))
            
            audio_segment = base_audio[frame_start_sample:frame_end_sample]
            if audio_segment.size == 0: continue 

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]
            
            cutoff_freq = min_cutoff + current_brightness * (max_cutoff - min_cutoff)
            resonance_q = min_res + current_detail * (max_res - min_res) 

            nyquist = 0.5 * self.sample_rate
            normal_cutoff = cutoff_freq / nyquist
            normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999) 

            b, a = butter(filter_order, normal_cutoff, btype='lowpass', analog=False, output='ba')
            filtered_segment, zi = lfilter(b, a, audio_segment, zi=zi)
            
            filtered_audio[frame_start_sample:frame_end_sample] = filtered_segment
            progress_bar_filter.progress((i + 1) / num_frames)
            status_text.text(f"🎶 Filtro Dinamico Frame {i + 1}/{num_frames} | Cutoff: {int(cutoff_freq)} Hz | Q: {resonance_q:.2f}")

        # --- Fase 2: Pitch Shifting e Time Stretching ---
        st.info("🔊 Applicazione Pitch Shifting e Time Stretching...")
        final_audio = np.zeros_like(filtered_audio) # Useremo questo array per l'output finale
        
        progress_bar_effects = st.progress(0, text="🔊 Applicazione Effetti Audio...")
        
        # Processa l'audio in blocchi per gli effetti di pitch/time.
        # È più efficiente applicare questi effetti su blocchi piuttosto che campione per campione.
        # Scegli una dimensione del blocco, ad esempio 2048 campioni o l'equivalente di un frame.
        
        # Qui potremmo affrontare una scelta progettuale:
        # 1. Applicare pitch/time stretch a ogni segmento *originale* prima del filtro,
        #    e poi unire. (più complesso con le durate che cambiano)
        # 2. Applicare pitch/time stretch all'audio *filtrato*. (più semplice)
        # 3. Applicare gli effetti in modo globale con valori medi, o per segmenti più grandi.
        
        # Per questa implementazione iniziale, applicheremo gli effetti in modo "per frame"
        # sull'audio già filtrato, gestendo il time stretching ri-campionando l'audio.

        # Pitch shift e time stretch con librosa richiedono audio float32
        filtered_audio = filtered_audio.astype(np.float32)

        # Inizializza l'audio finale con una dimensione stimata per il time stretch
        # La dimensione finale può variare leggermente, quindi gestiremo l'append
        output_segments = []
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(filtered_audio))
            
            audio_segment = filtered_audio[frame_start_sample:frame_end_sample]
            if audio_segment.size == 0: continue

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]

            # Mappa il dettaglio al pitch shift (es. più dettaglio = pitch più alto)
            pitch_shift_semitones = min_pitch_shift_semitones + current_detail * (max_pitch_shift_semitones - min_pitch_shift_semitones)
            # Mappa la luminosità al time stretch rate (es. più luminosità = più veloce)
            # Un rate di 1.0 significa velocità normale. >1.0 è più veloce, <1.0 è più lento.
            time_stretch_rate = min_time_stretch_rate + current_brightness * (max_time_stretch_rate - min_time_stretch_rate)
            time_stretch_rate = np.clip(time_stretch_rate, 0.5, 2.0) # Limita per evitare eccessi

            # Applica Pitch Shifting
            pitched_segment = librosa.effects.pitch_shift(
                y=audio_segment, 
                sr=self.sample_rate, 
                n_steps=pitch_shift_semitones
            )
            
            # Applica Time Stretching
            # librosa.effects.time_stretch richiede un DGT (Discrete Group Transform) che è un po' più complesso.
            # Un approccio più semplice (ma che può suonare meno bene per variazioni estreme)
            # è ri-campionare per il time stretch, ma librosa time_stretch è più robusto.
            # Per time_stretch, usiamo un 'hop_length' che influenza la granularità.
            # Per una gestione più semplice e robusta del time stretch con librosa, spesso si usa
            # librosa.stft per l'analisi e poi istft per la sintesi dopo la trasformazione.
            # Tuttavia, librosa.effects.time_stretch è un'interfaccia più semplice.
            
            # Per un time stretch che mantenga la stessa lunghezza di output del segmento originale,
            # ma con il contenuto "stretchato", dobbiamo calcolare il target_length.
            # `librosa.effects.time_stretch` cambia la durata. Per mantenere una durata costante
            # del *segmento di output*, si può fare un `resample` dopo, ma è più complesso.
            # Alternativa: usiamo il rate e lasciamo che la durata cambi, poi ri-campioniamo l'intero array finale.

            # Per la modularità, useremo time_stretch che modifica la durata del segmento.
            # Successivamente, dovremo ricampionare l'intero audio per farlo combaciare con la durata del video.
            stretched_segment = librosa.effects.time_stretch(y=pitched_segment, rate=time_stretch_rate)
            
            output_segments.append(stretched_segment)

            progress_bar_effects.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Effetti Audio Frame {i + 1}/{num_frames} | Pitch: {pitch_shift_semitones:.1f} semitoni | Stretch: {time_stretch_rate:.2f}")

        # Unisci tutti i segmenti processati
        combined_audio = np.concatenate(output_segments)

        # Ricampiona l'audio combinato alla durata esatta del video (se necessario)
        # Questo è cruciale perché time_stretch modifica la lunghezza.
        target_total_samples = int(num_frames * self.samples_per_frame)
        if len(combined_audio) != target_total_samples:
            st.info(f"🔄 Ricampionamento audio per adattarsi alla durata video (da {len(combined_audio)} a {target_total_samples} campioni)...")
            final_audio = librosa.resample(y=combined_audio, orig_sr=self.sample_rate, target_sr=self.sample_rate, res_type='kaiser_best', scale=False, fix=True, to_mono=True, axis=-1, length=target_total_samples)
        else:
            final_audio = combined_audio

        # Normalizza l'audio finale per evitare clipping
        if np.max(np.abs(final_audio)) > 0:
            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.9 
            
        return final_audio.astype(np.float32)


def main():
    st.set_page_config(page_title="🎵 VideoSound Gen - Sperimentale", layout="centered")
    st.title("🎬 VideoSound Gen - Sperimentale")
    st.markdown("###### by Loop507") 
    st.markdown("### Genera musica sperimentale da un video muto")
    st.markdown("Carica un video e osserva come le sue proprietà visive creano un paesaggio sonoro dinamico attraverso la sintesi sottrattiva.")

    uploaded_file = st.file_uploader("🎞️ Carica un file video (.mp4, .mov, ecc.)", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        if not validate_video_file(uploaded_file):
            return
            
        base_name_upload = os.path.splitext(uploaded_file.name)[0]
        unique_id = str(np.random.randint(10000, 99999)) 
        video_input_path = os.path.join("temp", f"{base_name_upload}_{unique_id}.mp4")
        os.makedirs("temp", exist_ok=True) 
        with open(video_input_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("🎥 Video caricato correttamente!")

        with st.spinner("📊 Analisi frame video (luminosità, dettaglio) in corso..."):
            brightness_data, detail_data, width, height, fps, video_duration = analyze_video_frames(video_input_path)
        
        if brightness_data is None: 
            return
        
        st.info(f"🎥 Durata video: {video_duration:.2f} secondi | Risoluzione Originale: {width}x{height} | FPS: {fps:.2f}")

        st.markdown("---")
        st.subheader("🎶 Configurazione Sintesi Audio Sperimentale")

        # Slider per il controllo dei range di mappatura (opzionale per l'utente)
        st.sidebar.header("Parametri Sintesi Sottrattiva")
        min_cutoff_user = st.sidebar.slider("Min Frequenza Taglio (Hz)", 20, 5000, 100)
        max_cutoff_user = st.sidebar.slider("Max Frequenza Taglio (Hz)", 1000, 20000, 8000)
        min_resonance_user = st.sidebar.slider("Min Risonanza (Q)", 0.1, 5.0, 0.5) 
        max_resonance_user = st.sidebar.slider("Max Risonanza (Q)", 1.0, 30.0, 10.0) 

        # --- Nuovi Slider per Pitch Shifting / Time Stretching ---
        st.sidebar.header("Parametri Pitch/Time Stretching")
        st.sidebar.markdown("*(Luminosità controlla il Time Stretching, Dettaglio il Pitch Shifting)*")
        min_pitch_shift_semitones = st.sidebar.slider("Min Pitch Shift (semitoni)", -24.0, 24.0, -12.0, 0.5)
        max_pitch_shift_semitones = st.sidebar.slider("Max Pitch Shift (semitoni)", -24.0, 24.0, 12.0, 0.5)
        # Rate: 1.0 = normale, <1.0 = più lento, >1.0 = più veloce
        min_time_stretch_rate = st.sidebar.slider("Min Time Stretch Rate", 0.1, 2.0, 0.8, 0.1) 
        max_time_stretch_rate = st.sidebar.slider("Max Time Stretch Rate", 0.5, 5.0, 1.5, 0.1) 


        st.markdown("---")
        st.subheader("⬇️ Opzioni di Download")
        
        # Selezione della risoluzione di output
        output_resolution_choice = st.selectbox(
            "Seleziona la risoluzione di output del video:",
            list(FORMAT_RESOLUTIONS.keys())
        )
        
        # Scegli cosa scaricare
        download_option = st.radio(
            "Cosa vuoi scaricare?",
            ("Video con Audio", "Solo Audio")
        )

        if not check_ffmpeg():
            st.warning("⚠️ FFmpeg non disponibile sul tuo sistema. L'unione o la ricodifica del video potrebbe non funzionare. Assicurati che FFmpeg sia installato e nel PATH.")
            
        if st.button("🎵 Genera e Prepara Download"):
            base_name_output = os.path.splitext(uploaded_file.name)[0]
            audio_output_path = os.path.join("temp", f"{base_name_output}_{unique_id}_generated_audio.wav") 
            final_video_path = os.path.join("temp", f"{base_name_output}_{unique_id}_final_videosound.mp4")
            
            os.makedirs("temp", exist_ok=True)

            audio_gen = AudioGenerator(AUDIO_SAMPLE_RATE, int(fps)) 
            
            total_samples = int(video_duration * AUDIO_SAMPLE_RATE)
            base_waveform = audio_gen.generate_base_waveform(total_samples)

            with st.spinner("🎧 Generazione audio sperimentale e applicazione filtri dinamici..."):
                generated_audio = audio_gen.apply_filter_dynamic(
                    base_waveform, 
                    brightness_data, 
                    detail_data,
                    min_cutoff=min_cutoff_user, 
                    max_cutoff=max_cutoff_user,
                    min_res=min_resonance_user, 
                    max_res=max_resonance_user,
                    min_pitch_shift_semitones=min_pitch_shift_semitones,
                    max_pitch_shift_semitones=max_pitch_shift_semitones,
                    min_time_stretch_rate=min_time_stretch_rate,
                    max_time_stretch_rate=max_time_stretch_rate
                )
            
            if generated_audio is None or generated_audio.size == 0:
                st.error("❌ Errore nella generazione dell'audio.")
                return

            try:
                sf.write(audio_output_path, generated_audio, AUDIO_SAMPLE_RATE)
                st.success(f"✅ Audio sperimentale generato e salvato in '{audio_output_path}'")
            except Exception as e:
                st.error(f"❌ Errore nel salvataggio dell'audio WAV: {str(e)}")
                return
            
            # Gestione del download in base alla scelta
            if download_option == "Solo Audio":
                with open(audio_output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica Solo Audio (WAV)", 
                        f,
                        file_name=f"videosound_sottrattiva_audio_{base_name_output}.wav",
                        mime="audio/wav" 
                    )
                if os.path.exists(video_input_path):
                    os.remove(video_input_path) 
                if os.path.exists(audio_output_path):
                    os.remove(audio_output_path)
                st.info("🗑️ File temporanei puliti.")

            elif download_option == "Video con Audio":
                if check_ffmpeg():
                    with st.spinner("🔗 Unione e ricodifica video/audio con FFmpeg (potrebbe richiedere tempo)..."):
                        target_width, target_height = FORMAT_RESOLUTIONS[output_resolution_choice]
                        
                        if output_resolution_choice == "Originale":
                            target_width = width
                            target_height = height

                        vf_complex = f"scale='min({target_width},iw*({target_height}/ih)):min({target_height},ih*({target_width}/iw))',pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black"

                        try:
                            subprocess.run([
                                "ffmpeg", "-y",
                                "-i", video_input_path, 
                                "-i", audio_output_path, 
                                "-c:v", "libx264", 
                                "-preset", "medium", 
                                "-crf", "23", 
                                "-vf", vf_complex, 
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
                                    file_name=f"videosound_sottrattiva_{base_name_output}_{output_resolution_choice.replace(' ', '_')}.mp4",
                                    mime="video/mp4"
                                )
                            
                            for temp_f in [video_input_path, audio_output_path, final_video_path]:
                                if os.path.exists(temp_f):
                                    os.remove(temp_f)
                            st.info("🗑️ File temporanei puliti.")

                        except subprocess.CalledProcessError as e:
                            st.error(f"❌ Errore FFmpeg durante l'unione/ricodifica: {e.stderr.decode()}")
                            st.code(e.stdout.decode() + e.stderr.decode()) 
                        except Exception as e:
                            st.error(f"❌ Errore generico durante l'unione/ricodifica: {str(e)}")
                else:
                    st.warning(f"⚠️ FFmpeg non trovato. Il video con audio non può essere unito o ricodificato. L'audio generato è disponibile in '{audio_output_path}'.")
                    with open(audio_output_path, "rb") as f:
                        st.download_button(
                            "⬇️ Scarica Solo Audio (WAV temporaneo)",
                            f,
                            file_name=f"videosound_sottrattiva_audio_{base_name_output}.wav",
                            mime="audio/wav"
                        )


if __name__ == "__main__":
    main()
