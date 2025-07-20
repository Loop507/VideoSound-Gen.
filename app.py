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
import librosa
import librosa.display

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

def analyze_video_frames(video_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Carica il video, estrae i frame e analizza luminosità, dettaglio/contrasto e movimento.
    Restituisce array di luminosità, dettaglio e movimento per frame, e info sul video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il file video.")
        return None, None, None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.error("❌ Impossibile leggere il framerate del video.")
        cap.release()
        return None, None, None, None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    if video_duration < MIN_DURATION:
        st.error(f"❌ Il video deve essere lungo almeno {MIN_DURATION} secondi. Durata attuale: {video_duration:.2f}s")
        cap.release()
        return None, None, None, None, None, None, None
    
    if video_duration > MAX_DURATION:
        st.warning(f"⚠️ Video troppo lungo ({video_duration:.1f}s). Verranno analizzati solo i primi {MAX_DURATION} secondi.")
        total_frames = int(MAX_DURATION * fps)
        video_duration = MAX_DURATION

    brightness_data = []
    detail_data = [] 
    movement_data = [] # Nuovo: per il movimento
    
    prev_gray_frame = None # Per il calcolo del movimento

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Luminosità Media: Media dei pixel
        brightness = np.mean(gray_frame) / 255.0 # Normalizza tra 0 e 1
        brightness_data.append(brightness)

        # 2. Dettaglio/Contrasto: Varianza dei pixel
        detail = np.std(gray_frame) / 255.0 # Normalizza tra 0 e 1
        detail_data.append(detail)

        # 3. Movimento: Differenza assoluta media tra frame consecutivi
        if prev_gray_frame is not None:
            # Calcola la differenza assoluta e normalizza per la dimensione del frame e il max pixel value (255)
            frame_diff = np.sum(cv2.absdiff(gray_frame, prev_gray_frame)) / (gray_frame.size * 255.0)
            movement_data.append(frame_diff)
        else:
            movement_data.append(0.0) # Primo frame non ha movimento rispetto a un precedente
        
        prev_gray_frame = gray_frame

        progress_bar.progress((i + 1) / total_frames)
        status_text.text(f"📊 Analisi Frame {i + 1}/{total_frames} | Lum: {brightness:.2f} | Det: {detail:.2f} | Mov: {movement_data[-1]:.2f}")

    cap.release()
    st.success("✅ Analisi video completata!")
    
    # Assicurati che movement_data abbia la stessa lunghezza degli altri array
    # Se il video ha solo 1 frame, movement_data sarà [0.0], quindi andrà bene
    if len(movement_data) < len(brightness_data):
        # Questo può accadere se il loop si interrompe prima o se c'è solo un frame
        # Per semplicità, replichiamo l'ultimo valore o aggiungiamo zeri
        # Per video di un solo frame, movement_data sarà [0.0] e gli altri [valore_frame1], quindi deve avere la stessa lunghezza
        if len(brightness_data) == 1 and len(movement_data) == 1:
            pass # Già allineato
        elif len(brightness_data) > 0:
            # Replicare l'ultimo valore per i frame mancanti (dovrebbe essere solo l'ultimo se si rompe il loop)
            # In un caso normale, length(movement_data) = length(brightness_data) - 1, quindi aggiungiamo 1
            if len(movement_data) == len(brightness_data) - 1:
                movement_data.append(movement_data[-1]) # Aggiungi l'ultimo valore
            else: # Caso più complesso, riempi con zeri o media
                while len(movement_data) < len(brightness_data):
                    movement_data.append(0.0) # Riempi con zeri

    return np.array(brightness_data), np.array(detail_data), np.array(movement_data), width, height, fps, video_duration

class AudioGenerator:
    def __init__(self, sample_rate: int, fps: int):
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_per_frame = self.sample_rate // self.fps 

    def generate_subtractive_waveform(self, duration_samples: int) -> np.ndarray:
        """Genera un'onda a dente di sega (sawtooth) come base per la sintesi sottrattiva."""
        base_freq = 220.0 
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)
        waveform = 2 * (t * base_freq - np.floor(t * base_freq + 0.5))
        return waveform.astype(np.float32)

    def generate_fm_layer(self, duration_samples: int,
                             brightness_data: np.ndarray, detail_data: np.ndarray, movement_data: np.ndarray, # Aggiunto movement_data
                             min_carrier_freq: float, max_carrier_freq: float,
                             min_modulator_freq: float, max_modulator_freq: float,
                             min_mod_index: float, max_mod_index: float) -> np.ndarray:
        """
        Genera un'onda FM modulata dai dati visivi (luminosità per carrier, movimento per modulator e index) come strato aggiuntivo.
        """
        st.info("🎵 Generazione Strato FM...")
        fm_audio_layer = np.zeros(duration_samples, dtype=np.float32)
        
        t_overall = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)

        num_frames = len(brightness_data)
        
        progress_bar = st.progress(0, text="🎵 Generazione Strato FM...")
        status_text = st.empty()

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, duration_samples)
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_brightness = brightness_data[i]
            current_detail = detail_data[i] # Mantenuto per completezza, ma non usato direttamente in FM qui
            current_movement = movement_data[i] # NUOVO

            # Mappatura dei parametri FM ai dati visivi
            # Carrier Frequency -> Luminosità (come prima)
            carrier_freq = min_carrier_freq + current_brightness * (max_carrier_freq - min_carrier_freq)
            
            # Modulator Frequency -> Movimento (NUOVO)
            modulator_freq = min_modulator_freq + current_movement * (max_modulator_freq - min_modulator_freq)
            
            # Modulation Index -> Movimento (NUOVO)
            mod_index = min_mod_index + current_movement * (max_mod_index - min_mod_index) 

            t_segment = t_overall[frame_start_sample:frame_end_sample]
            
            modulator_wave = np.sin(2 * np.pi * modulator_freq * t_segment)
            fm_segment = np.sin(2 * np.pi * carrier_freq * t_segment + mod_index * modulator_wave)
            
            fm_audio_layer[frame_start_sample:frame_end_sample] = fm_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎵 Strato FM Frame {i + 1}/{num_frames} | Fc: {int(carrier_freq)} Hz | Fm: {int(modulator_freq)} Hz | I: {mod_index:.2f}")
        
        return fm_audio_layer

    def process_audio_segments(self, base_audio: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray, 
                             min_cutoff: float, max_cutoff: float, min_res: float, max_res: float,
                             min_pitch_shift_semitones: float, max_pitch_shift_semitones: float,
                             min_time_stretch_rate: float, max_time_stretch_rate: float,
                             apply_filter: bool = True) -> np.ndarray:
        """
        Applica (opzionalmente) un filtro passa-basso dinamico, pitch shifting e time stretching 
        all'audio di base, modulato dai dati visivi.
        """
        st.info("🔊 Applicazione Effetti Audio dinamici (Filtro, Pitch, Time Stretch)...")
        
        processed_audio = base_audio.copy()
        
        filter_order = 4
        num_frames = len(brightness_data)
        status_text = st.empty()
        
        # --- Fase 1: Filtro Dinamico (se abilitato, per sintesi sottrattiva) ---
        if apply_filter:
            progress_bar_filter = st.progress(0, text="🎶 Applicazione Filtro Dinamico...")
            zi = np.zeros(filter_order) 
            for i in range(num_frames):
                frame_start_sample = i * self.samples_per_frame
                frame_end_sample = min((i + 1) * self.samples_per_frame, len(processed_audio))
                
                audio_segment = processed_audio[frame_start_sample:frame_end_sample]
                if audio_segment.size == 0: continue 

                current_brightness = brightness_data[i]
                current_detail = detail_data[i]
                
                # Mappatura: Frequenza di Taglio -> Luminosità, Risonanza -> Dettaglio
                cutoff_freq = min_cutoff + current_brightness * (max_cutoff - min_cutoff)
                resonance_q = min_res + current_detail * (max_res - min_res) 

                nyquist = 0.5 * self.sample_rate
                normal_cutoff = cutoff_freq / nyquist
                normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999) 

                b, a = butter(filter_order, normal_cutoff, btype='lowpass', analog=False, output='ba')
                filtered_segment, zi = lfilter(b, a, audio_segment, zi=zi)
                
                processed_audio[frame_start_sample:frame_end_sample] = filtered_segment
                progress_bar_filter.progress((i + 1) / num_frames)
                status_text.text(f"🎶 Filtro Dinamico Frame {i + 1}/{num_frames} | Cutoff: {int(cutoff_freq)} Hz | Q: {resonance_q:.2f}")

        # --- Fase 2: Pitch Shifting e Time Stretching ---
        st.info("🔊 Applicazione Pitch Shifting e Time Stretching...")
        
        processed_audio = processed_audio.astype(np.float32)

        output_segments = []
        
        progress_bar_effects = st.progress(0, text="🔊 Applicazione Effetti Audio...")
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(processed_audio))
            
            audio_segment = processed_audio[frame_start_sample:frame_end_sample]
            if audio_segment.size == 0: continue

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]

            # Mappatura: Pitch Shift -> Dettaglio, Time Stretch -> Luminosità
            pitch_shift_semitones = min_pitch_shift_semitones + current_detail * (max_pitch_shift_semitones - min_pitch_shift_semitones)
            time_stretch_rate = min_time_stretch_rate + current_brightness * (max_time_stretch_rate - min_time_stretch_rate)
            time_stretch_rate = np.clip(time_stretch_rate, 0.1, 5.0) 

            pitched_segment = librosa.effects.pitch_shift(
                y=audio_segment, 
                sr=self.sample_rate, 
                n_steps=pitch_shift_semitones
            )
            
            stretched_segment = librosa.effects.time_stretch(y=pitched_segment, rate=time_stretch_rate)
            
            output_segments.append(stretched_segment)

            progress_bar_effects.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Effetti Audio Frame {i + 1}/{num_frames} | Pitch: {pitch_shift_semitones:.1f} semitoni | Stretch: {time_stretch_rate:.2f}")

        combined_audio = np.concatenate(output_segments)

        target_total_samples = int(num_frames * self.samples_per_frame)
        if len(combined_audio) != target_total_samples:
            st.info(f"🔄 Ricampionamento audio per adattarsi alla durata video (da {len(combined_audio)} a {target_total_samples} campioni)...")
            final_audio = librosa.resample(y=combined_audio, orig_sr=self.sample_rate, target_sr=self.sample_rate, res_type='kaiser_best', scale=False, fix=True, to_mono=True, axis=-1, length=target_total_samples)
        else:
            final_audio = combined_audio

        if np.max(np.abs(final_audio)) > 0:
            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.9 
            
        return final_audio.astype(np.float32)


def main():
    st.set_page_config(page_title="🎵 VideoSound Gen - Sperimentale", layout="centered")
    st.title("🎬 VideoSound Gen - Sperimentale")
    st.markdown("###### by Loop507") 
    st.markdown("### Genera musica sperimentale da un video muto")
    st.markdown("Carica un video e osserva come le sue proprietà visive creano un paesaggio sonoro dinamico.")

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

        with st.spinner("📊 Analisi frame video (luminosità, dettaglio, movimento) in corso..."):
            brightness_data, detail_data, movement_data, width, height, fps, video_duration = analyze_video_frames(video_input_path)
        
        if brightness_data is None: 
            return
        
        st.info(f"🎥 Durata video: {video_duration:.2f} secondi | Risoluzione Originale: {width}x{height} | FPS: {fps:.2f}")

        st.markdown("---")
        st.subheader("🎶 Configurazione Sintesi Audio Sperimentale")

        # Inizializza tutti i parametri a zero/default
        min_cutoff_user, max_cutoff_user = 0, 0
        min_resonance_user, max_resonance_user = 0, 0
        min_carrier_freq_user, max_carrier_freq_user = 0, 0
        min_modulator_freq_user, max_modulator_freq_user = 0, 0
        min_mod_index_user, max_mod_index_user = 0, 0
        
        # --- Sezione Sintesi Sottrattiva (sempre attiva come base) ---
        st.sidebar.header("Generazione Suono Base")
        with st.sidebar.expander("Sintesi Sottrattiva (Filtro Passa-Basso)", expanded=True): # Espandi di default
            st.markdown("**Controlli:**")
            st.markdown("- **Frequenza di Taglio:** controllata dalla **Luminosità** del video.")
            st.markdown("- **Risonanza:** controllata dal **Dettaglio/Contrasto** del video.")
            min_cutoff_user = st.slider("Min Frequenza Taglio (Hz)", 20, 5000, 100, key="sub_min_cutoff")
            max_cutoff_user = st.slider("Max Frequenza Taglio (Hz)", 1000, 20000, 8000, key="sub_max_cutoff")
            min_resonance_user = st.slider("Min Risonanza (Q)", 0.1, 5.0, 0.5, key="sub_min_res") 
            max_resonance_user = st.slider("Max Risonanza (Q)", 1.0, 30.0, 10.0, key="sub_max_res") 
        
        # --- Sezione Sintesi FM (attivabile con checkbox) ---
        st.sidebar.markdown("---")
        enable_fm_synthesis = st.sidebar.checkbox("🔊 Abilita Sintesi FM (Strato aggiuntivo)", value=False)
        
        if enable_fm_synthesis:
            with st.sidebar.expander("Sintesi FM Parametri", expanded=True): # Espandi di default se abilitato
                st.markdown("**Controlli:**")
                st.markdown("- **Frequenza Carrier:** controllata dalla **Luminosità** del video.")
                st.markdown("- **Frequenza Modulator:** controllata dal **Movimento** del video.")
                st.markdown("- **Indice di Modulazione:** controllato dal **Movimento** del video.")
                min_carrier_freq_user = st.slider("Min Frequenza Carrier (Hz)", 20, 1000, 100, key="fm_min_carrier")
                max_carrier_freq_user = st.slider("Max Frequenza Carrier (Hz)", 500, 5000, 1000, key="fm_max_carrier")
                min_modulator_freq_user = st.slider("Min Frequenza Modulator (Hz)", 1, 500, 5, key="fm_min_modulator")
                max_modulator_freq_user = st.slider("Max Frequenza Modulator (Hz)", 10, 2000, 100, key="fm_max_modulator")
                min_mod_index_user = st.slider("Min Indice di Modulazione", 0.1, 10.0, 0.5, 0.1, key="fm_min_index")
                max_mod_index_user = st.slider("Max Indice di Modulazione", 1.0, 50.0, 5.0, 0.1, key="fm_max_index")
        
        # --- Sezione Pitch/Time Stretching (sempre attiva) ---
        st.sidebar.markdown("---")
        with st.sidebar.expander("Pitch Shifting / Time Stretching", expanded=True): # Espandi di default
            st.markdown("**Controlli:**")
            st.markdown("- **Time Stretch Rate:** controllato dalla **Luminosità** del video.")
            st.markdown("- **Pitch Shift:** controllato dal **Dettaglio/Contrasto** del video.")
            min_pitch_shift_semitones = st.slider("Min Pitch Shift (semitoni)", -24.0, 24.0, -12.0, 0.5, key="pitch_min")
            max_pitch_shift_semitones = st.slider("Max Pitch Shift (semitoni)", -24.0, 24.0, 12.0, 0.5, key="pitch_max")
            min_time_stretch_rate = st.slider("Min Time Stretch Rate", 0.1, 2.0, 0.8, 0.1, key="stretch_min") 
            max_time_stretch_rate = st.slider("Max Time Stretch Rate", 0.5, 5.0, 1.5, 0.1, key="stretch_max") 


        st.markdown("---")
        st.subheader("⬇️ Opzioni di Download")
        
        output_resolution_choice = st.selectbox(
            "Seleziona la risoluzione di output del video:",
            list(FORMAT_RESOLUTIONS.keys())
        )
        
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
            
            # --- Generazione dell'audio base (Sintesi Sottrattiva) ---
            st.info("🎵 Generazione dell'onda base (Sintesi Sottrattiva)...")
            primary_audio_waveform = audio_gen.generate_subtractive_waveform(total_samples)
            
            # --- Aggiungi lo strato FM se abilitato ---
            if enable_fm_synthesis:
                fm_layer = audio_gen.generate_fm_layer(
                    total_samples,
                    brightness_data, 
                    detail_data,
                    movement_data, # Passa i dati di movimento
                    min_carrier_freq_user, max_carrier_freq_user,
                    min_modulator_freq_user, max_modulator_freq_user,
                    min_mod_index_user, max_mod_index_user
                )
                # Mescola lo strato FM con l'audio sottrattivo
                # Normalizziamo leggermente per evitare clipping eccessivo quando si combinano
                primary_audio_waveform = (primary_audio_waveform * 0.7 + fm_layer * 0.3) 
                st.success("✅ Strato FM combinato con l'audio base!")

            # --- Processamento degli effetti dinamici (filtro, pitch, time stretch) ---
            with st.spinner("🎧 Applicazione effetti dinamici all'audio generato..."):
                generated_audio = audio_gen.process_audio_segments(
                    primary_audio_waveform, 
                    brightness_data, 
                    detail_data,
                    min_cutoff=min_cutoff_user, 
                    max_cutoff=max_cutoff_user,
                    min_res=min_resonance_user, 
                    max_res=max_resonance_user,
                    min_pitch_shift_semitones=min_pitch_shift_semitones,
                    max_pitch_shift_semitones=max_pitch_shift_semitones,
                    min_time_stretch_rate=min_time_stretch_rate,
                    max_time_stretch_rate=max_time_stretch_rate,
                    apply_filter=True # Il filtro si applica sempre all'onda base sottrattiva
                )
            
            if generated_audio is None or generated_audio.size == 0:
                st.error("❌ Errore nel processamento degli effetti audio.")
                return

            try:
                sf.write(audio_output_path, generated_audio, AUDIO_SAMPLE_RATE)
                st.success(f"✅ Audio sperimentale processato e salvato in '{audio_output_path}'")
            except Exception as e:
                st.error(f"❌ Errore nel salvataggio dell'audio WAV: {str(e)}")
                return
            
            # Gestione del download in base alla scelta
            if download_option == "Solo Audio":
                with open(audio_output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica Solo Audio (WAV)", 
                        f,
                        file_name=f"videosound_generato_audio_{base_name_output}.wav",
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
                                    file_name=f"videosound_generato_{base_name_output}_{output_resolution_choice.replace(' ', '_')}.mp4",
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
                            file_name=f"videosound_generato_audio_{base_name_output}.wav",
                            mime="audio/wav"
                        )


if __name__ == "__main__":
    main()
