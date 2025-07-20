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

def analyze_video_frames(video_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Carica il video, estrae i frame e analizza luminosità, dettaglio/contrasto e movimento.
    Restituisce array di luminosità, dettaglio, movimento e variazione di movimento per frame, e info sul video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il file video.")
        return None, None, None, None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.error("❌ Impossibile leggere il framerate del video.")
        cap.release()
        return None, None, None, None, None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Usato CAP_PROP_HEIGHT invece di CAP_PROP_FRAME_HEIGHT per consistenza
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    if video_duration < MIN_DURATION:
        st.error(f"❌ Il video deve essere lungo almeno {MIN_DURATION} secondi. Durata attuale: {video_duration:.2f}s")
        cap.release()
        return None, None, None, None, None, None, None, None
    
    if video_duration > MAX_DURATION:
        st.warning(f"⚠️ Video troppo lungo ({video_duration:.1f}s). Verranno analizzati solo i primi {MAX_DURATION} secondi.")
        total_frames = int(MAX_DURATION * fps)
        video_duration = MAX_DURATION

    brightness_data = []
    detail_data = [] 
    movement_data = [] 
    variation_movement_data = [] # Nuovo array per la variazione del movimento
    
    prev_gray_frame = None 
    prev_movement = 0.0 # Per calcolare la variazione del movimento

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
        current_movement = 0.0
        if prev_gray_frame is not None:
            current_movement = np.sum(cv2.absdiff(gray_frame, prev_gray_frame)) / (gray_frame.size * 255.0)
            movement_data.append(current_movement)
        else:
            movement_data.append(0.0) # Primo frame non ha movimento rispetto a un precedente
        
        # 4. Variazione del Movimento: Differenza assoluta tra il movimento corrente e il precedente
        variation_movement = abs(current_movement - prev_movement)
        variation_movement_data.append(variation_movement)
        
        prev_gray_frame = gray_frame
        prev_movement = current_movement # Aggiorna il movimento precedente per il prossimo calcolo

        progress_bar.progress((i + 1) / total_frames)
        status_text.text(f"📊 Analisi Frame {i + 1}/{total_frames} | Lum: {brightness:.2f} | Det: {detail:.2f} | Mov: {movement_data[-1]:.2f} | VarMov: {variation_movement_data[-1]:.2f}")

    cap.release()
    st.success("✅ Analisi video completata!")
    
    # Assicurati che movement_data e variation_movement_data abbiano la stessa lunghezza degli altri array
    max_len = len(brightness_data)
    if len(movement_data) < max_len:
        while len(movement_data) < max_len:
            movement_data.append(movement_data[-1] if len(movement_data) > 0 else 0.0)
    if len(variation_movement_data) < max_len:
        while len(variation_movement_data) < max_len:
            variation_movement_data.append(variation_movement_data[-1] if len(variation_movement_data) > 0 else 0.0)


    return np.array(brightness_data), np.array(detail_data), np.array(movement_data), np.array(variation_movement_data), width, height, fps, video_duration

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
                             brightness_data: np.ndarray, movement_data: np.ndarray, 
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
            current_movement = movement_data[i] 

            # Mappatura dei parametri FM ai dati visivi
            # Carrier Frequency -> Luminosità
            carrier_freq = min_carrier_freq + current_brightness * (max_carrier_freq - min_carrier_freq)
            
            # Modulator Frequency -> Movimento
            modulator_freq = min_modulator_freq + current_movement * (max_modulator_freq - min_modulator_freq)
            
            # Modulation Index -> Movimento
            mod_index = min_mod_index + current_movement * (max_mod_index - min_mod_index) 

            t_segment = t_overall[frame_start_sample:frame_end_sample]
            
            modulator_wave = np.sin(2 * np.pi * modulator_freq * t_segment)
            fm_segment = np.sin(2 * np.pi * carrier_freq * t_segment + mod_index * modulator_wave)
            
            fm_audio_layer[frame_start_sample:frame_end_sample] = fm_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎵 Strato FM Frame {i + 1}/{num_frames} | Fc: {int(carrier_freq)} Hz | Fm: {int(modulator_freq)} Hz | I: {mod_index:.2f}")
        
        return fm_audio_layer

    def add_noise_layer(self, base_audio: np.ndarray, detail_data: np.ndarray, 
                        min_noise_amp: float, max_noise_amp: float) -> np.ndarray:
        """
        Aggiunge un layer di rumore bianco all'audio di base, modulato dal dettaglio/contrasto.
        """
        st.info("🔊 Aggiunta Strato Rumore (Noise)...")
        noise_layer = np.zeros_like(base_audio, dtype=np.float32)
        num_frames = len(detail_data)
        
        progress_bar = st.progress(0, text="🔊 Aggiunta Strato Rumore (Noise)...")
        status_text = st.empty()

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(base_audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_detail = detail_data[i]
            
            # Mappatura: Ampiezza Rumore -> Dettaglio/Contrasto
            noise_amplitude = min_noise_amp + current_detail * (max_noise_amp - min_noise_amp)
            
            segment_length = frame_end_sample - frame_start_sample
            noise_segment = np.random.randn(segment_length).astype(np.float32) * noise_amplitude
            
            noise_layer[frame_start_sample:frame_end_sample] = noise_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Rumore Frame {i + 1}/{num_frames} | Amp: {noise_amplitude:.3f}")

        # Mescola il rumore con l'audio base
        combined_audio = base_audio + noise_layer
        return combined_audio

    def apply_glitch_effect(self, base_audio: np.ndarray, variation_movement_data: np.ndarray,
                            glitch_threshold: float, glitch_duration_frames: int, glitch_intensity: float) -> np.ndarray:
        """
        Applica effetti glitch (ripetizioni/salti) all'audio basati sulla variazione del movimento.
        """
        st.info("🔊 Applicazione Effetti Glitch...")
        glitched_audio = base_audio.copy()
        num_frames = len(variation_movement_data)
        
        progress_bar = st.progress(0, text="🔊 Applicazione Effetti Glitch...")
        status_text = st.empty()

        for i in range(num_frames):
            current_var_movement = variation_movement_data[i]
            
            if current_var_movement > glitch_threshold:
                # Determina l'intensità del glitch in base alla variazione del movimento
                # Maggiore la variazione, maggiore la probabilità/intensità del glitch
                glitch_factor = (current_var_movement - glitch_threshold) / (np.max(variation_movement_data) - glitch_threshold + 1e-6)
                glitch_factor = np.clip(glitch_factor, 0, 1) * glitch_intensity
                
                if np.random.rand() < glitch_factor: # Probabilità di innescare il glitch
                    start_frame = i
                    end_frame = min(i + glitch_duration_frames, num_frames -1)
                    
                    if start_frame >= end_frame: continue

                    # Calcola il segmento audio da glitchare
                    glitch_start_sample = start_frame * self.samples_per_frame
                    glitch_end_sample = min(end_frame * self.samples_per_frame, len(glitched_audio))
                    
                    if glitch_start_sample >= glitch_end_sample: continue

                    segment_to_glitch = glitched_audio[glitch_start_sample:glitch_end_sample]
                    
                    # Esempio di glitch: ripetizione di un piccolo segmento
                    if segment_to_glitch.size > 0:
                        repeat_len = min(int(self.samples_per_frame * 0.1), segment_to_glitch.size) # Ripete il 10% del frame
                        if repeat_len > 0:
                            repeated_segment = np.tile(segment_to_glitch[:repeat_len], int(glitch_duration_frames * 0.5)) # Ripete per una frazione della durata
                            
                            # Sostituisci il segmento glitchato con la ripetizione (o un pezzo di essa)
                            replace_len = min(len(repeated_segment), glitch_end_sample - glitch_start_sample)
                            glitched_audio[glitch_start_sample : glitch_start_sample + replace_len] = repeated_segment[:replace_len]
                            
                            # Avanza l'indice per evitare glitch sovrapposti immediatamente
                            i += glitch_duration_frames 
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Glitch Frame {i + 1}/{num_frames} | VarMov: {current_var_movement:.2f}")
        
        return glitched_audio

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
        
        # --- Fase 1: Filtro Dinamico (per sintesi sottrattiva) ---
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
    st.markdown("### Genera musica sperimentale da un video") 
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

        with st.spinner("📊 Analisi frame video (luminosità, dettaglio, movimento, variazione movimento) in corso..."):
            brightness_data, detail_data, movement_data, variation_movement_data, width, height, fps, video_duration = analyze_video_frames(video_input_path)
        
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
        min_noise_amp_user, max_noise_amp_user = 0, 0
        glitch_threshold_user, glitch_duration_frames_user, glitch_intensity_user = 0, 0, 0

        # --- Sezione Sintesi Sottrattiva (sempre attiva come base) ---
        st.sidebar.header("Generazione Suono Base")
        with st.sidebar.expander("Sintesi Sottrattiva (Filtro Passa-Basso)", expanded=True): 
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
            with st.sidebar.expander("Sintesi FM Parametri", expanded=True): 
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
        
        # --- Sezione Noise (attivabile con checkbox) ---
        st.sidebar.markdown("---")
        enable_noise_effect = st.sidebar.checkbox("🔇 Abilita Rumore (Noise)", value=False)

        if enable_noise_effect:
            with st.sidebar.expander("Parametri Rumore (Noise)", expanded=True):
                st.markdown("**Controllo:**")
                st.markdown("- **Intensità (Volume):** controllata dal **Dettaglio/Contrasto** del video.")
                min_noise_amp_user = st.slider("Min Ampiezza Rumore", 0.0, 0.5, 0.01, 0.01, key="noise_min_amp")
                max_noise_amp_user = st.slider("Max Ampiezza Rumore", 0.1, 1.0, 0.2, 0.01, key="noise_max_amp")

        # --- Sezione Glitch (attivabile con checkbox) ---
        st.sidebar.markdown("---")
        enable_glitch_effect = st.sidebar.checkbox("👾 Abilita Glitch Audio", value=False)

        if enable_glitch_effect:
            with st.sidebar.expander("Parametri Glitch Audio", expanded=True):
                st.markdown("**Controllo:**")
                st.markdown("- **Frequenza/Intensità:** innescata dai **Picchi di Movimento** del video.")
                glitch_threshold_user = st.slider("Soglia Variazione Movimento per Glitch", 0.0, 0.5, 0.05, 0.01, key="glitch_threshold")
                glitch_duration_frames_user = st.slider("Durata Glitch (frames)", 1, 10, 2, key="glitch_duration_frames")
                glitch_intensity_user = st.slider("Intensità Glitch", 0.1, 2.0, 1.0, 0.1, key="glitch_intensity")

        # --- Sezione Pitch/Time Stretching (sempre attiva) ---
        st.sidebar.markdown("---")
        with st.sidebar.expander("Pitch Shifting / Time Stretching", expanded=True): 
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
                    movement_data, 
                    min_carrier_freq_user, max_carrier_freq_user,
                    min_modulator_freq_user, max_modulator_freq_user,
                    min_mod_index_user, max_mod_index_user
                )
                # Mescola lo strato FM con l'audio sottrattivo
                primary_audio_waveform = (primary_audio_waveform * 0.7 + fm_layer * 0.3) 
                st.success("✅ Strato FM combinato con l'audio base!")

            # --- Aggiungi il layer di Rumore se abilitato ---
            if enable_noise_effect:
                primary_audio_waveform = audio_gen.add_noise_layer(
                    primary_audio_waveform, 
                    detail_data,
                    min_noise_amp_user, 
                    max_noise_amp_user
                )
                st.success("✅ Strato Rumore aggiunto!")

            # --- Applica effetti Glitch se abilitati ---
            if enable_glitch_effect:
                primary_audio_waveform = audio_gen.apply_glitch_effect(
                    primary_audio_waveform, 
                    variation_movement_data,
                    glitch_threshold_user, 
                    glitch_duration_frames_user,
                    glitch_intensity_user
                )
                st.success("✅ Effetti Glitch applicati!")

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
                    apply_filter=True 
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
