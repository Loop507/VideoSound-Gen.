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
        self.samples_per_frame = self.sample_rate // self.fps 

    def generate_subtractive_waveform(self, duration_samples: int) -> np.ndarray:
        """Genera un'onda a dente di sega (sawtooth) come base per la sintesi sottrattiva."""
        base_freq = 220.0 
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)
        waveform = 2 * (t * base_freq - np.floor(t * base_freq + 0.5))
        return waveform.astype(np.float32)

    def generate_fm_waveform(self, duration_samples: int,
                             brightness_data: np.ndarray, detail_data: np.ndarray,
                             min_carrier_freq: float, max_carrier_freq: float,
                             min_modulator_freq: float, max_modulator_freq: float,
                             min_mod_index: float, max_mod_index: float) -> np.ndarray:
        """
        Genera un'onda FM modulata dai dati visivi.
        luminosità -> carrier freq
        dettaglio -> modulator freq e/o modulation index
        """
        st.info("🎵 Inizializzazione della generazione audio sperimentale (Sintesi FM)...")
        generated_audio = np.zeros(duration_samples, dtype=np.float32)
        
        t_overall = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)

        num_frames = len(brightness_data)
        
        progress_bar = st.progress(0, text="🎵 Generazione Sintesi FM...")
        status_text = st.empty()

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, duration_samples)
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]

            # Mappatura dei parametri FM ai dati visivi
            carrier_freq = min_carrier_freq + current_brightness * (max_carrier_freq - min_carrier_freq)
            modulator_freq = min_modulator_freq + current_detail * (max_modulator_freq - min_modulator_freq)
            mod_index = min_mod_index + current_detail * (max_mod_index - min_mod_index) # Oppure usa brightness per l'indice

            # Genera il segmento audio FM per il frame corrente
            t_segment = t_overall[frame_start_sample:frame_end_sample]
            
            # Calcolo dell'onda modulante (sinusoidale)
            modulator_wave = np.sin(2 * np.pi * modulator_freq * t_segment)
            
            # Calcolo dell'onda portante (sinusoidale) modulata
            # FM Synthesis: Carrier_Amplitude * sin(2*pi*Carrier_Freq*t + Modulation_Index * sin(2*pi*Modulator_Freq*t))
            fm_segment = np.sin(2 * np.pi * carrier_freq * t_segment + mod_index * modulator_wave)
            
            generated_audio[frame_start_sample:frame_end_sample] = fm_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎵 Sintesi FM Frame {i + 1}/{num_frames} | Fc: {int(carrier_freq)} Hz | Fm: {int(modulator_freq)} Hz | I: {mod_index:.2f}")
        
        return generated_audio

    def process_audio_segments(self, base_audio: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray, 
                             min_cutoff: float, max_cutoff: float, min_res: float, max_res: float,
                             min_pitch_shift_semitones: float, max_pitch_shift_semitones: float,
                             min_time_stretch_rate: float, max_time_stretch_rate: float,
                             apply_filter: bool = True) -> np.ndarray: # Aggiungi un flag per il filtro
        """
        Applica (opzionalmente) un filtro passa-basso dinamico, pitch shifting e time stretching 
        all'audio di base, modulato dai dati visivi.
        """
        st.info("🔊 Applicazione Effetti Audio dinamici (Filtro, Pitch, Time Stretch)...")
        
        # Copia dell'audio di base per le modifiche
        processed_audio = base_audio.copy()
        
        filter_order = 4
        
        num_frames = len(brightness_data)
        
        status_text = st.empty()
        
        # --- Fase 1: Filtro Dinamico (se abilitato) ---
        if apply_filter:
            progress_bar_filter = st.progress(0, text="🎶 Applicazione Filtro Dinamico...")
            # Inizializza lo stato del filtro per una transizione più fluida
            zi = np.zeros(filter_order) 
            for i in range(num_frames):
                frame_start_sample = i * self.samples_per_frame
                frame_end_sample = min((i + 1) * self.samples_per_frame, len(processed_audio))
                
                audio_segment = processed_audio[frame_start_sample:frame_end_sample]
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
                
                processed_audio[frame_start_sample:frame_end_sample] = filtered_segment
                progress_bar_filter.progress((i + 1) / num_frames)
                status_text.text(f"🎶 Filtro Dinamico Frame {i + 1}/{num_frames} | Cutoff: {int(cutoff_freq)} Hz | Q: {resonance_q:.2f}")

        # --- Fase 2: Pitch Shifting e Time Stretching ---
        st.info("🔊 Applicazione Pitch Shifting e Time Stretching...")
        
        # Assicurati che l'audio sia float32 per librosa
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

            pitch_shift_semitones = min_pitch_shift_semitones + current_detail * (max_pitch_shift_semitones - min_pitch_shift_semitones)
            time_stretch_rate = min_time_stretch_rate + current_brightness * (max_time_stretch_rate - min_time_stretch_rate)
            time_stretch_rate = np.clip(time_stretch_rate, 0.1, 5.0) 

            # Applica Pitch Shifting
            pitched_segment = librosa.effects.pitch_shift(
                y=audio_segment, 
                sr=self.sample_rate, 
                n_steps=pitch_shift_semitones
            )
            
            # Applica Time Stretching
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

        with st.spinner("📊 Analisi frame video (luminosità, dettaglio) in corso..."):
            brightness_data, detail_data, width, height, fps, video_duration = analyze_video_frames(video_input_path)
        
        if brightness_data is None: 
            return
        
        st.info(f"🎥 Durata video: {video_duration:.2f} secondi | Risoluzione Originale: {width}x{height} | FPS: {fps:.2f}")

        st.markdown("---")
        st.subheader("🎶 Configurazione Sintesi Audio Sperimentale")

        # --- Menu a tendina per la scelta del Tipo di Sintesi ---
        synthesis_type = st.selectbox(
            "Seleziona il tipo di sintesi:",
            ("Sintesi Sottrattiva", "Sintesi FM")
        )

        # Inizializza tutti i parametri a zero/default per evitare errori
        min_cutoff_user, max_cutoff_user = 0, 0
        min_resonance_user, max_resonance_user = 0, 0
        min_carrier_freq_user, max_carrier_freq_user = 0, 0
        min_modulator_freq_user, max_modulator_freq_user = 0, 0
        min_mod_index_user, max_mod_index_user = 0, 0
        apply_filter = False # Default a False, abilitato solo per sottrattiva

        # Parametri Condizionali per la Sintesi Sottrattiva
        if synthesis_type == "Sintesi Sottrattiva":
            st.sidebar.header("Parametri Sintesi Sottrattiva")
            min_cutoff_user = st.sidebar.slider("Min Frequenza Taglio (Hz)", 20, 5000, 100)
            max_cutoff_user = st.sidebar.slider("Max Frequenza Taglio (Hz)", 1000, 20000, 8000)
            min_resonance_user = st.sidebar.slider("Min Risonanza (Q)", 0.1, 5.0, 0.5) 
            max_resonance_user = st.sidebar.slider("Max Risonanza (Q)", 1.0, 30.0, 10.0) 
            apply_filter = True # Il filtro è intrinseco alla sintesi sottrattiva
        elif synthesis_type == "Sintesi FM": # Sintesi FM
            st.sidebar.header("Parametri Sintesi FM")
            st.sidebar.markdown("*(Luminosità controlla Freq Carrier, Dettaglio controlla Freq Modulator e Indice)*")
            min_carrier_freq_user = st.sidebar.slider("Min Frequenza Carrier (Hz)", 20, 1000, 100)
            max_carrier_freq_user = st.sidebar.slider("Max Frequenza Carrier (Hz)", 500, 5000, 1000)
            min_modulator_freq_user = st.sidebar.slider("Min Frequenza Modulator (Hz)", 1, 500, 5)
            max_modulator_freq_user = st.sidebar.slider("Max Frequenza Modulator (Hz)", 10, 2000, 100)
            min_mod_index_user = st.sidebar.slider("Min Indice di Modulazione", 0.1, 10.0, 0.5, 0.1)
            max_mod_index_user = st.sidebar.slider("Max Indice di Modulazione", 1.0, 50.0, 5.0, 0.1)
            # apply_filter rimane False, poiché non si applica un filtro passa-basso a questo tipo di sintesi

        # Parametri Pitch/Time Stretching (sempre visibili)
        st.sidebar.header("Parametri Pitch/Time Stretching")
        st.sidebar.markdown("*(Luminosità controlla il Time Stretching, Dettaglio il Pitch Shifting)*")
        min_pitch_shift_semitones = st.sidebar.slider("Min Pitch Shift (semitoni)", -24.0, 24.0, -12.0, 0.5)
        max_pitch_shift_semitones = st.sidebar.slider("Max Pitch Shift (semitoni)", -24.0, 24.0, 12.0, 0.5)
        min_time_stretch_rate = st.sidebar.slider("Min Time Stretch Rate", 0.1, 2.0, 0.8, 0.1) 
        max_time_stretch_rate = st.sidebar.slider("Max Time Stretch Rate", 0.5, 5.0, 1.5, 0.1) 


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
            
            # --- Generazione dell'audio base in base al tipo di sintesi scelto ---
            base_audio_waveform = None
            if synthesis_type == "Sintesi Sottrattiva":
                base_audio_waveform = audio_gen.generate_subtractive_waveform(total_samples)
                st.success("✅ Onda base per Sintesi Sottrattiva generata!")
            elif synthesis_type == "Sintesi FM":
                base_audio_waveform = audio_gen.generate_fm_waveform(
                    total_samples,
                    brightness_data, 
                    detail_data,
                    min_carrier_freq_user, max_carrier_freq_user,
                    min_modulator_freq_user, max_modulator_freq_user,
                    min_mod_index_user, max_mod_index_user
                )
                st.success("✅ Onda base per Sintesi FM generata!")

            if base_audio_waveform is None or base_audio_waveform.size == 0:
                st.error("❌ Errore nella generazione dell'onda audio base.")
                return

            # --- Processamento degli effetti dinamici (filtro, pitch, time stretch) ---
            with st.spinner("🎧 Applicazione effetti dinamici all'audio generato..."):
                generated_audio = audio_gen.process_audio_segments(
                    base_audio_waveform, 
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
                    apply_filter=apply_filter 
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
