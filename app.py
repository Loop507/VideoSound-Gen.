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

def analyze_video_frames(video_path: str, progress_bar, status_text) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Carica il video, estrae i frame e analizza luminosità, dettaglio/contrasto, movimento e centro di massa orizzontale.
    Restituisce array di luminosità, dettaglio, movimento, variazione di movimento e centro di massa, e info sul video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Impossibile aprire il file video.")
        return None, None, None, None, None, None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.error("❌ Impossibile leggere il framerate del video.")
        cap.release()
        return None, None, None, None, None, None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    if video_duration < MIN_DURATION:
        st.error(f"❌ Il video deve essere lungo almeno {MIN_DURATION} secondi. Durata attuale: {video_duration:.2f}s")
        cap.release()
        return None, None, None, None, None, None, None, None, None
    
    if video_duration > MAX_DURATION:
        st.warning(f"⚠️ Video troppo lungo ({video_duration:.1f}s). Verranno analizzati solo i primi {MAX_DURATION} secondi.")
        total_frames = int(MAX_DURATION * fps)
        video_duration = MAX_DURATION

    brightness_data = []
    detail_data = [] 
    movement_data = [] 
    variation_movement_data = []
    horizontal_center_data = [] # Nuovo array per il centro di massa orizzontale
    
    prev_gray_frame = None 
    prev_movement = 0.0

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
            movement_data.append(0.0) 
        
        # 4. Variazione del Movimento: Differenza assoluta tra il movimento corrente e il precedente
        variation_movement = abs(current_movement - prev_movement)
        variation_movement_data.append(variation_movement)
        
        # 5. Centro di massa orizzontale (per la panoramica stereo)
        # Consideriamo i pixel "interessanti" (non troppo scuri o troppo chiari)
        # o semplicemente il centro di massa della luminosità
        if np.sum(gray_frame) > 0:
            moments = cv2.moments(gray_frame)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                normalized_cx = cx / width # Normalizza tra 0 (sinistra) e 1 (destra)
                horizontal_center_data.append(normalized_cx)
            else:
                horizontal_center_data.append(0.5) # Centro se nessun momento
        else:
            horizontal_center_data.append(0.5) # Centro se frame nero

        prev_gray_frame = gray_frame
        prev_movement = current_movement

        progress_bar.progress((i + 1) / total_frames)
        status_text.text(f"📊 Analisi Frame {i + 1}/{total_frames} | Lum: {brightness:.2f} | Det: {detail:.2f} | Mov: {movement_data[-1]:.2f} | VarMov: {variation_movement_data[-1]:.2f} | Pan: {horizontal_center_data[-1]:.2f}")

    cap.release()
    st.success("✅ Analisi video completata!")
    
    # Assicurati che tutti gli array abbiano la stessa lunghezza
    max_len = len(brightness_data)
    for arr in [movement_data, variation_movement_data, horizontal_center_data]:
        if len(arr) < max_len:
            while len(arr) < max_len:
                arr.append(arr[-1] if len(arr) > 0 else 0.5 if arr is horizontal_center_data else 0.0) # 0.5 per panning al centro

    return np.array(brightness_data), np.array(detail_data), np.array(movement_data), np.array(variation_movement_data), np.array(horizontal_center_data), width, height, fps, video_duration

class AudioGenerator:
    def __init__(self, sample_rate: int, fps: int):
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_per_frame = self.sample_rate // self.fps 

    def generate_subtractive_waveform(self, duration_samples: int, waveform_type: str = "sawtooth") -> np.ndarray:
        """Genera un'onda di base per la sintesi sottrattiva con diverse forme d'onda."""
        base_freq = 220.0 
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)
        
        if waveform_type == "sine":
            waveform = np.sin(2 * np.pi * base_freq * t)
        elif waveform_type == "square":
            waveform = np.sign(np.sin(2 * np.pi * base_freq * t))
        elif waveform_type == "triangle":
            waveform = 2 * np.abs(2 * (t * base_freq - np.floor(t * base_freq + 0.5))) - 1
        else: # Default is sawtooth
            waveform = 2 * (t * base_freq - np.floor(t * base_freq + 0.5))
            
        return waveform.astype(np.float32)

    def generate_fm_layer(self, duration_samples: int,
                             brightness_data: np.ndarray, movement_data: np.ndarray, 
                             min_carrier_freq: float, max_carrier_freq: float,
                             min_modulator_freq: float, max_modulator_freq: float,
                             min_mod_index: float, max_mod_index: float, progress_bar, status_text) -> np.ndarray:
        """
        Genera un'onda FM modulata dai dati visivi (luminosità per carrier, movimento per modulator e index) come strato aggiuntivo.
        """
        st.info("🎵 Generazione Strato FM...")
        fm_audio_layer = np.zeros(duration_samples, dtype=np.float32)
        
        t_overall = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)

        num_frames = len(brightness_data)
        
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

    def generate_granular_layer(self, duration_samples: int,
                                brightness_data: np.ndarray, detail_data: np.ndarray, movement_data: np.ndarray,
                                min_grain_freq: float, max_grain_freq: float,
                                min_grain_density: float, max_grain_density: float,
                                min_grain_duration: float, max_grain_duration: float, progress_bar, status_text
                                ) -> np.ndarray:
        """
        Genera un layer di sintesi granulare, modulato dai dati visivi.
        
        Parametri:
        - min_grain_freq, max_grain_freq: range di frequenza dei grani (Pito)
        - min_grain_density, max_grain_density: range di densità dei grani (Texture) - grani per secondo
        - min_grain_duration, max_grain_duration: range di durata dei grani (Drone) - in secondi
        """
        st.info("🎵 Generazione Strato Granulare...")
        granular_layer = np.zeros(duration_samples, dtype=np.float32)
        
        num_frames = len(brightness_data)
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, duration_samples)
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]
            current_movement = movement_data[i]

            # Mappatura dei parametri granulari ai dati visivi
            # Pitch (Pito) -> Luminosità
            grain_freq = min_grain_freq + current_brightness * (max_grain_freq - min_grain_freq)
            grain_freq = np.clip(grain_freq, 20, self.sample_rate / 2 - 100) # Assicura che la frequenza sia valida

            # Densità (Texture) -> Movimento (più movimento = più grani)
            grains_per_second = min_grain_density + current_movement * (max_grain_density - min_grain_density)
            
            # Durata Grano (Drone) -> Dettaglio/Contrasto (meno dettaglio = grani più lunghi per drone)
            # Invertiamo il dettaglio per il drone: basso dettaglio -> alta durata grano
            grain_duration_sec = max_grain_duration - current_detail * (max_grain_duration - min_grain_duration)
            grain_duration_sec = np.clip(grain_duration_sec, 0.005, 0.5) # Durata minima 5ms, max 500ms
            
            
            segment_duration_sec = (frame_end_sample - frame_start_sample) / self.sample_rate
            num_grains_in_segment = int(grains_per_second * segment_duration_sec)

            if num_grains_in_segment == 0:
                progress_bar.progress((i + 1) / num_frames)
                status_text.text(f"🎵 Granulare Frame {i + 1}/{num_frames} | Freq: {int(grain_freq)} Hz | Dens: {int(grains_per_second)}/s | Dur: {grain_duration_sec:.3f} s")
                continue

            grain_duration_samples = int(grain_duration_sec * self.sample_rate)
            
            # Creazione e sovrapposizione dei grani
            for _ in range(num_grains_in_segment):
                # Posizione casuale all'interno del segmento corrente
                grain_start_sample_in_segment = np.random.randint(0, max(1, (frame_end_sample - frame_start_sample) - grain_duration_samples))
                actual_grain_start = frame_start_sample + grain_start_sample_in_segment
                actual_grain_end = min(actual_grain_start + grain_duration_samples, duration_samples)
                
                if actual_grain_start >= actual_grain_end: continue

                t_grain = np.linspace(0, (actual_grain_end - actual_grain_start) / self.sample_rate, (actual_grain_end - actual_grain_start), endpoint=False)
                
                # Forma d'onda del grano (es. sinusoidale con inviluppo Hanning per evitare click)
                grain_waveform = np.sin(2 * np.pi * grain_freq * t_grain)
                
                # Applica inviluppo (Hanning window)
                window = np.hanning(len(grain_waveform))
                grain_waveform *= window
                
                # Normalizza per evitare clipping quando si sommano molti grani
                grain_waveform *= 0.1 

                granular_layer[actual_grain_start:actual_grain_end] += grain_waveform
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎵 Granulare Frame {i + 1}/{num_frames} | Freq: {int(grain_freq)} Hz | Dens: {int(grains_per_second)}/s | Dur: {grain_duration_sec:.3f} s")
        
        return granular_layer

    def add_noise_layer(self, base_audio: np.ndarray, detail_data: np.ndarray, 
                        min_noise_amp: float, max_noise_amp: float, progress_bar, status_text) -> np.ndarray:
        """
        Aggiunge un layer di rumore bianco all'audio di base, modulato dal dettaglio/contrasto.
        """
        st.info("🔊 Aggiunta Strato Rumore (Noise)...")
        noise_layer = np.zeros_like(base_audio, dtype=np.float32)
        num_frames = len(detail_data)
        
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
                            glitch_threshold: float, glitch_duration_frames: int, glitch_intensity: float, progress_bar, status_text) -> np.ndarray:
        """
        Applica effetti glitch (ripetizioni/salti) all'audio basati sulla variazione del movimento.
        """
        st.info("🔊 Applicazione Effetti Glitch...")
        glitched_audio = base_audio.copy()
        num_frames = len(variation_movement_data)
        
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

    def apply_delay_effect(self, audio: np.ndarray, 
                           movement_data: np.ndarray, detail_data: np.ndarray,
                           max_delay_time: float, max_feedback: float, progress_bar, status_text) -> np.ndarray:
        """
        Applica un effetto delay (eco) dinamico.
        - delay_time: controllato dal movimento (più movimento = delay più corto)
        - feedback: controllato dal dettaglio (più dettaglio = più ripetizioni)
        """
        st.info("🔊 Applicazione Effetto Delay...")
        
        num_frames = len(movement_data)
        delay_audio = audio.copy()
        
        # Buffer per il delay
        max_delay_samples = int(max_delay_time * self.sample_rate)
        delay_buffer = np.zeros(max_delay_samples, dtype=np.float32)
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_movement = movement_data[i]
            current_detail = detail_data[i]

            # Mappatura: Delay Time -> Movimento (inverso: più movimento -> meno delay)
            delay_time_sec = max_delay_time - current_movement * max_delay_time * 0.9 # Da max a 0.1*max
            delay_time_sec = np.clip(delay_time_sec, 0.05, max_delay_time) # Min 50ms delay
            delay_samples = int(delay_time_sec * self.sample_rate)
            delay_samples = np.clip(delay_samples, 1, max_delay_samples - 1) # Assicura che sia nell'intervallo

            # Mappatura: Feedback -> Dettaglio
            feedback_gain = max_feedback * current_detail 
            feedback_gain = np.clip(feedback_gain, 0.0, 0.95) # Impedisce feedback infinito

            segment = audio[frame_start_sample:frame_end_sample]
            output_segment = np.zeros_like(segment)

            for j in range(len(segment)):
                input_sample = segment[j]
                
                # Leggi dal buffer con delay
                # Ensure index is within bounds of delay_buffer
                delayed_sample_index = (delay_buffer.size - delay_samples + (frame_start_sample + j)) % delay_buffer.size
                delayed_sample = delay_buffer[delayed_sample_index]
                
                # Aggiungi all'output
                output_segment[j] = input_sample + delayed_sample
                
                # Scrivi nel buffer con feedback
                delay_buffer[ (frame_start_sample + j) % delay_buffer.size ] = input_sample + delayed_sample * feedback_gain
            
            delay_audio[frame_start_sample:frame_end_sample] = output_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Delay Frame {i + 1}/{num_frames} | Time: {delay_time_sec:.2f}s | Feedback: {feedback_gain:.2f}")

        return delay_audio
    
    def apply_reverb_effect(self, audio: np.ndarray, 
                            detail_data: np.ndarray, brightness_data: np.ndarray,
                            max_decay_time: float, max_wet_mix: float, progress_bar, status_text) -> np.ndarray:
        """
        Applica un effetto riverbero dinamico.
        - decay_time: controllato dal dettaglio (meno dettaglio = decadimento più lungo)
        - wet_mix: controllato dalla luminosità
        """
        st.info("🔊 Applicazione Effetto Riverbero...")
        
        num_frames = len(detail_data)
        
        # Inizializza un array per l'output riverberato
        output_reverb = np.zeros_like(audio)

        # Parametri base per una simulazione semplificata di riverbero
        # Un vero riverbero richiede algoritmi complessi (es. Schroeder, FDN)
        # Qui usiamo un approccio basato su più delay che decadono per simulare una coda di riverbero.
        # Questa è una semplificazione per scopi dimostrativi.
        num_delay_lines = 4 # Numero di linee di ritardo per simulare riflessioni multiple
        base_delay_ms = 50 # Millisecondi di ritardo base per ogni linea
        
        # Buffer per le linee di ritardo (una per linea)
        delay_line_buffers = [np.zeros(int((base_delay_ms / 1000) * self.sample_rate), dtype=np.float32) for _ in range(num_delay_lines)]
        delay_line_indices = [0] * num_delay_lines # Indici correnti per i buffer circolari
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_detail = detail_data[i]
            current_brightness = brightness_data[i]

            # Mappatura: Decay Time -> Dettaglio (inverso: meno dettaglio = più decadimento)
            decay_time_sec = max_decay_time - current_detail * max_decay_time * 0.9 # Da max a 0.1*max
            decay_time_sec = np.clip(decay_time_sec, 0.1, max_decay_time) # Min 100ms
            
            # Mappatura: Wet Mix -> Luminosità
            wet_mix_gain = max_wet_mix * current_brightness
            wet_mix_gain = np.clip(wet_mix_gain, 0.0, 1.0) # Da 0 (dry) a 1 (wet)

            # Il gain per ogni eco che si alimenta nel riverbero
            feedback_gain_reverb = np.exp(np.log(0.001) / (decay_time_sec * self.sample_rate)) # Calcola gain per decadimento

            segment_length = frame_end_sample - frame_start_sample
            if segment_length <= 0: continue

            audio_segment = audio[frame_start_sample:frame_end_sample]
            
            reverb_output_segment = np.zeros_like(audio_segment)

            for k in range(segment_length):
                dry_sample = audio_segment[k]
                
                # Calcola il segnale "wet" dalle linee di ritardo
                wet_signal = 0.0
                for line_idx in range(num_delay_lines):
                    current_buffer = delay_line_buffers[line_idx]
                    current_idx = delay_line_indices[line_idx]
                    
                    delayed_sample = current_buffer[current_idx]
                    wet_signal += delayed_sample
                    
                    # Aggiorna il buffer: input + feedback del proprio delay line (e potenzialmente cross-feedback)
                    # Semplificazione: feed-forward del dry_sample nel buffer
                    current_buffer[current_idx] = dry_sample + delayed_sample * feedback_gain_reverb
                    
                    delay_line_indices[line_idx] = (current_idx + 1) % len(current_buffer)
                
                # Miscela dry e wet
                reverb_output_segment[k] = dry_sample * (1 - wet_mix_gain) + wet_signal * wet_mix_gain / num_delay_lines # Normalizza wet signal
            
            output_reverb[frame_start_sample:frame_end_sample] = reverb_output_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Reverb Frame {i + 1}/{num_frames} | Decay: {decay_time_sec:.2f}s | Wet: {wet_mix_gain:.2f}")

        return output_reverb


    def apply_dynamic_filter(self, audio: np.ndarray, 
                             brightness_data: np.ndarray, detail_data: np.ndarray, 
                             min_cutoff: float, max_cutoff: float, min_res: float, max_res: float,
                             progress_bar, status_text, filter_type: str = "lowpass") -> np.ndarray:
        """
        Applica un filtro dinamico (Passa-Basso, Passa-Banda, Passa-Alto) all'audio.
        """
        st.info(f"🎶 Applicazione Filtro Dinamico ({filter_type})...")
        filtered_audio = audio.copy()
        
        filter_order = 4 # Ordine del filtro (bilanciamento tra steepness e performance)
        num_frames = len(brightness_data)
        
        zi_channels = [np.zeros(filter_order) for _ in range(audio.shape[1] if audio.ndim > 1 else 1)] # Per gestire stereo

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(filtered_audio))
            
            audio_segment = filtered_audio[frame_start_sample:frame_end_sample]
            if audio_segment.size == 0: continue 

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]
            
            # Mappatura: Frequenza di Taglio -> Luminosità, Risonanza -> Dettaglio
            cutoff_freq = min_cutoff + current_brightness * (max_cutoff - min_cutoff)
            resonance_q = min_res + current_detail * (max_res - min_res) 

            nyquist = 0.5 * self.sample_rate
            normal_cutoff = cutoff_freq / nyquist
            normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999) 

            # Calcola i coefficienti del filtro in base al tipo selezionato
            if filter_type == "lowpass":
                b, a = butter(filter_order, normal_cutoff, btype='lowpass', analog=False, output='ba')
            elif filter_type == "highpass":
                b, a = butter(filter_order, normal_cutoff, btype='highpass', analog=False, output='ba')
            elif filter_type == "bandpass":
                # Per passa-banda, normal_cutoff può essere una tupla (low, high)
                # Semplifichiamo usando la cutoff come centro e una larghezza di banda fissa
                bandwidth = 0.1 # Esempio: 10% della frequenza di taglio
                low_cut = np.clip(normal_cutoff - bandwidth/2, 0.001, 0.999)
                high_cut = np.clip(normal_cutoff + bandwidth/2, 0.001, 0.999)
                b, a = butter(filter_order, [low_cut, high_cut], btype='bandpass', analog=False, output='ba')
            else: # Fallback to lowpass
                b, a = butter(filter_order, normal_cutoff, btype='lowpass', analog=False, output='ba')

            if audio.ndim > 1: # Stereo
                for channel_idx in range(audio.shape[1]):
                    filtered_segment_channel, zi_channels[channel_idx] = lfilter(b, a, audio_segment[:, channel_idx], zi=zi_channels[channel_idx])
                    filtered_audio[frame_start_sample:frame_end_sample, channel_idx] = filtered_segment_channel
            else: # Mono
                filtered_segment, zi_channels[0] = lfilter(b, a, audio_segment, zi=zi_channels[0])
                filtered_audio[frame_start_sample:frame_end_sample] = filtered_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎶 Filtro Dinamico Frame {i + 1}/{num_frames} | Cutoff: {int(cutoff_freq)} Hz | Q: {resonance_q:.2f}")

        return filtered_audio.astype(np.float32)

    def apply_pitch_time_stretch(self, base_audio: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray,
                                 min_pitch_shift_semitones: float, max_pitch_shift_semitones: float,
                                 min_time_stretch_rate: float, max_time_stretch_rate: float, progress_bar, status_text) -> np.ndarray:
        """
        Applica pitch shifting e time stretching dinamici all'audio.
        """
        st.info("🔊 Applicazione Pitch Shifting e Time Stretching...")
        
        # Converte a mono se stereo per librosa.effects.pitch_shift/time_stretch
        if base_audio.ndim > 1:
            audio_mono = librosa.to_mono(base_audio.T)
        else:
            audio_mono = base_audio.copy()

        output_segments = []
        num_frames = len(brightness_data)
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio_mono))
            
            audio_segment = audio_mono[frame_start_sample:frame_end_sample]
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

            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Effetti Audio Frame {i + 1}/{num_frames} | Pitch: {pitch_shift_semitones:.1f} semitoni | Stretch: {time_stretch_rate:.2f}")

        combined_audio = np.concatenate(output_segments)

        target_total_samples = int(num_frames * self.samples_per_frame)
        if len(combined_audio) != target_total_samples:
            st.info(f"🔄 Ricampionamento audio per adattarsi alla durata video (da {len(combined_audio)} a {target_total_samples} campioni)...")
            final_audio = librosa.resample(y=combined_audio, orig_sr=self.sample_rate, target_sr=self.sample_rate, res_type='kaiser_best', scale=False, fix=True, to_mono=True, axis=-1, length=target_total_samples)
        else:
            final_audio = combined_audio
            
        return final_audio.astype(np.float32)

    def apply_modulation_effect(self, audio: np.ndarray, variation_movement_data: np.ndarray, detail_data: np.ndarray,
                                progress_bar, status_text, effect_type: str = "none", intensity: float = 0.5, rate: float = 0.1) -> np.ndarray:
        """
        Applica effetti di modulazione (Chorus, Flanger, Phaser) all'audio.
        Questi sono placeholder e necessitano di un'implementazione più robusta con librerie dedicate
        o algoritmi DSP manuali.
        """
        if effect_type == "none":
            return audio.copy()
            
        st.info(f"🔊 Applicazione Effetto {effect_type.capitalize()}...")
        modulated_audio = audio.copy()
        
        num_frames = len(variation_movement_data)
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(modulated_audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_var_movement = variation_movement_data[i]
            current_detail = detail_data[i]

            # Esempio di modulazione dei parametri dell'effetto
            # Questo è molto semplificato, un'implementazione reale sarebbe più complessa.
            # L'intensità dell'effetto può essere modulata dalla variazione del movimento
            # La velocità di modulazione (rate) può essere modulata dal dettaglio
            current_intensity = intensity * current_var_movement * 2 # Aumenta intensità con movimento
            current_rate = rate + current_detail * 0.5 # Aumenta rate con dettaglio

            segment = modulated_audio[frame_start_sample:frame_end_sample]

            # Placeholder per l'applicazione dell'effetto reale
            # In un'applicazione reale, si userebbe una libreria come PyDub, Pydubfx o implementazioni DSP
            # Per ora, simuliamo un leggero "movimento" aggiungendo un piccolo offset di pitch/delay
            # che cambia dinamicamente. NON è un vero chorus/flanger/phaser.

            if effect_type == "chorus":
                # Simula un leggero detune e delay variabile
                delay_amount = int(5 * current_intensity * self.sample_rate / 1000) # Max 5ms delay
                if delay_amount > 0:
                    delayed_segment = np.roll(segment, delay_amount)
                    modulated_audio[frame_start_sample:frame_end_sample] = segment + delayed_segment * 0.2 * np.sin(2 * np.pi * current_rate * (np.arange(len(segment)) / self.sample_rate))
                else:
                    modulated_audio[frame_start_sample:frame_end_sample] = segment
            elif effect_type == "flanger":
                # Simula un feedback delay variabile
                delay_amount = int(20 * current_intensity * self.sample_rate / 1000) # Max 20ms delay
                if delay_amount > 0:
                    delayed_segment = np.roll(segment, delay_amount)
                    modulated_audio[frame_start_sample:frame_end_sample] = segment + delayed_segment * 0.5 * np.cos(2 * np.pi * current_rate * (np.arange(len(segment)) / self.sample_rate))
                else:
                    modulated_audio[frame_start_sample:frame_end_sample] = segment
            elif effect_type == "phaser":
                # Phaser è più complesso, spesso usa filtri allpass. Qui una simulazione molto grezza.
                # Aggiunge un leggero spostamento di fase variabile.
                modulated_audio[frame_start_sample:frame_end_sample] = segment + np.roll(segment, int(5 * current_intensity)) * np.sin(2 * np.pi * current_rate * (np.arange(len(segment)) / self.sample_rate))
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🔊 {effect_type.capitalize()} Frame {i + 1}/{num_frames} | Int: {current_intensity:.2f} | Rate: {current_rate:.2f}")

        return modulated_audio


    def apply_panning(self, audio: np.ndarray, horizontal_center_data: np.ndarray, progress_bar, status_text) -> np.ndarray:
        """
        Applica la panoramica stereo all'audio basata sul centro di massa orizzontale del video.
        audio: mono array
        horizontal_center_data: array di valori da 0 (sinistra) a 1 (destra)
        Restituisce un array stereo (2 colonne)
        """
        st.info("🔊 Applicazione Panoramica Stereo...")
        
        num_frames = len(horizontal_center_data)
        stereo_audio = np.zeros((audio.shape[0], 2), dtype=np.float32)
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_pan_pos = horizontal_center_data[i] # 0 = left, 1 = right
            
            # Calcola il gain per i canali sinistro e destro
            # Metodo del seno/coseno per panning costante in potenza
            gain_left = np.cos(current_pan_pos * np.pi / 2)
            gain_right = np.sin(current_pan_pos * np.pi / 2)
            
            segment = audio[frame_start_sample:frame_end_sample]
            
            stereo_audio[frame_start_sample:frame_end_sample, 0] = segment * gain_left
            stereo_audio[frame_start_sample:frame_end_sample, 1] = segment * gain_right
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Panoramica Frame {i + 1}/{num_frames} | Pos: {current_pan_pos:.2f} | L: {gain_left:.2f} R: {gain_right:.2f}")

        return stereo_audio


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

        # Initialize progress bar and status text before analysis
        analysis_progress_bar = st.progress(0)
        analysis_status_text = st.empty()

        with st.spinner("📊 Analisi frame video (luminosità, dettaglio, movimento, variazione movimento, centro orizzontale) in corso..."):
            brightness_data, detail_data, movement_data, variation_movement_data, horizontal_center_data, width, height, fps, video_duration = analyze_video_frames(video_input_path, analysis_progress_bar, analysis_status_text)
        
        if brightness_data is None: 
            return
        
        st.info(f"🎥 Durata video: {video_duration:.2f} secondi | Risoluzione Originale: {width}x{height} | FPS: {fps:.2f}")

        st.markdown("---")
        st.subheader("🎶 Configurazione Sintesi Audio Sperimentale")

        # Inizializza tutti i parametri a zero/default per evitare ReferenceError
        min_cutoff_user, max_cutoff_user = 0, 0
        min_resonance_user, max_resonance_user = 0, 0
        waveform_type_user = "sawtooth" # Default waveform
        filter_type_user = "lowpass" # Default filter type

        min_carrier_freq_user, max_carrier_freq_user = 0, 0
        min_modulator_freq_user, max_modulator_freq_user = 0, 0
        min_mod_index_user, max_mod_index_user = 0, 0

        min_noise_amp_user, max_noise_amp_user = 0, 0
        glitch_threshold_user, glitch_duration_frames_user, glitch_intensity_user = 0, 0, 0
        min_pitch_shift_semitones, max_pitch_shift_semitones = 0, 0
        min_time_stretch_rate, max_time_stretch_rate = 0, 0
        min_grain_freq, max_grain_freq = 0, 0
        min_grain_density, max_grain_density = 0, 0
        min_grain_duration, max_grain_duration = 0, 0
        max_delay_time_user, max_delay_feedback_user = 0, 0
        max_reverb_decay_user, max_reverb_wet_user = 0, 0
        modulation_effect_type = "none"
        modulation_intensity = 0.5
        modulation_rate = 0.1

        # --- Sezione Sintesi Sottrattiva (con checkbox di abilitazione) ---
        st.sidebar.header("Generazione Suono Base")
        enable_subtractive_synthesis = st.sidebar.checkbox("🔊 **Abilita Sintesi Sottrattiva (Suono Base)**", value=True)
        with st.sidebar.expander("Sintesi Sottrattiva (Forma d'Onda & Filtro)", expanded=True): 
            st.markdown("#### Oscillatore")
            waveform_type_user = st.selectbox(
                "Forma d'Onda Oscillatore:",
                ("sawtooth", "square", "sine", "triangle"),
                key="waveform_type",
                disabled=not enable_subtractive_synthesis
            )
            st.markdown("#### Filtro Base (modulato dalla Luminosità/Dettaglio)")
            st.markdown("**Controlli:**")
            st.markdown("- **Frequenza di Taglio:** controllata dalla **Luminosità** del video.")
            st.markdown("- **Risonanza:** controllata dal **Dettaglio/Contrasto** del video.")
            min_cutoff_user = st.slider("Min Frequenza Taglio (Hz)", 20, 5000, 100, key="sub_min_cutoff", disabled=not enable_subtractive_synthesis)
            max_cutoff_user = st.slider("Max Frequenza Taglio (Hz)", 1000, 20000, 8000, key="sub_max_cutoff", disabled=not enable_subtractive_synthesis)
            min_resonance_user = st.slider("Min Risonanza (Q)", 0.1, 5.0, 0.5, key="sub_min_res", disabled=not enable_subtractive_synthesis) 
            max_resonance_user = st.slider("Max Risonanza (Q)", 1.0, 30.0, 10.0, key="sub_max_res", disabled=not enable_subtractive_synthesis) 
        
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

        # --- Sezione Sintesi Granulare (con checkbox di abilitazione) ---
        st.sidebar.markdown("---")
        enable_granular_synthesis = st.sidebar.checkbox("🎶 **Abilita Sintesi Granulare**", value=False)

        if enable_granular_synthesis:
            with st.sidebar.expander("Sintesi Granulare Parametri", expanded=True):
                st.markdown("**Controlli:**")
                st.markdown("- **Frequenza Grani (Pito):** controllata dalla **Luminosità** del video.")
                st.markdown("- **Densità Grani (Texture):** controllata dal **Movimento** del video.")
                st.markdown("- **Durata Grani (Drone):** controllata dal **Dettaglio/Contrasto** del video.")
                
                min_grain_freq = st.slider("Min Frequenza Grano (Hz) - Pitch/Pito", 50, 2000, 200, key="gran_min_freq")
                max_grain_freq = st.slider("Max Frequenza Grano (Hz) - Pitch/Pono", 1000, 8000, 1500, key="gran_max_freq")
                
                min_grain_density = st.slider("Min Densità Grani (Grani/sec) - Texture", 1, 100, 10, key="gran_min_dens")
                max_grain_density = st.slider("Max Densità Grani (Grani/sec) - Texture", 20, 500, 100, key="gran_max_dens")
                
                min_grain_duration = st.slider("Min Durata Grano (sec) - Drone", 0.005, 0.2, 0.01, 0.001, key="gran_min_dur")
                max_grain_duration = st.slider("Max Durata Grano (sec) - Drone", 0.05, 1.0, 0.2, 0.001, key="gran_max_dur")
        
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

        # --- Sezione Pitch/Time Stretching (con checkbox di abilitazione) ---
        st.sidebar.markdown("---")
        enable_pitch_time_stretch = st.sidebar.checkbox("🎶 **Abilita Pitch Shifting / Time Stretching**", value=True)
        with st.sidebar.expander("Pitch Shifting / Time Stretching", expanded=True): 
            st.markdown("**Controlli:**")
            st.markdown("- **Time Stretch Rate:** controllato dalla **Luminosità** del video.")
            st.markdown("- **Pitch Shift:** controllato dal **Dettaglio/Contrasto** del video.")
            min_pitch_shift_semitones = st.slider("Min Pitch Shift (semitoni)", -24.0, 24.0, -12.0, 0.5, key="pitch_min", disabled=not enable_pitch_time_stretch)
            max_pitch_shift_semitones = st.slider("Max Pitch Shift (semitoni)", -24.0, 24.0, 12.0, 0.5, key="pitch_max", disabled=not enable_pitch_time_stretch)
            min_time_stretch_rate = st.slider("Min Time Stretch Rate", 0.1, 2.0, 0.8, 0.1, key="stretch_min", disabled=not enable_pitch_time_stretch) 
            max_time_stretch_rate = st.slider("Max Time Stretch Rate", 0.5, 5.0, 1.5, 0.1, key="stretch_max", disabled=not enable_pitch_time_stretch) 

        # --- Sezione Effetti Sonori Dinamici Unificata ---
        st.sidebar.markdown("---")
        st.sidebar.header("Effetti Sonori Dinamici")
        enable_dynamic_effects = st.sidebar.checkbox("🎛️ **Abilita Effetti Sonori Dinamici**", value=False)

        if enable_dynamic_effects:
            with st.sidebar.expander("Filtri Avanzati, Modulazione, Delay & Riverbero", expanded=True):
                st.markdown("#### Filtri Avanzati (modulati dalla Luminosità/Dettaglio)")
                filter_type_user = st.selectbox(
                    "Tipo di Filtro:",
                    ("lowpass", "highpass", "bandpass"),
                    key="filter_type_adv"
                )
                st.markdown("**(I seguenti slider controllano il filtro selezionato sopra)**")
                min_cutoff_adv = st.slider("Min Frequenza Taglio (Hz) - Filtri Avanzati", 20, 5000, 100, key="adv_min_cutoff")
                max_cutoff_adv = st.slider("Max Frequenza Taglio (Hz) - Filtri Avanzati", 1000, 20000, 8000, key="adv_max_cutoff")
                min_resonance_adv = st.slider("Min Risonanza (Q) - Filtri Avanzati", 0.1, 5.0, 0.5, key="adv_min_res") 
                max_resonance_adv = st.slider("Max Risonanza (Q) - Filtri Avanzati", 1.0, 30.0, 10.0, key="adv_max_res") 

                st.markdown("#### Effetti di Modulazione (Chorus/Flanger/Phaser)")
                modulation_effect_type = st.selectbox(
                    "Seleziona Effetto di Modulazione:",
                    ("none", "chorus", "flanger", "phaser"),
                    key="mod_effect_type"
                )
                if modulation_effect_type != "none":
                    st.markdown("**Controlli:**")
                    st.markdown("- **Intensità:** controllata dalla **Variazione del Movimento**.")
                    st.markdown("- **Velocità:** controllata dal **Dettaglio/Contrasto**.")
                    modulation_intensity = st.slider("Intensità Effetto", 0.1, 1.0, 0.5, 0.1, key="mod_intensity")
                    modulation_rate = st.slider("Velocità Modulazione", 0.01, 1.0, 0.1, 0.01, key="mod_rate")


                st.markdown("#### Delay Dinamico")
                st.markdown("**Controlli:**")
                st.markdown("- **Tempo di Delay:** controllato dal **Movimento** del video.")
                st.markdown("- **Feedback:** controllato dal **Dettaglio/Contrasto** del video.")
                max_delay_time_user = st.slider("Max Tempo Delay (sec)", 0.1, 2.0, 0.5, 0.05, key="delay_time")
                max_delay_feedback_user = st.slider("Max Feedback Delay", 0.0, 0.9, 0.5, 0.05, key="delay_feedback")

                st.markdown("#### Riverbero Dinamico")
                st.markdown("**Controlli:**")
                st.markdown("- **Tempo di Decadimento:** controllato dal **Dettaglio/Contrasto** del video.")
                st.markdown("- **Mix Wet/Dry:** controllato dalla **Luminosità** del video.")
                max_reverb_decay_user = st.slider("Max Tempo Decadimento (sec)", 0.5, 10.0, 3.0, 0.1, key="reverb_decay")
                max_reverb_wet_user = st.slider("Max Mix Wet/Dry Riverbero", 0.0, 1.0, 0.4, 0.05, key="reverb_wet")


        st.markdown("---")
        st.subheader("⬇️ Opzioni di Download")
        
        output_resolution_choice = st.selectbox(
            "Seleziona la risoluzione di output del video:",
            list(FORMAT_RESOLUTIONS.keys()) # This is the positional argument
            # The next line would cause the error if it were a positional argument:
            # , "Originale" # This would be a positional argument after the keyword argument 'key'
        )
        # If output_resolution_choice was causing the error, it would be because you tried to specify a default value
        # as a positional argument after `list(FORMAT_RESOLUTIONS.keys())`.
        # The correct way to set a default for `selectbox` is using the `index` argument:
        # st.selectbox("Label", options, index=0) for "Originale"

        download_option = st.radio(
            "Cosa vuoi scaricare?",
            ("Video con Audio", "Solo Audio")
        )

        if not check_ffmpeg():
            st.warning("⚠️ FFmpeg non disponibile sul tuo sistema. L'unione o la ricodifica del video potrebbe non funzionare. Assicurati che FFmpeg sia installato e nel PATH.")
            
        if st.button("🎵 Genera e Prepara Download"):
            base_name_output = os.path.splitext(uploaded_file.name)[0]
            unique_id_audio = str(np.random.randint(10000, 99999)) 
            audio_output_path = os.path.join("temp", f"{base_name_output}_{unique_id_audio}_generated_audio.wav") 
            final_video_path = os.path.join("temp", f"{base_name_output}_{unique_id_audio}_final_videosound.mp4")
            
            os.makedirs("temp", exist_ok=True)

            audio_gen = AudioGenerator(AUDIO_SAMPLE_RATE, int(fps)) 
            
            total_samples = int(video_duration * AUDIO_SAMPLE_RATE)
            
            # Inizializza l'audio base a silenzio
            combined_audio_layers = np.zeros(total_samples, dtype=np.float32)

            # Define global progress bar and status text for audio generation
            audio_progress_bar = st.progress(0)
            audio_status_text = st.empty()

            # --- Generazione dell'audio base (Sintesi Sottrattiva) ---
            if enable_subtractive_synthesis:
                st.info("🎵 Generazione dell'onda base (Sintesi Sottrattiva)...")
                subtractive_layer = audio_gen.generate_subtractive_waveform(total_samples, waveform_type_user)
                combined_audio_layers += subtractive_layer
                st.success("✅ Strato Sintesi Sottrattiva generato!")
            else:
                st.info("🎵 Sintesi Sottrattiva disabilitata.")


            # --- Aggiungi lo strato FM se abilitato ---
            if enable_fm_synthesis:
                fm_layer = audio_gen.generate_fm_layer(
                    total_samples,
                    brightness_data, 
                    movement_data, 
                    min_carrier_freq_user, max_carrier_freq_user,
                    min_modulator_freq_user, max_modulator_freq_user,
                    min_mod_index_user, max_mod_index_user, 
                    audio_progress_bar, # Positional arg, correctly placed
                    audio_status_text   # Positional arg, correctly placed
                )
                combined_audio_layers += fm_layer * 0.5 # Aggiungi con un peso ridotto
                st.success("✅ Strato FM generato e combinato!")

            # --- Aggiungi lo strato Granulare se abilitato ---
            if enable_granular_synthesis:
                granular_layer = audio_gen.generate_granular_layer(
                    total_samples,
                    brightness_data, 
                    detail_data, 
                    movement_data,
                    min_grain_freq, max_grain_freq,
                    min_grain_density, max_grain_density,
                    min_grain_duration, max_grain_duration, 
                    audio_progress_bar, # Positional arg, correctly placed
                    audio_status_text   # Positional arg, correctly placed
                )
                combined_audio_layers += granular_layer * 0.5 # Aggiungi con un peso ridotto
                st.success("✅ Strato Granulare generato e combinato!")

            # --- Aggiungi il layer di Rumore se abilitato ---
            if enable_noise_effect:
                # Applica il rumore all'audio combinato fino a questo punto
                combined_audio_layers = audio_gen.add_noise_layer(
                    combined_audio_layers, 
                    detail_data,
                    min_noise_amp_user, 
                    max_noise_amp_user, 
                    audio_progress_bar, # Positional arg, correctly placed
                    audio_status_text   # Positional arg, correctly placed
                )
                st.success("✅ Strato Rumore aggiunto!")

            # --- Applica effetti Glitch se abilitati ---
            if enable_glitch_effect:
                # Applica il glitch all'audio combinato fino a questo punto
                combined_audio_layers = audio_gen.apply_glitch_effect(
                    combined_audio_layers, 
                    variation_movement_data,
                    glitch_threshold_user, 
                    glitch_duration_frames_user,
                    glitch_intensity_user, 
                    audio_progress_bar, # Positional arg, correctly placed
                    audio_status_text   # Positional arg, correctly placed
                )
                st.success("✅ Effetti Glitch applicati!")

            # --- Processamento degli effetti dinamici (filtro, pitch, time stretch) ---
            with st.spinner("🎧 Applicazione effetti dinamici all'audio generato..."):
                processed_audio = combined_audio_layers # Inizia con l'audio base/combinato
                
                # Applica Pitch Shifting e Time Stretching
                if enable_pitch_time_stretch:
                    processed_audio = audio_gen.apply_pitch_time_stretch(
                        processed_audio, 
                        brightness_data, 
                        detail_data,
                        min_pitch_shift_semitones=min_pitch_shift_semitones,
                        max_pitch_shift_semitones=max_pitch_shift_semitones,
                        min_time_stretch_rate=min_time_stretch_rate,
                        max_time_stretch_rate=max_time_stretch_rate, 
                        audio_progress_bar, # Positional arg, correctly placed
                        audio_status_text   # Positional arg, correctly placed
                    )
                    st.success("✅ Effetti pitch e stretch applicati!")
                else:
                    st.info("🎶 Pitch Shifting / Time Stretching disabilitato.")


                # --- Applicazione Effetti Sonori Dinamici unificati ---
                if enable_dynamic_effects:
                    # Applica Filtro Avanzato
                    # This is likely the line that caused the original SyntaxError.
                    # The `filter_type` argument was defined in `apply_dynamic_filter` with a default
                    # after `progress_bar` and `status_text`.
                    # In the *call* here, `filter_type` should be a keyword argument if `progress_bar` and `status_text` are positional.
                    # Or, if `progress_bar` and `status_text` are *also* keyword arguments, the order doesn't matter as much.
                    # Given the function definition was fixed to have progress_bar, status_text first, then filter_type as default:
                    # def apply_dynamic_filter(self, audio: np.ndarray, ..., progress_bar, status_text, filter_type: str = "lowpass")
                    # So, `progress_bar` and `status_text` should be positional, and `filter_type` keyword.
                    # Let's verify the calls.

                    processed_audio = audio_gen.apply_dynamic_filter(
                        processed_audio,
                        brightness_data,
                        detail_data,
                        min_cutoff=min_cutoff_adv,
                        max_cutoff=max_cutoff_adv,
                        min_res=min_resonance_adv,
                        max_res=max_resonance_adv,
                        audio_progress_bar,  # Positional argument
                        audio_status_text,   # Positional argument
                        filter_type=filter_type_user # Keyword argument, correctly placed after positional arguments
                    )
                    st.success("✅ Filtri avanzati applicati!")

                    # Applica Effetto di Modulazione (Chorus/Flanger/Phaser)
                    processed_audio = audio_gen.apply_modulation_effect(
                        processed_audio,
                        variation_movement_data,
                        detail_data,
                        audio_progress_bar, # Positional argument
                        audio_status_text,   # Positional argument
                        effect_type=modulation_effect_type, # Keyword argument
                        intensity=modulation_intensity,    # Keyword argument
                        rate=modulation_rate               # Keyword argument
                    )
                    st.success(f"✅ Effetto di Modulazione '{modulation_effect_type}' applicato!")

                    # Applica Delay
                    processed_audio = audio_gen.apply_delay_effect(
                        processed_audio,
                        movement_data,
                        detail_data,
                        max_delay_time_user,
                        max_delay_feedback_user, 
                        audio_progress_bar, # Positional argument
                        audio_status_text   # Positional argument
                    )
                    st.success("✅ Effetto Delay applicato!")

                    # Applica Riverbero
                    processed_audio = audio_gen.apply_reverb_effect(
                        processed_audio,
                        detail_data,
                        brightness_data,
                        max_reverb_decay_user,
                        max_reverb_wet_user, 
                        audio_progress_bar, # Positional argument
                        audio_status_text   # Positional argument
                    )
                    st.success("✅ Effetto Riverbero applicato!")
                else:
                    st.info("🎶 Effetti Sonori Dinamici disabilitati.")
            
            if processed_audio is None or processed_audio.size == 0:
                st.error("❌ Errore nel processamento degli effetti audio.")
                return

            # --- Normalizzazione Finale e Panoramica Stereo ---
            with st.spinner("🎧 Normalizzazione finale e Panoramica Stereo..."):
                # Normalizzazione prima del panning per evitare clipping
                if np.max(np.abs(processed_audio)) > 0:
                    processed_audio = processed_audio / np.max(np.abs(processed_audio)) * 0.9 
                
                # Applica panning stereo
                final_audio_stereo = audio_gen.apply_panning(
                    processed_audio, 
                    horizontal_center_data, 
                    audio_progress_bar, # Positional argument
                    audio_status_text   # Positional argument
                )
                st.success("✅ Normalizzazione e Panoramica Stereo completate!")


            try:
                # Salva l'audio stereo
                sf.write(audio_output_path, final_audio_stereo, AUDIO_SAMPLE_RATE)
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
