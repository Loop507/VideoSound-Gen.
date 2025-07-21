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
            carrier_wave = np.sin(2 * np.pi * carrier_freq * t_segment + mod_index * modulator_wave)
            
            fm_audio_layer[frame_start_sample:frame_end_sample] = carrier_wave
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎵 FM Layer | Frame {i + 1}/{num_frames} | Car: {int(carrier_freq)}Hz | Mod: {int(modulator_freq)}Hz | Index: {mod_index:.1f}")

        st.success("✅ Generazione Strato FM completata!")
        return fm_audio_layer.astype(np.float32)


    def generate_granular_layer(self, duration_samples: int, brightness_data: np.ndarray, 
                                 movement_data: np.ndarray, detail_data: np.ndarray,
                                 min_grain_freq: float, max_grain_freq: float, 
                                 min_grain_density: float, max_grain_density: float,
                                 min_grain_duration: float, max_grain_duration: float, progress_bar, status_text) -> np.ndarray:
        """Genera un layer di sintesi granulare modulato dai dati visivi."""
        st.info("✨ Generazione Strato Granulare...")
        granular_audio_layer = np.zeros(duration_samples, dtype=np.float32)
        
        num_frames = len(brightness_data)
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, duration_samples)
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_brightness = brightness_data[i]
            current_movement = movement_data[i]
            current_detail = detail_data[i]

            # Mappatura dei parametri granulari ai dati visivi
            grain_freq = min_grain_freq + current_brightness * (max_grain_freq - min_grain_freq)
            grain_density = int(min_grain_density + current_movement * (max_grain_density - min_grain_density))
            grain_duration_sec = min_grain_duration + current_detail * (max_grain_duration - min_grain_duration)
            grain_duration_samples = int(grain_duration_sec * self.sample_rate)

            # Genera grani all'interno del frame corrente
            for _ in range(grain_density):
                if grain_duration_samples == 0: continue

                grain_start = frame_start_sample + np.random.randint(0, self.samples_per_frame - grain_duration_samples + 1)
                grain_end = grain_start + grain_duration_samples

                if grain_end > duration_samples:
                    grain_end = duration_samples
                    grain_duration_samples = grain_end - grain_start
                    if grain_duration_samples <= 0: continue

                t_grain = np.linspace(0, grain_duration_samples / self.sample_rate, grain_duration_samples, endpoint=False)
                # Semplice onda sinusoidale per il grano con inviluppo hanning
                grain = np.sin(2 * np.pi * grain_freq * t_grain) * np.hanning(grain_duration_samples)

                granular_audio_layer[grain_start:grain_end] += grain

            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"✨ Granular Layer | Frame {i + 1}/{num_frames} | Freq: {int(grain_freq)}Hz | Dens: {grain_density} | Dur: {grain_duration_sec:.2f}s")
        
        st.success("✅ Generazione Strato Granulare completata!")
        return granular_audio_layer.astype(np.float32)


    def generate_noise_layer(self, duration_samples: int, detail_data: np.ndarray, min_amp: float, max_amp: float, progress_bar, status_text) -> np.ndarray:
        """Genera un layer di rumore bianco con ampiezza modulata dal dettaglio visivo."""
        st.info("🔊 Generazione Strato Rumore...")
        noise_audio_layer = np.zeros(duration_samples, dtype=np.float32)
        
        num_frames = len(detail_data)

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, duration_samples)

            if frame_start_sample >= frame_end_sample:
                continue
            
            current_detail = detail_data[i]
            amplitude = min_amp + current_detail * (max_amp - min_amp)

            segment_length = frame_end_sample - frame_start_sample
            noise_segment = (np.random.rand(segment_length) * 2 - 1) * amplitude
            noise_audio_layer[frame_start_sample:frame_end_sample] = noise_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🔊 Noise Layer | Frame {i + 1}/{num_frames} | Amp: {amplitude:.2f}")

        st.success("✅ Generazione Strato Rumore completata!")
        return noise_audio_layer.astype(np.float32)


    def apply_glitch_effect(self, audio: np.ndarray, variation_movement_data: np.ndarray, 
                            glitch_threshold: float, glitch_duration_frames: int, glitch_intensity: float, progress_bar, status_text) -> np.ndarray:
        """Applica un effetto glitch all'audio basato sulla variazione del movimento."""
        st.info("👾 Applicazione Effetto Glitch...")
        processed_audio = audio.copy()
        
        num_frames = len(variation_movement_data)
        
        glitch_duration_samples = glitch_duration_frames * self.samples_per_frame
        
        for i in range(num_frames):
            current_variation_movement = variation_movement_data[i]
            
            if current_variation_movement > glitch_threshold:
                frame_start_sample = i * self.samples_per_frame
                
                # Definisci la finestra del glitch
                glitch_segment_start = max(0, frame_start_sample - glitch_duration_samples // 2)
                glitch_segment_end = min(len(audio), frame_start_sample + glitch_duration_samples // 2)

                if glitch_segment_end - glitch_segment_start <= 0:
                    continue

                # Esegui un semplice glitch: ripeti, distorci o silenzia una piccola porzione
                # Qui facciamo una ripetizione/time stretch inverso per un effetto di "stutter"
                
                # Scegli un punto casuale all'interno del segmento di glitch da cui prelevare il suono
                source_start = np.random.randint(max(0, glitch_segment_start - self.samples_per_frame * 2), glitch_segment_start) # prendi da prima
                source_end = source_start + (glitch_segment_end - glitch_segment_start)

                if source_end > len(audio) or source_start < 0:
                    source_start = glitch_segment_start # Fallback se fuori limiti
                    source_end = glitch_segment_end

                if source_end - source_start <= 0:
                    continue

                # Preleva la porzione sorgente
                source_segment = audio[source_start:source_end]
                
                # Applica il glitch: per esempio, sovrascrivi con la porzione sorgente, magari con un'intensità
                # maggiore di quella originale per renderlo più evidente
                processed_audio[glitch_segment_start:glitch_segment_end] = source_segment * glitch_intensity
                
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"👾 Glitch Effect | Frame {i + 1}/{num_frames} | VarMov: {current_variation_movement:.2f}")

        st.success("✅ Applicazione Effetto Glitch completata!")
        return processed_audio.astype(np.float32)


    def apply_pitch_time_stretch(self, audio: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray, 
                                 min_pitch_shift: float, max_pitch_shift: float, 
                                 min_time_stretch: float, max_time_stretch: float, progress_bar, status_text) -> np.ndarray:
        """Applica pitch shifting e time stretching dinamici all'audio."""
        st.info("🎶 Applicazione Pitch/Time Stretch...")
        
        # librosa.effects.time_stretch e librosa.effects.pitch_shift richiedono un audio a singola dimensione
        # Li applicheremo per frame o segmenti per simulare la dinamicità
        
        # Per semplicità e performance, divideremo l'audio in segmenti più grandi dei singoli frame
        # Per esempio, 0.1 secondi per segmento
        segment_duration_samples = self.sample_rate // 10
        
        processed_audio = np.zeros_like(audio)
        current_sample = 0
        
        num_frames = len(brightness_data)
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio))

            if frame_start_sample >= frame_end_sample:
                continue

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]

            # Mappatura dei parametri
            pitch_shift_value = min_pitch_shift + current_detail * (max_pitch_shift - min_pitch_shift)
            time_stretch_rate = min_time_stretch + current_brightness * (max_time_stretch - min_time_stretch)
            
            # Applica per ogni frame (o un segmento di frame)
            segment = audio[frame_start_sample:frame_end_sample]
            
            if len(segment) > 0:
                try:
                    # Applica prima il pitch shift
                    pitched_segment = librosa.effects.pitch_shift(segment, sr=self.sample_rate, n_steps=pitch_shift_value)
                    
                    # Poi il time stretch (se il rate è > 0)
                    if time_stretch_rate > 0.01: # Evita divisione per zero o valori troppo piccoli
                        stretched_segment = librosa.effects.time_stretch(pitched_segment, rate=time_stretch_rate)
                    else:
                        stretched_segment = pitched_segment # Nessun time stretch se rate è quasi zero

                    # Assicurati che il segmento processato abbia la stessa lunghezza del segmento originale
                    # per evitare problemi di allineamento
                    if len(stretched_segment) > len(segment):
                        stretched_segment = stretched_segment[:len(segment)]
                    elif len(stretched_segment) < len(segment):
                        stretched_segment = np.pad(stretched_segment, (0, len(segment) - len(stretched_segment)))
                    
                    processed_audio[frame_start_sample:frame_end_sample] = stretched_segment

                except Exception as e:
                    # In caso di errore (es. segmenti troppo corti per librosa), usa il segmento originale
                    processed_audio[frame_start_sample:frame_end_sample] = segment
                    # st.warning(f"Problema con Pitch/Time Stretch al frame {i}: {e}") # Per debug
                    
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎶 Pitch/Time Stretch | Frame {i + 1}/{num_frames} | Pitch: {pitch_shift_value:.1f} | Time: {time_stretch_rate:.2f}")

        st.success("✅ Applicazione Pitch/Time Stretch completata!")
        return processed_audio.astype(np.float32)


    def create_dynamic_biquad_filter(self, audio: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray, 
                                     filter_type: str, min_cutoff: float, max_cutoff: float, 
                                     min_resonance: float, max_resonance: float, progress_bar, status_text) -> np.ndarray:
        """Applica un filtro biquad dinamico (LP, HP, BP) modulato da luminosità e dettaglio."""
        st.info(f"🎚️ Applicazione Filtro Dinamico ({filter_type.upper()})...")
        processed_audio = np.zeros_like(audio)
        
        num_frames = len(brightness_data)
        
        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]

            # Mappatura dei parametri del filtro
            cutoff_freq = min_cutoff + current_brightness * (max_cutoff - min_cutoff)
            Q_factor = min_resonance + current_detail * (max_resonance - min_resonance)
            Q_factor = max(0.1, Q_factor) # Evita Q troppo bassi o nulli

            nyquist = 0.5 * self.sample_rate
            normal_cutoff = cutoff_freq / nyquist

            if normal_cutoff > 0.99: # Clamping per evitare frequenze di taglio oltre Nyquist
                normal_cutoff = 0.99
            elif normal_cutoff < 0.01:
                normal_cutoff = 0.01
            
            if filter_type == "lowpass":
                b, a = butter(2, normal_cutoff, btype='low', analog=False, output='ba')
            elif filter_type == "highpass":
                b, a = butter(2, normal_cutoff, btype='high', analog=False, output='ba')
            elif filter_type == "bandpass":
                b, a = butter(2, normal_cutoff, btype='band', analog=False, output='ba', Wn=[normal_cutoff * 0.5, normal_cutoff * 1.5]) # Semplice BP
            else: # default lowpass
                b, a = butter(2, normal_cutoff, btype='low', analog=False, output='ba')

            # Applica il filtro al segmento audio del frame corrente
            segment = audio[frame_start_sample:frame_end_sample]
            if len(segment) > 0:
                filtered_segment = lfilter(b, a, segment)
                processed_audio[frame_start_sample:frame_end_sample] = filtered_segment
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🎚️ Filter Effect | Frame {i + 1}/{num_frames} | Cutoff: {int(cutoff_freq)}Hz | Q: {Q_factor:.2f}")

        st.success("✅ Applicazione Filtro Dinamico completata!")
        return processed_audio.astype(np.float32)

    def apply_modulation_effect(self, audio: np.ndarray, detail_data: np.ndarray, variation_movement_data: np.ndarray,
                                modulation_effect_type: str, modulation_intensity: float, modulation_rate: float, progress_bar, status_text) -> np.ndarray:
        """Applica effetti di modulazione dinamici (tremolo, vibrato, phaser) modulati da dati visivi."""
        if modulation_effect_type == "none":
            return audio.astype(np.float32)

        st.info(f"🌀 Applicazione Effetto di Modulazione ({modulation_effect_type.upper()})...")
        processed_audio = np.zeros_like(audio)
        
        num_frames = len(detail_data)

        # Frequenza base dell'LFO (modulazione)
        base_lfo_freq = 5.0 # Hz

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_sample, len(audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_detail = detail_data[i] # Per la velocità dell'LFO
            current_variation_movement = variation_movement_data[i] # Per l'intensità dell'effetto

            # Mappatura dei parametri di modulazione
            lfo_rate = base_lfo_freq + current_detail * (modulation_rate - base_lfo_freq)
            effect_amount = current_variation_movement * modulation_intensity # L'intensità massima è decisa dall'utente

            segment = audio[frame_start_sample:frame_end_sample]
            if len(segment) == 0: continue

            # Genera un LFO (Low Frequency Oscillator) per il segmento corrente
            t_segment = np.linspace(0, (frame_end_sample - frame_start_sample) / self.sample_rate, (frame_end_sample - frame_start_sample), endpoint=False)
            lfo_wave = np.sin(2 * np.pi * lfo_rate * t_segment)

            if modulation_effect_type == "tremolo":
                # Modula l'ampiezza
                mod_segment = segment * (1 + lfo_wave * effect_amount)
            elif modulation_effect_type == "vibrato":
                # Modula la fase per cambiare l'intonazione (pitch)
                # Questo è più complesso e richiede interpolazione, useremo una semplificazione
                pitch_shift_amount = lfo_wave * effect_amount * 0.1 # Piccoli spostamenti di semitoni
                mod_segment = librosa.effects.pitch_shift(segment, sr=self.sample_rate, n_steps=pitch_shift_amount)
                # Assicurati la stessa lunghezza
                if len(mod_segment) > len(segment): mod_segment = mod_segment[:len(segment)]
                elif len(mod_segment) < len(segment): mod_segment = np.pad(mod_segment, (0, len(segment) - len(mod_segment)))
            elif modulation_effect_type == "phaser":
                # Simula un phaser applicando un filtro che sposta la fase
                # Questo è molto complesso da implementare da zero dinamicamente per frame
                # Per una demo, potremmo applicare un filtro passa-banda con frequenza centrale modulata
                center_freq = 1000 + lfo_wave * effect_amount * 500 # Modula frequenza centrale
                nyquist = 0.5 * self.sample_rate
                normal_center = center_freq / nyquist
                if normal_center > 0.99: normal_center = 0.99
                elif normal_center < 0.01: normal_center = 0.01

                b, a = butter(2, normal_center, btype='band', analog=False, output='ba', Wn=[normal_center * 0.8, normal_center * 1.2])
                mod_segment = lfilter(b, a, segment)
            else:
                mod_segment = segment # Fallback

            processed_audio[frame_start_sample:frame_end_sample] = mod_segment

            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🌀 Mod. Effect | Frame {i + 1}/{num_frames} | Type: {modulation_effect_type.capitalize()} | Rate: {lfo_rate:.1f}Hz | Amt: {effect_amount:.2f}")

        st.success(f"✅ Applicazione Effetto di Modulazione ({modulation_effect_type.upper()}) completata!")
        return processed_audio.astype(np.float32)


    def apply_dynamic_delay(self, audio: np.ndarray, movement_data: np.ndarray, detail_data: np.ndarray, 
                            max_delay_time: float, max_delay_feedback: float, progress_bar, status_text) -> np.ndarray:
        """Applica un effetto delay dinamico modulato da movimento e dettaglio."""
        st.info("🕰️ Applicazione Delay Dinamico...")
        processed_audio = audio.copy()
        
        num_frames = len(movement_data)
        
        # Buffer per il delay
        delay_buffer = np.zeros(int(max_delay_time * self.sample_rate) + 1, dtype=np.float32)
        write_ptr = 0

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_movement = movement_data[i]
            current_detail = detail_data[i]

            # Mappatura dei parametri del delay
            delay_time_samples = int((current_movement * max_delay_time) * self.sample_rate)
            feedback_amount = current_detail * max_delay_feedback # Da 0 a max_delay_feedback

            # Assicurati che il delay_time_samples non superi la dimensione del buffer
            delay_time_samples = min(delay_time_samples, len(delay_buffer) - 1)
            delay_time_samples = max(0, delay_time_samples) # Non negativo

            # Applica il delay campione per campione nel segmento corrente
            for j in range(frame_start_sample, frame_end_sample):
                if j >= len(audio): break

                # Calcola il puntatore di lettura per il delay
                read_ptr = (write_ptr - delay_time_samples + len(delay_buffer)) % len(delay_buffer)
                
                # Suono ritardato
                delayed_sound = delay_buffer[read_ptr]
                
                # Aggiungi il suono ritardato all'output
                processed_audio[j] += delayed_sound 
                
                # Scrivi il suono corrente (con feedback) nel buffer di delay
                delay_buffer[write_ptr] = audio[j] + delayed_sound * feedback_amount
                
                write_ptr = (write_ptr + 1) % len(delay_buffer)
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🕰️ Delay Effect | Frame {i + 1}/{num_frames} | Time: {delay_time_samples / self.sample_rate:.2f}s | Feedback: {feedback_amount:.2f}")

        st.success("✅ Applicazione Delay Dinamico completata!")
        return processed_audio.astype(np.float32)


    def apply_dynamic_reverb(self, audio: np.ndarray, brightness_data: np.ndarray, detail_data: np.ndarray, 
                             max_reverb_decay: float, max_reverb_wet: float, progress_bar, status_text) -> np.ndarray:
        """Applica un effetto riverbero dinamico modulato da luminosità e dettaglio."""
        st.info("🌌 Applicazione Riverbero Dinamico...")
        
        # Una semplice simulazione di riverbero con un feedback delay line (molto rudimentale)
        # Per un riverbero più convincente, servirebbero algoritmi più complessi (es. allpass/comb filters)
        # Useremo una catena di delay con decadimento.

        processed_audio = np.zeros_like(audio)
        num_frames = len(brightness_data)

        # Parametri fissi per il riverbero (possono essere esposti come UI se serve più controllo)
        num_delays = 4 # Numero di delay lines per simulare early reflections
        base_delay_times = [0.015, 0.023, 0.031, 0.040] # tempi di ritardo in secondi
        
        # Buffer per ogni delay line
        delay_buffers = [np.zeros(int(t * self.sample_rate) + 1, dtype=np.float32) for t in base_delay_times]
        write_pointers = [0] * num_delays

        # Per il decadimento globale (ricoda)
        decay_buffer = np.zeros(int(max_reverb_decay * self.sample_rate) + 1, dtype=np.float32)
        decay_write_ptr = 0


        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_brightness = brightness_data[i]
            current_detail = detail_data[i]

            # Mappatura dei parametri del riverbero
            wet_level = current_brightness * max_reverb_wet # Mix tra segnale originale e riverberato
            decay_factor = current_detail * max_reverb_decay # Tempo di decadimento base

            # Loop per ogni campione nel frame corrente
            for j in range(frame_start_sample, frame_end_sample):
                if j >= len(audio): break

                input_sample = audio[j]
                reverb_output = 0.0

                # Early reflections (simulazione con multiple delay lines)
                for k in range(num_delays):
                    delay_time_samples = int(base_delay_times[k] * self.sample_rate)
                    read_ptr = (write_pointers[k] - delay_time_samples + len(delay_buffers[k])) % len(delay_buffers[k])
                    
                    reverb_output += delay_buffers[k][read_ptr] * 0.2 # Contributo delle early reflections
                    
                    # Scrivi nel buffer di early reflection
                    delay_buffers[k][write_pointers[k]] = input_sample + delay_buffers[k][read_ptr] * 0.3 # Piccolo feedback per dispersione
                    write_pointers[k] = (write_pointers[k] + 1) % len(delay_buffers[k])

                # Ricoda (simulazione con un delay principale e decadimento)
                decay_read_ptr = (decay_write_ptr - int(decay_factor * self.sample_rate) + len(decay_buffer)) % len(decay_buffer)
                reverb_output += decay_buffer[decay_read_ptr] * 0.5 # Contributo della ricoda
                decay_buffer[decay_write_ptr] = input_sample + decay_buffer[decay_read_ptr] * 0.7 # Feedback per la ricoda

                decay_write_ptr = (decay_write_ptr + 1) % len(decay_buffer)

                # Mix dry/wet
                processed_audio[j] = input_sample * (1 - wet_level) + reverb_output * wet_level
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"🌌 Reverb Effect | Frame {i + 1}/{num_frames} | Decay: {decay_factor:.2f}s | Wet: {wet_level:.2f}")

        st.success("✅ Applicazione Riverbero Dinamico completata!")
        return processed_audio.astype(np.float32)


    def apply_stereo_panning(self, audio_mono: np.ndarray, horizontal_center_data: np.ndarray, progress_bar, status_text) -> np.ndarray:
        """Applica la panoramica stereo dinamica basata sul centro di massa orizzontale."""
        st.info("↔️ Applicazione Panoramica Stereo...")
        
        # Inizializza l'audio stereo (2 canali)
        audio_stereo = np.zeros((len(audio_mono), 2), dtype=np.float32)
        
        num_frames = len(horizontal_center_data)

        for i in range(num_frames):
            frame_start_sample = i * self.samples_per_frame
            frame_end_sample = min((i + 1) * self.samples_per_frame, len(audio_mono))
            
            if frame_start_sample >= frame_end_sample:
                continue

            current_horizontal_center = horizontal_center_data[i]

            # Mappatura del centro orizzontale (0-1) a un valore di panning (-1 a 1, dove -1 è L, 0 è C, 1 è R)
            # Un valore di 0.5 (centro) nel video -> 0 nel panning
            # Un valore di 0 (sinistra) nel video -> -1 nel panning
            # Un valore di 1 (destra) nel video -> 1 nel panning
            panning_value = (current_horizontal_center * 2) - 1 

            # Calcolo dei guadagni per i canali sinistro e destro
            # Metodo di "panning a potenza costante" per evitare cali di volume
            gain_left = np.sqrt(0.5 * (1 - panning_value))
            gain_right = np.sqrt(0.5 * (1 + panning_value))
            
            segment = audio_mono[frame_start_sample:frame_end_sample]
            
            audio_stereo[frame_start_sample:frame_end_sample, 0] = segment * gain_left  # Canale sinistro
            audio_stereo[frame_start_sample:frame_end_sample, 1] = segment * gain_right # Canale destro
            
            progress_bar.progress((i + 1) / num_frames)
            status_text.text(f"↔️ Pan Effect | Frame {i + 1}/{num_frames} | Center: {current_horizontal_center:.2f} | Pan: {panning_value:.2f}")

        st.success("✅ Applicazione Panoramica Stereo completata!")
        return audio_stereo.astype(np.float32)

    @st.cache_data(show_spinner=False)
    def generate_audio(_self, audio_duration_seconds: float, brightness_data: np.ndarray, 
                       detail_data: np.ndarray, movement_data: np.ndarray, 
                       variation_movement_data: np.ndarray, horizontal_center_data: np.ndarray,
                       params: dict) -> np.ndarray:
        
        total_samples = int(audio_duration_seconds * _self.sample_rate)
        if total_samples == 0:
            st.warning("⚠️ Durata audio calcolata pari a zero campioni, impossibile generare.")
            return np.zeros((1, 2), dtype=np.float32) # Ritorna un array vuoto per evitare errori

        # Normalizza tutti i dati visivi per essere sicuri che siano tra 0 e 1
        # Già fatto durante l'analisi, ma un doppio controllo non guasta
        brightness_data = np.clip(brightness_data, 0, 1)
        detail_data = np.clip(detail_data, 0, 1)
        movement_data = np.clip(movement_data, 0, 1)
        variation_movement_data = np.clip(variation_movement_data, 0, 1)
        horizontal_center_data = np.clip(horizontal_center_data, 0, 1)

        st.info("🎵 Inizio generazione audio sperimentale...")
        
        # Array audio principale (mono)
        main_audio_mono = np.zeros(total_samples, dtype=np.float32)
        
        audio_progress_bar = st.progress(0, text="Generazione Audio...")
        audio_status_text = st.empty()

        # 1. Sintesi Sottrattiva (Base Sound)
        if params['enable_subtractive_synthesis']:
            subtractive_base_wave = _self.generate_subtractive_waveform(total_samples, params['waveform_type'])
            
            sub_filter_progress = st.progress(0, text="Applicazione Filtro Sottrattivo...")
            sub_filter_status = st.empty()
            filtered_subtractive_audio = _self.create_dynamic_biquad_filter(
                subtractive_base_wave, brightness_data, detail_data, 
                params['subtractive_filter_type'], params['min_cutoff'], params['max_cutoff'], 
                params['min_resonance'], params['max_resonance'], sub_filter_progress, sub_filter_status
            )
            main_audio_mono += filtered_subtractive_audio * params['subtractive_volume']
            del subtractive_base_wave, filtered_subtractive_audio
            gc.collect()


        # 2. Sintesi FM Layer
        if params['enable_fm_synthesis']:
            fm_layer_progress = st.progress(0, text="Generazione Strato FM...")
            fm_layer_status = st.empty()
            fm_audio = _self.generate_fm_layer(
                total_samples, brightness_data, movement_data, 
                params['min_carrier_freq'], params['max_carrier_freq'],
                params['min_modulator_freq'], params['max_modulator_freq'],
                params['min_mod_index'], params['max_mod_index'], fm_layer_progress, fm_layer_status
            )
            main_audio_mono += fm_audio * params['fm_volume']
            del fm_audio
            gc.collect()

        # 3. Sintesi Granulare Layer
        if params['enable_granular_synthesis']:
            granular_layer_progress = st.progress(0, text="Generazione Strato Granulare...")
            granular_layer_status = st.empty()
            granular_audio = _self.generate_granular_layer(
                total_samples, brightness_data, movement_data, detail_data,
                params['min_grain_freq'], params['max_grain_freq'], 
                params['min_grain_density'], params['max_grain_density'],
                params['min_grain_duration'], params['max_grain_duration'], granular_layer_progress, granular_layer_status
            )
            main_audio_mono += granular_audio * params['granular_volume']
            del granular_audio
            gc.collect()
        
        # 4. Noise Layer
        if params['enable_noise_effect']:
            noise_layer_progress = st.progress(0, text="Generazione Strato Rumore...")
            noise_layer_status = st.empty()
            noise_audio = _self.generate_noise_layer(
                total_samples, detail_data, params['min_noise_amp'], params['max_noise_amp'], noise_layer_progress, noise_layer_status
            )
            main_audio_mono += noise_audio * params['noise_volume']
            del noise_audio
            gc.collect()


        # Applica effetti globali (glitch, pitch/time stretch, effetti dinamici avanzati)
        # Questi effetti vengono applicati all'audio mono cumulativo prima del panning finale.
        temp_audio_mono = main_audio_mono.copy() # Lavora su una copia

        # 5. Glitch Effect
        if params['enable_glitch_effect']:
            glitch_progress = st.progress(0, text="Applicazione Effetto Glitch...")
            glitch_status = st.empty()
            temp_audio_mono = _self.apply_glitch_effect(
                temp_audio_mono, variation_movement_data, 
                params['glitch_threshold'], params['glitch_duration_frames'], 
                params['glitch_intensity'], glitch_progress, glitch_status
            )
            gc.collect()

        # 6. Pitch Shifting / Time Stretching
        if params['enable_pitch_time_stretch']:
            pt_progress = st.progress(0, text="Applicazione Pitch/Time Stretch...")
            pt_status = st.empty()
            temp_audio_mono = _self.apply_pitch_time_stretch(
                temp_audio_mono, brightness_data, detail_data, 
                params['min_pitch_shift'], params['max_pitch_shift'], 
                params['min_time_stretch'], params['max_time_stretch'], pt_progress, pt_status
            )
            gc.collect()
        
        # 7. Effetti Dinamici Avanzati
        if params['enable_dynamic_effects']:
            # Filtro Dinamico
            adv_filter_progress = st.progress(0, text="Applicazione Filtro Avanzato...")
            adv_filter_status = st.empty()
            temp_audio_mono = _self.create_dynamic_biquad_filter(
                temp_audio_mono, brightness_data, detail_data, 
                params['filter_type_adv'], params['min_cutoff_adv'], params['max_cutoff_adv'], 
                params['min_resonance_adv'], params['max_resonance_adv'], adv_filter_progress, adv_filter_status
            )
            gc.collect()

            # Effetto di Modulazione
            if params['modulation_effect_type'] != "none":
                mod_progress = st.progress(0, text="Applicazione Effetto di Modulazione...")
                mod_status = st.empty()
                temp_audio_mono = _self.apply_modulation_effect(
                    temp_audio_mono, detail_data, variation_movement_data, 
                    params['modulation_effect_type'], params['modulation_intensity'], 
                    params['modulation_rate'], mod_progress, mod_status
                )
                gc.collect()

            # Dynamic Delay
            delay_progress = st.progress(0, text="Applicazione Delay Dinamico...")
            delay_status = st.empty()
            temp_audio_mono = _self.apply_dynamic_delay(
                temp_audio_mono, movement_data, detail_data, 
                params['max_delay_time'], params['max_delay_feedback'], delay_progress, delay_status
            )
            gc.collect()

            # Dynamic Reverb
            reverb_progress = st.progress(0, text="Applicazione Riverbero Dinamico...")
            reverb_status = st.empty()
            temp_audio_mono = _self.apply_dynamic_reverb(
                temp_audio_mono, brightness_data, detail_data, 
                params['max_reverb_decay'], params['max_reverb_wet'], reverb_progress, reverb_status
            )
            gc.collect()


        # Normalizzazione finale dell'audio mono (per evitare clipping)
        max_abs_val = np.max(np.abs(temp_audio_mono))
        if max_abs_val > 0:
            temp_audio_mono = temp_audio_mono / max_abs_val * 0.9 # Normalizza a 90% del range massimo
        
        # 8. Panoramica Stereo
        panning_progress = st.progress(0, text="Applicazione Panoramica Stereo...")
        panning_status = st.empty()
        final_audio_stereo = _self.apply_stereo_panning(
            temp_audio_mono, horizontal_center_data, panning_progress, panning_status
        )
        gc.collect()

        st.success("✅ Generazione audio sperimentale completata!")
        audio_progress_bar.progress(100)
        audio_status_text.text("Generazione Audio: Completata!")

        return final_audio_stereo


def generate_audio_description(
    enable_subtractive_synthesis: bool, waveform_type: str, min_cutoff: float, max_cutoff: float, min_resonance: float, max_resonance: float,
    enable_fm_synthesis: bool, min_carrier_freq: float, max_carrier_freq: float, min_modulator_freq: float, max_modulator_freq: float, min_mod_index: float, max_mod_index: float,
    enable_granular_synthesis: bool, min_grain_freq: float, max_grain_freq: float, min_grain_density: float, max_grain_density: float, min_grain_duration: float, max_grain_duration: float,
    enable_noise_effect: bool, min_noise_amp: float, max_noise_amp: float,
    enable_glitch_effect: bool, glitch_threshold: float, glitch_duration_frames: int, glitch_intensity: float,
    enable_pitch_time_stretch: bool, min_pitch_shift: float, max_pitch_shift: float, min_time_stretch: float, max_time_stretch: float,
    enable_dynamic_effects: bool, filter_type_adv: str, min_cutoff_adv: float, max_cutoff_adv: float, min_resonance_adv: float, max_resonance_adv: float,
    modulation_effect_type: str, modulation_intensity: float, modulation_rate: float,
    max_delay_time: float, max_delay_feedback: float,
    max_reverb_decay: float, max_reverb_wet: float,
    video_duration: float,  # Aggiunto per info sulla durata
    fps: float # Aggiunto per info FPS
) -> str:
    """Genera una descrizione testuale del brano audio creato in base ai parametri scelti."""
    description_parts = [
        f"Questo brano sperimentale, della durata di circa {video_duration:.1f} secondi (basato su un video a {int(fps)} FPS), è stato generato modulando parametri audio dalle caratteristiche visive del tuo video."
    ]

    # Sintesi Sottrattiva
    if enable_subtractive_synthesis:
        description_parts.append(
            f"\n\n**Sintesi Sottrattiva (Suono Base):** Un oscillatore con forma d'onda **'{waveform_type}'** costituisce la base. La **Luminosità** del video controlla dinamicamente la frequenza di taglio di un filtro (da {int(min_cutoff)}Hz a {int(max_cutoff)}Hz), mentre il **Dettaglio/Contrasto** modula la sua risonanza (da {min_resonance:.1f}Q a {max_resonance:.1f}Q), creando variazioni timbriche basate sulla chiarezza dell'immagine."
        )

    # Sintesi FM
    if enable_fm_synthesis:
        description_parts.append(
            f"\n\n**Sintesi FM (Strato Armonico):** Un ulteriore strato di sintesi a modulazione di frequenza è stato aggiunto. La **Luminosità** del video modula la frequenza del carrier (da {int(min_carrier_freq)}Hz a {int(max_carrier_freq)}Hz). Il **Movimento** del video influenza sia la frequenza del modulatore (da {int(min_modulator_freq)}Hz a {int(max_modulator_freq)}Hz) che l'indice di modulazione (da {min_mod_index:.1f} a {max_mod_index:.1f}), generando trame armoniche complesse e dinamiche."
        )

    # Sintesi Granulare
    if enable_granular_synthesis:
        description_parts.append(
            f"\n\n**Sintesi Granulare (Texture):** Questo strato crea una tessitura sonora granulare. La **Luminosità** del video controlla la frequenza dei grani (da {int(min_grain_freq)}Hz a {int(max_grain_freq)}Hz, influenzando il 'pito'). Il **Movimento** determina la densità dei grani (da {int(min_grain_density)} a {int(max_grain_density)} grani/sec), mentre il **Dettaglio/Contrasto** del video modula la durata dei singoli grani (da {min_grain_duration*1000:.1f}ms a {max_grain_duration*1000:.1f}ms, per effetti 'drone')."
        )
    
    # Rumore (Noise)
    if enable_noise_effect:
        description_parts.append(
            f"\n\n**Strato di Rumore (Noise):** Un layer di rumore bianco è stato aggiunto, con la sua ampiezza (volume) controllata dal **Dettaglio/Contrasto** del video (da {min_noise_amp:.2f} a {max_noise_amp:.2f}). Maggiore è il dettaglio, più presente sarà il rumore."
        )

    # Glitch Audio
    if enable_glitch_effect:
        description_parts.append(
            f"\n\n**Effetti Glitch:** Picchi di **Variazione del Movimento** nel video (sopra la soglia di {glitch_threshold:.2f}) innescano effetti di 'glitch' nell'audio. Questi glitch durano circa {glitch_duration_frames} frame e hanno un'intensità massima di {glitch_intensity:.1f}, creando interruzioni e ripetizioni sonore."
        )

    # Pitch Shifting / Time Stretching
    if enable_pitch_time_stretch:
        description_parts.append(
            f"\n\n**Pitch Shifting e Time Stretching:** L'audio viene dinamicamente processato: il **Dettaglio/Contrasto** modula lo spostamento di tonalità (pitch shift, da {min_pitch_shift:.1f} a {max_pitch_shift:.1f} semitoni), mentre la **Luminosità** influenza la velocità di riproduzione (time stretch, da {min_time_stretch:.1f}x a {max_time_stretch:.1f}x), creando alterazioni temporali e tonali."
        )
    
    # Effetti Dinamici Avanzati
    if enable_dynamic_effects:
        description_parts.append("\n\n**Effetti Sonori Dinamici:**")
        description_parts.append(f"- **Filtro Avanzato ({filter_type_adv.capitalize()}):** La **Luminosità** e il **Dettaglio** del video modulano la frequenza di taglio (da {int(min_cutoff_adv)}Hz a {int(max_cutoff_adv)}Hz) e la risonanza (da {min_resonance_adv:.1f}Q a {max_resonance_adv:.1f}Q) di un filtro dinamico.")
        
        if modulation_effect_type != "none":
            description_parts.append(f"- **Effetto di Modulazione ({modulation_effect_type.capitalize()}):** L'intensità di questo effetto è controllata dalla **Variazione del Movimento**, e la velocità di modulazione dal **Dettaglio/Contrasto**, per un suono in costante mutamento.")
        
        description_parts.append(f"- **Delay Dinamico:** Il **Movimento** del video influenza il tempo di ritardo (fino a {max_delay_time:.2f}s), mentre il **Dettaglio/Contrasto** modula il feedback (fino a {max_delay_feedback:.2f}), creando effetti di eco variabili.")
        description_parts.append(f"- **Riverbero Dinamico:** Il **Dettaglio/Contrasto** del video controlla il tempo di decadimento del riverbero (fino a {max_reverb_decay:.1f}s), e la **Luminosità** regola il mix tra segnale pulito e riverberato (wet/dry, fino a {max_reverb_wet:.2f}), simulando ambienti acustici che cambiano con la scena.")

    description_parts.append(
        "\n\nLa **Panoramica Stereo** (spostamento del suono tra sinistra e destra) è guidata dal **Centro di Massa Orizzontale** degli elementi visivi nel video, creando un'esperienza spaziale dinamica."
    )

    return "\n".join(description_parts)

def main():
    st.set_page_config(
        page_title="Video to Experimental Audio",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📹🎶 Video to Experimental Audio Generator")
    st.write("Carica un video per generare un'esperienza sonora sperimentale basata sulle sue caratteristiche visive (luminosità, movimento, dettaglio, ecc.).")

    if not check_ffmpeg():
        st.error("🚨 FFmpeg non trovato! Assicurati che FFmpeg sia installato e configurato nel PATH del tuo sistema. Senza FFmpeg, non sarà possibile unire l'audio al video.")
        st.info("Per Windows: scarica da [ffmpeg.org](https://ffmpeg.org/download.html) e aggiungi la cartella `bin` al PATH di sistema.")
        st.info("Per macOS: `brew install ffmpeg` (con Homebrew)")
        st.info("Per Linux: `sudo apt update && sudo apt install ffmpeg`")

    st.sidebar.header("Parametri di Generazione Audio")

    # Colonna per il caricamento e l'anteprima
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        uploaded_file = st.file_uploader("Carica il tuo file video (Max 50MB)", type=["mp4", "mov", "avi", "mkv", "webm"])

    video_input_path = None
    if uploaded_file is not None:
        if not validate_video_file(uploaded_file):
            uploaded_file = None # Resetta il file se non valido
        else:
            base_name = os.path.splitext(uploaded_file.name)[0]
            video_input_path = os.path.join("/tmp", uploaded_file.name) # Usa /tmp per file temporanei
            with open(video_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Video '{uploaded_file.name}' caricato con successo.")

            with col_preview:
                st.video(video_input_path)

    # Espandi la sezione dei parametri in sidebar
    with st.sidebar:
        st.subheader("Controlli Globali")
        global_volume = st.slider("Volume Generale Audio", 0.0, 2.0, 1.0, 0.1)

        st.subheader("Sintesi Sottrattiva (Suono Base)")
        enable_subtractive_synthesis = st.checkbox("Abilita Sintesi Sottrattiva", value=True)
        if enable_subtractive_synthesis:
            subtractive_volume = st.slider("Volume Sottrattiva", 0.0, 1.0, 0.7, 0.05)
            waveform_type_user = st.selectbox("Forma d'Onda Base", ["sawtooth", "sine", "square", "triangle"])
            subtractive_filter_type = st.selectbox("Tipo Filtro Sottrattiva", ["lowpass", "highpass", "bandpass"])
            min_cutoff_user = st.slider("Frequenza Taglio Min (Hz)", 20, 5000, 50, 10)
            max_cutoff_user = st.slider("Frequenza Taglio Max (Hz)", 20, 10000, 2000, 10)
            min_resonance_user = st.slider("Risonanza Min (Q)", 0.1, 10.0, 0.7, 0.1)
            max_resonance_user = st.slider("Risonanza Max (Q)", 0.1, 20.0, 5.0, 0.1)
        else:
            subtractive_volume, waveform_type_user, subtractive_filter_type, min_cutoff_user, max_cutoff_user, min_resonance_user, max_resonance_user = (0.0, "sawtooth", "lowpass", 20, 2000, 0.7, 5.0)

        st.subheader("Sintesi FM (Strato Armonico)")
        enable_fm_synthesis = st.checkbox("Abilita Sintesi FM", value=False)
        if enable_fm_synthesis:
            fm_volume = st.slider("Volume FM", 0.0, 1.0, 0.5, 0.05)
            min_carrier_freq_user = st.slider("Frequenza Carrier Min (Hz)", 50, 1000, 200, 10)
            max_carrier_freq_user = st.slider("Frequenza Carrier Max (Hz)", 100, 5000, 800, 10)
            min_modulator_freq_user = st.slider("Frequenza Modulatore Min (Hz)", 10, 500, 50, 5)
            max_modulator_freq_user = st.slider("Frequenza Modulatore Max (Hz)", 50, 2000, 300, 5)
            min_mod_index_user = st.slider("Indice Modulazione Min", 0.1, 10.0, 1.0, 0.1)
            max_mod_index_user = st.slider("Indice Modulazione Max", 1.0, 20.0, 5.0, 0.1)
        else:
            fm_volume, min_carrier_freq_user, max_carrier_freq_user, min_modulator_freq_user, max_modulator_freq_user, min_mod_index_user, max_mod_index_user = (0.0, 200, 800, 50, 300, 1.0, 5.0)

        st.subheader("Sintesi Granulare (Texture)")
        enable_granular_synthesis = st.checkbox("Abilita Sintesi Granulare", value=False)
        if enable_granular_synthesis:
            granular_volume = st.slider("Volume Granulare", 0.0, 1.0, 0.4, 0.05)
            min_grain_freq = st.slider("Frequenza Grano Min (Hz)", 50, 2000, 100, 10)
            max_grain_freq = st.slider("Frequenza Grano Max (Hz)", 200, 5000, 1000, 10)
            min_grain_density = st.slider("Densità Grani Min (grani/sec)", 1, 100, 10, 1)
            max_grain_density = st.slider("Densità Grani Max (grani/sec)", 10, 500, 50, 5)
            min_grain_duration = st.slider("Durata Grano Min (sec)", 0.01, 0.5, 0.05, 0.01)
            max_grain_duration = st.slider("Durata Grano Max (sec)", 0.05, 2.0, 0.2, 0.01)
        else:
            granular_volume, min_grain_freq, max_grain_freq, min_grain_density, max_grain_density, min_grain_duration, max_grain_duration = (0.0, 100, 1000, 10, 50, 0.05, 0.2)
        
        st.subheader("Strato di Rumore")
        enable_noise_effect = st.checkbox("Abilita Strato di Rumore", value=False)
        if enable_noise_effect:
            noise_volume = st.slider("Volume Rumore", 0.0, 1.0, 0.3, 0.05)
            min_noise_amp_user = st.slider("Ampiezza Rumore Min", 0.0, 0.5, 0.1, 0.01)
            max_noise_amp_user = st.slider("Ampiezza Rumore Max", 0.1, 1.0, 0.4, 0.01)
        else:
            noise_volume, min_noise_amp_user, max_noise_amp_user = (0.0, 0.1, 0.4)

        st.subheader("Effetti Glitch")
        enable_glitch_effect = st.checkbox("Abilita Effetto Glitch", value=False)
        if enable_glitch_effect:
            glitch_threshold_user = st.slider("Soglia Glitch (Variazione Mov.)", 0.0, 1.0, 0.5, 0.05)
            glitch_duration_frames_user = st.slider("Durata Glitch (Frame)", 1, 10, 3, 1)
            glitch_intensity_user = st.slider("Intensità Glitch", 0.1, 5.0, 2.0, 0.1)
        else:
            glitch_threshold_user, glitch_duration_frames_user, glitch_intensity_user = (0.5, 3, 2.0)

        st.subheader("Pitch Shifting / Time Stretching")
        enable_pitch_time_stretch = st.checkbox("Abilita Pitch/Time Stretch", value=False)
        if enable_pitch_time_stretch:
            min_pitch_shift_semitones = st.slider("Pitch Shift Min (semitoni)", -12.0, 12.0, 0.0, 0.1)
            max_pitch_shift_semitones = st.slider("Pitch Shift Max (semitoni)", -12.0, 12.0, 5.0, 0.1)
            min_time_stretch_rate = st.slider("Time Stretch Min (Velocità)", 0.1, 2.0, 1.0, 0.05)
            max_time_stretch_rate = st.slider("Time Stretch Max (Velocità)", 0.5, 5.0, 2.0, 0.05)
        else:
            min_pitch_shift_semitones, max_pitch_shift_semitones, min_time_stretch_rate, max_time_stretch_rate = (0.0, 0.0, 1.0, 1.0)

        st.subheader("Effetti Dinamici Avanzati")
        enable_dynamic_effects = st.checkbox("Abilita Effetti Dinamici Avanzati", value=False)
        if enable_dynamic_effects:
            st.markdown("---")
            st.subheader("Filtro Avanzato")
            filter_type_user = st.selectbox("Tipo Filtro Avanzato", ["lowpass", "highpass", "bandpass"])
            min_cutoff_adv = st.slider("Frequenza Taglio Min (Avanzato)", 20, 5000, 100, 10, key="adv_min_cutoff")
            max_cutoff_adv = st.slider("Frequenza Taglio Max (Avanzato)", 20, 10000, 4000, 10, key="adv_max_cutoff")
            min_resonance_adv = st.slider("Risonanza Min (Avanzato)", 0.1, 10.0, 1.0, 0.1, key="adv_min_res")
            max_resonance_adv = st.slider("Risonanza Max (Avanzato)", 0.1, 20.0, 8.0, 0.1, key="adv_max_res")

            st.markdown("---")
            st.subheader("Effetto di Modulazione")
            modulation_effect_type = st.selectbox("Tipo Effetto Modulazione", ["none", "tremolo", "vibrato", "phaser"])
            if modulation_effect_type != "none":
                modulation_intensity = st.slider("Intensità Modulazione", 0.0, 1.0, 0.5, 0.05)
                modulation_rate = st.slider("Velocità Modulazione (Hz)", 1.0, 20.0, 10.0, 0.1)
            else:
                modulation_intensity, modulation_rate = (0.0, 0.0)

            st.markdown("---")
            st.subheader("Delay Dinamico")
            max_delay_time_user = st.slider("Tempo Max Delay (sec)", 0.0, 2.0, 0.5, 0.05)
            max_delay_feedback_user = st.slider("Feedback Max Delay", 0.0, 0.95, 0.7, 0.05)

            st.markdown("---")
            st.subheader("Riverbero Dinamico")
            max_reverb_decay_user = st.slider("Decadimento Max Riverbero (sec)", 0.0, 5.0, 1.5, 0.1)
            max_reverb_wet_user = st.slider("Wet Level Max Riverbero", 0.0, 1.0, 0.4, 0.05)
        else:
            filter_type_user, min_cutoff_adv, max_cutoff_adv, min_resonance_adv, max_resonance_adv = ("lowpass", 20, 2000, 0.7, 5.0)
            modulation_effect_type, modulation_intensity, modulation_rate = ("none", 0.0, 0.0)
            max_delay_time_user, max_delay_feedback_user = (0.0, 0.0)
            max_reverb_decay_user, max_reverb_wet_user = (0.0, 0.0)
    
    st.markdown("---")
    st.subheader("Opzioni di Esportazione")
    output_resolution_choice = st.selectbox(
        "Scegli la risoluzione del video di output (seleziona 'Originale' per mantenere)",
        list(FORMAT_RESOLUTIONS.keys())
    )
    
    # Bottone di generazione
    if st.button("🚀 Genera Audio da Video"):
        if video_input_path is None:
            st.error("⚠️ Per favore, carica un file video prima di generare.")
        else:
            progress_bar = st.progress(0, text="Analisi Video...")
            status_text = st.empty()

            brightness_data, detail_data, movement_data, variation_movement_data, horizontal_center_data, width, height, fps, video_duration = \
                analyze_video_frames(video_input_path, progress_bar, status_text)

            if brightness_data is None: # Se l'analisi video fallisce o il video è troppo corto
                st.error("❌ Impossibile procedere con la generazione audio a causa di problemi con l'analisi del video o durata non valida.")
                if os.path.exists(video_input_path):
                    os.remove(video_input_path)
                return

            # Prepara i parametri per la funzione generate_audio
            audio_gen_params = {
                'enable_subtractive_synthesis': enable_subtractive_synthesis,
                'subtractive_volume': subtractive_volume,
                'waveform_type': waveform_type_user,
                'subtractive_filter_type': subtractive_filter_type,
                'min_cutoff': min_cutoff_user,
                'max_cutoff': max_cutoff_user,
                'min_resonance': min_resonance_user,
                'max_resonance': max_resonance_user,

                'enable_fm_synthesis': enable_fm_synthesis,
                'fm_volume': fm_volume,
                'min_carrier_freq': min_carrier_freq_user,
                'max_carrier_freq': max_carrier_freq_user,
                'min_modulator_freq': min_modulator_freq_user,
                'max_modulator_freq': max_modulator_freq_user,
                'min_mod_index': min_mod_index_user,
                'max_mod_index': max_mod_index_user,

                'enable_granular_synthesis': enable_granular_synthesis,
                'granular_volume': granular_volume,
                'min_grain_freq': min_grain_freq,
                'max_grain_freq': max_grain_freq,
                'min_grain_density': min_grain_density,
                'max_grain_density': max_grain_density,
                'min_grain_duration': min_grain_duration,
                'max_grain_duration': max_grain_duration,

                'enable_noise_effect': enable_noise_effect,
                'noise_volume': noise_volume,
                'min_noise_amp': min_noise_amp_user,
                'max_noise_amp': max_noise_amp_user,

                'enable_glitch_effect': enable_glitch_effect,
                'glitch_threshold': glitch_threshold_user,
                'glitch_duration_frames': glitch_duration_frames_user,
                'glitch_intensity': glitch_intensity_user,

                'enable_pitch_time_stretch': enable_pitch_time_stretch,
                'min_pitch_shift': min_pitch_shift_semitones,
                'max_pitch_shift': max_pitch_shift_semitones,
                'min_time_stretch': min_time_stretch_rate,
                'max_time_stretch': max_time_stretch_rate,
                
                'enable_dynamic_effects': enable_dynamic_effects,
                'filter_type_adv': filter_type_user,
                'min_cutoff_adv': min_cutoff_adv,
                'max_cutoff_adv': max_cutoff_adv,
                'min_resonance_adv': min_resonance_adv,
                'max_resonance_adv': max_resonance_adv,
                'modulation_effect_type': modulation_effect_type,
                'modulation_intensity': modulation_intensity,
                'modulation_rate': modulation_rate,
                'max_delay_time': max_delay_time_user,
                'max_delay_feedback': max_delay_feedback_user,
                'max_reverb_decay': max_reverb_decay_user,
                'max_reverb_wet': max_reverb_wet_user,
            }

            audio_generator = AudioGenerator(AUDIO_SAMPLE_RATE, AUDIO_FPS)
            final_audio_stereo = audio_generator.generate_audio(
                video_duration, brightness_data, detail_data, movement_data, 
                variation_movement_data, horizontal_center_data, audio_gen_params
            ) * global_volume
            
            # Percorsi temporanei per l'output audio
            base_name_output = os.path.splitext(os.path.basename(uploaded_file.name))[0]
            audio_output_path = os.path.join("/tmp", f"generated_audio_{base_name_output}.wav")
            
            try:
                # Salva l'audio stereo
                sf.write(audio_output_path, final_audio_stereo, AUDIO_SAMPLE_RATE)
                st.success(f"✅ Audio sperimentale processato e salvato in '{audio_output_path}'")
            except Exception as e:
                st.error(f"❌ Errore nel salvataggio dell'audio WAV: {str(e)}")
                return
            
            st.markdown("---")
            st.subheader("📝 Descrizione del Brano Generato")
            
            # Chiamata alla funzione per generare la descrizione
            generated_description = generate_audio_description(
                enable_subtractive_synthesis, waveform_type_user, min_cutoff_user, max_cutoff_user, min_resonance_user, max_resonance_user,
                enable_fm_synthesis, min_carrier_freq_user, max_carrier_freq_user, min_modulator_freq_user, max_modulator_freq_user, min_mod_index_user, max_mod_index_user,
                enable_granular_synthesis, min_grain_freq, max_grain_freq, min_grain_density, max_grain_density, min_grain_duration, max_grain_duration,
                enable_noise_effect, min_noise_amp_user, max_noise_amp_user,
                enable_glitch_effect, glitch_threshold_user, glitch_duration_frames_user, glitch_intensity_user,
                enable_pitch_time_stretch, min_pitch_shift_semitones, max_pitch_shift_semitones, min_time_stretch_rate, max_time_stretch_rate,
                enable_dynamic_effects, filter_type_user, min_cutoff_adv, max_cutoff_adv, min_resonance_adv, max_resonance_adv,
                modulation_effect_type, modulation_intensity, modulation_rate,
                max_delay_time_user, max_delay_feedback_user,
                max_reverb_decay_user, max_reverb_wet_user,
                video_duration, # Passa la durata del video
                fps # Passa gli FPS del video
            )
            
            st.markdown(generated_description) # Usa markdown per formattare la descrizione
            st.markdown("---") # Separatore per il download


            # Processo di unione video e audio
            st.subheader("🎬 Esportazione Video con Audio")
            if check_ffmpeg():
                final_video_path = os.path.join("/tmp", f"final_video_{base_name_output}.mp4")
                
                target_width, target_height = FORMAT_RESOLUTIONS[output_resolution_choice]

                ffmpeg_command = [
                    "ffmpeg",
                    "-y", # Sovrascrivi file di output senza chiedere
                    "-i", video_input_path,
                    "-i", audio_output_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest", # Termina quando il video o l'audio più corto finisce
                ]

                if target_width > 0 and target_height > 0:
                    ffmpeg_command.extend(["-vf", f"scale={target_width}:{target_height},setsar=1:1"])
                else: # Mantieni risoluzione originale ma assicurati che sia compatibile (es. pari)
                    ffmpeg_command.extend(["-vf", "scale='iw-mod(iw,2)':'ih-mod(ih,2)',setsar=1:1"])
                
                ffmpeg_command.append(final_video_path)

                ffmpeg_progress = st.progress(0, text="Unione Video e Audio (FFmpeg)...")
                ffmpeg_status = st.empty()

                try:
                    # Esegui FFmpeg in un sottoprocesso
                    # Non possiamo mostrare un progresso preciso da FFmpeg senza parsing complesso del suo stderr
                    # Quindi, un progresso simulato o un messaggio di attesa
                    ffmpeg_status.text("⏳ Unione video e audio in corso, potrebbe richiedere tempo...")
                    subprocess.run(ffmpeg_command, check=True, capture_output=True)
                    ffmpeg_progress.progress(100)
                    ffmpeg_status.text("✅ Unione Video e Audio completata!")

                    st.success(f"🎉 Video finale con audio generato con successo!")
                    
                    with open(final_video_path, "rb") as f:
                        st.download_button(
                            "⬇️ Scarica Video con Audio (MP4)", 
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
                # Lasciamo qui l'opzione di scaricare solo l'audio come fallback se ffmpeg non è installato
                with open(audio_output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica Solo Audio (WAV temporaneo)",
                        f,
                        file_name=f"videosound_generato_audio_{base_name_output}.wav",
                        mime="audio/wav"
                    )


if __name__ == "__main__":
    main()
