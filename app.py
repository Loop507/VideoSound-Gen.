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
import time # Importa il modulo time

# --- (Mantieni tutte le costanti globali e le definizioni delle funzioni helper come check_ffmpeg, validate_video_file) ---
# --- (Mantieni l'intera classe AudioGenerator, è fondamentale per la generazione audio) ---

# Rimuovi analyze_live_frames se non la stai usando, o adattala se vuoi aggiungere la webcam in futuro.
# Per ora, ci concentriamo su "Carica Video" e "Solo Audio (Parametri Manuali)".

# La funzione analyze_video_frames (che già hai) sarà usata quando si carica un video.

def main():
    st.set_page_config(page_title="🎵 VideoSound Gen - Sperimentale", layout="centered")
    st.title("🎬 VideoSound Gen - Sperimentale")
    st.markdown("###### by Loop507")
    st.markdown("### Genera musica sperimentale da un video o da parametri manuali!")

    # --- NUOVA SEZIONE: Scelta della Sorgente di Input ---
    input_mode = st.radio(
        "Seleziona la sorgente per la generazione del suono:",
        ("Carica un Video", "Genera Suono Senza Video (Parametri Manuali)"),
        index=0, # "Carica un Video" è l'opzione predefinita
        help="Scegli se analizzare un video per generare il suono, o se creare un suono basato su parametri che imposterai manualmente."
    )

    uploaded_file = None
    video_input_path = None

    # Inizializzazione delle variabili che contengono i dati di modulazione
    brightness_data = None
    detail_data = None
    movement_data = None
    variation_movement_data = None
    horizontal_center_data = None

    # Inizializzazione delle informazioni sul "video" (virtuale o reale)
    width, height, fps, video_duration = 0, 0, 0.0, 0.0

    # --- LOGICA IN BASE ALLA SCELTA DELL'UTENTE ---
    if input_mode == "Carica un Video":
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

            analysis_progress_bar = st.progress(0)
            analysis_status_text = st.empty()

            with st.spinner("📊 Analisi frame video (luminosità, dettaglio, movimento, variazione movimento, centro orizzontale) in corso..."):
                brightness_data, detail_data, movement_data, variation_movement_data, horizontal_center_data, width, height, fps, video_duration = analyze_video_frames(video_input_path, analysis_progress_bar, analysis_status_text)

            if brightness_data is None:
                return # Si è verificato un errore nell'analisi del video

            st.info(f"🎥 Durata video: {video_duration:.2f} secondi | Risoluzione Originale: {width}x{height} | FPS: {fps:.2f}")
        else:
            st.info("Carica un video per iniziare o cambia modalità.")
            return # Ferma l'esecuzione se non c'è input video

    elif input_mode == "Genera Suono Senza Video (Parametri Manuali)":
        st.info("ℹ️ Generazione audio basata su parametri manuali. Regola i controlli qui sotto per modellare il tuo suono.")

        st.subheader("⚙️ Configurazione Parametri 'Visivi' Virtuali")
        # Permetti all'utente di definire la durata dell'audio
        virtual_duration = st.slider("Durata Audio Generato (secondi)", 5, MAX_DURATION, 30, help="Definisci per quanto tempo l'audio verrà generato.")
        
        # Definisci i "frames per secondo" virtuali (possono essere gli stessi di AUDIO_FPS)
        virtual_fps = AUDIO_FPS
        num_virtual_frames = int(virtual_duration * virtual_fps)

        # Controlli per i "dati visivi" virtuali
        st.markdown("---")
        st.markdown("**Andamento Luminosità (controlla cutoff filtro, carrier FM, pitch, ecc.)**")
        start_brightness = st.slider("Luminosità Iniziale (0.0=scuro, 1.0=chiaro)", 0.0, 1.0, 0.2, 0.05, key="sb")
        end_brightness = st.slider("Luminosità Finale (0.0=scuro, 1.0=chiaro)", 0.0, 1.0, 0.8, 0.05, key="eb")
        brightness_data = np.linspace(start_brightness, end_brightness, num_virtual_frames)

        st.markdown("---")
        st.markdown("**Andamento Dettaglio/Contrasto (controlla risonanza, modulator FM, noise, ecc.)**")
        start_detail = st.slider("Dettaglio Iniziale (0.0=sfocato, 1.0=nitido)", 0.0, 1.0, 0.8, 0.05, key="sd")
        end_detail = st.slider("Dettaglio Finale (0.0=sfocato, 1.0=nitido)", 0.0, 1.0, 0.2, 0.05, key="ed")
        detail_data = np.linspace(start_detail, end_detail, num_virtual_frames)

        st.markdown("---")
        st.markdown("**Andamento Movimento (controlla glitch, delay, densità granulare, ecc.)**")
        base_movement = np.linspace(0.1, 0.5, num_virtual_frames) # Movimento di base crescente
        random_movement_factor = st.slider("Fattore di Movimento Casuale", 0.0, 1.0, 0.2, 0.05, help="Aggiunge casualità all'andamento del movimento.")
        movement_data = base_movement + (np.random.rand(num_virtual_frames) - 0.5) * random_movement_factor
        movement_data = np.clip(movement_data, 0.0, 1.0) # Assicura che i valori rimangano tra 0 e 1
        variation_movement_data = np.abs(np.diff(movement_data, prepend=movement_data[0]))
        
        st.markdown("---")
        st.markdown("**Andamento Panning Orizzontale (controlla il bilanciamento stereo)**")
        start_pan = st.slider("Panning Iniziale (0.0=sinistra, 0.5=centro, 1.0=destra)", 0.0, 1.0, 0.5, 0.05, key="sp")
        end_pan = st.slider("Panning Finale (0.0=sinistra, 0.5=centro, 1.0=destra)", 0.0, 1.0, 0.5, 0.05, key="ep")
        horizontal_center_data = np.linspace(start_pan, end_pan, num_virtual_frames)

        # Imposta i valori per il resto del codice, come se venissero da un video
        video_duration = float(virtual_duration)
        fps = float(virtual_fps)
        width, height = 1280, 720 # Valori placeholder, non usati per la generazione audio

        st.success(f"✅ Dati 'visivi' virtuali generati per {virtual_duration} secondi di audio.")

    # Se a questo punto i dati non sono stati inizializzati, significa che non è stato caricato un video
    # e non è stata scelta la modalità manuale (o l'utente non ha ancora interagito).
    # In tal caso, fermiamo l'esecuzione.
    if brightness_data is None:
        return
        
    st.markdown("---")
    st.subheader("🎶 Configurazione Sintesi Audio Sperimentale")

    # --- (Qui continua il resto del tuo codice main() con tutti i controlli degli effetti e slider) ---
    # Assicurati che tutti i parametri degli effetti (min_cutoff_user, max_cutoff_user, etc.)
    # siano inizializzati prima di essere usati nei blocchi if st.button("Genera...").
    # L'inizializzazione che avevi già dovrebbe essere sufficiente, ad esempio:
    min_cutoff_user, max_cutoff_user = 0, 0
    min_resonance_user, max_resonance_user = 0, 0
    waveform_type_user = "sawtooth"
    filter_type_user = "lowpass"
    min_carrier_freq_user, max_carrier_freq_user = 0, 0
    min_modulator_freq_user, max_modulator_freq_user = 0, 0
    min_mod_index_user, max_mod_index_user = 0, 0
    min_noise_amp_user, max_noise_amp_user = 0, 0
    glitch_threshold_user, glitch_duration_frames_user, glitch_intensity_user = 0, 0, 0
    min_pitch_shift_semitones = 0
    max_pitch_shift_semitones = 0
    min_time_stretch_rate, max_time_stretch_rate = 0, 0
    min_grain_freq, max_grain_freq = 0, 0
    min_grain_density, max_grain_density = 0, 0
    min_grain_duration, max_grain_duration = 0, 0
    max_delay_time_user, max_delay_feedback_user = 0, 0
    max_reverb_decay_user, max_reverb_wet_user = 0, 0
    modulation_effect_type = "none"
    modulation_intensity = 0.5
    modulation_rate = 0.1

    min_cutoff_adv = 20
    max_cutoff_adv = 20000
    min_resonance_adv = 0.1
    max_resonance_adv = 30.0

    # ... (il resto della configurazione UI dei tuoi effetti in sidebar) ...
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
        st.markdown("- **Frequenza di Taglio:** controllata dalla **Luminosità** del video/input.")
        st.markdown("- **Risonanza:** controllata dal **Dettaglio/Contrasto** del video/input.")
        min_cutoff_user = st.slider("Min Frequenza Taglio (Hz)", 20, 5000, 100, key="sub_min_cutoff", disabled=not enable_subtractive_synthesis)
        max_cutoff_user = st.slider("Max Frequenza Taglio (Hz)", 1000, 20000, 8000, key="sub_max_cutoff", disabled=not enable_subtractive_synthesis)
        min_resonance_user = st.slider("Min Risonanza (Q)", 0.1, 5.0, 0.5, key="sub_min_res", disabled=not enable_subtractive_synthesis)
        max_resonance_user = st.slider("Max Risonanza (Q)", 1.0, 30.0, 10.0, key="sub_max_res", disabled=not enable_subtractive_synthesis)

    # ... (continua con le sezioni FM, Granular, Noise, Glitch, Pitch/Time, Dynamic Effects) ...
    # Ricorda di aggiornare i testi nelle descrizioni per riferirsi a "video/input" invece che solo "video".
    # Esempio: "controllata dalla **Luminosità** del video/input."

    st.markdown("---")
    st.subheader("⬇️ Opzioni di Download")

    output_resolution_choice = "Originale" # Default, sarà sovrascritto
    download_option_choices = ["Solo Audio"] # Default, se non c'è video

    if input_mode == "Carica un Video":
        output_resolution_choice = st.selectbox(
            "Seleziona la risoluzione di output del video:",
            list(FORMAT_RESOLUTIONS.keys())
        )
        download_option_choices = ["Video con Audio", "Solo Audio"]
    elif input_mode == "Genera Suono Senza Video (Parametri Manuali)":
        st.warning("⚠️ Stai generando solo audio. Le opzioni video non sono disponibili in questa modalità.")
        output_resolution_choice = "Originale" # Non usato, ma per coerenza
        download_option_choices = ["Solo Audio"]

    download_option = st.radio(
        "Cosa vuoi scaricare?",
        download_option_choices
    )

    if not check_ffmpeg():
        st.warning("⚠️ FFmpeg non disponibile sul tuo sistema. L'unione o la ricodifica del video potrebbe non funzionare. Assicurati che FFmpeg sia installato e nel PATH.")

    if st.button("🎵 Genera e Prepara Download"):
        
        base_name_output = "generated_sound"
        if input_mode == "Carica un Video" and uploaded_file is not None:
            base_name_output = os.path.splitext(uploaded_file.name)[0]
        elif input_mode == "Genera Suono Senza Video (Parametri Manuali)":
            base_name_output = "manual_params_sound" # Nome specifico per output manuale

        unique_id_audio = str(np.random.randint(10000, 99999))
        audio_output_path = os.path.join("temp", f"{base_name_output}_{unique_id_audio}_generated_audio.wav")
        final_video_path = os.path.join("temp", f"{base_name_output}_{unique_id_audio}_final_videosound.mp4")

        os.makedirs("temp", exist_ok=True)

        audio_gen = AudioGenerator(AUDIO_SAMPLE_RATE, int(fps))

        total_samples = int(video_duration * AUDIO_SAMPLE_RATE)

        audio_progress_bar = st.progress(0)
        audio_status_text = st.empty()

        # --- (Qui continua il tuo codice per la generazione e combinazione degli strati audio) ---
        # Questo blocco rimane lo stesso perché le variabili _data sono già state popolate
        # in base alla scelta dell'utente (o dal video, o dai parametri manuali).

        # --- Generazione dell'audio base (Sintesi Sottrattiva) ---
        if enable_subtractive_synthesis:
            st.info("🎵 Generazione dell'onda base (Sintesi Sottrattiva)...")
            subtractive_layer = audio_gen.generate_subtractive_waveform(total_samples, waveform_type_user)
            combined_audio_layers = subtractive_layer
            st.success("✅ Strato Sintesi Sottrattiva generato!")
        else:
            combined_audio_layers = np.zeros(total_samples, dtype=np.float32) # Assicura che sia inizializzato
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
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            combined_audio_layers += fm_layer * 0.5
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
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            combined_audio_layers += granular_layer * 0.5
            st.success("✅ Strato Granulare generato e combinato!")

        # --- Aggiungi il layer di Rumore se abilitato ---
        if enable_noise_effect:
            combined_audio_layers = audio_gen.add_noise_layer(
                combined_audio_layers,
                detail_data,
                min_noise_amp_user,
                max_noise_amp_user,
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
            )
            st.success("✅ Strato Rumore aggiunto!")

        # --- Applica effetti Glitch se abilitati ---
        if enable_glitch_effect:
            combined_audio_layers = audio_gen.apply_glitch_effect(
                combined_audio_layers,
                variation_movement_data,
                glitch_threshold_user,
                glitch_duration_frames_user,
                glitch_intensity_user,
                progress_bar=audio_progress_bar,
                status_text=audio_status_text
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
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text
                )
                st.success("✅ Effetti pitch e stretch applicati!")
            else:
                st.info("🎶 Pitch Shifting / Time Stretching disabilitato.")


            # --- Applicazione Effetti Sonori Dinamici unificati ---
            if enable_dynamic_effects:
                # Applica Filtro Avanzato
                processed_audio = audio_gen.apply_dynamic_filter(
                    processed_audio,
                    brightness_data,
                    detail_data,
                    min_cutoff=min_cutoff_adv,
                    max_cutoff=max_cutoff_adv,
                    min_res=min_resonance_adv,
                    max_res=max_resonance_adv,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text,
                    filter_type=filter_type_user
                )
                st.success("✅ Filtri avanzati applicati!")

                # Applica Effetto di Modulazione (Chorus/Flanger/Phaser)
                processed_audio = audio_gen.apply_modulation_effect(
                    processed_audio,
                    variation_movement_data,
                    detail_data,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text,
                    effect_type=modulation_effect_type,
                    intensity=modulation_intensity,
                    rate=modulation_rate
                )
                st.success(f"✅ Effetto di Modulazione '{modulation_effect_type}' applicato!")

                # Applica Delay
                processed_audio = audio_gen.apply_delay_effect(
                    processed_audio,
                    movement_data,
                    detail_data,
                    max_delay_time_user,
                    max_delay_feedback_user,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text
                )
                st.success("✅ Effetto Delay applicato!")

                # Applica Riverbero
                processed_audio = audio_gen.apply_reverb_effect(
                    processed_audio,
                    detail_data,
                    brightness_data,
                    max_reverb_decay_user,
                    max_reverb_wet_user,
                    progress_bar=audio_progress_bar,
                    status_text=audio_status_text
                )
                st.success("✅ Effetto Riverbero applicato!")
            else:
                st.info("🎶 Effetti Dinamici Avanzati disabilitati.")
            
            # --- Panning Stereo ---
            processed_audio_stereo = audio_gen.apply_stereo_panning(processed_audio, horizontal_center_data)
            st.success("✅ Panning stereo applicato!")

            # --- Normalizzazione finale ---
            final_audio = audio_gen.normalize_audio(processed_audio_stereo)
            st.success("✅ Audio normalizzato!")

        # --- Salvataggio audio ---
        st.info(f"💾 Salvataggio audio in '{audio_output_path}'...")
        try:
            sf.write(audio_output_path, final_audio, AUDIO_SAMPLE_RATE)
            st.success("✅ Audio salvato correttamente!")
        except Exception as e:
            st.error(f"❌ Errore durante il salvataggio dell'audio: {e}")
            return

        # --- Sezione Download ---
        if download_option == "Solo Audio":
            with open(audio_output_path, "rb") as f:
                st.download_button(
                    "⬇️ Scarica Solo Audio (WAV)",
                    f,
                    file_name=f"videosound_generato_audio_{base_name_output}.wav",
                    mime="audio/wav"
                )
            # Pulizia file temporanei
            if os.path.exists(audio_output_path):
                os.remove(audio_output_path)
            if video_input_path and os.path.exists(video_input_path): # Pulisci video input solo se esiste
                os.remove(video_input_path)
            st.info("🗑️ File temporanei puliti.")

        elif download_option == "Video con Audio" and input_mode == "Carica un Video": # Questa opzione è disponibile solo con un video caricato
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
        elif download_option == "Video con Audio" and input_mode == "Genera Suono Senza Video (Parametri Manuali)":
            st.error("❌ Non puoi scaricare un 'Video con Audio' se hai scelto di generare il suono senza caricare un video. Seleziona 'Solo Audio'.")

if __name__ == "__main__":
    main()
