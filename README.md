# Ambisonic Visualizer

Aplikacja desktopowa do wizualizacji nagrań ambisonicznych (WAV) w czasie rzeczywistym.
Łączy odtwarzanie audio z podglądem wideo/obrazu i interaktywną, obracaną
wizualizacją energii pola dźwiękowego.

## Co to jest i jak działa?

Program wczytuje plik WAV z wielokanałowym sygnałem ambisonicznym (1. – 3. rząd).
Podczas odtwarzania:

- **Audio** jest dekodowane i analizowane w czasie rzeczywistym, aby wyliczyć
  rozkład energii dźwięku wokół słuchacza.
- **Wizualizacja** (OpenGL) pokazuje tę energię jako dynamiczną, obracaną mapę.
- **Wideo/obraz** może być użyte jako tło, aby łatwiej skojarzyć scenę z dźwiękiem.

Kierunek „patrzenia” w wizualizacji wpływa na to, jak interpretowane są kanały
ambisoniczne (symulacja rotacji pola dźwiękowego).

## Wymagania

- **Python 3.11.9**
- Systemowy pakiet: `libportaudio2`

### Zależności Pythona

```
numpy==1.26.4
scipy==1.14.1
sounddevice==0.4.6
soundfile==0.13.1
pyfar==0.7.3
opencv-python==4.11.0.86
PyQt6==6.10.0
PyOpenGL==3.1.10
matplotlib==3.8.1
```

## Instalacja

1. Zainstaluj zależności systemowe (Debian/Ubuntu):
   ```bash
   sudo apt-get install libportaudio2
   ```
2. Zainstaluj pakiety Pythona:
   ```bash
   pip install -r requirements.txt
   ```

## Uruchomienie

```bash
python main.py
```

## Obsługa aplikacji

1. **Load Audio** – wybierz plik `.wav` z ambisoniką.
2. **Load Video** – opcjonalnie wybierz wideo/obraz (`.mp4`, `.jpg`, `.png`).
3. **Play** – uruchamia odtwarzanie i wizualizację.

### Kontrolki

- **Ambisonic Order** – wybór rzędu (Auto/1/2/3).
- **Gain (Boost)** – wzmocnienie sygnału audio.
- **Visual Focus** – ostrość/kontrast wizualizacji.
- **Panel boczny** – można zwinąć przyciskiem strzałki.

## Pliki w repozytorium (skrót)

- `main.py` – aplikacja GUI i obsługa sterowania.
- `processing.py` – odtwarzanie, dekodowanie i analiza ambisoniki.
- `graphics_engine.py` – silnik wizualizacji OpenGL.

## Wskazówki

- Najlepsze efekty uzyskasz na plikach WAV z poprawnym mapowaniem kanałów
  ambisonicznych (FOA/2OA/3OA).
- Jeśli nie słyszysz dźwięku, sprawdź urządzenie wyjściowe audio i uprawnienia
  systemu do korzystania z niego.
