# Projekt 1.4 - Rybki

## 1. Uwagi wstępne
Jeśli program się nie kompiluje, proszę postąpić następująco:

Kliknąć prawym przyciskiem na projekt (Boids) -> 
Build Dependencies -> 
Build Customizations -> 
wybrać dostępną wersję CUDA (projekt został bazowo skonfigurowany do używania CUDA 12.3)

## 2. Konfiguracja
Plik konfiguracyjny boids.config zawiera pola:

1. GPU - określa czy program ma działać w wersji CPU (wartość 0), czy GPU (wartość !0).
2. N_FOR_VIS - określa liczbę rybek w symulacji.