from pathlib import Path

# Einziger Ort, an dem der Pfad zu den Daten eingetragen wird.
# Jede Person, die dieses Projekt nutzt, passt NUR diese eine Zeile an.
BASE_PATH = Path(r"C:\Users\user\Documents\EAGLE\DeepLearning\Deep4dist")

# Checkpoints relativ zum Projektordner selbst (nicht zur externen Festplatte),
# damit das unabhängig vom Speicherort der Rohdaten funktioniert.
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints_4thtry"