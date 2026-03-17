from pathlib import Path
path=Path(r"D:\Projects\DLGenAi Project\dataset\messy_mashup\genres_stems\jazz")
songs = list(path.iterdir())
print(songs)