from music21 import converter, instrument, note, chord
import glob,pickle

notes=[]
for file in glob.glob("dataset/midi_files/*.mid"):
    midi=converter.parse(file)
    parts=instrument.partitionByInstrument(midi)
    notes_to_parse=parts.parts[0].recurse() if parts else midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element,note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element,chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

with open("notes.pkl","wb") as f:
    pickle.dump(notes,f)
print("Saved",len(notes),"notes")
