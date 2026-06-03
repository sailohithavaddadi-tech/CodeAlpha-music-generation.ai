from midiutil import MIDIFile

m=MIDIFile(1)
m.addTrackName(0,0,"AI Music")
m.addTempo(0,0,120)

scale=[60,62,64,65,67,69,71,72]
for i,n in enumerate(scale*2):
    m.addNote(0,0,n,i,1,100)

with open("generated_music/generated.mid","wb") as f:
    m.writeFile(f)

print("generated_music/generated.mid created")
