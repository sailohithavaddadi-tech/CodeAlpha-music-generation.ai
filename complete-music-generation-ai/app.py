from flask import Flask
app=Flask(__name__)

@app.route('/')
def home():
    return '<h1>Music Generation AI Project</h1><p>Run generate.py to create MIDI output.</p>'

app.run(debug=True)
