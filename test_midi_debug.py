#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug MIDI input to see what messages arrive."""

import mido

print("MIDI Input Debug Tool")
print("=" * 50)
print()

# List available ports
ports = mido.get_input_names()
print(f"Available MIDI inputs: {ports}")

if not ports:
    print("No MIDI devices found!")
    exit(1)

port_name = ports[0]
print(f"\nOpening: {port_name}")
print("Press keys, use sustain pedal, pitch bend, mod wheel...")
print("Press Ctrl+C to exit")
print("-" * 50)

try:
    with mido.open_input(port_name) as port:
        for msg in port:
            if msg.type == 'control_change':
                cc_names = {
                    1: "Mod Wheel",
                    7: "Volume",
                    10: "Pan",
                    64: "Sustain Pedal",
                    66: "Sostenuto",
                    67: "Soft Pedal",
                }
                cc_name = cc_names.get(msg.control, f"CC{msg.control}")
                print(f"CC: {cc_name} = {msg.value} (channel {msg.channel})")
            elif msg.type == 'note_on':
                print(f"Note ON:  {msg.note} vel={msg.velocity} (channel {msg.channel})")
            elif msg.type == 'note_off':
                print(f"Note OFF: {msg.note} vel={msg.velocity} (channel {msg.channel})")
            elif msg.type == 'pitchwheel':
                print(f"Pitch Bend: {msg.pitch} (channel {msg.channel})")
            else:
                print(f"Other: {msg}")
except KeyboardInterrupt:
    print("\nExiting...")
