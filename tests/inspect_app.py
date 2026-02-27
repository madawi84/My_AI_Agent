import src.main

print("commands:", [c.name for c in src.main.app.registered_commands])
for c in src.main.app.registered_commands:
    print(c.name, c.callback)