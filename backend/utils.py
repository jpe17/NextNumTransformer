import json

def load_config(filepath):
  """Loads a JSON configuration file and returns it as a dictionary."""
  try:
    with open(filepath, 'r') as f:
      config = json.load(f)
      return config
  except FileNotFoundError:
    print(f"Error: File not found at {filepath}")
    return None
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {filepath}")
    return None